
import csv
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The experiments folder is automatically on sys.path when this
# file is executed as: python experiments/train_unet_v4_hard_negative.py
from train_unet_v3_frangi import (
    load_case,
    build_model,
    calculate_metrics,
    predict_volume,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PATCH_SIZE = 64
EPOCHS = 8
STEPS_PER_EPOCH = 100

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

# Patch sampling probabilities. They sum to 1.
FOREGROUND_PROBABILITY = 0.45
HARD_NEGATIVE_PROBABILITY = 0.30
RANDOM_PROBABILITY = 0.25

# A hard-negative voxel is background with Frangi response
# above this patient's 90th background percentile.
HARD_NEGATIVE_PERCENTILE = 90

CACHE_PATIENTS = 2
THRESHOLD = 0.5
OVERLAP = 0.25

ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = ROOT / "data" / "splits.json"
DATA_ROOT = ROOT / "data" / "raw" / "train"
FRANGI_ROOT = ROOT / "outputs" / "frangi_features"

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"

BEST_PATH = CHECKPOINT_DIR / "unet_v4_hard_negative_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "unet_v4_hard_negative_latest.pt"

HISTORY_PATH = METRICS_DIR / "unet_v4_hard_negative_history.json"
VALIDATION_PATH = METRICS_DIR / "unet_v4_hard_negative_validation.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# CASE LOADING AND HARD-NEGATIVE THRESHOLD
# ============================================================

def load_case_v4(patient_id):

    image, frangi, label = load_case(patient_id)

    background = ~label

    # Calculate a patient-specific high-Frangi background threshold.
    # This is calculated only from background voxels.
    background_values = frangi[background]

    if background_values.size == 0:
        raise ValueError(f"No background voxels in {patient_id}")

    hard_threshold = float(
        np.percentile(
            background_values,
            HARD_NEGATIVE_PERCENTILE
        )
    )

    del background_values

    return image, frangi, label, hard_threshold


# ============================================================
# PATCH SAMPLING
# ============================================================

def random_center_from_mask(mask, rng, max_attempts=1000):
    """
    Samples a random voxel from a 3D mask without constructing
    a potentially huge np.argwhere coordinate array.
    """

    sx, sy, sz = mask.shape

    for _ in range(max_attempts):

        x = int(rng.integers(sx))
        y = int(rng.integers(sy))
        z = int(rng.integers(sz))

        if mask[x, y, z]:
            return x, y, z

    return None


def random_volume_center(shape, rng):

    return tuple(
        int(rng.integers(size))
        for size in shape
    )


def extract_patch(image, frangi, label, center, rng):

    p = PATCH_SIZE
    sx, sy, sz = image.shape

    cx, cy, cz = center

    x = int(np.clip(cx - p // 2, 0, sx - p))
    y = int(np.clip(cy - p // 2, 0, sy - p))
    z = int(np.clip(cz - p // 2, 0, sz - p))

    ct_patch = image[x:x+p, y:y+p, z:z+p]
    frangi_patch = frangi[x:x+p, y:y+p, z:z+p]
    label_patch = label[x:x+p, y:y+p, z:z+p]

    x_patch = np.stack(
        [ct_patch, frangi_patch],
        axis=0
    ).astype(np.float32, copy=True)

    y_patch = label_patch.astype(
        np.float32,
        copy=True
    )

    return x_patch, y_patch


def sample_patch_v4(image, frangi, label, hard_threshold, rng):

    background = ~label

    hard_mask = background & (frangi >= hard_threshold)

    choice = rng.random()

    # 1. Foreground-centred patch.
    if choice < FOREGROUND_PROBABILITY:

        center = random_center_from_mask(label, rng)

        if center is None:
            center = random_volume_center(image.shape, rng)

    # 2. High-Frangi hard-negative-centred patch.
    elif choice < FOREGROUND_PROBABILITY + HARD_NEGATIVE_PROBABILITY:

        center = random_center_from_mask(hard_mask, rng)

        if center is None:
            center = random_volume_center(image.shape, rng)

    # 3. Uniform random patch.
    else:

        center = random_volume_center(image.shape, rng)

    return extract_patch(
        image,
        frangi,
        label,
        center,
        rng
    )


# ============================================================
# LOSS
# ============================================================

def combined_loss(logits, target):

    # Binary cross-entropy encourages voxel-wise discrimination.
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target
    )

    # Soft Dice loss encourages region overlap.
    probabilities = torch.sigmoid(logits)

    dimensions = (1, 2, 3, 4)

    intersection = (
        probabilities * target
    ).sum(dim=dimensions)

    denominator = (
        probabilities.sum(dim=dimensions)
        + target.sum(dim=dimensions)
    )

    dice = (
        2.0 * intersection + 1e-5
    ) / (
        denominator + 1e-5
    )

    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(model, validation_ids, device, epoch):

    model.eval()

    rows = []

    for number, patient_id in enumerate(validation_ids, 1):

        start = time.time()

        image, frangi, label = load_case(patient_id)

        probabilities = predict_volume(
            model,
            image,
            frangi,
            device
        )

        prediction = probabilities >= THRESHOLD

        metrics = calculate_metrics(
            prediction,
            label
        )

        row = {
            "epoch": epoch,
            "patient_id": patient_id,
            **metrics,
            "threshold": THRESHOLD,
            "seconds": round(time.time() - start, 2)
        }

        rows.append(row)

        print(
            f"  {number}/{len(validation_ids)} {patient_id}: "
            f"Dice={metrics['dice']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Predicted={metrics['predicted_fraction']:.4%}"
        )

        del image, frangi, label, probabilities, prediction
        torch.cuda.empty_cache()

    summary = {
        "epoch": epoch,
        "mean_dice": float(
            np.mean([r["dice"] for r in rows])
        ),
        "median_dice": float(
            np.median([r["dice"] for r in rows])
        ),
        "mean_precision": float(
            np.mean([r["precision"] for r in rows])
        ),
        "mean_recall": float(
            np.mean([r["recall"] for r in rows])
        ),
        "mean_predicted_fraction": float(
            np.mean([r["predicted_fraction"] for r in rows])
        ),
        "threshold": THRESHOLD,
        "patients": len(rows),
        "test_set_evaluated": False
    }

    return rows, summary


# ============================================================
# PATIENT CACHE
# ============================================================

def get_cached_case(patient_id, cache):

    if patient_id in cache:

        cache.move_to_end(patient_id)
        return cache[patient_id]

    case = load_case_v4(patient_id)

    cache[patient_id] = case

    while len(cache) > CACHE_PATIENTS:
        cache.popitem(last=False)

    return case


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your neurovessel environment."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = set(split["test"])

    # Explicit split leakage checks.
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Train/validation leakage detected.")

    if set(train_ids) & test_ids:
        raise RuntimeError("Train/test leakage detected.")

    if set(validation_ids) & test_ids:
        raise RuntimeError("Validation/test leakage detected.")

    # Verify all required training and validation features.
    required_ids = train_ids + validation_ids

    missing = [
        pid for pid in required_ids
        if not (FRANGI_ROOT / f"{pid}.npy").exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing Frangi features: {missing}"
        )

    print("=" * 65)
    print("NEUROVESSEL V4 - HARD NEGATIVE TRAINING")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training patients:", len(train_ids))
    print("Validation patients:", len(validation_ids))
    print("Test patients:", len(test_ids))
    print("Test patients will NOT be evaluated.")

    print("\nInput channels: CT + Frangi")
    print("Patch size:", PATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Steps per epoch:", STEPS_PER_EPOCH)

    print("\nPatch sampling:")
    print(f"  Foreground: {FOREGROUND_PROBABILITY:.0%}")
    print(f"  Hard negative: {HARD_NEGATIVE_PROBABILITY:.0%}")
    print(f"  Random: {RANDOM_PROBABILITY:.0%}")

    print(
        "Hard-negative Frangi percentile:",
        HARD_NEGATIVE_PERCENTILE
    )

    print("\nLoss: 0.5 BCE + 0.5 soft Dice")
    print("Validation threshold:", THRESHOLD)

    device = torch.device("cuda")

    model = build_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

    rng = np.random.default_rng(SEED)

    history = []
    all_rows = []

    best_dice = -1.0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        model.train()

        losses = []
        epoch_start = time.time()

        shuffled_ids = list(train_ids)

        random.Random(SEED + epoch).shuffle(shuffled_ids)

        cache = OrderedDict()

        print(f"\n{'=' * 20} EPOCH {epoch}/{EPOCHS} {'=' * 20}")

        for step in range(1, STEPS_PER_EPOCH + 1):

            patient_id = shuffled_ids[
                (step - 1) % len(shuffled_ids)
            ]

            image, frangi, label, hard_threshold = (
                get_cached_case(patient_id, cache)
            )

            x_patch, label_patch = sample_patch_v4(
                image,
                frangi,
                label,
                hard_threshold,
                rng
            )

            x = torch.from_numpy(
                x_patch[None]
            ).to(device)

            y = torch.from_numpy(
                label_patch[None, None]
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True
            ):

                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().item()))

            if (
                step == 1
                or step % 20 == 0
                or step == STEPS_PER_EPOCH
            ):

                print(
                    f"Step {step}/{STEPS_PER_EPOCH} "
                    f"| Loss: {losses[-1]:.4f} "
                    f"| Patient: {patient_id}"
                )

            del x, y, logits, loss
            del x_patch, label_patch

        del cache

        # ====================================================
        # FULL-VOLUME VALIDATION
        # ====================================================

        print("\nRunning full-volume validation...")

        rows, summary = evaluate_validation(
            model,
            validation_ids,
            device,
            epoch
        )

        all_rows.extend(rows)

        record = {
            "epoch": epoch,
            "mean_training_loss": float(np.mean(losses)),
            "validation_mean_dice": summary["mean_dice"],
            "validation_median_dice": summary["median_dice"],
            "validation_mean_precision": summary["mean_precision"],
            "validation_mean_recall": summary["mean_recall"],
            "validation_mean_predicted_fraction":
                summary["mean_predicted_fraction"],
            "seconds": round(time.time() - epoch_start, 2)
        }

        history.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "seed": SEED,
            "architecture": "MONAI BasicUNet",
            "in_channels": 2,
            "input_channels": [
                "normalized_CT",
                "Frangi_vesselness"
            ],
            "features": (8, 16, 32, 64, 128, 8),
            "patch_size": PATCH_SIZE,
            "loss": "0.5 BCE + 0.5 soft Dice",
            "sampling": {
                "foreground_probability": FOREGROUND_PROBABILITY,
                "hard_negative_probability": HARD_NEGATIVE_PROBABILITY,
                "random_probability": RANDOM_PROBABILITY,
                "hard_negative_percentile": HARD_NEGATIVE_PERCENTILE
            },
            "learning_rate": LEARNING_RATE,
            "validation_summary": summary,
            "test_set_evaluated": False
        }

        torch.save(checkpoint, LATEST_PATH)

        if summary["mean_dice"] > best_dice:

            best_dice = summary["mean_dice"]

            torch.save(checkpoint, BEST_PATH)

            print("\nSaved new best V4 checkpoint.")

        HISTORY_PATH.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

        with VALIDATION_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(all_rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(all_rows)

        print("\nEpoch summary:")
        print("Training loss:", f"{record['mean_training_loss']:.4f}")
        print("Validation Dice:", f"{summary['mean_dice']:.4f}")
        print("Validation precision:", f"{summary['mean_precision']:.4f}")
        print("Validation recall:", f"{summary['mean_recall']:.4f}")
        print(
            "Predicted foreground:",
            f"{summary['mean_predicted_fraction']:.4%}"
        )

        print("Best validation Dice so far:", f"{best_dice:.4f}")

    print("\n" + "=" * 65)
    print("V4 TRAINING COMPLETE")
    print("=" * 65)

    print("Best validation Dice:", best_dice)
    print("Best checkpoint:", BEST_PATH)
    print("Latest checkpoint:", LATEST_PATH)
    print("History:", HISTORY_PATH)
    print("Validation CSV:", VALIDATION_PATH)
    print("Test patients were NOT evaluated.")


if __name__ == "__main__":
    main()
