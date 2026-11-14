
import csv
import json
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet


SEED = 42
PATCH_SIZE = 64
EPOCHS = 5
STEPS_PER_EPOCH = 100
LEARNING_RATE = 3e-4
FOREGROUND_PROBABILITY = 0.7
THRESHOLD = 0.5
OVERLAP = 0.25

DATA_ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT_DIR = Path("outputs/checkpoints")
METRICS_DIR = Path("outputs/metrics")

BEST_PATH = CHECKPOINT_DIR / "unet_v2_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "unet_v2_latest.pt"
HISTORY_PATH = METRICS_DIR / "unet_v2_history.json"
VALIDATION_PATH = METRICS_DIR / "unet_v2_validation.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_case(patient_id):
    folder = DATA_ROOT / patient_id
    image_path = folder / "image" / f"{patient_id}.nii.gz"
    label_path = folder / "label" / f"{patient_id}.nii.gz"

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(f"Shape mismatch: {patient_id}")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite CT: {patient_id}")

    if not label.any():
        raise ValueError(f"Empty label: {patient_id}")

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    if min(image.shape) < PATCH_SIZE:
        raise ValueError(f"Volume too small: {patient_id}")

    return image, label


def sample_patch(image, label, rng):
    p = PATCH_SIZE
    sx, sy, sz = image.shape

    foreground = np.argwhere(label)

    if len(foreground) and rng.random() < FOREGROUND_PROBABILITY:
        cx, cy, cz = foreground[rng.integers(len(foreground))]
        x = int(np.clip(cx - p // 2, 0, sx - p))
        y = int(np.clip(cy - p // 2, 0, sy - p))
        z = int(np.clip(cz - p // 2, 0, sz - p))
    else:
        x = int(rng.integers(sx - p + 1))
        y = int(rng.integers(sy - p + 1))
        z = int(rng.integers(sz - p + 1))

    image_patch = image[x:x+p, y:y+p, z:z+p].copy()
    label_patch = label[x:x+p, y:y+p, z:z+p].astype(np.float32)

    return image_patch, label_patch


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)

    dims = (1, 2, 3, 4)
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)

    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    dice_loss = 1 - dice.mean()

    return 0.5 * bce + 0.5 * dice_loss


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = np.logical_and(prediction, target).sum()
    predicted_count = prediction.sum()
    target_count = target.sum()
    denominator = predicted_count + target_count

    return {
        "dice": float(2 * intersection / denominator) if denominator else 1.0,
        "precision": float(intersection / predicted_count) if predicted_count else 0.0,
        "recall": float(intersection / target_count) if target_count else 0.0,
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
    }


def build_model(device):
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample="nontrainable",
    ).to(device)


def predict_volume(model, image, device):
    tensor = torch.from_numpy(
        np.ascontiguousarray(image[None, None], dtype=np.float32)
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=(PATCH_SIZE,) * 3,
                sw_batch_size=1,
                predictor=model,
                overlap=OVERLAP,
                mode="gaussian",
                sw_device=device,
                device=torch.device("cpu"),
            )

        probabilities = torch.sigmoid(logits)[0, 0].float().numpy()

    return probabilities


def evaluate_validation(model, patient_ids, device, epoch):
    model.eval()
    rows = []

    for number, patient_id in enumerate(patient_ids, start=1):
        start = time.time()
        image, target = load_case(patient_id)
        probabilities = predict_volume(model, image, device)
        prediction = probabilities >= THRESHOLD
        metrics = calculate_metrics(prediction, target)

        row = {
            "epoch": epoch,
            "patient_id": patient_id,
            **metrics,
            "seconds": round(time.time() - start, 2),
        }
        rows.append(row)

        print(
            f"  {number}/{len(patient_ids)} {patient_id}: "
            f"Dice={metrics['dice']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )

        del image, target, probabilities, prediction
        torch.cuda.empty_cache()

    summary = {
        "epoch": epoch,
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "median_dice": float(np.median([row["dice"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "patients": len(rows),
        "threshold": THRESHOLD,
        "test_set_evaluated": False,
    }

    return rows, summary


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the neurovessel environment.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = set(split["test"])

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Train/validation leakage")
    if set(train_ids) & test_ids or set(validation_ids) & test_ids:
        raise RuntimeError("Test-set leakage")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")
    print("Full-volume validation; test set remains untouched.")

    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    rng = np.random.default_rng(SEED)

    history = []
    all_rows = []
    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        epoch_start = time.time()
        shuffled_ids = list(train_ids)
        random.Random(SEED + epoch).shuffle(shuffled_ids)

        # Each epoch loads each patient at most once into this local cache.
        cache = {}

        print(f"\nEpoch {epoch}/{EPOCHS}")

        for step in range(1, STEPS_PER_EPOCH + 1):
            patient_id = shuffled_ids[(step - 1) % len(shuffled_ids)]

            if patient_id not in cache:
                cache[patient_id] = load_case(patient_id)

            image, label = cache[patient_id]
            image_patch, label_patch = sample_patch(image, label, rng)

            x = torch.from_numpy(image_patch[None, None]).to(device)
            y = torch.from_numpy(label_patch[None, None]).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                logits = model(x)
                loss = combined_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach()))

            if step == 1 or step % 20 == 0 or step == STEPS_PER_EPOCH:
                print(f"  Step {step}/{STEPS_PER_EPOCH}, loss={losses[-1]:.4f}")

            del x, y, logits, loss

        del cache

        print("Running full-volume validation...")
        rows, summary = evaluate_validation(
            model, validation_ids, device, epoch
        )
        all_rows.extend(rows)

        record = {
            "epoch": epoch,
            "mean_training_loss": float(np.mean(losses)),
            "validation_mean_dice": summary["mean_dice"],
            "validation_median_dice": summary["median_dice"],
            "validation_mean_precision": summary["mean_precision"],
            "validation_mean_recall": summary["mean_recall"],
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "seed": SEED,
            "patch_size": PATCH_SIZE,
            "architecture": "MONAI BasicUNet",
            "features": (8, 16, 32, 64, 128, 8),
            "upsample": "nontrainable",
            "loss": "0.5 BCE + 0.5 soft Dice",
            "validation_summary": summary,
            "test_set_evaluated": False,
        }

        torch.save(checkpoint, LATEST_PATH)

        if summary["mean_dice"] > best_dice:
            best_dice = summary["mean_dice"]
            torch.save(checkpoint, BEST_PATH)
            print("Saved new best checkpoint.")

        HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        with VALIDATION_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Epoch {epoch} training loss: {record['mean_training_loss']:.4f}")
        print(f"Mean validation Dice: {summary['mean_dice']:.4f}")

    print("\nV2 training complete.")
    print("Best validation Dice:", best_dice)
    print("Best checkpoint:", BEST_PATH)
    print("Training history:", HISTORY_PATH)
    print("Validation results:", VALIDATION_PATH)
    print("Test patients were not evaluated.")


if __name__ == "__main__":
    main()
