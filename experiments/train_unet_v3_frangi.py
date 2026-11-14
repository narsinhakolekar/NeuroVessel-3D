import csv
import json
import random
import time
from collections import OrderedDict
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
CACHE_PATIENTS = 2

DATA_ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
FRANGI_ROOT = Path("outputs/frangi_features")
CHECKPOINT_DIR = Path("outputs/checkpoints")
METRICS_DIR = Path("outputs/metrics")
BEST_PATH = CHECKPOINT_DIR / "unet_v3_frangi_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "unet_v3_frangi_latest.pt"
HISTORY_PATH = METRICS_DIR / "unet_v3_frangi_history.json"
VALIDATION_PATH = METRICS_DIR / "unet_v3_frangi_validation.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_case(patient_id):
    folder = DATA_ROOT / patient_id
    image_file = nib.load(str(folder / "image" / f"{patient_id}.nii.gz"))
    label_file = nib.load(str(folder / "label" / f"{patient_id}.nii.gz"))
    frangi_path = FRANGI_ROOT / f"{patient_id}.npy"

    if not frangi_path.exists():
        raise FileNotFoundError(f"Missing Frangi feature: {frangi_path}")
    if image_file.shape != label_file.shape:
        raise ValueError(f"CT/label shape mismatch: {patient_id}")
    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"CT/label affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0
    frangi = np.load(frangi_path, mmap_mode="r")

    if tuple(frangi.shape) != tuple(image.shape):
        raise ValueError(f"CT/Frangi shape mismatch: {patient_id}")
    if not np.isfinite(image).all() or not np.isfinite(frangi).all():
        raise ValueError(f"Non-finite input: {patient_id}")
    if not label.any():
        raise ValueError(f"Empty label: {patient_id}")
    if min(image.shape) < PATCH_SIZE:
        raise ValueError(f"Volume too small: {patient_id}")

    image = (np.clip(image, -1000, 500) + 1000.0) / 1500.0
    frangi = np.asarray(frangi, dtype=np.float32)
    return image, frangi, label


def sample_patch(image, frangi, label, rng):
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

    ct_patch = image[x:x+p, y:y+p, z:z+p]
    frangi_patch = frangi[x:x+p, y:y+p, z:z+p]
    label_patch = label[x:x+p, y:y+p, z:z+p].astype(np.float32)
    return np.stack([ct_patch, frangi_patch], axis=0).copy(), label_patch.copy()


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3, 4)
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    pred_count, target_count = prediction.sum(), target.sum()
    return {
        "dice": float(2 * intersection / (pred_count + target_count))
                if pred_count + target_count else 1.0,
        "precision": float(intersection / pred_count) if pred_count else 0.0,
        "recall": float(intersection / target_count) if target_count else 0.0,
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
    }


def build_model(device):
    return BasicUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        features=(8, 16, 32, 64, 128, 8), upsample="nontrainable"
    ).to(device)


def predict_volume(model, image, frangi, device):
    volume = np.stack([image, frangi], axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(volume[None], dtype=np.float32))
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = sliding_window_inference(
                inputs=tensor, roi_size=(PATCH_SIZE,) * 3, sw_batch_size=1,
                predictor=model, overlap=OVERLAP, mode="gaussian",
                sw_device=device, device=torch.device("cpu")
            )
        probabilities = torch.sigmoid(logits)[0, 0].float().numpy()
    return probabilities


def evaluate_validation(model, patient_ids, device, epoch):
    model.eval()
    rows = []
    for number, patient_id in enumerate(patient_ids, start=1):
        start = time.time()
        image, frangi, target = load_case(patient_id)
        probabilities = predict_volume(model, image, frangi, device)
        metrics = calculate_metrics(probabilities >= THRESHOLD, target)
        row = {
            "epoch": epoch, "patient_id": patient_id, **metrics,
            "seconds": round(time.time() - start, 2)
        }
        rows.append(row)
        print(
            f"  {number}/{len(patient_ids)} {patient_id}: "
            f"Dice={metrics['dice']:.4f}, Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )
        del image, frangi, target, probabilities
        torch.cuda.empty_cache()

    summary = {
        "epoch": epoch,
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "median_dice": float(np.median([r["dice"] for r in rows])),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "patients": len(rows), "threshold": THRESHOLD,
        "test_set_evaluated": False,
    }
    return rows, summary


def get_cached_case(patient_id, cache):
    if patient_id in cache:
        cache.move_to_end(patient_id)
        return cache[patient_id]
    case = load_case(patient_id)
    cache[patient_id] = case
    while len(cache) > CACHE_PATIENTS:
        cache.popitem(last=False)
    return case


def main():
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the neurovessel environment.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    train_ids, validation_ids = split["train"], split["validation"]
    test_ids = set(split["test"])

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Train/validation leakage")
    if set(train_ids) & test_ids or set(validation_ids) & test_ids:
        raise RuntimeError("Test-set leakage")

    required = train_ids + validation_ids
    missing = [pid for pid in required if not (FRANGI_ROOT / f"{pid}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Frangi features: {missing}")

    metadata_path = FRANGI_ROOT / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("test_patients_processed", False):
            raise RuntimeError("Metadata indicates test patients were processed.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")
    print("Input channels: normalized CT + Frangi vesselness")
    print(f"LRU patient cache: {CACHE_PATIENTS}")
    print("Full-volume validation; test set remains untouched.")

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    rng = np.random.default_rng(SEED)
    history, all_rows = [], []
    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        epoch_start = time.time()
        shuffled_ids = list(train_ids)
        random.Random(SEED + epoch).shuffle(shuffled_ids)
        cache = OrderedDict()
        print(f"\\nEpoch {epoch}/{EPOCHS}")

        for step in range(1, STEPS_PER_EPOCH + 1):
            patient_id = shuffled_ids[(step - 1) % len(shuffled_ids)]
            image, frangi, label = get_cached_case(patient_id, cache)
            x_patch, label_patch = sample_patch(image, frangi, label, rng)
            x = torch.from_numpy(x_patch[None]).to(device)
            y = torch.from_numpy(label_patch[None, None]).to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
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
            del x, y, logits, loss, x_patch, label_patch

        del cache
        print("Running full-volume validation...")
        rows, summary = evaluate_validation(model, validation_ids, device, epoch)
        all_rows.extend(rows)
        record = {
            "epoch": epoch, "mean_training_loss": float(np.mean(losses)),
            "validation_mean_dice": summary["mean_dice"],
            "validation_median_dice": summary["median_dice"],
            "validation_mean_precision": summary["mean_precision"],
            "validation_mean_recall": summary["mean_recall"],
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(record)

        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "seed": SEED,
            "patch_size": PATCH_SIZE, "architecture": "MONAI BasicUNet",
            "in_channels": 2, "input_channels": ["normalized_CT", "Frangi_vesselness"],
            "features": (8, 16, 32, 64, 128, 8), "upsample": "nontrainable",
            "loss": "0.5 BCE + 0.5 soft Dice",
            "frangi_feature_dir": str(FRANGI_ROOT),
            "validation_summary": summary, "test_set_evaluated": False,
        }
        torch.save(checkpoint, LATEST_PATH)
        if summary["mean_dice"] > best_dice:
            best_dice = summary["mean_dice"]
            torch.save(checkpoint, BEST_PATH)
            print("Saved new best checkpoint.")

        HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        with VALIDATION_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Epoch {epoch} training loss: {record['mean_training_loss']:.4f}")
        print(f"Mean validation Dice: {summary['mean_dice']:.4f}")

    print("\\nV3 Frangi training complete.")
    print("Best validation Dice:", best_dice)
    print("Best checkpoint:", BEST_PATH)
    print("Training history:", HISTORY_PATH)
    print("Validation results:", VALIDATION_PATH)
    print("Test patients were not evaluated.")


if __name__ == "__main__":
    main()
