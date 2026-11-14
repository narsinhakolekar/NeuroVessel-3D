
"""
NeuroVessel V5: Residual 3D U-Net
AirRC pulmonary artery segmentation.

Training:
- Patient-level train/validation split.
- 3D foreground-aware patch sampling.
- Compact residual 3D U-Net.
- GroupNorm for batch-size-1 training.
- Mixed precision on CUDA.
- BCE + soft Dice loss.
- Fixed validation patches for checkpoint selection.

The 26-patient test split is never loaded or evaluated here.
"""

import argparse
import csv
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed_airrc_v5"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "labels_pa"

SPLIT_FILE = (
    PROJECT_ROOT / "outputs" / "metrics" / "airrc_v5_splits.csv"
)

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RESIDUAL 3D U-NET
# ============================================================

class ResidualBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        groups = min(8, out_channels)

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(groups, out_channels)
            )

        else:
            self.shortcut = nn.Identity()

        self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        return self.activation(x)


class NeuroVesselV5(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base=8):

        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, base)
        self.enc2 = ResidualBlock3D(base, base * 2)
        self.enc3 = ResidualBlock3D(base * 2, base * 4)
        self.enc4 = ResidualBlock3D(base * 4, base * 8)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(
            base * 8,
            base * 16
        )

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base * 16, base * 8, 2, stride=2
        )
        self.dec4 = ResidualBlock3D(base * 16, base * 8)

        self.up3 = nn.ConvTranspose3d(
            base * 8, base * 4, 2, stride=2
        )
        self.dec3 = ResidualBlock3D(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(
            base * 4, base * 2, 2, stride=2
        )
        self.dec2 = ResidualBlock3D(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(
            base * 2, base, 2, stride=2
        )
        self.dec1 = ResidualBlock3D(base * 2, base)

        self.output = nn.Conv3d(
            base,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 3. LOSS FUNCTION
# ============================================================

def combined_loss(logits, target, pos_weight):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight
    )

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * target).sum()

    dice = (
        2.0 * intersection + 1.0
    ) / (
        probabilities.sum() + target.sum() + 1.0
    )

    dice_loss = 1.0 - dice

    return 0.5 * bce + 0.5 * dice_loss


# ============================================================
# 4. LOAD PATIENT SPLITS
# ============================================================

def find_column(df, candidates, required=True):

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Missing required column. Expected one of {candidates}. "
            f"Found: {list(df.columns)}"
        )

    return None


def load_splits():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    df = pd.read_csv(SPLIT_FILE)

    uid_col = find_column(
        df,
        ["series_instance_uid", "uid"]
    )

    split_col = find_column(df, ["split"])

    patient_col = find_column(
        df,
        ["patient_id"],
        required=False
    )

    contrast_col = find_column(
        df,
        ["contrast_status"],
        required=False
    )

    records = []
    seen_uids = set()

    for _, row in df.iterrows():

        uid = str(row[uid_col]).strip()
        split = str(row[split_col]).strip().lower()

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unexpected split '{split}' for {uid}"
            )

        if uid in seen_uids:
            raise ValueError(f"Duplicate UID: {uid}")

        seen_uids.add(uid)

        patient_id = (
            str(row[patient_col]).strip()
            if patient_col is not None
            else uid
        )

        contrast = (
            str(row[contrast_col]).strip()
            if contrast_col is not None
            else "Unknown"
        )

        image_path = IMAGE_DIR / f"{uid}_ct.nii.gz"
        mask_path = MASK_DIR / f"{uid}_pa.nii.gz"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        records.append({
            "uid": uid,
            "patient_id": patient_id,
            "split": split,
            "contrast_status": contrast,
            "image": image_path,
            "mask": mask_path
        })

    # Explicit patient-level leakage check.
    patients_by_split = {}

    for record in records:

        patients_by_split.setdefault(
            record["split"], set()
        ).add(record["patient_id"])

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]:

        overlap = (
            patients_by_split[a]
            & patients_by_split[b]
        )

        if overlap:
            raise RuntimeError(
                f"Patient leakage between {a} and {b}: "
                f"{list(overlap)[:5]}"
            )

    train = [r for r in records if r["split"] == "train"]

    validation = [
        r for r in records
        if r["split"] == "validation"
    ]

    test_count = sum(
        r["split"] == "test" for r in records
    )

    print("\nDATASET SPLITS")
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Locked test cases:", test_count)

    print("\nThe test images and masks will not be loaded.")

    return train, validation


# ============================================================
# 5. READ ONE CASE
# ============================================================

def load_case(record):

    image_obj = sitk.ReadImage(str(record["image"]))
    mask_obj = sitk.ReadImage(str(record["mask"]))

    # Create independent NumPy arrays.
    # Do not use GetArrayViewFromImage here because its
    # memory depends on the lifetime of the SimpleITK objects.

    image = sitk.GetArrayFromImage(image_obj)
    mask = sitk.GetArrayFromImage(mask_obj)

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)

    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch in {record['uid']}: "
            f"{image.shape} vs {mask.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"Non-finite CT values in {record['uid']}"
        )

    if not np.any(mask):
        raise ValueError(
            f"Empty PA mask in {record['uid']}"
        )

    if not np.isin(np.unique(mask), [0, 1]).all():
        raise ValueError(
            f"Mask is not binary: {record['uid']}"
        )

    return image, mask


# ============================================================
# 6. 3D PATCH SAMPLING
# ============================================================

def get_crop_bounds(shape, center, patch_size):

    starts = []

    for dim, coordinate in zip(shape, center):

        start = int(coordinate) - patch_size // 2

        start = max(0, min(start, dim - patch_size))

        starts.append(start)

    slices = tuple(
        slice(s, s + patch_size)
        for s in starts
    )

    return slices


def sample_background_center(mask, rng, patch_size):

    shape = mask.shape

    for _ in range(25):

        center = tuple(
            int(rng.integers(0, dim))
            for dim in shape
        )

        slices = get_crop_bounds(
            shape,
            center,
            patch_size
        )

        patch_mask = mask[slices]

        # Prefer negative or low-foreground patches.
        if patch_mask.mean() < 0.001:
            return center

    return tuple(
        int(rng.integers(0, dim))
        for dim in shape
    )


def extract_patch(
    image,
    mask,
    center,
    patch_size,
    window_low,
    window_high
):

    slices = get_crop_bounds(
        image.shape,
        center,
        patch_size
    )

    x = image[slices].copy()

    y = mask[slices].copy()

    if x.shape != (
        patch_size,
        patch_size,
        patch_size
    ):
        raise ValueError(
            f"Unexpected patch shape: {x.shape}"
        )

    # Clip and normalize the selected input window to [0, 1].
    x = np.clip(
        x,
        window_low,
        window_high
    )

    x = (
        (x - window_low)
        / (window_high - window_low)
    ).astype(np.float32)

    y = y.astype(np.float32)

    return x, y


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

def augment_patch(x, y):

    # Random flips along 3 spatial axes.
    for dimension in (1, 2, 3):

        if random.random() < 0.5:

            x = torch.flip(x, dims=(dimension,))
            y = torch.flip(y, dims=(dimension,))

    # 90-degree rotations in the axial plane.
    rotations = random.randint(0, 3)

    if rotations:

        x = torch.rot90(
            x,
            rotations,
            dims=(2, 3)
        )

        y = torch.rot90(
            y,
            rotations,
            dims=(2, 3)
        )

    # Mild CT intensity augmentation.
    scale = random.uniform(0.90, 1.10)

    shift = random.uniform(-0.05, 0.05)

    x = x * scale + shift

    x = x + torch.randn_like(x) * 0.01

    x = torch.clamp(x, 0.0, 1.0)

    return x, y


# ============================================================
# 8. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    scaler,
    train_records,
    device,
    args,
    epoch,
    amp,
    pos_weight
):

    model.train()

    rng = np.random.default_rng(
        args.seed + epoch
    )

    order = rng.permutation(len(train_records))

    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, index in enumerate(order, start=1):

        record = train_records[index]

        image, mask = load_case(record)

        positive_coordinates = np.argwhere(mask > 0)

        # 75% foreground-centred patches.
        if rng.random() < args.foreground_probability:

            selected = positive_coordinates[
                rng.integers(len(positive_coordinates))
            ]

            center = tuple(int(v) for v in selected)

        else:

            center = sample_background_center(
                mask,
                rng,
                args.patch_size
            )

        x_np, y_np = extract_patch(
            image,
            mask,
            center,
            args.patch_size,
            args.window_low,
            args.window_high
        )

        del image, mask, positive_coordinates

        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)
        y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x, y = augment_patch(x, y)

        try:

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp
            ):

                logits = model(x)

                loss = combined_loss(
                    logits,
                    y,
                    pos_weight
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        except torch.cuda.OutOfMemoryError as exc:

            torch.cuda.empty_cache()

            raise RuntimeError(
                "GPU memory exhausted. Rerun with "
                "--patch-size 64."
            ) from exc

        total_loss += float(loss.detach().item())

        if step % 20 == 0 or step == len(order):

            print(
                f"  Train {step:03d}/{len(order)} "
                f"loss={float(loss.detach().item()):.4f}"
            )

        del x, y, logits, loss, x_np, y_np

    return total_loss / len(order)


# ============================================================
# 9. DETERMINISTIC VALIDATION PATCHES
# ============================================================

def stable_seed(uid, base_seed):

    digest = hashlib.sha256(
        uid.encode("utf-8")
    ).hexdigest()

    return (
        int(digest[:8], 16) + base_seed
    ) % (2**32)


def validate(
    model,
    validation_records,
    device,
    args,
    amp,
    pos_weight
):

    model.eval()

    case_dice = []
    case_precision = []
    case_recall = []

    total_val_loss = 0.0
    total_patches = 0

    with torch.inference_mode():

        for case_index, record in enumerate(
            validation_records,
            start=1
        ):

            image, mask = load_case(record)

            rng = np.random.default_rng(
                stable_seed(record["uid"], args.seed)
            )

            positive_coordinates = np.argwhere(mask > 0)

            # Fixed positive and background patches.
            positive_center = tuple(
                int(v) for v in
                positive_coordinates[
                    rng.integers(len(positive_coordinates))
                ]
            )

            background_center = sample_background_center(
                mask,
                rng,
                args.patch_size
            )

            centers = [
                positive_center,
                background_center
            ]

            case_intersection = 0
            case_predicted = 0
            case_target = 0

            for center in centers:

                x_np, y_np = extract_patch(
                    image,
                    mask,
                    center,
                    args.patch_size,
                    args.window_low,
                    args.window_high
                )

                x = torch.from_numpy(
                    x_np
                ).unsqueeze(0).unsqueeze(0).to(device)

                y = torch.from_numpy(
                    y_np
                ).unsqueeze(0).unsqueeze(0).to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp
                ):

                    logits = model(x)

                    loss = combined_loss(
                        logits,
                        y,
                        pos_weight
                    )

                probabilities = torch.sigmoid(logits)

                prediction = (
                    probabilities >= 0.5
                ).float()

                case_intersection += int(
                    (prediction * y).sum().item()
                )

                case_predicted += int(
                    prediction.sum().item()
                )

                case_target += int(
                    y.sum().item()
                )

                total_val_loss += float(
                    loss.item()
                )

                total_patches += 1

                del x, y, logits, loss, prediction

            dice = (
                2.0 * case_intersection + 1.0
            ) / (
                case_predicted + case_target + 1.0
            )

            precision = (
                case_intersection / case_predicted
                if case_predicted else 0.0
            )

            recall = (
                case_intersection / case_target
                if case_target else 0.0
            )

            case_dice.append(dice)
            case_precision.append(precision)
            case_recall.append(recall)

            if case_index % 5 == 0 or (
                case_index == len(validation_records)
            ):

                print(
                    f"  Validation "
                    f"{case_index}/{len(validation_records)}"
                )

            del image, mask, positive_coordinates

    return {
        "loss": total_val_loss / max(total_patches, 1),
        "dice": float(np.mean(case_dice)),
        "precision": float(np.mean(case_precision)),
        "recall": float(np.mean(case_recall))
    }


# ============================================================
# 10. MAIN TRAINING FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--window-low",
        type=float,
        default=-1000
    )

    parser.add_argument(
        "--window-high",
        type=float,
        default=1000
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=96
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--foreground-probability",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.window_high <= args.window_low:
        raise ValueError("Invalid intensity window.")

    if args.patch_size < 32 or args.patch_size % 16 != 0:
        raise ValueError(
            "Patch size must be >=32 and divisible by 16."
        )

    # Reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp = device.type == "cuda"

    # Smoke test deliberately uses small patches and few cases.
    if args.smoke_test:

        args.patch_size = 64
        args.epochs = 1

        print("\nSMOKE TEST MODE")
        print("Training cases: 4")
        print("Validation cases: 2")
        print("Patch size: 64")

    train_records, validation_records = load_splits()

    if args.smoke_test:

        train_records = train_records[:4]
        validation_records = validation_records[:2]

    if not args.smoke_test:

        if len(train_records) != 202:
            raise RuntimeError(
                f"Expected 202 training cases, got {len(train_records)}"
            )

        if len(validation_records) != 26:
            raise RuntimeError(
                f"Expected 26 validation cases, got "
                f"{len(validation_records)}"
            )

    print("\n" + "=" * 60)
    print("NEUROVESSEL V5 TRAINING")
    print("=" * 60)

    print("Device:", device)

    if amp:
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    print("Architecture: Residual 3D U-Net")
    print("Base channels: 8")
    print("Patch:", args.patch_size)
    print("Batch size: 1")
    print("AMP:", amp)
    print("Intensity window:", args.window_low, args.window_high)
    print("Epochs:", args.epochs)

    # Initialize model.
    model = NeuroVesselV5(
        in_channels=1,
        out_channels=1,
        base=8
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp
    )

    pos_weight = torch.tensor(
        [3.0],
        dtype=torch.float32,
        device=device
    )

    patience = 8
    epochs_without_improvement = 0
    best_val_dice = -1.0

    if args.smoke_test:

        run_name = "airrc_v5_smoke"

    else:

        run_name = "airrc_v5_wm1000_wp1000"

    best_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    last_path = CHECKPOINT_DIR / f"{run_name}_last.pt"

    history_path = METRICS_DIR / f"{run_name}_history.csv"

    config_path = METRICS_DIR / f"{run_name}_config.json"

    config = {
        "architecture": "Residual 3D U-Net",
        "base_channels": 8,
        "patch_size": args.patch_size,
        "batch_size": 1,
        "epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "window_low": args.window_low,
        "window_high": args.window_high,
        "foreground_probability": args.foreground_probability,
        "positive_bce_weight": 3.0,
        "seed": args.seed,
        "amp": amp,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "validation_method": (
            "Two deterministic patches per validation patient; "
            "not full-volume Dice."
        )
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_patch_dice",
        "val_precision",
        "val_recall",
        "learning_rate",
        "epoch_seconds"
    ]

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        print(f"\n{'=' * 50}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 50)

        train_loss = train_one_epoch(
            model,
            optimizer,
            scaler,
            train_records,
            device,
            args,
            epoch,
            amp,
            pos_weight
        )

        val_metrics = validate(
            model,
            validation_records,
            device,
            args,
            amp,
            pos_weight
        )

        scheduler.step(val_metrics["dice"])

        lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start

        print("\nEPOCH RESULTS")
        print("Training loss:", round(train_loss, 5))
        print("Validation loss:", round(val_metrics["loss"], 5))
        print("Validation patch Dice:", round(val_metrics["dice"], 5))
        print("Validation precision:", round(val_metrics["precision"], 5))
        print("Validation recall:", round(val_metrics["recall"], 5))
        print("Learning rate:", lr)
        print("Time:", round(elapsed, 1), "seconds")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_patch_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "learning_rate": lr,
            "epoch_seconds": elapsed
        }

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writerow(row)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_patch_dice": best_val_dice,
            "config": config
        }

        # Always save the latest checkpoint.
        torch.save(state, last_path)

        if val_metrics["dice"] > best_val_dice:

            best_val_dice = val_metrics["dice"]

            state["best_val_patch_dice"] = best_val_dice

            torch.save(state, best_path)

            epochs_without_improvement = 0

            print(
                "\nBEST CHECKPOINT UPDATED:",
                best_path
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                patience
            )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (
            not args.smoke_test
            and epochs_without_improvement >= patience
        ):

            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print("Best validation patch Dice:", best_val_dice)
    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", last_path)
    print("Training history:", history_path)
    print("Configuration:", config_path)

    print(
        "\nIMPORTANT: These are patch-level validation metrics, "
        "not full-volume evaluation results."
    )

    print(
        "The 26-patient test set remains untouched."
    )


if __name__ == "__main__":
    main()
