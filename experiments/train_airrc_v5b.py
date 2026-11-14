
"""
NEUROVESSEL V5B
3D Pulmonary Artery Segmentation using a Residual 3D U-Net

Dataset:
    AirRC processed dataset
    202 training cases
    26 validation cases
    26 locked test cases

Important:
    Test images and masks are never loaded by this script.

Improvements over V5:
    1. Independent SimpleITK-to-NumPy copies.
    2. Two training patches per patient per epoch.
    3. Correct augmentation over D, H and W dimensions.
    4. Improved background patch sampling.
    5. Six deterministic validation patches per patient.
    6. Separate V5B checkpoints and metric files.

Author: NeuroVessel project
"""

# ============================================================
# 1. IMPORTS
# ============================================================

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
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed_airrc_v5"

IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "labels_pa"

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

SPLIT_CSV = METRICS_DIR / "airrc_v5_splits.csv"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. RESIDUAL 3D U-NET
# ============================================================

def get_group_count(channels):

    groups = min(8, channels)

    while channels % groups != 0:
        groups -= 1

    return groups


class ResidualBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        groups = get_group_count(out_channels)

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(
            groups,
            out_channels
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(
            groups,
            out_channels
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(
                    groups,
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.activation(out)

        return out


class NeuroVesselV5(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base=8
    ):

        super().__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # Encoder
        self.enc1 = ResidualBlock3D(
            in_channels,
            base
        )

        self.enc2 = ResidualBlock3D(
            base,
            base * 2
        )

        self.enc3 = ResidualBlock3D(
            base * 2,
            base * 4
        )

        self.enc4 = ResidualBlock3D(
            base * 4,
            base * 8
        )

        # Bottleneck
        self.bottleneck = ResidualBlock3D(
            base * 8,
            base * 16
        )

        # Decoder
        self.up4 = nn.ConvTranspose3d(
            base * 16,
            base * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ResidualBlock3D(
            base * 16,
            base * 8
        )

        self.up3 = nn.ConvTranspose3d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ResidualBlock3D(
            base * 8,
            base * 4
        )

        self.up2 = nn.ConvTranspose3d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ResidualBlock3D(
            base * 4,
            base * 2
        )

        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ResidualBlock3D(
            base * 2,
            base
        )

        self.output = nn.Conv3d(
            base,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 4. COMBINED BCE + DICE LOSS
# ============================================================

def combined_loss(logits, target, pos_weight):

    logits = logits.float()
    target = target.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight
    )

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * target
    ).sum()

    denominator = (
        probabilities.sum() + target.sum()
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    dice_loss = 1.0 - dice

    return 0.5 * bce + 0.5 * dice_loss


# ============================================================
# 5. DATASET SPLIT HELPERS
# ============================================================

def find_column(df, candidates, required=True):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    if required:

        raise ValueError(
            "Could not find a required CSV column.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_split(value):

    value = str(value).strip().lower()

    if value in ("train", "training"):
        return "train"

    if value in ("val", "valid", "validation", "validating"):
        return "validation"

    if value in ("test", "testing"):
        return "test"

    raise ValueError(
        f"Unknown dataset split value: {value}"
    )


def clean_string(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in ("nan", "none", "null"):
        return None

    return value


def uid_from_filename(filename):

    name = str(filename)

    for suffix in (
        "_ct.nii.gz",
        "_pa.nii.gz",
        ".nii.gz",
        ".nii"
    ):

        if name.lower().endswith(suffix):
            return name[:-len(suffix)]

    return Path(name).stem


def resolve_dataset_path(
    value,
    expected_directory,
    uid,
    expected_suffix
):

    # First try the path supplied by the CSV.
    if value is not None:

        path_value = clean_string(value)

        if path_value:

            supplied = Path(path_value)

            candidates = []

            if supplied.is_absolute():
                candidates.append(supplied)
            else:
                candidates.append(
                    PROJECT_ROOT / supplied
                )

                candidates.append(supplied)

            # Also try the filename in its expected folder.
            candidates.append(
                expected_directory / supplied.name
            )

            for candidate in candidates:

                if candidate.is_file():
                    return candidate.resolve()

    # Otherwise reconstruct the path from UID.
    if uid is not None:

        candidate = (
            expected_directory /
            f"{uid}{expected_suffix}"
        )

        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve processed dataset file.\n"
        f"UID: {uid}\n"
        f"CSV path: {value}\n"
        f"Expected folder: {expected_directory}"
    )


def load_splits():

    if not SPLIT_CSV.is_file():

        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_CSV}"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory missing: {IMAGE_DIR}"
        )

    if not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Mask directory missing: {MASK_DIR}"
        )

    df = pd.read_csv(SPLIT_CSV)

    split_col = find_column(
        df,
        [
            "split",
            "partition",
            "data_split",
            "dataset_split"
        ]
    )

    uid_col = find_column(
        df,
        [
            "uid",
            "case_uid",
            "case_id",
            "seriesuid",
            "series_uid",
            "series_instance_uid",
            "patient_id",
            "case",
            "id"
        ],
        required=False
    )

    image_col = find_column(
        df,
        [
            "image_path",
            "ct_path",
            "image_file",
            "ct_file",
            "image"
        ],
        required=False
    )

    mask_col = find_column(
        df,
        [
            "mask_path",
            "pa_mask_path",
            "label_path",
            "mask_file",
            "label_file",
            "mask"
        ],
        required=False
    )

    patient_col = find_column(
        df,
        [
            "patient_id",
            "subject_id",
            "patient",
            "subject"
        ],
        required=False
    )

    train_records = []
    validation_records = []

    test_count = 0

    # Test records are counted only.
    # Their image and mask paths are deliberately not resolved.
    for _, row in df.iterrows():

        split = normalize_split(
            row[split_col]
        )

        if split == "test":

            test_count += 1
            continue

        uid = None

        if uid_col is not None:
            uid = clean_string(row[uid_col])

        image_value = (
            row[image_col]
            if image_col is not None
            else None
        )

        mask_value = (
            row[mask_col]
            if mask_col is not None
            else None
        )

        # Derive UID from image filename if necessary.
        if uid is None and image_col is not None:

            image_text = clean_string(image_value)

            if image_text:
                uid = uid_from_filename(
                    Path(image_text).name
                )

        if uid is None and mask_col is not None:

            mask_text = clean_string(mask_value)

            if mask_text:
                uid = uid_from_filename(
                    Path(mask_text).name
                )

        if uid is None:

            raise ValueError(
                "Could not determine case UID for a "
                "training or validation row."
            )

        image_path = resolve_dataset_path(
            image_value,
            IMAGE_DIR,
            uid,
            "_ct.nii.gz"
        )

        mask_path = resolve_dataset_path(
            mask_value,
            MASK_DIR,
            uid,
            "_pa.nii.gz"
        )

        if patient_col is not None:

            patient_id = clean_string(
                row[patient_col]
            )

        else:

            patient_id = uid

        record = {
            "uid": uid,
            "patient_id": patient_id or uid,
            "image": image_path,
            "mask": mask_path
        }

        if split == "train":
            train_records.append(record)

        elif split == "validation":
            validation_records.append(record)

    if not train_records:
        raise RuntimeError("No training cases found.")

    if not validation_records:
        raise RuntimeError("No validation cases found.")

    # Check for patient-level leakage.
    train_patients = {
        r["patient_id"] for r in train_records
    }

    validation_patients = {
        r["patient_id"] for r in validation_records
    }

    overlap = train_patients.intersection(
        validation_patients
    )

    if overlap:

        raise RuntimeError(
            "Patient leakage detected between training "
            f"and validation: {list(overlap)[:5]}"
        )

    return (
        train_records,
        validation_records,
        test_count
    )


# ============================================================
# 6. READ ONE CASE
# ============================================================

def load_case(record):

    image_obj = sitk.ReadImage(
        str(record["image"])
    )

    mask_obj = sitk.ReadImage(
        str(record["mask"])
    )

    # IMPORTANT:
    # GetArrayFromImage creates independent NumPy arrays.
    # Do not replace it with GetArrayViewFromImage.
    # Views can reference invalid memory after local
    # SimpleITK image objects are destroyed.

    image = sitk.GetArrayFromImage(image_obj)
    mask = sitk.GetArrayFromImage(mask_obj)

    image = np.asarray(
        image,
        dtype=np.float32
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8
    )

    del image_obj
    del mask_obj

    if image.shape != mask.shape:

        raise ValueError(
            f"Shape mismatch in {record['uid']}: "
            f"{image.shape} vs {mask.shape}"
        )

    if not np.isfinite(image).all():

        raise ValueError(
            f"Non-finite CT values: {record['uid']}"
        )

    if not np.any(mask):

        raise ValueError(
            f"Empty PA mask: {record['uid']}"
        )

    if not np.isin(
        np.unique(mask),
        [0, 1]
    ).all():

        raise ValueError(
            f"Mask is not binary: {record['uid']}"
        )

    return image, mask


# ============================================================
# 7. PATCH SAMPLING
# ============================================================

def get_crop_bounds(
    shape,
    center,
    patch_size
):

    starts = []

    for dim, coordinate in zip(shape, center):

        if dim < patch_size:

            raise ValueError(
                f"Volume dimension {dim} is smaller "
                f"than patch size {patch_size}."
            )

        start = (
            int(coordinate) - patch_size // 2
        )

        start = max(
            0,
            min(start, dim - patch_size)
        )

        starts.append(start)

    slices = tuple(
        slice(s, s + patch_size)
        for s in starts
    )

    return slices


def sample_background_center(
    mask,
    rng,
    patch_size
):

    shape = mask.shape

    best_center = None
    best_fraction = float("inf")

    # Try to find a patch with minimal foreground.
    for _ in range(50):

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

        foreground_fraction = (
            np.count_nonzero(patch_mask)
            / patch_mask.size
        )

        if foreground_fraction < best_fraction:

            best_fraction = foreground_fraction
            best_center = center

        if foreground_fraction == 0.0:

            return center

    # Return the lowest-foreground candidate found.
    return best_center


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

    expected_shape = (
        patch_size,
        patch_size,
        patch_size
    )

    if x.shape != expected_shape:

        raise ValueError(
            f"Unexpected patch shape: {x.shape}"
        )

    # Clip CT intensity.
    x = np.clip(
        x,
        window_low,
        window_high
    )

    # Normalize to [0, 1].
    x = (
        (x - window_low)
        / (window_high - window_low)
    ).astype(np.float32)

    y = y.astype(np.float32)

    return x, y


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

def augment_patch(x, y):

    # Tensor shape:
    # (B, C, D, H, W)
    #
    # Spatial dimensions:
    # D = 2
    # H = 3
    # W = 4

    for dimension in (2, 3, 4):

        if random.random() < 0.5:

            x = torch.flip(
                x,
                dims=(dimension,)
            )

            y = torch.flip(
                y,
                dims=(dimension,)
            )

    # Rotate within the H-W plane.
    rotations = random.randint(0, 3)

    if rotations:

        x = torch.rot90(
            x,
            rotations,
            dims=(3, 4)
        )

        y = torch.rot90(
            y,
            rotations,
            dims=(3, 4)
        )

    # Mild intensity augmentation.
    scale = random.uniform(
        0.90,
        1.10
    )

    shift = random.uniform(
        -0.05,
        0.05
    )

    x = x * scale + shift

    x = x + torch.randn_like(x) * 0.01

    x = torch.clamp(
        x,
        0.0,
        1.0
    )

    return x, y


# ============================================================
# 9. TRAIN ONE EPOCH
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

    order = rng.permutation(
        len(train_records)
    )

    total_loss = 0.0
    total_steps = 0

    expected_steps = (
        len(train_records) * args.patches_per_case
    )

    for case_number, index in enumerate(
        order,
        start=1
    ):

        record = train_records[index]

        # Load each case once per epoch.
        image, mask = load_case(record)

        # More efficient than np.argwhere(mask > 0).
        positive_flat = np.flatnonzero(mask)

        if positive_flat.size == 0:

            raise ValueError(
                f"Empty PA mask: {record['uid']}"
            )

        for patch_number in range(
            1,
            args.patches_per_case + 1
        ):

            if (
                rng.random()
                < args.foreground_probability
            ):

                # Select an actual PA foreground voxel.
                flat_index = int(
                    positive_flat[
                        rng.integers(
                            positive_flat.size
                        )
                    ]
                )

                center = tuple(
                    int(v) for v in
                    np.unravel_index(
                        flat_index,
                        mask.shape
                    )
                )

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

            x = torch.from_numpy(
                x_np
            ).unsqueeze(0).unsqueeze(0)

            y = torch.from_numpy(
                y_np
            ).unsqueeze(0).unsqueeze(0)

            x = x.to(device)
            y = y.to(device)

            x, y = augment_patch(x, y)

            optimizer.zero_grad(
                set_to_none=True
            )

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

                if not torch.isfinite(loss):

                    raise RuntimeError(
                        "Non-finite training loss detected."
                    )

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

            except torch.cuda.OutOfMemoryError as exc:

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "GPU memory exhausted. "
                    "Try --patch-size 64."
                ) from exc

            current_loss = float(
                loss.detach().item()
            )

            total_loss += current_loss
            total_steps += 1

            if (
                total_steps % 20 == 0
                or total_steps == expected_steps
            ):

                print(
                    f"  Train "
                    f"{total_steps:04d}/{expected_steps} "
                    f"case={case_number}/{len(train_records)} "
                    f"patch={patch_number}/"
                    f"{args.patches_per_case} "
                    f"loss={current_loss:.4f}",
                    flush=True
                )

            del x, y, logits, loss
            del x_np, y_np

        # Release the complete volume.
        del image
        del mask
        del positive_flat

    return total_loss / max(total_steps, 1)


# ============================================================
# 10. DETERMINISTIC VALIDATION
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
                stable_seed(
                    record["uid"],
                    args.seed
                )
            )

            positive_flat = np.flatnonzero(mask)

            if positive_flat.size == 0:

                raise ValueError(
                    f"Empty validation mask: {record['uid']}"
                )

            number_of_patches = (
                args.validation_patches
            )

            # Approximately two-thirds foreground.
            number_of_foreground = max(
                1,
                int(
                    round(
                        number_of_patches * 2.0 / 3.0
                    )
                )
            )

            number_of_foreground = min(
                number_of_foreground,
                number_of_patches
            )

            foreground_segments = np.array_split(
                positive_flat,
                number_of_foreground
            )

            centers = []

            # Sample from separate portions of foreground
            # indices to increase spatial coverage.
            for segment in foreground_segments:

                if segment.size == 0:
                    continue

                flat_index = int(
                    segment[
                        rng.integers(
                            segment.size
                        )
                    ]
                )

                center = tuple(
                    int(v) for v in
                    np.unravel_index(
                        flat_index,
                        mask.shape
                    )
                )

                centers.append(center)

            number_of_background = (
                number_of_patches - len(centers)
            )

            for _ in range(number_of_background):

                center = sample_background_center(
                    mask,
                    rng,
                    args.patch_size
                )

                centers.append(center)

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

                probabilities = torch.sigmoid(
                    logits.float()
                )

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

                del x, y, logits, loss
                del prediction, probabilities
                del x_np, y_np

            # Aggregate overlapping patches within each case.
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

            if (
                case_index % 5 == 0
                or case_index == len(validation_records)
            ):

                print(
                    f"  Validation "
                    f"{case_index}/"
                    f"{len(validation_records)} "
                    f"patches={len(centers)}",
                    flush=True
                )

            del image
            del mask
            del positive_flat
            del centers

    return {
        "loss": (
            total_val_loss / max(total_patches, 1)
        ),
        "dice": float(np.mean(case_dice)),
        "precision": float(np.mean(case_precision)),
        "recall": float(np.mean(case_recall))
    }


# ============================================================
# 11. MAIN TRAINING FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NeuroVessel V5B training"
    )

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
        "--patches-per-case",
        type=int,
        default=2
    )

    parser.add_argument(
        "--validation-patches",
        type=int,
        default=6
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

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.window_high <= args.window_low:

        raise ValueError(
            "Invalid intensity window."
        )

    if (
        args.patch_size < 32
        or args.patch_size % 16 != 0
    ):

        raise ValueError(
            "Patch size must be >=32 and divisible by 16."
        )

    if args.patches_per_case < 1:

        raise ValueError(
            "Patches per case must be at least 1."
        )

    if args.validation_patches < 2:

        raise ValueError(
            "Validation patches must be at least 2."
        )

    if not 0.0 <= args.foreground_probability <= 1.0:

        raise ValueError(
            "Foreground probability must be between 0 and 1."
        )

    if args.epochs < 1:

        raise ValueError(
            "Epochs must be at least 1."
        )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Smoke test configuration
    # --------------------------------------------------------

    if args.smoke_test:

        args.patch_size = 64
        args.epochs = 1

        args.patches_per_case = 1
        args.validation_patches = 2

        print("\nSMOKE TEST MODE")
        print("Training cases: 4")
        print("Validation cases: 2")
        print("Patch size: 64")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_records,
        validation_records,
        test_count
    ) = load_splits()

    print("\nDATASET SPLITS")

    print(
        "Training:",
        len(train_records)
    )

    print(
        "Validation:",
        len(validation_records)
    )

    print(
        "Locked test cases:",
        test_count
    )

    print(
        "\nThe test images and masks will not be loaded."
    )

    if args.smoke_test:

        train_records = train_records[:4]
        validation_records = validation_records[:2]

    else:

        if len(train_records) != 202:

            raise RuntimeError(
                f"Expected 202 training cases, "
                f"got {len(train_records)}"
            )

        if len(validation_records) != 26:

            raise RuntimeError(
                f"Expected 26 validation cases, "
                f"got {len(validation_records)}"
            )

        if test_count != 26:

            raise RuntimeError(
                f"Expected 26 locked test cases, "
                f"got {test_count}"
            )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NEUROVESSEL V5B TRAINING")
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
                torch.cuda.get_device_properties(
                    0
                ).total_memory / 1024**3,
                2
            ),
            "GB"
        )

    print("Architecture: Residual 3D U-Net")
    print("Base channels: 8")
    print("Patch:", args.patch_size)
    print("Batch size: 1")
    print("AMP:", amp)

    print(
        "Training patches per patient:",
        args.patches_per_case
    )

    print(
        "Validation patches per patient:",
        args.validation_patches
    )

    print(
        "Intensity window:",
        args.window_low,
        args.window_high
    )

    print("Epochs:", args.epochs)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NeuroVesselV5(
        in_channels=1,
        out_channels=1,
        base=8
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    patience = 8

    epochs_without_improvement = 0

    best_val_dice = -1.0

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    if args.smoke_test:

        run_name = "airrc_v5b_smoke"

    else:

        run_name = (
            "airrc_v5b_wm1000_wp1000"
        )

    best_path = (
        CHECKPOINT_DIR /
        f"{run_name}_best.pt"
    )

    last_path = (
        CHECKPOINT_DIR /
        f"{run_name}_last.pt"
    )

    history_path = (
        METRICS_DIR /
        f"{run_name}_history.csv"
    )

    config_path = (
        METRICS_DIR /
        f"{run_name}_config.json"
    )

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

    config = {

        "project": "NeuroVessel",

        "version": "V5B",

        "task": "Binary pulmonary artery segmentation",

        "architecture": "Residual 3D U-Net",

        "base_channels": 8,

        "trainable_parameters": parameter_count,

        "patch_size": args.patch_size,

        "batch_size": 1,

        "patches_per_case": args.patches_per_case,

        "validation_patches_per_case": (
            args.validation_patches
        ),

        "epochs_requested": args.epochs,

        "learning_rate": args.lr,

        "window_low": args.window_low,

        "window_high": args.window_high,

        "foreground_probability": (
            args.foreground_probability
        ),

        "positive_bce_weight": 3.0,

        "optimizer": "AdamW",

        "scheduler": "ReduceLROnPlateau",

        "loss": "0.5 BCE + 0.5 Soft Dice",

        "seed": args.seed,

        "amp": amp,

        "device": str(device),

        "smoke_test": args.smoke_test,

        "validation_method": (
            "Deterministic multi-patch validation. "
            "Per-patient patch counts are aggregated, "
            "then macro-averaged. "
            "This is not full-volume Dice."
        ),

        "test_partition_loaded": False

    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # CSV history
    # --------------------------------------------------------

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

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            "=" * 50
        )

        # Training
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

        # Validation
        val_metrics = validate(
            model,
            validation_records,
            device,
            args,
            amp,
            pos_weight
        )

        scheduler.step(
            val_metrics["dice"]
        )

        lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time() - epoch_start
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print("\nEPOCH RESULTS")

        print(
            "Training loss:",
            round(train_loss, 5)
        )

        print(
            "Validation loss:",
            round(val_metrics["loss"], 5)
        )

        print(
            "Validation patch Dice:",
            round(val_metrics["dice"], 5)
        )

        print(
            "Validation precision:",
            round(val_metrics["precision"], 5)
        )

        print(
            "Validation recall:",
            round(val_metrics["recall"], 5)
        )

        print(
            "Learning rate:",
            lr
        )

        print(
            "Time:",
            round(elapsed, 1),
            "seconds"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        improved = (
            val_metrics["dice"] > best_val_dice
        )

        if improved:

            best_val_dice = (
                val_metrics["dice"]
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        state = {

            "epoch": epoch,

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
            ),

            "best_val_patch_dice": (
                best_val_dice
            ),

            "current_val_metrics": val_metrics,

            "config": config

        }

        # Save latest checkpoint.
        torch.save(
            state,
            last_path
        )

        # Save best checkpoint.
        if improved:

            torch.save(
                state,
                best_path
            )

            print(
                "\nBEST CHECKPOINT UPDATED:",
                best_path
            )

        else:

            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                patience
            )

        # ----------------------------------------------------
        # Memory cleanup
        # ----------------------------------------------------

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            not args.smoke_test
            and epochs_without_improvement >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        "Best validation patch Dice:",
        best_val_dice
    )

    print(
        "Best checkpoint:",
        best_path
    )

    print(
        "Latest checkpoint:",
        last_path
    )

    print(
        "Training history:",
        history_path
    )

    print(
        "Configuration:",
        config_path
    )

    print(
        "\nIMPORTANT: These are patch-level validation metrics, "
        "not full-volume evaluation results."
    )

    print(
        "The test partition remains untouched."
    )


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
