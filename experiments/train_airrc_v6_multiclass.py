
"""
NEUROVESSEL V6
THREE-CLASS PULMONARY ARTERY / PULMONARY VEIN SEGMENTATION

Classes:
    0 = Background (AirRC original classes 0, 1 and 2)
    1 = Pulmonary artery (AirRC original class 3)
    2 = Pulmonary vein (AirRC original class 4)

Architecture:
    Residual 3D U-Net
    Input channels: 1
    Output channels: 3

Training:
    One PA-centred patch and one PV-centred patch
    per training patient per epoch.

Validation:
    Deterministic PA-centred and PV-centred patches.
    Separate PA Dice and PV Dice.
    Combined foreground Dice for checkpoint selection.

IMPORTANT:
    This is a patch-level validation procedure.
    It is not full-volume evaluation.

    The locked test partition is never loaded.
    V5B checkpoints and outputs are not overwritten.
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

from train_airrc_v5b import (
    load_splits,
    PROJECT_ROOT,
    METRICS_DIR,
    CHECKPOINT_DIR
)


LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_airrc_v6"
    / "labels_multiclass"
)

V6_METRICS_DIR = (
    METRICS_DIR
    / "airrc_v6_multiclass"
)

V6_CHECKPOINT_DIR = CHECKPOINT_DIR

V6_METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V6_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CLASS CONFIGURATION
# ============================================================

NUM_CLASSES = 3

BACKGROUND = 0
PULMONARY_ARTERY = 1
PULMONARY_VEIN = 2

CLASS_NAMES = {
    0: "Background",
    1: "Pulmonary_Artery",
    2: "Pulmonary_Vein"
}

# Starting experimental class weights.
# Background is downweighted because it occupies
# most of the full volume.
#
# PA and PV receive equal weight because their
# annotation voxel counts are nearly equal.

CLASS_WEIGHTS = [0.20, 1.0, 1.0]


# ============================================================
# 4. RESIDUAL 3D U-NET
# ============================================================

def get_group_count(channels):

    groups = min(8, channels)

    while channels % groups != 0:
        groups -= 1

    return groups


class ResidualBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        groups = get_group_count(
            out_channels
        )

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

        return self.activation(out)


class NeuroVesselV6(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=3,
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

        # Three output classes:
        # Background, PA and PV.
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
        d4 = torch.cat(
            (d4, e4),
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            (d3, e3),
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            (d2, e2),
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            (d1, e1),
            dim=1
        )
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 5. MULTICLASS LOSS
# ============================================================

def multiclass_loss(
    logits,
    target,
    class_weights
):

    """
    Combination:
        0.5 weighted multiclass cross-entropy
        0.5 foreground multiclass soft Dice loss

    Background is excluded from the foreground Dice average.
    PA and PV are treated as separate foreground classes.
    """

    logits = logits.float()

    target = target.long()

    # Weighted cross-entropy.
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights
    )

    # Softmax probabilities.
    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # One-hot target:
    # (B,D,H,W) -> (B,C,D,H,W)
    one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES
    )

    one_hot = one_hot.permute(
        0, 4, 1, 2, 3
    ).float()

    dimensions = (
        0, 2, 3, 4
    )

    intersection = (
        probabilities * one_hot
    ).sum(
        dim=dimensions
    )

    denominator = (
        probabilities.sum(
            dim=dimensions
        )
        +
        one_hot.sum(
            dim=dimensions
        )
    )

    dice_per_class = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    # Only PA and PV contribute to foreground Dice loss.
    foreground_dice = dice_per_class[
        1:
    ].mean()

    dice_loss = (
        1.0 - foreground_dice
    )

    return (
        0.5 * ce
        + 0.5 * dice_loss
    )


# ============================================================
# 6. DATASET PATHS AND SPLITS
# ============================================================

def multiclass_label_path(uid):

    return (
        LABEL_DIR
        / f"{uid}_multiclass.nii.gz"
    )


def load_v6_splits():

    (
        train_records,
        validation_records,
        test_count
    ) = load_splits()

    if len(train_records) != 202:

        raise RuntimeError(
            f"Expected 202 training patients, "
            f"got {len(train_records)}"
        )

    if len(validation_records) != 26:

        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"got {len(validation_records)}"
        )

    if test_count != 26:

        raise RuntimeError(
            f"Expected 26 locked test patients, "
            f"got {test_count}"
        )

    # Patient-level split check.
    train_patients = {
        r["patient_id"]
        for r in train_records
    }

    val_patients = {
        r["patient_id"]
        for r in validation_records
    }

    overlap = train_patients.intersection(
        val_patients
    )

    if overlap:

        raise RuntimeError(
            "Patient leakage detected."
        )

    # Check all training and validation label paths.
    missing = []

    for record in (
        train_records + validation_records
    ):

        path = multiclass_label_path(
            record["uid"]
        )

        if not path.is_file():

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "Missing V6 multiclass labels:\n"
            + "\n".join(missing[:10])
        )

    return (
        train_records,
        validation_records,
        test_count
    )


# ============================================================
# 7. LOAD ONE V6 PATIENT
# ============================================================

def load_v6_case(record):

    image_obj = sitk.ReadImage(
        str(record["image"])
    )

    label_path = multiclass_label_path(
        record["uid"]
    )

    label_obj = sitk.ReadImage(
        str(label_path)
    )

    # Independent arrays; do not use array views.
    image = sitk.GetArrayFromImage(
        image_obj
    ).astype(
        np.float32,
        copy=False
    )

    labels = sitk.GetArrayFromImage(
        label_obj
    ).astype(
        np.uint8,
        copy=False
    )

    if image.shape != labels.shape:

        raise ValueError(
            f"Image/label shape mismatch: "
            f"{record['uid']} "
            f"{image.shape} vs {labels.shape}"
        )

    if (
        image_obj.GetSize()
        != label_obj.GetSize()
        or not np.allclose(
            image_obj.GetSpacing(),
            label_obj.GetSpacing(),
            atol=1e-5
        )
        or not np.allclose(
            image_obj.GetOrigin(),
            label_obj.GetOrigin(),
            atol=1e-4
        )
        or not np.allclose(
            image_obj.GetDirection(),
            label_obj.GetDirection(),
            atol=1e-5
        )
    ):

        raise ValueError(
            f"Geometry mismatch: {record['uid']}"
        )

    if not np.isfinite(image).all():

        raise ValueError(
            f"Non-finite CT values: {record['uid']}"
        )

    unique_labels = np.unique(
        labels
    )

    if not set(
        unique_labels.tolist()
    ).issubset({0, 1, 2}):

        raise ValueError(
            f"Invalid multiclass labels for "
            f"{record['uid']}: {unique_labels}"
        )

    for class_id in (
        PULMONARY_ARTERY,
        PULMONARY_VEIN
    ):

        if not np.any(labels == class_id):

            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} "
                f"is empty for {record['uid']}"
            )

    del image_obj
    del label_obj

    return image, labels


# ============================================================
# 8. PATCH HELPERS
# ============================================================

def get_crop_bounds(
    shape,
    center,
    patch_size
):

    starts = []

    for dim, coordinate in zip(
        shape,
        center
    ):

        if dim < patch_size:

            raise ValueError(
                f"Volume dimension {dim} "
                f"is smaller than patch {patch_size}."
            )

        start = (
            int(coordinate)
            - patch_size // 2
        )

        start = max(
            0,
            min(start, dim - patch_size)
        )

        starts.append(start)

    return tuple(
        slice(start, start + patch_size)
        for start in starts
    )


def sample_class_center(
    labels,
    class_id,
    rng
):

    flat_indices = np.flatnonzero(
        labels == class_id
    )

    if flat_indices.size == 0:

        raise ValueError(
            f"No voxels found for class {class_id}"
        )

    selected = int(
        flat_indices[
            rng.integers(
                flat_indices.size
            )
        ]
    )

    center = tuple(
        int(v) for v in
        np.unravel_index(
            selected,
            labels.shape
        )
    )

    del flat_indices

    return center


def extract_patch(
    image,
    labels,
    center,
    patch_size,
    window_low=-1000.0,
    window_high=1000.0
):

    slices = get_crop_bounds(
        image.shape,
        center,
        patch_size
    )

    x = image[slices].copy()

    y = labels[slices].copy()

    expected_shape = (
        patch_size,
        patch_size,
        patch_size
    )

    if x.shape != expected_shape:

        raise ValueError(
            f"Unexpected image patch shape: {x.shape}"
        )

    if y.shape != expected_shape:

        raise ValueError(
            f"Unexpected label patch shape: {y.shape}"
        )

    # Intensity clipping and normalization.
    x = np.clip(
        x,
        window_low,
        window_high
    )

    x = (
        (x - window_low)
        / (window_high - window_low)
    ).astype(
        np.float32,
        copy=False
    )

    y = y.astype(
        np.int64,
        copy=False
    )

    return x, y


# ============================================================
# 9. AUGMENTATION
# ============================================================

def augment_patch(x, y):

    # x shape: (B,C,D,H,W)
    # y shape: (B,D,H,W)

    # Three spatial flips.
    for x_dimension in (2, 3, 4):

        if random.random() < 0.5:

            y_dimension = (
                x_dimension - 1
            )

            x = torch.flip(
                x,
                dims=(x_dimension,)
            )

            y = torch.flip(
                y,
                dims=(y_dimension,)
            )

    # H-W rotations.
    rotations = random.randint(
        0, 3
    )

    if rotations:

        x = torch.rot90(
            x,
            rotations,
            dims=(3, 4)
        )

        y = torch.rot90(
            y,
            rotations,
            dims=(2, 3)
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

    x = (
        x
        + torch.randn_like(x) * 0.01
    )

    x = torch.clamp(
        x,
        0.0,
        1.0
    )

    return x, y


# ============================================================
# 10. SEED HELPER
# ============================================================

def stable_seed(uid, base_seed):

    digest = hashlib.sha256(
        uid.encode("utf-8")
    ).hexdigest()

    return (
        int(digest[:8], 16)
        + base_seed
    ) % (2**32)


# ============================================================
# 11. TRAIN ONE EPOCH
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
    class_weights
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

    patches_per_case = 2

    expected_steps = (
        len(train_records)
        * patches_per_case
    )

    for case_number, index in enumerate(
        order,
        start=1
    ):

        record = train_records[index]

        # Load patient volume only once.
        image, labels = load_v6_case(
            record
        )

        # One patch for PA and one for PV.
        patch_classes = [
            PULMONARY_ARTERY,
            PULMONARY_VEIN
        ]

        rng.shuffle(
            patch_classes
        )

        for patch_class in patch_classes:

            center = sample_class_center(
                labels,
                patch_class,
                rng
            )

            x_np, y_np = extract_patch(
                image,
                labels,
                center,
                args.patch_size
            )

            # Tensor shapes:
            # X = (1,1,D,H,W)
            # Y = (1,D,H,W)

            x = torch.from_numpy(
                x_np
            ).unsqueeze(0).unsqueeze(0)

            y = torch.from_numpy(
                y_np
            ).unsqueeze(0)

            x = x.to(device)
            y = y.to(device)

            x, y = augment_patch(
                x,
                y
            )

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

                    loss = multiclass_loss(
                        logits,
                        y,
                        class_weights
                    )

                if not torch.isfinite(loss):

                    raise RuntimeError(
                        "Non-finite V6 training loss."
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            except torch.cuda.OutOfMemoryError as exc:

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "V6 GPU memory exhausted. "
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
                    f"  Train {total_steps:04d}/"
                    f"{expected_steps} "
                    f"case={case_number}/"
                    f"{len(train_records)} "
                    f"class={CLASS_NAMES[patch_class]} "
                    f"loss={current_loss:.4f}",
                    flush=True
                )

            del x, y, logits, loss
            del x_np, y_np

        del image
        del labels

    return (
        total_loss / max(total_steps, 1)
    )


# ============================================================
# 12. VALIDATION
# ============================================================

def validate(
    model,
    validation_records,
    device,
    args,
    amp,
    class_weights
):

    model.eval()

    patient_pa_dice = []
    patient_pv_dice = []

    total_val_loss = 0.0
    total_patches = 0

    with torch.inference_mode():

        for case_index, record in enumerate(
            validation_records,
            start=1
        ):

            image, labels = load_v6_case(
                record
            )

            rng = np.random.default_rng(
                stable_seed(
                    record["uid"],
                    args.seed
                )
            )

            # Per-patient accumulated class counts.
            intersections = {
                1: 0,
                2: 0
            }

            predicted_counts = {
                1: 0,
                2: 0
            }

            target_counts = {
                1: 0,
                2: 0
            }

            # Sample a deterministic number of patches
            # centred on each foreground class.
            for class_id in (
                PULMONARY_ARTERY,
                PULMONARY_VEIN
            ):

                flat_indices = np.flatnonzero(
                    labels == class_id
                )

                count = min(
                    args.validation_samples_per_class,
                    flat_indices.size
                )

                selected_indices = rng.choice(
                    flat_indices,
                    size=count,
                    replace=False
                )

                for selected in selected_indices:

                    center = tuple(
                        int(v) for v in
                        np.unravel_index(
                            int(selected),
                            labels.shape
                        )
                    )

                    x_np, y_np = extract_patch(
                        image,
                        labels,
                        center,
                        args.patch_size
                    )

                    x = torch.from_numpy(
                        x_np
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    y = torch.from_numpy(
                        y_np
                    ).unsqueeze(0).to(device)

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp
                    ):

                        logits = model(x)

                        loss = multiclass_loss(
                            logits,
                            y,
                            class_weights
                        )

                    probabilities = torch.softmax(
                        logits.float(),
                        dim=1
                    )

                    prediction = (
                        probabilities.argmax(
                            dim=1
                        )
                    )

                    for metric_class in (
                        PULMONARY_ARTERY,
                        PULMONARY_VEIN
                    ):

                        predicted_class = (
                            prediction == metric_class
                        )

                        target_class = (
                            y == metric_class
                        )

                        intersections[metric_class] += int(
                            (
                                predicted_class
                                & target_class
                            ).sum().item()
                        )

                        predicted_counts[metric_class] += int(
                            predicted_class.sum().item()
                        )

                        target_counts[metric_class] += int(
                            target_class.sum().item()
                        )

                    total_val_loss += float(
                        loss.item()
                    )

                    total_patches += 1

                    del x, y, logits, loss
                    del probabilities, prediction
                    del x_np, y_np

                del flat_indices
                del selected_indices

            # Per-patient PA Dice.
            pa_denom = (
                predicted_counts[1]
                + target_counts[1]
            )

            pa_dice = (
                (2.0 * intersections[1] + 1.0)
                / (pa_denom + 1.0)
            )

            # Per-patient PV Dice.
            pv_denom = (
                predicted_counts[2]
                + target_counts[2]
            )

            pv_dice = (
                (2.0 * intersections[2] + 1.0)
                / (pv_denom + 1.0)
            )

            patient_pa_dice.append(
                pa_dice
            )

            patient_pv_dice.append(
                pv_dice
            )

            if (
                case_index % 5 == 0
                or case_index == len(validation_records)
            ):

                print(
                    f"  Validation "
                    f"{case_index}/"
                    f"{len(validation_records)}",
                    flush=True
                )

            del image
            del labels

    macro_pa_dice = float(
        np.mean(patient_pa_dice)
    )

    macro_pv_dice = float(
        np.mean(patient_pv_dice)
    )

    mean_foreground_dice = (
        macro_pa_dice
        + macro_pv_dice
    ) / 2.0

    return {

        "loss": (
            total_val_loss
            / max(total_patches, 1)
        ),

        "pa_patch_dice": macro_pa_dice,

        "pv_patch_dice": macro_pv_dice,

        "mean_foreground_patch_dice": (
            mean_foreground_dice
        )

    }


# ============================================================
# 13. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NeuroVessel V6 multiclass training"
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
        "--validation-samples-per-class",
        type=int,
        default=2
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
    # Validate arguments.
    # --------------------------------------------------------

    if (
        args.patch_size < 32
        or args.patch_size % 16 != 0
    ):

        raise ValueError(
            "Patch size must be >=32 and divisible by 16."
        )

    if args.epochs < 1:

        raise ValueError(
            "Epochs must be at least 1."
        )

    if args.lr <= 0:

        raise ValueError(
            "Learning rate must be positive."
        )

    if args.validation_samples_per_class < 1:

        raise ValueError(
            "Validation samples per class must be >=1."
        )

    # --------------------------------------------------------
    # Reproducibility.
    # --------------------------------------------------------

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = (
        device.type == "cuda"
    )

    # --------------------------------------------------------
    # Smoke-test configuration.
    # --------------------------------------------------------

    if args.smoke_test:

        args.patch_size = 64

        args.epochs = 1

        args.validation_samples_per_class = 1

        print("\nSMOKE TEST MODE")

        print(
            "Training patients: 4"
        )

        print(
            "Validation patients: 2"
        )

        print(
            "Patch size: 64"
        )

    # --------------------------------------------------------
    # Dataset.
    # --------------------------------------------------------

    (
        train_records,
        validation_records,
        test_count
    ) = load_v6_splits()

    if args.smoke_test:

        train_records = train_records[:4]

        validation_records = validation_records[:2]

    print("\nDATASET SPLITS")

    print(
        "Training patients:",
        len(train_records)
    )

    print(
        "Validation patients:",
        len(validation_records)
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "Test images and labels will not be loaded."
    )

    # --------------------------------------------------------
    # Training information.
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "NEUROVESSEL V6 — MULTICLASS TRAINING"
    )

    print("=" * 65)

    print(
        "Device:",
        device
    )

    if amp:

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**3,
                2
            ),
            "GB"
        )

    print(
        "Architecture: Residual 3D U-Net"
    )

    print(
        "Input channels: 1"
    )

    print(
        "Output classes: 3"
    )

    print(
        "Class mapping:",
        CLASS_NAMES
    )

    print(
        "Base channels: 8"
    )

    print(
        "Patch size:",
        args.patch_size
    )

    print(
        "Training patches per patient: 2"
    )

    print(
        "Training patch strategy: "
        "1 PA-centred + 1 PV-centred"
    )

    print(
        "Expected updates per epoch:",
        len(train_records) * 2
    )

    print(
        "Validation samples per class:",
        args.validation_samples_per_class
    )

    print(
        "Class weights:",
        CLASS_WEIGHTS
    )

    print(
        "Loss: 0.5 Weighted CE + 0.5 Foreground Dice"
    )

    print(
        "Intensity window: -1000 to 1000"
    )

    print(
        "AMP:",
        amp
    )

    print(
        "Epochs:",
        args.epochs
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = NeuroVesselV6(
        in_channels=1,
        out_channels=NUM_CLASSES,
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
    # Loss weights.
    # --------------------------------------------------------

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Optimizer and scheduler.
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

    # --------------------------------------------------------
    # Early stopping.
    # --------------------------------------------------------

    patience = 8

    epochs_without_improvement = 0

    best_val_dice = -1.0

    # --------------------------------------------------------
    # Output paths.
    # --------------------------------------------------------

    if args.smoke_test:

        run_name = (
            "airrc_v6_multiclass_smoke"
        )

    else:

        run_name = (
            "airrc_v6_multiclass"
        )

    best_path = (
        V6_CHECKPOINT_DIR
        / f"{run_name}_best.pt"
    )

    last_path = (
        V6_CHECKPOINT_DIR
        / f"{run_name}_last.pt"
    )

    history_path = (
        V6_METRICS_DIR
        / f"{run_name}_history.csv"
    )

    config_path = (
        V6_METRICS_DIR
        / f"{run_name}_config.json"
    )

    # --------------------------------------------------------
    # Configuration.
    # --------------------------------------------------------

    config = {

        "project": "NeuroVessel",

        "version": "V6",

        "task": (
            "Three-class pulmonary artery and "
            "pulmonary vein segmentation"
        ),

        "architecture": "Residual 3D U-Net",

        "base_channels": 8,

        "trainable_parameters": parameter_count,

        "input_channels": 1,

        "output_classes": 3,

        "class_mapping": {

            "0": (
                "Background: original classes 0, 1 and 2"
            ),

            "1": (
                "Pulmonary artery: original class 3"
            ),

            "2": (
                "Pulmonary vein: original class 4"
            )

        },

        "patch_size": args.patch_size,

        "batch_size": 1,

        "patches_per_patient": 2,

        "training_sampling": (
            "One PA-centred patch and one PV-centred "
            "patch per patient per epoch."
        ),

        "validation_samples_per_class": (
            args.validation_samples_per_class
        ),

        "epochs_requested": args.epochs,

        "learning_rate": args.lr,

        "weight_decay": 1e-5,

        "class_weights": CLASS_WEIGHTS,

        "loss": (
            "0.5 weighted cross entropy + "
            "0.5 foreground soft Dice"
        ),

        "window_low": -1000,

        "window_high": 1000,

        "optimizer": "AdamW",

        "scheduler": "ReduceLROnPlateau",

        "amp": amp,

        "device": str(device),

        "seed": args.seed,

        "smoke_test": args.smoke_test,

        "validation_method": (
            "Deterministic class-centred patches. "
            "Separate per-patient PA and PV patch Dice, "
            "then macro-average across patients. "
            "Not full-volume evaluation."
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
    # History CSV.
    # --------------------------------------------------------

    fieldnames = [

        "epoch",

        "train_loss",

        "val_loss",

        "val_pa_patch_dice",

        "val_pv_patch_dice",

        "val_mean_foreground_patch_dice",

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
    # Training loop.
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        print("\n" + "=" * 55)

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print("=" * 55)

        # Training.
        train_loss = train_one_epoch(
            model,
            optimizer,
            scaler,
            train_records,
            device,
            args,
            epoch,
            amp,
            class_weights
        )

        # Validation.
        val_metrics = validate(
            model,
            validation_records,
            device,
            args,
            amp,
            class_weights
        )

        monitor_dice = (
            val_metrics[
                "mean_foreground_patch_dice"
            ]
        )

        scheduler.step(
            monitor_dice
        )

        lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time() - epoch_start
        )

        # ----------------------------------------------------
        # Print metrics.
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
            "PA validation patch Dice:",
            round(
                val_metrics["pa_patch_dice"],
                5
            )
        )

        print(
            "PV validation patch Dice:",
            round(
                val_metrics["pv_patch_dice"],
                5
            )
        )

        print(
            "Mean foreground patch Dice:",
            round(monitor_dice, 5)
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
        # Append history.
        # ----------------------------------------------------

        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_metrics["loss"],

            "val_pa_patch_dice": (
                val_metrics["pa_patch_dice"]
            ),

            "val_pv_patch_dice": (
                val_metrics["pv_patch_dice"]
            ),

            "val_mean_foreground_patch_dice": (
                monitor_dice
            ),

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
        # Best model tracking.
        # ----------------------------------------------------

        improved = (
            monitor_dice > best_val_dice
        )

        if improved:

            best_val_dice = monitor_dice

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Checkpoint state.
        # ----------------------------------------------------

        state = {

            "epoch": epoch,

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
            ),

            "best_val_mean_foreground_patch_dice": (
                best_val_dice
            ),

            "current_val_metrics": (
                val_metrics
            ),

            "config": config

        }

        # Save latest.
        torch.save(
            state,
            last_path
        )

        # Save best.
        if improved:

            torch.save(
                state,
                best_path
            )

            print(
                "\nBEST V6 CHECKPOINT UPDATED:",
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
        # Cleanup.
        # ----------------------------------------------------

        gc.collect()

        if device.type == "cuda":

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Early stopping.
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
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V6 TRAINING FINISHED"
    )

    print("=" * 70)

    print(
        "Best mean foreground validation patch Dice:",
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
        "\nIMPORTANT:"
    )

    print(
        "These are patch-level validation metrics, "
        "not full-volume Dice."
    )

    print(
        "The locked test partition remains untouched."
    )


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
