
"""
NEUROVESSEL V6.1
THREE-CLASS SEGMENTATION WITH EXPLICIT BACKGROUND LEARNING

Classes:
    0 = Background
    1 = Pulmonary artery
    2 = Pulmonary vein

Changes from V6:
    - One additional background-centred patch per patient.
    - Initial CE class weights: [0.50, 1.0, 1.0].

Unchanged:
    - Residual 3D U-Net architecture.
    - PA and PV patch sampling.
    - CT normalization.
    - Augmentation.
    - Optimizer and scheduler.
    - Validation patch sampling method.

V6 outputs are not overwritten.
The locked test partition is never loaded.
"""

import argparse
import csv
import gc
import json
import random
import time

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_airrc_v5b import (
    PROJECT_ROOT,
    METRICS_DIR,
    CHECKPOINT_DIR,
)

from train_airrc_v6_multiclass import (
    NeuroVesselV6,
    load_v6_splits,
    load_v6_case,
    sample_class_center,
    extract_patch,
    augment_patch,
    multiclass_loss,
    validate,
    stable_seed,
    PULMONARY_ARTERY,
    PULMONARY_VEIN,
    CLASS_NAMES,
)


# ============================================================
# 1. V6.1 CONFIGURATION
# ============================================================

BACKGROUND = 0
PA = PULMONARY_ARTERY
PV = PULMONARY_VEIN

NUM_CLASSES = 3

# Initial experimental weights.
# Background receives more CE weight than in V6.
CLASS_WEIGHTS = [0.50, 1.0, 1.0]

# Exclude near-air voxels outside the body when
# choosing background patch centres.
BACKGROUND_MIN_HU = -950.0

V6_1_METRICS_DIR = (
    METRICS_DIR / "airrc_v6_1_multiclass"
)

V6_1_CHECKPOINT_DIR = CHECKPOINT_DIR

V6_1_METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V6_1_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. BACKGROUND CENTRE SAMPLING
# ============================================================

def sample_background_center(
    image,
    labels,
    rng,
    min_hu=BACKGROUND_MIN_HU,
    rounds=50,
    candidates_per_round=4096,
):
    """
    Sample a background voxel whose CT intensity is above
    min_hu. This avoids choosing most centres from external
    near-air regions.

    Uses rejection sampling rather than np.flatnonzero over
    the entire background class, which could allocate a very
    large array for full CT volumes.
    """

    shape = image.shape

    for _ in range(rounds):

        coordinates = np.column_stack([
            rng.integers(0, dim, size=candidates_per_round)
            for dim in shape
        ])

        indices = tuple(coordinates.T)

        candidate_labels = labels[indices]
        candidate_hu = image[indices]

        valid = (
            (candidate_labels == BACKGROUND)
            & (candidate_hu > min_hu)
        )

        valid_coordinates = coordinates[valid]

        if valid_coordinates.shape[0] > 0:

            selected_index = int(
                rng.integers(
                    0,
                    valid_coordinates.shape[0]
                )
            )

            selected = valid_coordinates[selected_index]

            return tuple(int(v) for v in selected)

    raise RuntimeError(
        "Could not find a suitable background centre. "
        "Check the CT intensity range or lower "
        "--background-min-hu."
    )


# ============================================================
# 3. TRAIN ONE V6.1 EPOCH
# ============================================================

def train_one_epoch_v61(
    model,
    optimizer,
    scaler,
    train_records,
    device,
    args,
    epoch,
    amp,
    class_weights,
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

    # Three patches per patient:
    # one PA, one PV and one background.
    patches_per_case = 3

    expected_steps = (
        len(train_records) * patches_per_case
    )

    for case_number, index in enumerate(
        order,
        start=1
    ):

        record = train_records[index]

        image, labels = load_v6_case(record)

        # One patch for each class.
        patch_classes = [
            PA,
            PV,
            BACKGROUND
        ]

        rng.shuffle(patch_classes)

        for patch_class in patch_classes:

            if patch_class == BACKGROUND:

                center = sample_background_center(
                    image,
                    labels,
                    rng,
                    min_hu=args.background_min_hu,
                )

            else:

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

            x = torch.from_numpy(
                x_np
            ).unsqueeze(0).unsqueeze(0)

            y = torch.from_numpy(
                y_np
            ).unsqueeze(0)

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

                    loss = multiclass_loss(
                        logits,
                        y,
                        class_weights
                    )

                if not torch.isfinite(loss):

                    raise RuntimeError(
                        "Non-finite V6.1 training loss."
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
                    "V6.1 GPU memory exhausted. "
                    "Retry with --patch-size 64."
                ) from exc

            current_loss = float(
                loss.detach().item()
            )

            total_loss += current_loss
            total_steps += 1

            if (
                total_steps % 30 == 0
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

        del image, labels

    return total_loss / max(total_steps, 1)


# ============================================================
# 4. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NeuroVessel V6.1 background-aware training"
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
        "--background-min-hu",
        type=float,
        default=BACKGROUND_MIN_HU
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

    if args.patch_size < 32 or args.patch_size % 16 != 0:
        raise ValueError(
            "Patch size must be >=32 and divisible by 16."
        )

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if args.lr <= 0:
        raise ValueError("Learning rate must be positive.")

    if args.validation_samples_per_class < 1:
        raise ValueError(
            "Validation samples per class must be >=1."
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
    # Smoke-test configuration
    # --------------------------------------------------------

    if args.smoke_test:

        args.patch_size = 64
        args.epochs = 1
        args.validation_samples_per_class = 1

        print("\nV6.1 SMOKE TEST MODE")
        print("Training subset: first 4 patients")
        print("Validation subset: first 2 patients")
        print("Patch size: 64")

    # --------------------------------------------------------
    # Dataset splits
    # --------------------------------------------------------

    train_records, validation_records, test_count = (
        load_v6_splits()
    )

    if args.smoke_test:

        train_records = train_records[:4]
        validation_records = validation_records[:2]

    print("\nDATASET SPLITS")
    print("Training patients:", len(train_records))
    print("Validation patients:", len(validation_records))
    print("Locked test patients:", test_count)
    print("Test images and labels will not be loaded.")

    # --------------------------------------------------------
    # Model and configuration
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("NEUROVESSEL V6.1 — BACKGROUND-AWARE TRAINING")
    print("=" * 68)

    print("Device:", device)

    if amp:

        print("GPU:", torch.cuda.get_device_name(0))

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**3,
                2
            ),
            "GB"
        )

    model = NeuroVesselV6(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base=8
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Architecture: Residual 3D U-Net")
    print("Trainable parameters:", f"{parameter_count:,}")
    print("Patch size:", args.patch_size)
    print("Output classes:", CLASS_NAMES)

    print("Patches per patient: 3")
    print("  1 PA-centred")
    print("  1 PV-centred")
    print("  1 background-centred")

    print(
        "Expected updates per epoch:",
        len(train_records) * 3
    )

    print("Class weights:", CLASS_WEIGHTS)
    print("Background centre HU minimum:", args.background_min_hu)
    print("Validation samples per class:", args.validation_samples_per_class)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
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

    patience = 8
    epochs_without_improvement = 0
    best_val_dice = -1.0

    # --------------------------------------------------------
    # V6.1-specific output paths
    # --------------------------------------------------------

    if args.smoke_test:
        run_name = "airrc_v6_1_multiclass_smoke"
    else:
        run_name = "airrc_v6_1_multiclass"

    best_path = (
        V6_1_CHECKPOINT_DIR / f"{run_name}_best.pt"
    )

    last_path = (
        V6_1_CHECKPOINT_DIR / f"{run_name}_last.pt"
    )

    history_path = (
        V6_1_METRICS_DIR / f"{run_name}_history.csv"
    )

    config_path = (
        V6_1_METRICS_DIR / f"{run_name}_config.json"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "project": "NeuroVessel",
        "version": "V6.1",
        "task": "Three-class PA/PV segmentation with explicit background learning",
        "architecture": "Residual 3D U-Net",
        "base_channels": 8,
        "trainable_parameters": parameter_count,
        "input_channels": 1,
        "output_classes": 3,
        "class_mapping": {
            "0": "Background",
            "1": "Pulmonary artery",
            "2": "Pulmonary vein"
        },
        "patch_size": args.patch_size,
        "patches_per_patient": 3,
        "training_sampling": [
            "One PA-centred patch",
            "One PV-centred patch",
            "One background-centred patch with CT intensity above configured HU minimum"
        ],
        "background_min_hu": args.background_min_hu,
        "class_weights": CLASS_WEIGHTS,
        "loss": "0.5 weighted cross entropy + 0.5 foreground soft Dice",
        "window_low": -1000,
        "window_high": 1000,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 1e-5,
        "scheduler": "ReduceLROnPlateau",
        "validation_method": (
            "Same deterministic PA-centred and PV-centred "
            "patch validation as V6. Metrics remain patch-level."
        ),
        "seed": args.seed,
        "amp": amp,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "test_partition_loaded": False
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # History CSV
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
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        print("\n" + "=" * 55)
        print(f"V6.1 EPOCH {epoch}/{args.epochs}")
        print("=" * 55)

        train_loss = train_one_epoch_v61(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_records=train_records,
            device=device,
            args=args,
            epoch=epoch,
            amp=amp,
            class_weights=class_weights
        )

        # Use the exact V6 deterministic patch validation.
        val_args = SimpleNamespace(
            seed=args.seed,
            patch_size=args.patch_size,
            validation_samples_per_class=args.validation_samples_per_class
        )

        val_metrics = validate(
            model,
            validation_records,
            device,
            val_args,
            amp,
            class_weights
        )

        monitor_dice = val_metrics[
            "mean_foreground_patch_dice"
        ]

        scheduler.step(monitor_dice)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        print("\nEPOCH RESULTS")

        print("Training loss:", round(train_loss, 5))
        print("Validation loss:", round(val_metrics["loss"], 5))

        print(
            "PA validation patch Dice:",
            round(val_metrics["pa_patch_dice"], 5)
        )

        print(
            "PV validation patch Dice:",
            round(val_metrics["pv_patch_dice"], 5)
        )

        print(
            "Mean foreground patch Dice:",
            round(monitor_dice, 5)
        )

        print("Learning rate:", lr)
        print("Time:", round(elapsed, 1), "seconds")

        # ----------------------------------------------------
        # Append history
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pa_patch_dice": val_metrics["pa_patch_dice"],
            "val_pv_patch_dice": val_metrics["pv_patch_dice"],
            "val_mean_foreground_patch_dice": monitor_dice,
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
        # Checkpoint tracking
        # ----------------------------------------------------

        improved = monitor_dice > best_val_dice

        if improved:

            best_val_dice = monitor_dice
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mean_foreground_patch_dice": best_val_dice,
            "current_val_metrics": val_metrics,
            "config": config
        }

        torch.save(state, last_path)

        if improved:

            torch.save(state, best_path)

            print("\nBEST V6.1 CHECKPOINT UPDATED:")
            print(best_path)

        else:

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

            print("\nV6.1 early stopping triggered.")
            break

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("NEUROVESSEL V6.1 TRAINING FINISHED")
    print("=" * 68)

    print(
        "Best mean foreground validation patch Dice:",
        best_val_dice
    )

    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", last_path)
    print("Training history:", history_path)
    print("Configuration:", config_path)

    print("\nIMPORTANT:")
    print("Validation metrics are patch-level, not full-volume Dice.")
    print("The V6 baseline outputs were not overwritten.")
    print("The locked test partition remains untouched.")


if __name__ == "__main__":
    main()
