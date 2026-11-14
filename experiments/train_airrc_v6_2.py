
"""
NEUROVESSEL V6.2
THREE-CLASS PA/PV SEGMENTATION

Controlled experiment:
- Same V6 Residual 3D U-Net architecture.
- Same V6.1 class weights and loss.
- Same 96^3 default patch size.
- Same optimizer and validation procedure.
- One PA patch + one PV patch + one background patch.
- Background patch selection:
    50% broad background: stored intensity > -950
    50% dense background: stored intensity > 0

Only training patients are used for training.
Validation patients are used only for validation.
Locked test patients are never loaded.

V6.1 checkpoints and metrics are not overwritten.
"""

import argparse
import csv
import gc
import json
import random
import time

from pathlib import Path

import numpy as np
import torch

import train_airrc_v6_multiclass as v6


# ============================================================
# 1. OUTPUT PATHS
# ============================================================

ROOT = v6.PROJECT_ROOT

CHECKPOINT_DIR = v6.CHECKPOINT_DIR

METRICS_DIR = (
    v6.METRICS_DIR / "airrc_v6_2_multiclass"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. V6.2 CONFIGURATION
# ============================================================

CLASS_WEIGHTS = [0.5, 1.0, 1.0]

BROAD_BACKGROUND_THRESHOLD = -950.0

DENSE_BACKGROUND_THRESHOLD = 0.0

DENSE_BACKGROUND_PROBABILITY = 0.50

BACKGROUND_SEARCH_BATCH = 8192

BACKGROUND_MAX_ATTEMPTS = 32

BASE_CHANNELS = 8

EARLY_STOPPING_PATIENCE = 8


# ============================================================
# 3. TRAINING-ONLY BACKGROUND SAMPLER
# ============================================================

def sample_background_center(
    image,
    labels,
    rng,
):
    """
    Select a ground-truth background voxel.

    First choose one of two sampling pools with equal
    probability:

        Broad:
            labels == 0 and image > -950

        Dense:
            labels == 0 and image > 0

    Uses rejection sampling to avoid creating a huge
    flat index array for all background voxels.

    Returns:
        center: (D, H, W) coordinate tuple
        sampling_type: descriptive string
    """

    use_dense = (
        rng.random()
        < DENSE_BACKGROUND_PROBABILITY
    )

    if use_dense:

        threshold = DENSE_BACKGROUND_THRESHOLD

        sampling_type = "Background_dense_gt_0"

    else:

        threshold = BROAD_BACKGROUND_THRESHOLD

        sampling_type = "Background_broad_gt_-950"

    flat_image = image.reshape(-1)

    flat_labels = labels.reshape(-1)

    total_voxels = flat_labels.size

    for attempt in range(
        BACKGROUND_MAX_ATTEMPTS
    ):

        candidates = rng.integers(
            0,
            total_voxels,
            size=BACKGROUND_SEARCH_BATCH,
            dtype=np.int64,
        )

        valid = (
            (flat_labels[candidates] == 0)
            & (flat_image[candidates] > threshold)
        )

        valid_indices = candidates[valid]

        if valid_indices.size > 0:

            selected = int(
                valid_indices[
                    rng.integers(
                        0,
                        valid_indices.size,
                    )
                ]
            )

            center = tuple(
                int(v)
                for v in np.unravel_index(
                    selected,
                    labels.shape,
                )
            )

            if labels[center] != 0:

                raise RuntimeError(
                    "Background sampler selected a non-background voxel."
                )

            if image[center] <= threshold:

                raise RuntimeError(
                    "Background sampler violated intensity threshold."
                )

            return center, sampling_type

    raise RuntimeError(
        f"Could not find an eligible background centre "
        f"after {BACKGROUND_MAX_ATTEMPTS} search batches. "
        f"Sampling pool: {sampling_type}"
    )


# ============================================================
# 4. ONE TRAINING EPOCH
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

    patches_per_patient = 3

    expected_steps = (
        len(train_records)
        * patches_per_patient
    )

    for case_number, index in enumerate(
        order,
        start=1,
    ):

        record = train_records[index]

        # Load one training patient.
        image, labels = v6.load_v6_case(
            record
        )

        # One patch for each foreground class.
        patch_specs = [
            ("PA", v6.PULMONARY_ARTERY),
            ("PV", v6.PULMONARY_VEIN),
        ]

        # Add one background patch.
        background_center, background_type = (
            sample_background_center(
                image,
                labels,
                rng,
            )
        )

        patch_specs.append(
            (background_type, None)
        )

        # Shuffle order of the three patches.
        rng.shuffle(
            patch_specs
        )

        for patch_name, class_id in patch_specs:

            if class_id is None:

                center = background_center

            else:

                center = v6.sample_class_center(
                    labels,
                    class_id,
                    rng,
                )

            x_np, y_np = v6.extract_patch(
                image,
                labels,
                center,
                args.patch_size,
            )

            # Tensor dimensions:
            # X = (1, 1, D, H, W)
            # Y = (1, D, H, W)

            x = torch.from_numpy(
                x_np
            ).unsqueeze(0).unsqueeze(0)

            y = torch.from_numpy(
                y_np
            ).unsqueeze(0)

            x = x.to(device)

            y = y.to(device)

            x, y = v6.augment_patch(
                x,
                y,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            try:

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):

                    logits = model(x)

                    loss = v6.multiclass_loss(
                        logits,
                        y,
                        class_weights,
                    )

                if not torch.isfinite(loss):

                    raise RuntimeError(
                        "Non-finite V6.2 training loss."
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            except torch.cuda.OutOfMemoryError as exc:

                if device.type == "cuda":

                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "V6.2 GPU memory exhausted. "
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
                    f"patch={patch_name} "
                    f"loss={current_loss:.4f}",
                    flush=True,
                )

            del x, y, logits, loss
            del x_np, y_np

        del image, labels
        del background_center

    if total_steps != expected_steps:

        raise RuntimeError(
            f"Expected {expected_steps} updates, "
            f"completed {total_steps}."
        )

    return (
        total_loss / max(total_steps, 1)
    )


# ============================================================
# 5. CHECKPOINT WRITER
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_dice,
    val_metrics,
    config,
):
    state = {
        "epoch": epoch,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "best_val_mean_foreground_patch_dice": (
            best_dice
        ),

        "current_val_metrics": (
            val_metrics
        ),

        "config": config,
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        state,
        temporary_path,
    )

    temporary_path.replace(
        path
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NeuroVessel V6.2 hard-background sampling training"
        )
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--validation-samples-per-class",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
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
    # Dataset partitions.
    # --------------------------------------------------------

    (
        train_records,
        validation_records,
        test_count,
    ) = v6.load_v6_splits()

    print("\nDATASET SPLITS")

    print(
        "Training patients:",
        len(train_records),
    )

    print(
        "Validation patients:",
        len(validation_records),
    )

    print(
        "Locked test patients:",
        test_count,
    )

    print(
        "Test images and labels will NOT be loaded."
    )

    # Explicit protection for the three previously
    # investigated validation patients.
    train_patient_ids = {
        r["patient_id"]
        for r in train_records
    }

    validation_patient_ids = {
        r["patient_id"]
        for r in validation_records
    }

    protected_validation = {
        "LIDC-IDRI-0529",
        "LIDC-IDRI-0763",
        "LIDC-IDRI-0848",
    }

    if train_patient_ids & protected_validation:

        raise RuntimeError(
            "A protected validation patient appears in training."
        )

    if not protected_validation.issubset(
        validation_patient_ids
    ):

        raise RuntimeError(
            "Protected validation patient split check failed."
        )

    if args.smoke_test:

        args.patch_size = 64

        args.epochs = 1

        args.validation_samples_per_class = 1

        train_records = train_records[:4]

        validation_records = validation_records[:2]

        print("\nSMOKE TEST MODE")

        print("Training patients: 4")

        print("Validation patients: 2")

        print("Patch size: 64")

    # --------------------------------------------------------
    # Experiment output names.
    # --------------------------------------------------------

    if args.smoke_test:

        run_name = (
            "airrc_v6_2_multiclass_smoke"
        )

    else:

        run_name = (
            "airrc_v6_2_multiclass"
        )

    best_path = (
        CHECKPOINT_DIR
        / f"{run_name}_best.pt"
    )

    last_path = (
        CHECKPOINT_DIR
        / f"{run_name}_last.pt"
    )

    history_path = (
        METRICS_DIR
        / f"{run_name}_history.csv"
    )

    config_path = (
        METRICS_DIR
        / f"{run_name}_config.json"
    )

    # --------------------------------------------------------
    # Experiment information.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V6.2 — CONTROLLED TRAINING"
    )

    print("=" * 70)

    print("Device:", device)

    if amp:

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**3,
                2,
            ),
            "GB",
        )

    print(
        "Architecture: Same V6 Residual 3D U-Net"
    )

    print(
        "Parameters: 1,435,027 expected"
    )

    print(
        "Patch size:",
        args.patch_size,
    )

    print(
        "Classes:",
        v6.CLASS_NAMES,
    )

    print(
        "Training patches per patient: 3"
    )

    print(
        "  1 PA-centred"
    )

    print(
        "  1 PV-centred"
    )

    print(
        "  1 background-centred"
    )

    print(
        "Background sampling mixture: 50% broad + 50% dense"
    )

    print(
        "Broad stored-intensity threshold:",
        BROAD_BACKGROUND_THRESHOLD,
    )

    print(
        "Dense stored-intensity threshold:",
        DENSE_BACKGROUND_THRESHOLD,
    )

    print(
        "Expected updates per epoch:",
        len(train_records) * 3,
    )

    print(
        "Class weights:",
        CLASS_WEIGHTS,
    )

    print(
        "Loss: 0.5 weighted CE + 0.5 foreground soft Dice"
    )

    print(
        "Validation samples per class:",
        args.validation_samples_per_class,
    )

    print(
        "Validation method: Same V6 patch-level procedure"
    )

    print(
        "AMP:",
        amp,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "\nStarting from a fresh model initialization."
    )

    print(
        "V6.1 checkpoint will NOT be loaded or overwritten."
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = v6.NeuroVesselV6(
        in_channels=1,
        out_channels=v6.NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Actual trainable parameters:",
        f"{parameter_count:,}",
    )

    expected_parameters = 1_435_027

    if parameter_count != expected_parameters:

        raise RuntimeError(
            f"Architecture mismatch: expected "
            f"{expected_parameters:,} parameters, "
            f"got {parameter_count:,}."
        )

    # --------------------------------------------------------
    # Loss weights.
    # --------------------------------------------------------

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Optimizer and scheduler.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
    )

    # --------------------------------------------------------
    # Experiment configuration.
    # --------------------------------------------------------

    config = {
        "project": "NeuroVessel",

        "version": "V6.2",

        "task": (
            "Three-class pulmonary artery and "
            "pulmonary vein segmentation"
        ),

        "architecture": "V6 Residual 3D U-Net",

        "base_channels": BASE_CHANNELS,

        "trainable_parameters": parameter_count,

        "input_channels": 1,

        "output_classes": 3,

        "class_mapping": v6.CLASS_NAMES,

        "patch_size": args.patch_size,

        "batch_size": 1,

        "patches_per_patient": 3,

        "training_sampling": {
            "PA": "One PA-centred patch",
            "PV": "One PV-centred patch",
            "background": {
                "broad_probability": 0.50,
                "dense_probability": 0.50,
                "broad_stored_intensity_exclusive": (
                    BROAD_BACKGROUND_THRESHOLD
                ),
                "dense_stored_intensity_exclusive": (
                    DENSE_BACKGROUND_THRESHOLD
                ),
                "ground_truth_condition": "label == 0",
            },
        },

        "class_weights": CLASS_WEIGHTS,

        "loss": (
            "0.5 weighted cross entropy + "
            "0.5 foreground soft Dice"
        ),

        "learning_rate": args.lr,

        "weight_decay": 1e-5,

        "optimizer": "AdamW",

        "scheduler": "ReduceLROnPlateau",

        "epochs_requested": args.epochs,

        "validation_samples_per_class": (
            args.validation_samples_per_class
        ),

        "window_low": -1000,

        "window_high": 1000,

        "seed": args.seed,

        "device": str(device),

        "amp": amp,

        "smoke_test": args.smoke_test,

        "train_patients": len(train_records),

        "validation_patients": len(validation_records),

        "locked_test_patients": test_count,

        "validation_method": (
            "Same deterministic class-centred patch validation "
            "procedure as V6. Validation Dice is patch-level, "
            "not full-volume."
        ),

        "hard_negative_method": (
            "Intensity-stratified ground-truth background "
            "sampling. Not model-mined or out-of-fold."
        ),

        "stored_intensity_calibration_verified": False,

        "test_partition_loaded": False,

        "initialization": "Fresh random initialization",

        "v6_1_checkpoint_loaded": False,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
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
        "epoch_seconds",
    ]

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

    # --------------------------------------------------------
    # Training loop.
    # --------------------------------------------------------

    best_val_dice = -1.0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        print("\n" + "=" * 60)

        print(
            f"V6.2 EPOCH {epoch}/{args.epochs}"
        )

        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            optimizer,
            scaler,
            train_records,
            device,
            args,
            epoch,
            amp,
            class_weights,
        )

        # Use the same patch-level validation function
        # as the original V6 pipeline.
        val_metrics = v6.validate(
            model,
            validation_records,
            device,
            args,
            amp,
            class_weights,
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

        print("\nEPOCH RESULTS")

        print(
            "Training loss:",
            round(train_loss, 5),
        )

        print(
            "Validation loss:",
            round(val_metrics["loss"], 5),
        )

        print(
            "PA validation patch Dice:",
            round(
                val_metrics["pa_patch_dice"],
                5,
            ),
        )

        print(
            "PV validation patch Dice:",
            round(
                val_metrics["pv_patch_dice"],
                5,
            ),
        )

        print(
            "Mean foreground patch Dice:",
            round(monitor_dice, 5),
        )

        print(
            "Learning rate:",
            lr,
        )

        print(
            "Time:",
            round(elapsed, 1),
            "seconds",
        )

        # ----------------------------------------------------
        # Save history.
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
            "epoch_seconds": elapsed,
        }

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writerow(row)

        # ----------------------------------------------------
        # Best checkpoint tracking.
        # ----------------------------------------------------

        improved = (
            monitor_dice > best_val_dice
        )

        if improved:

            best_val_dice = monitor_dice

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # Save latest checkpoint every epoch.
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_val_dice,
            val_metrics,
            config,
        )

        if improved:

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_dice,
                val_metrics,
                config,
            )

            print(
                "\nBEST V6.2 CHECKPOINT UPDATED:",
                best_path,
            )

        else:

            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                EARLY_STOPPING_PATIENCE,
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
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nV6.2 early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V6.2 TRAINING FINISHED"
    )

    print("=" * 70)

    print(
        "Best mean foreground validation patch Dice:",
        best_val_dice,
    )

    print(
        "Best checkpoint:",
        best_path,
    )

    print(
        "Latest checkpoint:",
        last_path,
    )

    print(
        "Training history:",
        history_path,
    )

    print(
        "Configuration:",
        config_path,
    )

    print("\nIMPORTANT")

    print(
        "Validation values are patch-level Dice."
    )

    print(
        "Run the full-volume validation evaluator separately."
    )

    print(
        "The locked test partition remains untouched."
    )

    print(
        "V6.1 checkpoints and metrics were not overwritten."
    )


if __name__ == "__main__":

    main()
