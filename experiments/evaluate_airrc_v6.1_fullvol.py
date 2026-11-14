
"""
NEUROVESSEL V6 / V6.1 — FULL-VOLUME MULTICLASS EVALUATION

Classes:
    0 = Background
    1 = Pulmonary artery
    2 = Pulmonary vein

Features:
    - Full-volume sliding-window inference
    - Gaussian overlap blending
    - Separate PA and PV Dice, precision and recall
    - Macro patient-level and micro voxel-level metrics
    - PA/PV confusion attribution
    - Predicted NIfTI masks preserving CT geometry
    - Five-panel visual comparisons
    - Separate output directories for V6 and V6.1

IMPORTANT:
    Only validation patients are evaluated.
    The locked test partition is never loaded.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import csv
import gc
import json
import time

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from monai.inferers import sliding_window_inference

from train_airrc_v5b import (
    PROJECT_ROOT,
    METRICS_DIR,
    CHECKPOINT_DIR,
)

from train_airrc_v6_multiclass import (
    NeuroVesselV6,
    load_v6_splits,
    multiclass_label_path,
    CLASS_NAMES,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

NUM_CLASSES = 3

BACKGROUND = 0
PA = 1
PV = 2

WINDOW_LOW = -1000.0
WINDOW_HIGH = 1000.0

DEFAULT_CHECKPOINT = (
    CHECKPOINT_DIR / "airrc_v6_multiclass_best.pt"
)

DEFAULT_RUN_NAME = "airrc_v6_fullvol"


# ============================================================
# 3. GEOMETRY AND DATA LOADING
# ============================================================

def geometry_matches(image_obj, label_obj):

    return (
        image_obj.GetSize() == label_obj.GetSize()

        and np.allclose(
            image_obj.GetSpacing(),
            label_obj.GetSpacing(),
            atol=1e-5
        )

        and np.allclose(
            image_obj.GetOrigin(),
            label_obj.GetOrigin(),
            atol=1e-4
        )

        and np.allclose(
            image_obj.GetDirection(),
            label_obj.GetDirection(),
            atol=1e-5
        )
    )


def load_evaluation_case(record):

    image_path = Path(record["image"])

    label_path = multiclass_label_path(
        record["uid"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    image_obj = sitk.ReadImage(
        str(image_path)
    )

    label_obj = sitk.ReadImage(
        str(label_path)
    )

    if not geometry_matches(
        image_obj,
        label_obj
    ):

        raise ValueError(
            f"CT/label geometry mismatch: {record['uid']}"
        )

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
            f"Shape mismatch for {record['uid']}: "
            f"{image.shape} vs {labels.shape}"
        )

    if not np.isfinite(image).all():

        raise ValueError(
            f"Non-finite CT values: {record['uid']}"
        )

    unique_labels = np.unique(labels)

    if not set(
        unique_labels.tolist()
    ).issubset({0, 1, 2}):

        raise ValueError(
            f"Unexpected labels in {record['uid']}: "
            f"{unique_labels}"
        )

    if not np.any(labels == PA):

        raise ValueError(
            f"Empty PA annotation: {record['uid']}"
        )

    if not np.any(labels == PV):

        raise ValueError(
            f"Empty PV annotation: {record['uid']}"
        )

    return image_obj, image, labels


# ============================================================
# 4. CT NORMALIZATION
# ============================================================

def normalize_ct(image):

    normalized = np.clip(
        image,
        WINDOW_LOW,
        WINDOW_HIGH
    ).astype(
        np.float32,
        copy=False
    )

    normalized = (
        normalized - WINDOW_LOW
    ) / (
        WINDOW_HIGH - WINDOW_LOW
    )

    return normalized.astype(
        np.float32,
        copy=False
    )


# ============================================================
# 5. FULL-VOLUME SLIDING-WINDOW INFERENCE
# ============================================================

def predict_full_volume(
    model,
    image,
    device,
    roi_size=96,
    overlap=0.5,
    sw_batch_size=1,
):

    normalized = normalize_ct(image)

    # Input remains on CPU.
    # Individual windows are transferred to GPU.
    volume = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    roi = (
        roi_size,
        roi_size,
        roi_size
    )

    amp = device.type == "cuda"

    with torch.inference_mode():

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp
        ):

            logits = sliding_window_inference(
                inputs=volume,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
                sw_device=device,
                device=torch.device("cpu"),
                progress=False,
            )

    if logits.shape[1] != NUM_CLASSES:

        raise RuntimeError(
            f"Expected 3 output channels, "
            f"received {logits.shape[1]}"
        )

    # MONAI has already blended overlapping windows.
    # Convert final logits into discrete class predictions.
    prediction = (
        logits.argmax(dim=1)
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
    )

    del volume
    del logits
    del normalized

    return prediction


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def make_confusion_matrix(
    ground_truth,
    prediction
):

    gt_flat = ground_truth.reshape(-1).astype(
        np.int64
    )

    pred_flat = prediction.reshape(-1).astype(
        np.int64
    )

    encoded = (
        gt_flat * NUM_CLASSES + pred_flat
    )

    matrix = np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )

    return matrix.astype(np.int64)


# ============================================================
# 7. CLASS METRICS
# ============================================================

def calculate_class_metrics(
    matrix,
    class_id
):

    true_positive = int(
        matrix[class_id, class_id]
    )

    predicted_voxels = int(
        matrix[:, class_id].sum()
    )

    ground_truth_voxels = int(
        matrix[class_id, :].sum()
    )

    false_positive = (
        predicted_voxels - true_positive
    )

    false_negative = (
        ground_truth_voxels - true_positive
    )

    dice_denominator = (
        predicted_voxels + ground_truth_voxels
    )

    dice = (
        2.0 * true_positive / dice_denominator
        if dice_denominator > 0
        else 0.0
    )

    precision = (
        true_positive / predicted_voxels
        if predicted_voxels > 0
        else 0.0
    )

    recall = (
        true_positive / ground_truth_voxels
        if ground_truth_voxels > 0
        else 0.0
    )

    return {

        "dice": float(dice),

        "precision": float(precision),

        "recall": float(recall),

        "true_positive": true_positive,

        "false_positive": false_positive,

        "false_negative": false_negative,

        "predicted_voxels": predicted_voxels,

        "ground_truth_voxels": ground_truth_voxels,
    }


def calculate_micro_metrics(matrix):

    pa_metrics = calculate_class_metrics(
        matrix,
        PA
    )

    pv_metrics = calculate_class_metrics(
        matrix,
        PV
    )

    true_positive = (
        pa_metrics["true_positive"]
        + pv_metrics["true_positive"]
    )

    predicted_foreground = (
        pa_metrics["predicted_voxels"]
        + pv_metrics["predicted_voxels"]
    )

    target_foreground = (
        pa_metrics["ground_truth_voxels"]
        + pv_metrics["ground_truth_voxels"]
    )

    denominator = (
        predicted_foreground + target_foreground
    )

    foreground_dice = (
        2.0 * true_positive / denominator
        if denominator > 0
        else 0.0
    )

    return {

        "pa": pa_metrics,

        "pv": pv_metrics,

        "foreground_micro_dice": float(
            foreground_dice
        ),
    }


# ============================================================
# 8. SAVE NIFTI PREDICTIONS
# ============================================================

def save_prediction(
    prediction,
    reference_image,
    output_path
):

    pred_image = sitk.GetImageFromArray(
        prediction.astype(np.uint8)
    )

    # Preserve physical geometry of original CT.
    pred_image.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        pred_image,
        str(output_path),
        True
    )


# ============================================================
# 9. VISUALIZATION HELPERS
# ============================================================

def make_base_image(ct_slice):

    gray = np.clip(
        (ct_slice - WINDOW_LOW)
        / (WINDOW_HIGH - WINDOW_LOW),
        0.0,
        1.0
    )

    return np.repeat(
        gray[..., None],
        3,
        axis=-1
    )


def blend_class(
    base,
    mask,
    color,
    alpha=0.58
):

    result = base.copy()

    color_array = np.asarray(
        color,
        dtype=np.float32
    )

    result[mask] = (
        (1.0 - alpha) * result[mask]
        + alpha * color_array
    )

    return result


def comparison_overlay(
    base,
    ground_truth,
    prediction,
    class_id,
    gt_color,
    pred_color,
    overlap_color
):

    result = base.copy()

    gt_mask = (
        ground_truth == class_id
    )

    pred_mask = (
        prediction == class_id
    )

    intersection = (
        gt_mask & pred_mask
    )

    gt_only = (
        gt_mask & (~pred_mask)
    )

    pred_only = (
        pred_mask & (~gt_mask)
    )

    result = blend_class(
        result,
        gt_only,
        gt_color
    )

    result = blend_class(
        result,
        pred_only,
        pred_color
    )

    result = blend_class(
        result,
        intersection,
        overlap_color
    )

    return result


# ============================================================
# 10. FIVE-PANEL COMPARISON FIGURE
# ============================================================

def save_comparison_figure(
    uid,
    ct,
    ground_truth,
    prediction,
    output_dir
):

    try:

        import matplotlib.pyplot as plt

    except ImportError as exc:

        raise RuntimeError(
            "Matplotlib is required. "
            "Install using: pip install matplotlib"
        ) from exc

    # Choose axial slice with largest combined
    # PA and PV annotation area.
    foreground_per_slice = np.count_nonzero(
        ground_truth > 0,
        axis=(1, 2)
    )

    slice_index = int(
        np.argmax(foreground_per_slice)
    )

    ct_slice = ct[slice_index]

    gt_slice = ground_truth[slice_index]

    pred_slice = prediction[slice_index]

    base = make_base_image(
        ct_slice
    )

    # Ground truth:
    # PA = green
    # PV = cyan

    gt_rgb = base.copy()

    gt_rgb = blend_class(
        gt_rgb,
        gt_slice == PA,
        (0.0, 1.0, 0.0)
    )

    gt_rgb = blend_class(
        gt_rgb,
        gt_slice == PV,
        (0.0, 0.85, 1.0)
    )

    # Prediction:
    # PA = red
    # PV = magenta

    pred_rgb = base.copy()

    pred_rgb = blend_class(
        pred_rgb,
        pred_slice == PA,
        (1.0, 0.15, 0.1)
    )

    pred_rgb = blend_class(
        pred_rgb,
        pred_slice == PV,
        (1.0, 0.0, 1.0)
    )

    # PA overlay.
    pa_comparison = comparison_overlay(
        base,
        gt_slice,
        pred_slice,
        PA,
        gt_color=(0.0, 1.0, 0.0),
        pred_color=(1.0, 0.0, 0.0),
        overlap_color=(1.0, 1.0, 0.0)
    )

    # PV overlay.
    pv_comparison = comparison_overlay(
        base,
        gt_slice,
        pred_slice,
        PV,
        gt_color=(0.0, 0.9, 1.0),
        pred_color=(1.0, 0.0, 1.0),
        overlap_color=(1.0, 1.0, 1.0)
    )

    panels = [

        (base, "Original CT"),

        (gt_rgb, "Ground Truth: PA + PV"),

        (pred_rgb, "Prediction: PA + PV"),

        (pa_comparison, "PA: GT / Prediction"),

        (pv_comparison, "PV: GT / Prediction"),
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(23, 5)
    )

    for ax, (panel, title) in zip(
        axes,
        panels
    ):

        ax.imshow(panel)

        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold"
        )

        ax.axis("off")

    fig.suptitle(
        f"NeuroVessel — {uid} — "
        f"Axial slice {slice_index}",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / f"{uid}_v6_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# 11. CSV HELPERS
# ============================================================

def write_confusion_csv(
    matrix,
    output_path
):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Ground truth",
            "Predicted background",
            "Predicted PA",
            "Predicted PV",
        ])

        for class_id in range(NUM_CLASSES):

            writer.writerow([
                CLASS_NAMES[class_id],
                *matrix[class_id].tolist()
            ])


def safe_fraction(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NeuroVessel V6/V6.1 full-volume evaluation"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT)
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=96
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional validation-only smoke test limit."
    )

    parser.add_argument(
        "--no-save-predictions",
        action="store_true"
    )

    parser.add_argument(
        "--no-figures",
        action="store_true"
    )

    # Separate output names for V6 and V6.1.
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if (
        args.patch_size < 32
        or args.patch_size % 16 != 0
    ):

        raise ValueError(
            "Patch size must be >=32 and divisible by 16."
        )

    if not 0.0 <= args.overlap < 1.0:

        raise ValueError(
            "Overlap must be >=0 and <1."
        )

    if args.sw_batch_size < 1:

        raise ValueError(
            "Sliding-window batch size must be >=1."
        )

    if args.limit < 0:

        raise ValueError(
            "Limit cannot be negative."
        )

    if (
        not args.run_name
        or any(
            char in args.run_name
            for char in '<>:"/\\|?*'
        )
    ):

        raise ValueError(
            "Invalid run name. Use letters, numbers, "
            "underscores or hyphens."
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            checkpoint_path
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = device.type == "cuda"

    print("\n" + "=" * 68)

    print(
        "NEUROVESSEL — FULL-VOLUME EVALUATION"
    )

    print("=" * 68)

    print("Device:", device)

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

    # --------------------------------------------------------
    # Load splits
    # --------------------------------------------------------

    train_records, validation_records, test_count = (
        load_v6_splits()
    )

    if len(validation_records) != 26:

        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"found {len(validation_records)}"
        )

    print("\nDATASET PARTITIONS")

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
        "Test images and labels will NOT be loaded."
    )

    if args.limit:

        validation_records = (
            validation_records[:args.limit]
        )

        print(
            "\nVALIDATION SMOKE TEST:",
            len(validation_records),
            "patients"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in state:

        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    checkpoint_config = state.get(
        "config",
        {}
    )

    if checkpoint_config.get(
        "output_classes",
        3
    ) != 3:

        raise ValueError(
            "Checkpoint configuration does not specify 3 classes."
        )

    model = NeuroVesselV6(
        in_channels=1,
        out_channels=3,
        base=8
    ).to(device)

    model.load_state_dict(
        state["model_state_dict"],
        strict=True
    )

    model.eval()

    checkpoint_epoch = state.get(
        "epoch",
        None
    )

    print(
        "Checkpoint epoch:",
        checkpoint_epoch
    )

    print(
        "Best patch validation Dice:",
        state.get(
            "best_val_mean_foreground_patch_dice",
            "not recorded"
        )
    )

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------------
    # Separate output directories
    # --------------------------------------------------------

    suffix = (
        f"_limit{args.limit}"
        if args.limit
        else ""
    )

    run_name = (
        args.run_name + suffix
    )

    prediction_dir = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / run_name
    )

    visualization_dir = (
        PROJECT_ROOT
        / "outputs"
        / "visualizations"
        / run_name
    )

    metrics_dir = (
        METRICS_DIR / run_name
    )

    for directory in (
        prediction_dir,
        visualization_dir,
        metrics_dir
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    case_csv_path = (
        metrics_dir
        / f"{run_name}_per_patient.csv"
    )

    confusion_csv_path = (
        metrics_dir
        / f"{run_name}_confusion_matrix.csv"
    )

    summary_json_path = (
        metrics_dir
        / f"{run_name}_summary.json"
    )

    # --------------------------------------------------------
    # CSV fields
    # --------------------------------------------------------

    fields = [

        "uid",

        "patient_id",

        "volume_shape",

        "pa_dice",

        "pa_precision",

        "pa_recall",

        "pv_dice",

        "pv_precision",

        "pv_recall",

        "mean_foreground_dice",

        "gt_pa_voxels",

        "pred_pa_voxels",

        "gt_pv_voxels",

        "pred_pv_voxels",

        "pv_misclassified_as_pa",

        "pa_misclassified_as_pv",

        "pa_fp_from_pv_fraction",

        "pv_fp_from_pa_fraction",

        "inference_seconds",
    ]

    total_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    patient_rows = []

    print("\n" + "=" * 68)

    print("FULL-VOLUME VALIDATION")

    print("=" * 68)

    # --------------------------------------------------------
    # Evaluate each validation patient
    # --------------------------------------------------------

    with open(
        case_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for case_index, record in enumerate(
            validation_records,
            start=1
        ):

            uid = record["uid"]

            patient_id = record["patient_id"]

            print(
                f"\n[{case_index}/{len(validation_records)}] "
                f"Patient: {patient_id} | UID: {uid}",
                flush=True
            )

            image_obj, image, labels = (
                load_evaluation_case(record)
            )

            print(
                "  Volume shape (D,H,W):",
                image.shape
            )

            start_time = time.time()

            prediction = predict_full_volume(
                model=model,
                image=image,
                device=device,
                roi_size=args.patch_size,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size
            )

            inference_seconds = (
                time.time() - start_time
            )

            if prediction.shape != labels.shape:

                raise RuntimeError(
                    f"Prediction geometry mismatch: {uid}"
                )

            matrix = make_confusion_matrix(
                labels,
                prediction
            )

            total_confusion += matrix

            pa_metrics = calculate_class_metrics(
                matrix,
                PA
            )

            pv_metrics = calculate_class_metrics(
                matrix,
                PV
            )

            mean_dice = (
                pa_metrics["dice"]
                + pv_metrics["dice"]
            ) / 2.0

            # Cross-class confusion.
            pv_as_pa = int(
                matrix[PV, PA]
            )

            pa_as_pv = int(
                matrix[PA, PV]
            )

            pa_fp_from_pv = safe_fraction(
                pv_as_pa,
                pa_metrics["false_positive"]
            )

            pv_fp_from_pa = safe_fraction(
                pa_as_pv,
                pv_metrics["false_positive"]
            )

            row = {

                "uid": uid,

                "patient_id": patient_id,

                "volume_shape": str(
                    tuple(image.shape)
                ),

                "pa_dice": pa_metrics["dice"],

                "pa_precision": pa_metrics["precision"],

                "pa_recall": pa_metrics["recall"],

                "pv_dice": pv_metrics["dice"],

                "pv_precision": pv_metrics["precision"],

                "pv_recall": pv_metrics["recall"],

                "mean_foreground_dice": mean_dice,

                "gt_pa_voxels": (
                    pa_metrics["ground_truth_voxels"]
                ),

                "pred_pa_voxels": (
                    pa_metrics["predicted_voxels"]
                ),

                "gt_pv_voxels": (
                    pv_metrics["ground_truth_voxels"]
                ),

                "pred_pv_voxels": (
                    pv_metrics["predicted_voxels"]
                ),

                "pv_misclassified_as_pa": pv_as_pa,

                "pa_misclassified_as_pv": pa_as_pv,

                "pa_fp_from_pv_fraction": pa_fp_from_pv,

                "pv_fp_from_pa_fraction": pv_fp_from_pa,

                "inference_seconds": inference_seconds,
            }

            patient_rows.append(row)

            writer.writerow(row)

            f.flush()

            print(
                f"  PA Dice={pa_metrics['dice']:.4f} "
                f"Precision={pa_metrics['precision']:.4f} "
                f"Recall={pa_metrics['recall']:.4f}"
            )

            print(
                f"  PV Dice={pv_metrics['dice']:.4f} "
                f"Precision={pv_metrics['precision']:.4f} "
                f"Recall={pv_metrics['recall']:.4f}"
            )

            print(
                f"  Mean foreground full-volume Dice="
                f"{mean_dice:.4f}"
            )

            print(
                f"  PV voxels predicted as PA={pv_as_pa:,}"
            )

            print(
                f"  Inference time={inference_seconds:.1f}s"
            )

            # ------------------------------------------------
            # Save NIfTI prediction
            # ------------------------------------------------

            if not args.no_save_predictions:

                prediction_path = (
                    prediction_dir
                    / f"{uid}_pred_multiclass.nii.gz"
                )

                save_prediction(
                    prediction,
                    image_obj,
                    prediction_path
                )

            # ------------------------------------------------
            # Save comparison figure
            # ------------------------------------------------

            if not args.no_figures:

                figure_path = save_comparison_figure(
                    uid,
                    image,
                    labels,
                    prediction,
                    visualization_dir
                )

                print(
                    "  Figure:",
                    figure_path.name
                )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            del image_obj
            del image
            del labels
            del prediction
            del matrix

            gc.collect()

            if device.type == "cuda":

                torch.cuda.empty_cache()

    # ========================================================
    # 13. AGGREGATE RESULTS
    # ========================================================

    micro = calculate_micro_metrics(
        total_confusion
    )

    pa_patient_dice = [
        row["pa_dice"]
        for row in patient_rows
    ]

    pv_patient_dice = [
        row["pv_dice"]
        for row in patient_rows
    ]

    mean_patient_dice = [
        row["mean_foreground_dice"]
        for row in patient_rows
    ]

    pa_patient_precision = [
        row["pa_precision"]
        for row in patient_rows
    ]

    pa_patient_recall = [
        row["pa_recall"]
        for row in patient_rows
    ]

    pv_patient_precision = [
        row["pv_precision"]
        for row in patient_rows
    ]

    pv_patient_recall = [
        row["pv_recall"]
        for row in patient_rows
    ]

    total_voxels = int(
        total_confusion.sum()
    )

    overall_accuracy = safe_fraction(
        int(np.trace(total_confusion)),
        total_voxels
    )

    # Background false positives and cross-class false positives.
    pa_fp = int(
        total_confusion[BACKGROUND, PA]
        + total_confusion[PV, PA]
    )

    pv_fp = int(
        total_confusion[BACKGROUND, PV]
        + total_confusion[PA, PV]
    )

    pv_as_pa = int(
        total_confusion[PV, PA]
    )

    pa_as_pv = int(
        total_confusion[PA, PV]
    )

    # ========================================================
    # 14. FINAL SUMMARY JSON
    # ========================================================

    summary = {

        "project": "NeuroVessel",

        "version": args.run_name,

        "task": (
            "Full-volume three-class pulmonary artery "
            "and pulmonary vein segmentation"
        ),

        "checkpoint": str(checkpoint_path),

        "checkpoint_epoch": checkpoint_epoch,

        "best_patch_validation_dice": state.get(
            "best_val_mean_foreground_patch_dice"
        ),

        "architecture": "Residual 3D U-Net",

        "base_channels": 8,

        "class_mapping": {
            "0": "Background",
            "1": "Pulmonary artery",
            "2": "Pulmonary vein"
        },

        "evaluation_protocol": {

            "method": "Full-volume sliding-window inference",

            "roi_size": [
                args.patch_size,
                args.patch_size,
                args.patch_size
            ],

            "overlap": args.overlap,

            "blend_mode": "Gaussian",

            "sw_batch_size": args.sw_batch_size,

            "window_low": WINDOW_LOW,

            "window_high": WINDOW_HIGH,

            "normalization": "Clip and scale to [0,1]",

            "classification": "Three-class argmax",
        },

        "dataset": {

            "training_patients_in_split": len(train_records),

            "validation_patients_in_split": 26,

            "validation_patients_evaluated": len(patient_rows),

            "locked_test_patients": test_count,

            "test_partition_loaded": False,
        },

        "macro_patient_average": {

            "pa_dice": float(
                np.mean(pa_patient_dice)
            ),

            "pa_precision": float(
                np.mean(pa_patient_precision)
            ),

            "pa_recall": float(
                np.mean(pa_patient_recall)
            ),

            "pv_dice": float(
                np.mean(pv_patient_dice)
            ),

            "pv_precision": float(
                np.mean(pv_patient_precision)
            ),

            "pv_recall": float(
                np.mean(pv_patient_recall)
            ),

            "mean_foreground_dice": float(
                np.mean(mean_patient_dice)
            ),
        },

        "micro_voxel_aggregate": {

            "pa": micro["pa"],

            "pv": micro["pv"],

            "foreground_micro_dice": (
                micro["foreground_micro_dice"]
            ),

            "overall_voxel_accuracy": overall_accuracy,

            "total_voxels": total_voxels,
        },

        "cross_class_confusion": {

            "pv_ground_truth_predicted_as_pa": pv_as_pa,

            "pa_ground_truth_predicted_as_pv": pa_as_pv,

            "fraction_of_pa_false_positives_from_pv": (
                safe_fraction(pv_as_pa, pa_fp)
            ),

            "fraction_of_pv_false_positives_from_pa": (
                safe_fraction(pa_as_pv, pv_fp)
            ),

            "fraction_of_ground_truth_pv_predicted_as_pa": (
                safe_fraction(
                    pv_as_pa,
                    int(total_confusion[PV, :].sum())
                )
            ),

            "fraction_of_ground_truth_pa_predicted_as_pv": (
                safe_fraction(
                    pa_as_pv,
                    int(total_confusion[PA, :].sum())
                )
            ),
        },

        "confusion_matrix": {

            "rows": [
                "Ground truth background",
                "Ground truth PA",
                "Ground truth PV"
            ],

            "columns": [
                "Predicted background",
                "Predicted PA",
                "Predicted PV"
            ],

            "counts": total_confusion.tolist(),
        },

        "outputs": {

            "per_patient_csv": str(case_csv_path),

            "confusion_matrix_csv": str(
                confusion_csv_path
            ),

            "summary_json": str(
                summary_json_path
            ),

            "predictions_directory": (
                None
                if args.no_save_predictions
                else str(prediction_dir)
            ),

            "visualizations_directory": (
                None
                if args.no_figures
                else str(visualization_dir)
            ),
        }
    }

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    write_confusion_csv(
        total_confusion,
        confusion_csv_path
    )

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    with open(
        summary_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # 15. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 68)

    print(
        "NEUROVESSEL — FULL-VOLUME RESULTS"
    )

    print("=" * 68)

    print(
        "\nRun name:",
        args.run_name
    )

    print(
        "Patients evaluated:",
        len(patient_rows)
    )

    print("\nMACRO PATIENT-AVERAGE METRICS")

    macro = summary[
        "macro_patient_average"
    ]

    for key, value in macro.items():

        print(
            f"{key:30s}: {value:.5f}"
        )

    print("\nMICRO VOXEL-AGGREGATE METRICS")

    print(
        "PA Dice:",
        f"{micro['pa']['dice']:.5f}"
    )

    print(
        "PV Dice:",
        f"{micro['pv']['dice']:.5f}"
    )

    print(
        "Foreground micro Dice:",
        f"{micro['foreground_micro_dice']:.5f}"
    )

    print("\nPA/PV CONFUSION")

    print(
        "Ground-truth PV predicted as PA:",
        f"{pv_as_pa:,}"
    )

    print(
        "Ground-truth PA predicted as PV:",
        f"{pa_as_pv:,}"
    )

    print("\nCONFUSION MATRIX")

    print(total_confusion)

    print("\nOUTPUT FILES")

    print(
        "Per-patient CSV:",
        case_csv_path
    )

    print(
        "Confusion CSV:",
        confusion_csv_path
    )

    print(
        "Summary JSON:",
        summary_json_path
    )

    if not args.no_save_predictions:

        print(
            "Predictions:",
            prediction_dir
        )

    if not args.no_figures:

        print(
            "Visualizations:",
            visualization_dir
        )

    print(
        "\nLOCKED TEST PARTITION WAS NOT LOADED."
    )

    print(
        "\nFULL-VOLUME EVALUATION FINISHED."
    )


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
