
"""
NEUROVESSEL V5B — FULL-VOLUME VALIDATION

Evaluates the best V5B checkpoint on all 26 validation patients.

Features:
    - MONAI sliding-window inference
    - Gaussian blending of overlapping windows
    - Full-volume Dice, precision, recall and specificity
    - Threshold comparison from 0.30 to 0.80
    - Per-patient and aggregate metrics
    - NIfTI prediction masks at a selected threshold
    - Locked test set is never loaded

Run:
    python -X faulthandler -u experiments/evaluate_airrc_v5b_fullvol.py

Author: NeuroVessel project
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import csv
import json
import time

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch

from monai.inferers import sliding_window_inference

# Import the exact V5B architecture and dataset loader.
from train_airrc_v5b import (
    NeuroVesselV5,
    load_splits,
    load_case,
    PROJECT_ROOT,
    CHECKPOINT_DIR,
    METRICS_DIR
)


# ============================================================
# 2. PATHS
# ============================================================

CHECKPOINT_PATH = (
    CHECKPOINT_DIR /
    "airrc_v5b_wm1000_wp1000_best.pt"
)

OUTPUT_METRICS_DIR = (
    METRICS_DIR /
    "airrc_v5b_fullvol"
)

OUTPUT_PREDICTION_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "predictions" /
    "airrc_v5b_fullvol"
)

OUTPUT_METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    probabilities,
    target,
    threshold
):

    prediction = probabilities >= threshold

    target = target.astype(bool, copy=False)

    intersection = int(
        np.count_nonzero(
            prediction & target
        )
    )

    predicted_count = int(
        np.count_nonzero(prediction)
    )

    target_count = int(
        np.count_nonzero(target)
    )

    total_voxels = int(target.size)

    false_positive = (
        predicted_count - intersection
    )

    false_negative = (
        target_count - intersection
    )

    true_negative = (
        total_voxels
        - intersection
        - false_positive
        - false_negative
    )

    denominator = (
        predicted_count + target_count
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * intersection
        ) / denominator

    precision = (
        intersection / predicted_count
        if predicted_count > 0
        else 0.0
    )

    recall = (
        intersection / target_count
        if target_count > 0
        else 0.0
    )

    specificity_denominator = (
        true_negative + false_positive
    )

    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "intersection": intersection,
        "predicted_voxels": predicted_count,
        "target_voxels": target_count,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


# ============================================================
# 4. SAVE PREDICTION AS NIFTI
# ============================================================

def save_prediction(
    prediction,
    source_image_path,
    output_path
):

    reference_image = sitk.ReadImage(
        str(source_image_path)
    )

    prediction_image = sitk.GetImageFromArray(
        prediction.astype(np.uint8)
    )

    # Preserve original image geometry.
    prediction_image.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        prediction_image,
        str(output_path),
        useCompression=True
    )

    del reference_image
    del prediction_image


def safe_filename(uid):

    return "".join(
        char if char.isalnum() or char in "-_"
        else "_"
        for char in str(uid)
    )


# ============================================================
# 5. FULL-VOLUME INFERENCE
# ============================================================

def run_full_volume_inference(
    model,
    image,
    device,
    amp,
    patch_size,
    overlap,
    sw_batch_size
):

    # Match the exact intensity preprocessing used in V5B.
    image = np.clip(
        image,
        -1000.0,
        1000.0
    )

    image = (
        (image + 1000.0) / 2000.0
    ).astype(np.float32, copy=False)

    # Input stays on CPU.
    # MONAI transfers individual windows to the GPU.
    inputs = torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0).unsqueeze(0)

    roi_size = (
        patch_size,
        patch_size,
        patch_size
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp
        ):

            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
                sigma_scale=0.125,
                sw_device=device,
                device=torch.device("cpu")
            )

        probabilities = torch.sigmoid(
            logits.float()
        )

    probabilities = (
        probabilities[0, 0]
        .contiguous()
        .numpy()
    )

    if probabilities.shape != image.shape:

        raise RuntimeError(
            "Prediction shape does not match CT shape: "
            f"{probabilities.shape} vs {image.shape}"
        )

    del inputs
    del logits

    return probabilities


# ============================================================
# 6. AGGREGATE METRICS
# ============================================================

def summarize_threshold(
    threshold,
    patient_rows
):

    rows = [
        row for row in patient_rows
        if row["threshold"] == threshold
    ]

    if not rows:

        raise ValueError(
            f"No patient results for threshold {threshold}"
        )

    dice_values = np.asarray(
        [row["dice"] for row in rows],
        dtype=np.float64
    )

    precision_values = np.asarray(
        [row["precision"] for row in rows],
        dtype=np.float64
    )

    recall_values = np.asarray(
        [row["recall"] for row in rows],
        dtype=np.float64
    )

    specificity_values = np.asarray(
        [row["specificity"] for row in rows],
        dtype=np.float64
    )

    # Global voxel-level totals.
    total_intersection = sum(
        row["intersection"] for row in rows
    )

    total_predicted = sum(
        row["predicted_voxels"] for row in rows
    )

    total_target = sum(
        row["target_voxels"] for row in rows
    )

    total_tn = sum(
        row["true_negative"] for row in rows
    )

    total_fp = sum(
        row["false_positive"] for row in rows
    )

    total_fn = sum(
        row["false_negative"] for row in rows
    )

    micro_dice_denominator = (
        total_predicted + total_target
    )

    micro_dice = (
        2 * total_intersection
        / micro_dice_denominator
        if micro_dice_denominator
        else 1.0
    )

    micro_precision = (
        total_intersection / total_predicted
        if total_predicted
        else 0.0
    )

    micro_recall = (
        total_intersection / total_target
        if total_target
        else 0.0
    )

    micro_specificity = (
        total_tn / (total_tn + total_fp)
        if (total_tn + total_fp)
        else 0.0
    )

    return {

        "threshold": threshold,

        "patients": len(rows),

        # Patient-wise macro averages.
        "macro_dice": float(
            np.mean(dice_values)
        ),

        "dice_std": float(
            np.std(dice_values, ddof=1)
            if len(dice_values) > 1
            else 0.0
        ),

        "median_dice": float(
            np.median(dice_values)
        ),

        "macro_precision": float(
            np.mean(precision_values)
        ),

        "macro_recall": float(
            np.mean(recall_values)
        ),

        "macro_specificity": float(
            np.mean(specificity_values)
        ),

        # Voxel-pooled metrics.
        "micro_dice": float(micro_dice),

        "micro_precision": float(
            micro_precision
        ),

        "micro_recall": float(
            micro_recall
        ),

        "micro_specificity": float(
            micro_specificity
        ),

        "total_intersection": total_intersection,

        "total_predicted_voxels": total_predicted,

        "total_target_voxels": total_target,

        "total_false_positive": total_fp,

        "total_false_negative": total_fn

    }


# ============================================================
# 7. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NeuroVessel V5B full-volume validation"
        )
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80
        ]
    )

    parser.add_argument(
        "--save-threshold",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    if not 0.0 <= args.overlap < 1.0:

        raise ValueError(
            "Overlap must be >=0 and <1."
        )

    if args.sw_batch_size < 1:

        raise ValueError(
            "Sliding-window batch size must be >=1."
        )

    if not args.thresholds:

        raise ValueError(
            "At least one threshold is required."
        )

    if any(
        threshold <= 0.0 or threshold >= 1.0
        for threshold in args.thresholds
    ):

        raise ValueError(
            "Thresholds must be between 0 and 1."
        )

    if not (
        0.0 < args.save_threshold < 1.0
    ):

        raise ValueError(
            "Save threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = device.type == "cuda"

    if device.type == "cuda":

        torch.backends.cudnn.benchmark = True

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "WARNING: CUDA unavailable. "
            "Full-volume CPU inference may be very slow."
        )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            f"Best V5B checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    print("\nLoading checkpoint:")

    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Invalid checkpoint: model_state_dict missing."
        )

    config = checkpoint.get(
        "config",
        {}
    )

    if config.get("version") != "V5B":

        raise RuntimeError(
            "Checkpoint does not identify itself as V5B."
        )

    patch_size = int(
        config.get("patch_size", 96)
    )

    window_low = float(
        config.get("window_low", -1000)
    )

    window_high = float(
        config.get("window_high", 1000)
    )

    if (
        window_low != -1000.0
        or window_high != 1000.0
    ):

        raise RuntimeError(
            "This evaluator currently expects the "
            "V5B [-1000, 1000] intensity window."
        )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Best patch validation Dice:",
        checkpoint.get("best_val_patch_dice")
    )

    print(
        "Patch size:",
        patch_size
    )

    print(
        "Intensity window:",
        window_low,
        window_high
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    base_channels = int(
        config.get("base_channels", 8)
    )

    model = NeuroVesselV5(
        in_channels=1,
        out_channels=1,
        base=base_channels
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    del checkpoint

    # --------------------------------------------------------
    # Load validation split
    # --------------------------------------------------------

    (
        train_records,
        validation_records,
        test_count
    ) = load_splits()

    # This evaluator never reads or loads test images.
    del train_records

    if len(validation_records) != 26:

        raise RuntimeError(
            "Expected 26 validation patients, "
            f"got {len(validation_records)}"
        )

    if test_count != 26:

        raise RuntimeError(
            "Expected 26 locked test patients, "
            f"got {test_count}"
        )

    print("\nDATASET")

    print(
        "Validation patients:",
        len(validation_records)
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "Test images and masks will not be loaded."
    )

    print("\nINFERENCE SETTINGS")

    print(
        "Patch:",
        patch_size
    )

    print(
        "Overlap:",
        args.overlap
    )

    print(
        "Sliding-window batch size:",
        args.sw_batch_size
    )

    print(
        "Thresholds:",
        args.thresholds
    )

    print(
        "Prediction save threshold:",
        args.save_threshold
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    patient_rows = []

    total_start = time.time()

    for case_index, record in enumerate(
        validation_records,
        start=1
    ):

        case_start = time.time()

        uid = record["uid"]

        print("\n" + "=" * 60)

        print(
            f"VALIDATION CASE {case_index}/"
            f"{len(validation_records)}"
        )

        print(
            "Patient:",
            uid
        )

        # Read only the validation case.
        image, mask = load_case(record)

        print(
            "Volume shape (Z,Y,X):",
            image.shape
        )

        probabilities = run_full_volume_inference(
            model=model,
            image=image,
            device=device,
            amp=amp,
            patch_size=patch_size,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size
        )

        target = mask > 0

        # Calculate metrics for every threshold.
        for threshold in args.thresholds:

            metrics = calculate_metrics(
                probabilities,
                target,
                threshold
            )

            row = {

                "uid": uid,

                "patient_id": record["patient_id"],

                "threshold": float(threshold),

                **metrics

            }

            patient_rows.append(row)

        # Save one binary prediction volume at the
        # requested reference threshold.
        saved_prediction = (
            probabilities >= args.save_threshold
        ).astype(np.uint8)

        save_dir = (
            OUTPUT_PREDICTION_DIR /
            f"threshold_{args.save_threshold:.2f}"
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            save_dir /
            f"{safe_filename(uid)}_pa_prediction.nii.gz"
        )

        save_prediction(
            saved_prediction,
            record["image"],
            output_path
        )

        # Display reference-threshold patient metrics.
        reference_metrics = calculate_metrics(
            probabilities,
            target,
            args.save_threshold
        )

        elapsed = (
            time.time() - case_start
        )

        print(
            f"Dice @ {args.save_threshold:.2f}: "
            f"{reference_metrics['dice']:.4f}"
        )

        print(
            f"Precision: "
            f"{reference_metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{reference_metrics['recall']:.4f}"
        )

        print(
            "Inference time:",
            round(elapsed, 1),
            "seconds"
        )

        print(
            "Saved prediction:",
            output_path
        )

        # Release current patient memory.
        del image
        del mask
        del target
        del probabilities
        del saved_prediction

        gc_collected = False

        if device.type == "cuda":

            torch.cuda.empty_cache()

        print(
            f"Completed {case_index}/"
            f"{len(validation_records)}",
            flush=True
        )

    # --------------------------------------------------------
    # Save per-patient metrics
    # --------------------------------------------------------

    per_patient_path = (
        OUTPUT_METRICS_DIR /
        "per_patient_metrics.csv"
    )

    patient_df = pd.DataFrame(
        patient_rows
    )

    patient_df.to_csv(
        per_patient_path,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate threshold metrics
    # --------------------------------------------------------

    summaries = []

    for threshold in args.thresholds:

        summary = summarize_threshold(
            threshold,
            patient_rows
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_METRICS_DIR /
        "threshold_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Select threshold using validation only
    # --------------------------------------------------------

    best_row = max(
        summaries,
        key=lambda row: row["macro_dice"]
    )

    best_threshold = best_row["threshold"]

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    total_elapsed = (
        time.time() - total_start
    )

    metadata = {

        "project": "NeuroVessel",

        "model": "V5B Residual 3D U-Net",

        "evaluation": "Full-volume validation",

        "checkpoint": str(CHECKPOINT_PATH),

        "checkpoint_epoch": int(
            config.get("epochs_requested", 30)
        ),

        "patch_size": patch_size,

        "overlap": args.overlap,

        "sw_batch_size": args.sw_batch_size,

        "intensity_window": [
            window_low,
            window_high
        ],

        "validation_patients": len(
            validation_records
        ),

        "locked_test_patients": test_count,

        "test_partition_loaded": False,

        "thresholds_evaluated": [
            float(x) for x in args.thresholds
        ],

        "best_validation_threshold": float(
            best_threshold
        ),

        "best_validation_macro_dice": float(
            best_row["macro_dice"]
        ),

        "saved_prediction_threshold": float(
            args.save_threshold
        ),

        "metric_definition": (
            "Full-volume metrics calculated over all voxels "
            "of each validation patient. Macro metrics are "
            "the unweighted average over patients. Micro "
            "metrics pool voxel counts across patients."
        ),

        "total_evaluation_seconds": total_elapsed

    }

    metadata_path = (
        OUTPUT_METRICS_DIR /
        "evaluation_summary.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "NEUROVESSEL V5B — FULL-VOLUME VALIDATION RESULTS"
    )

    print("=" * 75)

    display_columns = [
        "threshold",
        "macro_dice",
        "macro_precision",
        "macro_recall",
        "micro_dice"
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nBEST VALIDATION THRESHOLD")

    print(
        "Threshold:",
        best_threshold
    )

    print(
        "Macro full-volume Dice:",
        round(best_row["macro_dice"], 5)
    )

    print(
        "Macro precision:",
        round(best_row["macro_precision"], 5)
    )

    print(
        "Macro recall:",
        round(best_row["macro_recall"], 5)
    )

    print(
        "Micro full-volume Dice:",
        round(best_row["micro_dice"], 5)
    )

    print(
        "Macro Dice standard deviation:",
        round(best_row["dice_std"], 5)
    )

    print("\nOUTPUTS")

    print(
        "Per-patient metrics:",
        per_patient_path
    )

    print(
        "Threshold summary:",
        summary_path
    )

    print(
        "Evaluation configuration:",
        metadata_path
    )

    print(
        "Prediction masks:",
        OUTPUT_PREDICTION_DIR
    )

    print(
        "\nTotal evaluation time:",
        round(total_elapsed / 60.0, 2),
        "minutes"
    )

    print(
        "\nIMPORTANT: Threshold selection is based "
        "only on validation data."
    )

    print(
        "The locked test set remains untouched."
    )


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import gc

    main()
