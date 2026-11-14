
import csv
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from train_unet_v3_frangi import (
    DATA_ROOT,
    SPLIT_FILE,
    FRANGI_ROOT,
    PATCH_SIZE,
    OVERLAP,
    build_model,
    load_case,
    predict_volume,
)

# ---------------------------------------------------------
# FINAL LOCKED CONFIGURATION
# Do not tune these values using the test set.
# ---------------------------------------------------------

CHECKPOINT_PATH = Path("outputs/checkpoints/unet_v4_hard_negative_best.pt")

OUTPUT_DIR = Path("outputs/metrics/final_test_v4")
PATIENT_CSV = OUTPUT_DIR / "final_test_patient_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "final_test_summary.json"
COMPLETE_MARKER = OUTPUT_DIR / "FINAL_COMPLETE.json"

PROBABILITY_THRESHOLD = 0.75
MIN_COMPONENT_SIZE = 5000
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)

EXPECTED_TEST_PATIENTS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_test_inputs(patient_ids):
    """Check all test inputs before model evaluation."""

    print("\nPreflight: checking test files...")

    for patient_id in patient_ids:
        folder = DATA_ROOT / patient_id
        image_path = folder / "image" / f"{patient_id}.nii.gz"
        label_path = folder / "label" / f"{patient_id}.nii.gz"
        frangi_path = FRANGI_ROOT / f"{patient_id}.npy"

        for path in (image_path, label_path, frangi_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        if image_nii.shape != label_nii.shape:
            raise ValueError(f"CT/label shape mismatch: {patient_id}")

        if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-4):
            raise ValueError(f"CT/label affine mismatch: {patient_id}")

        frangi = np.load(frangi_path, mmap_mode="r")

        if tuple(frangi.shape) != tuple(image_nii.shape):
            raise ValueError(f"CT/Frangi shape mismatch: {patient_id}")

        if not np.isfinite(frangi).all():
            raise ValueError(f"Non-finite Frangi values: {patient_id}")

        del image_nii, label_nii, frangi

        print(f"  OK: {patient_id}")

    print("Preflight successful for all test patients.")


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    pred_count = tp + fp
    target_count = tp + fn

    dice = (
        2.0 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0 else 1.0
    )

    precision = tp / pred_count if pred_count else 0.0
    recall = tp / target_count if target_count else 0.0

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_voxels": int(pred_count),
        "target_voxels": int(target_count),
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
    }


def filter_components(prediction):
    """Retain 26-connected components with >= 5000 voxels."""

    labels, number = ndimage.label(
        prediction,
        structure=CONNECTIVITY
    )

    if number == 0:
        return np.zeros_like(prediction, dtype=bool), 0, 0

    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_COMPONENT_SIZE
    keep[0] = False

    filtered = keep[labels]

    remaining = int(np.count_nonzero(keep))

    return filtered, int(number), remaining


# ---------------------------------------------------------
# MAIN FINAL TEST
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("NEUROVESSEL - V4 FINAL TEST EVALUATION")
    print("=" * 65)

    # Refuse to overwrite an already completed final evaluation.
    if COMPLETE_MARKER.exists() or SUMMARY_JSON.exists():
        raise RuntimeError(
            "A completed final test report already exists. "
            "Refusing to evaluate the test set again or overwrite results."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    # Load split and verify patient separation.
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_ids = set(split["train"])
    validation_ids = set(split["validation"])
    test_ids = list(split["test"])

    if len(test_ids) != EXPECTED_TEST_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_PATIENTS} test patients, "
            f"found {len(test_ids)}."
        )

    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate patient IDs in test split.")

    if set(test_ids) & train_ids:
        raise ValueError("Test/train patient overlap detected.")

    if set(test_ids) & validation_ids:
        raise ValueError("Test/validation patient overlap detected.")

    print(f"\nTraining patients:   {len(train_ids)}")
    print(f"Validation patients: {len(validation_ids)}")
    print(f"Test patients:       {len(test_ids)}")

    print(f"\nDevice: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Threshold: {PROBABILITY_THRESHOLD}")
    print(f"Minimum component size: {MIN_COMPONENT_SIZE}")
    print("Connectivity: 26-connected")

    # Verify all inputs before starting evaluation.
    validate_test_inputs(test_ids)

    # Load checkpoint and model.
    print("\nLoading final V4 checkpoint...")

    checkpoint_hash = sha256_file(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    checkpoint_epoch = checkpoint.get("epoch", None)

    print("Checkpoint loaded successfully.")
    print(f"Checkpoint epoch metadata: {checkpoint_epoch}")

    # Prepare output directory only after preflight.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total_start = time.time()

    # Each test patient is evaluated exactly once.
    for index, patient_id in enumerate(test_ids, start=1):

        print("\n" + "-" * 60)
        print(f"Patient {index}/{len(test_ids)}: {patient_id}")

        patient_start = time.time()

        # Load CT, Frangi feature and ground-truth label.
        image, frangi, label = load_case(patient_id)

        # Full-volume prediction using the existing V4 inference code.
        probabilities = predict_volume(
            model,
            image,
            frangi,
            DEVICE
        )

        if probabilities.shape != label.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: {patient_id}"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite predictions: {patient_id}"
            )

        # Fixed probability threshold.
        raw_prediction = probabilities >= PROBABILITY_THRESHOLD

        # Fixed connected-component filtering.
        filtered_prediction, original_components, remaining_components = (
            filter_components(raw_prediction)
        )

        # Final metrics are calculated only on filtered predictions.
        metrics = calculate_metrics(filtered_prediction, label)

        row = {
            "patient_id": patient_id,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "minimum_component_size": MIN_COMPONENT_SIZE,
            "connectivity": 26,
            "original_components": original_components,
            "remaining_components": remaining_components,
            **metrics,
            "inference_seconds": round(time.time() - patient_start, 2),
        }

        rows.append(row)

        print(f"Dice:       {metrics['dice']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"Components: {original_components} -> {remaining_components}")
        print(f"FP voxels:  {metrics['fp']:,}")
        print(f"Time:       {row['inference_seconds']:.2f} seconds")

        del image, frangi, label
        del probabilities, raw_prediction, filtered_prediction

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # AGGREGATE RESULTS
    # -----------------------------------------------------

    print("\nCalculating final test summary...")

    metric_names = ["dice", "precision", "recall"]

    macro_summary = {}

    for metric in metric_names:
        values = np.asarray(
            [row[metric] for row in rows],
            dtype=np.float64
        )

        macro_summary[f"mean_{metric}"] = float(values.mean())
        macro_summary[f"std_{metric}"] = float(values.std(ddof=1))

    total_tp = sum(row["tp"] for row in rows)
    total_fp = sum(row["fp"] for row in rows)
    total_fn = sum(row["fn"] for row in rows)

    micro_summary = {
        "micro_dice": float(
            2 * total_tp / (2 * total_tp + total_fp + total_fn)
        ) if (2 * total_tp + total_fp + total_fn) else 1.0,
        "micro_precision": float(
            total_tp / (total_tp + total_fp)
        ) if (total_tp + total_fp) else 0.0,
        "micro_recall": float(
            total_tp / (total_tp + total_fn)
        ) if (total_tp + total_fn) else 0.0,
    }

    elapsed = time.time() - total_start

    summary = {
        "project": "NeuroVessel",
        "experiment": "V4 final held-out test evaluation",
        "evaluation_status": "completed",
        "test_set_evaluated": True,
        "test_patients_evaluated": len(rows),
        "test_patient_ids": test_ids,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_epoch_metadata": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_hash,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "minimum_component_size_voxels": MIN_COMPONENT_SIZE,
        "connectivity": 26,
        "patch_size": PATCH_SIZE,
        "sliding_window_overlap": OVERLAP,
        "inference_device": str(DEVICE),
        "macro_patient_metrics": macro_summary,
        "pooled_voxel_metrics": micro_summary,
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(total_fn),
        "total_inference_seconds": round(elapsed, 2),
        "evaluation_timestamp": datetime.now().astimezone().isoformat(),
        "selection_note": (
            "Threshold and component size were fixed using validation "
            "experiments before this held-out test evaluation. "
            "No test-set parameter tuning was performed."
        ),
    }

    # Save patient-level results.
    fieldnames = list(rows[0].keys())

    with open(PATIENT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save aggregate report.
    with open(SUMMARY_JSON, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # Completion marker is written last.
    with open(COMPLETE_MARKER, "w", encoding="utf-8") as file:
        json.dump(
            {
                "status": "completed",
                "patients_evaluated": len(rows),
                "summary_file": str(SUMMARY_JSON),
                "patient_results_file": str(PATIENT_CSV),
            },
            file,
            indent=4
        )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Patients evaluated: {len(rows)}")

    print("\nMACRO PATIENT-LEVEL METRICS")
    for metric in metric_names:
        print(
            f"{metric.capitalize():10s}: "
            f"{macro_summary[f'mean_{metric}']:.4f} "
            f"+/- {macro_summary[f'std_{metric}']:.4f}"
        )

    print("\nPOOLED VOXEL-LEVEL METRICS")
    for metric, value in micro_summary.items():
        print(f"{metric:20s}: {value:.4f}")

    print("\nSaved files:")
    print(PATIENT_CSV)
    print(SUMMARY_JSON)
    print(COMPLETE_MARKER)

    print("\nTest evaluation is complete. Do not tune using these results.")


if __name__ == "__main__":
    main()
