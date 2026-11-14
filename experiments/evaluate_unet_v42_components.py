
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v3_frangi import (
    load_case,
    build_model,
    predict_volume,
)

CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints"
    / "unet_v4_hard_negative_best.pt"
)

SPLIT_PATH = ROOT / "data" / "splits.json"

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics"
    / "v42_connected_components"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_CSV = OUTPUT_DIR / "v42_patient_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "v42_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "v42_summary.json"

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Fixed using the previous validation threshold experiment.
PROBABILITY_THRESHOLD = 0.75

# Minimum connected-component sizes in voxels.
# Zero means no component filtering.
MIN_COMPONENT_SIZES = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
]

# 26-connectivity in 3D.
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    dice = (
        2.0 * tp / dice_denominator
        if dice_denominator > 0 else 1.0
    )

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_voxels": int(prediction.sum()),
        "target_voxels": int(target.sum()),
        "predicted_fraction": float(prediction.mean()),
    }


# ============================================================
# CONNECTED COMPONENT ANALYSIS
# ============================================================

def analyze_components(binary_prediction):

    labels, number_of_components = ndimage.label(
        binary_prediction,
        structure=CONNECTIVITY
    )

    if number_of_components == 0:

        return labels, np.array([], dtype=np.int64)

    component_sizes = np.bincount(
        labels.ravel()
    )[1:]

    return labels, component_sizes


def filter_components(
    component_labels,
    component_sizes,
    minimum_size
):

    if minimum_size == 0:

        return component_labels > 0

    if len(component_sizes) == 0:

        return np.zeros(
            component_labels.shape,
            dtype=bool
        )

    keep_labels = (
        np.where(component_sizes >= minimum_size)[0] + 1
    )

    return np.isin(
        component_labels,
        keep_labels
    )


# ============================================================
# LOAD DATA SPLIT
# ============================================================

with open(SPLIT_PATH, "r", encoding="utf-8") as f:

    split = json.load(f)

validation_ids = split["validation"]
test_ids = set(split["test"])

assert len(validation_ids) == 15, (
    f"Expected 15 validation patients, "
    f"found {len(validation_ids)}"
)

assert not (
    set(validation_ids) & test_ids
), "Validation/test leakage detected!"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("NEUROVESSEL V4.2 - CONNECTED COMPONENT EXPERIMENT")
print("=" * 70)

print("Device:", DEVICE)
print("Probability threshold:", PROBABILITY_THRESHOLD)
print("Validation patients:", len(validation_ids))
print("Test patients:", len(test_ids))
print("Test patients will NOT be evaluated.")

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = build_model(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded V4 best checkpoint.")
print("Checkpoint epoch:", checkpoint.get("epoch"))


# ============================================================
# MAIN EVALUATION
# ============================================================

patient_rows = []

start_total = time.time()

for patient_number, patient_id in enumerate(
    validation_ids,
    start=1
):

    print("\n" + "-" * 65)

    print(
        f"Patient {patient_number}/{len(validation_ids)}: "
        f"{patient_id}"
    )

    start_patient = time.time()

    # Load CT, Frangi and ground truth.
    image, frangi, label = load_case(patient_id)

    target = np.asarray(label) > 0

    # Full-volume inference.
    probabilities = predict_volume(
        model,
        image,
        frangi,
        DEVICE
    )

    probabilities = np.squeeze(probabilities)

    if probabilities.shape != target.shape:

        raise ValueError(
            f"Shape mismatch for {patient_id}: "
            f"{probabilities.shape} vs {target.shape}"
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            f"Non-finite probabilities in {patient_id}"
        )

    # Initial binary prediction.
    binary_prediction = (
        probabilities >= PROBABILITY_THRESHOLD
    )

    # Connected-component analysis.
    component_labels, component_sizes = (
        analyze_components(binary_prediction)
    )

    original_component_count = len(component_sizes)

    largest_component = (
        int(component_sizes.max())
        if len(component_sizes) > 0 else 0
    )

    print(
        "Original predicted components:",
        original_component_count
    )

    print(
        "Largest component size:",
        largest_component
    )

    # Evaluate every component-size threshold.
    for minimum_size in MIN_COMPONENT_SIZES:

        filtered_prediction = filter_components(
            component_labels,
            component_sizes,
            minimum_size
        )

        metrics = calculate_metrics(
            filtered_prediction,
            target
        )

        remaining_components = int(
            np.sum(component_sizes >= minimum_size)
        )

        row = {
            "patient_id": patient_id,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "minimum_component_size": minimum_size,
            "original_components": original_component_count,
            "remaining_components": remaining_components,
            "largest_component_voxels": largest_component,
            **metrics,
        }

        patient_rows.append(row)

        print(
            f"MinSize={minimum_size:5d} | "
            f"Components={remaining_components:5d} | "
            f"Dice={metrics['dice']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"FP={metrics['fp']}"
        )

    print(
        "Patient processing time:",
        round(time.time() - start_patient, 2),
        "seconds"
    )

    del image, frangi, label
    del probabilities, binary_prediction
    del component_labels, component_sizes

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE PATIENT-WISE CSV
# ============================================================

if not patient_rows:

    raise RuntimeError("No patient results were generated.")

fieldnames = list(patient_rows[0].keys())

with open(
    PATIENT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(patient_rows)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

summary_rows = []

for minimum_size in MIN_COMPONENT_SIZES:

    rows = [
        row for row in patient_rows
        if row["minimum_component_size"] == minimum_size
    ]

    summary_rows.append({

        "minimum_component_size": minimum_size,

        "mean_dice": float(np.mean([
            row["dice"] for row in rows
        ])),

        "std_dice": float(np.std([
            row["dice"] for row in rows
        ])),

        "mean_precision": float(np.mean([
            row["precision"] for row in rows
        ])),

        "mean_recall": float(np.mean([
            row["recall"] for row in rows
        ])),

        "mean_fp": float(np.mean([
            row["fp"] for row in rows
        ])),

        "mean_fn": float(np.mean([
            row["fn"] for row in rows
        ])),

        "mean_remaining_components": float(np.mean([
            row["remaining_components"] for row in rows
        ])),

        "mean_predicted_fraction": float(np.mean([
            row["predicted_fraction"] for row in rows
        ])),

        "patients": len(rows),
    })


# ============================================================
# FIND BEST VALIDATION SETTING
# ============================================================

best_result = max(
    summary_rows,
    key=lambda row: row["mean_dice"]
)

print("\n" + "=" * 70)
print("V4.2 CONNECTED COMPONENT RESULTS")
print("=" * 70)

for row in summary_rows:

    print(
        f"MinSize={row['minimum_component_size']:5d} | "
        f"Dice={row['mean_dice']:.4f} | "
        f"Precision={row['mean_precision']:.4f} | "
        f"Recall={row['mean_recall']:.4f} | "
        f"FP={row['mean_fp']:.0f} | "
        f"Components={row['mean_remaining_components']:.1f}"
    )

print("\nBEST VALIDATION SETTING")

print(
    "Minimum component size:",
    best_result["minimum_component_size"]
)

print(
    "Mean Dice:",
    best_result["mean_dice"]
)

print(
    "Mean Precision:",
    best_result["mean_precision"]
)

print(
    "Mean Recall:",
    best_result["mean_recall"]
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

final_summary = {

    "experiment": "NeuroVessel V4.2 Connected Component Analysis",

    "checkpoint": str(CHECKPOINT_PATH),

    "checkpoint_epoch": checkpoint.get("epoch"),

    "probability_threshold": PROBABILITY_THRESHOLD,

    "validation_patients": len(validation_ids),

    "test_patients_evaluated": 0,

    "test_set_evaluated": False,

    "minimum_component_sizes": MIN_COMPONENT_SIZES,

    "best_validation_minimum_component_size":
        best_result["minimum_component_size"],

    "best_validation_mean_dice":
        best_result["mean_dice"],

    "best_validation_mean_precision":
        best_result["mean_precision"],

    "best_validation_mean_recall":
        best_result["mean_recall"],

    "results": summary_rows,
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_summary,
        f,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("V4.2 EXPERIMENT COMPLETED")
print("=" * 70)

print("Patient results:", PATIENT_CSV)
print("Summary CSV:", SUMMARY_CSV)
print("Summary JSON:", SUMMARY_JSON)

print(
    "Total time:",
    round(time.time() - start_total, 2),
    "seconds"
)

print("\nTest patients were NOT evaluated.")
