
import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints"
    / "unet_v4_hard_negative_best.pt"
)

SPLIT_PATH = ROOT / "data" / "splits.json"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "unet_v4_hard_negative_threshold_sweep.csv"
PATIENT_CSV = OUTPUT_DIR / "unet_v4_hard_negative_threshold_patientwise.csv"
PLOT_PATH = OUTPUT_DIR / "unet_v4_hard_negative_threshold_sweep.png"
SUMMARY_JSON = OUTPUT_DIR / "unet_v4_hard_negative_threshold_summary.json"

# ---------------------------------------------------------
# IMPORT EXISTING V3 MODEL AND INFERENCE FUNCTIONS
# ---------------------------------------------------------

sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v3_frangi import (
    load_case,
    build_model,
    predict_volume,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

THRESHOLDS = [
    round(float(x), 2)
    for x in np.arange(0.10, 1.00, 0.05)
]

THRESHOLDS.append(0.99)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(prediction, target):

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice_denominator = (2 * tp) + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    dice = (
        (2.0 * tp / dice_denominator)
        if dice_denominator > 0 else 1.0
    )

    precision = (
        (tp / precision_denominator)
        if precision_denominator > 0 else 0.0
    )

    recall = (
        (tp / recall_denominator)
        if recall_denominator > 0 else 0.0
    )

    predicted_fraction = prediction.mean()
    target_fraction = target.mean()

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_fraction": float(predicted_fraction),
        "target_fraction": float(target_fraction),
    }


# ---------------------------------------------------------
# LOAD VALIDATION PATIENTS ONLY
# ---------------------------------------------------------

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

validation_ids = split["validation"]
test_ids = set(split["test"])

assert len(validation_ids) == 15, (
    f"Expected 15 validation patients, got {len(validation_ids)}"
)

assert not (set(validation_ids) & test_ids), (
    "ERROR: Validation and test patients overlap!"
)

print("=" * 65)
print("NEUROVESSEL V4 - THRESHOLD OPTIMIZATION")
print("=" * 65)

print("Device:", DEVICE)
print("Validation patients:", len(validation_ids))
print("Test patients:", len(test_ids))
print("Test patients will NOT be evaluated.")
print("Thresholds:", THRESHOLDS)

# ---------------------------------------------------------
# LOAD BEST CHECKPOINT
# ---------------------------------------------------------

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model = build_model(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nLoaded best V4 checkpoint.")
print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))
print(
    "Checkpoint validation summary:",
    checkpoint.get("validation_summary", "Not available")
)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

patient_rows = []

for patient_number, patient_id in enumerate(
    validation_ids, start=1
):

    print(
        f"\nPatient {patient_number}/{len(validation_ids)}: "
        f"{patient_id}"
    )

    # Load CT, Frangi and ground truth
    image, frangi, label = load_case(patient_id)

    target = np.asarray(label) > 0

    # Run inference only once per patient.
    # Reuse probabilities for all thresholds.
    probabilities = predict_volume(
        model,
        image,
        frangi,
        DEVICE
    )

    probabilities = np.asarray(probabilities)
    probabilities = np.squeeze(probabilities)

    if probabilities.shape != target.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id}: "
            f"prediction={probabilities.shape}, "
            f"target={target.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"Non-finite predictions for {patient_id}"
        )

    print(
        "Probability range:",
        float(probabilities.min()),
        "to",
        float(probabilities.max())
    )

    # Evaluate every threshold
    for threshold in THRESHOLDS:

        prediction = probabilities >= threshold

        metrics = calculate_metrics(
            prediction,
            target
        )

        patient_rows.append({
            "patient_id": patient_id,
            "threshold": threshold,
            **metrics,
        })

        print(
            f"  Threshold={threshold:.2f} | "
            f"Dice={metrics['dice']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Predicted={metrics['predicted_fraction'] * 100:.3f}%"
        )

    del image, frangi, label, probabilities, target

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------
# SAVE PATIENT-WISE RESULTS
# ---------------------------------------------------------

patient_fieldnames = [
    "patient_id",
    "threshold",
    "dice",
    "precision",
    "recall",
    "predicted_fraction",
    "target_fraction",
]

with open(PATIENT_CSV, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=patient_fieldnames
    )

    writer.writeheader()
    writer.writerows(patient_rows)


# ---------------------------------------------------------
# AGGREGATE RESULTS ACROSS PATIENTS
# ---------------------------------------------------------

summary_rows = []

for threshold in THRESHOLDS:

    rows = [
        row for row in patient_rows
        if row["threshold"] == threshold
    ]

    summary_rows.append({
        "threshold": threshold,
        "mean_dice": float(np.mean(
            [r["dice"] for r in rows]
        )),
        "std_dice": float(np.std(
            [r["dice"] for r in rows]
        )),
        "mean_precision": float(np.mean(
            [r["precision"] for r in rows]
        )),
        "mean_recall": float(np.mean(
            [r["recall"] for r in rows]
        )),
        "mean_predicted_fraction": float(np.mean(
            [r["predicted_fraction"] for r in rows]
        )),
        "mean_target_fraction": float(np.mean(
            [r["target_fraction"] for r in rows]
        )),
        "patients": len(rows),
    })


# ---------------------------------------------------------
# FIND BEST THRESHOLD
# ---------------------------------------------------------

best_result = max(
    summary_rows,
    key=lambda row: row["mean_dice"]
)

print("\n" + "=" * 65)
print("THRESHOLD SWEEP SUMMARY")
print("=" * 65)

for row in summary_rows:

    print(
        f"Threshold={row['threshold']:.2f} | "
        f"Dice={row['mean_dice']:.4f} | "
        f"Precision={row['mean_precision']:.4f} | "
        f"Recall={row['mean_recall']:.4f} | "
        f"Predicted={row['mean_predicted_fraction'] * 100:.3f}%"
    )

print("\nBEST VALIDATION THRESHOLD")
print("Threshold:", best_result["threshold"])
print("Mean Dice:", best_result["mean_dice"])
print("Mean Precision:", best_result["mean_precision"])
print("Mean Recall:", best_result["mean_recall"])
print(
    "Predicted foreground:",
    best_result["mean_predicted_fraction"] * 100,
    "%"
)


# ---------------------------------------------------------
# SAVE SUMMARY CSV
# ---------------------------------------------------------

summary_fieldnames = list(summary_rows[0].keys())

with open(SUMMARY_CSV, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fieldnames
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ---------------------------------------------------------
# SAVE GRAPH
# ---------------------------------------------------------

threshold_values = [
    row["threshold"] for row in summary_rows
]

dice_values = [
    row["mean_dice"] for row in summary_rows
]

precision_values = [
    row["mean_precision"] for row in summary_rows
]

recall_values = [
    row["mean_recall"] for row in summary_rows
]

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_values,
    dice_values,
    marker="o",
    label="Mean Dice"
)

plt.plot(
    threshold_values,
    precision_values,
    marker="s",
    label="Mean Precision"
)

plt.plot(
    threshold_values,
    recall_values,
    marker="^",
    label="Mean Recall"
)

plt.axvline(
    best_result["threshold"],
    linestyle="--",
    label=f"Best Dice threshold = {best_result['threshold']:.2f}"
)

plt.xlabel("Probability Threshold")
plt.ylabel("Metric Value")
plt.title("NeuroVessel V4 - Validation Threshold Sweep")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=300)
plt.close()


# ---------------------------------------------------------
# SAVE JSON SUMMARY
# ---------------------------------------------------------

final_summary = {
    "model": "NeuroVessel V4 Hard Negative",
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_epoch": checkpoint.get("epoch"),
    "validation_patients": len(validation_ids),
    "test_patients_evaluated": 0,
    "thresholds": THRESHOLDS,
    "best_threshold": best_result["threshold"],
    "best_validation_dice": best_result["mean_dice"],
    "best_validation_precision": best_result["mean_precision"],
    "best_validation_recall": best_result["mean_recall"],
    "best_predicted_foreground_fraction": (
        best_result["mean_predicted_fraction"]
    ),
    "test_set_evaluated": False,
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(final_summary, f, indent=4)


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("V4 THRESHOLD SWEEP COMPLETED")
print("=" * 65)

print("Summary CSV:", SUMMARY_CSV)
print("Patient-wise CSV:", PATIENT_CSV)
print("Graph:", PLOT_PATH)
print("JSON summary:", SUMMARY_JSON)

print("\nTest patients were NOT evaluated.")
