
"""
NeuroVessel V4 - Frangi Response at Segmentation Errors

Analyzes Frangi responses at:
1. True Positive (TP)
2. False Positive (FP)
3. False Negative (FN)

Validation patients only.
Does not train or modify any model.
Does not access the test set.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v3_frangi import (
    load_case,
    build_model,
    predict_volume,
)

SPLIT_FILE = ROOT / "data" / "splits.json"

CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints"
    / "unet_v4_hard_negative_best.pt"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics"
    / "v4_frangi_error_analysis"
)

PATIENT_IDS = [
    "PA000314",
    "PA000194",
    "PA000005",
]

THRESHOLD = 0.75
MAX_SAMPLES = 50000
RANDOM_SEED = 42


# ============================================================
# VALIDATION SAFETY
# ============================================================

def validate_patient_ids():

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    validation_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    for patient_id in PATIENT_IDS:

        if patient_id not in validation_ids:
            raise ValueError(
                f"{patient_id} is not in validation split."
            )

        if patient_id in test_ids:
            raise RuntimeError(
                f"Test patient detected: {patient_id}"
            )

    print("Validation split safety check passed.")
    print("Test patients will NOT be accessed.")


# ============================================================
# MEMORY-EFFICIENT UNIFORM 3D SAMPLING
# ============================================================

def sample_mask_values(feature, mask, max_samples, rng):

    slice_counts = np.count_nonzero(
        mask, axis=(0, 1)
    ).astype(np.int64)

    total = int(slice_counts.sum())

    if total == 0:
        return np.empty(0, dtype=np.float32)

    n_samples = min(max_samples, total)

    exact = (
        slice_counts.astype(np.float64)
        * n_samples / total
    )

    allocation = np.floor(exact).astype(np.int64)

    remaining = n_samples - int(allocation.sum())

    if remaining > 0:

        remainders = exact - allocation

        valid = np.flatnonzero(
            allocation < slice_counts
        )

        order = valid[
            np.argsort(remainders[valid])[::-1]
        ]

        for z in order[:remaining]:
            allocation[z] += 1

    selected = []

    for z, count in enumerate(allocation):

        if count == 0:
            continue

        indices = np.flatnonzero(mask[:, :, z])

        if count < indices.size:

            indices = rng.choice(
                indices,
                size=int(count),
                replace=False
            )

        values = feature[:, :, z].reshape(-1)[indices]

        selected.append(
            values.astype(np.float32, copy=False)
        )

    if not selected:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(selected)


# ============================================================
# DISTRIBUTION STATISTICS
# ============================================================

def calculate_distribution(values):

    if values.size == 0:
        return {}

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def calculate_auc(positive, negative):

    if positive.size == 0 or negative.size == 0:
        return float("nan")

    y_true = np.concatenate([
        np.ones(positive.size, dtype=np.uint8),
        np.zeros(negative.size, dtype=np.uint8)
    ])

    y_score = np.concatenate([
        positive,
        negative
    ])

    return float(
        roc_auc_score(y_true, y_score)
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_distribution_plot(
    patient_id,
    tp_values,
    fp_values,
    fn_values
):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )

    categories = [
        ("TP", tp_values),
        ("FP", fp_values),
        ("FN", fn_values),
    ]

    for ax, (name, values) in zip(axes, categories):

        if values.size:

            ax.hist(
                values,
                bins=60,
                range=(0, 1),
                density=True,
                alpha=0.8
            )

            ax.axvline(
                np.mean(values),
                linestyle="--",
                label=f"Mean={np.mean(values):.4f}"
            )

            ax.legend()

        ax.set_title(f"{name} Frangi Distribution")
        ax.set_xlabel("Frangi response")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"V4 Frangi Error Analysis - {patient_id}",
        fontsize=15
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{patient_id}_frangi_distributions.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# PATIENT ANALYSIS
# ============================================================

def analyze_patient(
    patient_id,
    model,
    device,
    patient_number
):

    print("\n" + "=" * 65)
    print(f"PATIENT {patient_number}: {patient_id}")
    print("=" * 65)

    rng = np.random.default_rng(
        RANDOM_SEED + patient_number
    )

    image, frangi, label = load_case(patient_id)

    frangi = np.asarray(
        frangi, dtype=np.float32
    )

    label = np.asarray(label, dtype=bool)

    probabilities = predict_volume(
        model,
        image,
        frangi,
        device
    )

    prediction = probabilities >= THRESHOLD

    # Exact voxel categories.
    tp_mask = prediction & label
    fp_mask = prediction & ~label
    fn_mask = ~prediction & label

    # Sample Frangi responses uniformly across 3D volume.
    tp_values = sample_mask_values(
        frangi, tp_mask, MAX_SAMPLES, rng
    )

    fp_values = sample_mask_values(
        frangi, fp_mask, MAX_SAMPLES, rng
    )

    fn_values = sample_mask_values(
        frangi, fn_mask, MAX_SAMPLES, rng
    )

    distributions = {
        "TP": calculate_distribution(tp_values),
        "FP": calculate_distribution(fp_values),
        "FN": calculate_distribution(fn_values),
    }

    auc_tp_fp = calculate_auc(
        tp_values, fp_values
    )

    auc_tp_fn = calculate_auc(
        tp_values, fn_values
    )

    intersection = int(tp_mask.sum())

    dice = (
        2.0 * intersection
        / (int(prediction.sum()) + int(label.sum()))
    )

    precision = (
        intersection / int(prediction.sum())
        if prediction.sum() else 0.0
    )

    recall = (
        intersection / int(label.sum())
        if label.sum() else 0.0
    )

    print("\nSegmentation metrics:")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nFrangi distributions:")

    for category, stats in distributions.items():

        if not stats:
            print(f"{category}: No voxels")
            continue

        print(
            f"{category}: "
            f"Mean={stats['mean']:.6f}, "
            f"Median={stats['median']:.6f}, "
            f"P95={stats['p95']:.6f}, "
            f"Count={stats['count']}"
        )

    print("\nFrangi discrimination:")
    print(f"TP vs FP AUC: {auc_tp_fp:.4f}")
    print(f"TP vs FN AUC: {auc_tp_fn:.4f}")

    plot_path = save_distribution_plot(
        patient_id,
        tp_values,
        fp_values,
        fn_values
    )

    print("\nPlot saved:", plot_path)

    row = {
        "patient_id": patient_id,
        "threshold": THRESHOLD,
        "dice": dice,
        "precision": precision,
        "recall": recall,

        "tp_voxels": int(tp_mask.sum()),
        "fp_voxels": int(fp_mask.sum()),
        "fn_voxels": int(fn_mask.sum()),

        "tp_frangi_mean": distributions["TP"].get("mean"),
        "tp_frangi_median": distributions["TP"].get("median"),
        "tp_frangi_p95": distributions["TP"].get("p95"),

        "fp_frangi_mean": distributions["FP"].get("mean"),
        "fp_frangi_median": distributions["FP"].get("median"),
        "fp_frangi_p95": distributions["FP"].get("p95"),

        "fn_frangi_mean": distributions["FN"].get("mean"),
        "fn_frangi_median": distributions["FN"].get("median"),
        "fn_frangi_p95": distributions["FN"].get("p95"),

        "auc_tp_vs_fp": auc_tp_fp,
        "auc_tp_vs_fn": auc_tp_fn,
    }

    del image, frangi, label
    del probabilities, prediction
    del tp_mask, fp_mask, fn_mask

    torch.cuda.empty_cache()

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V4 - FRANGI ERROR ANALYSIS")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    validate_patient_ids()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading V4 checkpoint...")

    model = build_model(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown")
    )

    print("Threshold:", THRESHOLD)
    print("Patients:", PATIENT_IDS)
    print("Maximum samples per category:", MAX_SAMPLES)

    results = []

    for number, patient_id in enumerate(
        PATIENT_IDS,
        start=1
    ):

        row = analyze_patient(
            patient_id,
            model,
            device,
            number
        )

        results.append(row)

    # Save patient-level CSV.
    csv_path = (
        OUTPUT_DIR / "v4_frangi_error_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # Save summary JSON.
    summary = {
        "patients_analyzed": len(results),
        "patient_ids": PATIENT_IDS,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": THRESHOLD,
        "test_patients_evaluated": 0,
        "results": results,
    }

    json_path = (
        OUTPUT_DIR / "v4_frangi_error_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("FRANGI ERROR ANALYSIS COMPLETE")
    print("=" * 65)

    print("\nCSV:", csv_path)
    print("JSON:", json_path)

    print("\nOnly validation patients were analyzed.")
    print("No model training was performed.")
    print("The locked test set was NOT accessed.")


if __name__ == "__main__":
    main()
