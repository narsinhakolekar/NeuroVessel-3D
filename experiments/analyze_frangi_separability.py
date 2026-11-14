
"""
NeuroVessel: Frangi Feature Separability Analysis

Evaluates whether Frangi vesselness distinguishes:
1. Ground-truth pulmonary artery voxels.
2. Background voxels.
3. A 5-pixel 2D surrounding region around PA annotations.

Validation patients only.
No training. No test-set access.
"""

import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train_unet_v3_frangi import load_case


SPLIT_FILE = PROJECT_ROOT / "data" / "splits.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "frangi_separability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES_PER_CLASS = 50000
RANDOM_SEED = 42
CHUNK_SIZE = 2_000_000
RING_ITERATIONS = 5


# ---------------------------------------------------------
# Read validation patient IDs
# ---------------------------------------------------------

def get_validation_ids():

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    possible_keys = [
        "validation",
        "val",
        "validation_ids",
        "val_ids",
    ]

    for key in possible_keys:

        if key in splits:
            ids = splits[key]

            if isinstance(ids, list) and len(ids) > 0:
                return ids

    raise ValueError(
        "Could not find validation patients in data/splits.json. "
        f"Available keys: {list(splits.keys())}"
    )


# ---------------------------------------------------------
# Sample feature values without copying entire background
# ---------------------------------------------------------

def sample_masked_values(
    feature,
    mask,
    max_samples=MAX_SAMPLES_PER_CLASS,
    seed=42,
):
    """
    Randomly sample feature values where mask is True.

    Processes flattened arrays in chunks to reduce memory use.
    """

    rng = np.random.default_rng(seed)

    feature_flat = feature.reshape(-1)
    mask_flat = mask.reshape(-1)

    selected = []
    total_selected = 0

    for start in range(0, feature_flat.size, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, feature_flat.size)

        values = feature_flat[start:end]
        current_mask = mask_flat[start:end]

        candidates = values[current_mask]

        if candidates.size == 0:
            continue

        remaining = max_samples - total_selected

        if remaining <= 0:
            break

        if candidates.size > remaining:
            indices = rng.choice(
                candidates.size,
                size=remaining,
                replace=False,
            )
            candidates = candidates[indices]

        selected.append(candidates.astype(np.float32, copy=False))
        total_selected += candidates.size

        if total_selected >= max_samples:
            break

    if not selected:
        return np.array([], dtype=np.float32)

    return np.concatenate(selected)


# ---------------------------------------------------------
# Build surrounding PA region slice by slice
# ---------------------------------------------------------

def sample_surrounding_region(
    feature,
    label,
    max_samples=MAX_SAMPLES_PER_CLASS,
    seed=42,
):
    """
    Samples a 5-pixel 2D surrounding ring around the PA label
    independently on each axial slice.

    The ring excludes the PA annotation itself.
    """

    rng = np.random.default_rng(seed)

    selected = []
    total_selected = 0

    for z in range(label.shape[2]):

        gt_slice = label[:, :, z]

        if not gt_slice.any():
            continue

        dilated = binary_dilation(
            gt_slice,
            iterations=RING_ITERATIONS,
        )

        ring = dilated & (~gt_slice)

        values = feature[:, :, z][ring]

        if values.size == 0:
            continue

        remaining = max_samples - total_selected

        if remaining <= 0:
            break

        if values.size > remaining:
            indices = rng.choice(
                values.size,
                size=remaining,
                replace=False,
            )
            values = values[indices]

        selected.append(values.astype(np.float32, copy=False))
        total_selected += values.size

        if total_selected >= max_samples:
            break

    if not selected:
        return np.array([], dtype=np.float32)

    return np.concatenate(selected)


# ---------------------------------------------------------
# Calculate feature statistics
# ---------------------------------------------------------

def calculate_statistics(values):

    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values)),
    }


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("NEUROVESSEL - FRANGI FEATURE SEPARABILITY ANALYSIS")
    print("=" * 65)

    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Test patients will NOT be evaluated.")

    validation_ids = get_validation_ids()

    print("Validation patients:", len(validation_ids))
    print("Maximum samples per class:", MAX_SAMPLES_PER_CLASS)

    patient_results = []

    pooled_pa = []
    pooled_background = []

    # -----------------------------------------------------
    # Process each validation patient
    # -----------------------------------------------------

    for patient_index, patient_id in enumerate(validation_ids, 1):

        print(
            f"\nPatient {patient_index}/{len(validation_ids)}: "
            f"{patient_id}"
        )

        image, frangi, label = load_case(patient_id)

        image = np.asarray(image)
        frangi = np.asarray(frangi, dtype=np.float32)
        label = np.asarray(label) > 0

        if image.shape != frangi.shape or image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: "
                f"CT={image.shape}, Frangi={frangi.shape}, "
                f"Label={label.shape}"
            )

        if not np.isfinite(frangi).all():
            raise ValueError(
                f"Non-finite Frangi values in {patient_id}"
            )

        if not label.any():
            print("WARNING: Empty PA label. Skipping.")
            continue

        background = ~label

        # Sample PA and background feature values.
        pa_values = sample_masked_values(
            frangi,
            label,
            seed=RANDOM_SEED + patient_index,
        )

        bg_values = sample_masked_values(
            frangi,
            background,
            seed=RANDOM_SEED + 1000 + patient_index,
        )

        # Sample surrounding region.
        ring_values = sample_surrounding_region(
            frangi,
            label,
            seed=RANDOM_SEED + 2000 + patient_index,
        )

        if pa_values.size == 0 or bg_values.size == 0:
            print("WARNING: Insufficient samples. Skipping.")
            continue

        pa_stats = calculate_statistics(pa_values)
        bg_stats = calculate_statistics(bg_values)
        ring_stats = calculate_statistics(ring_values)

        # Frangi-only ROC-AUC.
        # This is a diagnostic feature-separability measure,
        # not a trained segmentation-model metric.
        y_true = np.concatenate([
            np.ones(pa_values.size, dtype=np.uint8),
            np.zeros(bg_values.size, dtype=np.uint8),
        ])

        y_score = np.concatenate([
            pa_values,
            bg_values,
        ])

        auc = float(roc_auc_score(y_true, y_score))

        # Mean ratio.
        if bg_stats["mean"] > 1e-12:
            mean_ratio = pa_stats["mean"] / bg_stats["mean"]
        else:
            mean_ratio = float("nan")

        result = {
            "patient_id": patient_id,
            "pa_mean": pa_stats["mean"],
            "pa_median": pa_stats["median"],
            "pa_p95": pa_stats["p95"],
            "background_mean": bg_stats["mean"],
            "background_median": bg_stats["median"],
            "background_p95": bg_stats["p95"],
            "surrounding_mean": ring_stats["mean"],
            "surrounding_median": ring_stats["median"],
            "surrounding_p95": ring_stats["p95"],
            "pa_background_mean_ratio": mean_ratio,
            "frangi_auc": auc,
            "pa_sample_count": pa_stats["count"],
            "background_sample_count": bg_stats["count"],
            "surrounding_sample_count": ring_stats["count"],
        }

        patient_results.append(result)

        pooled_pa.append(pa_values)
        pooled_background.append(bg_values)

        print(f"PA Frangi mean:          {pa_stats['mean']:.6f}")
        print(f"Background Frangi mean:  {bg_stats['mean']:.6f}")
        print(f"Surrounding Frangi mean: {ring_stats['mean']:.6f}")
        print(f"PA/background ratio:     {mean_ratio:.4f}")
        print(f"Frangi-only ROC-AUC:     {auc:.4f}")

    # -----------------------------------------------------
    # Save patient-level CSV
    # -----------------------------------------------------

    if not patient_results:
        raise RuntimeError("No validation patients were successfully analyzed.")

    csv_path = OUTPUT_DIR / "frangi_separability_validation.csv"

    fieldnames = list(patient_results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(patient_results)

    # -----------------------------------------------------
    # Aggregate validation summary
    # -----------------------------------------------------

    auc_values = [
        row["frangi_auc"]
        for row in patient_results
    ]

    ratio_values = [
        row["pa_background_mean_ratio"]
        for row in patient_results
        if np.isfinite(row["pa_background_mean_ratio"])
    ]

    mean_auc = float(np.mean(auc_values))
    mean_ratio = (
        float(np.mean(ratio_values))
        if ratio_values else float("nan")
    )

    pooled_pa_values = np.concatenate(pooled_pa)
    pooled_bg_values = np.concatenate(pooled_background)

    pooled_y = np.concatenate([
        np.ones(pooled_pa_values.size, dtype=np.uint8),
        np.zeros(pooled_bg_values.size, dtype=np.uint8),
    ])

    pooled_scores = np.concatenate([
        pooled_pa_values,
        pooled_bg_values,
    ])

    pooled_auc = float(
        roc_auc_score(pooled_y, pooled_scores)
    )

    summary = {
        "validation_patients_analyzed": len(patient_results),
        "mean_patient_auc": mean_auc,
        "pooled_sample_auc": pooled_auc,
        "mean_pa_background_ratio": mean_ratio,
        "mean_pa_frangi": float(np.mean([
            row["pa_mean"] for row in patient_results
        ])),
        "mean_background_frangi": float(np.mean([
            row["background_mean"] for row in patient_results
        ])),
        "mean_surrounding_frangi": float(np.nanmean([
            row["surrounding_mean"] for row in patient_results
        ])),
        "test_patients_evaluated": 0,
        "interpretation_note": (
            "Frangi-only ROC-AUC measures feature separability, "
            "not segmentation performance."
        ),
    }

    summary_path = OUTPUT_DIR / "frangi_separability_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FRANGI SEPARABILITY SUMMARY")
    print("=" * 65)

    print("Patients analyzed:", len(patient_results))
    print(f"Mean patient ROC-AUC: {mean_auc:.4f}")
    print(f"Pooled sampled ROC-AUC: {pooled_auc:.4f}")
    print(f"Mean PA/background ratio: {mean_ratio:.4f}")

    print(
        "Mean PA Frangi:",
        f"{summary['mean_pa_frangi']:.6f}"
    )

    print(
        "Mean background Frangi:",
        f"{summary['mean_background_frangi']:.6f}"
    )

    print(
        "Mean surrounding Frangi:",
        f"{summary['mean_surrounding_frangi']:.6f}"
    )

    # -----------------------------------------------------
    # Plot patient-level AUC
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    patient_names = [
        row["patient_id"]
        for row in patient_results
    ]

    patient_aucs = [
        row["frangi_auc"]
        for row in patient_results
    ]

    ax.bar(patient_names, patient_aucs)

    ax.axhline(
        0.5,
        linestyle="--",
        label="Random-ranking reference (0.5)"
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Validation patient")
    ax.set_title("Frangi Feature Separability Across Validation Patients")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "frangi_patient_auc.png"

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved files:")
    print(csv_path)
    print(summary_path)
    print(plot_path)

    print("\nTest patients were NOT evaluated.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()
