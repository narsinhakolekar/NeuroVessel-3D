
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


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v3_frangi import load_case

SPLIT_FILE = ROOT / "data" / "splits.json"

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics" / "frangi_separability_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 50000
RING_WIDTH = 5
RANDOM_SEED = 42

# Frangi features are expected to be normalized to [0, 1].
HISTOGRAM_BINS = 4096


# =========================================================
# VALIDATION SPLIT
# =========================================================

def get_validation_ids():

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for key in ["validation", "val", "validation_ids", "val_ids"]:
        if key in splits and isinstance(splits[key], list):
            if splits[key]:
                return splits[key]

    raise ValueError(
        f"Validation IDs not found. Available keys: {list(splits.keys())}"
    )


# =========================================================
# UNIFORM SAMPLING ACROSS THE COMPLETE 3D VOLUME
# =========================================================

def sample_uniform_3d(feature, mask, max_samples, rng):
    """
    Uniformly samples voxels from the complete 3D mask.

    Sampling is proportional to the number of eligible voxels
    in each axial slice. This prevents early slices from
    dominating the sample.
    """

    slice_counts = np.array(
        [np.count_nonzero(mask[:, :, z])
         for z in range(mask.shape[2])],
        dtype=np.int64
    )

    total = int(slice_counts.sum())

    if total == 0:
        return np.empty(0, dtype=np.float32)

    n_samples = min(max_samples, total)

    # Allocate samples proportionally to slice populations.
    exact = slice_counts.astype(np.float64) * n_samples / total
    allocation = np.floor(exact).astype(np.int64)

    # Distribute remaining samples according to largest remainders.
    remaining = n_samples - int(allocation.sum())

    if remaining > 0:
        remainders = exact - allocation
        valid = np.flatnonzero(allocation < slice_counts)

        if valid.size:
            order = valid[np.argsort(remainders[valid])[::-1]]
            for z in order[:remaining]:
                allocation[z] += 1

    selected = []

    for z, count in enumerate(allocation):

        if count == 0:
            continue

        slice_mask = mask[:, :, z]
        indices = np.flatnonzero(slice_mask)

        if count < indices.size:
            indices = rng.choice(
                indices,
                size=int(count),
                replace=False
            )

        values = feature[:, :, z].reshape(-1)[indices]
        selected.append(values.astype(np.float32, copy=False))

    if not selected:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(selected)


# =========================================================
# FIND HIGH-FRANGI BACKGROUND THRESHOLD
# =========================================================

def background_frangi_percentile(feature, background, percentile=90):
    """
    Estimates a background Frangi percentile using a histogram.
    This avoids making a large copy of all background values.
    """

    histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)

    for z in range(feature.shape[2]):

        values = feature[:, :, z][background[:, :, z]]

        if values.size == 0:
            continue

        hist, _ = np.histogram(
            values,
            bins=HISTOGRAM_BINS,
            range=(0.0, 1.0)
        )

        histogram += hist

    cumulative = np.cumsum(histogram)
    total = int(cumulative[-1])

    if total == 0:
        return 1.0

    target = int(np.ceil(percentile / 100.0 * total))
    bin_index = int(np.searchsorted(cumulative, target))

    bin_index = min(bin_index, HISTOGRAM_BINS - 1)

    return float((bin_index + 1) / HISTOGRAM_BINS)


# =========================================================
# SURROUNDING REGION
# =========================================================

def make_surrounding_ring(label, width=5):
    """
    Creates a 2D surrounding ring on every axial slice.
    The annotated PA itself is excluded.
    """

    ring = np.zeros(label.shape, dtype=bool)

    for z in range(label.shape[2]):

        gt_slice = label[:, :, z]

        if not gt_slice.any():
            continue

        expanded = binary_dilation(
            gt_slice,
            iterations=width
        )

        ring[:, :, z] = expanded & (~gt_slice)

    return ring


# =========================================================
# ROC-AUC
# =========================================================

def calculate_auc(positive, negative):

    if positive.size == 0 or negative.size == 0:
        return float("nan")

    y_true = np.concatenate([
        np.ones(positive.size, dtype=np.uint8),
        np.zeros(negative.size, dtype=np.uint8)
    ])

    y_score = np.concatenate([positive, negative])

    return float(roc_auc_score(y_true, y_score))


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    print("=" * 70)
    print("NEUROVESSEL - FRANGI SEPARABILITY V2")
    print("=" * 70)

    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Test patients will NOT be evaluated.")
    print("Sampling: uniform across complete 3D volume")

    validation_ids = get_validation_ids()

    print("Validation patients:", len(validation_ids))
    print("Maximum samples per class:", MAX_SAMPLES)

    results = []

    pooled = {
        "pa": [],
        "background": [],
        "ring": [],
        "hard_background": []
    }

    # -----------------------------------------------------
    # Process validation patients
    # -----------------------------------------------------

    for patient_number, patient_id in enumerate(validation_ids, 1):

        print(
            f"\nPatient {patient_number}/{len(validation_ids)}: "
            f"{patient_id}"
        )

        rng = np.random.default_rng(
            RANDOM_SEED + patient_number
        )

        image, frangi, label = load_case(patient_id)

        frangi = np.asarray(frangi, dtype=np.float32)
        label = np.asarray(label) > 0

        if frangi.shape != label.shape:
            raise ValueError(
                f"Shape mismatch in {patient_id}: "
                f"Frangi {frangi.shape}, label {label.shape}"
            )

        if not np.isfinite(frangi).all():
            raise ValueError(
                f"Non-finite Frangi values in {patient_id}"
            )

        if frangi.min() < -1e-5 or frangi.max() > 1.00001:
            raise ValueError(
                f"Frangi outside expected [0,1] range in {patient_id}: "
                f"{frangi.min()}, {frangi.max()}"
            )

        if not label.any():
            print("WARNING: Empty label. Skipping.")
            continue

        background = ~label

        # 1. Positive PA samples.
        pa_values = sample_uniform_3d(
            frangi, label, MAX_SAMPLES, rng
        )

        # 2. Random background samples.
        bg_values = sample_uniform_3d(
            frangi, background, MAX_SAMPLES, rng
        )

        # 3. Nearby surrounding ring samples.
        ring_mask = make_surrounding_ring(
            label, width=RING_WIDTH
        )

        ring_values = sample_uniform_3d(
            frangi, ring_mask, MAX_SAMPLES, rng
        )

        # 4. Hard negatives: background with high Frangi response.
        hard_threshold = background_frangi_percentile(
            frangi,
            background,
            percentile=90
        )

        hard_mask = background & (frangi >= hard_threshold)

        hard_values = sample_uniform_3d(
            frangi,
            hard_mask,
            MAX_SAMPLES,
            rng
        )

        if min(
            pa_values.size,
            bg_values.size,
            ring_values.size,
            hard_values.size
        ) == 0:
            print("WARNING: One or more classes empty. Skipping.")
            continue

        # Separate AUC measurements.
        auc_random = calculate_auc(pa_values, bg_values)
        auc_ring = calculate_auc(pa_values, ring_values)
        auc_hard = calculate_auc(pa_values, hard_values)

        row = {
            "patient_id": patient_id,
            "pa_mean": float(np.mean(pa_values)),
            "pa_median": float(np.median(pa_values)),
            "pa_p95": float(np.percentile(pa_values, 95)),

            "background_mean": float(np.mean(bg_values)),
            "background_median": float(np.median(bg_values)),
            "background_p95": float(np.percentile(bg_values, 95)),

            "ring_mean": float(np.mean(ring_values)),
            "ring_median": float(np.median(ring_values)),
            "ring_p95": float(np.percentile(ring_values, 95)),

            "hard_background_mean": float(np.mean(hard_values)),
            "hard_background_median": float(np.median(hard_values)),

            "hard_background_threshold": hard_threshold,

            "auc_pa_vs_random_background": auc_random,
            "auc_pa_vs_surrounding_ring": auc_ring,
            "auc_pa_vs_hard_background": auc_hard,

            "pa_samples": int(pa_values.size),
            "background_samples": int(bg_values.size),
            "ring_samples": int(ring_values.size),
            "hard_background_samples": int(hard_values.size)
        }

        results.append(row)

        pooled["pa"].append(pa_values)
        pooled["background"].append(bg_values)
        pooled["ring"].append(ring_values)
        pooled["hard_background"].append(hard_values)

        print(f"PA mean:                {row['pa_mean']:.6f}")
        print(f"Background mean:        {row['background_mean']:.6f}")
        print(f"Surrounding mean:       {row['ring_mean']:.6f}")
        print(f"Hard background mean:   {row['hard_background_mean']:.6f}")
        print(f"Hard negative cutoff:   {hard_threshold:.6f}")
        print(f"AUC vs random BG:       {auc_random:.4f}")
        print(f"AUC vs surrounding:     {auc_ring:.4f}")
        print(f"AUC vs hard BG:         {auc_hard:.4f}")

    if not results:
        raise RuntimeError("No patients were successfully analyzed.")

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = OUTPUT_DIR / "frangi_separability_v2_validation.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Patient-level summary
    # -----------------------------------------------------

    metric_names = [
        "auc_pa_vs_random_background",
        "auc_pa_vs_surrounding_ring",
        "auc_pa_vs_hard_background"
    ]

    summary = {
        "validation_patients_analyzed": len(results),
        "test_patients_evaluated": 0,
        "sampling_method": "Uniform across complete 3D volume",
        "samples_per_class_maximum": MAX_SAMPLES,
        "ring_width_pixels": RING_WIDTH
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for metric in metric_names:

        values = np.array(
            [r[metric] for r in results],
            dtype=np.float64
        )

        values = values[np.isfinite(values)]

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

        summary[metric] = {
            "mean": mean,
            "std": std,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values))
        }

        print(
            f"{metric}: mean={mean:.4f}, "
            f"std={std:.4f}, "
            f"min={np.min(values):.4f}, "
            f"max={np.max(values):.4f}"
        )

    # Patient-level bootstrap confidence intervals.
    rng = np.random.default_rng(RANDOM_SEED)

    for metric in metric_names:

        values = np.array(
            [r[metric] for r in results],
            dtype=np.float64
        )

        values = values[np.isfinite(values)]

        if values.size > 1:

            boot_means = np.array([
                np.mean(rng.choice(values, size=values.size, replace=True))
                for _ in range(5000)
            ])

            ci_low, ci_high = np.percentile(
                boot_means, [2.5, 97.5]
            )

            summary[metric]["bootstrap_95ci"] = [
                float(ci_low),
                float(ci_high)
            ]

            print(
                f"  Patient bootstrap 95% CI: "
                f"[{ci_low:.4f}, {ci_high:.4f}]"
            )

    # -----------------------------------------------------
    # Pooled sampled AUCs
    # -----------------------------------------------------

    pooled_auc = {}

    for comparison, negative_key in [
        ("pa_vs_random_background", "background"),
        ("pa_vs_surrounding_ring", "ring"),
        ("pa_vs_hard_background", "hard_background")
    ]:

        positive = np.concatenate(pooled["pa"])
        negative = np.concatenate(pooled[negative_key])

        auc = calculate_auc(positive, negative)

        pooled_auc[comparison] = auc

        print(
            f"Pooled sampled AUC {comparison}: {auc:.4f}"
        )

    summary["pooled_sample_auc"] = pooled_auc

    # -----------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------

    json_path = OUTPUT_DIR / "frangi_separability_v2_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # Plot three patient-level AUC comparisons
    # -----------------------------------------------------

    patient_names = [r["patient_id"] for r in results]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 13),
        sharex=True
    )

    plot_configs = [
        (
            "auc_pa_vs_random_background",
            "PA vs Random Background"
        ),
        (
            "auc_pa_vs_surrounding_ring",
            "PA vs Surrounding Ring"
        ),
        (
            "auc_pa_vs_hard_background",
            "PA vs High-Frangi Hard Background"
        )
    ]

    for ax, (key, title) in zip(axes, plot_configs):

        values = [r[key] for r in results]

        ax.bar(patient_names, values)
        ax.axhline(
            0.5,
            linestyle="--",
            label="Random-ranking reference"
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.legend()

    axes[-1].set_xlabel("Validation patient")
    axes[-1].tick_params(axis="x", rotation=45)

    fig.suptitle(
        "NeuroVessel: Corrected Frangi Separability Analysis",
        fontsize=16
    )

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "frangi_separability_v2.png"

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved results:")
    print(csv_path)
    print(json_path)
    print(plot_path)

    print("\nTest patients were NOT evaluated.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()
