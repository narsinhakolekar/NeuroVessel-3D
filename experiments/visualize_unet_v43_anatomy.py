"""
NEUROVESSEL V4.3
Raw-versus-connected-component-filtered anatomical comparison.

Validation-only analysis. Does not train the model and does not evaluate
the test set. Uses the V4 checkpoint, probability threshold 0.75, and
minimum connected-component size 5000 voxels, matching V4.2.

Outputs are written to:
outputs/visualizations/v4_3_anatomy/
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v3_frangi import load_case, build_model, predict_volume


# ------------------------- Configuration -------------------------

SPLIT_FILE = ROOT / "data" / "splits.json"
CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints" / "unet_v4_hard_negative_best.pt"
)
OUTPUT_DIR = ROOT / "outputs" / "visualizations" / "v4_3_anatomy"

PATIENT_IDS = ["PA000314", "PA000194", "PA000005"]

PROBABILITY_THRESHOLD = 0.75
MIN_COMPONENT_SIZE = 5000
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity

MAX_3D_POINTS = 15000
RANDOM_SEED = 42


# ------------------------- Split protection -------------------------

def verify_validation_patients():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        split = json.load(f)

    validation_ids = set(split["validation"])
    test_ids = set(split["test"])

    if len(split["validation"]) != 15:
        raise RuntimeError(
            f"Expected 15 validation patients, found {len(split['validation'])}"
        )

    for patient_id in PATIENT_IDS:
        if patient_id not in validation_ids:
            raise RuntimeError(
                f"{patient_id} is not in the validation split."
            )
        if patient_id in test_ids:
            raise RuntimeError(
                f"Test-set patient detected: {patient_id}"
            )

    if validation_ids & test_ids:
        raise RuntimeError("Validation/test split overlap detected.")

    print("Validation split verified.")
    print("Selected validation patients:", PATIENT_IDS)
    print("Test patients will NOT be evaluated.")


# ------------------------- Metrics -------------------------

def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    denominator = 2 * tp + fp + fn
    dice = 2.0 * tp / denominator if denominator else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "predicted_voxels": int(prediction.sum()),
        "ground_truth_voxels": int(target.sum()),
        "predicted_to_gt_ratio": (
            float(prediction.sum() / target.sum())
            if target.sum() else None
        ),
    }


# ------------------------- Component filtering -------------------------

def connected_component_filter(binary_prediction, minimum_size):
    """
    Keep components with at least minimum_size voxels.
    Uses the same 26-connectivity and inclusive >= cutoff as V4.2.
    """
    labels, component_count = ndimage.label(
        binary_prediction,
        structure=CONNECTIVITY
    )

    if component_count == 0:
        return np.zeros(binary_prediction.shape, dtype=bool), {
            "original_component_count": 0,
            "retained_component_count": 0,
            "largest_component_voxels": 0,
        }

    sizes = np.bincount(labels.ravel())[1:]
    keep_labels = np.flatnonzero(sizes >= minimum_size) + 1
    filtered = np.isin(labels, keep_labels)

    component_info = {
        "original_component_count": int(component_count),
        "retained_component_count": int(len(keep_labels)),
        "largest_component_voxels": int(sizes.max()),
    }

    del labels, sizes, keep_labels
    return filtered, component_info


# ------------------------- 2D visualization -------------------------

def extract_slice(volume, axis, index):
    if axis == 0:
        return volume[index, :, :].T
    if axis == 1:
        return volume[:, index, :].T
    return volume[:, :, index].T


def select_best_slice(target, axis):
    reduce_axes = tuple(i for i in range(3) if i != axis)
    counts = target.sum(axis=reduce_axes)
    index = int(np.argmax(counts))
    return index, int(counts[index])


def add_overlay(ax, ct_slice, gt_slice, pred_slice):
    ax.imshow(ct_slice, cmap="gray", origin="lower")
    if gt_slice.any():
        ax.contour(
            gt_slice.astype(np.uint8),
            levels=[0.5],
            colors=["lime"],
            linewidths=0.8,
            origin="lower"
        )
    if pred_slice.any():
        ax.contour(
            pred_slice.astype(np.uint8),
            levels=[0.5],
            colors=["red"],
            linewidths=0.8,
            origin="lower"
        )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_orthogonal_comparison(
    patient_id, image, target, raw_prediction, filtered_prediction,
    output_path
):
    false_positive = filtered_prediction & ~target
    false_negative = target & ~filtered_prediction

    orientations = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    titles = [
        "CT",
        "Ground truth",
        "Raw V4",
        "Filtered V4",
        "Raw overlay\nGreen=GT, Red=prediction",
        "Filtered overlay\nGreen=GT, Red=prediction",
        "Filtered false positives",
        "Filtered false negatives",
    ]

    fig, axes = plt.subplots(
        3, 8, figsize=(27, 12), squeeze=False
    )
    selected_slices = {}

    for row, (orientation, axis) in enumerate(orientations):
        index, gt_count = select_best_slice(target, axis)
        selected_slices[orientation] = {
            "index": index,
            "ground_truth_voxels_on_slice": gt_count,
        }

        ct = extract_slice(image, axis, index)
        gt = extract_slice(target, axis, index).astype(bool)
        raw = extract_slice(raw_prediction, axis, index).astype(bool)
        filtered = extract_slice(
            filtered_prediction, axis, index
        ).astype(bool)
        fp = extract_slice(false_positive, axis, index).astype(bool)
        fn = extract_slice(false_negative, axis, index).astype(bool)

        axes[row, 0].imshow(ct, cmap="gray", origin="lower")
        axes[row, 1].imshow(gt, cmap="gray", origin="lower")
        axes[row, 2].imshow(raw, cmap="gray", origin="lower")
        axes[row, 3].imshow(filtered, cmap="gray", origin="lower")
        add_overlay(axes[row, 4], ct, gt, raw)
        add_overlay(axes[row, 5], ct, gt, filtered)
        axes[row, 6].imshow(fp, cmap="Reds", origin="lower")
        axes[row, 7].imshow(fn, cmap="Blues", origin="lower")

        for col in range(8):
            axes[row, col].set_title(titles[col], fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(
            f"{orientation}\nSlice={index}\nGT voxels={gt_count}",
            fontsize=9
        )

    fig.suptitle(
        f"NeuroVessel V4.3 | {patient_id}\n"
        f"Threshold={PROBABILITY_THRESHOLD}, "
        f"minimum component size={MIN_COMPONENT_SIZE} voxels",
        fontsize=15,
        fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return selected_slices


# ------------------------- 3D point-cloud visualization -------------------------

def sample_points(mask, max_points, rng):
    coordinates = np.argwhere(mask)
    if len(coordinates) > max_points:
        indices = rng.choice(
            len(coordinates), size=max_points, replace=False
        )
        coordinates = coordinates[indices]
    return coordinates


def plot_3d_comparison(
    patient_id, target, raw_prediction, filtered_prediction, output_path
):
    rng = np.random.default_rng(RANDOM_SEED)
    masks = [
        ("Ground truth", target),
        ("Raw V4 prediction", raw_prediction),
        ("Filtered V4 prediction", filtered_prediction),
    ]

    fig = plt.figure(figsize=(18, 6))

    for position, (title, mask) in enumerate(masks, start=1):
        points = sample_points(mask, MAX_3D_POINTS, rng)
        ax = fig.add_subplot(1, 3, position, projection="3d")

        if len(points):
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=1, alpha=0.35
            )

        ax.set_title(f"{title}\n{int(mask.sum()):,} voxels")
        ax.set_xlabel("Voxel axis 0")
        ax.set_ylabel("Voxel axis 1")
        ax.set_zlabel("Voxel axis 2")

    fig.suptitle(
        f"NeuroVessel V4.3 - 3D point-cloud comparison - {patient_id}\n"
        "Coordinates are voxel indices; point clouds are randomly subsampled.",
        fontsize=13,
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Per-patient analysis -------------------------

def analyze_patient(patient_id, model, device):
    print("\n" + "=" * 68)
    print("PATIENT:", patient_id)
    print("=" * 68)

    image, frangi, label = load_case(patient_id)
    target = np.asarray(label) > 0

    probabilities = predict_volume(
        model, image, frangi, device
    )
    probabilities = np.squeeze(probabilities)

    if probabilities.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch for {patient_id}: "
            f"{probabilities.shape} vs {target.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"Non-finite prediction probabilities for {patient_id}"
        )

    raw_prediction = probabilities >= PROBABILITY_THRESHOLD
    filtered_prediction, component_info = connected_component_filter(
        raw_prediction, MIN_COMPONENT_SIZE
    )

    raw_metrics = calculate_metrics(raw_prediction, target)
    filtered_metrics = calculate_metrics(filtered_prediction, target)

    patient_dir = OUTPUT_DIR / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    orthogonal_path = patient_dir / "raw_vs_filtered_orthogonal.png"
    selected_slices = plot_orthogonal_comparison(
        patient_id,
        image,
        target,
        raw_prediction,
        filtered_prediction,
        orthogonal_path
    )

    three_d_path = patient_dir / "raw_vs_filtered_3d.png"
    plot_3d_comparison(
        patient_id,
        target,
        raw_prediction,
        filtered_prediction,
        three_d_path
    )

    result = {
        "patient_id": patient_id,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "minimum_component_size_voxels": MIN_COMPONENT_SIZE,
        "connectivity": 26,
        "raw_metrics": raw_metrics,
        "filtered_metrics": filtered_metrics,
        "component_analysis": component_info,
        "selected_slices": selected_slices,
        "test_set_evaluated": False,
    }

    with (patient_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=4)

    print(
        f"Raw      Dice={raw_metrics['dice']:.4f} | "
        f"Precision={raw_metrics['precision']:.4f} | "
        f"Recall={raw_metrics['recall']:.4f}"
    )
    print(
        f"Filtered Dice={filtered_metrics['dice']:.4f} | "
        f"Precision={filtered_metrics['precision']:.4f} | "
        f"Recall={filtered_metrics['recall']:.4f}"
    )
    print(
        "Components:",
        component_info["original_component_count"],
        "->",
        component_info["retained_component_count"]
    )
    print("Saved visualizations:", patient_dir)

    csv_row = {
        "patient_id": patient_id,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "minimum_component_size_voxels": MIN_COMPONENT_SIZE,
        "raw_dice": raw_metrics["dice"],
        "raw_precision": raw_metrics["precision"],
        "raw_recall": raw_metrics["recall"],
        "raw_fp_voxels": raw_metrics["fp_voxels"],
        "raw_fn_voxels": raw_metrics["fn_voxels"],
        "raw_predicted_voxels": raw_metrics["predicted_voxels"],
        "filtered_dice": filtered_metrics["dice"],
        "filtered_precision": filtered_metrics["precision"],
        "filtered_recall": filtered_metrics["recall"],
        "filtered_fp_voxels": filtered_metrics["fp_voxels"],
        "filtered_fn_voxels": filtered_metrics["fn_voxels"],
        "filtered_predicted_voxels": filtered_metrics["predicted_voxels"],
        "original_components": component_info["original_component_count"],
        "retained_components": component_info["retained_component_count"],
        "largest_component_voxels": component_info["largest_component_voxels"],
    }

    del image, frangi, label, probabilities
    del raw_prediction, filtered_prediction, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, csv_row


# ------------------------- Main -------------------------

def main():
    print("=" * 68)
    print("NEUROVESSEL V4.3 - RAW VS FILTERED ANATOMICAL COMPARISON")
    print("=" * 68)

    verify_validation_patients()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the neurovessel environment."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading V4 checkpoint...")
    model = build_model(device)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Probability threshold:", PROBABILITY_THRESHOLD)
    print("Minimum component size:", MIN_COMPONENT_SIZE)
    print("Test set will NOT be evaluated.")

    results = []
    csv_rows = []

    for patient_id in PATIENT_IDS:
        result, csv_row = analyze_patient(patient_id, model, device)
        results.append(result)
        csv_rows.append(csv_row)

    csv_path = OUTPUT_DIR / "v43_patient_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(csv_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "experiment": "NeuroVessel V4.3 raw versus filtered comparison",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "minimum_component_size_voxels": MIN_COMPONENT_SIZE,
        "connectivity": 26,
        "patients_analyzed": PATIENT_IDS,
        "test_patients_evaluated": 0,
        "test_set_evaluated": False,
        "results": results,
        "interpretation_note": (
            "The 5000-voxel component threshold was selected from prior "
            "validation experiments. This visualization is not an independent "
            "test and does not establish anatomical correctness. 3D plots use "
            "voxel-index coordinates and randomly subsampled point clouds."
        ),
    }

    json_path = OUTPUT_DIR / "v43_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 68)
    print("V4.3 COMPLETED")
    print("=" * 68)
    print("Output directory:", OUTPUT_DIR)
    print("Patient comparison CSV:", csv_path)
    print("Summary JSON:", json_path)
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
