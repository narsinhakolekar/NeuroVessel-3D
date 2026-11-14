"""
NEUROVESSEL V4.1
3D Anatomical Error Analysis

Features:
1. Axial, coronal and sagittal comparisons.
2. Ground-truth-aware slice selection.
3. 3D ground-truth versus prediction point-cloud visualization.
4. Connected-component analysis.
5. False-positive and false-negative measurements.
6. Patient-wise CSV and JSON reports.

Uses validation patients only.
Does not train the model.
Does not evaluate the test set.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import ndimage

# Reuse the existing V3 inference functions.
from train_unet_v3_frangi import (
    load_case,
    build_model,
    predict_volume,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = ROOT / "data" / "splits.json"

CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints"
    / "unet_v4_hard_negative_best.pt"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "visualizations"
    / "v4_3d_analysis"
)

PATIENT_IDS = [
    "PA000314",
    "PA000194",
    "PA000005",
]

THRESHOLD = 0.75

# Select slices containing substantial ground-truth area.
MIN_SLICE_FRACTION = 0.10

# Maximum number of points shown in each 3D point cloud.
MAX_3D_POINTS = 25000

# Random seed for reproducible point sampling.
SEED = 42

# 26-connectivity: includes face, edge and corner neighbours.
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


# ============================================================
# 2. VALIDATION SPLIT CHECK
# ============================================================

def verify_validation_patients():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    split = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    validation_ids = set(split["validation"])
    test_ids = set(split["test"])

    for patient_id in PATIENT_IDS:

        if patient_id not in validation_ids:
            raise ValueError(
                f"{patient_id} is not in validation split."
            )

        if patient_id in test_ids:
            raise RuntimeError(
                f"Test patient detected: {patient_id}"
            )

    print("Validation split verified.")
    print("Test patients will NOT be evaluated.")

    return split


# ============================================================
# 3. SELECT BEST SLICE
# ============================================================

def select_best_slice(label, axis):

    """
    Select the slice with the largest ground-truth area
    along the requested axis.

    axis=0: sagittal
    axis=1: coronal
    axis=2: axial
    """

    counts = label.sum(axis=tuple(
        i for i in range(3) if i != axis
    ))

    best_index = int(np.argmax(counts))

    return best_index, int(counts[best_index])


# ============================================================
# 4. GENERATE 2D ORTHOGONAL COMPARISONS
# ============================================================

def extract_slice(volume, axis, index):

    if axis == 0:
        return volume[index, :, :].T

    elif axis == 1:
        return volume[:, index, :].T

    else:
        return volume[:, :, index].T


def plot_orthogonal_analysis(
    patient_id,
    image,
    ground_truth,
    prediction,
    output_path
):

    false_positive = prediction & ~ground_truth
    false_negative = ground_truth & ~prediction

    orientations = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(23, 13),
        squeeze=False
    )

    titles = [
        "CT Image",
        "Ground Truth",
        "V4 Prediction",
        "Overlay",
        "False Positives",
        "False Negatives",
    ]

    selected_indices = {}

    for row, (orientation, axis) in enumerate(orientations):

        slice_index, foreground_count = select_best_slice(
            ground_truth,
            axis
        )

        selected_indices[orientation] = slice_index

        ct_slice = extract_slice(
            image,
            axis,
            slice_index
        )

        gt_slice = extract_slice(
            ground_truth,
            axis,
            slice_index
        ).astype(bool)

        pred_slice = extract_slice(
            prediction,
            axis,
            slice_index
        ).astype(bool)

        fp_slice = extract_slice(
            false_positive,
            axis,
            slice_index
        ).astype(bool)

        fn_slice = extract_slice(
            false_negative,
            axis,
            slice_index
        ).astype(bool)

        # CT image
        axes[row, 0].imshow(
            ct_slice,
            cmap="gray",
            origin="lower"
        )

        # Ground truth
        axes[row, 1].imshow(
            gt_slice,
            cmap="gray",
            origin="lower"
        )

        # Prediction
        axes[row, 2].imshow(
            pred_slice,
            cmap="gray",
            origin="lower"
        )

        # Overlay
        axes[row, 3].imshow(
            ct_slice,
            cmap="gray",
            origin="lower"
        )

        if gt_slice.any():
            axes[row, 3].contour(
                gt_slice.astype(np.uint8),
                levels=[0.5],
                colors=["lime"],
                linewidths=0.8,
                origin="lower"
            )

        if pred_slice.any():
            axes[row, 3].contour(
                pred_slice.astype(np.uint8),
                levels=[0.5],
                colors=["red"],
                linewidths=0.8,
                origin="lower"
            )

        # False positives
        axes[row, 4].imshow(
            fp_slice,
            cmap="Reds",
            origin="lower"
        )

        # False negatives
        axes[row, 5].imshow(
            fn_slice,
            cmap="Blues",
            origin="lower"
        )

        for col in range(6):

            axes[row, col].set_title(
                titles[col]
                if col != 3
                else "Overlay: Green=GT, Red=Prediction"
            )

            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(
            f"{orientation}\nIndex={slice_index}\n"
            f"GT voxels={foreground_count}",
            fontsize=10
        )

    fig.suptitle(
        f"NeuroVessel V4.1 - {patient_id}\n"
        f"Threshold = {THRESHOLD}",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return selected_indices


# ============================================================
# 5. CONNECTED COMPONENT ANALYSIS
# ============================================================

def connected_component_analysis(mask, name):

    """
    Analyze connected components inside the foreground
    bounding box to reduce memory usage.
    """

    coordinates = np.where(mask)

    if len(coordinates[0]) == 0:

        return {
            "component_count": 0,
            "largest_component_voxels": 0,
            "top_5_component_sizes": [],
        }

    slices = []

    for axis_coordinates in coordinates:

        start = max(
            0,
            int(axis_coordinates.min()) - 1
        )

        end = min(
            mask.shape[len(slices)],
            int(axis_coordinates.max()) + 2
        )

        slices.append(slice(start, end))

    cropped_mask = mask[tuple(slices)]

    labels, component_count = ndimage.label(
        cropped_mask,
        structure=CONNECTIVITY
    )

    component_sizes = np.bincount(
        labels.ravel()
    )[1:]

    component_sizes = np.sort(
        component_sizes
    )[::-1]

    largest = (
        int(component_sizes[0])
        if component_sizes.size
        else 0
    )

    top_sizes = [
        int(x)
        for x in component_sizes[:5]
    ]

    del labels, cropped_mask

    return {
        "component_count": int(component_count),
        "largest_component_voxels": largest,
        "top_5_component_sizes": top_sizes,
    }


# ============================================================
# 6. 3D POINT CLOUD VISUALIZATION
# ============================================================

def sample_points(mask, max_points, rng):

    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return coordinates

    if len(coordinates) > max_points:

        indices = rng.choice(
            len(coordinates),
            size=max_points,
            replace=False
        )

        coordinates = coordinates[indices]

    return coordinates


def plot_3d_comparison(
    patient_id,
    ground_truth,
    prediction,
    output_path
):

    rng = np.random.default_rng(SEED)

    gt_points = sample_points(
        ground_truth,
        MAX_3D_POINTS,
        rng
    )

    pred_points = sample_points(
        prediction,
        MAX_3D_POINTS,
        rng
    )

    fig = plt.figure(
        figsize=(17, 8)
    )

    # Ground-truth 3D point cloud.
    ax1 = fig.add_subplot(
        1, 2, 1,
        projection="3d"
    )

    if len(gt_points):

        ax1.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            s=1,
            alpha=0.35
        )

    ax1.set_title(
        "Ground Truth - 3D"
    )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Prediction 3D point cloud.
    ax2 = fig.add_subplot(
        1, 2, 2,
        projection="3d"
    )

    if len(pred_points):

        ax2.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            pred_points[:, 2],
            s=1,
            alpha=0.35
        )

    ax2.set_title(
        f"V4 Prediction - 3D (Threshold={THRESHOLD})"
    )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    fig.suptitle(
        f"NeuroVessel V4.1 - 3D Comparison - {patient_id}",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 7. METRICS AND ERROR ANALYSIS
# ============================================================

def calculate_error_metrics(
    ground_truth,
    prediction
):

    gt_count = int(ground_truth.sum())
    pred_count = int(prediction.sum())

    true_positive = int(
        np.logical_and(
            ground_truth,
            prediction
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            ~ground_truth,
            prediction
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            ground_truth,
            ~prediction
        ).sum()
    )

    dice = (
        2.0 * true_positive
        / (2 * true_positive + false_positive + false_negative)
        if (2 * true_positive + false_positive + false_negative)
        else 1.0
    )

    precision = (
        true_positive / pred_count
        if pred_count
        else 0.0
    )

    recall = (
        true_positive / gt_count
        if gt_count
        else 0.0
    )

    return {
        "ground_truth_voxels": gt_count,
        "predicted_voxels": pred_count,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_to_gt_ratio": (
            float(pred_count / gt_count)
            if gt_count else None
        ),
    }


# ============================================================
# 8. PROCESS ONE PATIENT
# ============================================================

def analyze_patient(
    patient_id,
    model,
    device
):

    print("\n" + "=" * 65)
    print("PATIENT:", patient_id)
    print("=" * 65)

    # Load CT, Frangi and ground truth.
    image, frangi, label = load_case(
        patient_id
    )

    print("Volume shape:", image.shape)

    ground_truth = np.asarray(
        label,
        dtype=bool
    )

    # V4 full-volume inference.
    probabilities = predict_volume(
        model,
        image,
        frangi,
        device
    )

    prediction = (
        probabilities >= THRESHOLD
    )

    # Error metrics.
    metrics = calculate_error_metrics(
        ground_truth,
        prediction
    )

    # Connected-component analysis.
    print("Analyzing connected components...")

    gt_components = connected_component_analysis(
        ground_truth,
        "Ground Truth"
    )

    pred_components = connected_component_analysis(
        prediction,
        "Prediction"
    )

    # Save outputs.
    patient_dir = OUTPUT_DIR / patient_id

    patient_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Orthogonal visualizations.
    orthogonal_path = (
        patient_dir / "orthogonal_analysis.png"
    )

    selected_slices = plot_orthogonal_analysis(
        patient_id,
        image,
        ground_truth,
        prediction,
        orthogonal_path
    )

    # 3D comparison.
    three_d_path = (
        patient_dir / "3d_comparison.png"
    )

    plot_3d_comparison(
        patient_id,
        ground_truth,
        prediction,
        three_d_path
    )

    result = {
        "patient_id": patient_id,
        "threshold": THRESHOLD,
        "metrics": metrics,
        "ground_truth_components": gt_components,
        "prediction_components": pred_components,
        "selected_slices": selected_slices,
        "test_set_evaluated": False,
    }

    json_path = (
        patient_dir / "analysis_summary.json"
    )

    json_path.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print("\nSegmentation results:")

    print(
        f"Dice: {metrics['dice']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        "Ground-truth voxels:",
        metrics["ground_truth_voxels"]
    )

    print(
        "Predicted voxels:",
        metrics["predicted_voxels"]
    )

    print(
        "False positives:",
        metrics["false_positive_voxels"]
    )

    print(
        "False negatives:",
        metrics["false_negative_voxels"]
    )

    print("\nConnected components:")

    print(
        "GT component count:",
        gt_components["component_count"]
    )

    print(
        "Prediction component count:",
        pred_components["component_count"]
    )

    print(
        "\nSaved:",
        patient_dir
    )

    del image
    del frangi
    del label
    del probabilities
    del prediction
    del ground_truth

    torch.cuda.empty_cache()

    return {
        "patient_id": patient_id,
        **metrics,
        "gt_component_count": gt_components["component_count"],
        "pred_component_count": pred_components["component_count"],
        "gt_largest_component": gt_components[
            "largest_component_voxels"
        ],
        "pred_largest_component": pred_components[
            "largest_component_voxels"
        ],
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V4.1")
    print("3D ANATOMICAL ERROR ANALYSIS")
    print("=" * 65)

    # Verify split.
    verify_validation_patients()

    # CUDA check.
    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable. Activate neurovessel environment."
        )

    device = torch.device("cuda")

    print("\nGPU:", torch.cuda.get_device_name(0))

    # Check checkpoint.
    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            CHECKPOINT_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load V4 model.
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

    print(
        "Prediction threshold:",
        THRESHOLD
    )

    print(
        "Test patients will NOT be evaluated."
    )

    # Analyze selected validation patients.
    all_results = []

    for patient_id in PATIENT_IDS:

        result = analyze_patient(
            patient_id,
            model,
            device
        )

        all_results.append(result)

    # Save overall CSV.
    csv_path = (
        OUTPUT_DIR / "v4_3d_analysis_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Save overall JSON.
    summary_path = (
        OUTPUT_DIR / "v4_3d_analysis_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "model": "NeuroVessel V4",
                "checkpoint_epoch": checkpoint.get("epoch"),
                "threshold": THRESHOLD,
                "patients_analyzed": PATIENT_IDS,
                "test_set_evaluated": False,
                "results": all_results,
            },
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("V4.1 ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nSummary CSV:")
    print(csv_path)

    print("\nSummary JSON:")
    print(summary_path)

    print("\nOnly validation patients were analyzed.")
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()