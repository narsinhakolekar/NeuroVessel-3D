"""
NeuroVessel V4 - Qualitative Error Analysis

Visualizes:
1. Original CT image
2. Ground-truth pulmonary artery mask
3. V4 predicted mask
4. Ground truth + prediction overlay
5. False-positive regions
6. False-negative regions

Uses validation patients only.
Does NOT evaluate the test set.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Import existing V3 helper functions.
from train_unet_v3_frangi import (
    load_case,
    build_model,
    predict_volume,
    calculate_metrics,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = ROOT / "data" / "splits.json"

CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "unet_v4_hard_negative_best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "visualizations"
    / "v4_error_analysis"
)

# Three representative validation patients.
PATIENT_IDS = [
    "PA000314",
    "PA000194",
    "PA000005",
]

# Best threshold found during V4 validation threshold sweep.
THRESHOLD = 0.75

# Number of axial slices per patient.
SLICES_PER_PATIENT = 3


# ============================================================
# SELECT REPRESENTATIVE AXIAL SLICES
# ============================================================

def choose_slices(label, number=3):

    # Count foreground voxels in every axial slice.
    slice_counts = np.asarray(
        label,
        dtype=np.uint8
    ).sum(axis=(0, 1))

    # Find slices containing annotated foreground.
    positive_slices = np.flatnonzero(
        slice_counts > 0
    )

    if positive_slices.size == 0:
        raise ValueError(
            "Ground-truth label contains no foreground voxels."
        )

    # Select approximately equally spaced slices.
    positions = np.linspace(
        0,
        positive_slices.size - 1,
        num=min(number, positive_slices.size)
    )

    selected = [
        int(positive_slices[int(round(pos))])
        for pos in positions
    ]

    # Include the slice containing the largest annotation.
    max_area_slice = int(np.argmax(slice_counts))

    if max_area_slice not in selected:

        if len(selected) >= number:
            selected[-1] = max_area_slice

        else:
            selected.append(max_area_slice)

    return sorted(set(selected))


# ============================================================
# GENERATE VISUALIZATION
# ============================================================

def visualize_patient(
    patient_id,
    image,
    label,
    probabilities,
    output_path
):

    # Convert probabilities into binary prediction.
    prediction = probabilities >= THRESHOLD

    # Select representative slices.
    slices = choose_slices(
        label,
        SLICES_PER_PATIENT
    )

    fig, axes = plt.subplots(
        len(slices),
        6,
        figsize=(22, 4.5 * len(slices)),
        squeeze=False
    )

    titles = [
        "Original CT",
        "Ground Truth",
        "V4 Prediction",
        "GT + Prediction Overlay",
        "False Positives",
        "False Negatives",
    ]

    for row, z in enumerate(slices):

        # Extract axial slices.
        ct = image[:, :, z].T

        gt = label[:, :, z].T.astype(bool)

        pred = prediction[:, :, z].T.astype(bool)

        # Error masks.
        false_positive = pred & ~gt

        false_negative = gt & ~pred

        # ----------------------------------------------------
        # 1. Original CT
        # ----------------------------------------------------

        axes[row, 0].imshow(
            ct,
            cmap="gray",
            origin="lower"
        )

        axes[row, 0].set_title(
            titles[0]
        )

        # ----------------------------------------------------
        # 2. Ground Truth
        # ----------------------------------------------------

        axes[row, 1].imshow(
            gt,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1
        )

        axes[row, 1].set_title(
            titles[1]
        )

        # ----------------------------------------------------
        # 3. V4 Prediction
        # ----------------------------------------------------

        axes[row, 2].imshow(
            pred,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1
        )

        axes[row, 2].set_title(
            f"{titles[2]} (threshold={THRESHOLD})"
        )

        # ----------------------------------------------------
        # 4. Overlay
        # ----------------------------------------------------

        axes[row, 3].imshow(
            ct,
            cmap="gray",
            origin="lower"
        )

        # Green contour = Ground truth.
        if gt.any():

            axes[row, 3].contour(
                gt.astype(np.uint8),
                levels=[0.5],
                colors=["lime"],
                linewidths=0.8,
                origin="lower"
            )

        # Red contour = Prediction.
        if pred.any():

            axes[row, 3].contour(
                pred.astype(np.uint8),
                levels=[0.5],
                colors=["red"],
                linewidths=0.8,
                origin="lower"
            )

        axes[row, 3].set_title(
            "Overlay\nGreen=GT, Red=Prediction"
        )

        # ----------------------------------------------------
        # 5. False Positives
        # ----------------------------------------------------

        axes[row, 4].imshow(
            false_positive,
            cmap="Reds",
            origin="lower",
            vmin=0,
            vmax=1
        )

        axes[row, 4].set_title(
            titles[4]
        )

        # ----------------------------------------------------
        # 6. False Negatives
        # ----------------------------------------------------

        axes[row, 5].imshow(
            false_negative,
            cmap="Blues",
            origin="lower",
            vmin=0,
            vmax=1
        )

        axes[row, 5].set_title(
            titles[5]
        )

        # Common formatting.
        for ax in axes[row]:

            ax.set_xlabel(
                f"Axial slice z={z}"
            )

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"NeuroVessel V4 - {patient_id} "
        f"- Threshold {THRESHOLD}",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return slices


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V4 - ERROR ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------------
    # Check CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable. Activate neurovessel environment."
        )

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    # --------------------------------------------------------
    # Load patient splits
    # --------------------------------------------------------

    split = json.loads(
        SPLIT_FILE.read_text(
            encoding="utf-8"
        )
    )

    validation_ids = set(
        split["validation"]
    )

    test_ids = set(
        split["test"]
    )

    # Safety check: only validation patients.
    for patient_id in PATIENT_IDS:

        if patient_id not in validation_ids:

            raise ValueError(
                f"{patient_id} is not a validation patient."
            )

        if patient_id in test_ids:

            raise RuntimeError(
                f"Test patient detected: {patient_id}"
            )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading V4 model...")

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

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    metric_rows = []

    for index, patient_id in enumerate(
        PATIENT_IDS,
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"[{index}/{len(PATIENT_IDS)}] "
            f"Processing {patient_id}"
        )

        # Load CT, Frangi and ground-truth label.
        image, frangi, label = load_case(
            patient_id
        )

        print(
            "CT shape:",
            image.shape
        )

        print(
            "Frangi shape:",
            frangi.shape
        )

        print(
            "Label shape:",
            label.shape
        )

        # Full-volume inference.
        probabilities = predict_volume(
            model,
            image,
            frangi,
            device
        )

        # Binary prediction.
        prediction = (
            probabilities >= THRESHOLD
        )

        # Calculate metrics.
        metrics = calculate_metrics(
            prediction,
            label
        )

        print("\nMetrics:")

        print(
            f"Dice:      {metrics['dice']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"Predicted foreground: "
            f"{metrics['predicted_fraction']:.4%}"
        )

        # Save visualization.
        output_path = (
            OUTPUT_DIR
            / f"{patient_id}_v4_error_analysis.png"
        )

        selected_slices = visualize_patient(
            patient_id,
            image,
            label,
            probabilities,
            output_path
        )

        print(
            "\nSelected axial slices:",
            selected_slices
        )

        print(
            "Visualization saved:",
            output_path
        )

        # Save patient metrics.
        metric_rows.append({

            "patient_id": patient_id,

            "threshold": THRESHOLD,

            "dice": metrics["dice"],

            "precision": metrics["precision"],

            "recall": metrics["recall"],

            "predicted_fraction": metrics[
                "predicted_fraction"
            ],

            "selected_slices": ";".join(
                map(str, selected_slices)
            ),

            "visualization": output_path.name

        })

        # Free memory.
        del image
        del frangi
        del label
        del probabilities
        del prediction

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "v4_selected_patient_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metric_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            metric_rows
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "V4 QUALITATIVE ERROR ANALYSIS COMPLETE"
    )

    print("=" * 65)

    print(
        "\nOutput folder:",
        OUTPUT_DIR
    )

    print(
        "\nMetrics CSV:",
        csv_path
    )

    print(
        "\nOnly validation patients were processed."
    )

    print(
        "The test set was NOT evaluated."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()