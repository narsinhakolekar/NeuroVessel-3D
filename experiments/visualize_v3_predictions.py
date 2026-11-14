
"""
NeuroVessel V3 - Pulmonary Artery Prediction Visualization

Uses:
    - V3 epoch-4 best checkpoint
    - Validation patient PA000005
    - Original CT volume
    - Frangi vesselness features
    - Ground-truth pulmonary artery segmentation

Generates:
    1. Original CT
    2. Ground-truth segmentation
    3. Prediction at threshold 0.5
    4. Prediction at threshold 0.9
    5. Ground-truth / prediction overlay
    6. Probability heatmap

No training is performed.
No test patients are accessed.
"""

# ============================================================
# 1. IMPORTS AND PROJECT PATH
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib

# Use a non-interactive backend to save images reliably on Windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Import functions from the V3 training script.
# Both Python files are inside the experiments folder.
from train_unet_v3_frangi import (
    BEST_PATH,
    build_model,
    load_case,
    predict_volume,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PATIENT_ID = "PA000005"

THRESHOLD_LOW = 0.50
THRESHOLD_HIGH = 0.90

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / f"v3_{PATIENT_ID}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. LOAD V3 MODEL
# ============================================================

def load_v3_model():

    print("\nLoading V3 checkpoint...")

    checkpoint_path = Path(BEST_PATH)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V3 checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "Unknown")
    )

    return model


# ============================================================
# 4. CONVERT MODEL OUTPUT TO NUMPY
# ============================================================

def to_numpy(array):

    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    return np.asarray(array)


# ============================================================
# 5. SAVE INDIVIDUAL IMAGE
# ============================================================

def save_image(array, filename, title, cmap="gray",
               show_colorbar=False):

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        array.T,
        cmap=cmap,
        origin="lower"
    )

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if show_colorbar:
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Probability"
        )

    fig.tight_layout()

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path.name)


# ============================================================
# 6. MAIN VISUALIZATION
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V3 - PREDICTION VISUALIZATION")
    print("=" * 65)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_v3_model()

    # --------------------------------------------------------
    # Load validation patient
    # --------------------------------------------------------

    print("\nLoading validation patient:", PATIENT_ID)

    image, frangi, label = load_case(PATIENT_ID)

    image = to_numpy(image).astype(np.float32)
    frangi = to_numpy(frangi).astype(np.float32)
    label = to_numpy(label) > 0

    print("\nLoaded data:")
    print("CT shape:", image.shape)
    print("Frangi shape:", frangi.shape)
    print("Label shape:", label.shape)

    if image.shape != frangi.shape:
        raise ValueError("CT and Frangi shapes do not match.")

    if image.shape != label.shape:
        raise ValueError("CT and label shapes do not match.")

    if not np.isfinite(image).all():
        raise ValueError("CT contains NaN or infinite values.")

    if not np.isfinite(frangi).all():
        raise ValueError("Frangi contains NaN or infinite values.")

    if not label.any():
        raise ValueError("Ground-truth label is empty.")

    # --------------------------------------------------------
    # Full-volume prediction
    # --------------------------------------------------------

    print("\nRunning V3 full-volume inference...")
    print("This may take several minutes.")

    probabilities = predict_volume(
        model,
        image,
        frangi,
        DEVICE
    )

    probabilities = to_numpy(probabilities).astype(np.float32)

    if probabilities.shape != label.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} "
            f"does not match label shape {label.shape}"
        )

    print("\nInference completed.")

    print(
        "Probability minimum:",
        float(probabilities.min())
    )

    print(
        "Probability maximum:",
        float(probabilities.max())
    )

    print(
        "Probability mean:",
        float(probabilities.mean())
    )

    # --------------------------------------------------------
    # Select representative axial slice
    # --------------------------------------------------------

    # Find the axial slice containing the largest
    # ground-truth pulmonary artery area.

    slice_areas = label.sum(axis=(0, 1))

    slice_index = int(np.argmax(slice_areas))

    if slice_areas[slice_index] == 0:
        raise ValueError("No foreground slice found.")

    ct_slice = image[:, :, slice_index]
    gt_slice = label[:, :, slice_index]
    probability_slice = probabilities[:, :, slice_index]

    print("\nSelected axial slice:", slice_index)

    print(
        "Ground-truth foreground pixels:",
        int(gt_slice.sum())
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    prediction_05 = probability_slice >= THRESHOLD_LOW
    prediction_09 = probability_slice >= THRESHOLD_HIGH

    print(
        "Predicted pixels at threshold 0.5:",
        int(prediction_05.sum())
    )

    print(
        "Predicted pixels at threshold 0.9:",
        int(prediction_09.sum())
    )

    # --------------------------------------------------------
    # Save six-panel comparison
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 12)
    )

    # Panel 1: Original CT
    axes[0, 0].imshow(
        ct_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[0, 0].set_title("1. Original CT")

    # Panel 2: Ground truth
    axes[0, 1].imshow(
        gt_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[0, 1].set_title("2. Ground Truth PA")

    # Panel 3: Prediction at 0.5
    axes[0, 2].imshow(
        prediction_05.T,
        cmap="gray",
        origin="lower"
    )

    axes[0, 2].set_title("3. Prediction - Threshold 0.5")

    # Panel 4: Prediction at 0.9
    axes[1, 0].imshow(
        prediction_09.T,
        cmap="gray",
        origin="lower"
    )

    axes[1, 0].set_title("4. Prediction - Threshold 0.9")

    # Panel 5: Overlay
    axes[1, 1].imshow(
        ct_slice.T,
        cmap="gray",
        origin="lower"
    )

    overlay = np.zeros(
        (*gt_slice.T.shape, 4),
        dtype=np.float32
    )

    gt_transposed = gt_slice.T
    pred_transposed = prediction_09.T

    # Ground truth: green
    overlay[gt_transposed] = [
        0.0, 1.0, 0.0, 0.65
    ]

    # Prediction: red
    overlay[pred_transposed] = [
        1.0, 0.0, 0.0, 0.55
    ]

    # Correctly overlapping pixels: yellow
    intersection = gt_transposed & pred_transposed

    overlay[intersection] = [
        1.0, 1.0, 0.0, 0.85
    ]

    axes[1, 1].imshow(
        overlay,
        origin="lower"
    )

    axes[1, 1].set_title(
        "5. Overlay: GT Green / Prediction Red"
    )

    # Panel 6: Probability heatmap
    heatmap = axes[1, 2].imshow(
        probability_slice.T,
        cmap="hot",
        vmin=0,
        vmax=1,
        origin="lower"
    )

    axes[1, 2].set_title(
        "6. Probability Heatmap"
    )

    fig.colorbar(
        heatmap,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.04,
        label="Predicted probability"
    )

    # Remove axes
    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"NeuroVessel V3 | {PATIENT_ID} | "
        f"Axial Slice {slice_index}",
        fontsize=17
    )

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR / "v3_six_panel_comparison.png"
    )

    fig.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSix-panel comparison saved.")

    # --------------------------------------------------------
    # Save individual images
    # --------------------------------------------------------

    save_image(
        ct_slice,
        "original_ct.png",
        "Original CT"
    )

    save_image(
        gt_slice,
        "ground_truth.png",
        "Ground Truth PA"
    )

    save_image(
        prediction_05,
        "prediction_threshold_05.png",
        "V3 Prediction Threshold 0.5"
    )

    save_image(
        prediction_09,
        "prediction_threshold_09.png",
        "V3 Prediction Threshold 0.9"
    )

    save_image(
        probability_slice,
        "probability_heatmap.png",
        "V3 Probability Heatmap",
        cmap="hot",
        show_colorbar=True
    )

    # --------------------------------------------------------
    # Save complete 3D prediction arrays
    # --------------------------------------------------------

    print("\nSaving 3D prediction arrays...")

    np.save(
        OUTPUT_DIR / "probability_volume.npy",
        probabilities.astype(np.float16)
    )

    np.save(
        OUTPUT_DIR / "prediction_threshold_05.npy",
        (probabilities >= THRESHOLD_LOW).astype(np.uint8)
    )

    np.save(
        OUTPUT_DIR / "prediction_threshold_09.npy",
        (probabilities >= THRESHOLD_HIGH).astype(np.uint8)
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nPatient:", PATIENT_ID)
    print("Selected slice:", slice_index)

    print("\nGenerated files:")

    for file_path in sorted(OUTPUT_DIR.iterdir()):

        print(" -", file_path.name)

    print("\nMain visualization:")
    print(comparison_path.resolve())

    print("\nNo model training was performed.")
    print("No test patients were evaluated.")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
