"""
NeuroVessel V4 - Improved Spatial Error Visualization

Six panels:
1. Original CT
2. Ground Truth Overlay
3. V4 Prediction Overlay
4. TP / FP / FN Error Map
5. Frangi Response
6. Enlarged Error Crop

Validation patients only.
No training. No test-set access.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom


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
    ROOT / "outputs" / "visualizations"
    / "v4_spatial_error_analysis_v2"
)

PATIENT_IDS = [
    "PA000314",
    "PA000194",
    "PA000005",
]

THRESHOLD = 0.75
NUM_SLICES = 6
CROP_PADDING = 35
MAX_DISPLAY_SIZE = 512


# ============================================================
# VALIDATE DATA SPLIT
# ============================================================

def validate_splits():

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

    print("Validation split verified.")
    print("Test set remains untouched.")


# ============================================================
# LOAD ORIGINAL CT
# ============================================================

def load_original_ct(patient_id):

    path = (
        ROOT / "data" / "raw" / "train"
        / patient_id / "image" / f"{patient_id}.nii.gz"
    )

    nii = nib.load(str(path))

    return np.asarray(
        nii.dataobj,
        dtype=np.float32
    )


# ============================================================
# DISPLAY RESIZING
# ============================================================

def resize_for_display(array, is_mask=False):

    height, width = array.shape

    scale = min(
        1.0,
        MAX_DISPLAY_SIZE / max(height, width)
    )

    if scale >= 1.0:
        return array

    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))

    return zoom(
        array,
        (
            new_height / height,
            new_width / width
        ),
        order=0 if is_mask else 1
    )


# ============================================================
# SELECT INFORMATIVE SLICES
# ============================================================

def select_slices(label, prediction, num_slices):

    gt_counts = label.sum(axis=(0, 1))
    pred_counts = prediction.sum(axis=(0, 1))

    combined = gt_counts + pred_counts

    valid = np.flatnonzero(combined > 0)

    if valid.size == 0:
        raise ValueError("No foreground found.")

    if valid.size <= num_slices:
        return valid.tolist()

    positions = np.linspace(
        0,
        valid.size - 1,
        num_slices
    )

    indices = np.round(positions).astype(int)

    return valid[indices].tolist()


# ============================================================
# CREATE ERROR MAP
# ============================================================

def create_error_map(prediction, label):

    tp = prediction & label
    fp = prediction & ~label
    fn = ~prediction & label

    error_map = np.zeros(
        label.shape,
        dtype=np.uint8
    )

    error_map[tp] = 1
    error_map[fp] = 2
    error_map[fn] = 3

    return error_map


# ============================================================
# CREATE CROP AROUND GROUND-TRUTH REGION
# ============================================================

def get_crop_bounds(label_slice, prediction_slice):

    region = label_slice

    if not region.any():
        region = prediction_slice

    height, width = region.shape

    if not region.any():
        return 0, height, 0, width

    coords = np.argwhere(region)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, int(y_min) - CROP_PADDING)
    y_max = min(height, int(y_max) + CROP_PADDING + 1)

    x_min = max(0, int(x_min) - CROP_PADDING)
    x_max = min(width, int(x_max) + CROP_PADDING + 1)

    crop_height = y_max - y_min
    crop_width = x_max - x_min

    side = max(crop_height, crop_width)

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    half = side // 2

    y_min = max(0, center_y - half)
    y_max = min(height, y_min + side)

    x_min = max(0, center_x - half)
    x_max = min(width, x_min + side)

    y_min = max(0, y_max - side)
    x_min = max(0, x_max - side)

    return y_min, y_max, x_min, x_max


# ============================================================
# OVERLAY HELPERS
# ============================================================

def show_binary_overlay(ax, ct, mask, color):

    ax.imshow(
        ct,
        cmap="gray",
        vmin=-1000,
        vmax=500
    )

    overlay = np.ma.masked_where(
        ~mask,
        mask.astype(np.float32)
    )

    cmap = ListedColormap([color])

    ax.imshow(
        overlay,
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.75,
        interpolation="nearest"
    )


def show_error_overlay(ax, error_map):

    error_cmap = ListedColormap([
        "black",
        "#00D084",  # TP
        "#FF3B30",  # FP
        "#FFD60A",  # FN
    ])

    ax.imshow(
        error_map,
        cmap=error_cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )


# ============================================================
# VISUALIZE ONE PATIENT
# ============================================================

def visualize_patient(patient_id, model, device):

    print("\n" + "=" * 65)
    print("Processing:", patient_id)
    print("=" * 65)

    image, frangi, label = load_case(patient_id)

    label = np.asarray(label, dtype=bool)
    frangi = np.asarray(frangi, dtype=np.float32)

    original_ct = load_original_ct(patient_id)

    if original_ct.shape != label.shape:
        raise ValueError(
            f"CT shape mismatch for {patient_id}"
        )

    # Exact V4 inference.
    probabilities = predict_volume(
        model,
        image,
        frangi,
        device
    )

    prediction = probabilities >= THRESHOLD

    tp = prediction & label
    fp = prediction & ~label
    fn = ~prediction & label

    intersection = int(tp.sum())

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

    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("TP:", int(tp.sum()))
    print("FP:", int(fp.sum()))
    print("FN:", int(fn.sum()))

    error_map = create_error_map(
        prediction,
        label
    )

    slice_indices = select_slices(
        label,
        prediction,
        NUM_SLICES
    )

    print("Selected slices:", slice_indices)

    # Six-column research visualization.
    fig, axes = plt.subplots(
        len(slice_indices),
        6,
        figsize=(23, 4.2 * len(slice_indices)),
        squeeze=False
    )

    titles = [
        "Original CT",
        "Ground Truth Overlay",
        "V4 Prediction Overlay",
        "TP / FP / FN",
        "Frangi Response",
        "Enlarged Error Crop",
    ]

    for column, title in enumerate(titles):

        axes[0, column].set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

    for row, z in enumerate(slice_indices):

        ct = resize_for_display(
            original_ct[:, :, z]
        )

        gt = resize_for_display(
            label[:, :, z],
            is_mask=True
        ).astype(bool)

        pred = resize_for_display(
            prediction[:, :, z],
            is_mask=True
        ).astype(bool)

        errors = resize_for_display(
            error_map[:, :, z],
            is_mask=True
        ).astype(np.uint8)

        frangi_slice = resize_for_display(
            frangi[:, :, z]
        )

        # 1. Original CT.
        axes[row, 0].imshow(
            ct,
            cmap="gray",
            vmin=-1000,
            vmax=500
        )

        # 2. Ground truth overlay.
        show_binary_overlay(
            axes[row, 1],
            ct,
            gt,
            "#00FF55"
        )

        # 3. Prediction overlay.
        show_binary_overlay(
            axes[row, 2],
            ct,
            pred,
            "#00CFFF"
        )

        # 4. Error map.
        show_error_overlay(
            axes[row, 3],
            errors
        )

        # 5. Frangi response.
        axes[row, 4].imshow(
            frangi_slice,
            cmap="inferno",
            vmin=0,
            vmax=1
        )

        # 6. Enlarged crop around GT region.
        y0, y1, x0, x1 = get_crop_bounds(
            label[:, :, z],
            prediction[:, :, z]
        )

        crop_ct = original_ct[
            y0:y1, x0:x1, z
        ]

        crop_gt = label[
            y0:y1, x0:x1, z
        ]

        crop_pred = prediction[
            y0:y1, x0:x1, z
        ]

        crop_error = error_map[
            y0:y1, x0:x1, z
        ]

        axes[row, 5].imshow(
            crop_ct,
            cmap="gray",
            vmin=-1000,
            vmax=500
        )

        # Ground truth overlay in yellow.
        gt_overlay = np.ma.masked_where(
            ~crop_gt,
            crop_gt.astype(np.float32)
        )

        axes[row, 5].imshow(
            gt_overlay,
            cmap=ListedColormap(["#FFD60A"]),
            vmin=0,
            vmax=1,
            alpha=0.75
        )

        # Prediction outline in cyan.
        if crop_pred.any():

            axes[row, 5].contour(
                crop_pred.astype(float),
                levels=[0.5],
                colors=["#00CFFF"],
                linewidths=0.8
            )

        axes[row, 0].set_ylabel(
            f"Slice {z}",
            fontsize=10
        )

        for ax in axes[row]:

            ax.set_xticks([])
            ax.set_yticks([])

    # Figure title.
    fig.suptitle(
        (
            f"NeuroVessel V4 Spatial Error Analysis V2 | "
            f"{patient_id}\n"
            f"Dice={dice:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"Threshold={THRESHOLD}"
        ),
        fontsize=15,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.005,
        (
            "GT overlay = Green | Prediction overlay = Cyan | "
            "Error map: Green=TP, Red=FP, Yellow=FN"
        ),
        ha="center",
        fontsize=11
    )

    fig.tight_layout(
        rect=[0, 0.025, 1, 0.94]
    )

    output_path = (
        OUTPUT_DIR
        / f"{patient_id}_spatial_error_analysis_v2.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)

    del image, frangi, label, original_ct
    del probabilities, prediction
    del tp, fp, fn, error_map

    torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V4 - IMPROVED SPATIAL ERROR ANALYSIS")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    validate_splits()

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

    for patient_id in PATIENT_IDS:

        visualize_patient(
            patient_id,
            model,
            device
        )

    print("\n" + "=" * 65)
    print("IMPROVED VISUALIZATION COMPLETE")
    print("=" * 65)

    print("Output:", OUTPUT_DIR)
    print("Validation patients only.")
    print("No training performed.")
    print("Test set NOT accessed.")


if __name__ == "__main__":
    main()