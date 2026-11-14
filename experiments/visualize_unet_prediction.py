
import json
from pathlib import Path

import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.networks.nets import UNet


SPLIT_FILE = Path("data/splits.json")
DATA_ROOT = Path("data/raw/train")
CHECKPOINT_PATH = Path("outputs/checkpoints/unet_baseline_best.pt")
OUTPUT_PATH = Path("outputs/visualizations/unet_validation_prediction.png")

PATCH_SIZE = 64


def load_case(patient_id):
    folder = DATA_ROOT / patient_id
    image_path = folder / "image" / f"{patient_id}.nii.gz"
    label_path = folder / "label" / f"{patient_id}.nii.gz"

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(f"Image/label shape mismatch: {patient_id}")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"Image/label affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite CT values: {patient_id}")

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    return image, label


def get_validation_patch(image, label):
    p = PATCH_SIZE
    sx, sy, sz = image.shape

    if min(sx, sy, sz) < p:
        raise ValueError(f"Volume smaller than patch size {p}")

    foreground = np.argwhere(label)

    if len(foreground) == 0:
        raise ValueError("Selected validation patient has an empty mask")

    # Deterministically choose a foreground voxel as the patch centre.
    cx, cy, cz = map(int, foreground[len(foreground) // 2])

    x = int(np.clip(cx - p // 2, 0, sx - p))
    y = int(np.clip(cy - p // 2, 0, sy - p))
    z = int(np.clip(cz - p // 2, 0, sz - p))

    image_patch = image[x:x + p, y:y + p, z:z + p].copy()
    label_patch = label[x:x + p, y:y + p, z:z + p].copy()

    return image_patch, label_patch


def calculate_dice(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    denominator = int(prediction.sum()) + int(target.sum())

    if denominator == 0:
        return float("nan")

    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / denominator


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate neurovessel.")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]

    if not validation_ids:
        raise ValueError("No validation patients in data/splits.json")

    patient_id = validation_ids[0]

    print("Validation patient:", patient_id)
    print("GPU:", torch.cuda.get_device_name(0))

    image, label = load_case(patient_id)
    image_patch, label_patch = get_validation_patch(image, label)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tensor = torch.from_numpy(
        image_patch[None, None, ...]
    ).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = (
            torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        )

    if probabilities.shape != label_patch.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} does not match "
            f"label shape {label_patch.shape}"
        )

    prediction = probabilities >= 0.5

    target_count = int(label_patch.sum())
    prediction_count = int(prediction.sum())
    dice = calculate_dice(prediction, label_patch)

    print("\nPatch diagnostics")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Patch shape:", image_patch.shape)
    print("Ground-truth foreground voxels:", target_count)
    print("Predicted foreground voxels:", prediction_count)
    print("Ground-truth foreground fraction:", float(label_patch.mean()))
    print("Predicted foreground fraction:", float(prediction.mean()))
    print(
        "Probability min/max:",
        float(probabilities.min()),
        float(probabilities.max()),
    )
    print("Patch Dice at threshold 0.5:", dice)

    # Threshold sensitivity is diagnostic only.
    # Do not choose a final threshold using this single patch.
    print("\nThreshold diagnostics")

    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pred = probabilities >= threshold

        print(
            f"threshold={threshold:.1f} | "
            f"foreground={pred.mean():.4f} | "
            f"Dice={calculate_dice(pred, label_patch):.4f}"
        )

    # Select the axial slice with the most ground-truth foreground
    # BEFORE using slice_index in any visualization.
    foreground_per_slice = label_patch.sum(axis=(0, 1))
    slice_index = int(np.argmax(foreground_per_slice))

    ct_slice = image_patch[:, :, slice_index].T
    target_slice = label_patch[:, :, slice_index].T
    prediction_slice = prediction[:, :, slice_index].T
    probability_slice = probabilities[:, :, slice_index].T

    output_dir = OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probability map: inspect the raw model output.
    probability_path = output_dir / "unet_validation_probability.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT")

    axes[1].imshow(
        probability_slice,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("Predicted probability")

    axes[2].imshow(
        target_slice,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Ground truth")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(f"Probability diagnostic | {patient_id} | slice {slice_index}")
    fig.tight_layout()
    fig.savefig(probability_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nProbability map saved to:", probability_path)

    # CT, ground truth overlay, and prediction overlay.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT — {patient_id}")

    axes[1].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        np.ma.masked_where(~target_slice, target_slice),
        cmap="Greens",
        alpha=0.65,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("Ground truth (green)")

    axes[2].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        np.ma.masked_where(~prediction_slice, prediction_slice),
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[2].set_title("Prediction (red, threshold 0.5)")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"NeuroVessel U-Net | patch Dice={dice:.4f} | slice={slice_index}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Visualization saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
