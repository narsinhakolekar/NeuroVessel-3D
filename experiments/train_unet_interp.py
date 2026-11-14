
import json
from pathlib import Path

import nibabel as nib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.networks.nets import BasicUNet


SPLIT_FILE = Path("data/splits.json")
DATA_ROOT = Path("data/raw/train")
CHECKPOINT_PATH = Path("outputs/checkpoints/unet_interp_best.pt")
OUTPUT_DIR = Path("outputs/visualizations")

PATCH_SIZE = 64


def load_case(patient_id):
    folder = DATA_ROOT / patient_id
    image_path = folder / "image" / f"{patient_id}.nii.gz"
    label_path = folder / "label" / f"{patient_id}.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(f"Image/label shape mismatch: {patient_id}")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"Image/label affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    return image, label


def get_foreground_patch(image, label):
    p = PATCH_SIZE
    sx, sy, sz = image.shape

    if min(sx, sy, sz) < p:
        raise ValueError(f"Volume smaller than patch size {p}")

    foreground = np.argwhere(label)

    if len(foreground) == 0:
        raise ValueError("The selected patient has an empty label mask.")

    # Choose a reproducible foreground voxel near the middle
    # of the foreground coordinate list.
    cx, cy, cz = map(int, foreground[len(foreground) // 2])

    x = int(np.clip(cx - p // 2, 0, sx - p))
    y = int(np.clip(cy - p // 2, 0, sy - p))
    z = int(np.clip(cz - p // 2, 0, sz - p))

    image_patch = image[x:x + p, y:y + p, z:z + p].copy()
    label_patch = label[x:x + p, y:y + p, z:z + p].copy()

    if image_patch.shape != (p, p, p):
        raise ValueError(f"Unexpected image patch shape: {image_patch.shape}")

    if label_patch.sum() == 0:
        raise RuntimeError(
            "Foreground-centred patch has no foreground voxels. "
            "Check image/label alignment and patch extraction."
        )

    return image_patch, label_patch, (x, y, z)


def calculate_dice(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    denominator = int(prediction.sum()) + int(target.sum())

    if denominator == 0:
        return float("nan")

    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / denominator



def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample=checkpoint.get("upsample", "nontrainable"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Architecture:", checkpoint.get("architecture", "unknown"))
    print("Upsampling:", checkpoint.get("upsample", "unknown"))

    return model, checkpoint



def save_probability_visualization(
    image_patch,
    label_patch,
    probabilities,
    slice_index,
    patient_id,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_slice = image_patch[:, :, slice_index].T
    label_slice = label_patch[:, :, slice_index].T
    probability_slice = probabilities[:, :, slice_index].T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT")

    im = axes[1].imshow(
        probability_slice,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predicted probability")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(label_slice, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Probability diagnostics | {patient_id} | slice {slice_index}"
    )
    fig.tight_layout()

    path = OUTPUT_DIR / "unet_interp_probability_diagnostics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def save_overlay_visualization(
    image_patch,
    label_patch,
    prediction,
    slice_index,
    patient_id,
    dice,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_slice = image_patch[:, :, slice_index].T
    target_slice = label_patch[:, :, slice_index].T
    prediction_slice = prediction[:, :, slice_index].T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT — {patient_id}")

    axes[1].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        np.ma.masked_where(~target_slice, target_slice),
        cmap="Greens",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground truth")

    axes[2].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        np.ma.masked_where(~prediction_slice, prediction_slice),
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Prediction at threshold 0.5")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"NeuroVessel interpolation U-Net | patch Dice={dice:.4f} "
        f"| slice={slice_index}"
    )
    fig.tight_layout()

    path = OUTPUT_DIR / "unet_interp_diagnostics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the neurovessel environment."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split.get("validation", [])

    if not validation_ids:
        raise ValueError("No validation patients found in data/splits.json")

    patient_id = validation_ids[0]
    print("Validation patient:", patient_id)

    image, label = load_case(patient_id)
    image_patch, label_patch, origin = get_foreground_patch(image, label)

    model, checkpoint = load_model(device)

    input_tensor = torch.from_numpy(
        image_patch[None, None, ...].astype(np.float32)
    ).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    if probabilities.shape != label_patch.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: "
            f"{probabilities.shape} vs {label_patch.shape}"
        )

    print("\nPatch diagnostics")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Patch origin:", origin)
    print("Patch shape:", image_patch.shape)
    print("Ground-truth foreground voxels:", int(label_patch.sum()))
    print("Ground-truth foreground fraction:", float(label_patch.mean()))
    print(
        "Probability min/max/mean:",
        float(probabilities.min()),
        float(probabilities.max()),
        float(probabilities.mean()),
    )

    print("\nThreshold diagnostics")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        prediction_at_threshold = probabilities >= threshold
        dice = calculate_dice(prediction_at_threshold, label_patch)

        print(
            f"threshold={threshold:.1f} | "
            f"predicted_fraction={prediction_at_threshold.mean():.4f} | "
            f"Dice={dice:.4f}"
        )

    # Use the middle slice containing the most target foreground.
    foreground_per_slice = label_patch.sum(axis=(0, 1))
    slice_index = int(np.argmax(foreground_per_slice))

    prediction = probabilities >= 0.5
    dice = calculate_dice(prediction, label_patch)

    probability_path = save_probability_visualization(
        image_patch,
        label_patch,
        probabilities,
        slice_index,
        patient_id,
    )

    overlay_path = save_overlay_visualization(
        image_patch,
        label_patch,
        prediction,
        slice_index,
        patient_id,
        dice,
    )

    print("\nSaved probability visualization:", probability_path)
    print("Saved overlay visualization:", overlay_path)
    print("Patch Dice at threshold 0.5:", dice)
    print(
        "\nReminder: this is one foreground-centred validation patch, "
        "not a full-volume or test-set evaluation."
    )


if __name__ == "__main__":
    main()
