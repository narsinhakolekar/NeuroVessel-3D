
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from monai.networks.nets import BasicUNet


ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT = Path("outputs/checkpoints/unet_interp_best.pt")
OUTPUT_DIR = Path("outputs/visualizations")

PATCH_SIZE = 64


def load_case(patient_id):
    folder = ROOT / patient_id

    image_file = nib.load(
        str(folder / "image" / f"{patient_id}.nii.gz")
    )
    label_file = nib.load(
        str(folder / "label" / f"{patient_id}.nii.gz")
    )

    if image_file.shape != label_file.shape:
        raise ValueError("Image and label shapes differ.")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError("Image and label affines differ.")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0

    image = (np.clip(image, -1000, 500) + 1000.0) / 1500.0

    return image, label


def extract_foreground_patch(image, label):
    p = PATCH_SIZE
    sx, sy, sz = label.shape

    foreground = np.argwhere(label)

    if len(foreground) == 0:
        raise ValueError("The selected patient's label mask is empty.")

    # Select a reproducible foreground voxel.
    cx, cy, cz = map(int, foreground[len(foreground) // 2])

    x = int(np.clip(cx - p // 2, 0, sx - p))
    y = int(np.clip(cy - p // 2, 0, sy - p))
    z = int(np.clip(cz - p // 2, 0, sz - p))

    image_patch = image[x:x + p, y:y + p, z:z + p].copy()
    label_patch = label[x:x + p, y:y + p, z:z + p].copy()

    if image_patch.shape != (p, p, p):
        raise ValueError(f"Unexpected patch shape: {image_patch.shape}")

    if int(label_patch.sum()) == 0:
        raise RuntimeError(
            "Foreground-centred patch unexpectedly contains no foreground."
        )

    return image_patch, label_patch, (x, y, z)


def dice_score(prediction, target):
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

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]

    if not validation_ids:
        raise ValueError("No validation patients found in data/splits.json")

    patient_id = validation_ids[0]
    print("Validation patient:", patient_id)

    image, label = load_case(patient_id)
    image_patch, label_patch, origin = extract_foreground_patch(
        image, label
    )

    # Must match the BasicUNet configuration used for training.
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample="nontrainable",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tensor = torch.from_numpy(
        image_patch[None, None].astype(np.float32)
    ).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(input_tensor)

        probabilities = (
            torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        )

    print("\nPatch diagnostics")
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

        print(
            f"threshold={threshold:.1f} | "
            f"predicted_fraction={prediction_at_threshold.mean():.4f} | "
            f"Dice={dice_score(prediction_at_threshold, label_patch):.4f}"
        )

    prediction = probabilities >= 0.5
    dice = dice_score(prediction, label_patch)

    # Choose the slice with the most ground-truth foreground.
    foreground_per_slice = label_patch.sum(axis=(0, 1))
    slice_index = int(np.argmax(foreground_per_slice))

    ct_slice = image_patch[:, :, slice_index].T
    target_slice = label_patch[:, :, slice_index].T
    probability_slice = probabilities[:, :, slice_index].T
    prediction_slice = prediction[:, :, slice_index].T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: probability diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT")

    im = axes[1].imshow(
        probability_slice,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predicted probability")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(target_slice, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Probability diagnostics | {patient_id} | slice {slice_index}"
    )
    fig.tight_layout()

    probability_path = OUTPUT_DIR / "unet_interp_probability_diagnostics.png"
    fig.savefig(probability_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: prediction overlay.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT")

    axes[1].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        np.ma.masked_where(~target_slice, target_slice),
        cmap="Greens",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground truth")

    axes[2].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
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
        f"NeuroVessel interpolation U-Net | Dice={dice:.4f} "
        f"| slice={slice_index}"
    )
    fig.tight_layout()

    overlay_path = OUTPUT_DIR / "unet_interp_diagnostics.png"
    fig.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nProbability visualization saved:", probability_path)
    print("Overlay visualization saved:", overlay_path)
    print("Patch Dice at threshold 0.5:", dice)
    print(
        "\nThis is a foreground-centred validation patch, "
        "not a full-volume evaluation."
    )


if __name__ == "__main__":
    main()
