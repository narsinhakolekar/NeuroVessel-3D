
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet


DATA_ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT = Path("outputs/checkpoints/unet_v2_best.pt")
OUTPUT_DIR = Path("outputs/visualizations/v2_errors")

THRESHOLD = 0.7
PATCH_SIZE = 64
PATIENT_COUNT = 3
SLICES_PER_PATIENT = 3


def load_case(patient_id):
    folder = DATA_ROOT / patient_id
    image_file = nib.load(
        str(folder / "image" / f"{patient_id}.nii.gz")
    )
    label_file = nib.load(
        str(folder / "label" / f"{patient_id}.nii.gz")
    )

    if image_file.shape != label_file.shape:
        raise ValueError(f"Image/label shape mismatch: {patient_id}")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"Image/label affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    target = np.asarray(label_file.dataobj) != 0

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite CT values: {patient_id}")

    if not target.any():
        raise ValueError(f"Empty ground-truth mask: {patient_id}")

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    return image, target


def predict_volume(model, image, device):
    tensor = torch.from_numpy(
        np.ascontiguousarray(image[None, None], dtype=np.float32)
    )

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=(PATCH_SIZE,) * 3,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
                sw_device=device,
                device=torch.device("cpu"),
            )

        probabilities = torch.sigmoid(logits)[0, 0].float().numpy()

    del tensor, logits
    return probabilities


def calculate_metrics(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    predicted_count = prediction.sum()
    target_count = target.sum()

    return {
        "dice": float(
            2 * intersection / (predicted_count + target_count)
        ) if predicted_count + target_count else 1.0,
        "precision": float(intersection / predicted_count)
        if predicted_count else 0.0,
        "recall": float(intersection / target_count)
        if target_count else 0.0,
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
    }


def choose_slices(target, count):
    # Select slices with substantial ground-truth foreground.
    # The array's third axis is used as the displayed slice axis.
    foreground_per_slice = target.sum(axis=(0, 1))
    candidates = np.flatnonzero(foreground_per_slice > 0)

    if len(candidates) == 0:
        return []

    order = candidates[
        np.argsort(foreground_per_slice[candidates])[::-1]
    ]

    chosen = []
    for z in order:
        if all(abs(int(z) - previous) >= 5 for previous in chosen):
            chosen.append(int(z))
        if len(chosen) == count:
            break

    return sorted(chosen)


def save_patient_figure(patient_id, image, target, probabilities, output_dir):
    prediction = probabilities >= THRESHOLD
    metrics = calculate_metrics(prediction, target)
    slices = choose_slices(target, SLICES_PER_PATIENT)

    if not slices:
        print(f"No foreground slices found for {patient_id}")
        return

    fig, axes = plt.subplots(
        len(slices), 4,
        figsize=(16, 4.5 * len(slices)),
        squeeze=False,
    )

    for row, z in enumerate(slices):
        ct = image[:, :, z].T
        gt = target[:, :, z].T
        pred = prediction[:, :, z].T

        axes[row, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"CT — slice {z}")

        axes[row, 1].imshow(ct, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(
            gt, cmap="Greens", alpha=0.55, vmin=0, vmax=1
        )
        axes[row, 1].set_title("Ground truth")

        axes[row, 2].imshow(ct, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(
            pred, cmap="Reds", alpha=0.55, vmin=0, vmax=1
        )
        axes[row, 2].set_title(f"Prediction (threshold {THRESHOLD})")

        # Green = true positive; red = false positive; blue = false negative.
        overlap = np.zeros((*gt.shape, 3), dtype=np.float32)
        overlap[gt & pred] = [0.0, 1.0, 0.0]
        overlap[pred & ~gt] = [1.0, 0.0, 0.0]
        overlap[gt & ~pred] = [0.0, 0.4, 1.0]

        axes[row, 3].imshow(ct, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlap, alpha=0.8)
        axes[row, 3].set_title("Overlap: TP green, FP red, FN blue")

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(
        f"{patient_id} | Dice={metrics['dice']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}",
        fontsize=13,
    )
    fig.tight_layout()

    output_path = output_dir / f"{patient_id}_v2_diagnostic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{patient_id}")
    print("  Metrics:", json.dumps(metrics, indent=2))
    print("  Saved:", output_path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]
    test_ids = set(split["test"])

    if set(validation_ids) & test_ids:
        raise RuntimeError("Validation/test overlap detected.")

    patient_ids = validation_ids[:PATIENT_COUNT]
    device = torch.device("cuda")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample="nontrainable",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("GPU:", torch.cuda.get_device_name(0))
    print("Threshold:", THRESHOLD)
    print("Validation patients:", patient_ids)
    print("Test patients will not be loaded.")

    for patient_id in patient_ids:
        print(f"\nPredicting {patient_id}...")
        image, target = load_case(patient_id)
        probabilities = predict_volume(model, image, device)

        save_patient_figure(
            patient_id, image, target, probabilities, OUTPUT_DIR
        )

        del image, target, probabilities
        torch.cuda.empty_cache()

    print("\nVisualization complete.")
    print("Output directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
