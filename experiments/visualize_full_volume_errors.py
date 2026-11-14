
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from evaluate_interp_validation import (
    SPLIT_FILE,
    load_case,
    load_model,
    predict_full_volume,
    calculate_metrics,
)

CHECKPOINT = Path("outputs/checkpoints/unet_interp_best.pt")
OUTPUT_DIR = Path("outputs/visualizations/full_volume_errors")
THRESHOLD = 0.60
PATIENTS_TO_INSPECT = 3
SLICES_PER_PATIENT = 3


def save_slice(patient_id, image, target, probabilities, z, metrics):
    prediction = probabilities[:, :, z] >= THRESHOLD
    target_slice = target[:, :, z]
    ct_slice = image[:, :, z].T
    target_slice = target_slice.T
    prediction = prediction.T
    probability_slice = probabilities[:, :, z].T

    false_positive = prediction & ~target_slice
    false_negative = target_slice & ~prediction

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("CT")

    axes[0, 1].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].imshow(
        np.ma.masked_where(~target_slice, target_slice),
        cmap="Greens",
        alpha=0.7,
        interpolation="nearest",
    )
    axes[0, 1].set_title("Ground truth")

    axes[0, 2].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].imshow(
        np.ma.masked_where(~prediction, prediction),
        cmap="Reds",
        alpha=0.7,
        interpolation="nearest",
    )
    axes[0, 2].set_title(f"Prediction (threshold {THRESHOLD})")

    axes[1, 0].imshow(
        probability_slice,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1, 0].set_title("Predicted probability")

    axes[1, 1].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].imshow(
        np.ma.masked_where(~false_positive, false_positive),
        cmap="Reds",
        alpha=0.8,
        interpolation="nearest",
    )
    axes[1, 1].set_title("False positives")

    axes[1, 2].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].imshow(
        np.ma.masked_where(~false_negative, false_negative),
        cmap="Blues",
        alpha=0.8,
        interpolation="nearest",
    )
    axes[1, 2].set_title("False negatives")

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        f"{patient_id} | axial index {z} | "
        f"volume Dice={metrics['dice']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )
    fig.tight_layout()

    path = OUTPUT_DIR / f"{patient_id}_z{z:03d}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]

    if set(validation_ids) & set(split["test"]):
        raise RuntimeError("Validation/test leakage detected.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT)
    print("Threshold:", THRESHOLD)
    print("Test patients remain untouched.")

    model = load_model(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_ids = validation_ids[:PATIENTS_TO_INSPECT]

    for patient_id in selected_ids:
        print(f"\nProcessing {patient_id}...")

        image, target = load_case(patient_id)
        probabilities = predict_full_volume(model, image, device)

        prediction = probabilities >= THRESHOLD
        metrics = calculate_metrics(prediction, target)

        print(
            f"Full-volume Dice={metrics['dice']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Predicted fraction="
            f"{metrics['predicted_foreground_fraction']:.4f} | "
            f"GT fraction={metrics['ground_truth_foreground_fraction']:.4f}"
        )

        # Select slices with the most ground-truth foreground.
        # This makes vessel-related errors easier to inspect.
        foreground_per_slice = target.sum(axis=(0, 1))
        candidate_slices = np.flatnonzero(foreground_per_slice > 0)

        if len(candidate_slices) == 0:
            print("No labelled slices; skipping visualization.")
            continue

        ranked = candidate_slices[
            np.argsort(foreground_per_slice[candidate_slices])[::-1]
        ]

        chosen = sorted(
            int(z) for z in ranked[:SLICES_PER_PATIENT]
        )

        for z in chosen:
            save_slice(
                patient_id, image, target, probabilities, z, metrics
            )

        del image, target, probabilities, prediction

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nSaved visualizations to:", OUTPUT_DIR)
    print("Inspect the false-positive and false-negative panels.")
    print("These images are validation diagnostics, not test results.")


if __name__ == "__main__":
    main()
