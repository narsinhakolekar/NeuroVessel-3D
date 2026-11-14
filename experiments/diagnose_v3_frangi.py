
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_unet_v3_frangi import (
    BEST_PATH,
    build_model,
    load_case,
    predict_volume,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PATIENT_ID = "PA000005"
THRESHOLDS = [0.5, 0.9]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / "v3_frangi_diagnostic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_v3_model():

    checkpoint_path = Path(BEST_PATH)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = build_model(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded V3 checkpoint.")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    return model


# --------------------------------------------------
# Main diagnostic
# --------------------------------------------------

def main():

    print("=" * 60)
    print("NEUROVESSEL V3 FRANGI DIAGNOSTIC")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load model
    model = load_v3_model()

    # Load validation patient
    print("\nLoading patient:", PATIENT_ID)

    image, frangi, label = load_case(PATIENT_ID)

    image = np.asarray(image, dtype=np.float32)
    frangi = np.asarray(frangi, dtype=np.float32)
    label = np.asarray(label) > 0

    print("CT shape:", image.shape)
    print("Frangi shape:", frangi.shape)
    print("Label shape:", label.shape)

    if not (image.shape == frangi.shape == label.shape):
        raise ValueError("CT, Frangi and label shapes differ.")

    # Full-volume prediction
    print("\nRunning full-volume inference...")

    probabilities = predict_volume(
        model,
        image,
        frangi,
        DEVICE
    )

    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    if probabilities.shape != label.shape:
        raise ValueError("Prediction and label shapes differ.")

    print("Inference completed.")

    # --------------------------------------------------
    # Select slice with largest ground-truth area
    # --------------------------------------------------

    slice_areas = label.sum(axis=(0, 1))
    slice_index = int(np.argmax(slice_areas))

    ct = image[:, :, slice_index]
    vesselness = frangi[:, :, slice_index]
    gt = label[:, :, slice_index]
    probability = probabilities[:, :, slice_index]

    print("Selected slice:", slice_index)

    # Predictions
    pred_05 = probability >= 0.5
    pred_09 = probability >= 0.9

    # False positives and false negatives
    false_positive = pred_05 & (~gt)
    false_negative = (~pred_05) & gt
    true_positive = pred_05 & gt

    # --------------------------------------------------
    # Print diagnostic statistics
    # --------------------------------------------------

    print("\nSLICE DIAGNOSTICS")
    print("-" * 40)

    print("Ground truth pixels:", int(gt.sum()))
    print("Prediction at 0.5:", int(pred_05.sum()))
    print("Prediction at 0.9:", int(pred_09.sum()))

    print("True positives:", int(true_positive.sum()))
    print("False positives:", int(false_positive.sum()))
    print("False negatives:", int(false_negative.sum()))

    print("Frangi minimum:", float(vesselness.min()))
    print("Frangi maximum:", float(vesselness.max()))
    print("Frangi mean:", float(vesselness.mean()))

    print("Probability minimum:", float(probability.min()))
    print("Probability maximum:", float(probability.max()))
    print("Probability mean:", float(probability.mean()))

    # --------------------------------------------------
    # Eight-panel visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(21, 12)
    )

    panels = [
        (ct, "Original CT", "gray"),
        (gt, "Ground Truth PA", "gray"),
        (vesselness, "Frangi Vesselness", "hot"),
        (probability, "V3 Probability", "hot"),
        (pred_05, "Prediction Threshold 0.5", "gray"),
        (pred_09, "Prediction Threshold 0.9", "gray"),
    ]

    for index, (data, title, cmap) in enumerate(panels):

        ax = axes.flat[index]

        im = ax.imshow(
            data.T,
            cmap=cmap,
            origin="lower"
        )

        ax.set_title(title)
        ax.axis("off")

        if title in ["Frangi Vesselness", "V3 Probability"]:
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

    # False-positive / false-negative map
    error_map = np.zeros(
        (*gt.T.shape, 3),
        dtype=np.float32
    )

    error_map[false_positive.T] = [1, 0, 0]
    error_map[false_negative.T] = [0, 0, 1]
    error_map[true_positive.T] = [0, 1, 0]

    axes[1, 2].imshow(
        error_map,
        origin="lower"
    )

    axes[1, 2].set_title(
        "Errors: FP Red / FN Blue / TP Green"
    )
    axes[1, 2].axis("off")

    # CT with Frangi overlay
    axes[1, 3].imshow(
        ct.T,
        cmap="gray",
        origin="lower"
    )

    axes[1, 3].imshow(
        vesselness.T,
        cmap="hot",
        alpha=0.45,
        origin="lower"
    )

    axes[1, 3].set_title("CT + Frangi Overlay")
    axes[1, 3].axis("off")

    fig.suptitle(
        f"NeuroVessel V3 Diagnostic | {PATIENT_ID} "
        f"| Slice {slice_index}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "v3_frangi_diagnostic.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Save individual arrays for further analysis
    np.save(OUTPUT_DIR / "frangi_slice.npy", vesselness)
    np.save(OUTPUT_DIR / "probability_slice.npy", probability)
    np.save(OUTPUT_DIR / "ground_truth_slice.npy", gt.astype(np.uint8))

    print("\nDiagnostic completed.")
    print("Output:", output_path.resolve())


if __name__ == "__main__":
    main()
