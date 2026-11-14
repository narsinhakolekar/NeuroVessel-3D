
import numpy as np
import nibabel as nib
from pathlib import Path

from src.filters.classical_filters import (
    gaussian_filter_3d,
    median_filter_3d,
    frangi_vesselness_3d,
)


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return dice, precision, recall


def get_thresholds(response, filter_name):
    if filter_name == "Frangi":
        positive = response[response > 1e-8]

        if positive.size == 0:
            return np.array([], dtype=np.float32)

        quantiles = np.linspace(0.10, 0.99, 60)
        return np.unique(np.quantile(positive, quantiles))

    quantiles = np.linspace(0.50, 0.999, 60)
    return np.unique(np.quantile(response, quantiles))


def main():
    root = Path("data/raw/train/PA000005")

    image_path = root / "image" / "PA000005.nii.gz"
    label_path = root / "label" / "PA000005.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"CT image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(
            f"Image shape {image_file.shape} differs from "
            f"label shape {label_file.shape}"
        )

    if not np.allclose(
        image_file.affine, label_file.affine, atol=1e-4
    ):
        raise ValueError(
            "Image and label affine matrices differ. "
            "Check spatial alignment before evaluation."
        )

    x0, y0, z0 = 192, 192, 125
    x1, y1, z1 = 320, 320, 189

    ct = np.asarray(
        image_file.dataobj[x0:x1, y0:y1, z0:z1],
        dtype=np.float32,
    )

    labels = np.asarray(
        label_file.dataobj[x0:x1, y0:y1, z0:z1]
    )

    if ct.size == 0 or labels.size == 0:
        raise ValueError("The selected crop is empty")

    if not np.isfinite(ct).all():
        raise ValueError("CT crop contains NaN or infinite values")

    mask = labels != 0

    if not mask.any():
        raise ValueError(
            "The selected crop contains no foreground labels. "
            "Choose a crop containing labelled structures."
        )

    # Convert CT intensities from Hounsfield units into [0, 1].
    ct = np.clip(ct, -1000, 500)
    ct = (ct + 1000.0) / 1500.0

    print("Crop shape:", ct.shape)
    print("Foreground voxels:", int(mask.sum()))
    print("Foreground fraction:", f"{mask.mean():.6f}")
    print("\nExploratory threshold results")
    print("-" * 78)

    responses = {
        "Gaussian": gaussian_filter_3d(ct, sigma=1.0),
        "Median": median_filter_3d(ct, size=3),
        "Frangi": frangi_vesselness_3d(
            ct,
            sigmas=(1, 2, 3),
        ),
    }

    results = {}

    for name, response in responses.items():
        if not np.isfinite(response).all():
            raise ValueError(f"{name} response contains non-finite values")

        thresholds = get_thresholds(response, name)

        if thresholds.size == 0:
            print(f"{name}: no usable thresholds")
            continue

        best = None

        for threshold in thresholds:
            prediction = response >= threshold

            dice, precision, recall = calculate_metrics(
                prediction, mask
            )

            if best is None or dice > best["dice"]:
                best = {
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "threshold": float(threshold),
                    "predicted_fraction": float(prediction.mean()),
                }

        results[name] = best

        print(
            f"{name}: "
            f"Dice={best['dice']:.4f}, "
            f"Precision={best['precision']:.4f}, "
            f"Recall={best['recall']:.4f}, "
            f"Threshold={best['threshold']:.6f}, "
            f"Predicted fraction={best['predicted_fraction']:.4f}"
        )

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "filter_baseline_PA000005_crop.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "filter,dice,precision,recall,threshold,predicted_fraction\n"
        )

        for name, result in results.items():
            file.write(
                f"{name},"
                f"{result['dice']:.6f},"
                f"{result['precision']:.6f},"
                f"{result['recall']:.6f},"
                f"{result['threshold']:.8f},"
                f"{result['predicted_fraction']:.6f}\n"
            )

    print("\nSaved metrics:", output_path)
    print(
        "\nIMPORTANT: These are exploratory results from one crop. "
        "Thresholds were selected using the same crop being evaluated, "
        "so these scores are not held-out test performance."
    )
    print(
        "Gaussian and median intensity thresholding are crude "
        "baselines, not dedicated segmentation algorithms."
    )


if __name__ == "__main__":
    main()
