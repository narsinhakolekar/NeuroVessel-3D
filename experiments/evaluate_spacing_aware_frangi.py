
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import frangi


ROOT = Path("data/raw/train")
SPLIT_PATH = Path("data/splits.json")
OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with a small validation subset to check the pipeline.
# Expand to all 15 validation patients after this run succeeds.
MAX_PATIENTS = 15

# Isotropic resampling spacing in millimetres.
TARGET_SPACING_MM = 1.0

# Physical Gaussian scales in millimetres.
PHYSICAL_SIGMAS_MM = (1.0, 2.0, 3.0)

# Thresholds are selected on the first validation patient only.
# The remaining patients are diagnostic checks, NOT an independent test set.
THRESHOLD_QUANTILES = np.linspace(0.50, 0.995, 40)


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))

    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
    }


def load_patient(patient_id):
    patient_dir = ROOT / patient_id

    image_path = patient_dir / "image" / f"{patient_id}.nii.gz"
    label_path = patient_dir / "label" / f"{patient_id}.nii.gz"

    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing image or label for {patient_id}: "
            f"{image_path}, {label_path}"
        )

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(f"{patient_id}: image/label shape mismatch")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"{patient_id}: image/label affine mismatch")

    ct = np.asarray(image_file.dataobj, dtype=np.float32)
    target = np.asarray(label_file.dataobj) != 0

    spacing = tuple(float(v) for v in image_file.header.get_zooms()[:3])

    if not np.isfinite(ct).all():
        raise ValueError(f"{patient_id}: CT contains NaN or infinity")

    if not target.any():
        raise ValueError(f"{patient_id}: label contains no foreground")

    # Window CT intensities to [0, 1].
    ct = np.clip(ct, -1000.0, 500.0)
    ct = (ct + 1000.0) / 1500.0

    return ct, target, spacing


def isotropic_resample(ct, target, spacing):
    # Array-axis zoom factors convert original voxel spacing to 1 mm.
    zoom_factors = tuple(s / TARGET_SPACING_MM for s in spacing)

    ct_iso = ndimage.zoom(
        ct, zoom=zoom_factors, order=1, mode="nearest",
        prefilter=False
    )
    target_iso = ndimage.zoom(
        target.astype(np.uint8),
        zoom=zoom_factors,
        order=0,
        mode="nearest",
        prefilter=False,
    ) > 0

    # Align shapes defensively in case of rounding differences.
    common_shape = tuple(
        min(a, b) for a, b in zip(ct_iso.shape, target_iso.shape)
    )
    slices = tuple(slice(0, n) for n in common_shape)

    return ct_iso[slices], target_iso[slices]


def compute_frangi(ct_iso):
    responses = []

    for sigma_mm in PHYSICAL_SIGMAS_MM:
        response = frangi(
            ct_iso,
            sigmas=(sigma_mm,),
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=False,
            mode="reflect",
        )
        responses.append(np.asarray(response, dtype=np.float32))

    return np.maximum.reduce(responses)


def choose_threshold(response, target):
    positive = response[response > 0]

    if positive.size == 0:
        return None

    thresholds = np.unique(
        np.quantile(positive, THRESHOLD_QUANTILES)
    )

    best_threshold = None
    best_dice = -1.0

    for threshold in thresholds:
        metrics = calculate_metrics(response >= threshold, target)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = float(threshold)

    return best_threshold


def main():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    validation_ids = split["validation"][:MAX_PATIENTS]

    print("Validation patients:", validation_ids)
    print("Target isotropic spacing:", TARGET_SPACING_MM, "mm")
    print("Physical Frangi scales:", PHYSICAL_SIGMAS_MM)
    print("Loading and processing CT volumes...")

    patient_data = []

    for patient_id in validation_ids:
        ct, target, spacing = load_patient(patient_id)
        ct_iso, target_iso = isotropic_resample(ct, target, spacing)
        response = compute_frangi(ct_iso)

        patient_data.append({
            "patient_id": patient_id,
            "spacing_mm": spacing,
            "original_shape": ct.shape,
            "isotropic_shape": ct_iso.shape,
            "target": target_iso,
            "response": response,
        })

        print(
            f"{patient_id}: original={ct.shape}, "
            f"spacing={spacing}, isotropic={ct_iso.shape}, "
            f"target fraction={target_iso.mean():.6f}, "
            f"response max={response.max():.6f}"
        )

    if not patient_data:
        raise RuntimeError("No validation patients were processed")

    # Tune threshold on only the first validation patient.
    # This is exploratory tuning, not a held-out evaluation.
    tuning_patient = patient_data[0]
    threshold = choose_threshold(
        tuning_patient["response"], tuning_patient["target"]
    )

    if threshold is None:
        raise RuntimeError("Frangi returned no positive response")

    print(
        f"\nExploratory threshold tuned on "
        f"{tuning_patient['patient_id']}: {threshold:.8f}"
    )

    rows = []

    for item in patient_data:
        metrics = calculate_metrics(
            item["response"] >= threshold, item["target"]
        )

        row = {
            "patient_id": item["patient_id"],
            "original_shape": str(item["original_shape"]),
            "original_spacing_mm": str(item["spacing_mm"]),
            "isotropic_shape": str(item["isotropic_shape"]),
            "target_spacing_mm": TARGET_SPACING_MM,
            "threshold": threshold,
            **metrics,
        }
        rows.append(row)

        print(
            f"{item['patient_id']}: "
            f"Dice={metrics['dice']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Predicted fraction={metrics['predicted_fraction']:.4f}, "
            f"Target fraction={metrics['target_fraction']:.4f}"
        )

    csv_path = OUTPUT_DIR / "spacing_aware_frangi_validation.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "spacing-aware Frangi exploratory baseline",
        "validation_patients": validation_ids,
        "threshold_tuning_patient": tuning_patient["patient_id"],
        "threshold": threshold,
        "target_spacing_mm": TARGET_SPACING_MM,
        "physical_sigmas_mm": list(PHYSICAL_SIGMAS_MM),
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "warning": (
            "Small validation subset; threshold tuned on a validation "
            "patient. Results are exploratory and are not held-out test "
            "performance. Frangi is not artery-specific."
        ),
    }

    json_path = OUTPUT_DIR / "spacing_aware_frangi_validation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved CSV:", csv_path)
    print("Saved summary:", json_path)
    print("Mean Dice:", f"{summary['mean_dice']:.4f}")
    print("This is an exploratory baseline, not a final research result.")


if __name__ == "__main__":
    main()
