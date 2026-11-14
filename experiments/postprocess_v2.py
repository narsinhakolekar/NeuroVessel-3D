
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet


DATA_ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT = Path("outputs/checkpoints/unet_v2_best.pt")
OUTPUT_DIR = Path("outputs/metrics")

THRESHOLD = 0.7
PATCH_SIZE = 64
MIN_COMPONENT_SIZES = [10, 100, 500]


def load_case(patient_id):
    folder = DATA_ROOT / patient_id

    image_file = nib.load(
        str(folder / "image" / f"{patient_id}.nii.gz")
    )
    label_file = nib.load(
        str(folder / "label" / f"{patient_id}.nii.gz")
    )

    if image_file.shape != label_file.shape:
        raise ValueError(f"Shape mismatch: {patient_id}")

    if not np.allclose(
        image_file.affine, label_file.affine, atol=1e-4
    ):
        raise ValueError(f"Affine mismatch: {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    target = np.asarray(label_file.dataobj) != 0

    if not np.isfinite(image).all() or not target.any():
        raise ValueError(f"Invalid image or label: {patient_id}")

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    return image, target


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
        "predicted_voxels": int(predicted_count),
    }


def connected_component_predictions(binary_prediction):
    # Label 26-connected components in 3D.
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, number = ndimage.label(
        binary_prediction, structure=structure
    )

    if number == 0:
        empty = np.zeros_like(binary_prediction, dtype=bool)
        return {"largest_component": empty}

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0

    largest_label = int(np.argmax(component_sizes))
    results = {
        "largest_component": labels == largest_label
    }

    for minimum_size in MIN_COMPONENT_SIZES:
        # Keep all components meeting this size threshold.
        keep_labels = np.flatnonzero(
            component_sizes >= minimum_size
        )
        keep_labels = keep_labels[keep_labels != 0]

        results[f"min_size_{minimum_size}"] = np.isin(
            labels, keep_labels
        )

    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not SPLIT_FILE.exists() or not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing split file or model checkpoint."
        )

    device = torch.device("cuda")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]
    test_ids = set(split["test"])

    if set(validation_ids) & test_ids:
        raise RuntimeError("Validation/test leakage detected.")

    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample="nontrainable",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("GPU:", torch.cuda.get_device_name(0))
    print("Validation patients:", len(validation_ids))
    print("Threshold:", THRESHOLD)
    print("Test set will not be evaluated.")

    rows = []

    for index, patient_id in enumerate(validation_ids, start=1):
        image, target = load_case(patient_id)

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                image[None, None], dtype=np.float32
            )
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

            probabilities = (
                torch.sigmoid(logits)[0, 0].float().numpy()
            )

        raw_prediction = probabilities >= THRESHOLD
        predictions = {
            "raw": raw_prediction,
            **connected_component_predictions(raw_prediction),
        }

        print(
            f"\n[{index}/{len(validation_ids)}] {patient_id}"
        )

        for method, prediction in predictions.items():
            metrics = calculate_metrics(prediction, target)
            row = {
                "patient_id": patient_id,
                "threshold": THRESHOLD,
                "method": method,
                **metrics,
            }
            rows.append(row)

            print(
                f"  {method:20s} "
                f"Dice={metrics['dice']:.4f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f}"
            )

        del image, target, tensor, logits, probabilities
        del raw_prediction, predictions
        torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "v2_postprocessing_validation.csv"
    summary_path = (
        OUTPUT_DIR / "v2_postprocessing_summary.json"
    )

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    methods = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []

    for method in methods:
        subset = [row for row in rows if row["method"] == method]

        summaries.append({
            "method": method,
            "patients": len(subset),
            "mean_dice": float(
                np.mean([row["dice"] for row in subset])
            ),
            "median_dice": float(
                np.median([row["dice"] for row in subset])
            ),
            "mean_precision": float(
                np.mean([row["precision"] for row in subset])
            ),
            "mean_recall": float(
                np.mean([row["recall"] for row in subset])
            ),
            "mean_predicted_fraction": float(
                np.mean([
                    row["predicted_fraction"] for row in subset
                ])
            ),
        })

    summary_path.write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print("\n=== VALIDATION SUMMARY ===")
    for item in summaries:
        print(
            f"{item['method']:20s} "
            f"Dice={item['mean_dice']:.4f} "
            f"Precision={item['mean_precision']:.4f} "
            f"Recall={item['mean_recall']:.4f} "
            f"Predicted fraction="
            f"{item['mean_predicted_fraction']:.4f}"
        )

    raw = next(item for item in summaries if item["method"] == "raw")
    best = max(summaries, key=lambda item: item["mean_dice"])

    print("\nBest tested method:", best["method"])
    print("Raw baseline mean Dice:", round(raw["mean_dice"], 4))
    print("Best mean Dice:", round(best["mean_dice"], 4))
    print("Dice change:", round(best["mean_dice"] - raw["mean_dice"], 4))
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
