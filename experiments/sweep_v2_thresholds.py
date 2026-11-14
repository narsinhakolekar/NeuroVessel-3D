
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet

DATA_ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT = Path("outputs/checkpoints/unet_v2_best.pt")
OUTPUT_DIR = Path("outputs/metrics")
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PATCH_SIZE = 64


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

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
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
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

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
    print("Evaluating validation patients only.")

    rows = []

    for index, patient_id in enumerate(validation_ids, start=1):
        image, target = load_case(patient_id)
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

        print(f"{index}/{len(validation_ids)}: {patient_id}")

        for threshold in THRESHOLDS:
            metrics = calculate_metrics(probabilities >= threshold, target)
            rows.append({
                "patient_id": patient_id,
                "threshold": threshold,
                **metrics,
            })

        del image, target, tensor, logits, probabilities
        torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "v2_threshold_sweep.csv"
    summary_path = OUTPUT_DIR / "v2_threshold_sweep_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for threshold in THRESHOLDS:
        subset = [r for r in rows if r["threshold"] == threshold]
        summaries.append({
            "threshold": threshold,
            "mean_dice": float(np.mean([r["dice"] for r in subset])),
            "median_dice": float(np.median([r["dice"] for r in subset])),
            "mean_precision": float(np.mean([r["precision"] for r in subset])),
            "mean_recall": float(np.mean([r["recall"] for r in subset])),
            "mean_predicted_fraction": float(
                np.mean([r["predicted_fraction"] for r in subset])
            ),
            "mean_target_fraction": float(
                np.mean([r["target_fraction"] for r in subset])
            ),
        })

    summary_path.write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print("\nThreshold results:")
    for item in summaries:
        print(
            f"Threshold={item['threshold']:.1f} | "
            f"Dice={item['mean_dice']:.4f} | "
            f"Precision={item['mean_precision']:.4f} | "
            f"Recall={item['mean_recall']:.4f} | "
            f"Predicted fraction={item['mean_predicted_fraction']:.4f}"
        )

    best = max(summaries, key=lambda item: item["mean_dice"])
    print("\nBest tested validation threshold:", best["threshold"])
    print("Mean Dice at that threshold:", round(best["mean_dice"], 4))
    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Test set remains untouched.")


if __name__ == "__main__":
    main()
