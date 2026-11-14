
import csv
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet
from tqdm import tqdm


ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
CHECKPOINT_PATH = Path("outputs/checkpoints/unet_interp_best.pt")

OUTPUT_DIR = Path("outputs/metrics")
CSV_PATH = OUTPUT_DIR / "interp_full_volume_validation.csv"
SUMMARY_PATH = OUTPUT_DIR / "interp_full_volume_summary.json"

PATCH_SIZE = 64
OVERLAP = 0.25
THRESHOLD = 0.5


def load_case(patient_id):
    folder = ROOT / patient_id

    image_path = folder / "image" / f"{patient_id}.nii.gz"
    label_path = folder / "label" / f"{patient_id}.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    image_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    if image_file.shape != label_file.shape:
        raise ValueError(f"Shape mismatch for {patient_id}")

    if not np.allclose(image_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch for {patient_id}")

    image = np.asarray(image_file.dataobj, dtype=np.float32)
    label = np.asarray(label_file.dataobj) != 0

    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite CT values in {patient_id}")

    if not label.any():
        raise ValueError(f"Empty ground-truth mask for {patient_id}")

    image = np.clip(image, -1000, 500)
    image = (image + 1000.0) / 1500.0

    return image, label


def calculate_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    intersection = int(np.logical_and(prediction, target).sum())
    predicted_count = int(prediction.sum())
    target_count = int(target.sum())

    denominator = predicted_count + target_count

    dice = (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )

    precision = (
        intersection / predicted_count
        if predicted_count > 0
        else 0.0
    )

    recall = (
        intersection / target_count
        if target_count > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_voxels": predicted_count,
        "ground_truth_voxels": target_count,
        "predicted_foreground_fraction": float(prediction.mean()),
        "ground_truth_foreground_fraction": float(target.mean()),
    }


def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # These settings must match experiments/train_unet_interp.py.
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(8, 16, 32, 64, 128, 8),
        upsample="nontrainable",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    return model


def predict_full_volume(model, image, device):
    # NIfTI array axes are preserved; input shape is [B, C, X, Y, Z].
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image[None, None, ...])
    ).to(device)

    probability_chunks = []

    # Sliding-window inference limits GPU memory use.
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            probabilities = sliding_window_inference(
                inputs=image_tensor,
                roi_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                sw_batch_size=1,
                predictor=model,
                overlap=OVERLAP,
                mode="constant",
            )

            probabilities = torch.sigmoid(probabilities)

        probabilities = probabilities[0, 0].float().cpu().numpy()

    del image_tensor

    return probabilities


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the neurovessel environment."
        )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Inference patch:", PATCH_SIZE)
    print("Sliding-window overlap:", OVERLAP)
    print("Prediction threshold:", THRESHOLD)

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    validation_ids = split["validation"]
    test_ids = set(split["test"])

    if not validation_ids:
        raise ValueError("Validation split is empty.")

    if set(validation_ids) & test_ids:
        raise RuntimeError("Validation/test patient leakage detected.")

    print("Validation patients:", len(validation_ids))
    print("Test patients remain untouched.")

    model = load_model(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for patient_id in tqdm(validation_ids, desc="Validation patients"):
        start_time = time.time()

        image, target = load_case(patient_id)

        probabilities = predict_full_volume(
            model,
            image,
            device,
        )

        if probabilities.shape != target.shape:
            raise ValueError(
                f"Prediction shape mismatch for {patient_id}: "
                f"{probabilities.shape} vs {target.shape}"
            )

        prediction = probabilities >= THRESHOLD
        metrics = calculate_metrics(prediction, target)

        row = {
            "patient_id": patient_id,
            "threshold": THRESHOLD,
            "volume_shape": "x".join(map(str, target.shape)),
            **metrics,
            "inference_seconds": round(time.time() - start_time, 2),
        }

        rows.append(row)

        print(
            f"\n{patient_id} | "
            f"Dice={metrics['dice']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Predicted fraction="
            f"{metrics['predicted_foreground_fraction']:.4f}"
        )

        del image, target, probabilities, prediction

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Save progress after each patient so completed evaluations persist.
        with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    dice_values = [row["dice"] for row in rows]
    precision_values = [row["precision"] for row in rows]
    recall_values = [row["recall"] for row in rows]

    summary = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        ).get("epoch", "unknown"),
        "validation_patient_count": len(rows),
        "threshold": THRESHOLD,
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "mean_patient_dice": float(np.mean(dice_values)),
        "std_patient_dice": float(np.std(dice_values)),
        "median_patient_dice": float(np.median(dice_values)),
        "mean_patient_precision": float(np.mean(precision_values)),
        "mean_patient_recall": float(np.mean(recall_values)),
        "minimum_patient_dice": float(np.min(dice_values)),
        "maximum_patient_dice": float(np.max(dice_values)),
        "patient_results": rows,
        "test_set_evaluated": False,
        "evaluation_type": "full-volume sliding-window validation",
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FULL-VOLUME VALIDATION SUMMARY")
    print("=" * 60)
    print("Patients evaluated:", len(rows))
    print(f"Mean patient Dice:      {summary['mean_patient_dice']:.4f}")
    print(f"Median patient Dice:    {summary['median_patient_dice']:.4f}")
    print(f"Std patient Dice:       {summary['std_patient_dice']:.4f}")
    print(f"Mean patient precision: {summary['mean_patient_precision']:.4f}")
    print(f"Mean patient recall:    {summary['mean_patient_recall']:.4f}")
    print(f"Minimum patient Dice:   {summary['minimum_patient_dice']:.4f}")
    print(f"Maximum patient Dice:   {summary['maximum_patient_dice']:.4f}")
    print("\nPer-patient CSV:", CSV_PATH)
    print("Summary JSON:", SUMMARY_PATH)
    print("Test patients were not evaluated.")


if __name__ == "__main__":
    main()
