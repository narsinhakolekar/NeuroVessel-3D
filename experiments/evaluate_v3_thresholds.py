import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_unet_v3_frangi import (
    BEST_PATH,
    SPLIT_FILE,
    build_model,
    calculate_metrics,
    load_case,
    predict_volume,
)

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_CSV = OUTPUT_DIR / "unet_v3_frangi_threshold_sweep.csv"


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    validation_ids = split["validation"]

    print("Validation patients:", len(validation_ids))
    print("Test patients will NOT be evaluated.")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    # Load best V3 checkpoint
    checkpoint = torch.load(
        BEST_PATH,
        map_location=device,
        weights_only=False
    )

    model = build_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])

    all_results = []

    # One prediction per patient; reuse probabilities for every threshold.
    for i, patient_id in enumerate(validation_ids, start=1):

        print(f"\nPatient {i}/{len(validation_ids)}: {patient_id}")

        image, frangi, target = load_case(patient_id)

        probabilities = predict_volume(
            model,
            image,
            frangi,
            device
        )

        print(
            "Probability range:",
            float(probabilities.min()),
            float(probabilities.max())
        )

        for threshold in THRESHOLDS:

            prediction = probabilities >= threshold

            metrics = calculate_metrics(
                prediction,
                target
            )

            row = {
                "patient_id": patient_id,
                "threshold": threshold,
                **metrics
            }

            all_results.append(row)

        del image, frangi, target, probabilities
        torch.cuda.empty_cache()

    # Macro-average metrics across patients.
    summary = []

    for threshold in THRESHOLDS:

        rows = [
            r for r in all_results
            if r["threshold"] == threshold
        ]

        summary.append({
            "threshold": threshold,
            "mean_dice": np.mean([r["dice"] for r in rows]),
            "mean_precision": np.mean([r["precision"] for r in rows]),
            "mean_recall": np.mean([r["recall"] for r in rows]),
            "mean_predicted_fraction": np.mean(
                [r["predicted_fraction"] for r in rows]
            ),
            "mean_target_fraction": np.mean(
                [r["target_fraction"] for r in rows]
            )
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== THRESHOLD SWEEP ==========")

    print(
        f"{'Threshold':>10} "
        f"{'Dice':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Predicted %':>14}"
    )

    for r in summary:

        print(
            f"{r['threshold']:>10.2f} "
            f"{r['mean_dice']:>10.4f} "
            f"{r['mean_precision']:>12.4f} "
            f"{r['mean_recall']:>10.4f} "
            f"{r['mean_predicted_fraction'] * 100:>13.3f}%"
        )

    best = max(summary, key=lambda x: x["mean_dice"])

    print("\nBest threshold:", best["threshold"])
    print("Best mean Dice:", best["mean_dice"])

    print("\nSaved results:", OUTPUT_CSV)
    print("Test patients were not evaluated.")


if __name__ == "__main__":
    main()