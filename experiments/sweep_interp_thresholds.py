
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_interp_validation import (
    CHECKPOINT_PATH,
    SPLIT_FILE,
    OUTPUT_DIR,
    load_case,
    load_model,
    predict_full_volume,
    calculate_metrics,
)

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
CSV_PATH = OUTPUT_DIR / "interp_threshold_sweep.csv"
SUMMARY_PATH = OUTPUT_DIR / "interp_threshold_sweep_summary.json"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    validation_ids = split["validation"]

    if not validation_ids:
        raise ValueError("Validation split is empty.")

    if set(validation_ids) & set(split["test"]):
        raise RuntimeError("Validation/test leakage detected.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Validation patients:", len(validation_ids))
    print("Thresholds:", THRESHOLDS)
    print("Test patients remain untouched.")

    model = load_model(device)
    rows = []

    for patient_id in tqdm(validation_ids, desc="Full-volume inference"):
        image, target = load_case(patient_id)
        probabilities = predict_full_volume(model, image, device)

        if probabilities.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {patient_id}: "
                f"{probabilities.shape} vs {target.shape}"
            )

        for threshold in THRESHOLDS:
            prediction = probabilities >= threshold
            metrics = calculate_metrics(prediction, target)

            rows.append({
                "patient_id": patient_id,
                "threshold": threshold,
                **metrics,
            })

        del image, target, probabilities

        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}

    for threshold in THRESHOLDS:
        group = [r for r in rows if r["threshold"] == threshold]

        summary[str(threshold)] = {
            "mean_dice": float(np.mean([r["dice"] for r in group])),
            "median_dice": float(np.median([r["dice"] for r in group])),
            "mean_precision": float(
                np.mean([r["precision"] for r in group])
            ),
            "mean_recall": float(np.mean([r["recall"] for r in group])),
            "mean_predicted_fraction": float(
                np.mean([r["predicted_foreground_fraction"] for r in group])
            ),
            "mean_ground_truth_fraction": float(
                np.mean([r["ground_truth_foreground_fraction"] for r in group])
            ),
        }

    result = {
        "checkpoint": str(CHECKPOINT_PATH),
        "validation_patient_count": len(validation_ids),
        "thresholds": list(THRESHOLDS),
        "results": summary,
        "test_set_evaluated": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nFULL-VOLUME THRESHOLD SWEEP")
    print("-" * 88)
    print(
        f"{'Threshold':>9} {'Dice':>9} {'Precision':>11} "
        f"{'Recall':>9} {'Pred. frac.':>12} {'GT frac.':>10}"
    )

    for threshold in THRESHOLDS:
        m = summary[str(threshold)]
        print(
            f"{threshold:>9.2f} "
            f"{m['mean_dice']:>9.4f} "
            f"{m['mean_precision']:>11.4f} "
            f"{m['mean_recall']:>9.4f} "
            f"{m['mean_predicted_fraction']:>12.4f} "
            f"{m['mean_ground_truth_fraction']:>10.4f}"
        )

    print("\nPer-patient results:", CSV_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Test patients were not evaluated.")


if __name__ == "__main__":
    main()
