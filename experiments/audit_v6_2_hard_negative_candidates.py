
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd

from train_airrc_v6_multiclass import (
    load_v6_splits,
    load_v6_case,
)


# ============================================================
# NEUROVESSEL V6.2 — TRAINING-ONLY CANDIDATE AUDIT
#
# Counts background voxels eligible for intensity-stratified
# background patch sampling.
#
# Thresholds:
#   CT > -500
#   CT > -300
#   CT > 0
#
# IMPORTANT:
#   Only the 202 training patients are loaded.
#   Validation and locked test images/labels are not loaded.
#   No model inference or training.
#   No checkpoint or dataset modification.
#
# This is an intensity-based candidate audit, NOT model-based
# out-of-fold hard-negative mining.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics"
    / "airrc_v6_2_candidate_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [-500.0, -300.0, 0.0]

ROWS = []


def main():

    print("=" * 70)
    print("NEUROVESSEL V6.2 — TRAINING-ONLY CANDIDATE AUDIT")
    print("=" * 70)

    print("\nLoading patient split metadata...")

    train_records, validation_records, test_count = (
        load_v6_splits()
    )

    train_patients = {
        record["patient_id"]
        for record in train_records
    }

    validation_patients = {
        record["patient_id"]
        for record in validation_records
    }

    print("Training patients:", len(train_records))
    print("Validation patients:", len(validation_records))
    print("Locked test patients:", test_count)

    if len(train_records) != 202:
        raise RuntimeError(
            f"Expected 202 training patients, got {len(train_records)}"
        )

    if len(validation_records) != 26:
        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"got {len(validation_records)}"
        )

    if test_count != 26:
        raise RuntimeError(
            f"Expected 26 test patients, got {test_count}"
        )

    if train_patients & validation_patients:
        raise RuntimeError(
            "Train/validation patient leakage detected."
        )

    # The three previously investigated cases must not
    # enter this training-only candidate audit.
    protected_validation_patients = {
        "LIDC-IDRI-0529",
        "LIDC-IDRI-0763",
        "LIDC-IDRI-0848",
    }

    if protected_validation_patients & train_patients:
        raise RuntimeError(
            "A protected validation patient appears in training."
        )

    if not protected_validation_patients.issubset(
        validation_patients
    ):
        raise RuntimeError(
            "One or more diagnostic patients are not in validation."
        )

    print("\nSPLIT CHECK: PASSED")
    print("Only training records will be loaded.")
    print("Validation images and labels will NOT be loaded.")
    print("Locked test images and labels will NOT be loaded.")

    totals = {
        threshold: 0
        for threshold in THRESHOLDS
    }

    total_background = 0
    total_voxels = 0

    print("\nAuditing training patients...")

    for index, record in enumerate(
        train_records,
        start=1,
    ):

        patient = record["patient_id"]
        uid = record["uid"]

        print(
            f"\n[{index}/{len(train_records)}] {patient}"
        )

        # Loads only this training patient's CT and V6 label.
        image, labels = load_v6_case(record)

        if image.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: {patient}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Non-finite CT values: {patient}"
            )

        background = labels == 0

        background_count = int(
            np.count_nonzero(background)
        )

        voxel_count = int(image.size)

        total_background += background_count
        total_voxels += voxel_count

        row = {
            "patient_id": patient,
            "uid": uid,
            "volume_shape": str(image.shape),
            "total_voxels": voxel_count,
            "background_voxels": background_count,
            "background_fraction": (
                background_count / voxel_count
                if voxel_count else 0.0
            ),
            "background_intensity_median": (
                float(np.median(image[background]))
                if background_count else np.nan
            ),
        }

        for threshold in THRESHOLDS:

            # Candidate centres must be ground-truth background
            # and have CT intensity above the threshold.
            eligible = (
                background
                & (image > threshold)
            )

            count = int(
                np.count_nonzero(eligible)
            )

            row[
                f"background_centres_ct_gt_{int(threshold)}"
            ] = count

            row[
                f"eligible_fraction_of_background_gt_{int(threshold)}"
            ] = (
                count / background_count
                if background_count else 0.0
            )

            totals[threshold] += count

            print(
                f"  Background centres CT > {threshold:>6.0f}: "
                f"{count:,}"
            )

        ROWS.append(row)

        print(
            "  Total background voxels:",
            f"{background_count:,}",
        )

        # Release one patient before loading the next.
        del image, labels, background
        gc.collect()

    # --------------------------------------------------------
    # Save per-patient audit.
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR / "training_hard_negative_candidates.csv"
    )

    result = pd.DataFrame(ROWS)

    result.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    summary = {
        "experiment": "NeuroVessel V6.2 candidate audit",
        "data_scope": "training partition only",
        "training_patients": len(train_records),
        "validation_patients": len(validation_records),
        "locked_test_patients": test_count,
        "validation_images_loaded": False,
        "locked_test_images_loaded": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "thresholds": {},
    }

    print("\n" + "=" * 70)
    print("TRAINING-ONLY CANDIDATE AUDIT SUMMARY")
    print("=" * 70)

    print("Training patients:", len(train_records))

    print("Total CT voxels:", f"{total_voxels:,}")

    print("Total background voxels:", f"{total_background:,}")

    if total_voxels:
        print(
            "Overall background fraction:",
            f"{total_background / total_voxels:.4%}",
        )

    for threshold in THRESHOLDS:

        key = str(int(threshold))

        summary["thresholds"][key] = {
            "ct_threshold_exclusive": threshold,
            "eligible_background_centres": totals[threshold],
            "fraction_of_all_background": (
                totals[threshold] / total_background
                if total_background else 0.0
            ),
        }

        print(
            f"\nBackground centres with CT > {threshold:.0f}: "
            f"{totals[threshold]:,}"
        )

        print(
            "Fraction of all background:",
            f"{summary['thresholds'][key]['fraction_of_all_background']:.4%}",
        )

    summary_path = (
        OUTPUT_DIR / "training_hard_negative_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nOUTPUT FILES:")
    print(output_csv)
    print(summary_path)

    print("\nNo validation images or labels were loaded.")
    print("No locked test images or labels were loaded.")
    print("No model inference or training was performed.")

    print("\nAUDIT COMPLETED.")


if __name__ == "__main__":
    main()
