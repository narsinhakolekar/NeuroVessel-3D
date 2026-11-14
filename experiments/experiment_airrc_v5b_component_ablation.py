
"""
NEUROVESSEL V5B
CONNECTED-COMPONENT SIZE ABLATION

Purpose:
    Test whether removing small disconnected prediction
    components improves full-volume PA segmentation.

Evaluation:
    All 26 validation patients.
    26-connectivity.
    Prediction threshold: 0.70.

Component volume cutoffs:
    0, 10, 25, 50, 100, 250 mm3.

IMPORTANT:
    - Original predictions are never modified.
    - No filtered predictions are saved over baseline.
    - The model is not retrained.
    - The locked test partition is never loaded.
"""

import gc
import json
import time

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy import ndimage as ndi

from train_airrc_v5b import (
    load_splits,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.70

COMPONENT_CUTOFFS_MM3 = [
    0,
    10,
    25,
    50,
    100,
    250
]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v5b_fullvol"
    / "threshold_0.70"
)

OUTPUT_DIR = (
    METRICS_DIR
    / "airrc_v5b_component_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Same 26-connectivity used in the original audit.
STRUCTURE_26 = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(uid):

    import re

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(uid)
    )


def get_prediction_path(uid):

    filename = (
        safe_filename(uid)
        + "_pa_prediction.nii.gz"
    )

    return PREDICTION_DIR / filename


def calculate_metrics(prediction, target):

    intersection = int(
        np.count_nonzero(
            prediction & target
        )
    )

    predicted_voxels = int(
        np.count_nonzero(prediction)
    )

    target_voxels = int(
        np.count_nonzero(target)
    )

    fp = predicted_voxels - intersection

    fn = target_voxels - intersection

    dice = (
        2.0 * intersection
        / (predicted_voxels + target_voxels)
        if predicted_voxels + target_voxels > 0
        else 1.0
    )

    precision = (
        intersection / predicted_voxels
        if predicted_voxels > 0
        else 0.0
    )

    recall = (
        intersection / target_voxels
        if target_voxels > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": intersection,
        "fp": fp,
        "fn": fn,
        "predicted_voxels": predicted_voxels,
        "ground_truth_voxels": target_voxels
    }


# ============================================================
# ONE PATIENT
# ============================================================

def evaluate_patient(record):

    uid = record["uid"]

    print(
        "\nProcessing:",
        uid
    )

    prediction_path = get_prediction_path(uid)

    if not prediction_path.is_file():

        raise FileNotFoundError(
            f"Prediction missing: {prediction_path}"
        )

    # Load processed ground truth.
    gt_object = sitk.ReadImage(
        str(record["mask"])
    )

    gt = sitk.GetArrayFromImage(
        gt_object
    ) > 0

    # Load original prediction.
    pred_object = sitk.ReadImage(
        str(prediction_path)
    )

    pred = sitk.GetArrayFromImage(
        pred_object
    ) > 0

    # Verify geometry.
    geometry_matches = (
        gt_object.GetSize()
        == pred_object.GetSize()

        and gt_object.GetSpacing()
        == pred_object.GetSpacing()

        and gt_object.GetOrigin()
        == pred_object.GetOrigin()

        and gt_object.GetDirection()
        == pred_object.GetDirection()
    )

    if not geometry_matches:

        raise RuntimeError(
            f"Geometry mismatch: {uid}"
        )

    if gt.shape != pred.shape:

        raise RuntimeError(
            f"Array shape mismatch: {uid}"
        )

    spacing = gt_object.GetSpacing()

    # SimpleITK spacing is XYZ.
    # Arrays are ZYX; voxel volume is unchanged.
    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    # --------------------------------------------------------
    # Connected components of original prediction.
    # --------------------------------------------------------

    labels = np.zeros(
        pred.shape,
        dtype=np.int32
    )

    number_of_components = ndi.label(
        pred,
        structure=STRUCTURE_26,
        output=labels
    )

    component_sizes = np.bincount(
        labels.ravel()
    )[1:]

    original_metrics = calculate_metrics(
        pred,
        gt
    )

    original_component_count = int(
        number_of_components
    )

    print(
        "Original Dice:",
        round(original_metrics["dice"], 5)
    )

    print(
        "Original components:",
        original_component_count
    )

    rows = []

    # --------------------------------------------------------
    # Evaluate each minimum physical component size.
    # --------------------------------------------------------

    for cutoff_mm3 in COMPONENT_CUTOFFS_MM3:

        # Component index zero represents background.
        keep_component = np.zeros(
            number_of_components + 1,
            dtype=bool
        )

        # Keep all components above the physical
        # volume cutoff.
        component_volumes_mm3 = (
            component_sizes * voxel_volume_mm3
        )

        keep_component[1:] = (
            component_volumes_mm3 >= cutoff_mm3
        )

        # Create a temporary filtered prediction.
        # This does NOT modify the original NIfTI.
        filtered_prediction = (
            keep_component[labels]
        )

        metrics = calculate_metrics(
            filtered_prediction,
            gt
        )

        remaining_components = int(
            np.count_nonzero(
                keep_component[1:]
            )
        )

        removed_components = (
            original_component_count
            - remaining_components
        )

        removed_voxels = (
            original_metrics["predicted_voxels"]
            - metrics["predicted_voxels"]
        )

        removed_fraction = (
            removed_voxels
            / original_metrics["predicted_voxels"]
            if original_metrics["predicted_voxels"]
            else 0.0
        )

        row = {

            "uid": uid,

            "minimum_component_volume_mm3": (
                cutoff_mm3
            ),

            "geometry_matches": geometry_matches,

            "original_component_count": (
                original_component_count
            ),

            "remaining_components": (
                remaining_components
            ),

            "removed_components": (
                removed_components
            ),

            "removed_prediction_voxels": (
                removed_voxels
            ),

            "removed_prediction_fraction": (
                removed_fraction
            ),

            "voxel_volume_mm3": (
                voxel_volume_mm3
            ),

            **metrics

        }

        rows.append(row)

        print(
            f"Cutoff={cutoff_mm3:6.1f} mm3 | "
            f"Dice={metrics['dice']:.5f} | "
            f"Precision={metrics['precision']:.5f} | "
            f"Recall={metrics['recall']:.5f} | "
            f"Remaining components={remaining_components}"
        )

        del filtered_prediction
        del keep_component

    del labels
    del component_sizes
    del component_volumes_mm3

    del gt
    del pred
    del gt_object
    del pred_object

    gc.collect()

    return rows


# ============================================================
# AGGREGATION
# ============================================================

def summarize_results(df):

    summaries = []

    for cutoff in COMPONENT_CUTOFFS_MM3:

        subset = df[
            df["minimum_component_volume_mm3"]
            == cutoff
        ]

        total_tp = int(
            subset["tp"].sum()
        )

        total_fp = int(
            subset["fp"].sum()
        )

        total_fn = int(
            subset["fn"].sum()
        )

        total_pred = int(
            subset["predicted_voxels"].sum()
        )

        total_gt = int(
            subset["ground_truth_voxels"].sum()
        )

        micro_dice = (
            2.0 * total_tp
            / (total_pred + total_gt)
            if total_pred + total_gt
            else 1.0
        )

        micro_precision = (
            total_tp / total_pred
            if total_pred
            else 0.0
        )

        micro_recall = (
            total_tp / total_gt
            if total_gt
            else 0.0
        )

        summaries.append({

            "minimum_component_volume_mm3": cutoff,

            "patients": len(subset),

            "macro_dice": float(
                subset["dice"].mean()
            ),

            "dice_std": float(
                subset["dice"].std(ddof=1)
            ),

            "macro_precision": float(
                subset["precision"].mean()
            ),

            "macro_recall": float(
                subset["recall"].mean()
            ),

            "micro_dice": float(micro_dice),

            "micro_precision": float(
                micro_precision
            ),

            "micro_recall": float(
                micro_recall
            ),

            "total_tp": total_tp,

            "total_fp": total_fp,

            "total_fn": total_fn,

            "mean_remaining_components": float(
                subset["remaining_components"].mean()
            ),

            "mean_removed_components": float(
                subset["removed_components"].mean()
            ),

            "mean_removed_prediction_fraction": float(
                subset["removed_prediction_fraction"].mean()
            )

        })

    return pd.DataFrame(summaries)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V5B — COMPONENT SIZE ABLATION"
    )

    print("=" * 70)

    print(
        "Prediction threshold:",
        THRESHOLD
    )

    print(
        "Component connectivity: 26"
    )

    print(
        "Component volume cutoffs:",
        COMPONENT_CUTOFFS_MM3
    )

    print(
        "Original predictions will not be modified."
    )

    # --------------------------------------------------------
    # Validation split only.
    # --------------------------------------------------------

    _, validation_records, test_count = load_splits()

    if len(validation_records) != 26:

        raise RuntimeError(
            f"Expected 26 validation cases, "
            f"got {len(validation_records)}"
        )

    if test_count != 26:

        raise RuntimeError(
            f"Expected 26 locked test cases, "
            f"got {test_count}"
        )

    print("\nDATASET")

    print(
        "Validation patients:",
        len(validation_records)
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "Test images and masks will not be loaded."
    )

    # Check all predictions before execution.
    missing = [
        str(get_prediction_path(r["uid"]))
        for r in validation_records
        if not get_prediction_path(r["uid"]).is_file()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing predictions:\n"
            + "\n".join(missing[:10])
        )

    # --------------------------------------------------------
    # Run all patients.
    # --------------------------------------------------------

    all_rows = []

    for index, record in enumerate(
        validation_records,
        start=1
    ):

        print(
            f"\nPATIENT {index}/{len(validation_records)}"
        )

        patient_rows = evaluate_patient(
            record
        )

        all_rows.extend(
            patient_rows
        )

        print(
            f"Completed {index}/"
            f"{len(validation_records)}",
            flush=True
        )

    # --------------------------------------------------------
    # Per-patient results.
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        all_rows
    )

    patient_path = (
        OUTPUT_DIR
        / "per_patient_component_ablation.csv"
    )

    patient_df.to_csv(
        patient_path,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate results.
    # --------------------------------------------------------

    summary_df = summarize_results(
        patient_df
    )

    summary_path = (
        OUTPUT_DIR
        / "component_ablation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Identify best cutoff on validation.
    # --------------------------------------------------------

    best_row = summary_df.loc[
        summary_df["macro_dice"].idxmax()
    ]

    baseline_row = summary_df.loc[
        summary_df[
            "minimum_component_volume_mm3"
        ] == 0
    ].iloc[0]

    metadata = {

        "project": "NeuroVessel",

        "model": "V5B Residual 3D U-Net",

        "experiment": "Connected-component size ablation",

        "threshold": THRESHOLD,

        "connectivity": 26,

        "validation_patients": 26,

        "locked_test_patients": test_count,

        "test_partition_loaded": False,

        "component_cutoffs_mm3": (
            COMPONENT_CUTOFFS_MM3
        ),

        "original_prediction_modified": False,

        "component_filtering_applied_to_saved_masks": False,

        "baseline_macro_dice": float(
            baseline_row["macro_dice"]
        ),

        "best_validation_cutoff_mm3": float(
            best_row[
                "minimum_component_volume_mm3"
            ]
        ),

        "best_validation_macro_dice": float(
            best_row["macro_dice"]
        ),

        "best_validation_micro_dice": float(
            best_row["micro_dice"]
        ),

        "runtime_seconds": (
            time.time() - start_time
        ),

        "warning": (
            "Component filtering may remove genuine small "
            "pulmonary artery branches. Results are exploratory "
            "and require visual review."
        )

    }

    metadata_path = (
        OUTPUT_DIR
        / "component_ablation_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "COMPONENT SIZE ABLATION RESULTS"
    )

    print("=" * 75)

    display_columns = [

        "minimum_component_volume_mm3",

        "macro_dice",

        "micro_dice",

        "macro_precision",

        "macro_recall",

        "mean_remaining_components",

        "mean_removed_prediction_fraction"

    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nBEST VALIDATION CUTOFF")

    print(
        "Minimum component volume:",
        best_row[
            "minimum_component_volume_mm3"
        ],
        "mm3"
    )

    print(
        "Macro Dice:",
        round(
            best_row["macro_dice"],
            5
        )
    )

    print(
        "Baseline macro Dice:",
        round(
            baseline_row["macro_dice"],
            5
        )
    )

    print("\nOUTPUTS")

    print(
        "Per-patient results:",
        patient_path
    )

    print(
        "Aggregate summary:",
        summary_path
    )

    print(
        "Metadata:",
        metadata_path
    )

    print(
        "\nOriginal saved prediction masks remain unchanged."
    )

    print(
        "No model parameters were modified."
    )

    print(
        "Locked test set remains untouched."
    )

    print(
        "Runtime:",
        round(
            metadata["runtime_seconds"],
            1
        ),
        "seconds"
    )


if __name__ == "__main__":

    main()
