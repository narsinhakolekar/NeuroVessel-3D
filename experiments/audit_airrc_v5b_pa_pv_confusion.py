
"""
NEUROVESSEL V5B
PULMONARY ARTERY vs PULMONARY VEIN CONFUSION AUDIT

Evaluates all 26 validation patients.

Original AirRC labels:
    0 = Background
    1 = Airway lumen
    2 = Airway wall
    3 = Pulmonary artery
    4 = Pulmonary vein

V5B:
    Binary PA prediction
    Threshold = 0.70

Measurements:
    1. PA true positives
    2. PA false negatives
    3. False positives overlapping PV
    4. False positives overlapping airway lumen
    5. False positives overlapping airway wall
    6. False positives overlapping background
    7. Fraction of false positives attributable to PV
    8. Fraction of PV voxels predicted as PA
    9. Patient-level and pooled metrics

SAFETY:
    - Validation partition only.
    - Locked test images and masks are not loaded.
    - Original annotations are never modified.
    - Original predictions are never modified.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import gc
import json
import time

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from train_airrc_v5b import (
    load_splits,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

THRESHOLD = 0.70

ANNOTATION_DIR = Path(
    r"C:\Users\narsi\Downloads\AirRC_annotations_extracted\labelsTr"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v5b_fullvol"
    / "threshold_0.70"
)

OUTPUT_DIR = (
    METRICS_DIR
    / "airrc_v5b_pa_pv_confusion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = {
    0: "Background",
    1: "Airway_Lumen",
    2: "Airway_Wall",
    3: "Pulmonary_Artery",
    4: "Pulmonary_Vein"
}


# ============================================================
# 3. HELPERS
# ============================================================

def safe_filename(uid):

    import re

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(uid)
    )


def get_annotation_path(uid):

    return (
        ANNOTATION_DIR
        / f"{uid}.nii.gz"
    )


def get_prediction_path(uid):

    filename = (
        safe_filename(uid)
        + "_pa_prediction.nii.gz"
    )

    return PREDICTION_DIR / filename


def geometry_matches(a, b):

    size_ok = (
        a.GetSize() == b.GetSize()
    )

    spacing_ok = np.allclose(
        a.GetSpacing(),
        b.GetSpacing(),
        atol=1e-5
    )

    origin_ok = np.allclose(
        a.GetOrigin(),
        b.GetOrigin(),
        atol=1e-4
    )

    direction_ok = np.allclose(
        a.GetDirection(),
        b.GetDirection(),
        atol=1e-5
    )

    return bool(
        size_ok
        and spacing_ok
        and origin_ok
        and direction_ok
    )


def safe_divide(a, b):

    return float(a / b) if b else 0.0


# ============================================================
# 4. AUDIT ONE PATIENT
# ============================================================

def audit_patient(record):

    uid = record["uid"]

    print("\n" + "=" * 65)

    print(
        "PATIENT:",
        uid
    )

    annotation_path = get_annotation_path(uid)

    prediction_path = get_prediction_path(uid)

    if not annotation_path.is_file():

        raise FileNotFoundError(
            f"Original annotation missing: {annotation_path}"
        )

    if not prediction_path.is_file():

        raise FileNotFoundError(
            f"V5B prediction missing: {prediction_path}"
        )

    # --------------------------------------------------------
    # Load original four-class annotations.
    # --------------------------------------------------------

    annotation_obj = sitk.ReadImage(
        str(annotation_path)
    )

    labels = sitk.GetArrayFromImage(
        annotation_obj
    )

    # --------------------------------------------------------
    # Load V5B prediction.
    # --------------------------------------------------------

    prediction_obj = sitk.ReadImage(
        str(prediction_path)
    )

    prediction = (
        sitk.GetArrayFromImage(
            prediction_obj
        ) > 0
    )

    # --------------------------------------------------------
    # Load processed PA mask and CT headers for verification.
    # --------------------------------------------------------

    pa_obj = sitk.ReadImage(
        str(record["mask"])
    )

    ct_obj = sitk.ReadImage(
        str(record["image"])
    )

    processed_pa = (
        sitk.GetArrayFromImage(
            pa_obj
        ) > 0
    )

    geometry_annotation_ct = geometry_matches(
        annotation_obj,
        ct_obj
    )

    geometry_annotation_pa = geometry_matches(
        annotation_obj,
        pa_obj
    )

    geometry_annotation_prediction = geometry_matches(
        annotation_obj,
        prediction_obj
    )

    if not (
        geometry_annotation_ct
        and geometry_annotation_pa
        and geometry_annotation_prediction
    ):

        raise RuntimeError(
            f"Geometry mismatch for {uid}"
        )

    if not (
        labels.shape
        == prediction.shape
        == processed_pa.shape
    ):

        raise RuntimeError(
            f"Array shape mismatch for {uid}"
        )

    # --------------------------------------------------------
    # Confirm original PA class matches processed PA.
    # --------------------------------------------------------

    pa = labels == 3

    pv = labels == 4

    airway_lumen = labels == 1

    airway_wall = labels == 2

    background = labels == 0

    pa_mismatch = int(
        np.count_nonzero(
            pa != processed_pa
        )
    )

    if pa_mismatch != 0:

        raise RuntimeError(
            f"PA annotation mismatch for {uid}: "
            f"{pa_mismatch} voxels"
        )

    # --------------------------------------------------------
    # Check label values.
    # --------------------------------------------------------

    unique_classes = np.unique(labels)

    unknown_classes = [
        int(value)
        for value in unique_classes
        if int(value) not in CLASS_NAMES
    ]

    if unknown_classes:

        raise RuntimeError(
            f"Unexpected annotation classes for {uid}: "
            f"{unknown_classes}"
        )

    # --------------------------------------------------------
    # Count all original annotation classes.
    # --------------------------------------------------------

    class_voxels = {
        class_id: int(
            np.count_nonzero(
                labels == class_id
            )
        )
        for class_id in CLASS_NAMES
    }

    # --------------------------------------------------------
    # Predicted PA overlap with each true class.
    # --------------------------------------------------------

    predicted_pa_voxels = int(
        np.count_nonzero(prediction)
    )

    pred_on_background = int(
        np.count_nonzero(
            prediction & background
        )
    )

    pred_on_airway_lumen = int(
        np.count_nonzero(
            prediction & airway_lumen
        )
    )

    pred_on_airway_wall = int(
        np.count_nonzero(
            prediction & airway_wall
        )
    )

    pred_on_pa = int(
        np.count_nonzero(
            prediction & pa
        )
    )

    pred_on_pv = int(
        np.count_nonzero(
            prediction & pv
        )
    )

    # --------------------------------------------------------
    # Consistency check:
    # Every predicted voxel must belong to exactly one
    # original annotation class.
    # --------------------------------------------------------

    attributed_prediction_voxels = (
        pred_on_background
        + pred_on_airway_lumen
        + pred_on_airway_wall
        + pred_on_pa
        + pred_on_pv
    )

    if (
        attributed_prediction_voxels
        != predicted_pa_voxels
    ):

        raise RuntimeError(
            f"Class attribution does not sum to prediction "
            f"for {uid}"
        )

    # --------------------------------------------------------
    # False-positive and false-negative metrics.
    # --------------------------------------------------------

    total_pa_voxels = class_voxels[3]

    total_pv_voxels = class_voxels[4]

    tp = pred_on_pa

    fp = (
        predicted_pa_voxels - tp
    )

    fn = (
        total_pa_voxels - tp
    )

    tn = (
        labels.size
        - tp
        - fp
        - fn
    )

    # Class-specific FP attribution.
    fp_sum = (
        pred_on_background
        + pred_on_airway_lumen
        + pred_on_airway_wall
        + pred_on_pv
    )

    if fp_sum != fp:

        raise RuntimeError(
            f"False-positive attribution mismatch for {uid}"
        )

    dice = safe_divide(
        2 * tp,
        2 * tp + fp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    # --------------------------------------------------------
    # PA-PV confusion measures.
    # --------------------------------------------------------

    pv_overlap_fraction_of_fp = safe_divide(
        pred_on_pv,
        fp
    )

    pv_overlap_fraction_of_all_predictions = safe_divide(
        pred_on_pv,
        predicted_pa_voxels
    )

    pv_voxel_prediction_rate = safe_divide(
        pred_on_pv,
        total_pv_voxels
    )

    # Fraction of all false positives contributed
    # by each non-PA class.
    background_fp_share = safe_divide(
        pred_on_background,
        fp
    )

    airway_lumen_fp_share = safe_divide(
        pred_on_airway_lumen,
        fp
    )

    airway_wall_fp_share = safe_divide(
        pred_on_airway_wall,
        fp
    )

    pv_fp_share = pv_overlap_fraction_of_fp

    # --------------------------------------------------------
    # Per-patient results.
    # --------------------------------------------------------

    row = {

        "uid": uid,

        "patient_id": record["patient_id"],

        "threshold": THRESHOLD,

        "geometry_annotation_ct": (
            geometry_annotation_ct
        ),

        "geometry_annotation_processed_pa": (
            geometry_annotation_pa
        ),

        "geometry_annotation_prediction": (
            geometry_annotation_prediction
        ),

        "original_pa_exact_match": (
            pa_mismatch == 0
        ),

        "pa_mismatch_voxels": pa_mismatch,

        "total_voxels": int(labels.size),

        "predicted_pa_voxels": (
            predicted_pa_voxels
        ),

        "gt_pa_voxels": (
            total_pa_voxels
        ),

        "gt_pv_voxels": (
            total_pv_voxels
        ),

        "gt_airway_lumen_voxels": (
            class_voxels[1]
        ),

        "gt_airway_wall_voxels": (
            class_voxels[2]
        ),

        "gt_background_voxels": (
            class_voxels[0]
        ),

        "true_positive_pa_voxels": tp,

        "false_negative_pa_voxels": fn,

        "false_positive_total_voxels": fp,

        "predicted_pa_on_background": (
            pred_on_background
        ),

        "predicted_pa_on_airway_lumen": (
            pred_on_airway_lumen
        ),

        "predicted_pa_on_airway_wall": (
            pred_on_airway_wall
        ),

        "predicted_pa_on_pv": (
            pred_on_pv
        ),

        "pv_overlap_fraction_of_false_positives": (
            pv_overlap_fraction_of_fp
        ),

        "pv_overlap_fraction_of_all_predictions": (
            pv_overlap_fraction_of_all_predictions
        ),

        "pv_voxel_prediction_rate": (
            pv_voxel_prediction_rate
        ),

        "background_share_of_false_positives": (
            background_fp_share
        ),

        "airway_lumen_share_of_false_positives": (
            airway_lumen_fp_share
        ),

        "airway_wall_share_of_false_positives": (
            airway_wall_fp_share
        ),

        "pv_share_of_false_positives": (
            pv_fp_share
        ),

        "dice": dice,

        "precision": precision,

        "recall": recall,

        "specificity": specificity

    }

    # --------------------------------------------------------
    # Long-form class confusion rows.
    # --------------------------------------------------------

    class_rows = []

    for class_id, class_name in CLASS_NAMES.items():

        total_class_voxels = class_voxels[class_id]

        predicted_on_class = {

            0: pred_on_background,

            1: pred_on_airway_lumen,

            2: pred_on_airway_wall,

            3: pred_on_pa,

            4: pred_on_pv

        }[class_id]

        class_rows.append({

            "uid": uid,

            "patient_id": record["patient_id"],

            "actual_class_id": class_id,

            "actual_class_name": class_name,

            "total_class_voxels": total_class_voxels,

            "predicted_as_pa_voxels": predicted_on_class,

            "fraction_of_class_predicted_as_pa": (
                safe_divide(
                    predicted_on_class,
                    total_class_voxels
                )
            )

        })

    print(
        "PA true positives:",
        tp
    )

    print(
        "PA false negatives:",
        fn
    )

    print(
        "False positives:",
        fp
    )

    print(
        "False positives on PV:",
        pred_on_pv
    )

    print(
        "PV share of false positives:",
        round(pv_fp_share, 5)
    )

    print(
        "Percentage of PV voxels predicted as PA:",
        round(pv_voxel_prediction_rate, 5)
    )

    print(
        "Full-volume Dice:",
        round(dice, 5)
    )

    # Cleanup.
    del labels
    del prediction
    del processed_pa

    del pa
    del pv
    del airway_lumen
    del airway_wall
    del background

    del annotation_obj
    del prediction_obj
    del pa_obj
    del ct_obj

    gc.collect()

    return row, class_rows


# ============================================================
# 5. AGGREGATE RESULTS
# ============================================================

def create_summary(patient_df):

    columns = [
        "dice",
        "precision",
        "recall",
        "specificity",
        "pv_overlap_fraction_of_false_positives",
        "pv_overlap_fraction_of_all_predictions",
        "pv_voxel_prediction_rate",
        "background_share_of_false_positives",
        "airway_lumen_share_of_false_positives",
        "airway_wall_share_of_false_positives"
    ]

    summary = {

        "project": "NeuroVessel",

        "model": "V5B Residual 3D U-Net",

        "experiment": "PA versus PV confusion audit",

        "threshold": THRESHOLD,

        "patients_evaluated": int(
            len(patient_df)
        ),

        "test_partition_loaded": False,

        "original_pa_annotation_exact_match": bool(
            patient_df["original_pa_exact_match"].all()
        ),

        "metrics": {}

    }

    for column in columns:

        values = patient_df[column].astype(
            float
        )

        summary["metrics"][column] = {

            "mean": float(
                values.mean()
            ),

            "median": float(
                values.median()
            ),

            "std": float(
                values.std(ddof=1)
            )

        }

    # --------------------------------------------------------
    # Pooled voxel totals.
    # --------------------------------------------------------

    count_columns = [

        "predicted_pa_voxels",

        "gt_pa_voxels",

        "gt_pv_voxels",

        "true_positive_pa_voxels",

        "false_negative_pa_voxels",

        "false_positive_total_voxels",

        "predicted_pa_on_background",

        "predicted_pa_on_airway_lumen",

        "predicted_pa_on_airway_wall",

        "predicted_pa_on_pv"

    ]

    totals = {
        column: int(
            patient_df[column].sum()
        )
        for column in count_columns
    }

    summary["voxel_totals"] = totals

    tp = totals["true_positive_pa_voxels"]

    fp = totals["false_positive_total_voxels"]

    fn = totals["false_negative_pa_voxels"]

    pred_count = totals["predicted_pa_voxels"]

    pv_count = totals["gt_pv_voxels"]

    summary["micro_dice"] = safe_divide(
        2 * tp,
        2 * tp + fp + fn
    )

    summary["micro_precision"] = safe_divide(
        tp,
        tp + fp
    )

    summary["micro_recall"] = safe_divide(
        tp,
        tp + fn
    )

    summary["pooled_pv_share_of_false_positives"] = (
        safe_divide(
            totals["predicted_pa_on_pv"],
            fp
        )
    )

    summary["pooled_pv_voxel_prediction_rate"] = (
        safe_divide(
            totals["predicted_pa_on_pv"],
            pv_count
        )
    )

    summary["pooled_pv_overlap_fraction_of_all_predictions"] = (
        safe_divide(
            totals["predicted_pa_on_pv"],
            pred_count
        )
    )

    summary["pooled_background_share_of_false_positives"] = (
        safe_divide(
            totals["predicted_pa_on_background"],
            fp
        )
    )

    summary["pooled_airway_lumen_share_of_false_positives"] = (
        safe_divide(
            totals["predicted_pa_on_airway_lumen"],
            fp
        )
    )

    summary["pooled_airway_wall_share_of_false_positives"] = (
        safe_divide(
            totals["predicted_pa_on_airway_wall"],
            fp
        )
    )

    return summary


# ============================================================
# 6. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 75)

    print(
        "NEUROVESSEL V5B — PA/PV CONFUSION AUDIT"
    )

    print("=" * 75)

    print(
        "Prediction threshold:",
        THRESHOLD
    )

    print(
        "Original annotation directory:",
        ANNOTATION_DIR
    )

    print(
        "Prediction directory:",
        PREDICTION_DIR
    )

    if not ANNOTATION_DIR.is_dir():

        raise FileNotFoundError(
            f"Annotation directory missing: {ANNOTATION_DIR}"
        )

    if not PREDICTION_DIR.is_dir():

        raise FileNotFoundError(
            f"Prediction directory missing: {PREDICTION_DIR}"
        )

    # --------------------------------------------------------
    # Load validation split only.
    # --------------------------------------------------------

    _, validation_records, test_count = load_splits()

    if len(validation_records) != 26:

        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"got {len(validation_records)}"
        )

    if test_count != 26:

        raise RuntimeError(
            f"Expected 26 locked test patients, "
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

    print("\nORIGINAL LABEL CLASSES")

    for class_id, name in CLASS_NAMES.items():

        print(
            f"{class_id}: {name}"
        )

    # --------------------------------------------------------
    # Check all files before starting.
    # --------------------------------------------------------

    missing = []

    for record in validation_records:

        annotation = get_annotation_path(
            record["uid"]
        )

        prediction = get_prediction_path(
            record["uid"]
        )

        if not annotation.is_file():
            missing.append(str(annotation))

        if not prediction.is_file():
            missing.append(str(prediction))

    if missing:

        raise FileNotFoundError(
            "Missing annotation or prediction files:\n"
            + "\n".join(missing[:20])
        )

    # --------------------------------------------------------
    # Run audit.
    # --------------------------------------------------------

    patient_rows = []
    all_class_rows = []

    for index, record in enumerate(
        validation_records,
        start=1
    ):

        print(
            f"\nPATIENT {index}/26"
        )

        patient_row, class_rows = audit_patient(
            record
        )

        patient_rows.append(
            patient_row
        )

        all_class_rows.extend(
            class_rows
        )

        print(
            f"Completed {index}/26",
            flush=True
        )

    # --------------------------------------------------------
    # Save patient-level results.
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        patient_rows
    )

    patient_csv = (
        OUTPUT_DIR
        / "per_patient_pa_pv_confusion.csv"
    )

    patient_df.to_csv(
        patient_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save long-form class confusion table.
    # --------------------------------------------------------

    class_df = pd.DataFrame(
        all_class_rows
    )

    class_csv = (
        OUTPUT_DIR
        / "per_patient_class_confusion.csv"
    )

    class_df.to_csv(
        class_csv,
        index=False
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary = create_summary(
        patient_df
    )

    summary["runtime_seconds"] = (
        time.time() - start_time
    )

    summary["interpretation_limitations"] = [

        "Prediction overlap with PV is a spatial "
        "false-positive attribution, not proof of "
        "the model's internal semantic reasoning.",

        "False positives outside PA and PV may involve "
        "other vessels, airways, background or annotation "
        "boundary differences.",

        "This analysis does not establish that all "
        "remaining errors are caused by PA-PV confusion.",

        "All comparisons use original AirRC class labels "
        "and verified matching image geometry."

    ]

    summary_json = (
        OUTPUT_DIR
        / "pa_pv_confusion_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    totals = summary["voxel_totals"]

    print("\n" + "=" * 75)

    print(
        "NEUROVESSEL V5B — PA/PV CONFUSION SUMMARY"
    )

    print("=" * 75)

    print(
        "Patients evaluated:",
        len(patient_df)
    )

    print(
        "Micro Dice:",
        round(summary["micro_dice"], 5)
    )

    print(
        "Micro precision:",
        round(summary["micro_precision"], 5)
    )

    print(
        "Micro recall:",
        round(summary["micro_recall"], 5)
    )

    print("\nFALSE-POSITIVE ATTRIBUTION")

    print(
        "Total false-positive voxels:",
        totals["false_positive_total_voxels"]
    )

    print(
        "Overlapping PV:",
        totals["predicted_pa_on_pv"]
    )

    print(
        "Overlapping airway lumen:",
        totals["predicted_pa_on_airway_lumen"]
    )

    print(
        "Overlapping airway wall:",
        totals["predicted_pa_on_airway_wall"]
    )

    print(
        "Overlapping background:",
        totals["predicted_pa_on_background"]
    )

    print("\nPV-SPECIFIC MEASUREMENTS")

    print(
        "Pooled fraction of false positives on PV:",
        round(
            summary[
                "pooled_pv_share_of_false_positives"
            ],
            5
        )
    )

    print(
        "Pooled fraction of PV voxels predicted as PA:",
        round(
            summary[
                "pooled_pv_voxel_prediction_rate"
            ],
            5
        )
    )

    print(
        "Pooled fraction of all predicted PA voxels on PV:",
        round(
            summary[
                "pooled_pv_overlap_fraction_of_all_predictions"
            ],
            5
        )
    )

    print("\nOUTPUTS")

    print(
        "Patient results:",
        patient_csv
    )

    print(
        "Class confusion table:",
        class_csv
    )

    print(
        "Aggregate summary:",
        summary_json
    )

    print(
        "\nNo model weights or prediction masks were changed."
    )

    print(
        "The locked test set remains untouched."
    )

    print(
        "Runtime:",
        round(
            summary["runtime_seconds"],
            2
        ),
        "seconds"
    )


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
