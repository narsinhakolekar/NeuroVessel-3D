
"""
NEUROVESSEL V6
THREE-CLASS LABEL PREPARATION

Classes:
    0 = Background (original classes 0, 1 and 2)
    1 = Pulmonary Artery (original class 3)
    2 = Pulmonary Vein (original class 4)

Processes:
    202 training patients
    26 validation patients

Does NOT:
    Load or process test annotations.
    Modify original AirRC annotations.
    Modify V5B data or predictions.
"""

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
# PATHS
# ============================================================

ANNOTATION_DIR = Path(
    r"C:\Users\narsi\Downloads\AirRC_annotations_extracted\labelsTr"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_airrc_v6"
    / "labels_multiclass"
)

REPORT_DIR = (
    METRICS_DIR
    / "airrc_v6_multiclass"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASS MAPPING
# ============================================================

# Original AirRC:
# 0 = Background
# 1 = Airway lumen
# 2 = Airway wall
# 3 = Pulmonary artery
# 4 = Pulmonary vein

# V6:
# 0 = Background
# 1 = Pulmonary artery
# 2 = Pulmonary vein

CLASS_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2
}


# ============================================================
# HELPERS
# ============================================================

def get_annotation_path(uid):

    return (
        ANNOTATION_DIR
        / f"{uid}.nii.gz"
    )


def get_output_path(uid):

    return (
        OUTPUT_DIR
        / f"{uid}_multiclass.nii.gz"
    )


def geometry_matches(a, b):

    return (

        a.GetSize() == b.GetSize()

        and np.allclose(
            a.GetSpacing(),
            b.GetSpacing(),
            atol=1e-5
        )

        and np.allclose(
            a.GetOrigin(),
            b.GetOrigin(),
            atol=1e-4
        )

        and np.allclose(
            a.GetDirection(),
            b.GetDirection(),
            atol=1e-5
        )
    )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(record, split):

    uid = record["uid"]

    annotation_path = get_annotation_path(uid)

    output_path = get_output_path(uid)

    if not annotation_path.is_file():

        raise FileNotFoundError(
            f"Annotation missing: {annotation_path}"
        )

    print(
        "\nProcessing:",
        uid,
        "| Split:",
        split
    )

    # --------------------------------------------------------
    # Load original annotation.
    # --------------------------------------------------------

    annotation_obj = sitk.ReadImage(
        str(annotation_path)
    )

    original = sitk.GetArrayFromImage(
        annotation_obj
    )

    # --------------------------------------------------------
    # Load processed CT and PA mask.
    # --------------------------------------------------------

    ct_obj = sitk.ReadImage(
        str(record["image"])
    )

    pa_obj = sitk.ReadImage(
        str(record["mask"])
    )

    processed_pa = (
        sitk.GetArrayFromImage(
            pa_obj
        ) > 0
    )

    # --------------------------------------------------------
    # Verify geometry.
    # --------------------------------------------------------

    if not geometry_matches(
        annotation_obj,
        ct_obj
    ):

        raise RuntimeError(
            f"Original annotation and CT geometry mismatch: {uid}"
        )

    if not geometry_matches(
        annotation_obj,
        pa_obj
    ):

        raise RuntimeError(
            f"Original annotation and PA geometry mismatch: {uid}"
        )

    if original.shape != processed_pa.shape:

        raise RuntimeError(
            f"Array shape mismatch: {uid}"
        )

    # --------------------------------------------------------
    # Check source labels.
    # --------------------------------------------------------

    unique_values = np.unique(
        original
    )

    invalid = [
        int(v)
        for v in unique_values
        if int(v) not in CLASS_MAP
    ]

    if invalid:

        raise RuntimeError(
            f"Unknown source classes for {uid}: {invalid}"
        )

    # --------------------------------------------------------
    # Exact PA label consistency.
    # --------------------------------------------------------

    original_pa = (
        original == 3
    )

    pa_mismatch = int(
        np.count_nonzero(
            original_pa != processed_pa
        )
    )

    if pa_mismatch != 0:

        raise RuntimeError(
            f"PA mask mismatch for {uid}: {pa_mismatch}"
        )

    # --------------------------------------------------------
    # Build new three-class segmentation.
    # --------------------------------------------------------

    multiclass = np.zeros(
        original.shape,
        dtype=np.uint8
    )

    multiclass[original == 3] = 1
    multiclass[original == 4] = 2

    # --------------------------------------------------------
    # Verify class distribution.
    # --------------------------------------------------------

    class_counts = np.bincount(
        multiclass.ravel(),
        minlength=3
    )

    background_voxels = int(
        class_counts[0]
    )

    pa_voxels = int(
        class_counts[1]
    )

    pv_voxels = int(
        class_counts[2]
    )

    if pa_voxels == 0:

        raise RuntimeError(
            f"Empty PA class for {uid}"
        )

    if pv_voxels == 0:

        raise RuntimeError(
            f"Empty PV class for {uid}"
        )

    # The new multiclass PA must equal original PA.
    if not np.array_equal(
        multiclass == 1,
        original == 3
    ):

        raise RuntimeError(
            f"Multiclass PA conversion failed: {uid}"
        )

    # The new multiclass PV must equal original PV.
    if not np.array_equal(
        multiclass == 2,
        original == 4
    ):

        raise RuntimeError(
            f"Multiclass PV conversion failed: {uid}"
        )

    # --------------------------------------------------------
    # Save as NIfTI with original CT geometry.
    # --------------------------------------------------------

    output_obj = sitk.GetImageFromArray(
        multiclass
    )

    output_obj.CopyInformation(
        ct_obj
    )

    sitk.WriteImage(
        output_obj,
        str(output_path),
        useCompression=True
    )

    # --------------------------------------------------------
    # Verify saved file.
    # --------------------------------------------------------

    verification_obj = sitk.ReadImage(
        str(output_path)
    )

    verification = sitk.GetArrayFromImage(
        verification_obj
    )

    saved_geometry_ok = geometry_matches(
        verification_obj,
        ct_obj
    )

    if not saved_geometry_ok:

        raise RuntimeError(
            f"Saved geometry verification failed: {uid}"
        )

    if not np.array_equal(
        verification,
        multiclass
    ):

        raise RuntimeError(
            f"Saved voxel verification failed: {uid}"
        )

    saved_classes = np.unique(
        verification
    )

    if not set(
        saved_classes.tolist()
    ).issubset({0, 1, 2}):

        raise RuntimeError(
            f"Unexpected saved class values: {uid}"
        )

    # --------------------------------------------------------
    # Manifest row.
    # --------------------------------------------------------

    row = {

        "uid": uid,

        "patient_id": record["patient_id"],

        "split": split,

        "original_annotation": str(
            annotation_path
        ),

        "image_path": str(
            record["image"]
        ),

        "original_pa_mask": str(
            record["mask"]
        ),

        "multiclass_label_path": str(
            output_path
        ),

        "background_voxels": background_voxels,

        "pa_voxels": pa_voxels,

        "pv_voxels": pv_voxels,

        "pa_mismatch_voxels": pa_mismatch,

        "geometry_verified": saved_geometry_ok,

        "saved_file_verified": True

    }

    print(
        "Background:",
        background_voxels
    )

    print(
        "PA:",
        pa_voxels
    )

    print(
        "PV:",
        pv_voxels
    )

    print(
        "Verified:",
        saved_geometry_ok
    )

    # Cleanup.
    del original
    del multiclass
    del verification

    del annotation_obj
    del ct_obj
    del pa_obj
    del output_obj
    del verification_obj

    del processed_pa
    del original_pa

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V6 — THREE-CLASS DATA PREPARATION"
    )

    print("=" * 70)

    print("\nCLASS MAPPING")

    print("0 = Background")
    print("1 = Pulmonary Artery")
    print("2 = Pulmonary Vein")

    print(
        "\nOriginal annotations:",
        ANNOTATION_DIR
    )

    print(
        "Output labels:",
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Dataset splits
    # --------------------------------------------------------

    train_records, val_records, test_count = load_splits()

    if len(train_records) != 202:

        raise RuntimeError(
            f"Expected 202 training patients, "
            f"got {len(train_records)}"
        )

    if len(val_records) != 26:

        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"got {len(val_records)}"
        )

    if test_count != 26:

        raise RuntimeError(
            f"Expected 26 locked test patients, "
            f"got {test_count}"
        )

    train_ids = {
        r["patient_id"] for r in train_records
    }

    val_ids = {
        r["patient_id"] for r in val_records
    }

    if train_ids.intersection(val_ids):

        raise RuntimeError(
            "Patient leakage between train and validation."
        )

    print("\nDATASET")

    print(
        "Training patients:",
        len(train_records)
    )

    print(
        "Validation patients:",
        len(val_records)
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "\nTest annotations and test images will not be loaded."
    )

    # --------------------------------------------------------
    # Process training and validation only.
    # --------------------------------------------------------

    rows = []
    errors = []

    total_records = (
        len(train_records) + len(val_records)
    )

    for split, records in [

        ("train", train_records),

        ("validation", val_records)

    ]:

        for index, record in enumerate(
            records,
            start=1
        ):

            overall_index = len(rows) + len(errors) + 1

            print(
                f"\n[{overall_index}/{total_records}] "
                f"{split.upper()} "
                f"{index}/{len(records)}"
            )

            try:

                row = process_patient(
                    record,
                    split
                )

                rows.append(row)

            except Exception as exc:

                errors.append({

                    "uid": record["uid"],

                    "split": split,

                    "error": str(exc)

                })

                print(
                    "ERROR:",
                    exc
                )

    # --------------------------------------------------------
    # Save manifest.
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(
        rows
    )

    manifest_path = (
        REPORT_DIR
        / "airrc_v6_multiclass_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate class counts.
    # --------------------------------------------------------

    if len(manifest_df):

        total_background = int(
            manifest_df["background_voxels"].sum()
        )

        total_pa = int(
            manifest_df["pa_voxels"].sum()
        )

        total_pv = int(
            manifest_df["pv_voxels"].sum()
        )

    else:

        total_background = 0
        total_pa = 0
        total_pv = 0

    train_success = int(
        (
            manifest_df["split"] == "train"
        ).sum()
    ) if len(manifest_df) else 0

    val_success = int(
        (
            manifest_df["split"] == "validation"
        ).sum()
    ) if len(manifest_df) else 0

    all_verified = (
        len(rows) == 228
        and len(errors) == 0
        and bool(
            manifest_df["geometry_verified"].all()
        )
        and bool(
            manifest_df["saved_file_verified"].all()
        )
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    summary = {

        "project": "NeuroVessel",

        "model_version": "V6",

        "task": (
            "Three-class pulmonary artery and "
            "pulmonary vein segmentation"
        ),

        "class_mapping": {

            "0": "Background (original classes 0, 1 and 2)",

            "1": "Pulmonary Artery (original class 3)",

            "2": "Pulmonary Vein (original class 4)"

        },

        "training_patients_expected": 202,

        "training_patients_processed": train_success,

        "validation_patients_expected": 26,

        "validation_patients_processed": val_success,

        "locked_test_patients": test_count,

        "test_partition_loaded": False,

        "total_background_voxels": total_background,

        "total_pa_voxels": total_pa,

        "total_pv_voxels": total_pv,

        "successful_patients": len(rows),

        "failed_patients": len(errors),

        "all_verified": all_verified,

        "errors": errors,

        "manifest": str(manifest_path),

        "label_output_directory": str(
            OUTPUT_DIR
        ),

        "runtime_seconds": (
            time.time() - start_time
        )

    }

    summary_path = (
        REPORT_DIR
        / "airrc_v6_multiclass_summary.json"
    )

    with open(
        summary_path,
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

    print("\n" + "=" * 75)

    print(
        "NEUROVESSEL V6 — DATA PREPARATION SUMMARY"
    )

    print("=" * 75)

    print(
        "Training processed:",
        train_success,
        "/ 202"
    )

    print(
        "Validation processed:",
        val_success,
        "/ 26"
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "Background voxels:",
        total_background
    )

    print(
        "PA voxels:",
        total_pa
    )

    print(
        "PV voxels:",
        total_pv
    )

    print(
        "Failed patients:",
        len(errors)
    )

    print(
        "ALL VERIFIED:",
        all_verified
    )

    print(
        "\nManifest:",
        manifest_path
    )

    print(
        "Summary:",
        summary_path
    )

    print(
        "\nTest annotations and images were not loaded."
    )

    print(
        "Original V5B data and predictions remain unchanged."
    )

    print(
        "Runtime:",
        round(
            summary["runtime_seconds"],
            2
        ),
        "seconds"
    )

    if not all_verified:

        raise RuntimeError(
            "V6 dataset preparation was incomplete. "
            "Inspect the saved error report."
        )


if __name__ == "__main__":

    main()
