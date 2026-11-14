
"""
NEUROVESSEL V5B
ORIGINAL AIRRC PA/PV ANNOTATION VERIFICATION

Checks all 26 validation patients.

Verifies:
    1. Original four-class NIfTI annotations exist.
    2. Annotation class IDs and voxel counts.
    3. Original annotation geometry matches processed CT.
    4. Original annotation geometry matches processed PA mask.
    5. Original class 3 matches the processed binary PA mask.
    6. Pulmonary vein class 4 is present and quantified.

Does not:
    - Load locked test images or masks.
    - Modify datasets.
    - Modify model predictions.
    - Train the model.
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
    METRICS_DIR
    / "airrc_v5b_pa_pv_annotation_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GEOMETRY VERIFICATION
# ============================================================

def compare_geometry(a, b):

    size_match = (
        a.GetSize() == b.GetSize()
    )

    spacing_match = np.allclose(
        a.GetSpacing(),
        b.GetSpacing(),
        atol=1e-5
    )

    origin_match = np.allclose(
        a.GetOrigin(),
        b.GetOrigin(),
        atol=1e-4
    )

    direction_match = np.allclose(
        a.GetDirection(),
        b.GetDirection(),
        atol=1e-5
    )

    return {
        "size_match": bool(size_match),
        "spacing_match": bool(spacing_match),
        "origin_match": bool(origin_match),
        "direction_match": bool(direction_match),
        "all_match": bool(
            size_match
            and spacing_match
            and origin_match
            and direction_match
        )
    }


# ============================================================
# VERIFY ONE PATIENT
# ============================================================

def audit_patient(record):

    uid = record["uid"]

    annotation_path = (
        ANNOTATION_DIR
        / f"{uid}.nii.gz"
    )

    print("\n" + "-" * 65)
    print("Patient:", uid)

    if not annotation_path.is_file():

        raise FileNotFoundError(
            f"Original annotation missing: {annotation_path}"
        )

    # --------------------------------------------------------
    # Load original multiclass annotation.
    # --------------------------------------------------------

    annotation_obj = sitk.ReadImage(
        str(annotation_path)
    )

    annotation = sitk.GetArrayFromImage(
        annotation_obj
    )

    # --------------------------------------------------------
    # Load processed CT and binary PA mask.
    # --------------------------------------------------------

    ct_obj = sitk.ReadImage(
        str(record["image"])
    )

    pa_obj = sitk.ReadImage(
        str(record["mask"])
    )

    ct = sitk.GetArrayFromImage(
        ct_obj
    )

    processed_pa = (
        sitk.GetArrayFromImage(pa_obj) > 0
    )

    if annotation.shape != processed_pa.shape:

        raise RuntimeError(
            f"Annotation and processed PA array shapes "
            f"differ for {uid}: "
            f"{annotation.shape} vs {processed_pa.shape}"
        )

    if ct.shape != processed_pa.shape:

        raise RuntimeError(
            f"Processed CT and PA shapes differ for {uid}"
        )

    # --------------------------------------------------------
    # Geometry checks.
    # --------------------------------------------------------

    geometry_ct = compare_geometry(
        annotation_obj,
        ct_obj
    )

    geometry_pa = compare_geometry(
        annotation_obj,
        pa_obj
    )

    # --------------------------------------------------------
    # Inspect original class distribution.
    # --------------------------------------------------------

    class_values, class_counts = np.unique(
        annotation,
        return_counts=True
    )

    class_distribution = {
        str(int(label)): int(count)
        for label, count in zip(
            class_values,
            class_counts
        )
    }

    # Known AirRC classes:
    # 0 Background
    # 1 Airway lumen
    # 2 Airway wall
    # 3 Pulmonary artery
    # 4 Pulmonary vein

    source_pa = annotation == 3
    source_pv = annotation == 4

    # --------------------------------------------------------
    # Compare original PA class with processed PA mask.
    # --------------------------------------------------------

    pa_difference = (
        source_pa != processed_pa
    )

    mismatch_voxels = int(
        np.count_nonzero(pa_difference)
    )

    exact_pa_match = (
        mismatch_voxels == 0
    )

    # PA and PV should be mutually exclusive
    # in a single multiclass label volume.
    pa_pv_overlap = int(
        np.count_nonzero(
            source_pa & source_pv
        )
    )

    # --------------------------------------------------------
    # Counts.
    # --------------------------------------------------------

    pa_voxels = int(
        np.count_nonzero(source_pa)
    )

    pv_voxels = int(
        np.count_nonzero(source_pv)
    )

    if pa_voxels == 0:
        raise RuntimeError(
            f"Original PA class is empty for {uid}"
        )

    if pv_voxels == 0:
        print(
            "WARNING: No class-4 PV voxels found."
        )

    geometry_ok = (
        geometry_ct["all_match"]
        and geometry_pa["all_match"]
    )

    verified = (
        geometry_ok
        and exact_pa_match
        and pa_pv_overlap == 0
    )

    row = {

        "uid": uid,

        "patient_id": record["patient_id"],

        "annotation_path": str(annotation_path),

        "annotation_shape_zyx": str(
            annotation.shape
        ),

        "class_values": ",".join(
            str(int(x))
            for x in class_values
        ),

        "class_distribution_json": json.dumps(
            class_distribution
        ),

        "pa_voxels_original_class3": pa_voxels,

        "pv_voxels_original_class4": pv_voxels,

        "processed_pa_voxels": int(
            np.count_nonzero(processed_pa)
        ),

        "pa_pv_overlap_voxels": pa_pv_overlap,

        "pa_mismatch_voxels": mismatch_voxels,

        "exact_pa_match": exact_pa_match,

        "annotation_ct_geometry_match": (
            geometry_ct["all_match"]
        ),

        "annotation_processed_pa_geometry_match": (
            geometry_pa["all_match"]
        ),

        "verified": verified

    }

    print(
        "Original classes:",
        class_distribution
    )

    print(
        "PA voxels, class 3:",
        pa_voxels
    )

    print(
        "PV voxels, class 4:",
        pv_voxels
    )

    print(
        "PA mismatch voxels:",
        mismatch_voxels
    )

    print(
        "Annotation vs CT geometry:",
        geometry_ct["all_match"]
    )

    print(
        "Annotation vs processed PA geometry:",
        geometry_pa["all_match"]
    )

    print(
        "VERIFIED:",
        verified
    )

    # Cleanup.
    del annotation
    del source_pa
    del source_pv
    del processed_pa
    del pa_difference
    del ct

    del annotation_obj
    del ct_obj
    del pa_obj

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V5B — ORIGINAL PA/PV ANNOTATION AUDIT"
    )

    print("=" * 70)

    if not ANNOTATION_DIR.is_dir():

        raise FileNotFoundError(
            f"Annotation directory missing: {ANNOTATION_DIR}"
        )

    annotation_files = list(
        ANNOTATION_DIR.glob("*.nii.gz")
    )

    preview_files = list(
        ANNOTATION_DIR.glob("*.nii.png")
    )

    print(
        "Original NIfTI annotations:",
        len(annotation_files)
    )

    print(
        "PNG previews:",
        len(preview_files)
    )

    print(
        "Only NIfTI annotations will be loaded."
    )

    # --------------------------------------------------------
    # Load validation records only.
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

    # --------------------------------------------------------
    # Audit.
    # --------------------------------------------------------

    rows = []

    errors = []

    for index, record in enumerate(
        validation_records,
        start=1
    ):

        print(
            f"\nPATIENT {index}/26"
        )

        try:

            row = audit_patient(record)

            rows.append(row)

        except Exception as exc:

            errors.append({

                "uid": record["uid"],

                "error": str(exc)

            })

            print(
                "AUDIT ERROR:",
                exc
            )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    csv_path = (
        OUTPUT_DIR
        / "pa_pv_annotation_verification.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    total_pa = int(
        df["pa_voxels_original_class3"].sum()
    ) if len(df) else 0

    total_pv = int(
        df["pv_voxels_original_class4"].sum()
    ) if len(df) else 0

    verified_count = int(
        df["verified"].sum()
    ) if len(df) else 0

    summary = {

        "project": "NeuroVessel",

        "experiment": "Original PA/PV annotation verification",

        "validation_patients_expected": 26,

        "validation_patients_audited": len(df),

        "locked_test_patients": test_count,

        "test_partition_loaded": False,

        "annotation_nifti_files": len(
            annotation_files
        ),

        "annotation_png_previews": len(
            preview_files
        ),

        "verified_patients": verified_count,

        "patients_with_errors": len(errors),

        "total_original_pa_voxels": total_pa,

        "total_original_pv_voxels": total_pv,

        "all_patients_verified": (
            len(df) == 26
            and verified_count == 26
            and len(errors) == 0
        ),

        "errors": errors,

        "runtime_seconds": (
            time.time() - start_time
        )

    }

    json_path = (
        OUTPUT_DIR
        / "pa_pv_annotation_verification_summary.json"
    )

    with open(
        json_path,
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

    print("\n" + "=" * 70)

    print(
        "PA/PV ANNOTATION VERIFICATION SUMMARY"
    )

    print("=" * 70)

    print(
        "Validation patients audited:",
        len(df)
    )

    print(
        "Verified patients:",
        verified_count
    )

    print(
        "Patients with errors:",
        len(errors)
    )

    print(
        "Total original PA voxels:",
        total_pa
    )

    print(
        "Total original PV voxels:",
        total_pv
    )

    print(
        "All patients verified:",
        summary["all_patients_verified"]
    )

    print(
        "\nCSV:",
        csv_path
    )

    print(
        "JSON:",
        json_path
    )

    print(
        "\nThe locked test partition remains untouched."
    )


if __name__ == "__main__":

    main()
