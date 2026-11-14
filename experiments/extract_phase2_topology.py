
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy import ndimage
from skimage.morphology import skeletonize


# ============================================================
# NEUROVESSEL — PHASE 2C
# 3D SKELETON AND DIGITAL TOPOLOGY DESCRIPTORS
# ============================================================

sys.path.insert(0, ".")

import train_airrc_v5b as m


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PREDICTION_DIR = Path(
    "outputs/predictions/airrc_v6_1_final_test"
)

OUTPUT_DIR = Path("outputs/metrics/phase2_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV = OUTPUT_DIR / "phase2_topology_26cases.csv"
SUMMARY_CSV = OUTPUT_DIR / "phase2_topology_summary.csv"
CONFIG_JSON = OUTPUT_DIR / "phase2_topology_config.json"

CLASSES = {
    1: "PA",
    2: "PV",
}

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)

# Neighbourhood used to identify local skeleton degree.
# This is a digital-topology descriptor, not anatomical branch detection.
NEIGHBOURHOOD = np.ones((3, 3, 3), dtype=np.uint8)
NEIGHBOURHOOD[1, 1, 1] = 0


# ------------------------------------------------------------
# SKELETON DESCRIPTORS
# ------------------------------------------------------------

def analyse_skeleton(binary_mask, spacing_xyz):

    binary_mask = np.asarray(binary_mask, dtype=bool)

    if not np.any(binary_mask):
        return {
            "skeleton_voxels": 0,
            "skeleton_length_approx_mm": 0.0,
            "skeleton_components_26n": 0,
            "endpoint_candidates": 0,
            "junction_candidates": 0,
            "junction_density_per_1000_skeleton_voxels": 0.0,
        }

    # scikit-image skeletonize uses 3D Lee skeletonization
    skeleton = skeletonize(binary_mask, method="lee")

    skeleton_voxels = int(np.count_nonzero(skeleton))

    # Approximate physical length:
    # Each skeleton voxel is assigned the mean voxel spacing.
    # This is a coarse proxy, NOT a true graph-based centreline length.
    mean_spacing = float(np.mean(spacing_xyz))

    length_approx_mm = skeleton_voxels * mean_spacing

    # Count connected skeleton components using 26-connectivity
    _, component_count = ndimage.label(
        skeleton,
        structure=CONNECTIVITY_26,
    )

    # Count occupied neighbours for every skeleton voxel
    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        NEIGHBOURHOOD,
        mode="constant",
        cval=0,
    )

    # Digital endpoint candidates:
    # skeleton voxels with exactly one occupied neighbour
    endpoints = skeleton & (neighbour_count == 1)

    # Digital junction candidates:
    # skeleton voxels with at least three occupied neighbours
    junctions = skeleton & (neighbour_count >= 3)

    endpoint_count = int(np.count_nonzero(endpoints))
    junction_count = int(np.count_nonzero(junctions))

    junction_density = (
        1000.0 * junction_count / skeleton_voxels
        if skeleton_voxels > 0 else 0.0
    )

    return {
        "skeleton_voxels": skeleton_voxels,
        "skeleton_length_approx_mm": length_approx_mm,
        "skeleton_components_26n": int(component_count),
        "endpoint_candidates": endpoint_count,
        "junction_candidates": junction_count,
        "junction_density_per_1000_skeleton_voxels": junction_density,
    }


# ------------------------------------------------------------
# LOAD LOCKED TEST MANIFEST
# ------------------------------------------------------------

df = pd.read_csv(m.SPLIT_CSV)

split_col = m.find_column(
    df,
    ["split", "partition", "data_split", "dataset_split"],
)

uid_col = m.find_column(
    df,
    [
        "uid",
        "case_uid",
        "case_id",
        "seriesuid",
        "series_instance_uid",
        "patient_id",
        "case",
        "id",
    ],
)

test_rows = df[
    df[split_col].map(m.normalize_split) == "test"
].copy()

if len(test_rows) != 26:
    raise RuntimeError(
        f"Expected 26 locked test cases, found {len(test_rows)}"
    )

test_uids = [
    m.clean_string(value)
    for value in test_rows[uid_col]
]

if len(set(test_uids)) != 26:
    raise RuntimeError("Duplicate test UIDs detected.")

patient_col = m.find_column(
    df,
    ["patient_id"],
)

test_patients = [
    str(value)
    for value in test_rows[patient_col]
]

if len(set(test_patients)) != 26:
    raise RuntimeError("Duplicate patient IDs detected.")


# ------------------------------------------------------------
# VERIFY PREDICTION FILE SET
# ------------------------------------------------------------

expected_files = {
    uid + "_pred_multiclass.nii.gz"
    for uid in test_uids
}

actual_files = {
    path.name
    for path in PREDICTION_DIR.glob("*_pred_multiclass.nii.gz")
}

if expected_files != actual_files:
    missing = expected_files - actual_files
    extra = actual_files - expected_files

    raise RuntimeError(
        f"Prediction file mismatch.\n"
        f"Missing: {sorted(missing)}\n"
        f"Extra: {sorted(extra)}"
    )


# ------------------------------------------------------------
# PROCESS TEST CASES
# ------------------------------------------------------------

records = []

print("\nNEUROVESSEL PHASE 2C")
print("3D SKELETON AND DIGITAL TOPOLOGY")
print("=" * 70)

for _, row in test_rows.iterrows():

    uid = m.clean_string(row[uid_col])
    patient = str(row[patient_col])

    ct_path = m.resolve_dataset_path(
        None,
        m.IMAGE_DIR,
        uid,
        "_ct.nii.gz",
    )

    pred_path = PREDICTION_DIR / (
        uid + "_pred_multiclass.nii.gz"
    )

    if not ct_path.is_file():
        raise FileNotFoundError(ct_path)

    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)

    ct = sitk.ReadImage(str(ct_path))
    pred = sitk.ReadImage(str(pred_path))

    # Verify prediction geometry against processed CT
    geometry_matches = (
        ct.GetSize() == pred.GetSize()
        and np.allclose(
            ct.GetSpacing(),
            pred.GetSpacing(),
            atol=1e-5,
        )
        and np.allclose(
            ct.GetOrigin(),
            pred.GetOrigin(),
            atol=1e-4,
        )
        and np.allclose(
            ct.GetDirection(),
            pred.GetDirection(),
            atol=1e-5,
        )
    )

    if not geometry_matches:
        raise RuntimeError(
            f"Geometry mismatch: {patient}"
        )

    mask = sitk.GetArrayFromImage(pred)

    unexpected = set(np.unique(mask).tolist()) - {0, 1, 2}

    if unexpected:
        raise RuntimeError(
            f"Unexpected labels for {patient}: {unexpected}"
        )

    spacing_xyz = ct.GetSpacing()

    if (
        not np.all(np.isfinite(spacing_xyz))
        or np.any(np.asarray(spacing_xyz) <= 0)
    ):
        raise RuntimeError(
            f"Invalid spacing for {patient}"
        )

    result = {
        "patient_id": patient,
        "series_instance_uid": uid,
        "spacing_x_mm": spacing_xyz[0],
        "spacing_y_mm": spacing_xyz[1],
        "spacing_z_mm": spacing_xyz[2],
        "geometry_verified": True,
    }

    for class_id, class_name in CLASSES.items():

        binary = mask == class_id

        descriptors = analyse_skeleton(
            binary,
            spacing_xyz,
        )

        for feature_name, value in descriptors.items():

            result[
                f"{class_name}_{feature_name}"
            ] = value

    records.append(result)

    print(
        f"{patient}: "
        f"PA skeleton="
        f"{result['PA_skeleton_voxels']}, "
        f"PV skeleton="
        f"{result['PV_skeleton_voxels']}"
    )


# ------------------------------------------------------------
# SAVE PER-CASE RESULTS
# ------------------------------------------------------------

features = pd.DataFrame(records)

if len(features) != 26:
    raise RuntimeError("Not all 26 cases were processed.")

if features["patient_id"].nunique() != 26:
    raise RuntimeError("Patient uniqueness verification failed.")

features.to_csv(FEATURE_CSV, index=False)


# ------------------------------------------------------------
# DESCRIPTIVE SUMMARY
# ------------------------------------------------------------

summary_rows = []

for class_name in CLASSES.values():

    for feature in [
        "skeleton_voxels",
        "skeleton_length_approx_mm",
        "skeleton_components_26n",
        "endpoint_candidates",
        "junction_candidates",
        "junction_density_per_1000_skeleton_voxels",
    ]:

        column = f"{class_name}_{feature}"
        values = features[column]

        summary_rows.append({
            "vessel_class": class_name,
            "feature": feature,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_CSV, index=False)


# ------------------------------------------------------------
# SAVE CONFIGURATION AND LIMITATIONS
# ------------------------------------------------------------

configuration = {
    "project": "NeuroVessel",
    "phase": "2C",
    "cases": 26,
    "prediction_source": str(PREDICTION_DIR),
    "classes": {
        "1": "PA",
        "2": "PV",
    },
    "skeletonization": "scikit-image skeletonize, method=lee",
    "connectivity": "26-neighbour",
    "endpoint_rule": "exactly one occupied 26-neighbour",
    "junction_rule": "at least three occupied 26-neighbours",
    "length_estimate": (
        "skeleton voxel count multiplied by mean voxel spacing; "
        "coarse proxy, not graph-based physical centreline length"
    ),
    "limitations": [
        "Predicted masks, not ground-truth anatomy",
        "Digital endpoints and junctions are not validated anatomical branches",
        "Skeletonization can introduce topology artifacts",
        "No PH diagnosis or risk prediction",
        "No test-set tuning performed",
    ],
}

with open(CONFIG_JSON, "w", encoding="utf-8") as file:
    json.dump(configuration, file, indent=4)


# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 2C COMPLETED")
print("=" * 70)

print("Cases processed:", len(features))

print("\nPA mean skeleton voxels:",
      round(features["PA_skeleton_voxels"].mean(), 2))

print("PV mean skeleton voxels:",
      round(features["PV_skeleton_voxels"].mean(), 2))

print("\nPA endpoint candidates:",
      round(features["PA_endpoint_candidates"].mean(), 2))

print("PV endpoint candidates:",
      round(features["PV_endpoint_candidates"].mean(), 2))

print("\nPA junction candidates:",
      round(features["PA_junction_candidates"].mean(), 2))

print("PV junction candidates:",
      round(features["PV_junction_candidates"].mean(), 2))

print("\nSaved files:")
print(FEATURE_CSV.resolve())
print(SUMMARY_CSV.resolve())
print(CONFIG_JSON.resolve())

print("\nIMPORTANT:")
print("- Skeleton measurements describe predicted masks.")
print("- Endpoint and junction counts are digital candidates.")
print("- Approximate length is not a graph-based centreline length.")
print("- No clinical PH inference is performed.")
