
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, ".")
import train_airrc_v5b as m


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PREDICTION_DIR = Path(
    "outputs/predictions/airrc_v6_1_final_test"
)

OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV = OUTPUT_DIR / "phase2_vascular_features_26cases.csv"
SUMMARY_CSV = OUTPUT_DIR / "phase2_vascular_feature_summary.csv"

CLASSES = {
    1: "PA",
    2: "PV",
}

# 26-neighbour connectivity in 3D
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


# --------------------------------------------------
# LOAD LOCKED TEST MANIFEST
# --------------------------------------------------

df = pd.read_csv(m.SPLIT_CSV)

split_col = m.find_column(
    df, ["split", "partition", "data_split", "dataset_split"]
)

uid_col = m.find_column(
    df,
    ["uid", "case_uid", "case_id", "seriesuid",
     "series_instance_uid", "patient_id", "case", "id"]
)

test_rows = df[
    df[split_col].map(m.normalize_split) == "test"
].copy()

if len(test_rows) != 26:
    raise RuntimeError(
        f"Expected 26 locked test cases, found {len(test_rows)}"
    )


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

records = []

print("\nPHASE 2: VASCULAR FEATURE EXTRACTION")
print("=" * 75)

for _, row in test_rows.iterrows():

    uid = m.clean_string(row[uid_col])
    patient = str(row.get("patient_id", uid))

    ct_path = m.resolve_dataset_path(
        None, m.IMAGE_DIR, uid, "_ct.nii.gz"
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

    # Verify physical geometry again
    geometry_matches = (
        ct.GetSize() == pred.GetSize()
        and np.allclose(
            ct.GetSpacing(), pred.GetSpacing(), atol=1e-5
        )
        and np.allclose(
            ct.GetOrigin(), pred.GetOrigin(), atol=1e-4
        )
        and np.allclose(
            ct.GetDirection(), pred.GetDirection(), atol=1e-5
        )
    )

    if not geometry_matches:
        raise RuntimeError(
            f"Geometry mismatch for {patient}"
        )

    mask = sitk.GetArrayFromImage(pred)

    unexpected = set(np.unique(mask).tolist()) - {0, 1, 2}
    if unexpected:
        raise RuntimeError(
            f"Unexpected labels for {patient}: {unexpected}"
        )

    # SimpleITK spacing order: x, y, z
    spacing = ct.GetSpacing()
    voxel_volume_mm3 = float(np.prod(spacing))

    result = {
        "patient_id": patient,
        "series_instance_uid": uid,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_volume_mm3,
    }

    for class_id, class_name in CLASSES.items():

        binary = (mask == class_id)

        voxel_count = int(np.count_nonzero(binary))

        # 1 cm3 = 1000 mm3
        physical_volume_cm3 = (
            voxel_count * voxel_volume_mm3 / 1000.0
        )

        if voxel_count > 0:

            component_labels, component_count = ndimage.label(
                binary,
                structure=CONNECTIVITY_26
            )

            component_sizes = np.bincount(
                component_labels.ravel()
            )[1:]

            largest_component = int(component_sizes.max())

            largest_fraction = (
                largest_component / voxel_count
            )

            components_ge_10 = int(
                np.count_nonzero(component_sizes >= 10)
            )

            components_ge_100 = int(
                np.count_nonzero(component_sizes >= 100)
            )

        else:
            component_count = 0
            largest_component = 0
            largest_fraction = 0.0
            components_ge_10 = 0
            components_ge_100 = 0

        result.update({
            f"{class_name}_voxel_count": voxel_count,
            f"{class_name}_volume_cm3": physical_volume_cm3,
            f"{class_name}_component_count_26n": int(component_count),
            f"{class_name}_largest_component_voxels": largest_component,
            f"{class_name}_largest_component_fraction": largest_fraction,
            f"{class_name}_components_ge_10_voxels": components_ge_10,
            f"{class_name}_components_ge_100_voxels": components_ge_100,
        })

    records.append(result)

    print(
        f"{patient}: "
        f"PA={result['PA_volume_cm3']:.2f} cm3, "
        f"PV={result['PV_volume_cm3']:.2f} cm3"
    )


# --------------------------------------------------
# SAVE PER-CASE FEATURES
# --------------------------------------------------

features = pd.DataFrame(records)

features.to_csv(FEATURE_CSV, index=False)


# --------------------------------------------------
# DESCRIPTIVE SUMMARY
# --------------------------------------------------

summary_rows = []

for class_name in CLASSES.values():

    for feature in [
        "voxel_count",
        "volume_cm3",
        "component_count_26n",
        "largest_component_fraction",
        "components_ge_10_voxels",
        "components_ge_100_voxels",
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


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n" + "=" * 75)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 75)

print("Cases processed:", len(features))
print("PA mean volume (cm3):",
      round(features["PA_volume_cm3"].mean(), 3))
print("PV mean volume (cm3):",
      round(features["PV_volume_cm3"].mean(), 3))

print("\nPer-case features:")
print(FEATURE_CSV.resolve())

print("\nDescriptive summary:")
print(SUMMARY_CSV.resolve())

print("\nIMPORTANT:")
print("- Features describe predicted masks, not ground-truth anatomy.")
print("- Connected components are not anatomical branches.")
print("- No PH diagnosis or risk prediction is performed.")
