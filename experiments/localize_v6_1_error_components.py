
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


# ============================================================
# NEUROVESSEL V6.1 — 3D ERROR COMPONENT LOCALIZATION
#
# Identifies spatial coordinates of the largest:
#   1. False-positive PA/PV components
#   2. Missed PA/PV ground-truth components
#
# Read-only diagnostic.
# Does NOT modify predictions, labels, or checkpoints.
# Locked test patients are never loaded.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "outputs" / "predictions" / "airrc_v6_1_fullvol"

LABEL_DIR = (
    ROOT / "data" / "processed_airrc_v6" / "labels_multiclass"
)

INPUT_DIR = (
    ROOT / "outputs" / "metrics" / "airrc_v6_1_error_analysis"
)

OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Explicit patient IDs and UIDs.
PATIENTS = {
    "LIDC-IDRI-0529":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",

    "LIDC-IDRI-0763":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",

    "LIDC-IDRI-0848":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
}


# V6 label mapping:
# 0 = Background
# 1 = PA
# 2 = PV
CLASS_IDS = {
    "PA": 1,
    "PV": 2,
}

TOP_N = 3

STRUCTURE = ndimage.generate_binary_structure(3, 3)

FP_TABLE_PATH = INPUT_DIR / "largest_false_positive_components.csv"
FN_TABLE_PATH = INPUT_DIR / "largest_missed_components.csv"

OUTPUT_PATH = OUTPUT_DIR / "component_localization.csv"


def get_mask_coordinates(mask):
    coords = np.where(mask)

    if coords[0].size == 0:
        return None

    return coords


def validate_input_files():

    required_files = [
        FP_TABLE_PATH,
        FN_TABLE_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required diagnostic CSV missing: {path}"
            )

    fp_table = pd.read_csv(FP_TABLE_PATH)
    fn_table = pd.read_csv(FN_TABLE_PATH)

    required_columns = {
        "patient_id",
        "class",
        "component_id",
        "component_voxels",
        "error_voxels",
    }

    for name, table in [
        ("False-positive", fp_table),
        ("Missed-component", fn_table),
    ]:
        missing = required_columns - set(table.columns)

        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {sorted(missing)}"
            )

    return fp_table, fn_table


def localize_components(
    patient,
    uid,
    class_name,
    table,
    error_type,
    pred,
    gt,
    affine,
):

    class_id = CLASS_IDS[class_name]
    other_id = 3 - class_id

    if error_type == "FP":
        component_mask = pred == class_id
    elif error_type == "FN":
        component_mask = gt == class_id
    else:
        raise ValueError(f"Unknown error type: {error_type}")

    print(f"\n{patient} | {class_name} | {error_type}")

    # Recreate components using the exact same connectivity.
    component_map, n_components = ndimage.label(
        component_mask,
        structure=STRUCTURE,
    )

    patient_rows = table[
        (table["patient_id"] == patient)
        & (table["class"] == class_name)
    ].copy()

    patient_rows = patient_rows.sort_values(
        "error_voxels",
        ascending=False,
    ).head(TOP_N)

    if patient_rows.empty:
        print("No components found in diagnostic CSV.")
        del component_map
        return []

    localized_rows = []

    for _, record in patient_rows.iterrows():

        component_id = int(record["component_id"])

        if component_id < 1 or component_id > n_components:
            print(
                f"Skipping invalid component ID: {component_id}"
            )
            continue

        # Retrieve the selected connected component.
        full_component = component_map == component_id

        if error_type == "FP":
            # Predicted PA/PV but ground truth is another class.
            error_mask = full_component & (gt != class_id)
        else:
            # Ground-truth PA/PV but prediction is another class.
            error_mask = full_component & (pred != class_id)

        coords = get_mask_coordinates(error_mask)

        if coords is None:
            print(f"Component {component_id}: no errors found.")
            continue

        error_voxels = int(coords[0].size)

        centroid = np.array([
            float(np.mean(axis))
            for axis in coords
        ])

        # NIfTI affine converts voxel coordinates to RAS+ mm.
        world = nib.affines.apply_affine(
            affine,
            centroid,
        )

        minimum = [
            int(np.min(axis))
            for axis in coords
        ]

        maximum = [
            int(np.max(axis))
            for axis in coords
        ]

        if error_type == "FP":

            background_errors = int(
                np.count_nonzero(gt[error_mask] == 0)
            )

            other_class_errors = int(
                np.count_nonzero(gt[error_mask] == other_id)
            )

        else:

            background_errors = int(
                np.count_nonzero(pred[error_mask] == 0)
            )

            other_class_errors = int(
                np.count_nonzero(pred[error_mask] == other_id)
            )

        result = {
            "patient_id": patient,
            "uid": uid,
            "class": class_name,
            "error_type": error_type,
            "component_id": component_id,
            "component_voxels": int(record["component_voxels"]),
            "error_voxels": error_voxels,

            "centroid_i": centroid[0],
            "centroid_j": centroid[1],
            "centroid_k": centroid[2],

            "world_x_mm_RAS": world[0],
            "world_y_mm_RAS": world[1],
            "world_z_mm_RAS": world[2],

            "bbox_i_min": minimum[0],
            "bbox_i_max": maximum[0],
            "bbox_j_min": minimum[1],
            "bbox_j_max": maximum[1],
            "bbox_k_min": minimum[2],
            "bbox_k_max": maximum[2],

            "background_error_voxels": background_errors,
            "other_vessel_error_voxels": other_class_errors,
        }

        localized_rows.append(result)

        print(
            f"Component {component_id}: "
            f"total errors={error_voxels:,}"
        )

        print(
            f"  Voxel centroid (i,j,k): "
            f"{centroid.round(2)}"
        )

        print(
            f"  World centroid RAS (mm): "
            f"{world.round(2)}"
        )

        print(
            f"  Background errors: {background_errors:,}"
        )

        print(
            f"  Other-vessel errors: {other_class_errors:,}"
        )

    del component_map

    return localized_rows


def analyze_patient(patient, uid, fp_table, fn_table):

    print("\n" + "=" * 70)
    print("PATIENT:", patient)
    print("=" * 70)

    pred_path = (
        PRED_DIR / f"{uid}_pred_multiclass.nii.gz"
    )

    gt_path = (
        LABEL_DIR / f"{uid}_multiclass.nii.gz"
    )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction missing: {pred_path}"
        )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth label missing: {gt_path}"
        )

    print("Loading prediction:", pred_path.name)

    pred_img = nib.load(str(pred_path))

    print("Loading ground truth:", gt_path.name)

    gt_img = nib.load(str(gt_path))

    if pred_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch for {patient}: "
            f"{pred_img.shape} vs {gt_img.shape}"
        )

    if not np.allclose(
        pred_img.affine,
        gt_img.affine,
        atol=1e-3,
    ):
        raise ValueError(
            f"Affine mismatch for patient {patient}"
        )

    pred = np.asarray(
        pred_img.dataobj,
        dtype=np.uint8,
    )

    gt = np.asarray(
        gt_img.dataobj,
        dtype=np.uint8,
    )

    if not np.isin(pred, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected prediction labels in {patient}"
        )

    if not np.isin(gt, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected ground-truth labels in {patient}"
        )

    print("Volume shape:", pred.shape)
    print("Voxel spacing:", gt_img.header.get_zooms()[:3])

    localized_rows = []

    for class_name in ["PA", "PV"]:

        localized_rows.extend(
            localize_components(
                patient,
                uid,
                class_name,
                fp_table,
                "FP",
                pred,
                gt,
                gt_img.affine,
            )
        )

        localized_rows.extend(
            localize_components(
                patient,
                uid,
                class_name,
                fn_table,
                "FN",
                pred,
                gt,
                gt_img.affine,
            )
        )

    del pred, gt

    return localized_rows


def main():

    print("=" * 70)
    print("NEUROVESSEL V6.1 — 3D ERROR LOCALIZATION")
    print("=" * 70)

    print("Patients:", len(PATIENTS))
    print("Top components per class and error type:", TOP_N)
    print("Connectivity: 26-connected")
    print("Locked test patients will NOT be loaded.")
    print("No model inference or training will be performed.")

    fp_table, fn_table = validate_input_files()

    all_rows = []

    for patient, uid in PATIENTS.items():

        patient_rows = analyze_patient(
            patient,
            uid,
            fp_table,
            fn_table,
        )

        all_rows.extend(patient_rows)

    result_df = pd.DataFrame(all_rows)

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ERROR LOCALIZATION COMPLETED")
    print("=" * 70)

    print("Localized components:", len(result_df))

    print("\nOUTPUT FILE:")
    print(OUTPUT_PATH)

    if not result_df.empty:
        print("\nLOCALIZATION SUMMARY:")
        print(
            result_df[
                [
                    "patient_id",
                    "class",
                    "error_type",
                    "component_id",
                    "error_voxels",
                    "world_x_mm_RAS",
                    "world_y_mm_RAS",
                    "world_z_mm_RAS",
                ]
            ].to_string(index=False)
        )

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":
    main()
