"""
NeuroVessel: locked-test ground-truth vs V6.1 prediction comparison.

Reads only the 26 UIDs marked "test" in outputs/metrics/airrc_v5_splits.csv.
Does not modify CTs, labels, predictions, checkpoints, or existing metrics.

Outputs:
  outputs/metrics/phase2_gt_comparison/case_metrics.csv
  outputs/metrics/phase2_gt_comparison/summary.csv
  outputs/metrics/phase2_gt_comparison/qc.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize


# ------------------------------ PATHS ------------------------------

PROJECT_ROOT = Path(".")
MANIFEST = PROJECT_ROOT / "outputs/metrics/airrc_v5_splits.csv"
CT_DIR = PROJECT_ROOT / "data/processed_airrc_v5/images"
GT_DIR = PROJECT_ROOT / "data/processed_airrc_v6/labels_multiclass"
PRED_DIR = PROJECT_ROOT / "outputs/predictions/airrc_v6_1_final_test"

OUT_DIR = PROJECT_ROOT / "outputs/metrics/phase2_gt_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUT_DIR / "case_metrics.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
QC_JSON = OUT_DIR / "qc.json"

CLASSES = {1: "PA", 2: "PV"}
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


# ------------------------------ HELPERS ------------------------------

def find_column(frame, candidates):
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(
        f"Could not find any of {candidates} in manifest columns "
        f"{list(frame.columns)}"
    )


def geometry_equal(a, b):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def segmentation_metrics(gt, pred):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))

    denom = 2 * tp + fp + fn
    dice = safe_ratio(2 * tp, denom)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
    }


def mask_descriptors(binary, voxel_volume_mm3, mean_spacing_mm):
    binary = np.asarray(binary, dtype=bool)
    voxel_count = int(np.count_nonzero(binary))

    if voxel_count == 0:
        return {
            "voxel_count": 0,
            "volume_cm3": 0.0,
            "component_count_26n": 0,
            "largest_component_fraction": 0.0,
            "skeleton_voxels": 0,
            "skeleton_length_approx_mm": 0.0,
        }

    component_labels, component_count = ndimage.label(
        binary, structure=CONNECTIVITY_26
    )
    component_sizes = np.bincount(component_labels.ravel())[1:]
    largest_fraction = (
        float(component_sizes.max()) / voxel_count
        if component_sizes.size else 0.0
    )

    # 3D Lee skeletonization. These are digital descriptors, not
    # validated anatomical branch or centreline measurements.
    skeleton = skeletonize(binary, method="lee")
    skeleton_voxels = int(np.count_nonzero(skeleton))

    return {
        "voxel_count": voxel_count,
        "volume_cm3": voxel_count * voxel_volume_mm3 / 1000.0,
        "component_count_26n": int(component_count),
        "largest_component_fraction": largest_fraction,
        "skeleton_voxels": skeleton_voxels,
        "skeleton_length_approx_mm": skeleton_voxels * mean_spacing_mm,
    }


# ------------------------------ MANIFEST ------------------------------

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

manifest = pd.read_csv(MANIFEST)
split_col = find_column(
    manifest, ["split", "partition", "data_split", "dataset_split"]
)
uid_col = find_column(
    manifest,
    ["series_instance_uid", "uid", "case_uid", "case_id", "seriesuid",
     "patient_id", "case", "id"],
)
patient_col = find_column(manifest, ["patient_id"])

test_rows = manifest[
    manifest[split_col].astype(str).str.strip().str.lower() == "test"
].copy()

if len(test_rows) != 26:
    raise RuntimeError(f"Expected 26 test rows, found {len(test_rows)}")

test_rows[uid_col] = test_rows[uid_col].astype(str).str.strip()
if test_rows[uid_col].nunique() != 26:
    raise RuntimeError("Duplicate test UIDs found in manifest.")

expected_pred_names = {
    f"{uid}_pred_multiclass.nii.gz" for uid in test_rows[uid_col]
}
actual_pred_names = {
    p.name for p in PRED_DIR.glob("*_pred_multiclass.nii.gz")
}
if expected_pred_names != actual_pred_names:
    raise RuntimeError(
        "Prediction directory does not exactly match locked test UIDs.\n"
        f"Missing: {sorted(expected_pred_names - actual_pred_names)}\n"
        f"Extra: {sorted(actual_pred_names - expected_pred_names)}"
    )


# ------------------------------ EVALUATION ------------------------------

records = []
qc_cases = []
print("\nNEUROVESSEL: GROUND TRUTH vs V6.1 PREDICTION")
print("=" * 76)
print("Test cases:", len(test_rows))
print("No training, threshold tuning, or prediction modification is performed.")

for _, row in test_rows.iterrows():
    uid = str(row[uid_col]).strip()
    patient = str(row[patient_col])

    ct_path = CT_DIR / f"{uid}_ct.nii.gz"
    gt_path = GT_DIR / f"{uid}_multiclass.nii.gz"
    pred_path = PRED_DIR / f"{uid}_pred_multiclass.nii.gz"

    for label, path in (
        ("CT", ct_path), ("ground truth", gt_path), ("prediction", pred_path)
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing for {uid}: {path}")

    ct_img = sitk.ReadImage(str(ct_path))
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))

    if not geometry_equal(ct_img, gt_img):
        raise RuntimeError(f"CT/ground-truth geometry mismatch: {uid}")
    if not geometry_equal(ct_img, pred_img):
        raise RuntimeError(f"CT/prediction geometry mismatch: {uid}")
    if not geometry_equal(gt_img, pred_img):
        raise RuntimeError(f"Ground-truth/prediction geometry mismatch: {uid}")

    gt_arr = sitk.GetArrayFromImage(gt_img)
    pred_arr = sitk.GetArrayFromImage(pred_img)

    gt_values = set(np.unique(gt_arr).tolist())
    pred_values = set(np.unique(pred_arr).tolist())

    if not gt_values.issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected ground-truth labels for {uid}: {gt_values}")
    if not pred_values.issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected prediction labels for {uid}: {pred_values}")
    if gt_arr.shape != pred_arr.shape:
        raise RuntimeError(f"Array shape mismatch: {uid}")

    spacing_xyz = np.asarray(ct_img.GetSpacing(), dtype=float)
    if not np.all(np.isfinite(spacing_xyz)) or np.any(spacing_xyz <= 0):
        raise RuntimeError(f"Invalid voxel spacing for {uid}")

    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    mean_spacing_mm = float(np.mean(spacing_xyz))

    result = {
        "patient_id": patient,
        "series_instance_uid": uid,
        "geometry_verified": True,
        "spacing_x_mm": spacing_xyz[0],
        "spacing_y_mm": spacing_xyz[1],
        "spacing_z_mm": spacing_xyz[2],
        "voxel_volume_mm3": voxel_volume_mm3,
        "gt_label_values": ",".join(map(str, sorted(gt_values))),
        "pred_label_values": ",".join(map(str, sorted(pred_values))),
    }

    for class_id, class_name in CLASSES.items():
        gt_mask = gt_arr == class_id
        pred_mask = pred_arr == class_id

        seg = segmentation_metrics(gt_mask, pred_mask)
        gt_desc = mask_descriptors(
            gt_mask, voxel_volume_mm3, mean_spacing_mm
        )
        pred_desc = mask_descriptors(
            pred_mask, voxel_volume_mm3, mean_spacing_mm
        )

        for metric, value in seg.items():
            result[f"{class_name}_{metric}"] = value

        for metric, value in gt_desc.items():
            result[f"{class_name}_gt_{metric}"] = value
        for metric, value in pred_desc.items():
            result[f"{class_name}_pred_{metric}"] = value
            result[f"{class_name}_pred_minus_gt_{metric}"] = (
                value - gt_desc[metric]
            )

    records.append(result)
    qc_cases.append({
        "uid": uid,
        "patient_id": patient,
        "geometry_verified": True,
        "gt_labels": sorted(int(v) for v in gt_values),
        "prediction_labels": sorted(int(v) for v in pred_values),
        "array_shape_zyx": list(gt_arr.shape),
        "spacing_xyz_mm": spacing_xyz.tolist(),
    })

    print(
        f"{patient}: "
        f"PA Dice={result['PA_dice']:.4f}, "
        f"PV Dice={result['PV_dice']:.4f}"
    )


# ------------------------------ SAVE RESULTS ------------------------------

case_df = pd.DataFrame(records)
if len(case_df) != 26 or case_df["series_instance_uid"].nunique() != 26:
    raise RuntimeError("Final case count or UID uniqueness check failed.")

case_df.to_csv(CASE_CSV, index=False)

summary_rows = []
summary_metrics = [
    "dice", "precision", "recall",
    "gt_volume_cm3", "pred_volume_cm3", "pred_minus_gt_volume_cm3",
    "gt_component_count_26n", "pred_component_count_26n",
    "pred_minus_gt_component_count_26n",
    "gt_largest_component_fraction", "pred_largest_component_fraction",
    "gt_skeleton_voxels", "pred_skeleton_voxels",
    "pred_minus_gt_skeleton_voxels",
    "gt_skeleton_length_approx_mm", "pred_skeleton_length_approx_mm",
    "pred_minus_gt_skeleton_length_approx_mm",
]

for class_name in CLASSES.values():
    for metric in summary_metrics:
        column = f"{class_name}_{metric}"
        values = case_df[column].astype(float)
        summary_rows.append({
            "class": class_name,
            "metric": metric,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "median": float(values.median()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

qc = {
    "project": "NeuroVessel",
    "evaluation": "locked_test_ground_truth_vs_v6_1_prediction",
    "test_cases_expected": 26,
    "test_cases_processed": len(case_df),
    "unique_test_uids": int(case_df["series_instance_uid"].nunique()),
    "all_geometry_verified": bool(case_df["geometry_verified"].all()),
    "all_label_values_valid": True,
    "ct_directory": str(CT_DIR),
    "ground_truth_directory": str(GT_DIR),
    "prediction_directory": str(PRED_DIR),
    "ground_truth_label_mapping": {"0": "background", "1": "PA", "2": "PV"},
    "prediction_label_mapping": {"0": "background", "1": "PA", "2": "PV"},
    "component_connectivity": "26-neighbour",
    "skeletonization": "scikit-image skeletonize, method=lee",
    "approximate_skeleton_length": (
        "skeleton voxel count multiplied by mean voxel spacing; "
        "coarse proxy, not a graph-based physical centreline length"
    ),
    "interpretation_limitations": [
        "Topology descriptors are computed on digital masks.",
        "Connected components are not anatomical branches.",
        "Skeleton voxel counts and approximate lengths are not validated "
        "anatomical centreline measurements.",
        "This is an evaluation only; no model or threshold tuning was done.",
        "No pulmonary hypertension diagnosis or risk prediction is performed."
    ],
    "cases": qc_cases,
    "outputs": [str(CASE_CSV), str(SUMMARY_CSV), str(QC_JSON)],
}

with open(QC_JSON, "w", encoding="utf-8") as f:
    json.dump(qc, f, indent=2)

print("\n" + "=" * 76)
print("GROUND-TRUTH COMPARISON COMPLETED")
print("=" * 76)
print("Cases processed:", len(case_df))
print("All geometries verified:", bool(case_df["geometry_verified"].all()))
for class_name in CLASSES.values():
    print(
        f"{class_name} mean Dice: "
        f"{case_df[f'{class_name}_dice'].mean():.4f}"
    )
print("\nSaved:")
print(CASE_CSV.resolve())
print(SUMMARY_CSV.resolve())
print(QC_JSON.resolve())
print("\nNo existing predictions, checkpoints, or Phase 2 outputs were modified.")
