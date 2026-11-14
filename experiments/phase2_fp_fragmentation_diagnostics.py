"""
NeuroVessel: locked-test false-positive and fragmentation diagnostics.

Read-only analysis of the 26 locked test cases.
Does NOT modify CTs, labels, predictions, checkpoints, or previous metrics.

Reports, separately for PA and PV:
- TP / FP / FN voxel counts and false-positive fraction
- predicted connected-component count and size distribution
- component-level overlap with same-class ground truth
- largest predicted component size and same-class GT overlap
- predicted components with zero same-class GT overlap

Important:
- Components are digital 26-connected components, not anatomical branches.
- "Outside GT" means outside the same-class reference mask, not outside lungs.
- No lung mask is assumed or synthesized.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


MANIFEST = Path("outputs/metrics/airrc_v5_splits.csv")
CT_DIR = Path("data/processed_airrc_v5/images")
GT_DIR = Path("data/processed_airrc_v6/labels_multiclass")
PRED_DIR = Path("outputs/predictions/airrc_v6_1_final_test")
OUT_DIR = Path("outputs/metrics/phase2_fp_fragmentation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUT_DIR / "case_diagnostics.csv"
COMPONENT_CSV = OUT_DIR / "predicted_component_diagnostics.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"
QC_JSON = OUT_DIR / "qc.json"

CLASSES = {1: "PA", 2: "PV"}
STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)


def find_column(df, candidates):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in columns:
            return columns[name.lower()]
    raise KeyError(f"Missing expected column. Tried {candidates}; found {list(df.columns)}")


def geometry_matches(a, b):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def div(a, b):
    return float(a / b) if b else 0.0


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(MANIFEST)
    split_col = find_column(df, ["split", "partition", "data_split", "dataset_split"])
    uid_col = find_column(
        df, ["series_instance_uid", "uid", "case_uid", "case_id",
             "seriesuid", "patient_id", "case", "id"]
    )
    patient_col = find_column(df, ["patient_id"])

    test = df[df[split_col].astype(str).str.strip().str.lower() == "test"].copy()
    if len(test) != 26 or test[uid_col].astype(str).nunique() != 26:
        raise RuntimeError("Expected 26 unique locked test UIDs.")

    expected_predictions = {f"{str(u).strip()}_pred_multiclass.nii.gz" for u in test[uid_col]}
    actual_predictions = {p.name for p in PRED_DIR.glob("*_pred_multiclass.nii.gz")}
    if expected_predictions != actual_predictions:
        raise RuntimeError(
            f"Prediction set mismatch. Missing={sorted(expected_predictions-actual_predictions)}; "
            f"extra={sorted(actual_predictions-expected_predictions)}"
        )

    case_rows = []
    component_rows = []
    geometry_checked = 0

    print("\nNEUROVESSEL — LOCKED TEST FALSE-POSITIVE / FRAGMENTATION DIAGNOSTICS")
    print("=" * 78)

    for _, row in test.iterrows():
        uid = str(row[uid_col]).strip()
        patient = str(row[patient_col])

        ct_path = CT_DIR / f"{uid}_ct.nii.gz"
        gt_path = GT_DIR / f"{uid}_multiclass.nii.gz"
        pred_path = PRED_DIR / f"{uid}_pred_multiclass.nii.gz"
        for p in (ct_path, gt_path, pred_path):
            if not p.is_file():
                raise FileNotFoundError(p)

        ct_img = sitk.ReadImage(str(ct_path))
        gt_img = sitk.ReadImage(str(gt_path))
        pred_img = sitk.ReadImage(str(pred_path))

        if not geometry_matches(ct_img, gt_img):
            raise RuntimeError(f"CT/GT geometry mismatch: {uid}")
        if not geometry_matches(ct_img, pred_img):
            raise RuntimeError(f"CT/prediction geometry mismatch: {uid}")
        geometry_checked += 1

        gt = sitk.GetArrayFromImage(gt_img)
        pred = sitk.GetArrayFromImage(pred_img)

        if gt.shape != pred.shape:
            raise RuntimeError(f"Array shape mismatch: {uid}")
        if not set(np.unique(gt).tolist()).issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected GT labels: {uid}")
        if not set(np.unique(pred).tolist()).issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected prediction labels: {uid}")

        case = {"patient_id": patient, "series_instance_uid": uid,
                "geometry_verified": True}

        for class_id, class_name in CLASSES.items():
            gt_mask = gt == class_id
            pred_mask = pred == class_id

            tp = int(np.count_nonzero(gt_mask & pred_mask))
            fp = int(np.count_nonzero(~gt_mask & pred_mask))
            fn = int(np.count_nonzero(gt_mask & ~pred_mask))
            pred_count = int(np.count_nonzero(pred_mask))

            labels, n_components = ndimage.label(
                pred_mask, structure=STRUCTURE_26
            )
            sizes = np.bincount(labels.ravel())[1:]
            sizes_sorted = np.sort(sizes)[::-1]
            if len(sizes):
                component_overlap = []
                for component_id, size in enumerate(sizes, start=1):
                    comp = labels == component_id
                    overlap_voxels = int(np.count_nonzero(comp & gt_mask))
                    overlap_fraction = div(overlap_voxels, int(size))
                    component_overlap.append(overlap_fraction)
                    component_rows.append({
                        "patient_id": patient,
                        "series_instance_uid": uid,
                        "class": class_name,
                        "component_id": component_id,
                        "component_voxels": int(size),
                        "same_class_gt_overlap_voxels": overlap_voxels,
                        "same_class_gt_overlap_fraction": overlap_fraction,
                        "has_any_same_class_gt_overlap": overlap_voxels > 0,
                    })
                largest_size = int(sizes_sorted[0])
                largest_id = int(np.argmax(sizes)) + 1
                largest_mask = labels == largest_id
                largest_overlap_voxels = int(np.count_nonzero(largest_mask & gt_mask))
                largest_overlap_fraction = div(largest_overlap_voxels, largest_size)
                zero_overlap_components = int(np.count_nonzero(np.asarray(component_overlap) == 0))
                median_size = float(np.median(sizes))
                p90_size = float(np.percentile(sizes, 90))
            else:
                largest_size = 0
                largest_overlap_voxels = 0
                largest_overlap_fraction = 0.0
                zero_overlap_components = 0
                median_size = 0.0
                p90_size = 0.0

            case.update({
                f"{class_name}_gt_voxels": int(np.count_nonzero(gt_mask)),
                f"{class_name}_pred_voxels": pred_count,
                f"{class_name}_tp_voxels": tp,
                f"{class_name}_fp_voxels": fp,
                f"{class_name}_fn_voxels": fn,
                f"{class_name}_fp_fraction_of_prediction": div(fp, pred_count),
                f"{class_name}_component_count_26n": int(n_components),
                f"{class_name}_component_size_median_voxels": median_size,
                f"{class_name}_component_size_p90_voxels": p90_size,
                f"{class_name}_largest_component_voxels": largest_size,
                f"{class_name}_largest_component_gt_overlap_fraction": largest_overlap_fraction,
                f"{class_name}_zero_gt_overlap_components": zero_overlap_components,
            })

        case_rows.append(case)
        print(
            f"{patient}: PA FP={case['PA_fp_voxels']:,}, "
            f"components={case['PA_component_count_26n']}; "
            f"PV FP={case['PV_fp_voxels']:,}, "
            f"components={case['PV_component_count_26n']}"
        )

    case_df = pd.DataFrame(case_rows)
    component_df = pd.DataFrame(component_rows)
    case_df.to_csv(CASE_CSV, index=False)
    component_df.to_csv(COMPONENT_CSV, index=False)

    summary_rows = []
    summary_columns = [
        "fp_voxels", "fp_fraction_of_prediction",
        "component_count_26n", "component_size_median_voxels",
        "component_size_p90_voxels", "largest_component_voxels",
        "largest_component_gt_overlap_fraction",
        "zero_gt_overlap_components",
    ]
    for class_name in CLASSES.values():
        for metric in summary_columns:
            values = case_df[f"{class_name}_{metric}"].astype(float)
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
        "evaluation": "locked_test_false_positive_fragmentation_diagnostics",
        "expected_cases": 26,
        "processed_cases": len(case_df),
        "unique_test_uids": int(case_df["series_instance_uid"].nunique()),
        "geometry_checks_passed": geometry_checked,
        "all_label_values_valid": True,
        "connectivity": "26-neighbour",
        "component_overlap_definition": (
            "fraction of each predicted component's voxels overlapping "
            "the same-class ground-truth mask"
        ),
        "false_positive_definition": (
            "predicted class voxels outside the same-class ground-truth mask"
        ),
        "lung_region_analysis": (
            "not performed; no independently validated lung mask was supplied"
        ),
        "limitations": [
            "Connected components are digital objects, not anatomical branches.",
            "Zero same-class overlap does not prove a component is outside the lungs.",
            "No components were removed and no predictions were modified.",
            "No model, threshold, or post-processing tuning was performed.",
            "Test set remains evaluation-only."
        ],
        "outputs": [str(CASE_CSV), str(COMPONENT_CSV), str(SUMMARY_CSV), str(QC_JSON)]
    }
    with open(QC_JSON, "w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)

    print("\nDIAGNOSTICS COMPLETED")
    print("Cases:", len(case_df))
    print("Component records:", len(component_df))
    print("Geometry checks:", geometry_checked, "/ 26")
    for class_name in CLASSES.values():
        print(
            f"{class_name}: mean FP fraction="
            f"{case_df[f'{class_name}_fp_fraction_of_prediction'].mean():.4f}; "
            f"mean components="
            f"{case_df[f'{class_name}_component_count_26n'].mean():.1f}; "
            f"mean zero-overlap components="
            f"{case_df[f'{class_name}_zero_gt_overlap_components'].mean():.1f}"
        )
    print("\nSaved:")
    print(CASE_CSV.resolve())
    print(COMPONENT_CSV.resolve())
    print(SUMMARY_CSV.resolve())
    print(QC_JSON.resolve())
    print("\nNo source data, predictions, checkpoints, or prior metrics were modified.")


if __name__ == "__main__":
    main()
