
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


# ============================================================
# NEUROVESSEL V6.1 — TARGETED 3D ERROR ANALYSIS
# Read-only analysis of three validation patients.
# Does not modify predictions, labels, or checkpoints.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "outputs" / "predictions" / "airrc_v6_1_fullvol"
LABEL_DIR = ROOT / "data" / "processed_airrc_v6" / "labels_multiclass"

OUTPUT_DIR = ROOT / "outputs" / "metrics" / "airrc_v6_1_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = {
    "LIDC-IDRI-0529":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",

    "LIDC-IDRI-0763":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",

    "LIDC-IDRI-0848":
        "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
}

CLASSES = {
    1: ("PA", 2),
    2: ("PV", 1),
}

# 26-connectivity in 3D.
STRUCTURE = ndimage.generate_binary_structure(3, 3)

summary_rows = []
fp_component_rows = []
fn_component_rows = []


def safe_div(a, b):
    return float(a / b) if b else 0.0


def component_overlap(component_map, target_mask, count):
    """Count target voxels belonging to each connected component."""
    return np.bincount(
        component_map[target_mask],
        minlength=count + 1
    )[1:].astype(np.int64)


def inspect_components(
    patient,
    class_name,
    component_map,
    sizes,
    overlaps,
    spacing_volume,
    output_rows,
    kind,
):
    """Save the ten components with the largest errors."""

    if kind == "false_positive":
        errors = sizes - overlaps
        order = np.argsort(errors)[::-1]
    else:
        errors = sizes - overlaps
        order = np.argsort(errors)[::-1]

    saved = 0

    for index in order:
        error_voxels = int(errors[index])

        if error_voxels <= 0 or saved >= 10:
            break

        size = int(sizes[index])
        overlap = int(overlaps[index])

        output_rows.append({
            "patient_id": patient,
            "class": class_name,
            "component_id": int(index + 1),
            "component_voxels": size,
            "correct_class_overlap_voxels": overlap,
            "error_voxels": error_voxels,
            "error_fraction": safe_div(error_voxels, size),
            "component_volume_mm3": size * spacing_volume,
        })

        saved += 1


def analyze_case(patient, uid):
    print("\n" + "=" * 65)
    print("PATIENT:", patient)
    print("=" * 65)

    pred_path = PRED_DIR / f"{uid}_pred_multiclass.nii.gz"
    gt_path = LABEL_DIR / f"{uid}_multiclass.nii.gz"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction missing: {pred_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    print("Loading prediction...")
    pred_img = nib.load(str(pred_path))

    print("Loading ground truth...")
    gt_img = nib.load(str(gt_path))

    if pred_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch: prediction={pred_img.shape}, "
            f"ground truth={gt_img.shape}"
        )

    if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-3):
        raise ValueError("Prediction and label affine matrices differ.")

    pred = np.asarray(pred_img.dataobj).astype(np.uint8, copy=False)
    gt = np.asarray(gt_img.dataobj).astype(np.uint8, copy=False)

    if not np.isin(pred, [0, 1, 2]).all():
        raise ValueError("Unexpected prediction labels.")

    if not np.isin(gt, [0, 1, 2]).all():
        raise ValueError("Unexpected ground-truth labels.")

    spacing = gt_img.header.get_zooms()[:3]
    voxel_volume = float(np.prod(spacing))

    print("Volume shape:", pred.shape)
    print("Voxel spacing:", spacing)
    print(f"Voxel volume: {voxel_volume:.5f} mm3")

    for class_id, (class_name, other_id) in CLASSES.items():

        print(f"\n--- {class_name} ANALYSIS ---")

        pred_mask = pred == class_id
        gt_mask = gt == class_id

        tp = int(np.count_nonzero(pred_mask & gt_mask))
        pred_count = int(np.count_nonzero(pred_mask))
        gt_count = int(np.count_nonzero(gt_mask))

        fp = pred_count - tp
        fn = gt_count - tp

        fp_background = int(
            np.count_nonzero(pred_mask & (gt == 0))
        )

        fp_other_vessel = int(
            np.count_nonzero(pred_mask & (gt == other_id))
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        dice = safe_div(2 * tp, 2 * tp + fp + fn)

        # ----------------------------------------------------
        # Predicted connected components:
        # identify false-positive regions.
        # ----------------------------------------------------

        print("Finding predicted connected components...")

        pred_components, n_pred = ndimage.label(
            pred_mask,
            structure=STRUCTURE
        )

        pred_sizes = np.bincount(
            pred_components.ravel(),
            minlength=n_pred + 1
        )[1:].astype(np.int64)

        pred_overlap = component_overlap(
            pred_components, gt_mask, n_pred
        )

        fp_by_component = pred_sizes - pred_overlap

        completely_false = pred_overlap == 0

        inspect_components(
            patient,
            class_name,
            pred_components,
            pred_sizes,
            pred_overlap,
            voxel_volume,
            fp_component_rows,
            "false_positive",
        )

        n_completely_false = int(np.count_nonzero(completely_false))

        completely_false_voxels = int(
            pred_sizes[completely_false].sum()
        )

        del pred_components

        # ----------------------------------------------------
        # Ground-truth connected components:
        # identify missed regions and branches.
        # ----------------------------------------------------

        print("Finding ground-truth connected components...")

        gt_components, n_gt = ndimage.label(
            gt_mask,
            structure=STRUCTURE
        )

        gt_sizes = np.bincount(
            gt_components.ravel(),
            minlength=n_gt + 1
        )[1:].astype(np.int64)

        gt_overlap = component_overlap(
            gt_components, pred_mask, n_gt
        )

        fn_by_component = gt_sizes - gt_overlap

        completely_missed = gt_overlap == 0

        inspect_components(
            patient,
            class_name,
            gt_components,
            gt_sizes,
            gt_overlap,
            voxel_volume,
            fn_component_rows,
            "false_negative",
        )

        n_completely_missed = int(
            np.count_nonzero(completely_missed)
        )

        completely_missed_voxels = int(
            gt_sizes[completely_missed].sum()
        )

        del gt_components

        # ----------------------------------------------------
        # Save summary.
        # ----------------------------------------------------

        row = {
            "patient_id": patient,
            "class": class_name,
            "gt_voxels": gt_count,
            "pred_voxels": pred_count,
            "true_positive_voxels": tp,
            "false_positive_voxels": fp,
            "false_negative_voxels": fn,
            "fp_from_background": fp_background,
            "fp_from_other_vessel": fp_other_vessel,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "predicted_components": n_pred,
            "completely_false_components": n_completely_false,
            "completely_false_component_voxels":
                completely_false_voxels,
            "ground_truth_components": n_gt,
            "completely_missed_components": n_completely_missed,
            "completely_missed_component_voxels":
                completely_missed_voxels,
        }

        summary_rows.append(row)

        print(f"Dice: {dice:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Predicted components: {n_pred}")
        print(f"Ground-truth components: {n_gt}")
        print(f"Completely false components: {n_completely_false}")
        print(f"Completely missed components: {n_completely_missed}")
        print(f"FP from background: {fp_background:,}")
        print(f"FP from other vessel: {fp_other_vessel:,}")

        del pred_mask, gt_mask

    del pred, gt


def main():
    print("=" * 65)
    print("NEUROVESSEL V6.1 — 3D ERROR ANALYSIS")
    print("=" * 65)

    print("Patients:", len(PATIENTS))
    print("Connectivity: 26-connected")
    print("Locked test patients will NOT be loaded.")
    print("No model training or checkpoint modification.")

    for patient, uid in PATIENTS.items():
        analyze_case(patient, uid)

    summary_path = OUTPUT_DIR / "component_summary.csv"
    fp_path = OUTPUT_DIR / "largest_false_positive_components.csv"
    fn_path = OUTPUT_DIR / "largest_missed_components.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(fp_component_rows).to_csv(fp_path, index=False)
    pd.DataFrame(fn_component_rows).to_csv(fn_path, index=False)

    print("\n" + "=" * 65)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nSUMMARY:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nOUTPUT FILES:")
    print(summary_path)
    print(fp_path)
    print(fn_path)

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":
    main()
