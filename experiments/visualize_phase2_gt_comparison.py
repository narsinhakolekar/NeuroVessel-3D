"""
NeuroVessel qualitative review: CT / ground truth / V6.1 prediction overlays.

Selects three locked-test cases by mean(PA Dice, PV Dice):
  - best combined Dice
  - median combined Dice
  - lowest combined Dice

For each case, exports axial, coronal and sagittal views.
Slices are selected independently per plane at the slice with the
largest ground-truth PA+PV voxel count.

No model inference, tuning, or modification of source data occurs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS_CSV = Path("outputs/metrics/phase2_gt_comparison/case_metrics.csv")
CT_DIR = Path("data/processed_airrc_v5/images")
GT_DIR = Path("data/processed_airrc_v6/labels_multiclass")
PRED_DIR = Path("outputs/predictions/airrc_v6_1_final_test")
OUT_DIR = Path("outputs/visualizations/phase2_gt_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Display window only; source CT data is not modified.
HU_MIN, HU_MAX = -1000.0, 500.0

# Consistent colors:
# GT PA=cyan, GT PV=yellow; predicted PA=red, predicted PV=magenta.
GT_COLORS = {1: (0.0, 0.9, 1.0), 2: (1.0, 0.85, 0.0)}
PRED_COLORS = {1: (1.0, 0.15, 0.1), 2: (1.0, 0.1, 0.85)}


def load_case(uid):
    ct_path = CT_DIR / f"{uid}_ct.nii.gz"
    gt_path = GT_DIR / f"{uid}_multiclass.nii.gz"
    pred_path = PRED_DIR / f"{uid}_pred_multiclass.nii.gz"

    for p in (ct_path, gt_path, pred_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    ct_img = sitk.ReadImage(str(ct_path))
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))

    def same_geometry(a, b):
        return (
            a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
        )

    if not same_geometry(ct_img, gt_img):
        raise RuntimeError(f"CT/GT geometry mismatch: {uid}")
    if not same_geometry(ct_img, pred_img):
        raise RuntimeError(f"CT/prediction geometry mismatch: {uid}")

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)

    if ct.shape != gt.shape or ct.shape != pred.shape:
        raise RuntimeError(f"Array shape mismatch: {uid}")
    if not set(np.unique(gt).tolist()).issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected GT labels: {uid}")
    if not set(np.unique(pred).tolist()).issubset({0, 1, 2}):
        raise RuntimeError(f"Unexpected prediction labels: {uid}")

    return ct, gt, pred


def choose_slices(gt):
    vessel = (gt == 1) | (gt == 2)
    if not np.any(vessel):
        raise RuntimeError("Ground truth contains no PA/PV voxels.")

    # Arrays are z,y,x. Pick the most populated GT slice in each plane.
    axial_counts = vessel.sum(axis=(1, 2))
    coronal_counts = vessel.sum(axis=(0, 2))
    sagittal_counts = vessel.sum(axis=(0, 1))

    return {
        "Axial": int(np.argmax(axial_counts)),
        "Coronal": int(np.argmax(coronal_counts)),
        "Sagittal": int(np.argmax(sagittal_counts)),
    }


def get_plane(array, plane, index):
    if plane == "Axial":
        return array[index, :, :]
    if plane == "Coronal":
        return array[:, index, :]
    if plane == "Sagittal":
        return array[:, :, index]
    raise ValueError(plane)


def overlay_rgb(ct_slice, label_slice, colors, alpha=0.58):
    gray = np.clip(ct_slice, HU_MIN, HU_MAX)
    gray = (gray - HU_MIN) / (HU_MAX - HU_MIN)
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    for class_id, color in colors.items():
        mask = label_slice == class_id
        if np.any(mask):
            rgb[mask] = (
                (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color)
            )
    return np.clip(rgb, 0, 1)


def error_overlay(ct_slice, gt_slice, pred_slice):
    """GT-only errors are blue; prediction-only errors are orange."""
    gray = np.clip(ct_slice, HU_MIN, HU_MAX)
    gray = (gray - HU_MIN) / (HU_MAX - HU_MIN)
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    gt_fg = gt_slice > 0
    pred_fg = pred_slice > 0
    fn = gt_fg & ~pred_fg
    fp = pred_fg & ~gt_fg
    class_confusion = gt_fg & pred_fg & (gt_slice != pred_slice)

    rgb[fn] = (0.1, 0.45, 1.0)          # missed GT vessel
    rgb[fp] = (1.0, 0.45, 0.0)          # extra predicted vessel
    rgb[class_confusion] = (1.0, 0.0, 1.0)  # PA/PV class disagreement
    return rgb


def main():
    if not METRICS_CSV.is_file():
        raise FileNotFoundError(
            f"Run compare_phase2_gt_predictions.py first: {METRICS_CSV}"
        )

    df = pd.read_csv(METRICS_CSV)
    if len(df) != 26 or df["series_instance_uid"].nunique() != 26:
        raise RuntimeError("Expected exactly 26 unique locked test cases.")

    df["mean_fg_dice"] = (df["PA_dice"] + df["PV_dice"]) / 2.0
    ordered = df.sort_values("mean_fg_dice").reset_index(drop=True)

    selected = [
        ("worst", ordered.iloc[0]),
        ("median", ordered.iloc[len(ordered) // 2]),
        ("best", ordered.iloc[-1]),
    ]

    selection_rows = []

    for rank, row in selected:
        uid = str(row["series_instance_uid"])
        patient = str(row["patient_id"])
        ct, gt, pred = load_case(uid)
        slices = choose_slices(gt)

        fig, axes = plt.subplots(3, 4, figsize=(18, 13))
        fig.suptitle(
            f"{rank.upper()} CASE — {patient}\n"
            f"PA Dice={row['PA_dice']:.3f} | "
            f"PV Dice={row['PV_dice']:.3f} | "
            f"Mean foreground Dice={row['mean_fg_dice']:.3f}",
            fontsize=15,
            fontweight="bold",
        )

        columns = [
            "CT",
            "Ground truth overlay\nPA=cyan, PV=yellow",
            "V6.1 prediction overlay\nPA=red, PV=magenta",
            "Difference view\nBlue=missed, orange=extra,\nmagenta=PA/PV disagreement",
        ]

        for col, title in enumerate(columns):
            axes[0, col].set_title(title, fontsize=10)

        for r, (plane, index) in enumerate(slices.items()):
            ct_sl = get_plane(ct, plane, index)
            gt_sl = get_plane(gt, plane, index)
            pred_sl = get_plane(pred, plane, index)

            images = [
                np.clip(ct_sl, HU_MIN, HU_MAX),
                overlay_rgb(ct_sl, gt_sl, GT_COLORS),
                overlay_rgb(ct_sl, pred_sl, PRED_COLORS),
                error_overlay(ct_sl, gt_sl, pred_sl),
            ]

            for c, img in enumerate(images):
                if c == 0:
                    axes[r, c].imshow(img, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
                else:
                    axes[r, c].imshow(img)
                axes[r, c].axis("off")

            axes[r, 0].set_ylabel(
                f"{plane}\nindex={index}",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = OUT_DIR / f"{rank}_{patient}_orthogonal_comparison.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        selection_rows.append({
            "selection": rank,
            "patient_id": patient,
            "series_instance_uid": uid,
            "PA_dice": float(row["PA_dice"]),
            "PV_dice": float(row["PV_dice"]),
            "mean_fg_dice": float(row["mean_fg_dice"]),
            "axial_slice_index": slices["Axial"],
            "coronal_slice_index": slices["Coronal"],
            "sagittal_slice_index": slices["Sagittal"],
            "figure": str(out_path),
        })

        print(
            f"{rank.upper()}: {patient} | "
            f"PA={row['PA_dice']:.4f}, PV={row['PV_dice']:.4f}, "
            f"mean={row['mean_fg_dice']:.4f}"
        )
        print("Saved:", out_path.resolve())

    selection_csv = OUT_DIR / "selected_cases.csv"
    pd.DataFrame(selection_rows).to_csv(selection_csv, index=False)

    print("\nQUALITATIVE VISUALIZATION COMPLETED")
    print("Figures directory:", OUT_DIR.resolve())
    print("Selection summary:", selection_csv.resolve())
    print("Source CTs, labels, predictions and model files were not modified.")


if __name__ == "__main__":
    main()
