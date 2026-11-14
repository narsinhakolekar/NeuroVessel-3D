
"""
NEUROVESSEL V5B
BEFORE VS AFTER COMPONENT FILTERING

Comparison:
    Original prediction at threshold 0.70
    vs.
    26-connected component filtering at 250 mm3

Six representative validation cases.

The script:
    - Does not modify original predictions.
    - Does not modify ground-truth masks.
    - Does not retrain the model.
    - Does not load the locked test set.
"""

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from train_airrc_v5b import (
    load_splits,
    load_case,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.70
CUTOFF_MM3 = 250.0

AUDIT_CSV = (
    METRICS_DIR
    / "airrc_v5b_component_ablation"
    / "per_patient_component_ablation.csv"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v5b_fullvol"
    / "threshold_0.70"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / "airrc_v5b_component_filter_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STRUCTURE_26 = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# HELPERS
# ============================================================

def safe_filename(uid):

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(uid)
    )


def prediction_path(uid):

    return (
        PREDICTION_DIR
        / (
            safe_filename(uid)
            + "_pa_prediction.nii.gz"
        )
    )


def calculate_metrics(pred, gt):

    tp = int(
        np.count_nonzero(pred & gt)
    )

    pred_count = int(
        np.count_nonzero(pred)
    )

    gt_count = int(
        np.count_nonzero(gt)
    )

    fp = pred_count - tp
    fn = gt_count - tp

    dice = (
        2.0 * tp / (pred_count + gt_count)
        if pred_count + gt_count
        else 1.0
    )

    precision = (
        tp / pred_count
        if pred_count else 0.0
    )

    recall = (
        tp / gt_count
        if gt_count else 0.0
    )

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def apply_component_filter(
    prediction,
    voxel_volume_mm3
):

    labels = np.empty(
        prediction.shape,
        dtype=np.int32
    )

    number_of_components = ndi.label(
        prediction,
        structure=STRUCTURE_26,
        output=labels
    )

    counts = np.bincount(
        labels.ravel()
    )

    component_sizes = counts[1:]

    component_volumes = (
        component_sizes * voxel_volume_mm3
    )

    keep = np.zeros(
        number_of_components + 1,
        dtype=bool
    )

    keep[1:] = (
        component_volumes >= CUTOFF_MM3
    )

    filtered = keep[labels]

    retained_components = int(
        np.count_nonzero(keep[1:])
    )

    removed_voxels = (
        int(np.count_nonzero(prediction))
        - int(np.count_nonzero(filtered))
    )

    del labels, counts, component_sizes
    del component_volumes, keep

    return (
        filtered,
        number_of_components,
        retained_components,
        removed_voxels
    )


# ============================================================
# SELECT REPRESENTATIVE PATIENTS
# ============================================================


def select_cases(df):

    baseline = df[
        df["minimum_component_volume_mm3"] == 0
    ].copy()

    filtered = df[
        df["minimum_component_volume_mm3"] == CUTOFF_MM3
    ].copy()

    if len(baseline) != 26 or len(filtered) != 26:
        raise RuntimeError(
            "Expected 26 baseline and 26 filtered patient records."
        )

    merged = baseline.merge(
        filtered,
        on="uid",
        suffixes=("_base", "_filtered")
    )

    # Calculate these fractions from the actual voxel counts.
    merged["fp_fraction_of_prediction_base"] = (
        merged["fp_base"]
        / merged["predicted_voxels_base"].clip(lower=1)
    )

    merged["fn_fraction_of_ground_truth_base"] = (
        merged["fn_base"]
        / merged["ground_truth_voxels_base"].clip(lower=1)
    )

    merged["dice_change"] = (
        merged["dice_filtered"]
        - merged["dice_base"]
    )

    # Six representative diagnostic cases.
    # Correct sort directions:
    criteria = [

        # Largest positive Dice improvement.
        ("Largest improvement", "dice_change", False),

        # Largest negative Dice change.
        ("Largest decrease", "dice_change", True),

        # Lowest original Dice.
        ("Weakest baseline", "dice_base", True),

        # Highest original false-positive fraction.
        ("Highest baseline FP fraction",
         "fp_fraction_of_prediction_base", False),

        # Highest original false-negative fraction.
        ("Highest baseline FN fraction",
         "fn_fraction_of_ground_truth_base", False),

        # Highest original Dice.
        ("Strongest baseline", "dice_base", False)

    ]

    selected = []
    used_uids = set()

    for label, column, ascending in criteria:

        candidates = merged.sort_values(
            column,
            ascending=ascending
        )

        chosen = None

        for _, row in candidates.iterrows():

            uid = str(row["uid"])

            if uid not in used_uids:
                chosen = row
                break

        if chosen is not None:

            selected.append(
                (label, chosen)
            )

            used_uids.add(
                str(chosen["uid"])
            )

    if len(selected) != 6:
        raise RuntimeError(
            f"Expected 6 unique cases, selected {len(selected)}."
        )

    return selected



# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_figure(
    category,
    row,
    record
):

    uid = str(row["uid"])

    print(
        "\nVisualizing:",
        category,
        uid
    )

    image, mask = load_case(record)

    gt = mask > 0

    pred_obj = sitk.ReadImage(
        str(prediction_path(uid))
    )

    pred = (
        sitk.GetArrayFromImage(pred_obj) > 0
    )

    reference_obj = sitk.ReadImage(
        str(record["image"])
    )

    geometry_ok = (
        pred_obj.GetSize()
        == reference_obj.GetSize()

        and pred_obj.GetSpacing()
        == reference_obj.GetSpacing()

        and pred_obj.GetOrigin()
        == reference_obj.GetOrigin()

        and pred_obj.GetDirection()
        == reference_obj.GetDirection()
    )

    if not geometry_ok:

        raise RuntimeError(
            f"Geometry mismatch: {uid}"
        )

    if pred.shape != gt.shape:

        raise RuntimeError(
            f"Mask shape mismatch: {uid}"
        )

    spacing = reference_obj.GetSpacing()

    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    filtered, original_components, retained, removed_voxels = (
        apply_component_filter(
            pred,
            voxel_volume_mm3
        )
    )

    baseline_metrics = calculate_metrics(
        pred,
        gt
    )

    filtered_metrics = calculate_metrics(
        filtered,
        gt
    )

    # Select one axial slice with maximum total baseline
    # disagreement. Every panel uses this same slice.
    fp = pred & ~gt
    fn = gt & ~pred

    error_per_slice = (
        np.count_nonzero(fp, axis=(1, 2))
        + np.count_nonzero(fn, axis=(1, 2))
    )

    z = int(
        np.argmax(error_per_slice)
    )

    ct_slice = np.clip(
        image[z],
        -1000,
        1000
    )

    gt_slice = gt[z]
    pred_slice = pred[z]
    filtered_slice = filtered[z]

    baseline_fp = (
        pred_slice & ~gt_slice
    )

    baseline_fn = (
        gt_slice & ~pred_slice
    )

    filtered_fp = (
        filtered_slice & ~gt_slice
    )

    filtered_fn = (
        gt_slice & ~filtered_slice
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 11)
    )

    fig.suptitle(
        "NeuroVessel V5B — Component Filter Comparison\n"
        f"{category} | Patient: {uid}\n"
        f"Dice: {baseline_metrics['dice']:.4f} → "
        f"{filtered_metrics['dice']:.4f} "
        f"| Change: "
        f"{filtered_metrics['dice'] - baseline_metrics['dice']:+.4f}",
        fontsize=14,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Panel 1: Original CT
    # --------------------------------------------------------

    axes[0, 0].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=1000,
        origin="lower"
    )

    axes[0, 0].set_title(
        f"1. Original CT | Axial Z={z}"
    )

    # --------------------------------------------------------
    # Panel 2: Ground truth
    # --------------------------------------------------------

    axes[0, 1].imshow(
        gt_slice.astype(np.uint8),
        cmap=ListedColormap(
            ["black", "lime"]
        ),
        vmin=0,
        vmax=1,
        origin="lower"
    )

    axes[0, 1].set_title(
        "2. Ground Truth PA"
    )

    # --------------------------------------------------------
    # Panel 3: Original prediction
    # --------------------------------------------------------

    axes[0, 2].imshow(
        pred_slice.astype(np.uint8),
        cmap=ListedColormap(
            ["black", "red"]
        ),
        vmin=0,
        vmax=1,
        origin="lower"
    )

    axes[0, 2].set_title(
        f"3. Original V5B Prediction\n"
        f"Dice={baseline_metrics['dice']:.4f}"
    )

    # --------------------------------------------------------
    # Panel 4: Filtered prediction
    # --------------------------------------------------------

    axes[1, 0].imshow(
        filtered_slice.astype(np.uint8),
        cmap=ListedColormap(
            ["black", "orange"]
        ),
        vmin=0,
        vmax=1,
        origin="lower"
    )

    axes[1, 0].set_title(
        f"4. Filtered Prediction\n"
        f"Dice={filtered_metrics['dice']:.4f}"
    )

    # --------------------------------------------------------
    # Panel 5: Original error map
    # --------------------------------------------------------

    axes[1, 1].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=1000,
        origin="lower"
    )

    error_base = np.zeros(
        (*gt_slice.shape, 4),
        dtype=np.float32
    )

    error_base[baseline_fp] = [
        1.0, 0.0, 0.0, 0.75
    ]

    error_base[baseline_fn] = [
        0.0, 0.85, 1.0, 0.80
    ]

    axes[1, 1].imshow(
        error_base,
        origin="lower"
    )

    axes[1, 1].set_title(
        "5. Original Errors\nFP=red | FN=cyan"
    )

    # --------------------------------------------------------
    # Panel 6: Filtered error map
    # --------------------------------------------------------

    axes[1, 2].imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=1000,
        origin="lower"
    )

    error_filtered = np.zeros(
        (*gt_slice.shape, 4),
        dtype=np.float32
    )

    error_filtered[filtered_fp] = [
        1.0, 0.0, 0.0, 0.75
    ]

    error_filtered[filtered_fn] = [
        0.0, 0.85, 1.0, 0.80
    ]

    axes[1, 2].imshow(
        error_filtered,
        origin="lower"
    )

    axes[1, 2].set_title(
        "6. Filtered Errors\nFP=red | FN=cyan"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.text(
        0.5,
        0.045,
        (
            f"Original: Precision={baseline_metrics['precision']:.4f}, "
            f"Recall={baseline_metrics['recall']:.4f}     |     "
            f"Filtered: Precision={filtered_metrics['precision']:.4f}, "
            f"Recall={filtered_metrics['recall']:.4f}"
        ),
        ha="center",
        fontsize=10
    )

    fig.text(
        0.5,
        0.018,
        (
            f"Component cutoff: {CUTOFF_MM3:.0f} mm³ | "
            f"Components: {original_components} → {retained} | "
            f"Removed voxels: {removed_voxels:,}"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.07, 1, 0.90]
    )

    output_path = (
        OUTPUT_DIR
        / f"comparison_{safe_filename(uid)}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    result = {

        "uid": uid,

        "selection": category,

        "threshold": THRESHOLD,

        "cutoff_mm3": CUTOFF_MM3,

        "original_dice": baseline_metrics["dice"],

        "filtered_dice": filtered_metrics["dice"],

        "dice_change": (
            filtered_metrics["dice"]
            - baseline_metrics["dice"]
        ),

        "original_precision": (
            baseline_metrics["precision"]
        ),

        "filtered_precision": (
            filtered_metrics["precision"]
        ),

        "original_recall": (
            baseline_metrics["recall"]
        ),

        "filtered_recall": (
            filtered_metrics["recall"]
        ),

        "original_components": original_components,

        "retained_components": retained,

        "removed_prediction_voxels": removed_voxels,

        "visualization": str(output_path)

    }

    print(
        "Original Dice:",
        round(baseline_metrics["dice"], 5)
    )

    print(
        "Filtered Dice:",
        round(filtered_metrics["dice"], 5)
    )

    print(
        "Dice change:",
        round(
            result["dice_change"],
            5
        )
    )

    print(
        "Saved:",
        output_path
    )

    del image, mask, gt, pred, filtered
    del pred_obj, reference_obj
    del fp, fn, error_per_slice

    gc.collect()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V5B — COMPONENT FILTER VISUAL COMPARISON"
    )

    print("=" * 70)

    if not AUDIT_CSV.is_file():

        raise FileNotFoundError(
            f"Component audit CSV missing: {AUDIT_CSV}"
        )

    # Validation records only.
    _, validation_records, test_count = load_splits()

    if len(validation_records) != 26:

        raise RuntimeError(
            "Expected 26 validation patients."
        )

    if test_count != 26:

        raise RuntimeError(
            "Expected 26 locked test patients."
        )

    print(
        "Validation patients:",
        len(validation_records)
    )

    print(
        "Locked test patients:",
        test_count
    )

    print(
        "Test images will not be loaded."
    )

    df = pd.read_csv(
        AUDIT_CSV
    )

    selected = select_cases(df)

    record_lookup = {
        r["uid"]: r
        for r in validation_records
    }

    results = []

    for index, (category, row) in enumerate(
        selected,
        start=1
    ):

        uid = str(row["uid"])

        print(
            f"\nCASE {index}/{len(selected)}"
        )

        if uid not in record_lookup:

            raise RuntimeError(
                f"Case not in validation split: {uid}"
            )

        result = create_figure(
            category,
            row,
            record_lookup[uid]
        )

        results.append(result)

    output_csv = (
        OUTPUT_DIR
        / "component_filter_visual_comparison.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "COMPONENT FILTER VISUAL COMPARISON COMPLETE"
    )

    print("=" * 70)

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Summary CSV:",
        output_csv
    )

    print(
        "Original prediction masks remain unchanged."
    )

    print(
        "Locked test set remains untouched."
    )


if __name__ == "__main__":

    main()
