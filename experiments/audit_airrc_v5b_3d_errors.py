
"""
NEUROVESSEL V5B
3D FALSE-POSITIVE, FALSE-NEGATIVE AND CONNECTIVITY AUDIT

Dataset:
    AirRC validation partition only

Model:
    V5B Residual 3D U-Net

Prediction threshold:
    0.70

Analyses:
    1. Full-volume confusion metrics
    2. False-positive and false-negative volume
    3. 6-connected component analysis
    4. 26-connected component analysis
    5. Largest-component fraction
    6. Small-component statistics
    7. Predicted-component overlap with ground truth
    8. Error visualizations in axial, coronal and sagittal planes

IMPORTANT:
    - Does not train or modify the model.
    - Does not modify prediction masks.
    - Does not access the locked test partition.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import gc
import json
import re
import time

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from train_airrc_v5b import (
    load_splits,
    load_case,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

THRESHOLD = 0.70

MIN_COMPONENT_VOXELS = 10

MEDIUM_COMPONENT_VOXELS = 100

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v5b_fullvol"
    / "threshold_0.70"
)

OUTPUT_DIR = (
    METRICS_DIR
    / "airrc_v5b_3d_audit"
)

ERROR_VIS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / "airrc_v5b_3d_errors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ERROR_VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CONNECTIVITY STRUCTURES
# ============================================================

# 6-connectivity:
# Face-sharing neighbours only.

STRUCTURE_6 = ndi.generate_binary_structure(
    rank=3,
    connectivity=1
)

# 26-connectivity:
# Face, edge and corner neighbours.

STRUCTURE_26 = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def safe_filename(uid):

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(uid)
    )


def prediction_path_for(uid):

    filename = (
        safe_filename(uid)
        + "_pa_prediction.nii.gz"
    )

    return PREDICTION_DIR / filename


def check_geometry(prediction_obj, reference_obj):

    return (
        prediction_obj.GetSize()
        == reference_obj.GetSize()

        and prediction_obj.GetSpacing()
        == reference_obj.GetSpacing()

        and prediction_obj.GetOrigin()
        == reference_obj.GetOrigin()

        and prediction_obj.GetDirection()
        == reference_obj.GetDirection()
    )


def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ============================================================
# 5. CONNECTED COMPONENT ANALYSIS
# ============================================================

def component_summary(
    mask,
    structure,
    reference_mask=None,
    min_size=MIN_COMPONENT_VOXELS,
    medium_size=MEDIUM_COMPONENT_VOXELS
):

    """
    Calculates connected components in a 3D binary mask.

    The function reports statistics only.
    It does not remove any components.

    If reference_mask is provided, it also checks
    how many components intersect that reference mask.
    """

    foreground_voxels = int(
        np.count_nonzero(mask)
    )

    if foreground_voxels == 0:

        return {

            "component_count": 0,

            "largest_component_voxels": 0,

            "largest_component_fraction": 0.0,

            "median_component_voxels": 0.0,

            "mean_component_voxels": 0.0,

            "components_ge_10": 0,

            "components_ge_100": 0,

            "small_components_lt_10": 0,

            "small_component_voxel_fraction": 0.0,

            "top_5_component_sizes": "",

            "components_intersecting_reference": 0,

            "components_not_intersecting_reference": 0

        }

    # Allocate label array explicitly.
    labels = np.empty(
        mask.shape,
        dtype=np.int32
    )

    number_of_components = ndi.label(
        mask,
        structure=structure,
        output=labels
    )

    # Component sizes.
    counts = np.bincount(
        labels.ravel()
    )

    sizes = counts[1:]

    if sizes.size == 0:

        del labels

        return {
            "component_count": 0,
            "largest_component_voxels": 0,
            "largest_component_fraction": 0.0,
            "median_component_voxels": 0.0,
            "mean_component_voxels": 0.0,
            "components_ge_10": 0,
            "components_ge_100": 0,
            "small_components_lt_10": 0,
            "small_component_voxel_fraction": 0.0,
            "top_5_component_sizes": "",
            "components_intersecting_reference": 0,
            "components_not_intersecting_reference": 0
        }

    largest = int(
        sizes.max()
    )

    small_component_mask = (
        sizes < min_size
    )

    small_voxels = int(
        sizes[small_component_mask].sum()
    )

    # Top five component sizes.
    top_sizes = sorted(
        sizes.tolist(),
        reverse=True
    )[:5]

    # Check how many predicted components intersect
    # the reference mask.
    if reference_mask is not None:

        overlapping_labels = np.bincount(
            labels[reference_mask].ravel(),
            minlength=number_of_components + 1
        )

        intersecting = int(
            np.count_nonzero(
                overlapping_labels[1:]
            )
        )

        not_intersecting = (
            number_of_components - intersecting
        )

    else:

        intersecting = 0
        not_intersecting = 0

    result = {

        "component_count": int(
            number_of_components
        ),

        "largest_component_voxels": largest,

        "largest_component_fraction": safe_divide(
            largest,
            foreground_voxels
        ),

        "median_component_voxels": float(
            np.median(sizes)
        ),

        "mean_component_voxels": float(
            np.mean(sizes)
        ),

        "components_ge_10": int(
            np.count_nonzero(
                sizes >= min_size
            )
        ),

        "components_ge_100": int(
            np.count_nonzero(
                sizes >= medium_size
            )
        ),

        "small_components_lt_10": int(
            np.count_nonzero(
                small_component_mask
            )
        ),

        "small_component_voxel_fraction": (
            safe_divide(
                small_voxels,
                foreground_voxels
            )
        ),

        "top_5_component_sizes": ",".join(
            str(v) for v in top_sizes
        ),

        "components_intersecting_reference": (
            intersecting
        ),

        "components_not_intersecting_reference": (
            not_intersecting
        )

    }

    del labels
    del counts
    del sizes

    return result


# ============================================================
# 6. FULL-VOLUME ERROR METRICS
# ============================================================

def calculate_error_metrics(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        bool,
        copy=False
    )

    ground_truth = ground_truth.astype(
        bool,
        copy=False
    )

    tp = int(
        np.count_nonzero(
            prediction & ground_truth
        )
    )

    fp_mask = (
        prediction & ~ground_truth
    )

    fn_mask = (
        ground_truth & ~prediction
    )

    tn = int(
        ground_truth.size
        - tp
        - np.count_nonzero(fp_mask)
        - np.count_nonzero(fn_mask)
    )

    fp = int(
        np.count_nonzero(fp_mask)
    )

    fn = int(
        np.count_nonzero(fn_mask)
    )

    pred_count = int(
        np.count_nonzero(prediction)
    )

    gt_count = int(
        np.count_nonzero(ground_truth)
    )

    dice = safe_divide(
        2 * tp,
        pred_count + gt_count
    )

    precision = safe_divide(
        tp,
        pred_count
    )

    recall = safe_divide(
        tp,
        gt_count
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    fp_fraction = safe_divide(
        fp,
        pred_count
    )

    fn_fraction = safe_divide(
        fn,
        gt_count
    )

    pred_to_gt_ratio = safe_divide(
        pred_count,
        gt_count
    )

    results = {

        "dice": dice,

        "precision": precision,

        "recall": recall,

        "specificity": specificity,

        "true_positive_voxels": tp,

        "false_positive_voxels": fp,

        "false_negative_voxels": fn,

        "true_negative_voxels": tn,

        "predicted_voxels": pred_count,

        "ground_truth_voxels": gt_count,

        "fp_fraction_of_prediction": fp_fraction,

        "fn_fraction_of_ground_truth": fn_fraction,

        "prediction_to_ground_truth_ratio": (
            pred_to_gt_ratio
        )

    }

    return results, fp_mask, fn_mask


# ============================================================
# 7. SINGLE PATIENT 3D AUDIT
# ============================================================

def audit_patient(
    record,
    threshold=THRESHOLD
):

    uid = record["uid"]

    print("\n" + "=" * 65)

    print(
        "AUDITING PATIENT:",
        uid
    )

    # --------------------------------------------------------
    # Load processed CT and GT mask.
    # --------------------------------------------------------

    image, mask = load_case(record)

    ground_truth = (
        mask > 0
    )

    # --------------------------------------------------------
    # Load V5B prediction.
    # --------------------------------------------------------

    pred_path = prediction_path_for(uid)

    if not pred_path.is_file():

        raise FileNotFoundError(
            f"Prediction not found: {pred_path}"
        )

    prediction_obj = sitk.ReadImage(
        str(pred_path)
    )

    reference_obj = sitk.ReadImage(
        str(record["image"])
    )

    geometry_ok = check_geometry(
        prediction_obj,
        reference_obj
    )

    if not geometry_ok:

        raise RuntimeError(
            f"Geometry mismatch for patient: {uid}"
        )

    prediction = sitk.GetArrayFromImage(
        prediction_obj
    )

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            f"Shape mismatch for patient: {uid}. "
            f"Prediction={prediction.shape}, "
            f"GT={ground_truth.shape}"
        )

    prediction = (
        prediction > 0
    )

    # Physical voxel volume in mL.
    spacing = reference_obj.GetSpacing()

    voxel_volume_ml = (
        float(np.prod(spacing)) / 1000.0
    )

    # --------------------------------------------------------
    # Confusion and error metrics.
    # --------------------------------------------------------

    metrics, fp_mask, fn_mask = (
        calculate_error_metrics(
            prediction,
            ground_truth
        )
    )

    pred_voxels = metrics[
        "predicted_voxels"
    ]

    gt_voxels = metrics[
        "ground_truth_voxels"
    ]

    print(
        "Full-volume Dice:",
        round(metrics["dice"], 5)
    )

    print(
        "False-positive fraction:",
        round(
            metrics["fp_fraction_of_prediction"],
            5
        )
    )

    print(
        "False-negative fraction:",
        round(
            metrics["fn_fraction_of_ground_truth"],
            5
        )
    )

    # --------------------------------------------------------
    # 3D connected-component analysis.
    # --------------------------------------------------------

    print(
        "\nRunning 3D connected-component analysis..."
    )

    # Predicted mask: 26 connectivity.
    pred_26 = component_summary(
        prediction,
        STRUCTURE_26,
        reference_mask=ground_truth
    )

    # Ground-truth mask: 26 connectivity.
    gt_26 = component_summary(
        ground_truth,
        STRUCTURE_26
    )

    # Prediction: strict 6 connectivity.
    pred_6 = component_summary(
        prediction,
        STRUCTURE_6
    )

    # Ground truth: strict 6 connectivity.
    gt_6 = component_summary(
        ground_truth,
        STRUCTURE_6
    )

    # False-positive mask: 26 connectivity.
    fp_26 = component_summary(
        fp_mask,
        STRUCTURE_26
    )

    # False-negative mask: 26 connectivity.
    fn_26 = component_summary(
        fn_mask,
        STRUCTURE_26
    )

    print(
        "Predicted components (26-connectivity):",
        pred_26["component_count"]
    )

    print(
        "Ground-truth components (26-connectivity):",
        gt_26["component_count"]
    )

    print(
        "Largest predicted component fraction:",
        round(
            pred_26["largest_component_fraction"],
            5
        )
    )

    print(
        "Predicted components not intersecting GT:",
        pred_26[
            "components_not_intersecting_reference"
        ]
    )

    # --------------------------------------------------------
    # Assemble patient report.
    # --------------------------------------------------------

    row = {

        "uid": uid,

        "patient_id": record["patient_id"],

        "threshold": threshold,

        "geometry_matches": geometry_ok,

        "shape_z": image.shape[0],

        "shape_y": image.shape[1],

        "shape_x": image.shape[2],

        "spacing_x_mm": spacing[0],

        "spacing_y_mm": spacing[1],

        "spacing_z_mm": spacing[2],

        "voxel_volume_ml": voxel_volume_ml,

        **metrics,

        "predicted_volume_ml": (
            pred_voxels * voxel_volume_ml
        ),

        "ground_truth_volume_ml": (
            gt_voxels * voxel_volume_ml
        ),

        "false_positive_volume_ml": (
            metrics["false_positive_voxels"]
            * voxel_volume_ml
        ),

        "false_negative_volume_ml": (
            metrics["false_negative_voxels"]
            * voxel_volume_ml
        )

    }

    # Prefix component information.
    for key, value in pred_26.items():

        row[
            "pred_26_" + key
        ] = value

    for key, value in gt_26.items():

        row[
            "gt_26_" + key
        ] = value

    for key, value in pred_6.items():

        row[
            "pred_6_" + key
        ] = value

    for key, value in gt_6.items():

        row[
            "gt_6_" + key
        ] = value

    for key, value in fp_26.items():

        row[
            "fp_26_" + key
        ] = value

    for key, value in fn_26.items():

        row[
            "fn_26_" + key
        ] = value

    # Cleanup.
    del image
    del mask
    del ground_truth
    del prediction
    del prediction_obj
    del reference_obj

    del fp_mask
    del fn_mask

    del pred_26
    del gt_26
    del pred_6
    del gt_6
    del fp_26
    del fn_26

    gc.collect()

    return row


# ============================================================
# 8. AGGREGATE SUMMARY
# ============================================================

def create_aggregate_summary(df):

    columns = [

        "dice",

        "precision",

        "recall",

        "specificity",

        "fp_fraction_of_prediction",

        "fn_fraction_of_ground_truth",

        "prediction_to_ground_truth_ratio",

        "pred_26_component_count",

        "gt_26_component_count",

        "pred_26_largest_component_fraction",

        "pred_26_small_components_lt_10",

        "fp_26_component_count",

        "fn_26_component_count",

        "pred_26_components_not_intersecting_reference"

    ]

    summary = {

        "project": "NeuroVessel",

        "model": "V5B Residual 3D U-Net",

        "audit": "3D error and connected-component analysis",

        "threshold": THRESHOLD,

        "patients_evaluated": int(len(df)),

        "locked_test_set_loaded": False,

        "connectivity_definition": (
            "6-connectivity uses face neighbours; "
            "26-connectivity includes face, edge "
            "and corner neighbours."
        ),

        "minimum_component_size_reported": (
            MIN_COMPONENT_VOXELS
        ),

        "component_pruning_applied": False,

        "metrics": {},

        "interpretation_limitations": [

            "Connected components are voxel-connectivity "
            "measures, not anatomical branch labels.",

            "Disconnected regions can reflect annotation "
            "gaps as well as model fragmentation.",

            "A predicted component intersecting the GT "
            "is not necessarily anatomically correct.",

            "Centerline and branch-level topology require "
            "a separate analysis."

        ]

    }

    for column in columns:

        values = df[column].astype(
            float
        )

        summary["metrics"][column] = {

            "mean": float(values.mean()),

            "median": float(values.median()),

            "std": float(values.std(ddof=1)),

            "minimum": float(values.min()),

            "maximum": float(values.max())

        }

    summary["total_false_positive_voxels"] = int(
        df["false_positive_voxels"].sum()
    )

    summary["total_false_negative_voxels"] = int(
        df["false_negative_voxels"].sum()
    )

    summary["total_predicted_voxels"] = int(
        df["predicted_voxels"].sum()
    )

    summary["total_ground_truth_voxels"] = int(
        df["ground_truth_voxels"].sum()
    )

    summary["macro_dice"] = float(
        df["dice"].mean()
    )

    # Pooled micro Dice.
    total_tp = int(
        df["true_positive_voxels"].sum()
    )

    total_pred = int(
        df["predicted_voxels"].sum()
    )

    total_gt = int(
        df["ground_truth_voxels"].sum()
    )

    summary["micro_dice"] = safe_divide(
        2 * total_tp,
        total_pred + total_gt
    )

    return summary


# ============================================================
# 9. ERROR VISUALIZATION
# ============================================================

def create_error_figure(
    record,
    patient_metrics
):

    uid = record["uid"]

    print(
        "\nCreating error visualization:",
        uid
    )

    image, mask = load_case(record)

    ground_truth = mask > 0

    prediction_obj = sitk.ReadImage(
        str(prediction_path_for(uid))
    )

    prediction = (
        sitk.GetArrayFromImage(
            prediction_obj
        ) > 0
    )

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            f"Visualization shape mismatch: {uid}"
        )

    # Error masks.
    fp_mask = (
        prediction & ~ground_truth
    )

    fn_mask = (
        ground_truth & ~prediction
    )

    # Select slice containing maximum combined error
    # in each of the three anatomical planes.

    z = int(
        np.argmax(
            (
                np.count_nonzero(
                    fp_mask,
                    axis=(1, 2)
                )
                +
                np.count_nonzero(
                    fn_mask,
                    axis=(1, 2)
                )
            )
        )
    )

    y = int(
        np.argmax(
            (
                np.count_nonzero(
                    fp_mask,
                    axis=(0, 2)
                )
                +
                np.count_nonzero(
                    fn_mask,
                    axis=(0, 2)
                )
            )
        )
    )

    x = int(
        np.argmax(
            (
                np.count_nonzero(
                    fp_mask,
                    axis=(0, 1)
                )
                +
                np.count_nonzero(
                    fn_mask,
                    axis=(0, 1)
                )
            )
        )
    )

    # Extract slices.
    planes = [

        (
            "Axial",
            z,
            image[z, :, :],
            ground_truth[z, :, :],
            prediction[z, :, :],
            fp_mask[z, :, :],
            fn_mask[z, :, :]
        ),

        (
            "Coronal",
            y,
            image[:, y, :],
            ground_truth[:, y, :],
            prediction[:, y, :],
            fp_mask[:, y, :],
            fn_mask[:, y, :]
        ),

        (
            "Sagittal",
            x,
            image[:, :, x],
            ground_truth[:, :, x],
            prediction[:, :, x],
            fp_mask[:, :, x],
            fn_mask[:, :, x]
        )

    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    dice = patient_metrics["dice"]

    fig.suptitle(
        "NeuroVessel V5B — 3D Error Audit\n"
        f"Patient: {uid}\n"
        f"Full-volume Dice @ {THRESHOLD:.2f}: {dice:.4f}",
        fontsize=14,
        fontweight="bold"
    )

    for column, plane in enumerate(planes):

        (
            name,
            index,
            ct_slice,
            gt_slice,
            pred_slice,
            fp_slice,
            fn_slice
        ) = plane

        ct_slice = np.clip(
            ct_slice,
            -1000,
            1000
        )

        # ----------------------------------------------------
        # Top row: GT and prediction contours.
        # ----------------------------------------------------

        ax = axes[0, column]

        ax.imshow(
            ct_slice,
            cmap="gray",
            vmin=-1000,
            vmax=1000,
            origin="lower"
        )

        if np.any(gt_slice):

            ax.contour(
                gt_slice.astype(float),
                levels=[0.5],
                colors=["lime"],
                linewidths=1.0,
                origin="lower"
            )

        if np.any(pred_slice):

            ax.contour(
                pred_slice.astype(float),
                levels=[0.5],
                colors=["red"],
                linewidths=0.9,
                origin="lower"
            )

        ax.set_title(
            f"{name} | index={index}\n"
            "Ground Truth vs Prediction"
        )

        ax.axis("off")

        # ----------------------------------------------------
        # Bottom row: FP and FN regions.
        # ----------------------------------------------------

        ax = axes[1, column]

        ax.imshow(
            ct_slice,
            cmap="gray",
            vmin=-1000,
            vmax=1000,
            origin="lower"
        )

        rgba = np.zeros(
            (
                fp_slice.shape[0],
                fp_slice.shape[1],
                4
            ),
            dtype=np.float32
        )

        # Red = False positive.
        rgba[fp_slice] = [
            1.0,
            0.05,
            0.05,
            0.70
        ]

        # Cyan = False negative.
        rgba[fn_slice] = [
            0.0,
            0.85,
            1.0,
            0.75
        ]

        ax.imshow(
            rgba,
            origin="lower"
        )

        ax.set_title(
            f"{name} | FP (red), FN (cyan)"
        )

        ax.axis("off")

    legend_elements = [

        Line2D(
            [0], [0],
            color="lime",
            lw=2,
            label="Ground Truth"
        ),

        Line2D(
            [0], [0],
            color="red",
            lw=2,
            label="Prediction"
        ),

        Patch(
            facecolor=(1, 0.05, 0.05, 0.7),
            label="False Positive"
        ),

        Patch(
            facecolor=(0, 0.85, 1, 0.75),
            label="False Negative"
        )

    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False
    )

    plt.tight_layout(
        rect=[0, 0.06, 1, 0.92]
    )

    output_path = (
        ERROR_VIS_DIR
        / f"error_audit_{safe_filename(uid)}.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path
    )

    del image
    del mask
    del ground_truth
    del prediction
    del prediction_obj
    del fp_mask
    del fn_mask

    gc.collect()

    return output_path


# ============================================================
# 10. SELECT CASES FOR VISUAL REVIEW
# ============================================================

def select_visual_cases(df):

    selected = {}

    # Three worst Dice cases.
    worst = df.sort_values(
        "dice",
        ascending=True
    ).head(3)

    for _, row in worst.iterrows():

        selected[row["uid"]] = row

    # Strongest case.
    best = df.sort_values(
        "dice",
        ascending=False
    ).iloc[0]

    selected[best["uid"]] = best

    # Highest false-positive fraction.
    high_fp = df.sort_values(
        "fp_fraction_of_prediction",
        ascending=False
    ).iloc[0]

    selected[high_fp["uid"]] = high_fp

    # Highest false-negative fraction.
    high_fn = df.sort_values(
        "fn_fraction_of_ground_truth",
        ascending=False
    ).iloc[0]

    selected[high_fn["uid"]] = high_fn

    return list(
        selected.values()
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)

    print(
        "NEUROVESSEL V5B — 3D ERROR AND CONNECTIVITY AUDIT"
    )

    print("=" * 70)

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Prediction directory:",
        PREDICTION_DIR
    )

    if not PREDICTION_DIR.is_dir():

        raise FileNotFoundError(
            f"Prediction directory missing: "
            f"{PREDICTION_DIR}"
        )

    # --------------------------------------------------------
    # Load validation split only.
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

    print(
        "\nConnectivity:",
        "6-connected and 26-connected"
    )

    print(
        "Small component threshold:",
        MIN_COMPONENT_VOXELS,
        "voxels"
    )

    # --------------------------------------------------------
    # Verify prediction files before starting.
    # --------------------------------------------------------

    missing = []

    for record in validation_records:

        path = prediction_path_for(
            record["uid"]
        )

        if not path.is_file():

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "Missing prediction files:\n"
            + "\n".join(missing[:10])
        )

    # --------------------------------------------------------
    # Audit all patients.
    # --------------------------------------------------------

    patient_rows = []

    for index, record in enumerate(
        validation_records,
        start=1
    ):

        print(
            f"\nPATIENT {index}/"
            f"{len(validation_records)}"
        )

        row = audit_patient(
            record,
            THRESHOLD
        )

        patient_rows.append(row)

        print(
            f"Completed {index}/"
            f"{len(validation_records)}",
            flush=True
        )

    # --------------------------------------------------------
    # Save per-patient report.
    # --------------------------------------------------------

    df = pd.DataFrame(
        patient_rows
    )

    per_patient_path = (
        OUTPUT_DIR
        / "per_patient_3d_error_audit.csv"
    )

    df.to_csv(
        per_patient_path,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate summary.
    # --------------------------------------------------------

    aggregate = create_aggregate_summary(
        df
    )

    aggregate["runtime_seconds"] = (
        time.time() - start_time
    )

    aggregate_path = (
        OUTPUT_DIR
        / "aggregate_3d_error_summary.json"
    )

    with open(
        aggregate_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            aggregate,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Visual selection.
    # --------------------------------------------------------

    selected_rows = select_visual_cases(
        df
    )

    record_lookup = {
        record["uid"]: record
        for record in validation_records
    }

    visual_rows = []

    for index, row in enumerate(
        selected_rows,
        start=1
    ):

        uid = row["uid"]

        print(
            f"\nERROR VISUALIZATION "
            f"{index}/{len(selected_rows)}"
        )

        output_path = create_error_figure(
            record_lookup[uid],
            row
        )

        visual_rows.append({

            "uid": uid,

            "dice": row["dice"],

            "fp_fraction_of_prediction": (
                row["fp_fraction_of_prediction"]
            ),

            "fn_fraction_of_ground_truth": (
                row["fn_fraction_of_ground_truth"]
            ),

            "visualization": str(output_path)

        })

    visual_csv = (
        OUTPUT_DIR
        / "selected_error_visualizations.csv"
    )

    pd.DataFrame(
        visual_rows
    ).to_csv(
        visual_csv,
        index=False
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "NEUROVESSEL V5B — 3D AUDIT SUMMARY"
    )

    print("=" * 75)

    display_columns = [

        "uid",

        "dice",

        "fp_fraction_of_prediction",

        "fn_fraction_of_ground_truth",

        "pred_26_component_count",

        "pred_26_largest_component_fraction",

        "pred_26_components_not_intersecting_reference"

    ]

    print(
        df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nAGGREGATE RESULTS")

    print(
        "Patients:",
        len(df)
    )

    print(
        "Macro full-volume Dice:",
        round(
            aggregate["macro_dice"],
            5
        )
    )

    print(
        "Micro full-volume Dice:",
        round(
            aggregate["micro_dice"],
            5
        )
    )

    print(
        "Total false-positive voxels:",
        aggregate["total_false_positive_voxels"]
    )

    print(
        "Total false-negative voxels:",
        aggregate["total_false_negative_voxels"]
    )

    print(
        "Mean predicted 26-connected components:",
        round(
            df["pred_26_component_count"].mean(),
            2
        )
    )

    print(
        "Mean predicted largest-component fraction:",
        round(
            df[
                "pred_26_largest_component_fraction"
            ].mean(),
            5
        )
    )

    print("\nOUTPUTS")

    print(
        "Patient audit:",
        per_patient_path
    )

    print(
        "Aggregate summary:",
        aggregate_path
    )

    print(
        "Error visualizations:",
        ERROR_VIS_DIR
    )

    print(
        "Visualization selection:",
        visual_csv
    )

    print(
        "\nRuntime:",
        round(
            aggregate["runtime_seconds"] / 60.0,
            2
        ),
        "minutes"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No segmentation components were removed."
    )

    print(
        "No model parameters were changed."
    )

    print(
        "The locked test set remains untouched."
    )

    print(
        "Connected-component measurements are not "
        "a substitute for anatomical branch analysis."
    )


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
