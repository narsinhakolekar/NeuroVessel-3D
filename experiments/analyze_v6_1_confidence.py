
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from scipy import ndimage

# Import the exact existing V6 architecture and split loader.
from train_airrc_v6_multiclass import (
    NeuroVesselV6,
    load_v6_splits,
)


# ============================================================
# NEUROVESSEL V6.1 — PREDICTION CONFIDENCE ANALYSIS
#
# Existing checkpoint: V6.1 epoch 29
# Task: 3-class BG / PA / PV segmentation
#
# Three difficult validation patients:
#   0529: PV false positives
#   0763: PV false positives
#   0848: PA false negatives
#
# Outputs:
#   1. confidence_group_summary.csv
#   2. confidence_calibration_bins.csv
#   3. confidence_matched_pairs.csv
#   4. confidence_matched_summary.csv
#   5. confidence_inference_checks.csv
#   6. Three probability distribution figures
#
# Read-only diagnostic.
# No training or checkpoint modification.
# Locked test partition is never loaded.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT / "outputs" / "checkpoints"
    / "airrc_v6_1_multiclass_best.pt"
)

PRED_DIR = (
    ROOT / "outputs" / "predictions"
    / "airrc_v6_1_fullvol"
)

LABEL_DIR = (
    ROOT / "data" / "processed_airrc_v6"
    / "labels_multiclass"
)

SPATIAL_DIR = (
    ROOT / "outputs" / "metrics"
    / "airrc_v6_1_spatial_hard_negative"
)

PAIR_INPUT = (
    SPATIAL_DIR / "spatial_matched_pairs.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics"
    / "airrc_v6_1_confidence_analysis"
)

FIGURE_DIR = (
    ROOT / "outputs" / "visualizations"
    / "airrc_v6_1_confidence_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Model and inference settings
# ------------------------------------------------------------

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "Background",
    1: "PA",
    2: "PV",
}

PATCH_SIZE = 96

ROI_SIZE = (96, 96, 96)

SW_BATCH_SIZE = 1

OVERLAP = 0.5

AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_OUTPUT = torch.device("cpu")


# ------------------------------------------------------------
# Diagnostic settings
# ------------------------------------------------------------

STRUCTURE = ndimage.generate_binary_structure(3, 3)

ROI_MARGIN = 24

NEIGHBORHOOD_ITERATIONS = 20

MAX_GROUP_SAMPLES = 15000

RANDOM_SEED = 2026

CALIBRATION_BINS = 10

MIN_ARGMAX_AGREEMENT = 0.99


# ------------------------------------------------------------
# Case definitions
# component IDs are based on IJK/NIfTI array order.
# Arrays below are reconstructed in that same order.
# ------------------------------------------------------------

CASES = [
    {
        "patient": "LIDC-IDRI-0529",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",
        "class_name": "PV",
        "class_id": 2,
        "component_id": 763,
        "error_type": "FP",
    },
    {
        "patient": "LIDC-IDRI-0763",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",
        "class_name": "PV",
        "class_id": 2,
        "component_id": 16,
        "error_type": "FP",
    },
    {
        "patient": "LIDC-IDRI-0848",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
        "class_name": "PA",
        "class_id": 1,
        "component_id": 1,
        "error_type": "FN",
    },
]


GROUP_ROWS = []
BIN_ROWS = []
PAIR_ROWS = []
PAIR_SUMMARY_ROWS = []
INFERENCE_ROWS = []


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def validate_geometry(reference, other, description):

    if reference.GetSize() != other.GetSize():
        raise ValueError(
            f"Size mismatch: {description}"
        )

    checks = [
        (
            reference.GetSpacing(),
            other.GetSpacing(),
            1e-5,
            "spacing",
        ),
        (
            reference.GetOrigin(),
            other.GetOrigin(),
            1e-4,
            "origin",
        ),
        (
            reference.GetDirection(),
            other.GetDirection(),
            1e-5,
            "direction",
        ),
    ]

    for first, second, tolerance, name in checks:

        if not np.allclose(
            first,
            second,
            atol=tolerance,
        ):
            raise ValueError(
                f"{name} mismatch: {description}"
            )


def roi_slices(mask, margin):

    coords = np.where(mask)

    if coords[0].size == 0:
        raise ValueError("Cannot create an ROI from an empty mask.")

    result = []

    for axis in range(3):

        start = max(
            0,
            int(coords[axis].min()) - margin,
        )

        stop = min(
            mask.shape[axis],
            int(coords[axis].max()) + margin + 1,
        )

        result.append(
            slice(start, stop)
        )

    return tuple(result)


def sample_coordinates(mask, maximum, rng):

    flat = np.flatnonzero(mask)

    if flat.size == 0:

        return np.empty(
            (0, 3),
            dtype=np.int64,
        )

    if flat.size > maximum:

        selected = rng.choice(
            flat,
            size=maximum,
            replace=False,
        )

    else:
        selected = flat

    coordinates = np.column_stack(
        np.unravel_index(
            selected,
            mask.shape,
        )
    )

    return coordinates.astype(
        np.int64,
        copy=False,
    )


def extract_probabilities(logits, coordinates_dhw):

    """
    Extract only selected voxel probabilities.

    logits shape:
        (1, 3, D, H, W)

    coordinates:
        (N, 3) in D,H,W order

    Returns:
        (N, 3) probabilities for BG, PA and PV.
    """

    if len(coordinates_dhw) == 0:

        return np.empty(
            (0, NUM_CLASSES),
            dtype=np.float32,
        )

    coords = np.asarray(
        coordinates_dhw,
        dtype=np.int64,
    )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 coordinates; got {coords.shape}"
        )

    depth, height, width = logits.shape[-3:]

    if (
        np.any(coords[:, 0] < 0)
        or np.any(coords[:, 0] >= depth)
        or np.any(coords[:, 1] < 0)
        or np.any(coords[:, 1] >= height)
        or np.any(coords[:, 2] < 0)
        or np.any(coords[:, 2] >= width)
    ):
        raise ValueError(
            "Confidence coordinates are outside the logit volume."
        )

    index = torch.as_tensor(
        coords,
        dtype=torch.long,
        device="cpu",
    )

    d = index[:, 0]
    h = index[:, 1]
    w = index[:, 2]

    # Convert logits to D,H,W,C for straightforward indexing.
    logits_dhwc = logits[0].permute(
        1, 2, 3, 0
    )

    selected_logits = logits_dhwc[
        d, h, w, :
    ].float()

    probabilities = torch.softmax(
        selected_logits,
        dim=1,
    )

    return probabilities.numpy()


def entropy_values(probabilities):

    p = np.clip(
        probabilities,
        1e-8,
        1.0,
    )

    entropy = -np.sum(
        p * np.log(p),
        axis=1,
    )

    return entropy / np.log(NUM_CLASSES)


def calibration_bins(
    patient,
    class_name,
    group_name,
    confidence,
    accuracy,
):

    bin_ids = np.minimum(
        (confidence * CALIBRATION_BINS).astype(int),
        CALIBRATION_BINS - 1,
    )

    ece = 0.0

    total = len(confidence)

    if total == 0:
        return np.nan

    for bin_id in range(CALIBRATION_BINS):

        selected = bin_ids == bin_id

        count = int(
            np.count_nonzero(selected)
        )

        if count == 0:
            continue

        mean_confidence = float(
            np.mean(confidence[selected])
        )

        mean_accuracy = float(
            np.mean(accuracy[selected])
        )

        fraction = count / total

        ece += fraction * abs(
            mean_accuracy - mean_confidence
        )

        BIN_ROWS.append({
            "patient_id": patient,
            "class": class_name,
            "group": group_name,
            "bin": bin_id,
            "bin_lower": bin_id / CALIBRATION_BINS,
            "bin_upper": (bin_id + 1) / CALIBRATION_BINS,
            "count": count,
            "mean_confidence": mean_confidence,
            "mean_accuracy": mean_accuracy,
            "absolute_calibration_gap": abs(
                mean_accuracy - mean_confidence
            ),
        })

    return float(ece)


# ============================================================
# 2. CONFIDENCE STATISTICS FOR A GROUP
# ============================================================

def analyze_group(
    patient,
    class_name,
    class_id,
    group_name,
    mask_ijk,
    roi_origin_ijk,
    logits,
    pred_dhw,
    gt_dhw,
    rng,
):

    total_voxels = int(
        np.count_nonzero(mask_ijk)
    )

    # Convert the ROI mask from I,J,K to D,H,W.
    mask_dhw = np.transpose(
        mask_ijk,
        (2, 1, 0),
    )

    local_coords_dhw = sample_coordinates(
        mask_dhw,
        MAX_GROUP_SAMPLES,
        rng,
    )

    sampled_count = len(local_coords_dhw)

    if sampled_count == 0:

        print(
            f"  {group_name}: empty"
        )

        GROUP_ROWS.append({
            "patient_id": patient,
            "class": class_name,
            "group": group_name,
            "total_voxels": total_voxels,
            "sampled_voxels": 0,
        })

        return None

    # ROI origin is I,J,K.
    # Model coordinates are D,H,W.
    roi_origin_dhw = roi_origin_ijk[::-1]

    global_coords_dhw = (
        local_coords_dhw
        + roi_origin_dhw
    )

    probabilities = extract_probabilities(
        logits,
        global_coords_dhw,
    )

    # Original model predictions and GT at the same voxels.
    d = global_coords_dhw[:, 0]
    h = global_coords_dhw[:, 1]
    w = global_coords_dhw[:, 2]

    saved_prediction = pred_dhw[d, h, w]
    ground_truth = gt_dhw[d, h, w]

    calculated_prediction = np.argmax(
        probabilities,
        axis=1,
    )

    agreement = (
        calculated_prediction == saved_prediction
    )

    accuracy = (
        calculated_prediction == ground_truth
    ).astype(np.float32)

    max_confidence = probabilities.max(
        axis=1
    )

    target_probability = probabilities[
        :,
        class_id,
    ]

    gt_probability = probabilities[
        np.arange(sampled_count),
        ground_truth,
    ]

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    entropy = entropy_values(
        probabilities
    )

    one_hot = np.eye(
        NUM_CLASSES,
        dtype=np.float32,
    )[ground_truth]

    brier = np.sum(
        (probabilities - one_hot) ** 2,
        axis=1,
    )

    nll = -np.log(
        np.clip(
            gt_probability,
            1e-8,
            1.0,
        )
    )

    ece = calibration_bins(
        patient,
        class_name,
        group_name,
        max_confidence,
        accuracy,
    )

    row = {
        "patient_id": patient,
        "class": class_name,
        "group": group_name,

        "total_voxels": total_voxels,
        "sampled_voxels": sampled_count,

        "prediction_agreement_with_saved": float(
            np.mean(agreement)
        ),

        "accuracy_vs_ground_truth": float(
            np.mean(accuracy)
        ),

        "mean_predicted_confidence": float(
            np.mean(max_confidence)
        ),

        "median_predicted_confidence": float(
            np.median(max_confidence)
        ),

        "fraction_confidence_ge_0_80": float(
            np.mean(max_confidence >= 0.80)
        ),

        "fraction_confidence_ge_0_90": float(
            np.mean(max_confidence >= 0.90)
        ),

        "mean_target_class_probability": float(
            np.mean(target_probability)
        ),

        "median_target_class_probability": float(
            np.median(target_probability)
        ),

        "mean_gt_class_probability": float(
            np.mean(gt_probability)
        ),

        "mean_p_background": float(
            np.mean(probabilities[:, 0])
        ),

        "mean_p_pa": float(
            np.mean(probabilities[:, 1])
        ),

        "mean_p_pv": float(
            np.mean(probabilities[:, 2])
        ),

        "mean_entropy_normalized": float(
            np.mean(entropy)
        ),

        "median_entropy_normalized": float(
            np.median(entropy)
        ),

        "mean_top1_top2_margin": float(
            np.mean(margin)
        ),

        "multiclass_brier_score": float(
            np.mean(brier)
        ),

        "mean_negative_log_likelihood": float(
            np.mean(nll)
        ),

        "expected_calibration_error": ece,
    }

    GROUP_ROWS.append(row)

    print(
        f"  {group_name}: "
        f"n={total_voxels:,}, "
        f"sampled={sampled_count:,}, "
        f"mean confidence={row['mean_predicted_confidence']:.4f}, "
        f"target P={row['mean_target_class_probability']:.4f}, "
        f"accuracy={row['accuracy_vs_ground_truth']:.4f}, "
        f"agreement={row['prediction_agreement_with_saved']:.5f}"
    )

    return {
        "coordinates": global_coords_dhw,
        "probabilities": probabilities,
        "saved_prediction": saved_prediction,
        "ground_truth": ground_truth,
        "accuracy": accuracy,
        "confidence": max_confidence,
        "target_probability": target_probability,
        "entropy": entropy,
    }


# ============================================================
# 3. MATCHED-PAIR PROBABILITY ANALYSIS
# ============================================================

def analyze_matched_pairs(
    patient,
    case,
    pair_table,
    masks_ijk,
    roi_origin_ijk,
    logits,
    pred_dhw,
    gt_dhw,
):

    subset = pair_table[
        pair_table["patient_id"] == patient
    ].copy()

    if subset.empty:

        print(
            "No spatial matched pairs found for:",
            patient,
        )

        return

    required_columns = [
        "error_group",
        "control_group",
        "error_i",
        "error_j",
        "error_k",
        "control_i",
        "control_j",
        "control_k",
        "match_distance",
    ]

    missing = (
        set(required_columns)
        - set(subset.columns)
    )

    if missing:
        raise ValueError(
            f"Matched-pair CSV missing columns: {sorted(missing)}"
        )

    class_id = case["class_id"]

    # Global IJK coordinates from the spatial analysis.
    error_ijk = subset[
        ["error_i", "error_j", "error_k"]
    ].to_numpy(dtype=np.int64)

    control_ijk = subset[
        ["control_i", "control_j", "control_k"]
    ].to_numpy(dtype=np.int64)

    # Convert IJK -> DHW.
    error_dhw = error_ijk[:, [2, 1, 0]]

    control_dhw = control_ijk[:, [2, 1, 0]]

    # Check every pair against its original ROI masks.
    error_local = (
        error_ijk - roi_origin_ijk
    )

    control_local = (
        control_ijk - roi_origin_ijk
    )

    roi_shape = next(
        iter(masks_ijk.values())
    ).shape

    for coords, description in [
        (error_local, "error"),
        (control_local, "control"),
    ]:

        outside = np.any(
            (coords < 0)
            | (coords >= np.asarray(roi_shape)),
            axis=1,
        )

        if np.any(outside):

            raise ValueError(
                f"{description} coordinates outside ROI "
                f"for {patient}"
            )

    for index, row in enumerate(
        subset.itertuples(index=False)
    ):

        error_name = row.error_group
        control_name = row.control_group

        if error_name not in masks_ijk:
            raise ValueError(
                f"Unknown error group: {error_name}"
            )

        if control_name not in masks_ijk:
            raise ValueError(
                f"Unknown control group: {control_name}"
            )

        ei, ej, ek = error_local[index]
        ci, cj, ck = control_local[index]

        if not masks_ijk[error_name][ei, ej, ek]:
            raise ValueError(
                f"Error pair does not match its mask: "
                f"{patient}, row {index}"
            )

        if not masks_ijk[control_name][ci, cj, ck]:
            raise ValueError(
                f"Control pair does not match its mask: "
                f"{patient}, row {index}"
            )

    print(
        f"\nMatched probability pairs: {len(subset):,}"
    )

    # Extract probabilities for both members of every pair.
    error_prob = extract_probabilities(
        logits,
        error_dhw,
    )

    control_prob = extract_probabilities(
        logits,
        control_dhw,
    )

    error_pred = np.argmax(
        error_prob,
        axis=1,
    )

    control_pred = np.argmax(
        control_prob,
        axis=1,
    )

    error_saved = pred_dhw[
        error_dhw[:, 0],
        error_dhw[:, 1],
        error_dhw[:, 2],
    ]

    control_saved = pred_dhw[
        control_dhw[:, 0],
        control_dhw[:, 1],
        control_dhw[:, 2],
    ]

    error_gt = gt_dhw[
        error_dhw[:, 0],
        error_dhw[:, 1],
        error_dhw[:, 2],
    ]

    control_gt = gt_dhw[
        control_dhw[:, 0],
        control_dhw[:, 1],
        control_dhw[:, 2],
    ]

    error_confidence = error_prob.max(axis=1)
    control_confidence = control_prob.max(axis=1)

    error_entropy = entropy_values(error_prob)
    control_entropy = entropy_values(control_prob)

    error_target = error_prob[:, class_id]
    control_target = control_prob[:, class_id]

    # Append probability values to original matched pairs.
    for class_number, class_name in CLASS_NAMES.items():

        subset[f"error_p_{class_name.lower()}"] = (
            error_prob[:, class_number]
        )

        subset[f"control_p_{class_name.lower()}"] = (
            control_prob[:, class_number]
        )

    subset["error_predicted_class"] = error_pred
    subset["control_predicted_class"] = control_pred

    subset["error_ground_truth_class"] = error_gt
    subset["control_ground_truth_class"] = control_gt

    subset["error_max_probability"] = error_confidence
    subset["control_max_probability"] = control_confidence

    subset["error_target_probability"] = error_target
    subset["control_target_probability"] = control_target

    subset["delta_target_probability"] = (
        error_target - control_target
    )

    subset["delta_max_probability"] = (
        error_confidence - control_confidence
    )

    subset["error_entropy"] = error_entropy
    subset["control_entropy"] = control_entropy

    subset["delta_entropy"] = (
        error_entropy - control_entropy
    )

    subset["error_correct"] = (
        error_pred == error_gt
    )

    subset["control_correct"] = (
        control_pred == control_gt
    )

    subset["error_saved_prediction_agreement"] = (
        error_pred == error_saved
    )

    subset["control_saved_prediction_agreement"] = (
        control_pred == control_saved
    )

    PAIR_ROWS.extend(
        subset.to_dict("records")
    )

    # Summary for each matched comparison.
    group_columns = [
        "error_group",
        "control_group",
    ]

    for key, group in subset.groupby(
        group_columns
    ):

        error_name, control_name = key

        PAIR_SUMMARY_ROWS.append({
            "patient_id": patient,
            "class": case["class_name"],
            "error_group": error_name,
            "control_group": control_name,
            "matched_pairs": len(group),

            "error_target_probability_median": float(
                group["error_target_probability"].median()
            ),

            "control_target_probability_median": float(
                group["control_target_probability"].median()
            ),

            "median_target_probability_difference": float(
                group["delta_target_probability"].median()
            ),

            "error_max_confidence_median": float(
                group["error_max_probability"].median()
            ),

            "control_max_confidence_median": float(
                group["control_max_probability"].median()
            ),

            "error_entropy_median": float(
                group["error_entropy"].median()
            ),

            "control_entropy_median": float(
                group["control_entropy"].median()
            ),

            "error_prediction_accuracy": float(
                group["error_correct"].mean()
            ),

            "control_prediction_accuracy": float(
                group["control_correct"].mean()
            ),

            "error_prediction_agreement": float(
                group["error_saved_prediction_agreement"].mean()
            ),

            "control_prediction_agreement": float(
                group["control_saved_prediction_agreement"].mean()
            ),
        })

    del error_prob, control_prob


# ============================================================
# 4. CONFIDENCE FIGURE
# ============================================================

def save_confidence_figure(
    patient,
    class_name,
    group_results,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    plotted = 0

    for group_name, result in group_results.items():

        if result is None:
            continue

        values = result["target_probability"]

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=np.linspace(0, 1, 31),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=group_name,
        )

        plotted += 1

    ax.set_title(
        f"{patient} — {class_name} Probability Distribution",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel(
        f"Model probability of target class: {class_name}"
    )

    ax.set_ylabel("Probability density")

    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)

    if plotted:
        ax.legend(fontsize=8)

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / f"{patient}_{class_name}_confidence.png"
    )

    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Confidence figure:", path)


# ============================================================
# 5. PROCESS ONE VALIDATION PATIENT
# ============================================================

def process_case(
    case,
    record,
    model,
    pair_table,
):

    patient = case["patient"]
    uid = case["uid"]

    class_id = case["class_id"]
    other_id = 3 - class_id

    class_name = case["class_name"]
    component_id = case["component_id"]

    rng = np.random.default_rng(
        RANDOM_SEED + component_id
    )

    print("\n" + "=" * 70)
    print("PATIENT:", patient)
    print("TARGET CLASS:", class_name)
    print("COMPONENT ID:", component_id)
    print("=" * 70)

    # --------------------------------------------------------
    # Load original image, GT and saved predictions.
    # --------------------------------------------------------

    image_path = Path(
        record["image"]
    )

    label_path = (
        LABEL_DIR / f"{uid}_multiclass.nii.gz"
    )

    prediction_path = (
        PRED_DIR / f"{uid}_pred_multiclass.nii.gz"
    )

    for path in [
        image_path,
        label_path,
        prediction_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading CT...")
    ct_img = sitk.ReadImage(
        str(image_path)
    )

    print("Loading ground truth...")
    gt_img = sitk.ReadImage(
        str(label_path)
    )

    print("Loading saved prediction...")
    pred_img = sitk.ReadImage(
        str(prediction_path)
    )

    validate_geometry(
        ct_img,
        gt_img,
        f"{patient}: CT versus GT",
    )

    validate_geometry(
        ct_img,
        pred_img,
        f"{patient}: CT versus prediction",
    )

    image = sitk.GetArrayFromImage(
        ct_img
    ).astype(
        np.float32,
        copy=False,
    )

    gt_dhw = sitk.GetArrayFromImage(
        gt_img
    ).astype(
        np.uint8,
        copy=False,
    )

    pred_dhw = sitk.GetArrayFromImage(
        pred_img
    ).astype(
        np.uint8,
        copy=False,
    )

    if not (
        image.shape == gt_dhw.shape == pred_dhw.shape
    ):
        raise ValueError(
            f"Shape mismatch for {patient}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"CT contains non-finite values: {patient}"
        )

    if not np.isin(gt_dhw, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected GT classes: {patient}"
        )

    if not np.isin(pred_dhw, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected prediction classes: {patient}"
        )

    print("Volume shape D,H,W:", image.shape)

    # --------------------------------------------------------
    # Preprocessing exactly as V6/V6.1 training.
    # --------------------------------------------------------

    print("Applying training intensity preprocessing...")

    image_normalized = np.clip(
        image,
        -1000.0,
        1000.0,
    )

    image_normalized = (
        (image_normalized + 1000.0)
        / 2000.0
    ).astype(
        np.float32,
        copy=False,
    )

    inputs = torch.from_numpy(
        np.ascontiguousarray(
            image_normalized[None, None]
        )
    )

    # Input stays on CPU; MONAI transfers patches to GPU.
    inputs = inputs.to(
        CPU_OUTPUT
    )

    # --------------------------------------------------------
    # Sliding-window inference.
    # --------------------------------------------------------

    print("\nRunning V6.1 probability inference...")
    print("ROI:", ROI_SIZE)
    print("Overlap:", OVERLAP)
    print("Blend mode: Gaussian")
    print("Sliding-window batch size:", SW_BATCH_SIZE)
    print("Output accumulation device: CPU")

    torch.cuda.empty_cache()

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=AMP and DEVICE.type == "cuda",
        ):

            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=OVERLAP,
                mode="gaussian",
                sw_device=DEVICE,
                device=CPU_OUTPUT,
            )

    if logits.shape != (
        1,
        NUM_CLASSES,
        *image.shape,
    ):

        raise ValueError(
            f"Unexpected logit output shape: {logits.shape}"
        )

    print(
        "Logits shape:",
        tuple(logits.shape),
    )

    print(
        "Logits accumulation device:",
        logits.device,
    )

    # Input arrays no longer needed for inference.
    del inputs
    del image_normalized
    del image
    del ct_img

    gc.collect()

    # --------------------------------------------------------
    # Reconstruct the exact 3D component.
    #
    # Training/inference array order is D,H,W.
    # Previous component IDs were generated in I,J,K.
    # Convert predictions/labels to I,J,K before labeling.
    # --------------------------------------------------------

    print("\nReconstructing selected error component...")

    pred_ijk = np.transpose(
        pred_dhw,
        (2, 1, 0),
    )

    gt_ijk = np.transpose(
        gt_dhw,
        (2, 1, 0),
    )

    if case["error_type"] == "FP":

        source_mask = (
            pred_ijk == class_id
        )

    else:

        source_mask = (
            gt_ijk == class_id
        )

    component_map, n_components = ndimage.label(
        source_mask,
        structure=STRUCTURE,
    )

    if component_id < 1 or component_id > n_components:

        raise ValueError(
            f"Component {component_id} not found "
            f"for {patient}; total={n_components}"
        )

    component = (
        component_map == component_id
    )

    if case["error_type"] == "FP":

        error_background = (
            component & (gt_ijk == 0)
        )

        error_other = (
            component & (gt_ijk == other_id)
        )

    else:

        error_background = (
            component & (pred_ijk == 0)
        )

        error_other = (
            component & (pred_ijk == other_id)
        )

    print(
        "Background error voxels:",
        f"{np.count_nonzero(error_background):,}",
    )

    print(
        "Other-vessel error voxels:",
        f"{np.count_nonzero(error_other):,}",
    )

    print("Component reconstruction: PASSED")

    # --------------------------------------------------------
    # Local ROI.
    # --------------------------------------------------------

    slices = roi_slices(
        component,
        ROI_MARGIN,
    )

    roi_origin_ijk = np.array(
        [s.start for s in slices],
        dtype=np.int64,
    )

    gt_roi = gt_ijk[slices]
    pred_roi = pred_ijk[slices]

    error_bg_roi = error_background[slices]
    error_other_roi = error_other[slices]

    # --------------------------------------------------------
    # Build local control groups.
    # --------------------------------------------------------

    neighborhood = ndimage.binary_dilation(
        error_bg_roi,
        structure=STRUCTURE,
        iterations=NEIGHBORHOOD_ITERATIONS,
    )

    neighborhood &= ~error_bg_roi

    local_true_negative = (
        neighborhood
        & (gt_roi == 0)
        & (pred_roi == 0)
    )

    local_correct_target = (
        neighborhood
        & (gt_roi == class_id)
        & (pred_roi == class_id)
    )

    local_correct_other = (
        neighborhood
        & (gt_roi == other_id)
        & (pred_roi == other_id)
    )

    masks_ijk = {
        "error_from_background": error_bg_roi,
        "error_from_other_vessel": error_other_roi,
        "local_true_negative_background": local_true_negative,
        "local_correct_target": local_correct_target,
        "local_correct_other_vessel": local_correct_other,
    }

    # --------------------------------------------------------
    # Confidence summary for each group.
    # --------------------------------------------------------

    print("\nCONFIDENCE GROUP ANALYSIS:")

    group_results = {}

    for group_name, mask in masks_ijk.items():

        group_results[group_name] = analyze_group(
            patient=patient,
            class_name=class_name,
            class_id=class_id,
            group_name=group_name,
            mask_ijk=mask,
            roi_origin_ijk=roi_origin_ijk,
            logits=logits,
            pred_dhw=pred_dhw,
            gt_dhw=gt_dhw,
            rng=rng,
        )

    # --------------------------------------------------------
    # Matched-pair probability analysis.
    # --------------------------------------------------------

    analyze_matched_pairs(
        patient=patient,
        case=case,
        pair_table=pair_table,
        masks_ijk=masks_ijk,
        roi_origin_ijk=roi_origin_ijk,
        logits=logits,
        pred_dhw=pred_dhw,
        gt_dhw=gt_dhw,
    )

    # --------------------------------------------------------
    # Inference agreement check.
    # --------------------------------------------------------

    agreements = []

    for result in group_results.values():

        if result is not None:
            agreements.extend(
                result["probabilities"].argmax(axis=1)
                == result["saved_prediction"]
            )

    patient_agreement = (
        float(np.mean(agreements))
        if agreements else np.nan
    )

    print(
        "\nArgmax agreement with saved V6.1 prediction:",
        f"{patient_agreement:.6%}",
    )

    INFERENCE_ROWS.append({
        "patient_id": patient,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "roi_size": str(ROI_SIZE),
        "overlap": OVERLAP,
        "blend_mode": "gaussian",
        "argmax_agreement": patient_agreement,
        "status": (
            "PASS"
            if patient_agreement >= MIN_ARGMAX_AGREEMENT
            else "CHECK_INFERENCE_SETTINGS"
        ),
    })

    # --------------------------------------------------------
    # Save figure.
    # --------------------------------------------------------

    save_confidence_figure(
        patient,
        class_name,
        group_results,
    )

    # --------------------------------------------------------
    # Cleanup.
    # --------------------------------------------------------

    del logits
    del pred_dhw
    del gt_dhw

    del pred_ijk
    del gt_ijk

    del component_map
    del component
    del source_mask

    del error_background
    del error_other

    del gt_roi
    del pred_roi

    del error_bg_roi
    del error_other_roi

    del neighborhood

    del local_true_negative
    del local_correct_target
    del local_correct_other

    del masks_ijk
    del group_results

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 6. LOAD MODEL AND VALIDATION SPLITS
# ============================================================

def main():

    global CHECKPOINT_EPOCH

    print("=" * 70)

    print(
        "NEUROVESSEL V6.1 — "
        "PREDICTION CONFIDENCE ANALYSIS"
    )

    print("=" * 70)

    if DEVICE.type != "cuda":

        raise RuntimeError(
            "CUDA is unavailable. "
            "Activate the neurovessel environment and verify GPU."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not PAIR_INPUT.exists():
        raise FileNotFoundError(
            "Previous spatial matched-pair CSV missing:\n"
            f"{PAIR_INPUT}"
        )

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading patient splits...")

    train_records, validation_records, test_count = (
        load_v6_splits()
    )

    print("Training patients:", len(train_records))
    print("Validation patients:", len(validation_records))
    print("Locked test patients:", test_count)

    if (
        len(train_records) != 202
        or len(validation_records) != 26
        or test_count != 26
    ):

        raise RuntimeError(
            "Unexpected dataset partition counts."
        )

    # Only validation records are used.
    validation_by_uid = {
        record["uid"]: record
        for record in validation_records
    }

    for case in CASES:

        if case["uid"] not in validation_by_uid:

            raise RuntimeError(
                f"{case['patient']} is not in validation split."
            )

    print(
        "Selected diagnostic patients:",
        len(CASES),
    )

    print("Locked test images and labels will NOT be loaded.")

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    print("\nLoading checkpoint:")

    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint has no model_state_dict."
        )

    CHECKPOINT_EPOCH = int(
        checkpoint.get("epoch", -1)
    )

    print("Checkpoint epoch:", CHECKPOINT_EPOCH)

    print(
        "Best patch validation Dice:",
        checkpoint.get(
            "best_val_mean_foreground_patch_dice",
            "Not available",
        ),
    )

    print("Building exact NeuroVesselV6 architecture...")

    model = NeuroVesselV6(
        in_channels=1,
        out_channels=3,
        base=8,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{parameter_count:,}",
    )

    print("Checkpoint loaded successfully.")

    del checkpoint

    # --------------------------------------------------------
    # Load previously saved spatial matches.
    # --------------------------------------------------------

    print("\nLoading spatial matched pairs...")

    pair_table = pd.read_csv(
        PAIR_INPUT
    )

    if "patient_id" not in pair_table.columns:
        raise ValueError(
            "Matched-pair CSV does not contain patient_id."
        )

    print(
        "Spatial matched-pair records:",
        f"{len(pair_table):,}",
    )

    # --------------------------------------------------------
    # Process the three validation cases.
    # --------------------------------------------------------

    for case in CASES:

        record = validation_by_uid[
            case["uid"]
        ]

        process_case(
            case=case,
            record=record,
            model=model,
            pair_table=pair_table,
        )

    # --------------------------------------------------------
    # Export CSV outputs.
    # --------------------------------------------------------

    group_path = (
        OUTPUT_DIR / "confidence_group_summary.csv"
    )

    bins_path = (
        OUTPUT_DIR / "confidence_calibration_bins.csv"
    )

    pairs_path = (
        OUTPUT_DIR / "confidence_matched_pairs.csv"
    )

    pair_summary_path = (
        OUTPUT_DIR / "confidence_matched_summary.csv"
    )

    check_path = (
        OUTPUT_DIR / "confidence_inference_checks.csv"
    )

    group_df = pd.DataFrame(
        GROUP_ROWS
    )

    bins_df = pd.DataFrame(
        BIN_ROWS
    )

    pairs_df = pd.DataFrame(
        PAIR_ROWS
    )

    pair_summary_df = pd.DataFrame(
        PAIR_SUMMARY_ROWS
    )

    checks_df = pd.DataFrame(
        INFERENCE_ROWS
    )

    group_df.to_csv(
        group_path,
        index=False,
    )

    bins_df.to_csv(
        bins_path,
        index=False,
    )

    pairs_df.to_csv(
        pairs_path,
        index=False,
    )

    pair_summary_df.to_csv(
        pair_summary_path,
        index=False,
    )

    checks_df.to_csv(
        check_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "V6.1 PREDICTION CONFIDENCE ANALYSIS COMPLETED"
    )

    print("=" * 70)

    print("\nCONFIDENCE GROUP SUMMARY:")

    if not group_df.empty:

        columns = [
            "patient_id",
            "group",
            "total_voxels",
            "sampled_voxels",
            "mean_predicted_confidence",
            "mean_target_class_probability",
            "accuracy_vs_ground_truth",
            "mean_entropy_normalized",
            "expected_calibration_error",
        ]

        print(
            group_df[columns].to_string(
                index=False
            )
        )

    print("\nINFERENCE CHECKS:")

    print(
        checks_df.to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES:")

    print(group_path)
    print(bins_path)
    print(pairs_path)
    print(pair_summary_path)
    print(check_path)

    print("\nFIGURE DIRECTORY:")
    print(FIGURE_DIR)

    if (
        not checks_df.empty
        and (
            checks_df["argmax_agreement"]
            < MIN_ARGMAX_AGREEMENT
        ).any()
    ):

        print(
            "\nWARNING: At least one inference agreement "
            "is below the required threshold."
        )

        print(
            "Check that sliding-window preprocessing and "
            "inference settings match the saved evaluator."
        )

    else:

        print(
            "\nAll inference agreement checks passed."
        )

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":
    main()
