
from pathlib import Path
import gc

import nibabel as nib
import numpy as np
import pandas as pd

from scipy import ndimage
from scipy.spatial import cKDTree


# ============================================================
# NEUROVESSEL V6.1
# SPATIALLY MATCHED HARD-NEGATIVE ANALYSIS
#
# Three validation patients:
#   0529: PV false positives
#   0763: PV false positives
#   0848: PA false negatives
#
# Read-only diagnostic.
# No training, inference, or prediction modification.
# Locked test partition is never loaded.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

CT_DIR = ROOT / "data" / "processed_airrc_v5" / "images"

PRED_DIR = (
    ROOT / "outputs" / "predictions" / "airrc_v6_1_fullvol"
)

LABEL_DIR = (
    ROOT / "data" / "processed_airrc_v6" / "labels_multiclass"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics"
    / "airrc_v6_1_spatial_hard_negative"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PATIENT CONFIGURATION
# ============================================================

CASES = [
    {
        "patient": "LIDC-IDRI-0529",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",
        "class_name": "PV",
        "class_id": 2,
        "component_id": 763,
        "error_type": "FP",
        "expected_background_errors": 123719,
        "expected_other_errors": 0,
    },
    {
        "patient": "LIDC-IDRI-0763",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",
        "class_name": "PV",
        "class_id": 2,
        "component_id": 16,
        "error_type": "FP",
        "expected_background_errors": 79642,
        "expected_other_errors": 3547,
    },
    {
        "patient": "LIDC-IDRI-0848",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
        "class_name": "PA",
        "class_id": 1,
        "component_id": 1,
        "error_type": "FN",
        "expected_background_errors": 49581,
        "expected_other_errors": 10989,
    },
]


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

STRUCTURE = ndimage.generate_binary_structure(3, 3)

ROI_MARGIN = 24

NEIGHBORHOOD_ITERATIONS = 20

MAX_MATCH_DISTANCE = 20.0

MAX_MATCH_ERRORS = 5000

MAX_GROUP_SAMPLES = 20000

NEAREST_CANDIDATES = 32

RANDOM_SEED = 2026


SUMMARY_ROWS = []
PAIR_SUMMARY_ROWS = []
MATCH_ROWS = []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sample_coordinates(mask, maximum, rng):
    """
    Sample three-dimensional coordinates from a 3D mask.

    Returns an Nx3 array.
    """

    if mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, got {mask.shape}"
        )

    flat = np.flatnonzero(mask)

    if flat.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    if flat.size > maximum:
        selected = rng.choice(
            flat,
            size=maximum,
            replace=False,
        )
    else:
        selected = flat

    coordinates = np.column_stack(
        np.unravel_index(selected, mask.shape)
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"Invalid coordinate shape: {coordinates.shape}"
        )

    return coordinates.astype(np.int32, copy=False)


def get_roi_slices(mask, margin):
    """Create a 3D crop around a selected component."""

    coords = np.where(mask)

    if coords[0].size == 0:
        raise ValueError("Cannot construct ROI from empty mask.")

    slices = []

    for axis in range(3):

        start = max(
            0,
            int(coords[axis].min()) - margin,
        )

        stop = min(
            mask.shape[axis],
            int(coords[axis].max()) + margin + 1,
        )

        slices.append(slice(start, stop))

    return tuple(slices)


def calculate_local_statistics(volume, window):
    """Local mean and standard deviation."""

    mean = ndimage.uniform_filter(
        volume,
        size=window,
        mode="nearest",
    )

    mean_square = ndimage.uniform_filter(
        np.square(volume),
        size=window,
        mode="nearest",
    )

    variance = np.maximum(
        mean_square - np.square(mean),
        0.0,
    )

    std = np.sqrt(variance).astype(np.float32)

    return mean, std


def build_features(ct):
    """
    Construct CT and local spatial image features.

    Training-style clipping:
    [-1000, 1000] -> [0, 1]
    """

    raw = ct.astype(np.float32, copy=False)

    clipped = np.clip(
        raw,
        -1000.0,
        1000.0,
    ).astype(np.float32)

    normalized = (
        clipped + 1000.0
    ) / 2000.0

    print("  Calculating local 3x3x3 features...")

    mean3, std3 = calculate_local_statistics(
        clipped,
        3,
    )

    print("  Calculating local 7x7x7 features...")

    mean7, std7 = calculate_local_statistics(
        clipped,
        7,
    )

    print("  Calculating gradient magnitude...")

    gi, gj, gk = np.gradient(clipped)

    gradient = np.sqrt(
        gi ** 2 + gj ** 2 + gk ** 2
    ).astype(np.float32)

    del gi, gj, gk

    return {
        "stored_intensity": raw,
        "model_clipped_intensity": clipped,
        "model_normalized_intensity": normalized,
        "local_mean_3": mean3,
        "local_std_3": std3,
        "local_mean_7": mean7,
        "local_std_7": std7,
        "gradient_magnitude": gradient,
    }


def extract_features(feature_arrays, coordinates):
    """Extract feature values at Nx3 coordinates."""

    if len(coordinates) == 0:
        return {}

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 coordinates, got {coordinates.shape}"
        )

    index = tuple(coordinates.T)

    return {
        name: array[index]
        for name, array in feature_arrays.items()
    }


def describe_values(values):
    """Calculate distribution statistics."""

    if len(values) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


# ============================================================
# SPATIAL MATCHING
# ============================================================

def match_nearest_unique(
    error_coords,
    control_mask,
    roi_origin,
    affine,
    distance_units,
):
    """
    Match error voxels with unique nearby control voxels.

    Input error_coords must already be an Nx3 coordinate array.

    Each control can be used once per comparison.
    Matching uses physical coordinates through the NIfTI affine.
    """

    if error_coords.ndim != 2 or error_coords.shape[1] != 3:
        raise ValueError(
            f"ERROR: Expected Nx3 coordinates, received "
            f"{error_coords.shape}"
        )

    control_coords = np.argwhere(control_mask)

    if len(error_coords) == 0 or len(control_coords) == 0:

        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.array([], dtype=np.float64),
            len(control_coords),
        )

    # Convert ROI coordinates to global voxel coordinates.
    error_global = error_coords + roi_origin

    control_global = control_coords + roi_origin

    # Convert voxel coordinates to world coordinates.
    error_world = nib.affines.apply_affine(
        affine,
        error_global,
    )

    control_world = nib.affines.apply_affine(
        affine,
        control_global,
    )

    tree = cKDTree(control_world)

    k = min(
        NEAREST_CANDIDATES,
        len(control_world),
    )

    distances, indices = tree.query(
        error_world,
        k=k,
        distance_upper_bound=MAX_MATCH_DISTANCE,
    )

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Greedy one-to-one assignment.
    # Errors with the closest first candidate are processed first.
    order = np.argsort(distances[:, 0])

    used_controls = set()

    matched_errors = []
    matched_controls = []
    matched_distances = []

    for error_index in order:

        for rank in range(k):

            distance = float(
                distances[error_index, rank]
            )

            control_index = int(
                indices[error_index, rank]
            )

            if not np.isfinite(distance):
                continue

            if control_index >= len(control_world):
                continue

            if distance > MAX_MATCH_DISTANCE:
                continue

            if control_index in used_controls:
                continue

            used_controls.add(control_index)

            matched_errors.append(
                error_coords[error_index]
            )

            matched_controls.append(
                control_coords[control_index]
            )

            matched_distances.append(distance)

            break

    if not matched_errors:

        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.array([], dtype=np.float64),
            len(control_coords),
        )

    return (
        np.asarray(matched_errors, dtype=np.int32),
        np.asarray(matched_controls, dtype=np.int32),
        np.asarray(matched_distances, dtype=np.float64),
        len(control_coords),
    )


def record_pairs(
    patient,
    class_name,
    error_group,
    control_group,
    error_mask,
    control_mask,
    roi_origin,
    affine,
    distance_units,
    feature_arrays,
    rng,
):
    """
    Sample error coordinates once and match them to controls.

    IMPORTANT:
    No second sampling is performed after coordinate extraction.
    """

    # Extract the error coordinates exactly once.
    sampled_errors = sample_coordinates(
        error_mask,
        MAX_MATCH_ERRORS,
        rng,
    )

    if sampled_errors.ndim != 2 or sampled_errors.shape[1] != 3:
        raise ValueError(
            f"Coordinate shape error: {sampled_errors.shape}"
        )

    (
        matched_errors,
        matched_controls,
        distances,
        available_controls,
    ) = match_nearest_unique(
        sampled_errors,
        control_mask,
        roi_origin,
        affine,
        distance_units,
    )

    matched_count = len(distances)

    match_rate = (
        matched_count / len(sampled_errors)
        if len(sampled_errors)
        else 0.0
    )

    result = {
        "patient_id": patient,
        "class": class_name,
        "error_group": error_group,
        "control_group": control_group,
        "error_samples": len(sampled_errors),
        "available_controls": available_controls,
        "matched_pairs": matched_count,
        "match_rate": float(match_rate),
        "median_match_distance": (
            float(np.median(distances))
            if matched_count else np.nan
        ),
        "p95_match_distance": (
            float(np.percentile(distances, 95))
            if matched_count else np.nan
        ),
        "distance_units": distance_units,
    }

    if matched_count == 0:

        PAIR_SUMMARY_ROWS.append(result)

        print(
            f"    {control_group}: "
            "no unique matches found."
        )

        return

    error_features = extract_features(
        feature_arrays,
        matched_errors,
    )

    control_features = extract_features(
        feature_arrays,
        matched_controls,
    )

    # Paired feature differences.
    for name in feature_arrays:

        delta = (
            error_features[name]
            - control_features[name]
        )

        result[f"{name}_delta_mean"] = float(
            np.mean(delta)
        )

        result[f"{name}_delta_median"] = float(
            np.median(delta)
        )

    PAIR_SUMMARY_ROWS.append(result)

    # Convert ROI coordinates to global coordinates.
    error_global = matched_errors + roi_origin

    control_global = matched_controls + roi_origin

    error_world = nib.affines.apply_affine(
        affine,
        error_global,
    )

    control_world = nib.affines.apply_affine(
        affine,
        control_global,
    )

    # Individual pair records.
    for n in range(matched_count):

        row = {
            "patient_id": patient,
            "class": class_name,
            "error_group": error_group,
            "control_group": control_group,
            "match_distance": float(distances[n]),
            "distance_units": distance_units,

            "error_i": int(error_global[n, 0]),
            "error_j": int(error_global[n, 1]),
            "error_k": int(error_global[n, 2]),

            "control_i": int(control_global[n, 0]),
            "control_j": int(control_global[n, 1]),
            "control_k": int(control_global[n, 2]),

            "error_world_x": float(error_world[n, 0]),
            "error_world_y": float(error_world[n, 1]),
            "error_world_z": float(error_world[n, 2]),

            "control_world_x": float(control_world[n, 0]),
            "control_world_y": float(control_world[n, 1]),
            "control_world_z": float(control_world[n, 2]),
        }

        for name in feature_arrays:

            error_value = float(
                error_features[name][n]
            )

            control_value = float(
                control_features[name][n]
            )

            row[f"error_{name}"] = error_value
            row[f"control_{name}"] = control_value

            row[f"delta_{name}"] = (
                error_value - control_value
            )

        MATCH_ROWS.append(row)

    print(
        f"    {control_group}: "
        f"{matched_count}/{len(sampled_errors)} matched, "
        f"median distance={np.median(distances):.2f} "
        f"{distance_units}"
    )


# ============================================================
# GROUP SUMMARIES
# ============================================================

def summarize_group(
    patient,
    class_name,
    group_name,
    mask,
    features,
    target_tree,
    affine,
    distance_units,
    roi_origin,
    rng,
):
    """Summarize feature distributions for one group."""

    coords = sample_coordinates(
        mask,
        MAX_GROUP_SAMPLES,
        rng,
    )

    total_voxels = int(
        np.count_nonzero(mask)
    )

    if len(coords) == 0:
        print(f"  {group_name}: empty")
        return

    values = extract_features(
        features,
        coords,
    )

    global_coords = coords + roi_origin

    world_coords = nib.affines.apply_affine(
        affine,
        global_coords,
    )

    # Distance from each sample to its nearest annotated target voxel.
    distances, _ = target_tree.query(
        world_coords,
        k=1,
    )

    row = {
        "patient_id": patient,
        "class": class_name,
        "group": group_name,
        "total_voxels": total_voxels,
        "sampled_voxels": len(coords),
        "distance_units": distance_units,
        "distance_to_gt_target_median": float(
            np.median(distances)
        ),
        "distance_to_gt_target_p95": float(
            np.percentile(distances, 95)
        ),
    }

    for feature_name, feature_values in values.items():

        stats = describe_values(feature_values)

        for stat_name, stat_value in stats.items():

            row[f"{feature_name}_{stat_name}"] = stat_value

    row["fraction_clipped_low"] = float(
        np.mean(values["stored_intensity"] < -1000)
    )

    row["fraction_clipped_high"] = float(
        np.mean(values["stored_intensity"] > 1000)
    )

    SUMMARY_ROWS.append(row)

    print(
        f"  {group_name}: "
        f"voxels={total_voxels:,}, "
        f"median intensity="
        f"{np.median(values['stored_intensity']):.2f}, "
        f"median distance="
        f"{np.median(distances):.2f} {distance_units}"
    )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_case(case):

    patient = case["patient"]
    uid = case["uid"]

    class_name = case["class_name"]
    class_id = case["class_id"]
    other_id = 3 - class_id

    component_id = case["component_id"]

    rng = np.random.default_rng(
        RANDOM_SEED + component_id
    )

    print("\n" + "=" * 70)
    print("PATIENT:", patient)
    print("CLASS:", class_name)
    print("ERROR TYPE:", case["error_type"])
    print("=" * 70)

    # --------------------------------------------------------
    # Load source files.
    # --------------------------------------------------------

    ct_path = CT_DIR / f"{uid}_ct.nii.gz"

    pred_path = (
        PRED_DIR / f"{uid}_pred_multiclass.nii.gz"
    )

    gt_path = (
        LABEL_DIR / f"{uid}_multiclass.nii.gz"
    )

    for path in [ct_path, pred_path, gt_path]:

        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading CT...")
    ct_img = nib.load(str(ct_path))

    print("Loading prediction...")
    pred_img = nib.load(str(pred_path))

    print("Loading ground truth...")
    gt_img = nib.load(str(gt_path))

    # --------------------------------------------------------
    # Geometry validation.
    # --------------------------------------------------------

    if not (
        ct_img.shape == pred_img.shape == gt_img.shape
    ):
        raise ValueError(
            f"Shape mismatch for {patient}"
        )

    if not (
        np.allclose(ct_img.affine, gt_img.affine, atol=1e-3)
        and np.allclose(pred_img.affine, gt_img.affine, atol=1e-3)
    ):
        raise ValueError(
            f"Affine mismatch for {patient}"
        )

    units = gt_img.header.get_xyzt_units()[0]

    if units == "mm":
        distance_units = "mm"
    else:
        distance_units = units or "affine_units"

        print(
            "WARNING: NIfTI spatial units are not mm:",
            repr(units)
        )

    affine = gt_img.affine

    spacing = gt_img.header.get_zooms()[:3]

    print("Volume shape:", ct_img.shape)
    print("Voxel spacing:", spacing)
    print("Spatial units:", distance_units)

    # --------------------------------------------------------
    # Load arrays.
    # --------------------------------------------------------

    ct = np.asarray(
        ct_img.dataobj,
        dtype=np.float32,
    )

    pred = np.asarray(
        pred_img.dataobj,
        dtype=np.uint8,
    )

    gt = np.asarray(
        gt_img.dataobj,
        dtype=np.uint8,
    )

    if not np.isfinite(ct).all():
        raise ValueError(
            f"CT contains non-finite values: {patient}"
        )

    if not np.isin(pred, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected prediction classes: {patient}"
        )

    if not np.isin(gt, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected GT classes: {patient}"
        )

    # --------------------------------------------------------
    # Reconstruct exact connected component.
    # --------------------------------------------------------

    if case["error_type"] == "FP":

        component_source = pred == class_id

    else:

        component_source = gt == class_id

    component_map, n_components = ndimage.label(
        component_source,
        structure=STRUCTURE,
    )

    if component_id < 1 or component_id > n_components:
        raise ValueError(
            f"Selected component not found: {patient}"
        )

    component = component_map == component_id

    if case["error_type"] == "FP":

        error_background = (
            component & (gt == 0)
        )

        error_other = (
            component & (gt == other_id)
        )

    else:

        error_background = (
            component & (pred == 0)
        )

        error_other = (
            component & (pred == other_id)
        )

    # --------------------------------------------------------
    # Validate exact error counts.
    # --------------------------------------------------------

    bg_count = int(
        np.count_nonzero(error_background)
    )

    other_count = int(
        np.count_nonzero(error_other)
    )

    if bg_count != case["expected_background_errors"]:
        raise ValueError(
            f"Background error mismatch for {patient}: "
            f"{bg_count} vs "
            f"{case['expected_background_errors']}"
        )

    if other_count != case["expected_other_errors"]:
        raise ValueError(
            f"Other-vessel error mismatch for {patient}: "
            f"{other_count} vs "
            f"{case['expected_other_errors']}"
        )

    print("Background errors:", f"{bg_count:,}")
    print("Other-vessel errors:", f"{other_count:,}")
    print("Component validation: PASSED")

    # --------------------------------------------------------
    # Extract a local 3D ROI.
    # --------------------------------------------------------

    slices = get_roi_slices(
        component,
        ROI_MARGIN,
    )

    roi_origin = np.array(
        [s.start for s in slices],
        dtype=np.int32,
    )

    ct_roi = ct[slices]

    pred_roi = pred[slices]

    gt_roi = gt[slices]

    component_roi = component[slices]

    error_bg_roi = error_background[slices]

    error_other_roi = error_other[slices]

    print("Local ROI shape:", ct_roi.shape)

    # --------------------------------------------------------
    # Build local neighborhood around background errors.
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

    # --------------------------------------------------------
    # Generate features.
    # --------------------------------------------------------

    print("\nBuilding local features...")

    features = build_features(ct_roi)

    # --------------------------------------------------------
    # Ground-truth target distance index.
    # --------------------------------------------------------

    print("Building target-vessel spatial index...")

    target_coords = np.argwhere(gt == class_id)

    if len(target_coords) == 0:
        raise ValueError(
            f"No GT target voxels found: {patient}"
        )

    target_world = nib.affines.apply_affine(
        affine,
        target_coords,
    )

    target_tree = cKDTree(target_world)

    # --------------------------------------------------------
    # Feature group summaries.
    # --------------------------------------------------------

    print("\nFEATURE GROUP SUMMARIES:")

    groups = [
        (
            "error_from_background",
            error_bg_roi,
        ),
        (
            "error_from_other_vessel",
            error_other_roi,
        ),
        (
            "local_true_negative_background",
            local_true_negative,
        ),
        (
            "local_correct_target",
            local_correct_target,
        ),
        (
            "local_correct_other_vessel",
            local_correct_other,
        ),
    ]

    for group_name, mask in groups:

        summarize_group(
            patient=patient,
            class_name=class_name,
            group_name=group_name,
            mask=mask,
            features=features,
            target_tree=target_tree,
            affine=affine,
            distance_units=distance_units,
            roi_origin=roi_origin,
            rng=rng,
        )

    # --------------------------------------------------------
    # Spatial matching.
    # --------------------------------------------------------

    print("\nSPATIALLY MATCHED COMPARISONS:")

    error_groups = [
        (
            "error_from_background",
            error_bg_roi,
        ),
        (
            "error_from_other_vessel",
            error_other_roi,
        ),
    ]

    control_groups = [
        (
            "local_true_negative_background",
            local_true_negative,
        ),
        (
            "local_correct_target",
            local_correct_target,
        ),
        (
            "local_correct_other_vessel",
            local_correct_other,
        ),
    ]

    for error_name, error_mask in error_groups:

        for control_name, control_mask in control_groups:

            record_pairs(
                patient=patient,
                class_name=class_name,
                error_group=error_name,
                control_group=control_name,
                error_mask=error_mask,
                control_mask=control_mask,
                roi_origin=roi_origin,
                affine=affine,
                distance_units=distance_units,
                feature_arrays=features,
                rng=rng,
            )

    # --------------------------------------------------------
    # Release memory.
    # --------------------------------------------------------

    del ct, pred, gt

    del component_source
    del component_map
    del component

    del error_background
    del error_other

    del ct_roi
    del pred_roi
    del gt_roi

    del component_roi

    del error_bg_roi
    del error_other_roi

    del neighborhood

    del local_true_negative
    del local_correct_target
    del local_correct_other

    del features

    del target_coords
    del target_world
    del target_tree

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "NEUROVESSEL V6.1 — "
        "SPATIALLY MATCHED HARD-NEGATIVE ANALYSIS"
    )

    print("=" * 70)

    print("Patients:", len(CASES))
    print("Neighborhood:", NEIGHBORHOOD_ITERATIONS, "voxels")
    print("Maximum match distance:", MAX_MATCH_DISTANCE)
    print("Maximum error samples:", MAX_MATCH_ERRORS)

    print("Features:")
    print("  - Stored CT intensity")
    print("  - Model-clipped intensity")
    print("  - Normalized intensity")
    print("  - Local mean and standard deviation")
    print("  - 3D gradient magnitude")

    print("\nLocked test partition will NOT be loaded.")
    print("No training or model inference.")

    for case in CASES:
        process_case(case)

    # --------------------------------------------------------
    # Export results.
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR / "spatial_feature_groups.csv"
    )

    pairs_path = (
        OUTPUT_DIR / "spatial_matched_pairs.csv"
    )

    matching_path = (
        OUTPUT_DIR / "spatial_matching_summary.csv"
    )

    summary_df = pd.DataFrame(SUMMARY_ROWS)

    pairs_df = pd.DataFrame(MATCH_ROWS)

    matching_df = pd.DataFrame(PAIR_SUMMARY_ROWS)

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    pairs_df.to_csv(
        pairs_path,
        index=False,
    )

    matching_df.to_csv(
        matching_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("SPATIAL HARD-NEGATIVE ANALYSIS COMPLETED")

    print("=" * 70)

    print("\nFEATURE GROUP RESULTS:")

    if not summary_df.empty:

        cols = [
            "patient_id",
            "group",
            "total_voxels",
            "sampled_voxels",
            "stored_intensity_median",
            "local_std_3_median",
            "local_std_7_median",
            "gradient_magnitude_median",
            "distance_to_gt_target_median",
        ]

        print(
            summary_df[cols].to_string(index=False)
        )

    print("\nMATCHING RESULTS:")

    if not matching_df.empty:

        cols = [
            "patient_id",
            "error_group",
            "control_group",
            "error_samples",
            "available_controls",
            "matched_pairs",
            "match_rate",
            "median_match_distance",
        ]

        print(
            matching_df[cols].to_string(index=False)
        )

    print("\nOUTPUT FILES:")

    print(summary_path)
    print(pairs_path)
    print(matching_path)

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":
    main()
