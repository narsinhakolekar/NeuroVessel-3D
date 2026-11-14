
from pathlib import Path
import gc

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage


# ============================================================
# NEUROVESSEL V6.1 — CT INTENSITY DISTRIBUTION ANALYSIS
#
# Patients:
#   LIDC-IDRI-0529: PV false positive
#   LIDC-IDRI-0763: PV false positive
#   LIDC-IDRI-0848: PA false negative
#
# Comparisons:
#   1. Incorrectly classified voxels
#   2. Correctly classified target-class voxels
#   3. Nearby ground-truth background
#   4. Error voxels confused with the other vessel class
#
# Read-only diagnostic. No training or inference.
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

METRIC_DIR = (
    ROOT / "outputs" / "metrics" / "airrc_v6_1_error_analysis"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "metrics" / "airrc_v6_1_intensity_analysis"
)

FIGURE_DIR = (
    ROOT / "outputs" / "visualizations"
    / "airrc_v6_1_intensity_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Exact validation cases and selected 3D components.
# Coordinates are (i, j, k).
# ------------------------------------------------------------

CASES = [
    {
        "patient": "LIDC-IDRI-0529",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",
        "class_name": "PV",
        "class_id": 2,
        "error_type": "FP",
        "component_id": 763,
        "expected_errors": 123719,
    },
    {
        "patient": "LIDC-IDRI-0763",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",
        "class_name": "PV",
        "class_id": 2,
        "error_type": "FP",
        "component_id": 16,
        "expected_errors": 83189,
    },
    {
        "patient": "LIDC-IDRI-0848",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
        "class_name": "PA",
        "class_id": 1,
        "error_type": "FN",
        "component_id": 1,
        "expected_errors": 60570,
    },
]


# All components use 26-connectivity.
STRUCTURE = ndimage.generate_binary_structure(3, 3)

# Nearby background shell: five voxels from error region.
# Dataset spacing in these cases is 1 mm per voxel.
SHELL_ITERATIONS = 5

# Maximum sampled voxels per intensity group.
MAX_SAMPLE = 120000

RANDOM_SEED = 42

summary_rows = []


def sample_intensities(ct, mask, rng):

    """Extract a deterministic random sample of CT values."""

    flat_indices = np.flatnonzero(mask)
    total_voxels = int(flat_indices.size)

    if total_voxels == 0:
        return np.array([], dtype=np.float32), 0

    if total_voxels > MAX_SAMPLE:

        selected = rng.choice(
            flat_indices,
            size=MAX_SAMPLE,
            replace=False,
        )

    else:
        selected = flat_indices

    values = ct.ravel()[selected]

    return values.astype(np.float32, copy=False), total_voxels


def add_group(
    patient,
    class_name,
    group_name,
    ct,
    mask,
    rng,
    samples,
):

    values, total_voxels = sample_intensities(
        ct,
        mask,
        rng,
    )

    if total_voxels == 0:
        print(f"  {group_name}: no voxels")
        return

    row = {
        "patient_id": patient,
        "class": class_name,
        "group": group_name,
        "total_voxels": total_voxels,
        "sampled_voxels": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "fraction_below_minus_500": float(
            np.mean(values < -500)
        ),
    }

    summary_rows.append(row)

    samples[group_name] = values

    print(
        f"  {group_name}: "
        f"n={total_voxels:,}, "
        f"median={row['median']:.2f}, "
        f"mean={row['mean']:.2f}, "
        f"std={row['std']:.2f}"
    )


def process_case(case):

    patient = case["patient"]
    uid = case["uid"]

    class_name = case["class_name"]
    class_id = case["class_id"]

    error_type = case["error_type"]
    component_id = case["component_id"]

    expected_errors = case["expected_errors"]

    other_id = 3 - class_id

    rng = np.random.default_rng(
        RANDOM_SEED + component_id
    )

    print("\n" + "=" * 70)
    print("PATIENT:", patient)
    print("TARGET CLASS:", class_name)
    print("ERROR TYPE:", error_type)
    print("=" * 70)

    # --------------------------------------------------------
    # Input files.
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

    # --------------------------------------------------------
    # Load volumes.
    # --------------------------------------------------------

    print("Loading CT...")
    ct_img = nib.load(str(ct_path))

    print("Loading prediction...")
    pred_img = nib.load(str(pred_path))

    print("Loading ground truth...")
    gt_img = nib.load(str(gt_path))

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

    spacing = gt_img.header.get_zooms()[:3]

    if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3):
        raise ValueError(
            f"Unexpected voxel spacing for {patient}: {spacing}"
        )

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
        raise ValueError(f"CT contains non-finite values: {patient}")

    print("Volume shape:", ct.shape)
    print("Voxel spacing:", spacing)

    print(
        "CT intensity range:",
        float(np.min(ct)),
        "to",
        float(np.max(ct)),
    )

    print(
        "CT intensity percentiles:",
        np.percentile(ct, [1, 5, 25, 50, 75, 95, 99]).round(2),
    )

    # --------------------------------------------------------
    # Identify the exact selected component.
    # --------------------------------------------------------

    if error_type == "FP":

        target_mask = pred == class_id

    elif error_type == "FN":

        target_mask = gt == class_id

    else:
        raise ValueError(error_type)

    component_map, n_components = ndimage.label(
        target_mask,
        structure=STRUCTURE,
    )

    if component_id > n_components:
        raise ValueError(
            f"Component {component_id} not found for {patient}"
        )

    selected_component = component_map == component_id

    if error_type == "FP":

        error_mask = (
            selected_component
            & (gt != class_id)
        )

    else:

        error_mask = (
            selected_component
            & (pred != class_id)
        )

    error_count = int(np.count_nonzero(error_mask))

    if error_count != expected_errors:

        raise ValueError(
            f"Error-count mismatch for {patient}: "
            f"expected={expected_errors:,}, "
            f"actual={error_count:,}"
        )

    print(
        "Selected component error count:",
        f"{error_count:,} — VALIDATED",
    )

    # --------------------------------------------------------
    # Divide errors by their confusing class.
    # --------------------------------------------------------

    if error_type == "FP":

        error_background = error_mask & (gt == 0)
        error_other_vessel = error_mask & (gt == other_id)

    else:

        error_background = error_mask & (pred == 0)
        error_other_vessel = error_mask & (pred == other_id)

    # Correctly classified target-class voxels.
    true_positive = (
        (pred == class_id)
        & (gt == class_id)
    )

    # Nearby GT background around this error region.
    # Exclude the error component itself.
    expanded = ndimage.binary_dilation(
        error_mask,
        structure=STRUCTURE,
        iterations=SHELL_ITERATIONS,
    )

    nearby_background = (
        expanded
        & (~error_mask)
        & (gt == 0)
    )

    # --------------------------------------------------------
    # Record intensity distributions.
    # --------------------------------------------------------

    samples = {}

    print("\nINTENSITY GROUPS:")

    add_group(
        patient,
        class_name,
        "error_from_background",
        ct,
        error_background,
        rng,
        samples,
    )

    add_group(
        patient,
        class_name,
        "error_from_other_vessel",
        ct,
        error_other_vessel,
        rng,
        samples,
    )

    add_group(
        patient,
        class_name,
        "correct_target_class",
        ct,
        true_positive,
        rng,
        samples,
    )

    add_group(
        patient,
        class_name,
        "nearby_gt_background",
        ct,
        nearby_background,
        rng,
        samples,
    )

    # --------------------------------------------------------
    # Generate intensity histogram.
    # --------------------------------------------------------

    nonempty = [
        values
        for values in samples.values()
        if values.size > 0
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for group_name, values in samples.items():

        if values.size == 0:
            continue

        ax.hist(
            values,
            bins=65,
            density=True,
            alpha=0.40,
            label=f"{group_name} (n={values.size:,})",
        )

    ax.set_title(
        f"{patient} — {class_name} CT Intensity Comparison",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel("Stored CT intensity (original units)")
    ax.set_ylabel("Probability density")

    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()

    figure_path = (
        FIGURE_DIR / f"{patient}_{class_name}_intensity.png"
    )

    fig.savefig(
        figure_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved histogram:", figure_path)

    # --------------------------------------------------------
    # Cleanup.
    # --------------------------------------------------------

    del ct
    del pred
    del gt

    del target_mask
    del component_map
    del selected_component
    del error_mask

    del error_background
    del error_other_vessel

    del true_positive
    del expanded
    del nearby_background

    del samples
    del nonempty

    gc.collect()


def main():

    print("=" * 70)

    print(
        "NEUROVESSEL V6.1 — CT INTENSITY ANALYSIS"
    )

    print("=" * 70)

    print("Patients:", len(CASES))
    print("Nearby background shell:", SHELL_ITERATIONS, "voxels")
    print("Maximum samples per group:", MAX_SAMPLE)
    print("Locked test partition will NOT be loaded.")
    print("No training or model inference.")

    for case in CASES:
        process_case(case)

    # --------------------------------------------------------
    # Export summary CSV.
    # --------------------------------------------------------

    result = pd.DataFrame(summary_rows)

    output_path = (
        OUTPUT_DIR / "ct_intensity_summary.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)

    print("INTENSITY ANALYSIS COMPLETED")

    print("=" * 70)

    print("\nSUMMARY:")

    if not result.empty:

        print(
            result[
                [
                    "patient_id",
                    "class",
                    "group",
                    "total_voxels",
                    "median",
                    "mean",
                    "std",
                    "p05",
                    "p95",
                ]
            ].to_string(index=False)
        )

    print("\nCSV OUTPUT:")
    print(output_path)

    print("\nHISTOGRAM DIRECTORY:")
    print(FIGURE_DIR)

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":
    main()
