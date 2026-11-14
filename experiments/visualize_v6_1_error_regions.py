
from pathlib import Path
import gc

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy import ndimage


# ============================================================
# NEUROVESSEL V6.1
# SELECTED 3D ERROR COMPONENT VISUALIZATION
#
# 0529: Largest PV false-positive component
# 0763: Largest PV false-positive component
# 0848: Largest PA false-negative component
#
# Correctly isolates the selected 3D component.
# Corrects k-index plane coordinate orientation.
# Validates expected error voxel counts.
#
# Read-only. No training or inference.
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
    ROOT / "outputs" / "visualizations"
    / "airrc_v6_1_error_regions"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET COMPONENTS
# Coordinates: NIfTI array indices (i, j, k)
# ============================================================

CASES = [

    {
        "patient": "LIDC-IDRI-0529",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085",
        "class_name": "PV",
        "class_id": 2,
        "error_type": "FALSE POSITIVE",
        "center": (270, 186, 36),
        "component_id": 763,
        "expected_error_voxels": 123719,
    },

    {
        "patient": "LIDC-IDRI-0763",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.215785045378334625097907422785",
        "class_name": "PV",
        "class_id": 2,
        "error_type": "FALSE POSITIVE",
        "center": (174, 156, 119),
        "component_id": 16,
        "expected_error_voxels": 83189,
    },

    {
        "patient": "LIDC-IDRI-0848",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.252697338970999211181671881792",
        "class_name": "PA",
        "class_id": 1,
        "error_type": "FALSE NEGATIVE",
        "center": (166, 170, 178),
        "component_id": 1,
        "expected_error_voxels": 60570,
    },

]


# 26-connected 3D connectivity.
STRUCTURE = ndimage.generate_binary_structure(3, 3)

# Display radius in voxels.
RADIUS = 64


# ============================================================
# CROPPING AND VIEW EXTRACTION
# ============================================================

def crop_bounds(center, length):

    low = max(0, center - RADIUS)
    high = min(length, center + RADIUS + 1)

    return low, high


def extract_view(volume, view, center):

    i, j, k = center

    ni, nj, nk = volume.shape

    if view == "k":

        # Axial-like index plane.
        # Rows = i, columns = j.
        i0, i1 = crop_bounds(i, ni)
        j0, j1 = crop_bounds(j, nj)

        image = volume[i0:i1, j0:j1, k]

        extent = (
            j0 - 0.5,
            j1 - 0.5,
            i0 - 0.5,
            i1 - 0.5,
        )

        xlabel = "Voxel index j"
        ylabel = "Voxel index i"

        marker = (j, i)

        title = f"k-index plane: k = {k}"

    elif view == "j":

        # Rows = k, columns = i.
        i0, i1 = crop_bounds(i, ni)
        k0, k1 = crop_bounds(k, nk)

        image = volume[i0:i1, j, k0:k1].T

        extent = (
            i0 - 0.5,
            i1 - 0.5,
            k0 - 0.5,
            k1 - 0.5,
        )

        xlabel = "Voxel index i"
        ylabel = "Voxel index k"

        marker = (i, k)

        title = f"j-index plane: j = {j}"

    elif view == "i":

        # Rows = k, columns = j.
        j0, j1 = crop_bounds(j, nj)
        k0, k1 = crop_bounds(k, nk)

        image = volume[i, j0:j1, k0:k1].T

        extent = (
            j0 - 0.5,
            j1 - 0.5,
            k0 - 0.5,
            k1 - 0.5,
        )

        xlabel = "Voxel index j"
        ylabel = "Voxel index k"

        marker = (j, k)

        title = f"i-index plane: i = {i}"

    else:
        raise ValueError(f"Invalid view: {view}")

    return image, extent, xlabel, ylabel, marker, title


# ============================================================
# DRAW ONE ORTHOGONAL VIEW
# ============================================================

def draw_view(
    ax,
    ct,
    gt_mask,
    pred_mask,
    selected_error_mask,
    view,
    center,
    error_type,
):

    ct2d, extent, xlabel, ylabel, marker, title = (
        extract_view(ct, view, center)
    )

    gt2d, _, _, _, _, _ = extract_view(
        gt_mask, view, center
    )

    pred2d, _, _, _, _, _ = extract_view(
        pred_mask, view, center
    )

    error2d, _, _, _, _, _ = extract_view(
        selected_error_mask, view, center
    )

    # CT window from the current view.
    low, high = np.percentile(
        ct2d,
        [2, 98],
    )

    if high <= low:
        low = float(np.min(ct2d))
        high = float(np.max(ct2d)) + 1e-5

    ax.imshow(
        ct2d,
        cmap="gray",
        origin="lower",
        extent=extent,
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )

    # Ground-truth target class boundary.
    if np.any(gt2d):

        ax.contour(
            gt2d.astype(np.float32),
            levels=[0.5],
            colors=["lime"],
            linewidths=1.3,
            origin="lower",
            extent=extent,
        )

    # Predicted target class boundary.
    if np.any(pred2d):

        ax.contour(
            pred2d.astype(np.float32),
            levels=[0.5],
            colors=["red"],
            linewidths=1.2,
            origin="lower",
            extent=extent,
        )

    # IMPORTANT:
    # Only the selected 3D component's error voxels
    # are highlighted here.
    if np.any(error2d):

        if error_type == "FALSE POSITIVE":
            error_color = "magenta"
        else:
            error_color = "orange"

        masked_error = np.ma.masked_where(
            ~error2d,
            error2d.astype(np.float32),
        )

        ax.imshow(
            masked_error,
            cmap=ListedColormap([error_color]),
            alpha=0.55,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )

    # Selected component centroid.
    ax.scatter(
        [marker[0]],
        [marker[1]],
        marker="+",
        color="white",
        s=90,
        linewidths=2,
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    ax.set_aspect("equal")
    ax.grid(False)


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_case(case):

    patient = case["patient"]
    uid = case["uid"]

    class_name = case["class_name"]
    class_id = case["class_id"]

    error_type = case["error_type"]

    center = case["center"]

    component_id = case["component_id"]

    expected_errors = case["expected_error_voxels"]

    print("\n" + "=" * 70)
    print("PATIENT:", patient)
    print("CLASS:", class_name)
    print("ERROR:", error_type)
    print("COMPONENT ID:", component_id)
    print("CENTROID:", center)
    print("=" * 70)

    # --------------------------------------------------------
    # Locate files.
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
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load volumes.
    # --------------------------------------------------------

    print("Loading CT...")

    ct_img = nib.load(str(ct_path))

    print("Loading prediction...")

    pred_img = nib.load(str(pred_path))

    print("Loading ground truth...")

    gt_img = nib.load(str(gt_path))

    # --------------------------------------------------------
    # Verify geometry.
    # --------------------------------------------------------

    if not (
        ct_img.shape == pred_img.shape == gt_img.shape
    ):

        raise ValueError(
            f"Volume shape mismatch: {patient}\n"
            f"CT: {ct_img.shape}\n"
            f"Prediction: {pred_img.shape}\n"
            f"Ground truth: {gt_img.shape}"
        )

    if not (
        np.allclose(
            ct_img.affine,
            gt_img.affine,
            atol=1e-3,
        )
        and
        np.allclose(
            pred_img.affine,
            gt_img.affine,
            atol=1e-3,
        )
    ):

        raise ValueError(
            f"Affine mismatch for patient {patient}"
        )

    # --------------------------------------------------------
    # Load data.
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
            f"Non-finite CT values: {patient}"
        )

    if not np.isin(pred, [0, 1, 2]).all():

        raise ValueError(
            f"Unexpected prediction classes: {patient}"
        )

    if not np.isin(gt, [0, 1, 2]).all():

        raise ValueError(
            f"Unexpected ground-truth classes: {patient}"
        )

    if any(
        center[d] < 0 or center[d] >= ct.shape[d]
        for d in range(3)
    ):

        raise ValueError(
            f"Centroid outside volume: {patient}"
        )

    print("Volume shape:", ct.shape)

    print(
        "Voxel spacing:",
        gt_img.header.get_zooms()[:3],
    )

    print("Centroid:", center)

    # --------------------------------------------------------
    # Create target-class masks.
    # --------------------------------------------------------

    gt_mask = gt == class_id

    pred_mask = pred == class_id

    # --------------------------------------------------------
    # Reconstruct the same 26-connected components.
    # --------------------------------------------------------

    if error_type == "FALSE POSITIVE":

        print("Reconstructing predicted components...")

        component_map, n_components = ndimage.label(
            pred_mask,
            structure=STRUCTURE,
        )

        if component_id > n_components:

            raise ValueError(
                f"Component {component_id} not found. "
                f"Total components: {n_components}"
            )

        selected_component = (
            component_map == component_id
        )

        # Only wrong-class voxels inside this component.
        selected_error_mask = (
            selected_component
            & (gt != class_id)
        )

    elif error_type == "FALSE NEGATIVE":

        print("Reconstructing ground-truth components...")

        component_map, n_components = ndimage.label(
            gt_mask,
            structure=STRUCTURE,
        )

        if component_id > n_components:

            raise ValueError(
                f"Component {component_id} not found. "
                f"Total components: {n_components}"
            )

        selected_component = (
            component_map == component_id
        )

        # Only missed voxels inside this component.
        selected_error_mask = (
            selected_component
            & (pred != class_id)
        )

    else:

        raise ValueError(
            f"Unsupported error type: {error_type}"
        )

    # --------------------------------------------------------
    # Validate selected component error count.
    # --------------------------------------------------------

    selected_component_voxels = int(
        np.count_nonzero(selected_component)
    )

    actual_errors = int(
        np.count_nonzero(selected_error_mask)
    )

    print(
        "Selected component voxels:",
        f"{selected_component_voxels:,}",
    )

    print(
        "Selected component error voxels:",
        f"{actual_errors:,}",
    )

    print(
        "Expected error voxels:",
        f"{expected_errors:,}",
    )

    if actual_errors != expected_errors:

        raise ValueError(
            "\nERROR COUNT VALIDATION FAILED!\n"
            f"Patient: {patient}\n"
            f"Expected: {expected_errors:,}\n"
            f"Actual: {actual_errors:,}\n"
            "No figure will be saved."
        )

    print("Component validation: PASSED")

    del component_map, selected_component

    # --------------------------------------------------------
    # Generate three perpendicular views.
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    for ax, view in zip(
        axes,
        ["k", "j", "i"],
    ):

        draw_view(
            ax=ax,
            ct=ct,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            selected_error_mask=selected_error_mask,
            view=view,
            center=center,
            error_type=error_type,
        )

    # --------------------------------------------------------
    # Legend.
    # --------------------------------------------------------

    error_color = (
        "magenta"
        if error_type == "FALSE POSITIVE"
        else "orange"
    )

    legend_handles = [

        Line2D(
            [0],
            [0],
            color="lime",
            lw=2,
            label="Ground truth",
        ),

        Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            label="Prediction",
        ),

        Line2D(
            [0],
            [0],
            color=error_color,
            lw=5,
            label="Selected component error",
        ),

        Line2D(
            [0],
            [0],
            color="white",
            marker="+",
            linestyle="none",
            markersize=10,
            label="Error centroid",
        ),

    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
    )

    # --------------------------------------------------------
    # Title.
    # --------------------------------------------------------

    fig.suptitle(
        f"{patient} | V6.1 {class_name} {error_type}\n"
        f"Component {component_id} | "
        f"Centroid {center} | "
        f"{actual_errors:,} selected error voxels",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0.10, 1, 0.88],
    )

    # --------------------------------------------------------
    # Save figure.
    # --------------------------------------------------------

    output_name = (
        f"{patient}_{class_name}_"
        f"{error_type.replace(' ', '_')}.png"
    )

    output_path = OUTPUT_DIR / output_name

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)

    # --------------------------------------------------------
    # Release memory.
    # --------------------------------------------------------

    del ct
    del pred
    del gt

    del gt_mask
    del pred_mask
    del selected_error_mask

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "NEUROVESSEL V6.1 — "
        "CORRECTED ERROR VISUALIZATION"
    )

    print("=" * 70)

    print("Patients:", len(CASES))

    print("Views per patient: 3")

    print("Connectivity: 26-connected")

    print("Only selected components will be highlighted.")

    print("Locked test partition will NOT be loaded.")

    print("No model inference or training.")

    for case in CASES:

        process_case(case)

    print("\n" + "=" * 70)

    print("ALL VISUALIZATIONS COMPLETED")

    print("=" * 70)

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nExpected selected-component error counts:")

    for case in CASES:

        print(
            case["patient"],
            case["class_name"],
            f"{case['expected_error_voxels']:,}",
        )

    print("\nLocked test partition was not loaded.")


if __name__ == "__main__":

    main()
