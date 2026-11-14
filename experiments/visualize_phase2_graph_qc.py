
from pathlib import Path
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "outputs/predictions/airrc_v6_1_final_test"

GRAPH_CSV = (
    ROOT / "outputs/metrics/phase2_graph"
    / "phase2_graph_26cases.csv"
)

BRANCH_CSV = (
    ROOT / "outputs/metrics/phase2_graph"
    / "phase2_graph_branches.csv"
)

OUT = ROOT / "outputs/visualizations/phase2_graph_qc"

OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

# Downsample for visualization only.
# Original prediction masks remain unchanged.
MAX_POINTS = 25000

# Select low, median and high graph-path-count cases.
REPRESENTATIVE_CASES = ["low", "median", "high"]

RANDOM_SEED = 42

# -------------------------------------------------------
# LOAD METADATA
# -------------------------------------------------------

metrics = pd.read_csv(GRAPH_CSV)
branches = pd.read_csv(BRANCH_CSV)

if metrics["uid"].nunique() != 26:
    raise RuntimeError("Expected 26 unique patients.")

if len(metrics) != 52:
    raise RuntimeError("Expected 52 PA/PV records.")

rng = np.random.default_rng(RANDOM_SEED)

def select_cases(class_name):

    subset = metrics[
        metrics["class"] == class_name
    ].copy()

    subset = subset.sort_values("graph_paths")

    return {
        "low": subset.iloc[0],
        "median": subset.iloc[len(subset) // 2],
        "high": subset.iloc[-1],
    }

# -------------------------------------------------------
# 3D PLOT HELPERS
# -------------------------------------------------------

def draw_3d_points(ax, coordinates, title):

    if len(coordinates) == 0:
        ax.set_title(title + " (empty)")
        return

    if len(coordinates) > MAX_POINTS:
        selected = rng.choice(
            len(coordinates),
            MAX_POINTS,
            replace=False
        )
        coordinates = coordinates[selected]

    z, y, x = coordinates.T

    ax.scatter(
        x, y, z,
        s=0.12,
        alpha=0.45
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=25, azim=45)

def create_case_figure(uid, class_id, class_name, category):

    pred_path = (
        PRED_DIR
        / f"{uid}_pred_multiclass.nii.gz"
    )

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    image = sitk.ReadImage(str(pred_path))
    array = sitk.GetArrayFromImage(image)

    mask = array == class_id

    if not np.any(mask):
        print(f"WARNING: Empty {class_name} mask: {uid}")
        return

    skeleton = skeletonize(mask, method="lee")

    mask_points = np.argwhere(mask)
    skeleton_points = np.argwhere(skeleton)

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    draw_3d_points(
        ax1,
        mask_points,
        f"{class_name} predicted mask"
    )

    draw_3d_points(
        ax2,
        skeleton_points,
        f"{class_name} Lee skeleton"
    )

    fig.suptitle(
        f"{class_name} | {category.upper()} graph-path case\n"
        f"UID: {uid}",
        fontsize=10
    )

    fig.tight_layout()

    filename = f"{uid}_{class_name}_{category}.png"

    fig.savefig(
        OUT / filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return filename

# -------------------------------------------------------
# GRAPH-PATH DISTRIBUTION FIGURE
# -------------------------------------------------------

def create_distribution_figure():

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    for ax, class_name in zip(
        axes, ["PA", "PV"]
    ):

        subset = branches[
            branches["class"] == class_name
        ]

        lengths = subset[
            "branch_distance_mm"
        ].dropna()

        ax.hist(
            lengths,
            bins=60
        )

        ax.set_title(
            f"{class_name} graph-path length distribution"
        )

        ax.set_xlabel("Graph-path distance (mm)")
        ax.set_ylabel("Number of paths")

    fig.tight_layout()

    fig.savefig(
        OUT / "graph_path_length_distributions.png",
        dpi=200
    )

    plt.close(fig)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    figure_records = []

    print("=" * 60)
    print("NEUROVESSEL PHASE 2 GRAPH QC VISUALIZATION")
    print("=" * 60)

    for class_id, class_name in [
        (1, "PA"),
        (2, "PV")
    ]:

        selected = select_cases(class_name)

        for category in REPRESENTATIVE_CASES:

            row = selected[category]
            uid = row["uid"]

            print(
                f"\n{class_name} | {category} | "
                f"graph paths={int(row['graph_paths'])}"
            )

            filename = create_case_figure(
                uid,
                class_id,
                class_name,
                category
            )

            figure_records.append({
                "class": class_name,
                "selection": category,
                "uid": uid,
                "graph_paths": int(row["graph_paths"]),
                "skeleton_voxels": int(row["skeleton_voxels"]),
                "degree_1_nodes": int(row["degree_1_nodes"]),
                "degree_3plus_nodes": int(row["degree_3plus_nodes"]),
                "figure": filename,
            })

    create_distribution_figure()

    selection_df = pd.DataFrame(figure_records)

    selection_df.to_csv(
        OUT / "representative_case_selection.csv",
        index=False
    )

    config = {
        "experiment": "Phase 2 graph visual QC",
        "patients_selected_per_class": 3,
        "classes": ["PA", "PV"],
        "selection": "Low, median and high graph-path counts",
        "maximum_points_per_plot": MAX_POINTS,
        "downsampling": "Visualization only",
        "prediction_masks_modified": False,
        "test_set_used_for_tuning": False,
        "purpose": "Qualitative visual inspection",
    }

    with open(
        OUT / "visualization_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 60)
    print("VISUAL QC FIGURES GENERATED")
    print("=" * 60)

    print("Output directory:", OUT)
    print("Representative figures:", len(figure_records))
    print("Distribution figure generated.")
    print("Original prediction masks modified: NO")

    print("\nSelected cases:")
    print(selection_df.to_string(index=False))

if __name__ == "__main__":
    main()