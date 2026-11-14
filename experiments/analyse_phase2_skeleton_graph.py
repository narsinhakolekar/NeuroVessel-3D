
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skan
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v6_1_final_test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "phase2_graph"
)

EXPECTED_CASES = 26
CLASSES = {
    1: "PA",
    2: "PV",
}

# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def get_graph_node_counts(graph):
    """
    Count graph nodes by their number of connected neighbours.

    These are digital graph-degree candidates, not validated
    anatomical endpoints or vessel bifurcations.
    """
    degrees = np.asarray(graph.getnnz(axis=1)).ravel()

    return {
        "degree_0_nodes": int(np.sum(degrees == 0)),
        "degree_1_nodes": int(np.sum(degrees == 1)),
        "degree_2_nodes": int(np.sum(degrees == 2)),
        "degree_3plus_nodes": int(np.sum(degrees >= 3)),
        "total_graph_nodes": int(len(degrees)),
    }

def analyse_one_class(mask, spacing_xyz, class_name, uid):
    """
    Extract a 3D skeleton graph and graph-path measurements.

    NumPy array order is Z,Y,X.
    SimpleITK spacing order is X,Y,Z.
    Therefore spacing is reversed before passing to Skan.
    """

    binary = mask.astype(bool)

    voxel_count = int(np.count_nonzero(binary))

    result = {
        "uid": uid,
        "class": class_name,
        "mask_voxels": voxel_count,
        "skeleton_voxels": 0,
        "graph_paths": 0,
        "total_graph_path_length_mm": 0.0,
        "mean_path_length_mm": 0.0,
        "median_path_length_mm": 0.0,
        "max_path_length_mm": 0.0,
        "min_path_length_mm": 0.0,
        "degree_0_nodes": 0,
        "degree_1_nodes": 0,
        "degree_2_nodes": 0,
        "degree_3plus_nodes": 0,
        "total_graph_nodes": 0,
    }

    branch_rows = []

    if voxel_count == 0:
        result["status"] = "EMPTY_MASK"
        return result, branch_rows

    # Lee 3D skeletonization
    skeleton_array = skeletonize(binary, method="lee")

    skeleton_voxels = int(np.count_nonzero(skeleton_array))
    result["skeleton_voxels"] = skeleton_voxels

    if skeleton_voxels == 0:
        result["status"] = "EMPTY_SKELETON"
        return result, branch_rows

    # Convert X,Y,Z spacing into NumPy Z,Y,X axis order.
    spacing_zyx = tuple(float(x) for x in spacing_xyz[::-1])

    sk = Skeleton(
        skeleton_array,
        spacing=spacing_zyx
    )

    graph = sk.graph

    node_counts = get_graph_node_counts(graph)
    result.update(node_counts)

    # Skan branch/path summary
    summary_df = summarize(sk, separator="_")

    if summary_df.empty:
        result["status"] = "NO_GRAPH_PATHS"
        return result, branch_rows

    path_lengths = np.asarray(
        sk.path_lengths(),
        dtype=np.float64
    )

    path_lengths = path_lengths[
        np.isfinite(path_lengths)
    ]

    result["graph_paths"] = int(len(path_lengths))

    if len(path_lengths) > 0:
        result["total_graph_path_length_mm"] = float(
            np.sum(path_lengths)
        )
        result["mean_path_length_mm"] = float(
            np.mean(path_lengths)
        )
        result["median_path_length_mm"] = float(
            np.median(path_lengths)
        )
        result["max_path_length_mm"] = float(
            np.max(path_lengths)
        )
        result["min_path_length_mm"] = float(
            np.min(path_lengths)
        )

    # Retain individual graph-path measurements.
    for path_index, row in summary_df.iterrows():
        branch_rows.append({
            "uid": uid,
            "class": class_name,
            "path_index": int(path_index),
            "source_node": int(row["node_id_src"]),
            "destination_node": int(row["node_id_dst"]),
            "branch_distance_mm": float(
                row["branch_distance"]
            ),
            "euclidean_distance_mm": float(
                row["euclidean_distance"]
            ),
            "branch_type_code": int(row["branch_type"]),
        })

    result["status"] = "OK"

    return result, branch_rows

# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not PREDICTION_DIR.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {PREDICTION_DIR}"
        )

    prediction_files = sorted(
        PREDICTION_DIR.glob("*_pred_multiclass.nii.gz")
    )

    print("=" * 65)
    print("NEUROVESSEL - PHASE 2 GRAPH-BASED SKELETON ANALYSIS")
    print("=" * 65)

    print("Prediction directory:", PREDICTION_DIR)
    print("Found prediction files:", len(prediction_files))

    if len(prediction_files) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} test predictions, "
            f"but found {len(prediction_files)}."
        )

    case_rows = []
    branch_rows = []
    processed_uids = set()

    for case_number, pred_path in enumerate(
        prediction_files, start=1
    ):

        uid = pred_path.name.replace(
            "_pred_multiclass.nii.gz", ""
        )

        if uid in processed_uids:
            raise RuntimeError(f"Duplicate UID: {uid}")

        processed_uids.add(uid)

        print(
            f"\n[{case_number}/{EXPECTED_CASES}] "
            f"Processing {uid}"
        )

        image = sitk.ReadImage(str(pred_path))

        array = sitk.GetArrayFromImage(image)

        spacing_xyz = image.GetSpacing()

        if array.ndim != 3:
            raise ValueError(
                f"{uid}: Expected 3D image, got {array.shape}"
            )

        if not np.all(np.isfinite(spacing_xyz)):
            raise ValueError(f"{uid}: Invalid spacing")

        if any(float(s) <= 0 for s in spacing_xyz):
            raise ValueError(f"{uid}: Non-positive spacing")

        labels = set(
            np.unique(array).astype(int).tolist()
        )

        if not labels.issubset({0, 1, 2}):
            raise ValueError(
                f"{uid}: Unexpected labels {labels}"
            )

        print("  Shape Z,Y,X:", array.shape)
        print("  Spacing X,Y,Z:", spacing_xyz)

        for class_id, class_name in CLASSES.items():

            print(f"  Analysing {class_name}...")

            mask = array == class_id

            result, paths = analyse_one_class(
                mask=mask,
                spacing_xyz=spacing_xyz,
                class_name=class_name,
                uid=uid
            )

            case_rows.append(result)
            branch_rows.extend(paths)

            print(
                f"    Skeleton voxels: "
                f"{result['skeleton_voxels']}"
            )
            print(
                f"    Graph paths: "
                f"{result['graph_paths']}"
            )
            print(
                f"    Total path length: "
                f"{result['total_graph_path_length_mm']:.2f} mm"
            )
            print(
                f"    Graph status: {result['status']}"
            )

    # -----------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------

    case_df = pd.DataFrame(case_rows)
    branch_df = pd.DataFrame(branch_rows)

    case_csv = OUTPUT_DIR / "phase2_graph_26cases.csv"
    branch_csv = OUTPUT_DIR / "phase2_graph_branches.csv"
    summary_csv = OUTPUT_DIR / "phase2_graph_summary.csv"

    case_df.to_csv(case_csv, index=False)
    branch_df.to_csv(branch_csv, index=False)

    # Descriptive statistics, separately by vascular class.
    numeric_columns = [
        "mask_voxels",
        "skeleton_voxels",
        "graph_paths",
        "total_graph_path_length_mm",
        "mean_path_length_mm",
        "median_path_length_mm",
        "max_path_length_mm",
        "degree_1_nodes",
        "degree_3plus_nodes",
        "total_graph_nodes",
    ]

    summary_rows = []

    for class_name in CLASSES.values():

        subset = case_df[
            case_df["class"] == class_name
        ]

        for metric in numeric_columns:

            values = pd.to_numeric(
                subset[metric],
                errors="coerce"
            ).dropna()

            summary_rows.append({
                "class": class_name,
                "metric": metric,
                "n_cases": int(values.count()),
                "mean": float(values.mean()) if len(values) else np.nan,
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    # -----------------------------------------------------
    # QUALITY CONTROL
    # -----------------------------------------------------

    expected_rows = EXPECTED_CASES * len(CLASSES)

    if len(case_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} case-class rows; "
            f"found {len(case_df)}"
        )

    if case_df.duplicated(["uid", "class"]).any():
        raise RuntimeError("Duplicate UID/class records detected")

    if set(case_df["uid"]) != processed_uids:
        raise RuntimeError("UID consistency check failed")

    if not (case_df["status"] == "OK").all():
        print("\nWARNING: Some case-class analyses need review.")
        print(
            case_df.loc[
                case_df["status"] != "OK",
                ["uid", "class", "status"]
            ].to_string(index=False)
        )

    config = {
        "experiment": "Phase 2 graph-based skeleton analysis",
        "cases": EXPECTED_CASES,
        "classes": CLASSES,
        "skeletonization": "scikit-image skeletonize, Lee method",
        "graph_library": "Skan",
        "skan_version": skan.__version__,
        "array_axis_order": "Z,Y,X",
        "spacing_source_order": "SimpleITK X,Y,Z",
        "skan_spacing_order": "Z,Y,X",
        "graph_endpoint_definition": "Graph nodes with degree 1",
        "graph_junction_candidate_definition": "Graph nodes with degree >= 3",
        "interpretation_warning": (
            "Graph-degree candidates and graph paths are digital "
            "skeleton measurements, not validated anatomical "
            "endpoints, bifurcations or branches."
        ),
        "prediction_masks_modified": False,
        "test_set_used_for_tuning": False,
    }

    config_path = OUTPUT_DIR / "phase2_graph_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    qc_path = OUTPUT_DIR / "phase2_graph_qc.txt"

    with open(qc_path, "w", encoding="utf-8") as f:
        f.write("NEUROVESSEL PHASE 2 GRAPH ANALYSIS QC\n")
        f.write("=" * 50 + "\n")
        f.write(f"Prediction files: {len(prediction_files)}\n")
        f.write(f"Unique UIDs: {len(processed_uids)}\n")
        f.write(f"Case-class rows: {len(case_df)}\n")
        f.write(f"Expected rows: {expected_rows}\n")
        f.write(
            f"Successful analyses: "
            f"{int((case_df['status'] == 'OK').sum())}\n"
        )
        f.write(
            f"Non-OK analyses: "
            f"{int((case_df['status'] != 'OK').sum())}\n"
        )
        f.write("Duplicate UID/class records: 0\n")
        f.write("Prediction masks modified: NO\n")
        f.write("Test-set tuning: NO\n")

    print("\n" + "=" * 65)
    print("GRAPH ANALYSIS COMPLETED")
    print("=" * 65)

    print("Patient-level CSV:", case_csv)
    print("Graph-path CSV:", branch_csv)
    print("Summary CSV:", summary_csv)
    print("Configuration:", config_path)
    print("QC report:", qc_path)

    print("\nCase-class rows:", len(case_df))
    print("Graph paths:", len(branch_df))
    print(
        "Successful analyses:",
        int((case_df["status"] == "OK").sum()),
        "/",
        expected_rows
    )

    print("\nSUMMARY:")
    print(
        summary_df[
            summary_df["metric"].isin([
                "skeleton_voxels",
                "graph_paths",
                "total_graph_path_length_mm",
                "degree_1_nodes",
                "degree_3plus_nodes",
            ])
        ].to_string(index=False)
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nEXPERIMENT FAILED:", str(exc))
        raise