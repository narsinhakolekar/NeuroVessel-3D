
"""
NEUROVESSEL V5B
PROFESSOR-READY FOUR-PANEL COMPARISON

Generates three figures:
    1. Weakest validation case
    2. Median validation case
    3. Strongest validation case

Each figure contains:
    A. Original CT
    B. Ground Truth PA
    C. Predicted PA
    D. Overlay comparison

Uses:
    Processed AirRC validation CT and masks
    V5B full-volume predictions at threshold 0.70

The locked test partition is never accessed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from train_airrc_v5b import (
    load_splits,
    load_case,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

THRESHOLD = 0.70

METRICS_PATH = (
    METRICS_DIR
    / "airrc_v5b_fullvol"
    / "per_patient_metrics.csv"
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
    / "professor_comparisons_v5b"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HELPERS
# ============================================================

def safe_filename(uid):

    return "".join(
        c if c.isalnum() or c in "-_"
        else "_"
        for c in str(uid)
    )


def select_representative_cases(df):

    df = df[
        np.isclose(
            df["threshold"].astype(float),
            THRESHOLD
        )
    ].copy()

    df = df.sort_values(
        "dice"
    ).reset_index(drop=True)

    if len(df) != 26:

        raise RuntimeError(
            f"Expected 26 validation patients, "
            f"found {len(df)}"
        )

    weakest = df.iloc[0]

    median_index = len(df) // 2

    median = df.iloc[median_index]

    strongest = df.iloc[-1]

    return [
        ("Weakest", weakest),
        ("Median", median),
        ("Strongest", strongest)
    ]


def check_geometry(prediction_obj, ct_obj):

    return (
        prediction_obj.GetSize()
        == ct_obj.GetSize()

        and prediction_obj.GetSpacing()
        == ct_obj.GetSpacing()

        and prediction_obj.GetOrigin()
        == ct_obj.GetOrigin()

        and prediction_obj.GetDirection()
        == ct_obj.GetDirection()
    )


# ============================================================
# 3. CREATE ONE FOUR-PANEL FIGURE
# ============================================================

def create_comparison(
    category,
    record,
    dice,
    precision,
    recall
):

    uid = record["uid"]

    print("\n" + "=" * 65)

    print(
        f"Generating {category} case: {uid}"
    )

    print(
        "Full-volume Dice:",
        round(float(dice), 4)
    )

    # --------------------------------------------------------
    # Load original CT and ground truth
    # --------------------------------------------------------

    image, mask = load_case(record)

    ground_truth = mask > 0

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    filename = (
        safe_filename(uid)
        + "_pa_prediction.nii.gz"
    )

    prediction_path = (
        PREDICTION_DIR / filename
    )

    if not prediction_path.is_file():

        raise FileNotFoundError(
            f"Prediction missing: {prediction_path}"
        )

    prediction_obj = sitk.ReadImage(
        str(prediction_path)
    )

    prediction = sitk.GetArrayFromImage(
        prediction_obj
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Verify geometry
    # --------------------------------------------------------

    ct_obj = sitk.ReadImage(
        str(record["image"])
    )

    geometry_ok = check_geometry(
        prediction_obj,
        ct_obj
    )

    if not geometry_ok:

        raise RuntimeError(
            f"Prediction geometry mismatch: {uid}"
        )

    if prediction.shape != ground_truth.shape:

        raise RuntimeError(
            "Prediction and GT array dimensions differ."
        )

    prediction = prediction > 0

    # --------------------------------------------------------
    # Select a useful axial slice.
    # Select the slice containing the largest GT PA area.
    # All four panels will use the SAME slice.
    # --------------------------------------------------------

    foreground_per_slice = np.count_nonzero(
        ground_truth,
        axis=(1, 2)
    )

    z = int(
        np.argmax(foreground_per_slice)
    )

    ct_slice = image[z, :, :]

    gt_slice = ground_truth[z, :, :]

    pred_slice = prediction[z, :, :]

    # --------------------------------------------------------
    # Prepare figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    fig.suptitle(
        f"NeuroVessel V5B | {category} Validation Case\n"
        f"Full-volume Dice: {float(dice):.4f} | "
        f"Threshold: {THRESHOLD:.2f}",
        fontsize=17,
        fontweight="bold",
        y=0.97
    )

    # --------------------------------------------------------
    # PANEL 1 — Original CT
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=1000,
        origin="lower"
    )

    ax.set_title(
        f"1. Original CT | Axial Z={z}",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # --------------------------------------------------------
    # PANEL 2 — Ground Truth PA
    # --------------------------------------------------------

    ax = axes[0, 1]

    ax.set_facecolor("black")

    gt_cmap = ListedColormap([
        "black",
        "lime"
    ])

    ax.imshow(
        gt_slice.astype(np.uint8),
        cmap=gt_cmap,
        vmin=0,
        vmax=1,
        origin="lower"
    )

    ax.set_title(
        "2. Ground Truth PA",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # --------------------------------------------------------
    # PANEL 3 — Predicted PA
    # --------------------------------------------------------

    ax = axes[1, 0]

    ax.set_facecolor("black")

    pred_cmap = ListedColormap([
        "black",
        "red"
    ])

    ax.imshow(
        pred_slice.astype(np.uint8),
        cmap=pred_cmap,
        vmin=0,
        vmax=1,
        origin="lower"
    )

    ax.set_title(
        f"3. Predicted PA | Threshold={THRESHOLD:.2f}",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # --------------------------------------------------------
    # PANEL 4 — Ground Truth vs Prediction
    # --------------------------------------------------------

    ax = axes[1, 1]

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
            linewidths=1.1,
            origin="lower"
        )

    if np.any(pred_slice):

        ax.contour(
            pred_slice.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=1.0,
            origin="lower"
        )

    ax.set_title(
        "4. Ground Truth vs AI Prediction",
        fontsize=13,
        fontweight="bold"
    )

    ax.axis("off")

    # --------------------------------------------------------
    # Metric information
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.045,
        (
            f"Full-volume Dice: {float(dice):.4f}   |   "
            f"Precision: {float(precision):.4f}   |   "
            f"Recall: {float(recall):.4f}"
        ),
        ha="center",
        fontsize=11
    )

    fig.text(
        0.5,
        0.018,
        (
            "Green = Ground Truth PA    |    "
            "Red = Predicted PA"
        ),
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0.02, 0.08, 0.98, 0.91]
    )

    output_path = (
        OUTPUT_DIR
        / f"professor_{category.lower()}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(
        "Geometry verified:",
        geometry_ok
    )

    print(
        "Saved:",
        output_path
    )

    # Release arrays and image objects.
    del image
    del mask
    del ground_truth
    del prediction
    del prediction_obj
    del ct_obj

    return output_path


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)

    print(
        "NEUROVESSEL V5B — PROFESSOR COMPARISON GENERATOR"
    )

    print("=" * 65)

    if not METRICS_PATH.is_file():

        raise FileNotFoundError(
            f"Metrics not found: {METRICS_PATH}"
        )

    if not PREDICTION_DIR.is_dir():

        raise FileNotFoundError(
            f"Prediction directory missing: "
            f"{PREDICTION_DIR}"
        )

    # Load validation cases only.
    _, validation_records, test_count = load_splits()

    if len(validation_records) != 26:

        raise RuntimeError(
            "Expected exactly 26 validation patients."
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

    df = pd.read_csv(
        METRICS_PATH
    )

    selected = select_representative_cases(
        df
    )

    record_lookup = {
        record["uid"]: record
        for record in validation_records
    }

    summary = []

    for category, row in selected:

        uid = str(row["uid"])

        if uid not in record_lookup:

            raise RuntimeError(
                f"Validation case not found: {uid}"
            )

        record = record_lookup[uid]

        output_path = create_comparison(
            category=category,
            record=record,
            dice=row["dice"],
            precision=row["precision"],
            recall=row["recall"]
        )

        summary.append({

            "category": category,

            "uid": uid,

            "dice": float(row["dice"]),

            "precision": float(row["precision"]),

            "recall": float(row["recall"]),

            "threshold": THRESHOLD,

            "figure": str(output_path)

        })

    summary_path = (
        OUTPUT_DIR
        / "professor_comparison_summary.csv"
    )

    pd.DataFrame(
        summary
    ).to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 65)

    print(
        "ALL THREE COMPARISON FIGURES GENERATED"
    )

    print("=" * 65)

    print(
        "Output folder:",
        OUTPUT_DIR
    )

    print(
        "Summary:",
        summary_path
    )

    print(
        "\nThe locked test set remains untouched."
    )


if __name__ == "__main__":

    main()
