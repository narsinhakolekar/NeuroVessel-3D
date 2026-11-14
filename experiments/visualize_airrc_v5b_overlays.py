
"""
NEUROVESSEL V5B
Full-volume prediction visual quality audit.

Selects:
    3 lowest-Dice validation patients
    2 median-performing validation patients
    1 highest-Dice validation patient

Displays:
    Original CT
    Ground truth PA contour: green
    Predicted PA contour: red

Uses threshold 0.70 predictions.
Does not access the locked test set.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from train_airrc_v5b import (
    load_splits,
    load_case,
    PROJECT_ROOT,
    METRICS_DIR
)


# ============================================================
# PATHS
# ============================================================

THRESHOLD = 0.70

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "airrc_v5b_fullvol"
    / "threshold_0.70"
)

METRICS_PATH = (
    METRICS_DIR
    / "airrc_v5b_fullvol"
    / "per_patient_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / "airrc_v5b_fullvol_threshold_070"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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


def select_cases(df):

    df = df[
        np.isclose(
            df["threshold"].astype(float),
            THRESHOLD
        )
    ].copy()

    if len(df) != 26:

        raise RuntimeError(
            f"Expected 26 patients, found {len(df)}"
        )

    df = df.sort_values(
        "dice"
    ).reset_index(drop=True)

    selected = []

    # Three weakest cases.
    selected.extend(
        df.iloc[:3].to_dict("records")
    )

    # Two cases closest to the median Dice.
    median_dice = df["dice"].median()

    median_candidates = df.iloc[3:-1].copy()

    median_candidates["distance"] = (
        median_candidates["dice"] - median_dice
    ).abs()

    for _, row in (
        median_candidates
        .sort_values("distance")
        .head(2)
        .iterrows()
    ):

        selected.append(
            row.to_dict()
        )

    # One strongest case.
    selected.append(
        df.iloc[-1].to_dict()
    )

    return selected


def show_overlay(
    ax,
    ct_slice,
    gt_slice,
    pred_slice,
    title
):

    ct_slice = np.clip(
        ct_slice,
        -1000,
        1000
    )

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
        title,
        fontsize=11
    )

    ax.axis("off")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NEUROVESSEL V5B VISUAL QUALITY AUDIT")
    print("=" * 65)

    if not METRICS_PATH.is_file():

        raise FileNotFoundError(
            f"Metrics not found: {METRICS_PATH}"
        )

    if not PREDICTION_DIR.is_dir():

        raise FileNotFoundError(
            f"Prediction directory not found: "
            f"{PREDICTION_DIR}"
        )

    # Load only validation records.
    _, validation_records, test_count = load_splits()

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

    selected_cases = select_cases(df)

    record_lookup = {
        record["uid"]: record
        for record in validation_records
    }

    # Line legend.
    legend_elements = [
        Line2D(
            [0], [0],
            color="lime",
            lw=2,
            label="Ground truth PA"
        ),
        Line2D(
            [0], [0],
            color="red",
            lw=2,
            label="Predicted PA"
        )
    ]

    summary_rows = []

    for index, item in enumerate(
        selected_cases,
        start=1
    ):

        uid = str(item["uid"])

        if uid not in record_lookup:

            raise RuntimeError(
                f"UID not found in validation split: {uid}"
            )

        record = record_lookup[uid]

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

        print("\n" + "-" * 60)

        print(
            f"CASE {index}/6:",
            uid
        )

        print(
            "Full-volume Dice:",
            round(float(item["dice"]), 4)
        )

        # Load CT and ground truth.
        image, mask = load_case(record)

        # Load predicted NIfTI.
        prediction_object = sitk.ReadImage(
            str(prediction_path)
        )

        prediction = sitk.GetArrayFromImage(
            prediction_object
        ).astype(np.uint8)

        # Check array dimensions.
        if prediction.shape != mask.shape:

            raise RuntimeError(
                f"Prediction and mask shape mismatch: "
                f"{prediction.shape} vs {mask.shape}"
            )

        # Check image geometry.
        ct_object = sitk.ReadImage(
            str(record["image"])
        )

        geometry_matches = (
            prediction_object.GetSpacing()
            == ct_object.GetSpacing()
            and prediction_object.GetOrigin()
            == ct_object.GetOrigin()
            and prediction_object.GetDirection()
            == ct_object.GetDirection()
        )

        if not geometry_matches:

            raise RuntimeError(
                f"Prediction geometry mismatch: {uid}"
            )

        gt = mask > 0
        pred = prediction > 0

        # Select high-foreground slices in each plane.
        z = int(
            np.argmax(
                np.count_nonzero(gt, axis=(1, 2))
            )
        )

        y = int(
            np.argmax(
                np.count_nonzero(gt, axis=(0, 2))
            )
        )

        x = int(
            np.argmax(
                np.count_nonzero(gt, axis=(0, 1))
            )
        )

        # 3 orthogonal planes.
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(17, 6)
        )

        fig.suptitle(
            f"NeuroVessel V5B | Case {index}: {uid}\n"
            f"Full-volume Dice @ 0.70: "
            f"{float(item['dice']):.4f}",
            fontsize=12
        )

        show_overlay(
            axes[0],
            image[z, :, :],
            gt[z, :, :],
            pred[z, :, :],
            f"Axial | Z={z}"
        )

        show_overlay(
            axes[1],
            image[:, y, :],
            gt[:, y, :],
            pred[:, y, :],
            f"Coronal | Y={y}"
        )

        show_overlay(
            axes[2],
            image[:, :, x],
            gt[:, :, x],
            pred[:, :, x],
            f"Sagittal | X={x}"
        )

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            frameon=False
        )

        plt.tight_layout(
            rect=[0, 0.08, 1, 0.91]
        )

        output_path = (
            OUTPUT_DIR
            / f"case_{index}_{safe_filename(uid)}.png"
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        summary_rows.append({

            "selection": index,

            "uid": uid,

            "dice": item["dice"],

            "precision": item["precision"],

            "recall": item["recall"],

            "geometry_matches": geometry_matches,

            "visualization": str(output_path)

        })

        print(
            "Geometry matches:",
            geometry_matches
        )

        print(
            "Saved:",
            output_path
        )

        del image
        del mask
        del prediction
        del prediction_object
        del ct_object
        del gt
        del pred

    summary_path = (
        OUTPUT_DIR
        / "visual_audit_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("VISUAL AUDIT IMAGES GENERATED")
    print("=" * 65)

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Summary:",
        summary_path
    )

    print(
        "\nThe locked test set was not accessed."
    )


if __name__ == "__main__":

    main()
