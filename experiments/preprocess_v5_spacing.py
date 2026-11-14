
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output, resample_from_to


# ============================================================
# NEUROVESSEL V5 - SPACING STANDARDIZATION
# ============================================================

TARGET_SPACING = (0.65, 0.65, 1.0)

ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = ROOT / "data" / "splits.json"
RAW_ROOT = ROOT / "data" / "raw" / "train"
OUTPUT_ROOT = ROOT / "data" / "processed_v5"

SUMMARY_PATH = (
    ROOT / "outputs" / "metrics" / "v5_preprocessing_summary.json"
)

PROCESS_SPLITS = ("train", "validation")
FORCE_REPROCESS = False


# ============================================================
# HELPERS
# ============================================================

def save_nifti(data, reference, output_path, dtype):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = reference.header.copy()
    header.set_data_dtype(dtype)

    output = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
        header=header
    )

    output.set_qform(reference.affine, code=1)
    output.set_sform(reference.affine, code=1)

    nib.save(output, str(output_path))


def validate_split(split):

    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = set(split["test"])

    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("Duplicate training patient IDs.")

    if len(validation_ids) != len(set(validation_ids)):
        raise RuntimeError("Duplicate validation patient IDs.")

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Train-validation leakage.")

    if set(train_ids) & test_ids:
        raise RuntimeError("Training-test leakage.")

    if set(validation_ids) & test_ids:
        raise RuntimeError("Validation-test leakage.")

    if not train_ids or not validation_ids:
        raise RuntimeError("Training or validation split is empty.")

    return {
        "train": train_ids,
        "validation": validation_ids,
    }


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient_id, split_name):

    source = RAW_ROOT / patient_id

    ct_path = (
        source / "image" / f"{patient_id}.nii.gz"
    )

    label_path = (
        source / "label" / f"{patient_id}.nii.gz"
    )

    output_case = OUTPUT_ROOT / split_name / patient_id

    output_ct = (
        output_case / "image" / f"{patient_id}.nii.gz"
    )

    output_label = (
        output_case / "label" / f"{patient_id}.nii.gz"
    )

    geometry_path = output_case / "geometry.json"

    # Resume support.
    existing = [
        output_ct.exists(),
        output_label.exists(),
        geometry_path.exists(),
    ]

    if all(existing) and not FORCE_REPROCESS:
        print(f"SKIP {patient_id}: already processed")
        return "skipped"

    if any(existing) and not FORCE_REPROCESS:
        raise RuntimeError(
            f"Incomplete previous output for {patient_id}. "
            "Inspect or remove its processed_v5 folder before retrying."
        )

    if not ct_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing CT or label for {patient_id}"
        )

    start = time.time()

    ct_img = nib.load(str(ct_path))
    label_img = nib.load(str(label_path))

    # Check original image/label geometry.
    if ct_img.shape != label_img.shape:
        raise ValueError(
            f"{patient_id}: original CT-label shape mismatch"
        )

    if not np.allclose(
        ct_img.affine,
        label_img.affine,
        atol=1e-4
    ):
        raise ValueError(
            f"{patient_id}: original CT-label affine mismatch"
        )

    original_ct_shape = list(ct_img.shape)
    original_spacing = list(
        map(float, ct_img.header.get_zooms()[:3])
    )
    original_orientation = list(
        nib.aff2axcodes(ct_img.affine)
    )
    original_affine = ct_img.affine.tolist()

    # --------------------------------------------------------
    # Resample CT to target physical spacing.
    # Linear interpolation is used for intensity images.
    # Outside values are set to -1000 HU (air).
    # --------------------------------------------------------

    ct_resampled = resample_to_output(
        ct_img,
        voxel_sizes=TARGET_SPACING,
        order=1,
        mode="constant",
        cval=-1000.0
    )

    # --------------------------------------------------------
    # Resample LABEL onto the exact CT output grid.
    # Nearest-neighbour interpolation preserves label classes.
    # --------------------------------------------------------

    label_resampled = resample_from_to(
        label_img,
        (ct_resampled.shape, ct_resampled.affine),
        order=0,
        mode="constant",
        cval=0
    )

    ct_data = np.asarray(
        ct_resampled.dataobj,
        dtype=np.float32
    )

    label_data = np.asarray(
        label_resampled.dataobj
    )

    if not np.isfinite(ct_data).all():
        raise ValueError(
            f"{patient_id}: non-finite CT values after resampling"
        )

    # Convert to binary mask.
    label_data = (label_data > 0).astype(np.uint8)

    if not label_data.any():
        raise ValueError(
            f"{patient_id}: empty label after resampling"
        )

    if ct_data.shape != label_data.shape:
        raise RuntimeError(
            f"{patient_id}: resampled shape mismatch"
        )

    if not np.allclose(
        ct_resampled.affine,
        label_resampled.affine,
        atol=1e-5
    ):
        raise RuntimeError(
            f"{patient_id}: resampled affine mismatch"
        )

    output_orientation = list(
        nib.aff2axcodes(ct_resampled.affine)
    )

    # --------------------------------------------------------
    # Save outputs separately from original data.
    # --------------------------------------------------------

    save_nifti(
        ct_data,
        ct_resampled,
        output_ct,
        np.float32
    )

    save_nifti(
        label_data,
        label_resampled,
        output_label,
        np.uint8
    )

    metadata = {
        "patient_id": patient_id,
        "split": split_name,
        "target_spacing_mm": list(TARGET_SPACING),
        "original_shape": original_ct_shape,
        "original_spacing_mm": original_spacing,
        "original_orientation": original_orientation,
        "original_affine": original_affine,
        "resampled_shape": list(ct_data.shape),
        "resampled_spacing_mm": list(
            map(float, ct_resampled.header.get_zooms()[:3])
        ),
        "resampled_orientation": output_orientation,
        "resampled_affine": ct_resampled.affine.tolist(),
        "label_foreground_voxels": int(label_data.sum()),
        "label_values": [int(v) for v in np.unique(label_data)],
        "ct_interpolation": "linear",
        "label_interpolation": "nearest",
        "ct_outside_value_hu": -1000.0,
        "test_set_processed": False,
    }

    geometry_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    elapsed = time.time() - start

    print(
        f"OK {patient_id} | "
        f"{original_ct_shape} -> {ct_data.shape} | "
        f"Foreground={label_data.sum():,} | "
        f"{elapsed:.1f}s"
    )

    return "processed"


# ============================================================
# MAIN
# ============================================================

def main():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    split_data = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    splits = validate_split(split_data)

    total = sum(len(v) for v in splits.values())

    print("=" * 70)
    print("NEUROVESSEL V5 - SPACING PREPROCESSING")
    print("=" * 70)

    print("Target spacing:", TARGET_SPACING)
    print("Training patients:", len(splits["train"]))
    print("Validation patients:", len(splits["validation"]))
    print("Total to process:", total)
    print("Test patients: EXCLUDED")
    print("Output:", OUTPUT_ROOT)

    print("\nStarting preprocessing...\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_spacing_mm": list(TARGET_SPACING),
        "train_patients": len(splits["train"]),
        "validation_patients": len(splits["validation"]),
        "test_patients_processed": False,
        "results": {},
    }

    processed = 0
    skipped = 0

    for split_name in PROCESS_SPLITS:

        patient_ids = splits[split_name]

        print("\n" + "=" * 65)
        print(f"PROCESSING {split_name.upper()}")
        print("=" * 65)

        for index, patient_id in enumerate(patient_ids, 1):

            print(
                f"\n[{index}/{len(patient_ids)}] ",
                end=""
            )

            status = process_patient(
                patient_id,
                split_name
            )

            if status == "processed":
                processed += 1
            else:
                skipped += 1

    summary["processed_cases"] = processed
    summary["skipped_cases"] = skipped
    summary["total_cases"] = total

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("V5 PREPROCESSING COMPLETE")
    print("=" * 70)

    print("Newly processed:", processed)
    print("Already processed:", skipped)
    print("Total cases:", total)
    print("Summary:", SUMMARY_PATH)
    print("Test set accessed: NO")


if __name__ == "__main__":
    main()
