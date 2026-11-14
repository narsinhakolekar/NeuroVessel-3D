
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.filters import frangi


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = PROJECT_ROOT / "data" / "splits.json"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "train"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "frangi_features"

REPORT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "test_frangi_precomputation.json"
)

TARGET_SPACING_MM = 1.0
PHYSICAL_SIGMAS_MM = (1.0, 2.0, 3.0)

HU_MIN = -1000
HU_MAX = 500


# ============================================================
# LOAD CT
# ============================================================

def load_ct(path):

    nii = nib.load(str(path))

    image = nii.get_fdata(dtype=np.float32)

    spacing = np.asarray(
        nii.header.get_zooms()[:3],
        dtype=np.float64
    )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"Non-finite CT values found: {path}"
        )

    if np.any(spacing <= 0):
        raise ValueError(
            f"Invalid voxel spacing: {spacing}"
        )

    return image, spacing


# ============================================================
# FIT FEATURE TO ORIGINAL SHAPE
# ============================================================

def fit_to_shape(volume, target_shape):

    slices = tuple(
        slice(0, min(volume.shape[i], target_shape[i]))
        for i in range(3)
    )

    volume = volume[slices]

    padding = tuple(
        (
            0,
            max(0, target_shape[i] - volume.shape[i])
        )
        for i in range(3)
    )

    if any(pad[1] > 0 for pad in padding):

        volume = np.pad(
            volume,
            padding,
            mode="edge"
        )

    return volume.astype(np.float32)


# ============================================================
# COMPUTE FRANGI
# Same parameters as original preprocessing script
# ============================================================

def compute_frangi(image, spacing):

    # Resample CT to approximately 1 mm isotropic spacing.

    factors = tuple(
        spacing[i] / TARGET_SPACING_MM
        for i in range(3)
    )

    isotropic = zoom(
        image,
        factors,
        order=1,
        mode="nearest",
        prefilter=False
    )

    isotropic = np.asarray(
        isotropic,
        dtype=np.float32
    )

    # Multiscale Frangi vessel enhancement.

    responses = []

    for sigma in PHYSICAL_SIGMAS_MM:

        print(
            f"    Frangi sigma: {sigma} mm",
            flush=True
        )

        response = frangi(
            isotropic,
            sigmas=(sigma,),
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=False,
            mode="reflect"
        )

        response = np.nan_to_num(
            response,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(np.float32)

        responses.append(response)

        del response

    # Maximum response across scales.

    vesselness = np.maximum.reduce(responses)

    del responses

    vesselness = np.clip(
        vesselness,
        0.0,
        None
    )

    # Resample feature back to original CT grid.

    reverse_factors = tuple(
        image.shape[i] / vesselness.shape[i]
        for i in range(3)
    )

    original_grid = zoom(
        vesselness,
        reverse_factors,
        order=1,
        mode="nearest",
        prefilter=False
    )

    del vesselness

    original_grid = fit_to_shape(
        original_grid,
        image.shape
    )

    # Per-volume robust normalization.

    positive = original_grid[
        original_grid > 0
    ]

    if positive.size > 0:

        scale = float(
            np.percentile(positive, 99.9)
        )

        if scale > 0:
            original_grid /= scale

    original_grid = np.clip(
        original_grid,
        0.0,
        1.0
    )

    return original_grid.astype(np.float32)


# ============================================================
# VERIFY EXISTING FEATURE
# ============================================================

def is_valid_feature(path, expected_shape):

    if not path.exists():
        return False

    try:

        existing = np.load(
            path,
            mmap_mode="r"
        )

        valid = (
            existing.shape == expected_shape
            and np.issubdtype(
                existing.dtype,
                np.floating
            )
            and np.all(np.isfinite(existing))
        )

        del existing

        return bool(valid)

    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        splits = json.load(f)

    train_ids = set(splits["train"])

    validation_ids = set(
        splits["validation"]
    )

    test_ids = list(
        splits["test"]
    )

    # Safety checks.

    if len(test_ids) != 15:

        raise ValueError(
            f"Expected 15 test patients, found {len(test_ids)}"
        )

    if set(test_ids) & (
        train_ids | validation_ids
    ):

        raise ValueError(
            "Test patients overlap train/validation"
        )

    print("=" * 65)
    print("NEUROVESSEL: TEST-ONLY FRANGI PRECOMPUTATION")
    print("=" * 65)

    print(f"Test patients: {len(test_ids)}")
    print(f"Target spacing: {TARGET_SPACING_MM} mm")
    print(f"Frangi sigmas: {PHYSICAL_SIGMAS_MM}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("Test labels will NOT be accessed.")
    print("=" * 65)

    processed = []
    skipped = []
    failed = []

    # --------------------------------------------------------
    # PROCESS TEST PATIENTS
    # --------------------------------------------------------

    for index, patient_id in enumerate(
        test_ids,
        start=1
    ):

        patient_start = time.time()

        print(
            f"\n[{index}/{len(test_ids)}] {patient_id}",
            flush=True
        )

        image_path = (
            DATA_ROOT
            / patient_id
            / "image"
            / f"{patient_id}.nii.gz"
        )

        output_path = (
            OUTPUT_DIR
            / f"{patient_id}.npy"
        )

        temporary_path = (
            OUTPUT_DIR
            / f"{patient_id}.tmp.npy"
        )

        try:

            # Read CT only. Never read the label.

            if not image_path.exists():

                raise FileNotFoundError(
                    image_path
                )

            image, spacing = load_ct(
                image_path
            )

            print(
                f"CT shape: {image.shape}"
            )

            print(
                f"Spacing: {spacing}"
            )

            # Skip an existing valid feature.

            if is_valid_feature(
                output_path,
                image.shape
            ):

                print(
                    "Valid existing feature found. Skipping."
                )

                skipped.append(patient_id)

                del image

                continue

            # Do not overwrite an invalid existing file.

            if output_path.exists():

                raise FileExistsError(
                    f"Invalid existing output: {output_path}. "
                    "Manual inspection required."
                )

            # Clean up a stale temporary file from an interrupted run.

            if temporary_path.exists():

                temporary_path.unlink()

            # Match original V2/V3/V4 CT preprocessing.

            image = np.clip(
                image,
                HU_MIN,
                HU_MAX
            )

            image = (
                (image - HU_MIN)
                / (HU_MAX - HU_MIN)
            ).astype(np.float32)

            # Compute Frangi features.

            print(
                "Computing multiscale Frangi features...",
                flush=True
            )

            feature = compute_frangi(
                image,
                spacing
            )

            if feature.shape != image.shape:

                raise ValueError(
                    f"Feature shape mismatch: "
                    f"{feature.shape} vs {image.shape}"
                )

            if not np.all(np.isfinite(feature)):

                raise ValueError(
                    "Feature contains non-finite values"
                )

            if (
                feature.min() < 0
                or feature.max() > 1
            ):

                raise ValueError(
                    "Feature outside expected [0,1] range"
                )

            # Save as float16 to a temporary file.

            print(
                "Saving feature...",
                flush=True
            )

            with open(
                temporary_path,
                "wb"
            ) as f:

                np.save(
                    f,
                    feature.astype(np.float16)
                )

            # IMPORTANT:
            # Load without mmap_mode so Windows does not
            # keep the temporary file locked.

            saved = np.load(
                temporary_path
            )

            if (
                saved.shape != image.shape
                or not np.all(np.isfinite(saved))
            ):

                del saved

                temporary_path.unlink(
                    missing_ok=True
                )

                raise ValueError(
                    "Saved feature verification failed"
                )

            feature_min = float(saved.min())
            feature_max = float(saved.max())

            saved_shape = saved.shape

            # Release file handle before rename.

            del saved

            # Never overwrite an existing output.

            if output_path.exists():

                temporary_path.unlink(
                    missing_ok=True
                )

                raise FileExistsError(
                    "Output appeared during processing. "
                    "Refusing to overwrite."
                )

            # Atomic rename.

            temporary_path.replace(
                output_path
            )

            print(
                f"Saved shape: {saved_shape}"
            )

            print(
                f"Feature range: "
                f"[{feature_min:.4f}, {feature_max:.4f}]"
            )

            print(
                f"Saved: {output_path}"
            )

            print(
                f"Patient time: "
                f"{time.time() - patient_start:.1f} seconds"
            )

            processed.append(patient_id)

            # Release large arrays.

            del image
            del feature

        except KeyboardInterrupt:

            print(
                "\nProcessing interrupted by user.",
                flush=True
            )

            raise

        except Exception as error:

            print(
                f"ERROR for {patient_id}: {error}",
                flush=True
            )

            failed.append({
                "patient_id": patient_id,
                "error": str(error)
            })

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    complete = []

    for patient_id in test_ids:

        image_path = (
            DATA_ROOT
            / patient_id
            / "image"
            / f"{patient_id}.nii.gz"
        )

        output_path = (
            OUTPUT_DIR
            / f"{patient_id}.npy"
        )

        try:

            nii = nib.load(
                str(image_path)
            )

            expected_shape = nii.shape[:3]

            if is_valid_feature(
                output_path,
                expected_shape
            ):

                complete.append(patient_id)

        except Exception:
            pass

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    summary = {

        "purpose": (
            "Test-only Frangi feature preparation"
        ),

        "test_patients": test_ids,

        "target_spacing_mm": TARGET_SPACING_MM,

        "physical_sigmas_mm": list(
            PHYSICAL_SIGMAS_MM
        ),

        "alpha": 0.5,

        "beta": 0.5,

        "gamma": None,

        "hu_clip": [
            HU_MIN,
            HU_MAX
        ],

        "normalization": (
            "Per-volume 99.9th percentile "
            "of positive responses"
        ),

        "storage_dtype": "float16",

        "labels_accessed": False,

        "processed": processed,

        "skipped": skipped,

        "failed": failed,

        "verified_complete_patients": complete,

        "test_features_complete": (
            len(complete) == len(test_ids)
        ),

        "elapsed_minutes": (
            time.time() - start_time
        ) / 60

    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TEST FEATURE PRECOMPUTATION SUMMARY")
    print("=" * 65)

    print(
        f"Processed: {len(processed)}"
    )

    print(
        f"Skipped: {len(skipped)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"Verified complete: {len(complete)}/{len(test_ids)}"
    )

    print(
        f"All features complete: "
        f"{summary['test_features_complete']}"
    )

    print(
        f"Elapsed time: "
        f"{summary['elapsed_minutes']:.2f} minutes"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()
