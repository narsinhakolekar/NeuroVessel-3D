
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
METADATA_FILE = OUTPUT_DIR / "metadata.json"

TARGET_SPACING_MM = 1.0
PHYSICAL_SIGMAS_MM = (1.0, 2.0, 3.0)

MAX_PATIENTS = None       # First run: test only one patient
FORCE_RECOMPUTE = False

HU_MIN = -1000
HU_MAX = 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_nifti(path):
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=np.float64)

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values found in {path}")

    return nii, data, spacing


def fit_to_shape(volume, target_shape):
    """Crop or edge-pad a volume to exactly match target_shape."""

    slices = tuple(
        slice(0, min(volume.shape[i], target_shape[i]))
        for i in range(3)
    )
    volume = volume[slices]

    padding = tuple(
        (0, max(0, target_shape[i] - volume.shape[i]))
        for i in range(3)
    )

    if any(pad[1] > 0 for pad in padding):
        volume = np.pad(volume, padding, mode="edge")

    return volume.astype(np.float32)


def compute_frangi(ct):
    """Compute multiscale 3D Frangi vesselness on a 1 mm grid."""

    spacing = ct["spacing"]
    image = ct["image"]

    # Resample CT to approximately isotropic 1 mm spacing.
    factors = tuple(spacing[i] / TARGET_SPACING_MM for i in range(3))

    isotropic = zoom(
        image,
        factors,
        order=1,
        mode="nearest",
        prefilter=False
    )

    isotropic = np.asarray(isotropic, dtype=np.float32)

    # Multiscale Frangi vessel enhancement.
    responses = []

    for sigma in PHYSICAL_SIGMAS_MM:
        response = frangi(
            isotropic,
            sigmas=(sigma,),
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=False,
            mode="reflect"
        )

        responses.append(
            np.nan_to_num(
                response,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ).astype(np.float32)
        )

    vesselness = np.maximum.reduce(responses)
    vesselness = np.clip(vesselness, 0.0, None)

    # Resample the feature back to the original CT grid.
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

    original_grid = fit_to_shape(original_grid, image.shape)

    # Consistent per-volume robust normalization.
    positive = original_grid[original_grid > 0]

    if positive.size > 0:
        scale = float(np.percentile(positive, 99.9))
        if scale > 0:
            original_grid = original_grid / scale

    original_grid = np.clip(original_grid, 0.0, 1.0)

    return original_grid.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Never process test patients during feature preparation.
    patient_ids = list(splits["train"]) + list(splits["validation"])

    if MAX_PATIENTS is not None:
        patient_ids = patient_ids[:MAX_PATIENTS]

    print("=" * 65)
    print("NEUROVESSEL: FRANGI FEATURE PRECOMPUTATION")
    print("=" * 65)
    print(f"Patients selected: {len(patient_ids)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target spacing: {TARGET_SPACING_MM} mm")
    print(f"Frangi sigmas: {PHYSICAL_SIGMAS_MM} mm")
    print()

    processed = []
    skipped = []
    failed = []

    for index, patient_id in enumerate(patient_ids, start=1):

        patient_start = time.time()

        image_path = (
            DATA_ROOT / patient_id / "image" / f"{patient_id}.nii.gz"
        )
        label_path = (
            DATA_ROOT / patient_id / "label" / f"{patient_id}.nii.gz"
        )

        output_path = OUTPUT_DIR / f"{patient_id}.npy"

        print(f"\n[{index}/{len(patient_ids)}] Patient: {patient_id}")

        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            if not label_path.exists():
                raise FileNotFoundError(label_path)

            # Load CT and label only for geometry validation.
            image_nii, image, spacing = load_nifti(image_path)
            label_nii, label, _ = load_nifti(label_path)

            if image.shape != label.shape:
                raise ValueError(
                    f"CT/label shape mismatch: {image.shape} vs {label.shape}"
                )

            if not np.allclose(
                image_nii.affine,
                label_nii.affine,
                atol=1e-4
            ):
                raise ValueError("CT and label affine matrices differ")

            if np.any(spacing <= 0):
                raise ValueError(f"Invalid voxel spacing: {spacing}")

            print(f"Original CT shape: {image.shape}")
            print(f"Original spacing: {spacing}")

            if output_path.exists() and not FORCE_RECOMPUTE:
                existing = np.load(output_path, mmap_mode="r")

                if existing.shape == image.shape:
                    print("Valid cached feature found. Skipping.")
                    skipped.append(patient_id)
                    continue

                print("Cached shape mismatch. Recomputing.")

            # Match the intensity preprocessing used by UNet V2.
            image = np.clip(image, HU_MIN, HU_MAX)
            image = (image - HU_MIN) / (HU_MAX - HU_MIN)
            image = image.astype(np.float32)

            ct_data = {
                "image": image,
                "spacing": spacing
            }

            print("Computing multiscale Frangi features...")
            feature = compute_frangi(ct_data)

            if feature.shape != image.shape:
                raise ValueError(
                    f"Feature shape mismatch: {feature.shape} vs {image.shape}"
                )

            np.save(output_path, feature.astype(np.float16))

            # Verify the saved file.
            saved = np.load(output_path, mmap_mode="r")

            if saved.shape != image.shape:
                raise ValueError("Saved feature verification failed")

            print(f"Feature shape: {saved.shape}")
            print(f"Feature range: [{saved.min():.4f}, {saved.max():.4f}]")
            print(f"Saved: {output_path}")
            print(f"Patient time: {time.time() - patient_start:.1f} seconds")

            processed.append(patient_id)

        except Exception as error:
            print(f"ERROR for {patient_id}: {error}")
            failed.append({
                "patient_id": patient_id,
                "error": str(error)
            })

    metadata = {
        "target_spacing_mm": TARGET_SPACING_MM,
        "physical_sigmas_mm": list(PHYSICAL_SIGMAS_MM),
        "normalization": "per-volume 99.9th percentile of positive responses",
        "storage_dtype": "float16",
        "processed_patients": processed,
        "skipped_patients": skipped,
        "failed_patients": failed,
        "test_patients_processed": False
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 65)
    print("PRECOMPUTATION SUMMARY")
    print("=" * 65)
    print(f"Processed: {len(processed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 65)


if __name__ == "__main__":
    main()
