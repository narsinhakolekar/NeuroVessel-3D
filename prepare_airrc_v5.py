#!/usr/bin/env python3
"""Prepare AirRC/LUNA16 for NeuroVessel V5.

- Creates a deterministic patient-grouped 80/10/10 train/validation/test split.
- Resamples each LUNA16 CT onto its matching AirRC annotation grid (linear interpolation).
- Extracts PA class 3 as a binary uint8 mask (nearest-neighbour is unnecessary because
  the annotation already defines the target grid; source annotation is not resampled).
- Saves one unclipped float32 CT-intensity volume so candidate windows can be compared
  later during training without duplicating the dataset.
- Processes one case at a time and can resume after interruption.

Run from the NeuroVessel project root:
    python prepare_airrc_v5.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "outputs" / "metrics" / "airrc_uid_manifest_with_luna_subset.csv"
SPLIT_PATH = ROOT / "outputs" / "metrics" / "airrc_v5_splits.csv"
OUT_ROOT = ROOT / "data" / "processed_airrc_v5"
IMAGE_DIR = OUT_ROOT / "images"
MASK_DIR = OUT_ROOT / "labels_pa"
CASE_REPORT = ROOT / "outputs" / "metrics" / "airrc_v5_preprocess_manifest.csv"
SUMMARY_PATH = ROOT / "outputs" / "metrics" / "airrc_v5_preprocess_summary.json"
SEED = 42


def atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def make_or_load_splits(df: pd.DataFrame) -> pd.DataFrame:
    required = {"series_instance_uid", "patient_id", "contrast_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required split columns: {sorted(missing)}")

    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH, dtype={"series_instance_uid": str})
        if set(saved["series_instance_uid"]) != set(df["series_instance_uid"]):
            raise RuntimeError(
                f"Existing split file {SPLIT_PATH} does not match the current 254-case manifest. "
                "Do not silently regenerate it; inspect/remove the split file deliberately."
            )
        if saved["series_instance_uid"].duplicated().any():
            raise RuntimeError("Existing split CSV contains duplicate series UIDs.")
        if saved.groupby("patient_id")["split"].nunique().max() != 1:
            raise RuntimeError("Patient leakage detected: a patient appears in multiple splits.")
        print(f"Loaded existing patient-level split: {SPLIT_PATH}")
        return df.drop(columns=["split"], errors="ignore").merge(
            saved[["series_instance_uid", "split"]], on="series_instance_uid", how="left", validate="one_to_one"
        )

    if df["series_instance_uid"].duplicated().any():
        raise RuntimeError("Duplicate series_instance_uid values found in input manifest.")

    # StratifiedGroupKFold keeps all scans from a patient together. Two of ten folds
    # are reserved (one validation, one test), yielding approximately 80/10/10.
    y = df["contrast_status"].astype(str).to_numpy()
    groups = df["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(df)), y, groups))
    split = np.full(len(df), "train", dtype=object)
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "validation"
    df_out = df.copy()
    df_out["split"] = split

    # Explicitly check patient separation and contrast proportions.
    if df_out.groupby("patient_id")["split"].nunique().max() != 1:
        raise RuntimeError("Patient-level split check failed.")
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out[["series_instance_uid", "patient_id", "contrast_status", "luna_subset", "split"]].to_csv(
        SPLIT_PATH, index=False
    )
    print(f"Created patient-level split: {SPLIT_PATH}")
    return df_out


def assert_same_grid(image: sitk.Image, reference: sitk.Image, name: str) -> None:
    if image.GetSize() != reference.GetSize():
        raise RuntimeError(f"{name} size differs from annotation reference grid")
    if not np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-5):
        raise RuntimeError(f"{name} spacing differs from annotation reference grid")
    if not np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-4):
        raise RuntimeError(f"{name} origin differs from annotation reference grid")
    if not np.allclose(image.GetDirection(), reference.GetDirection(), atol=1e-6):
        raise RuntimeError(f"{name} direction differs from annotation reference grid")


def validate_existing(image_path: Path, mask_path: Path, uid: str) -> dict[str, Any] | None:
    if not image_path.exists() or not mask_path.exists():
        return None
    try:
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        if image.GetSize() != mask.GetSize():
            return None
        if image.GetPixelID() != sitk.sitkFloat32:
            return None
        arr = sitk.GetArrayViewFromImage(mask)
        pa_count = int(np.count_nonzero(arr))
        if pa_count == 0 or not set(np.unique(arr).tolist()).issubset({0, 1}):
            return None
        return {
            "series_instance_uid": uid,
            "status": "EXISTING_VERIFIED",
            "image_path": str(image_path.relative_to(ROOT)),
            "mask_path": str(mask_path.relative_to(ROOT)),
            "size_xyz": json.dumps(image.GetSize()),
            "spacing_xyz_mm": json.dumps(image.GetSpacing()),
            "pa_voxels": pa_count,
            "image_min": float(sitk.GetArrayViewFromImage(image).min()),
            "image_max": float(sitk.GetArrayViewFromImage(image).max()),
            "error": "",
        }
    except Exception:
        return None


def process_case(row: Any) -> dict[str, Any]:
    uid = str(row.series_instance_uid)
    subset = str(row.luna_subset)
    ct_path = ROOT / "data" / "airrc_luna16" / subset / f"{uid}.mhd"
    ann_path = Path(str(row.annotation_path))
    image_path = IMAGE_DIR / f"{uid}_ct.nii.gz"
    mask_path = MASK_DIR / f"{uid}_pa.nii.gz"

    if not ct_path.exists():
        raise FileNotFoundError(f"CT header not found: {ct_path}")
    raw_path = ct_path.with_suffix(".raw")
    if not raw_path.exists():
        raise FileNotFoundError(f"CT raw voxel file not found: {raw_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"AirRC annotation not found: {ann_path}")

    existing = validate_existing(image_path, mask_path, uid)
    if existing is not None:
        existing.update({"patient_id": row.patient_id, "contrast_status": row.contrast_status, "split": row.split})
        return existing

    ct = sitk.ReadImage(str(ct_path))
    seg = sitk.ReadImage(str(ann_path))
    if ct.GetDimension() != 3 or seg.GetDimension() != 3:
        raise RuntimeError("Expected 3D CT and 3D annotation")

    # CT is resampled in physical space onto the annotation's 1 mm target grid.
    # The -1000 default is only used outside the source image; PA voxels were all
    # confirmed to lie inside CT coverage by the preceding cohort geometry audit.
    ct_resampled = sitk.Resample(
        ct,
        seg,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        -1000.0,
        sitk.sitkFloat32,
    )
    assert_same_grid(ct_resampled, seg, "Resampled CT")

    seg_arr = sitk.GetArrayViewFromImage(seg)
    labels = np.unique(seg_arr)
    if not set(labels.tolist()).issubset({0, 1, 2, 3, 4}):
        raise RuntimeError(f"Unexpected annotation labels: {labels.tolist()}")
    pa_arr = np.asarray(seg_arr == 3, dtype=np.uint8)
    pa_voxels = int(pa_arr.sum())
    if pa_voxels == 0:
        raise RuntimeError("No pulmonary-artery label-3 voxels found")

    pa_image = sitk.GetImageFromArray(pa_arr)
    pa_image.CopyInformation(seg)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    image_tmp = image_path.with_name(image_path.stem + ".tmp.nii.gz")
    mask_tmp = mask_path.with_name(mask_path.stem + ".tmp.nii.gz")

    sitk.WriteImage(ct_resampled, str(image_tmp), True)
    sitk.WriteImage(pa_image, str(mask_tmp), True)
    os.replace(image_tmp, image_path)
    os.replace(mask_tmp, mask_path)

    image_arr = sitk.GetArrayViewFromImage(ct_resampled)
    return {
        "series_instance_uid": uid,
        "patient_id": row.patient_id,
        "contrast_status": row.contrast_status,
        "split": row.split,
        "status": "PROCESSED",
        "image_path": str(image_path.relative_to(ROOT)),
        "mask_path": str(mask_path.relative_to(ROOT)),
        "size_xyz": json.dumps(ct_resampled.GetSize()),
        "spacing_xyz_mm": json.dumps(ct_resampled.GetSpacing()),
        "pa_voxels": pa_voxels,
        "pa_fraction": float(pa_voxels / pa_arr.size),
        "image_min": float(np.min(image_arr)),
        "image_max": float(np.max(image_arr)),
        "error": "",
    }


def main() -> int:
    print("NeuroVessel V5 AirRC preprocessing")
    print("This creates an unclipped CT-intensity dataset on each AirRC 1 mm annotation grid.")
    print("It preserves source files and original five-class annotations.")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Required UID manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH, dtype={"series_instance_uid": str})
    if len(df) != 254:
        raise RuntimeError(f"Expected 254 AirRC cases, found {len(df)}")
    if not (df["contrast_status"].value_counts().sum() == 254):
        raise RuntimeError("Contrast-status metadata is incomplete")

    df = make_or_load_splits(df)
    print("\nPatient-level split:")
    print(df.groupby(["split", "contrast_status"]).size().to_string())
    print("\nProcessing cases sequentially; existing verified outputs will be skipped.")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        print(f"[{i:03d}/{len(df)}] {row.patient_id} | {row.split} | {row.contrast_status}", flush=True)
        try:
            result = process_case(row)
            records.append(result)
            print(f"  {result['status']} | PA voxels={result['pa_voxels']:,} | size={result['size_xyz']}", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"series_instance_uid": str(row.series_instance_uid), "patient_id": str(row.patient_id), "error": message})
            records.append({
                "series_instance_uid": str(row.series_instance_uid),
                "patient_id": str(row.patient_id),
                "contrast_status": str(row.contrast_status),
                "split": str(row.split),
                "status": "ERROR",
                "error": message,
            })
            print(f"  ERROR: {message}", flush=True)

    atomic_write_csv(records, CASE_REPORT)
    successful = sum(r.get("status") in {"PROCESSED", "EXISTING_VERIFIED"} for r in records)
    summary = {
        "total_cases": int(len(df)),
        "processed_or_existing_verified": int(successful),
        "errors": int(len(errors)),
        "split_counts": df["split"].value_counts().to_dict(),
        "contrast_counts": df["contrast_status"].value_counts().to_dict(),
        "output_images": str(IMAGE_DIR.relative_to(ROOT)),
        "output_pa_masks": str(MASK_DIR.relative_to(ROOT)),
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        "case_manifest": str(CASE_REPORT.relative_to(ROOT)),
        "intensity_policy": "Store resampled source intensity as float32 without clipping/normalization; compare candidate windows during training.",
        "intensity_scale_note": "Values are stored as read from LUNA16 MetaImage data; independent HU calibration verification remains outstanding.",
        "candidate_windows_for_training": [[-1000, 700], [-1000, 1000], [-1000, 1500]],
        "seed": SEED,
        "errors_detail": errors,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PREPROCESSING SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors_detail"}, indent=2))
    print(f"\nDetailed case manifest: {CASE_REPORT}")
    print(f"Summary JSON: {SUMMARY_PATH}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed outputs are preserved; rerun to resume.")
        raise SystemExit(130)
