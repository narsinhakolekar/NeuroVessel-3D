#!/usr/bin/env python3
"""Integrity audit for NeuroVessel V5 preprocessed AirRC volumes.

Run from the NeuroVessel project root:
    python verify_airrc_v5_dataset.py

Reads one CT/mask pair at a time, checks file completeness, grid equality,
finite image values, binary/non-empty PA masks, manifest consistency and
patient-level train/validation/test separation. Does not modify source or
processed data.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "outputs/metrics/airrc_v5_splits.csv"
PREP = ROOT / "outputs/metrics/airrc_v5_preprocess_manifest.csv"
IMG_DIR = ROOT / "data/processed_airrc_v5/images"
MSK_DIR = ROOT / "data/processed_airrc_v5/labels_pa"
OUT_CSV = ROOT / "outputs/metrics/airrc_v5_dataset_integrity.csv"
OUT_JSON = ROOT / "outputs/metrics/airrc_v5_dataset_integrity_summary.json"
EXPECTED = 254


def check_grid(a: sitk.Image, b: sitk.Image) -> list[str]:
    problems = []
    if a.GetDimension() != 3 or b.GetDimension() != 3:
        problems.append("not_3d")
    if a.GetSize() != b.GetSize():
        problems.append("size_mismatch")
    if not np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5):
        problems.append("spacing_mismatch")
    if not np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4):
        problems.append("origin_mismatch")
    if not np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6):
        problems.append("direction_mismatch")
    return problems


def main() -> int:
    for p in (SPLITS, PREP):
        if not p.exists():
            raise FileNotFoundError(f"Required report not found: {p}")
    split = pd.read_csv(SPLITS, dtype={"series_instance_uid": str})
    prep = pd.read_csv(PREP, dtype={"series_instance_uid": str})
    required = {"series_instance_uid", "patient_id", "split", "contrast_status"}
    if not required.issubset(split.columns):
        raise RuntimeError(f"Split manifest lacks columns: {sorted(required - set(split.columns))}")
    if not {"series_instance_uid", "status", "pa_voxels"}.issubset(prep.columns):
        raise RuntimeError("Preprocessing manifest lacks required columns")
    if len(split) != EXPECTED or len(prep) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} rows in both manifests; got split={len(split)}, prep={len(prep)}")
    if split.series_instance_uid.duplicated().any() or prep.series_instance_uid.duplicated().any():
        raise RuntimeError("Duplicate series UID in manifests")
    if set(split.series_instance_uid) != set(prep.series_instance_uid):
        raise RuntimeError("Split and preprocessing manifests have different UID sets")
    if split.groupby("patient_id").split.nunique().max() != 1:
        raise RuntimeError("Patient leakage detected across splits")

    merged = split.merge(prep[["series_instance_uid", "status", "pa_voxels"]], on="series_instance_uid", validate="one_to_one")
    rows = []
    print(f"V5 dataset integrity audit: {len(merged)} cases", flush=True)
    for i, r in enumerate(merged.itertuples(index=False), 1):
        uid = str(r.series_instance_uid)
        ip = IMG_DIR / f"{uid}_ct.nii.gz"
        mp = MSK_DIR / f"{uid}_pa.nii.gz"
        issues = []
        rec = {"series_instance_uid": uid, "patient_id": str(r.patient_id), "split": str(r.split),
               "contrast_status": str(r.contrast_status), "preprocess_status": str(r.status),
               "image_exists": ip.exists(), "mask_exists": mp.exists()}
        if not ip.exists(): issues.append("missing_image")
        if not mp.exists(): issues.append("missing_mask")
        if issues:
            rec.update(status="FAIL", issues=";".join(issues)); rows.append(rec)
            print(f"[{i:03d}/{len(merged)}] {r.patient_id} | FAIL | {rec['issues']}", flush=True)
            continue
        try:
            image = sitk.ReadImage(str(ip))
            mask = sitk.ReadImage(str(mp))
            issues.extend(check_grid(image, mask))
            if image.GetPixelID() != sitk.sitkFloat32:
                issues.append("image_not_float32")
            ia = sitk.GetArrayViewFromImage(image)
            ma = sitk.GetArrayViewFromImage(mask)
            if not np.isfinite(ia).all():
                issues.append("nonfinite_image_values")
            labels = np.unique(ma)
            if not set(labels.tolist()).issubset({0, 1}):
                issues.append("mask_not_binary")
            pa_count = int(np.count_nonzero(ma))
            if pa_count == 0:
                issues.append("empty_pa_mask")
            if pa_count != int(r.pa_voxels):
                issues.append("pa_voxel_count_mismatch")
            rec.update(size_xyz=json.dumps(image.GetSize()), spacing_xyz_mm=json.dumps(image.GetSpacing()),
                       image_min=float(np.min(ia)), image_max=float(np.max(ia)),
                       pa_voxels=pa_count, pa_fraction=float(pa_count / ma.size),
                       mask_labels=json.dumps(labels.tolist()))
        except Exception as e:
            issues.append(f"read_error:{type(e).__name__}:{e}")
        rec["status"] = "PASS" if not issues else "FAIL"
        rec["issues"] = ";".join(issues)
        rows.append(rec)
        print(f"[{i:03d}/{len(merged)}] {r.patient_id} | {rec['status']} | split={r.split} | PA={rec.get('pa_voxels','?'):,}" if "pa_voxels" in rec else f"[{i:03d}/{len(merged)}] {r.patient_id} | FAIL | {rec['issues']}", flush=True)

    report = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_CSV, index=False)
    summary = {
        "total_cases": int(len(report)),
        "pass": int((report.status == "PASS").sum()),
        "fail": int((report.status == "FAIL").sum()),
        "split_counts": {str(k): int(v) for k, v in split.split.value_counts().items()},
        "contrast_counts": {str(k): int(v) for k, v in split.contrast_status.value_counts().items()},
        "patient_leakage": bool(split.groupby("patient_id").split.nunique().max() > 1),
        "total_pa_voxels": int(report.loc[report.status == "PASS", "pa_voxels"].sum()) if "pa_voxels" in report else 0,
        "csv_report": str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== DATASET INTEGRITY SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV: {OUT_CSV}\nJSON: {OUT_JSON}")
    return 0 if summary["pass"] == EXPECTED and summary["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
