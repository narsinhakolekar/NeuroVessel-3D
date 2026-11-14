#!/usr/bin/env python3
"""Memory-conscious CT intensity audit for AirRC/LUNA16.

Run from the NeuroVessel project root:
    python audit_airrc_intensity.py

Reads each MetaImage (.mhd/.raw) sequentially, computes exact range and
threshold fractions over all voxels, and estimates percentiles from a
reproducible random sample (default 300k voxels per scan).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path.cwd()
MANIFEST = ROOT / "outputs" / "metrics" / "airrc_uid_manifest_with_luna_subset.csv"
OUT_DIR = ROOT / "outputs" / "metrics"
CSV_OUT = OUT_DIR / "airrc_intensity_audit.csv"
JSON_OUT = OUT_DIR / "airrc_intensity_audit_summary.json"
MAX_SAMPLE = 300_000
PERCENTILES = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
THRESHOLDS = [-1000, -900, -700, 0, 300, 500, 1000, 2000, 3000]


def stable_seed(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:4], "little")


def pct_summary(frame: pd.DataFrame, column: str) -> dict:
    vals = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return {"median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def audit_one(row, index: int) -> dict:
    uid = str(row.series_instance_uid)
    subset = str(row.luna_subset)
    patient = str(row.patient_id)
    status = str(row.contrast_status)
    mhd = ROOT / "data" / "airrc_luna16" / subset / f"{uid}.mhd"
    raw = mhd.with_suffix(".raw")
    base = {
        "index": index,
        "patient_id": patient,
        "series_instance_uid": uid,
        "contrast_status": status,
        "luna_subset": subset,
        "mhd_path": str(mhd),
        "status": "ERROR",
    }
    if not mhd.is_file() or not raw.is_file():
        base["error"] = f"Missing image file(s): mhd={mhd.is_file()}, raw={raw.is_file()}"
        return base

    image = sitk.ReadImage(str(mhd))
    arr = sitk.GetArrayViewFromImage(image)  # Z,Y,X view; no full-volume copy
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        base["error"] = "Empty CT volume"
        return base

    sample_n = min(n, MAX_SAMPLE)
    if sample_n == n:
        sample = np.asarray(flat)
    else:
        rng = np.random.default_rng(stable_seed(uid))
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        sample = np.asarray(flat[sample_idx])

    q = np.percentile(sample, PERCENTILES)
    result = {
        **base,
        "status": "PASS",
        "size_x": int(image.GetSize()[0]),
        "size_y": int(image.GetSize()[1]),
        "size_z": int(image.GetSize()[2]),
        "spacing_x_mm": float(image.GetSpacing()[0]),
        "spacing_y_mm": float(image.GetSpacing()[1]),
        "spacing_z_mm": float(image.GetSpacing()[2]),
        "origin_lps_x": float(image.GetOrigin()[0]),
        "origin_lps_y": float(image.GetOrigin()[1]),
        "origin_lps_z": float(image.GetOrigin()[2]),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "voxel_count": n,
        "sample_count": int(sample_n),
        "min_exact": int(np.min(flat)),
        "max_exact": int(np.max(flat)),
        "sample_mean": float(np.mean(sample, dtype=np.float64)),
        "sample_std": float(np.std(sample, dtype=np.float64)),
    }
    for p, value in zip(PERCENTILES, q):
        result[f"sample_p{str(p).replace('.', '_')}"] = float(value)
    for t in THRESHOLDS:
        result[f"pct_le_{t}" if t < 0 else f"pct_ge_{t}"] = float(
            (np.count_nonzero(flat <= t) if t < 0 else np.count_nonzero(flat >= t)) / n * 100.0
        )
    # A useful indicator of whether the scan values resemble HU-like CT data.
    result["pct_outside_int16_ct_range"] = float(
        (np.count_nonzero((flat < -1200) | (flat > 3500))) / n * 100.0
    )
    return result


def main() -> int:
    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    required = {"series_instance_uid", "patient_id", "contrast_status", "luna_subset"}
    missing = required - set(manifest.columns)
    if missing:
        print("ERROR: manifest is missing columns:", sorted(missing))
        return 2

    print(f"AirRC CT intensity audit: {len(manifest)} scans")
    print(f"Sampling up to {MAX_SAMPLE:,} voxels per scan for percentiles; threshold fractions and min/max are exact.")
    rows = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        out = audit_one(row, i)
        rows.append(out)
        if out["status"] == "PASS":
            print(
                f"[{i:03d}/{len(manifest)}] {out['patient_id']} | {out['contrast_status']} | "
                f"range=[{out['min_exact']}, {out['max_exact']}] | "
                f"p1/p50/p99={out['sample_p1']:.1f}/{out['sample_p50']:.1f}/{out['sample_p99']:.1f}"
            )
        else:
            print(f"[{i:03d}/{len(manifest)}] {out['patient_id']} | ERROR | {out.get('error', 'unknown error')}")
        # Checkpoint after every scan so partial results survive interruption.
        pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    frame = pd.DataFrame(rows)
    passed = frame[frame["status"] == "PASS"].copy()
    summary = {
        "total_cases": int(len(frame)),
        "pass": int((frame["status"] == "PASS").sum()),
        "error": int((frame["status"] != "PASS").sum()),
        "percentiles_method": f"Reproducible random sample, up to {MAX_SAMPLE} voxels per scan",
        "threshold_fractions_and_min_max": "Exact over all voxels",
        "overall": {},
        "by_contrast_status": {},
        "csv_report": str(CSV_OUT.relative_to(ROOT)),
    }
    columns = ["sample_p0_5", "sample_p1", "sample_p5", "sample_p25", "sample_p50", "sample_p75", "sample_p95", "sample_p99", "sample_p99_5", "sample_mean", "sample_std"]
    for col in columns:
        summary["overall"][col] = pct_summary(passed, col)
    for status, group in passed.groupby("contrast_status"):
        summary["by_contrast_status"][str(status)] = {
            "cases": int(len(group)),
            "stats": {col: pct_summary(group, col) for col in columns},
        }
    for col in ["pct_le_-1000", "pct_le_-900", "pct_le_-700", "pct_ge_300", "pct_ge_500", "pct_ge_1000", "pct_ge_2000", "pct_ge_3000", "pct_outside_int16_ct_range"]:
        summary["overall"][col] = pct_summary(passed, col)
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== INTENSITY AUDIT SUMMARY ===")
    print(json.dumps({k: summary[k] for k in ["total_cases", "pass", "error", "by_contrast_status"]}, indent=2))
    print(f"Per-scan CSV: {CSV_OUT}")
    print(f"Summary JSON: {JSON_OUT}")
    return 0 if summary["error"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed per-case results remain saved in the CSV.")
        raise SystemExit(130)
