#!/usr/bin/env python3
"""Audit CT intensity values inside AirRC pulmonary-artery (label 3) masks.

Reads one case at a time, resamples the LUNA16 CT onto the AirRC annotation
physical grid with linear interpolation, and reports PA-region intensity
statistics and the amount of PA signal affected by candidate clipping windows.
Original CTs and annotations are never modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path.cwd()
MANIFEST = ROOT / "outputs" / "metrics" / "airrc_uid_manifest_with_luna_subset.csv"
OUT_DIR = ROOT / "outputs" / "metrics"
CSV_OUT = OUT_DIR / "airrc_pa_intensity_audit.csv"
JSON_OUT = OUT_DIR / "airrc_pa_intensity_audit_summary.json"
PA_LABEL = 3
WINDOWS = [(-1000, 500), (-1000, 700), (-1000, 1000), (-1000, 1500)]
PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, PERCENTILES)
    return {f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(PERCENTILES, q)}


def aggregate_group(frame: pd.DataFrame) -> dict:
    out = {"cases": int(len(frame)), "total_pa_voxels": int(frame["pa_voxels"].sum())}
    metric_cols = [
        "pa_p1", "pa_p5", "pa_p50", "pa_p95", "pa_p99", "pa_p99_5",
        "pa_p99_9", "pa_min", "pa_max", "frac_pa_below_minus1000",
        "frac_pa_above_500", "frac_pa_above_700", "frac_pa_above_1000",
        "frac_pa_above_1500", "frac_pa_above_3000",
    ]
    stats = {}
    for col in metric_cols:
        x = frame[col].dropna().to_numpy(dtype=float)
        if x.size:
            stats[col] = {
                "median_across_cases": float(np.median(x)),
                "p10_across_cases": float(np.percentile(x, 10)),
                "p90_across_cases": float(np.percentile(x, 90)),
                "min_across_cases": float(np.min(x)),
                "max_across_cases": float(np.max(x)),
            }
    out["per_case_statistics"] = stats
    total = int(frame["pa_voxels"].sum())
    for lo, hi in WINDOWS:
        below = int(frame[f"count_pa_below_{abs(lo)}"].sum())
        above = int(frame[f"count_pa_above_{hi}"].sum())
        out[f"window_{lo}_to_{hi}"] = {
            "pa_voxels_below_lower_bound": below,
            "pa_voxels_above_upper_bound": above,
            "fraction_pa_voxels_clipped": float((below + above) / total) if total else None,
        }
    return out


def audit_one(row) -> dict:
    uid = str(row.series_instance_uid)
    subset = str(row.luna_subset)
    ct_path = ROOT / "data" / "airrc_luna16" / subset / f"{uid}.mhd"
    mask_path = Path(str(row.annotation_path))
    if not ct_path.exists():
        raise FileNotFoundError(f"CT header missing: {ct_path}")
    if not (ct_path.parent / f"{uid}.raw").exists():
        raise FileNotFoundError(f"CT raw missing: {ct_path.parent / (uid + '.raw')}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Annotation missing: {mask_path}")

    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))

    # Align CT intensities to the exact AirRC annotation grid.
    identity = sitk.Transform(3, sitk.sitkIdentity)
    ct_on_mask = sitk.Resample(
        ct, mask, identity, sitk.sitkLinear, -2048.0, sitk.sitkFloat32
    )
    ct_arr = sitk.GetArrayViewFromImage(ct_on_mask)
    mask_arr = sitk.GetArrayViewFromImage(mask)
    pa_mask = mask_arr == PA_LABEL
    n_pa = int(np.count_nonzero(pa_mask))
    if n_pa == 0:
        raise ValueError(f"No label-{PA_LABEL} voxels in {uid}")

    vals = np.asarray(ct_arr[pa_mask], dtype=np.float32)
    qs = percentile_dict(vals)
    rec = {
        "series_instance_uid": uid,
        "patient_id": str(row.patient_id),
        "contrast_status": str(row.contrast_status),
        "luna_subset": subset,
        "pa_voxels": n_pa,
        "pa_min": float(np.min(vals)),
        "pa_max": float(np.max(vals)),
        "pa_mean": float(np.mean(vals, dtype=np.float64)),
        "pa_std": float(np.std(vals, dtype=np.float64)),
    }
    for key, value in qs.items():
        rec[f"pa_{key}"] = value

    thresholds = [-1000, -500, 0, 100, 300, 500, 700, 1000, 1500, 2000, 3000]
    for t in thresholds:
        rec[f"frac_pa_below_{str(t).replace('-', 'minus')}"] = float(np.count_nonzero(vals < t) / n_pa)
        rec[f"frac_pa_above_{t}"] = float(np.count_nonzero(vals > t) / n_pa)
    rec["frac_pa_below_minus1000"] = rec["frac_pa_below_minus1000"]
    for lo, hi in WINDOWS:
        rec[f"count_pa_below_{abs(lo)}"] = int(np.count_nonzero(vals < lo))
        rec[f"count_pa_above_{hi}"] = int(np.count_nonzero(vals > hi))

    # Check resampled image and annotation geometry are identical.
    same_grid = (
        ct_on_mask.GetSize() == mask.GetSize()
        and np.allclose(ct_on_mask.GetSpacing(), mask.GetSpacing(), atol=1e-5)
        and np.allclose(ct_on_mask.GetOrigin(), mask.GetOrigin(), atol=1e-4)
        and np.allclose(ct_on_mask.GetDirection(), mask.GetDirection(), atol=1e-5)
    )
    rec["resampled_ct_matches_mask_grid"] = bool(same_grid)
    if not same_grid:
        raise RuntimeError(f"Resampled CT does not match annotation grid: {uid}")

    del vals, pa_mask, ct_arr, mask_arr, ct_on_mask, mask, ct
    return rec


def main() -> int:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    needed = {"series_instance_uid", "patient_id", "contrast_status", "annotation_path", "luna_subset"}
    missing = needed - set(manifest.columns)
    if missing:
        print(f"Manifest missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    rows = []
    errors = []
    print(f"AirRC PA-region intensity audit: {len(manifest)} cases")
    print("CT is linearly resampled onto each annotation grid; only label-3 voxels are analyzed.")
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        try:
            rec = audit_one(row)
            rows.append(rec)
            print(
                f"[{i:03d}/{len(manifest)}] {rec['patient_id']} | {rec['contrast_status']} "
                f"| PA n={rec['pa_voxels']} | p1/p50/p99="
                f"{rec['pa_p1']:.1f}/{rec['pa_p50']:.1f}/{rec['pa_p99']:.1f}"
            )
        except Exception as exc:
            errors.append({"series_instance_uid": str(row.series_instance_uid),
                           "patient_id": str(row.patient_id), "error": repr(exc)})
            print(f"[{i:03d}/{len(manifest)}] {row.patient_id} | ERROR | {exc}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(CSV_OUT, index=False)
    summary = {
        "total_cases": int(len(manifest)),
        "pass": int(len(rows)),
        "error": int(len(errors)),
        "total_pa_voxels": int(frame["pa_voxels"].sum()) if not frame.empty else 0,
        "by_contrast_status": {},
        "candidate_windows": {f"{lo}_to_{hi}": {"description": "PA voxel clipping impact"} for lo, hi in WINDOWS},
        "errors": errors,
        "csv_report": str(CSV_OUT),
    }
    for status, group in frame.groupby("contrast_status") if not frame.empty else []:
        summary["by_contrast_status"][str(status)] = aggregate_group(group)
    # Also report all cases pooled together, weighted by PA voxel count for clipping fractions.
    if not frame.empty:
        summary["all_cases"] = aggregate_group(frame)
    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PA INTENSITY AUDIT SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_contrast_status", "all_cases", "errors")}, indent=2))
    print("\n=== BY CONTRAST STATUS ===")
    print(json.dumps(summary["by_contrast_status"], indent=2))
    if "all_cases" in summary:
        print("\n=== ALL CASES / CANDIDATE WINDOW CLIPPING ===")
        print(json.dumps({k: v for k, v in summary["all_cases"].items() if k.startswith("window_")}, indent=2))
    print(f"\nPer-case CSV: {CSV_OUT}")
    print(f"Summary JSON: {JSON_OUT}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
