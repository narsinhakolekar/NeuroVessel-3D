from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path.cwd()
MANIFEST = ROOT / "outputs" / "metrics" / "airrc_uid_manifest_with_luna_subset.csv"
OUT_DIR = ROOT / "outputs" / "metrics"
OUT_CSV = OUT_DIR / "airrc_pa_core_intensity.csv"
OUT_JSON = OUT_DIR / "airrc_pa_core_intensity_summary.json"

# These are descriptive thresholds, not a selected preprocessing window.
THRESHOLDS = (-500, -200, 0, 100, 300, 500, 700, 1000)


def stats(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{k}": None for k in ("n", "p1", "p5", "p50", "p95", "p99", "mean", "min", "max")}
    p = np.percentile(values, [1, 5, 50, 95, 99])
    out = {
        f"{prefix}_n": int(values.size),
        f"{prefix}_p1": float(p[0]),
        f"{prefix}_p5": float(p[1]),
        f"{prefix}_p50": float(p[2]),
        f"{prefix}_p95": float(p[3]),
        f"{prefix}_p99": float(p[4]),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }
    for t in THRESHOLDS:
        out[f"{prefix}_frac_below_{t}"] = float(np.mean(values < t))
        out[f"{prefix}_frac_above_{t}"] = float(np.mean(values > t))
    return out


def main() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 254:
        print(f"WARNING: expected 254 manifest rows, found {len(rows)}")

    results: list[dict] = []
    errors: list[dict] = []
    structure = ndi.generate_binary_structure(3, 1)  # 6-connected erosion, conservative core

    for i, row in enumerate(rows, 1):
        uid = row["series_instance_uid"]
        patient = row["patient_id"]
        subset = row["luna_subset"]
        contrast = row["contrast_status"]
        ct_path = ROOT / "data" / "airrc_luna16" / subset / f"{uid}.mhd"
        mask_path = Path(row["annotation_path"])

        try:
            if not ct_path.exists():
                raise FileNotFoundError(f"CT header missing: {ct_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask missing: {mask_path}")

            ct = sitk.ReadImage(str(ct_path))
            mask_img = sitk.ReadImage(str(mask_path))
            # Resample CT in physical space onto the exact annotation grid.
            ct_on_mask = sitk.Resample(
                ct,
                mask_img,
                sitk.Transform(3, sitk.sitkIdentity),
                sitk.sitkLinear,
                -1000.0,
                sitk.sitkFloat32,
            )
            image = sitk.GetArrayFromImage(ct_on_mask)
            labels = sitk.GetArrayViewFromImage(mask_img)
            pa = labels == 3
            pa_count = int(pa.sum())
            if pa_count == 0:
                raise ValueError("No class-3 PA voxels")

            # Erode by one 6-connected voxel to focus on mask interiors and reduce
            # boundary/partial-volume effects. Thin peripheral branches may vanish;
            # their contribution is retained in the full-mask and shell statistics.
            core = ndi.binary_erosion(pa, structure=structure, iterations=1, border_value=0)
            shell = pa & ~core
            core_count = int(core.sum())
            row_out = {
                "patient_id": patient,
                "series_instance_uid": uid,
                "luna_subset": subset,
                "contrast_status": contrast,
                "pa_voxels": pa_count,
                "core_voxels_after_1voxel_erosion": core_count,
                "core_fraction_of_pa": float(core_count / pa_count),
                "ct_size_xyz": str(ct.GetSize()),
                "mask_size_xyz": str(mask_img.GetSize()),
                "ct_spacing_xyz": str(ct.GetSpacing()),
                "mask_spacing_xyz": str(mask_img.GetSpacing()),
            }
            row_out.update(stats(image[pa], "pa_all"))
            row_out.update(stats(image[core], "pa_core"))
            row_out.update(stats(image[shell], "pa_shell"))
            results.append(row_out)
            print(
                f"[{i:03d}/{len(rows)}] {patient} | {contrast} | "
                f"PA={pa_count:,} core={core_count:,} | "
                f"p50 all/core/shell="
                f"{row_out['pa_all_p50']:.1f}/"
                f"{row_out['pa_core_p50'] if row_out['pa_core_p50'] is not None else float('nan'):.1f}/"
                f"{row_out['pa_shell_p50'] if row_out['pa_shell_p50'] is not None else float('nan'):.1f}"
            )
        except Exception as e:
            errors.append({"patient_id": patient, "series_instance_uid": uid, "error": repr(e)})
            print(f"[{i:03d}/{len(rows)}] {patient} | ERROR | {e}")

    if results:
        fieldnames = list(results[0].keys())
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    summary: dict = {
        "total_cases": len(rows),
        "processed": len(results),
        "errors": len(errors),
        "erosion_definition": "one iteration of 6-connected 3D binary erosion on PA label 3",
        "intensity_note": "Values are reported as read from the LUNA16 MetaImage data after linear resampling; HU calibration has not been independently verified by this script.",
        "csv_report": str(OUT_CSV),
        "errors_detail": errors,
        "groups": {},
    }
    for group in sorted({r["contrast_status"] for r in results}):
        group_rows = [r for r in results if r["contrast_status"] == group]
        summary["groups"][group] = {"cases": len(group_rows)}
        for key in ("pa_all_p50", "pa_core_p50", "pa_shell_p50", "pa_all_p95", "pa_core_p95", "pa_all_frac_below_-200", "pa_core_frac_below_-200", "pa_all_frac_above_100", "pa_core_frac_above_100"):
            vals = np.array([r[key] for r in group_rows if r.get(key) is not None], dtype=np.float64)
            if vals.size:
                summary["groups"][group][key] = {
                    "median_across_cases": float(np.median(vals)),
                    "p10_across_cases": float(np.percentile(vals, 10)),
                    "p90_across_cases": float(np.percentile(vals, 90)),
                }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PA CORE INTENSITY SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors_detail"}, indent=2))
    print(f"\nPer-case CSV: {OUT_CSV}")
    print(f"Summary JSON: {OUT_JSON}")
    if errors:
        print("\nERROR DETAILS:")
        print(json.dumps(errors, indent=2))
    return 0 if not errors and len(results) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
