from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

MANIFEST = Path('outputs/metrics/airrc_uid_manifest_with_luna_subset.csv')
DOWNLOAD_MANIFEST = Path('outputs/metrics/airrc_download_manifest.csv')
OUTPUT_CSV = Path('outputs/metrics/airrc_geometry_audit.csv')
OUTPUT_JSON = Path('outputs/metrics/airrc_geometry_audit_summary.json')
CHUNK_VOXELS = 250_000
TOL = 1e-5
EXPECTED_LABELS = {0, 1, 2, 3, 4}


def matrix_from_image(img: sitk.Image) -> np.ndarray:
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    return direction @ np.diag(spacing)


def corners_from_geometry(size, origin, direction, spacing):
    size = np.asarray(size, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)
    A = np.asarray(direction, dtype=np.float64).reshape(3, 3) @ np.diag(np.asarray(spacing, dtype=np.float64))
    indices = np.array([[x, y, z] for x in (0, size[0]-1) for y in (0, size[1]-1) for z in (0, size[2]-1)], dtype=np.float64).T
    pts = origin[:, None] + A @ indices
    return pts.min(axis=1), pts.max(axis=1)


def direction_is_orthonormal(direction) -> bool:
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return bool(np.allclose(d.T @ d, np.eye(3), atol=1e-4) and abs(abs(np.linalg.det(d)) - 1.0) < 1e-4)


def read_ct_info(mhd_path: Path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(mhd_path))
    reader.ReadImageInformation()
    size = tuple(int(x) for x in reader.GetSize())
    spacing = tuple(float(x) for x in reader.GetSpacing())
    origin = tuple(float(x) for x in reader.GetOrigin())
    direction = tuple(float(x) for x in reader.GetDirection())
    return size, spacing, origin, direction


def audit_case(row, download_row):
    uid = str(row.series_instance_uid)
    patient = str(row.patient_id)
    subset = str(row.luna_subset)
    annotation_path = Path(str(row.annotation_path))
    ct_dir = Path('data/airrc_luna16') / subset
    mhd_path = ct_dir / f'{uid}.mhd'
    raw_path = ct_dir / f'{uid}.raw'
    result = {
        'series_instance_uid': uid,
        'patient_id': patient,
        'luna_subset': subset,
        'contrast_status': str(row.contrast_status),
        'annotation_path': str(annotation_path),
        'mhd_path': str(mhd_path),
        'raw_path': str(raw_path),
        'status': 'PASS',
        'issues': '',
    }
    issues = []

    if not annotation_path.is_file():
        issues.append('MISSING_ANNOTATION')
    if not mhd_path.is_file():
        issues.append('MISSING_MHD')
    if not raw_path.is_file():
        issues.append('MISSING_RAW')
    if issues:
        result['status'] = 'FAIL'
        result['issues'] = ';'.join(issues)
        return result

    expected_raw = int(download_row.get('expected_raw_bytes', 0))
    actual_raw = raw_path.stat().st_size
    result['expected_raw_bytes'] = expected_raw
    result['actual_raw_bytes'] = actual_raw
    if expected_raw and actual_raw != expected_raw:
        issues.append('RAW_SIZE_MISMATCH')

    ct_size, ct_spacing, ct_origin, ct_direction = read_ct_info(mhd_path)
    seg = sitk.ReadImage(str(annotation_path))
    seg_size = tuple(int(x) for x in seg.GetSize())
    seg_spacing = tuple(float(x) for x in seg.GetSpacing())
    seg_origin = tuple(float(x) for x in seg.GetOrigin())
    seg_direction = tuple(float(x) for x in seg.GetDirection())

    result.update({
        'ct_size_xyz': 'x'.join(map(str, ct_size)),
        'ct_spacing_xyz_mm': ','.join(f'{x:.6g}' for x in ct_spacing),
        'ct_origin_lps_mm': ','.join(f'{x:.6f}' for x in ct_origin),
        'ct_direction': ','.join(f'{x:.6g}' for x in ct_direction),
        'mask_size_xyz': 'x'.join(map(str, seg_size)),
        'mask_spacing_xyz_mm': ','.join(f'{x:.6g}' for x in seg_spacing),
        'mask_origin_lps_mm': ','.join(f'{x:.6f}' for x in seg_origin),
        'mask_direction': ','.join(f'{x:.6g}' for x in seg_direction),
    })

    ct_ok = direction_is_orthonormal(ct_direction) and all(np.isfinite(ct_spacing)) and all(x > 0 for x in ct_spacing)
    seg_ok = direction_is_orthonormal(seg_direction) and all(np.isfinite(seg_spacing)) and all(x > 0 for x in seg_spacing)
    if not ct_ok:
        issues.append('INVALID_CT_GEOMETRY')
    if not seg_ok:
        issues.append('INVALID_MASK_GEOMETRY')

    ct_min, ct_max = corners_from_geometry(ct_size, ct_origin, ct_direction, ct_spacing)
    seg_min, seg_max = corners_from_geometry(seg_size, seg_origin, seg_direction, seg_spacing)
    result['ct_bounds_min_lps_mm'] = ','.join(f'{x:.4f}' for x in ct_min)
    result['ct_bounds_max_lps_mm'] = ','.join(f'{x:.4f}' for x in ct_max)
    result['mask_bounds_min_lps_mm'] = ','.join(f'{x:.4f}' for x in seg_min)
    result['mask_bounds_max_lps_mm'] = ','.join(f'{x:.4f}' for x in seg_max)

    arr = sitk.GetArrayViewFromImage(seg)  # z,y,x view; one case at a time
    labels, counts = np.unique(arr, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(labels, counts)}
    result['label_values'] = ','.join(map(str, sorted(label_counts)))
    result['pa_voxels'] = label_counts.get(3, 0)
    result['airway_lumen_voxels'] = label_counts.get(1, 0)
    result['airway_wall_voxels'] = label_counts.get(2, 0)
    result['pulmonary_vein_voxels'] = label_counts.get(4, 0)
    unexpected = set(label_counts) - EXPECTED_LABELS
    if unexpected:
        issues.append('UNEXPECTED_MASK_LABELS:' + ','.join(map(str, sorted(unexpected))))
    if result['pa_voxels'] <= 0:
        issues.append('EMPTY_PA_LABEL')

    # Map each class-3 voxel centre from mask index space into CT continuous index space.
    pa_zyx = np.argwhere(arr == 3)
    if len(pa_zyx):
        pa_xyz = pa_zyx[:, ::-1].astype(np.float64, copy=False)
        ct_A = matrix_from_image(type('ImageGeometry', (), {
            'GetDirection': lambda self: ct_direction,
            'GetSpacing': lambda self: ct_spacing,
        })())
        seg_A = matrix_from_image(seg)
        delta = np.asarray(seg_origin, dtype=np.float64) - np.asarray(ct_origin, dtype=np.float64)
        ct_dims = np.asarray(ct_size, dtype=np.float64)
        outside_centres = 0
        outside_fov = 0
        ci_min = np.full(3, np.inf)
        ci_max = np.full(3, -np.inf)
        for start in range(0, len(pa_xyz), CHUNK_VOXELS):
            p = pa_xyz[start:start + CHUNK_VOXELS].T
            ci = np.linalg.solve(ct_A, delta[:, None] + seg_A @ p)
            ci_min = np.minimum(ci_min, ci.min(axis=1))
            ci_max = np.maximum(ci_max, ci.max(axis=1))
            outside_centres += int(np.any((ci < -TOL) | (ci > (ct_dims - 1 + TOL)), axis=0).sum())
            outside_fov += int(np.any((ci < (-0.5 - TOL)) | (ci > (ct_dims - 0.5 + TOL)), axis=0).sum())
        result['pa_outside_ct_center_domain'] = outside_centres
        result['pa_outside_ct_fov'] = outside_fov
        result['pa_ct_index_min_xyz'] = ','.join(f'{x:.4f}' for x in ci_min)
        result['pa_ct_index_max_xyz'] = ','.join(f'{x:.4f}' for x in ci_max)
        if outside_fov:
            issues.append('PA_VOXELS_OUTSIDE_CT_FOV')
        elif outside_centres:
            issues.append('PA_VOXELS_NEAR_CT_EDGE')
    else:
        result['pa_outside_ct_center_domain'] = None
        result['pa_outside_ct_fov'] = None
        result['pa_ct_index_min_xyz'] = ''
        result['pa_ct_index_max_xyz'] = ''

    if 'PA_VOXELS_OUTSIDE_CT_FOV' in issues or any(x.startswith(('MISSING_', 'RAW_SIZE_MISMATCH', 'INVALID_', 'EMPTY_PA', 'UNEXPECTED_')) for x in issues):
        result['status'] = 'FAIL'
    elif issues:
        result['status'] = 'WARN'
    result['issues'] = ';'.join(issues)
    return result


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f'Missing {MANIFEST}; run from NeuroVessel project root.')
    if not DOWNLOAD_MANIFEST.is_file():
        raise FileNotFoundError(f'Missing {DOWNLOAD_MANIFEST}; complete the CT downloader first.')

    manifest = pd.read_csv(MANIFEST)
    downloads = pd.read_csv(DOWNLOAD_MANIFEST)
    if len(manifest) != 254 or manifest['series_instance_uid'].nunique() != 254:
        raise ValueError('Expected 254 unique AirRC annotations in the UID manifest.')
    if 'status' not in downloads.columns:
        raise ValueError('Download manifest has no status column.')
    done = downloads[downloads['status'].astype(str).str.lower().eq('complete')]
    if done['series_instance_uid'].nunique() != 254:
        raise ValueError(f"Download manifest reports only {done['series_instance_uid'].nunique()} completed unique series.")
    download_lookup = {str(r['series_instance_uid']): r for _, r in done.iterrows()}

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f'AirRC geometry audit: {len(manifest)} CT-mask pairs', flush=True)
    print('This checks metadata geometry and the physical location of every pulmonary-artery (label 3) voxel.', flush=True)
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        uid = str(row.series_instance_uid)
        if uid not in download_lookup:
            out = {'series_instance_uid': uid, 'patient_id': str(row.patient_id), 'status': 'FAIL', 'issues': 'MISSING_DOWNLOAD_MANIFEST_ENTRY'}
        else:
            out = audit_case(row, download_lookup[uid])
        results.append(out)
        print(f"[{i:03d}/254] {out.get('patient_id', 'unknown')} | {out['status']} | PA voxels={out.get('pa_voxels', 'n/a')} | {out.get('issues', '')}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    summary = {
        'total_cases': int(len(df)),
        'pass': int((df['status'] == 'PASS').sum()),
        'warning': int((df['status'] == 'WARN').sum()),
        'fail': int((df['status'] == 'FAIL').sum()),
        'total_pa_voxels': int(pd.to_numeric(df.get('pa_voxels', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()),
        'total_pa_voxels_outside_ct_fov': int(pd.to_numeric(df.get('pa_outside_ct_fov', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()),
        'cases_with_pa_voxels_near_ct_edge': int(((pd.to_numeric(df.get('pa_outside_ct_center_domain', pd.Series(dtype=float)), errors='coerce').fillna(0) > 0) & (pd.to_numeric(df.get('pa_outside_ct_fov', pd.Series(dtype=float)), errors='coerce').fillna(0) == 0)).sum()),
        'csv_report': str(OUTPUT_CSV),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== AUDIT SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print(f'Full report: {OUTPUT_CSV}')
    print(f'Summary: {OUTPUT_JSON}')
    return 0 if summary['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
