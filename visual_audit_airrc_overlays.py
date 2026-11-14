from pathlib import Path
import csv
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
MANIFEST = ROOT / 'outputs' / 'metrics' / 'airrc_uid_manifest_with_luna_subset.csv'
OUTDIR = ROOT / 'outputs' / 'visualizations' / 'airrc_alignment'
PATIENTS = [
    'LIDC-IDRI-1001',  # first geometry-validated sample
    'LIDC-IDRI-0278',  # example with high PA median intensity
    'LIDC-IDRI-0187',  # non-contrast case with low PA median intensity
    'LIDC-IDRI-0470',  # high PA median intensity example
]


def find_ct(uid: str, subset: str) -> Path:
    preferred = ROOT / 'data' / 'airrc_luna16' / subset / f'{uid}.mhd'
    if preferred.exists():
        return preferred
    candidates = list((ROOT / 'data' / 'airrc_luna16').rglob(f'{uid}.mhd'))
    if not candidates:
        raise FileNotFoundError(f'Cannot find CT header for {uid} under data/airrc_luna16')
    # Prefer a regular subset folder over the earlier sample folder, if both exist.
    candidates.sort(key=lambda p: (p.parent.name != subset, 'sample' in p.parts, len(p.parts)))
    return candidates[0]


def main() -> int:
    if not MANIFEST.exists():
        print(f'Missing manifest: {MANIFEST}')
        return 2
    manifest = pd.read_csv(MANIFEST)
    required = {'patient_id', 'series_instance_uid', 'luna_subset', 'annotation_path'}
    missing = required.difference(manifest.columns)
    if missing:
        print('Manifest missing columns:', sorted(missing))
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for patient in PATIENTS:
        rows = manifest.loc[manifest['patient_id'].astype(str) == patient]
        if len(rows) != 1:
            print(f'{patient}: expected one manifest row, found {len(rows)}; skipping')
            continue
        row = rows.iloc[0]
        uid = str(row['series_instance_uid'])
        subset = str(row['luna_subset'])
        ann_path = Path(str(row['annotation_path']))
        ct_path = find_ct(uid, subset)
        if not ann_path.exists():
            raise FileNotFoundError(f'Annotation not found: {ann_path}')

        print(f'[{patient}] reading CT and annotation...')
        ct = sitk.ReadImage(str(ct_path))
        seg = sitk.ReadImage(str(ann_path))
        ct_ref = sitk.Resample(ct, seg, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
        ct_arr = sitk.GetArrayFromImage(ct_ref)
        seg_arr = sitk.GetArrayViewFromImage(seg)
        pa = (seg_arr == 3)
        if not np.any(pa):
            print(f'{patient}: no label-3 voxels; skipping')
            continue

        z_counts = pa.sum(axis=(1, 2))
        z0 = int(np.argmax(z_counts))
        yy, xx = np.where(pa[z0])
        y0 = int(np.median(yy))
        x0 = int(np.median(xx))
        values = ct_arr[pa]
        p1, p50, p95, p99 = np.percentile(values, [1, 50, 95, 99])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Axial, coronal and sagittal planes through a dense PA slice/representative location.
        panels = [
            (ct_arr[z0, :, :], pa[z0, :, :], f'Axial z={z0}'),
            (ct_arr[:, y0, :], pa[:, y0, :], f'Coronal y={y0}'),
            (ct_arr[:, :, x0], pa[:, :, x0], f'Sagittal x={x0}'),
        ]
        for ax, (img, mask2d, title) in zip(axes, panels):
            ax.imshow(img, cmap='gray', vmin=-1000, vmax=700, origin='lower')
            if np.any(mask2d):
                ax.contour(mask2d.astype(np.uint8), levels=[0.5], colors=['red'], linewidths=0.7)
            ax.set_title(title)
            ax.set_axis_off()
        contrast = str(row.get('contrast_status', 'unknown'))
        fig.suptitle(
            f'{patient} | {contrast}\nPA voxels={int(pa.sum()):,} | '
            f'PA HU p1/p50/p95/p99={p1:.1f}/{p50:.1f}/{p95:.1f}/{p99:.1f}',
            fontsize=11,
        )
        fig.tight_layout()
        out = OUTDIR / f'{patient}_overlay.png'
        fig.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)

        summary.append({
            'patient_id': patient,
            'contrast_status': contrast,
            'series_instance_uid': uid,
            'ct_path': str(ct_path),
            'annotation_path': str(ann_path),
            'pa_voxels': int(pa.sum()),
            'pa_p1': float(p1),
            'pa_p50': float(p50),
            'pa_p95': float(p95),
            'pa_p99': float(p99),
            'overlay_png': str(out),
        })
        print(f'  saved: {out}')
        print(f'  PA HU p1/p50/p95/p99: {p1:.1f}/{p50:.1f}/{p95:.1f}/{p99:.1f}')

    csv_path = OUTDIR / 'visual_audit_summary.csv'
    if summary:
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print('\nVISUAL OVERLAY AUDIT FINISHED')
    print('Cases visualized:', len(summary))
    print('Output folder:', OUTDIR)
    print('Summary CSV:', csv_path)
    print('Review the red contours against vessel anatomy before fixing HU clipping.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
