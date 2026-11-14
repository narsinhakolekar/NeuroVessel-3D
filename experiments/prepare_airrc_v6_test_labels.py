#!/usr/bin/env python3
"""Prepare ONLY the 26 locked AirRC test ground-truth labels for final evaluation.

This script does not open CT images, train a model, or touch train/validation labels.
It converts the original 5-class AirRC annotation to the V6 mapping:
    source 0/1/2 -> 0 (background)
    source 3       -> 1 (pulmonary artery)
    source 4       -> 2 (pulmonary vein)

Run from the NeuroVessel project root:
    python -X faulthandler -u experiments\\prepare_airrc_v6_test_labels.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
SPLIT_CSV = METRICS_DIR / "airrc_v5_splits.csv"
MANIFEST_CSV = METRICS_DIR / "airrc_uid_manifest_with_luna_subset.csv"
OUTPUT_DIR = ROOT / "data" / "processed_airrc_v6" / "labels_multiclass"
ORIGINAL_ANNOTATION_DIR = (
    Path.home() / "Downloads" / "AirRC_annotations_extracted" / "labelsTr"
)

EXPECTED = {"train": 202, "validation": 26, "test": 26}
SOURCE_LABELS = {0, 1, 2, 3, 4}


def find_column(df: pd.DataFrame, choices: list[str], required: bool = True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in choices:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise KeyError(
            f"Could not find any column from {choices}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def clean(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value


def normalize_split(value) -> str:
    value = clean(value)
    if value is None:
        raise ValueError("Found an empty split value in split manifest")
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "train": "train", "training": "train",
        "val": "validation", "valid": "validation",
        "validation": "validation", "validate": "validation",
        "test": "test", "testing": "test",
    }
    if value not in aliases:
        raise ValueError(f"Unrecognized split value: {value!r}")
    return aliases[value]


def resolve_annotation_path(value, uid: str) -> Path:
    candidates: list[Path] = []
    value_text = clean(value)
    if value_text:
        supplied = Path(value_text)
        if supplied.is_absolute():
            candidates.append(supplied)
        else:
            candidates.extend([ROOT / supplied, Path.cwd() / supplied])
            candidates.append(ORIGINAL_ANNOTATION_DIR / supplied.name)
    candidates.append(ORIGINAL_ANNOTATION_DIR / f"{uid}.nii.gz")

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Original annotation not found for test UID {uid}. Tried:\n{tried}"
    )


def load_test_manifest_rows():
    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Split CSV missing: {SPLIT_CSV}")
    if not MANIFEST_CSV.is_file():
        raise FileNotFoundError(f"UID/annotation manifest missing: {MANIFEST_CSV}")

    split_df = pd.read_csv(SPLIT_CSV, dtype=str)
    split_col = find_column(split_df, ["split", "partition", "data_split", "dataset_split"])
    split_uid_col = find_column(
        split_df,
        ["uid", "case_uid", "case_id", "seriesuid", "series_uid", "series_instance_uid", "id"],
    )
    split_patient_col = find_column(
        split_df, ["patient_id", "subject_id", "patient", "subject"], required=False
    )

    split_df = split_df.copy()
    split_df["_uid"] = split_df[split_uid_col].map(clean)
    split_df["_split"] = split_df[split_col].map(normalize_split)
    if split_df["_uid"].isna().any():
        raise RuntimeError("Split manifest contains an empty UID")
    if split_df["_uid"].duplicated().any():
        dups = split_df.loc[split_df["_uid"].duplicated(), "_uid"].tolist()
        raise RuntimeError(f"Duplicate UIDs in split manifest: {dups[:5]}")

    counts = split_df["_split"].value_counts().to_dict()
    counts = {k: int(counts.get(k, 0)) for k in EXPECTED}
    if counts != EXPECTED:
        raise RuntimeError(f"Expected split counts {EXPECTED}, found {counts}")

    patient_sets = {k: set() for k in EXPECTED}
    for _, row in split_df.iterrows():
        uid = row["_uid"]
        patient = clean(row[split_patient_col]) if split_patient_col else uid
        patient_sets[row["_split"]].add(patient or uid)
    for a, first in enumerate(("train", "validation", "test")):
        for second in ("train", "validation", "test")[a + 1:]:
            overlap = patient_sets[first] & patient_sets[second]
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between {first} and {second}: {sorted(overlap)[:5]}"
                )

    manifest = pd.read_csv(MANIFEST_CSV, dtype=str)
    manifest_uid_col = find_column(
        manifest,
        ["uid", "case_uid", "case_id", "seriesuid", "series_uid", "series_instance_uid"],
    )
    annotation_col = find_column(
        manifest,
        ["annotation_path", "label_path", "segmentation_path", "mask_path", "label_file"],
    )
    manifest = manifest.copy()
    manifest["_uid"] = manifest[manifest_uid_col].map(clean)
    if manifest["_uid"].isna().any() or manifest["_uid"].duplicated().any():
        raise RuntimeError("UID manifest has empty or duplicate UIDs")

    test_rows = split_df.loc[split_df["_split"] == "test", ["_uid"]].merge(
        manifest[["_uid", annotation_col]], on="_uid", how="left", validate="one_to_one"
    )
    if len(test_rows) != EXPECTED["test"] or test_rows[annotation_col].isna().any():
        missing = test_rows.loc[test_rows[annotation_col].isna(), "_uid"].tolist()
        raise RuntimeError(
            f"Could not map all 26 test UIDs to original annotation paths. Missing: {missing}"
        )

    return test_rows, annotation_col


def convert_one(uid: str, annotation_value) -> dict:
    source_path = resolve_annotation_path(annotation_value, uid)
    source = sitk.ReadImage(str(source_path))
    if source.GetDimension() != 3:
        raise RuntimeError(f"Expected a 3D annotation for {uid}, got {source.GetDimension()}D")

    source_array = sitk.GetArrayFromImage(source)
    labels = set(int(x) for x in np.unique(source_array))
    if not labels.issubset(SOURCE_LABELS):
        raise RuntimeError(f"Unexpected source labels for {uid}: {sorted(labels)}")

    pa_count = int(np.count_nonzero(source_array == 3))
    pv_count = int(np.count_nonzero(source_array == 4))
    if pa_count == 0 or pv_count == 0:
        raise RuntimeError(
            f"Missing vessel class for {uid}: PA voxels={pa_count}, PV voxels={pv_count}"
        )

    # Exact V6 remapping used by train/validation labels.
    target_array = np.zeros(source_array.shape, dtype=np.uint8)
    target_array[source_array == 3] = 1
    target_array[source_array == 4] = 2

    target = sitk.GetImageFromArray(target_array)
    target.CopyInformation(source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{uid}_multiclass.nii.gz"
    if output_path.exists():
        existing_image = sitk.ReadImage(str(output_path))
        existing_array = sitk.GetArrayFromImage(existing_image)
        if (
            existing_array.shape != target_array.shape
            or not np.array_equal(existing_array, target_array)
            or existing_image.GetSize() != target.GetSize()
            or not np.allclose(existing_image.GetSpacing(), target.GetSpacing(), atol=1e-6)
            or not np.allclose(existing_image.GetOrigin(), target.GetOrigin(), atol=1e-6)
            or not np.allclose(existing_image.GetDirection(), target.GetDirection(), atol=1e-7)
        ):
            raise RuntimeError(
                f"Existing test label does not match source annotation exactly: {output_path}"
            )
        status = "EXISTING_VERIFIED"
    else:
        temp_path = OUTPUT_DIR / f"{uid}_multiclass.tmp.nii.gz"
        sitk.WriteImage(target, str(temp_path), True)
        os.replace(temp_path, output_path)
        status = "CREATED"

    # Reopen and verify persisted output.
    saved = sitk.ReadImage(str(output_path))
    saved_array = sitk.GetArrayFromImage(saved)
    if not np.array_equal(saved_array, target_array):
        raise RuntimeError(f"Saved label failed voxel equality check: {output_path}")
    out_labels = set(int(x) for x in np.unique(saved_array))
    if not out_labels.issubset({0, 1, 2}) or not {1, 2}.issubset(out_labels):
        raise RuntimeError(f"Invalid V6 label values for {uid}: {sorted(out_labels)}")

    return {
        "uid": uid,
        "status": status,
        "source_annotation": str(source_path),
        "output_label": str(output_path),
        "size_xyz": json.dumps(saved.GetSize()),
        "spacing_xyz_mm": json.dumps(saved.GetSpacing()),
        "background_voxels": int(np.count_nonzero(saved_array == 0)),
        "pa_voxels": int(np.count_nonzero(saved_array == 1)),
        "pv_voxels": int(np.count_nonzero(saved_array == 2)),
        "verified_exact_source_remap": True,
    }


def main() -> int:
    print("=" * 72)
    print("NEUROVESSEL V6 — TEST-ONLY GROUND-TRUTH PREPARATION")
    print("=" * 72)
    print("This script reads original annotations for the 26 locked TEST cases only.")
    print("It does NOT read CT images, train/validation volumes, or model checkpoints.")
    print("Mapping: source {0,1,2}->0, source 3->1 (PA), source 4->2 (PV)")
    print(f"Split manifest: {SPLIT_CSV}")
    print(f"Original annotation directory fallback: {ORIGINAL_ANNOTATION_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    rows, annotation_col = load_test_manifest_rows()
    print(f"Verified split counts: {EXPECTED}")
    print(f"Test annotations to process: {len(rows)}\n")

    results = []
    for index, row in enumerate(rows.itertuples(index=False, name=None), start=1):
        uid, annotation_value = row
        print(f"[{index:02d}/26] {uid}", flush=True)
        result = convert_one(uid, annotation_value)
        results.append(result)
        print(
            f"  {result['status']} | PA={result['pa_voxels']:,} "
            f"PV={result['pv_voxels']:,} | grid={result['size_xyz']}",
            flush=True,
        )

    report_csv = METRICS_DIR / "airrc_v6_test_label_preparation.csv"
    summary_json = METRICS_DIR / "airrc_v6_test_label_preparation_summary.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report_csv, index=False)
    summary = {
        "purpose": "Prepare ground-truth labels for one final independent locked test evaluation",
        "split_counts": EXPECTED,
        "test_cases_prepared": len(results),
        "source_label_mapping": {"0,1,2": 0, "3": 1, "4": 2},
        "ct_images_read": 0,
        "train_or_validation_volumes_read": 0,
        "checkpoint_loaded": False,
        "all_test_labels_verified": True,
        "results_csv": str(report_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("TEST LABEL PREPARATION COMPLETE")
    print(f"Verified labels: {len(results)}/26")
    print(f"CSV report: {report_csv}")
    print(f"Summary: {summary_json}")
    print("No CT images were opened and no model inference/training was performed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nFATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
