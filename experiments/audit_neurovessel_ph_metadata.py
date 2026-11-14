"""
NeuroVessel — PH clinical metadata audit

Purpose:
    Inventory tabular/text metadata inside the NeuroVessel project and look
    for potential pulmonary-hypertension outcomes or clinical variables.

Safety / scope:
    - Does NOT read, modify, or move NIfTI/CT/segmentation volumes.
    - Does NOT access model checkpoints or predictions.
    - Reads only metadata-like files (CSV/TSV/JSON/TXT/XML/YAML/XLSX/XLS).
    - A keyword match is only a candidate, NOT proof of a valid PH label.

Run from project root or save in NeuroVessel/experiments and execute there:
    python -X faulthandler -u experiments/audit_neurovessel_ph_metadata.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [PROJECT_ROOT / "data", PROJECT_ROOT / "outputs"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics" / "ph_metadata_audit"

METADATA_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".txt", ".xml", ".yaml", ".yml",
    ".xlsx", ".xls",
}
SKIP_DIR_NAMES = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".venv", "venv", "env", "site-packages",
    "predictions", "visualizations", "checkpoints",
}
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_ROWS_PER_TABLE = 200_000

# Deliberately use explicit terms; generic "PH" matching produces many false hits.
KEYWORDS = {
    "pulmonary_hypertension": re.compile(r"pulmonary\s+hypertension", re.I),
    "pulmonary_arterial_hypertension": re.compile(r"pulmonary\s+arterial\s+hypertension", re.I),
    "pah_token": re.compile(r"\bPAH\b", re.I),
    "mpap": re.compile(r"\bmPAP\b|mean\s+pulmonary\s+arterial\s+pressure", re.I),
    "right_heart_catheterization": re.compile(r"right\s+heart\s+catheter(?:ization|isation)?|\bRHC\b", re.I),
    "rvsp": re.compile(r"\bRVSP\b|right\s+ventricular\s+systolic\s+pressure", re.I),
    "echocardiography": re.compile(r"echocardiogra(?:phy|phic)|\becho\b", re.I),
    "ph_diagnosis_or_outcome": re.compile(r"PH\s*(?:diagnosis|status|label|outcome)|(?:diagnosis|status|outcome)\s*of\s*PH", re.I),
}
GENERAL_CLINICAL_TERMS = re.compile(
    r"diagnos|clinical|outcome|disease|histolog|malignan|benign|patient|age|sex|gender|smok|survival|follow.?up",
    re.I,
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)


def scan_csv(path: Path, delimiter: str) -> dict[str, Any]:
    result: dict[str, Any] = {"headers": [], "rows_scanned": 0, "keyword_hits": {}, "clinical_header_candidates": []}
    hit_examples: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration:
            return result
        result["headers"] = [h.strip() for h in headers]
        result["clinical_header_candidates"] = [h for h in result["headers"] if GENERAL_CLINICAL_TERMS.search(h)]
        for row in reader:
            result["rows_scanned"] += 1
            joined = " | ".join(row)
            for name, pattern in KEYWORDS.items():
                if pattern.search(joined):
                    hit_examples.setdefault(name, [])
                    if len(hit_examples[name]) < 5:
                        hit_examples[name].append(joined[:300])
            if result["rows_scanned"] >= MAX_ROWS_PER_TABLE:
                result["row_scan_limit_reached"] = True
                break
    result["keyword_hits"] = hit_examples
    return result


def scan_text_like(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"keyword_hits": {}, "clinical_term_lines": []}
    if path.stat().st_size > MAX_TEXT_BYTES:
        result["skipped_reason"] = f"larger than {MAX_TEXT_BYTES} bytes"
        return result
    hits: dict[str, list[str]] = {}
    clinical_lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for name, pattern in KEYWORDS.items():
                if pattern.search(stripped):
                    hits.setdefault(name, [])
                    if len(hits[name]) < 5:
                        hits[name].append(f"L{line_number}: {stripped[:300]}")
            if GENERAL_CLINICAL_TERMS.search(stripped) and len(clinical_lines) < 10:
                clinical_lines.append(f"L{line_number}: {stripped[:300]}")
    result["keyword_hits"] = hits
    result["clinical_term_lines"] = clinical_lines
    return result


def scan_xlsx(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"sheets": [], "keyword_hits": {}, "clinical_header_candidates": []}
    try:
        import openpyxl  # type: ignore
    except Exception:
        result["skipped_reason"] = "openpyxl is not installed; workbook was listed but not read"
        return result
    hits: dict[str, list[str]] = {}
    headers_found: list[str] = []
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            sheet_info = {"name": sheet.title, "max_row": sheet.max_row, "max_column": sheet.max_column}
            result["sheets"].append(sheet_info)
            for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                values = [safe_text(v) for v in row]
                joined = " | ".join(values)
                if row_index == 1:
                    headers_found.extend(v for v in values if v and GENERAL_CLINICAL_TERMS.search(v))
                for name, pattern in KEYWORDS.items():
                    if pattern.search(joined):
                        hits.setdefault(name, [])
                        if len(hits[name]) < 5:
                            hits[name].append(f"{sheet.title}!R{row_index}: {joined[:300]}")
                if row_index >= MAX_ROWS_PER_TABLE:
                    break
    finally:
        workbook.close()
    result["keyword_hits"] = hits
    result["clinical_header_candidates"] = headers_found
    return result


def main() -> int:
    print("=" * 76)
    print("NEUROVESSEL — PULMONARY HYPERTENSION METADATA AUDIT")
    print("=" * 76)
    print(f"Project root: {PROJECT_ROOT}")
    print("Scope: metadata files only; no CT, NIfTI masks, predictions, or checkpoints will be opened.")

    files: list[Path] = []
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            print(f"WARNING: scan root not found: {scan_root}")
            continue
        for current, dirnames, filenames in __import__("os").walk(scan_root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIR_NAMES]
            current_path = Path(current)
            for filename in filenames:
                path = current_path / filename
                if path.suffix.lower() in METADATA_EXTENSIONS:
                    # Avoid scanning our own generated report on repeat runs.
                    if OUTPUT_DIR in path.parents:
                        continue
                    files.append(path)

    files = sorted(set(files))
    print(f"Metadata-like files discovered: {len(files)}")

    records: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        ext = path.suffix.lower()
        record: dict[str, Any] = {
            "relative_path": str(path.relative_to(PROJECT_ROOT)),
            "extension": ext,
            "size_bytes": path.stat().st_size,
        }
        try:
            if ext in {".csv", ".tsv"}:
                record["inspection"] = scan_csv(path, "\t" if ext == ".tsv" else ",")
            elif ext == ".xlsx":
                record["inspection"] = scan_xlsx(path)
            elif ext == ".xls":
                record["inspection"] = {"skipped_reason": "legacy .xls listed but not parsed by this script"}
            else:
                record["inspection"] = scan_text_like(path)
        except Exception as exc:
            record["inspection_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        print(f"[{index:03d}/{len(files):03d}] {record['relative_path']}")

    candidate_files = []
    for record in records:
        inspection = record.get("inspection", {})
        hits = inspection.get("keyword_hits", {}) if isinstance(inspection, dict) else {}
        if hits or inspection.get("clinical_header_candidates") or inspection.get("clinical_term_lines"):
            candidate_files.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "ph_metadata_audit.json"
    txt_path = OUTPUT_DIR / "ph_metadata_audit_report.txt"

    summary = {
        "project": "NeuroVessel",
        "audit_scope": "metadata-like files under data/ and outputs/",
        "data_volumes_opened": False,
        "checkpoints_opened": False,
        "metadata_files_discovered": len(files),
        "candidate_metadata_files": len(candidate_files),
        "keyword_definitions": list(KEYWORDS),
        "important_note": "Keyword matches are leads for manual verification, not validated PH outcome labels. Absence of a match does not prove no external clinical metadata exists.",
        "files": records,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "NEUROVESSEL — PH METADATA AUDIT REPORT",
        "=" * 60,
        f"Project root: {PROJECT_ROOT}",
        f"Metadata-like files discovered: {len(files)}",
        f"Files containing possible clinical/PH terms: {len(candidate_files)}",
        "",
        "CANDIDATE FILES",
        "-" * 60,
    ]
    if not candidate_files:
        lines.append("No PH/clinical keyword candidates were found in the scanned metadata files.")
    for record in candidate_files:
        lines.append(f"\n{record['relative_path']}")
        inspection = record.get("inspection", {})
        if inspection.get("headers"):
            lines.append("  Headers: " + ", ".join(inspection["headers"]))
        if inspection.get("clinical_header_candidates"):
            lines.append("  Clinical-looking headers: " + ", ".join(inspection["clinical_header_candidates"]))
        if inspection.get("keyword_hits"):
            for keyword, examples in inspection["keyword_hits"].items():
                lines.append(f"  {keyword}:")
                lines.extend(f"    - {example}" for example in examples[:3])
        if inspection.get("clinical_term_lines"):
            lines.extend("  " + x for x in inspection["clinical_term_lines"][:5])
        if inspection.get("skipped_reason"):
            lines.append("  Note: " + inspection["skipped_reason"])
    lines += [
        "",
        "INTERPRETATION",
        "- Keyword matches are not validated PH outcomes.",
        "- Check labels against an authoritative clinical reference (ideally RHC-confirmed diagnosis/hemodynamics).",
        "- No PH prediction model should be trained until this label audit is resolved.",
        "- This audit did not read CT/NIfTI volumes, masks, predictions, or checkpoints.",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 76)
    print("AUDIT FINISHED")
    print(f"Metadata files discovered: {len(files)}")
    print(f"Candidate metadata files: {len(candidate_files)}")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")
    print("No CT/NIfTI volumes, masks, predictions, or checkpoints were opened.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.")
        raise SystemExit(130)
