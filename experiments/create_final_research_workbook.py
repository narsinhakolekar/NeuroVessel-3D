"""
NeuroVessel final research-results workbook generator.

Run from the NeuroVessel project root:
    python experiments/create_final_research_workbook.py

Creates:
    outputs/reports/NeuroVessel_Final_Research_Results.xlsx

The script reads existing metrics only. It does NOT train, infer, alter predictions,
modify checkpoints, or overwrite the historical NeuroVessel_Experimental_Results.xlsx.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import ColorScaleRule


ROOT = Path.cwd()
METRICS = ROOT / "outputs" / "metrics"
REPORTS = ROOT / "outputs" / "reports"
OUTPUT = REPORTS / "NeuroVessel_Final_Research_Results.xlsx"
HISTORICAL = REPORTS / "NeuroVessel_Experimental_Results.xlsx"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[MISSING] {path.relative_to(ROOT)}")
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"data": data}
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[MISSING] {path.relative_to(ROOT)}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def pct(value: Any) -> Any:
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return value


def summary_value(data: dict[str, Any] | None, *keys: str, default=None):
    if not data:
        return default
    for key in keys:
        if key in data:
            return data[key]
    return default


def add_dataframe(wb: Workbook, title: str, df: pd.DataFrame | None,
                  note: str | None = None) -> None:
    ws = wb.create_sheet(title[:31])
    if note:
        ws.append([note])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(df.columns) if df is not None and len(df.columns) else 1))
        ws["A1"].font = Font(italic=True, color="666666")
        ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
        header_row = 3
    else:
        header_row = 1

    if df is None or df.empty:
        ws.cell(header_row, 1, "No source file found; see project metrics directory.")
        return

    for col_idx, name in enumerate(df.columns, 1):
        ws.cell(header_row, col_idx, str(name))
    for row_idx, row in enumerate(df.itertuples(index=False, name=None), header_row + 1):
        for col_idx, value in enumerate(row, 1):
            if pd.isna(value):
                value = None
            elif hasattr(value, "item"):
                value = value.item()
            ws.cell(row_idx, col_idx, value)

    ws.freeze_panes = f"A{header_row + 1}"
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(df.columns))}{header_row + len(df)}"
    if len(df) and len(df.columns):
        ref = f"A{header_row}:{get_column_letter(len(df.columns))}{header_row + len(df)}"
        safe = "".join(ch for ch in title if ch.isalnum())[:18] or "Data"
        table = Table(displayName=f"T{safe}", ref=ref)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True,
                                              showColumnStripes=False)
        try:
            ws.add_table(table)
        except ValueError:
            pass


def epoch_summary(path: Path, version: str, threshold_note: str) -> pd.DataFrame:
    df = read_csv(path)
    if df is None or df.empty or "epoch" not in df.columns:
        return pd.DataFrame(columns=["version", "epoch", "patients", "mean_dice", "std_dice",
                                     "mean_precision", "mean_recall", "threshold_note"])
    agg = df.groupby("epoch").agg(
        patients=("patient_id", "nunique"),
        mean_dice=("dice", "mean"),
        std_dice=("dice", "std"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
    ).reset_index()
    agg.insert(0, "version", version)
    agg["threshold_note"] = threshold_note
    return agg


def normalize_fullvol(label: str, path: Path) -> pd.DataFrame:
    data = read_json(path)
    if not data:
        return pd.DataFrame([{"version": label, "status": "summary missing"}])
    rows = []
    # Support the known summary schema while retaining original summary values.
    candidates = [
        ("PA", "pa_dice", "pa_precision", "pa_recall"),
        ("PV", "pv_dice", "pv_precision", "pv_recall"),
    ]
    for cls, dice_key, prec_key, rec_key in candidates:
        dice = summary_value(data, dice_key, f"{cls.lower()}_macro_dice",
                             f"macro_{cls.lower()}_dice")
        precision = summary_value(data, prec_key, f"{cls.lower()}_macro_precision",
                                  f"macro_{cls.lower()}_precision")
        recall = summary_value(data, rec_key, f"{cls.lower()}_macro_recall",
                               f"macro_{cls.lower()}_recall")
        if dice is not None or precision is not None or recall is not None:
            rows.append({"version": label, "class": cls, "dice": dice,
                         "precision": precision, "recall": recall,
                         "patients": summary_value(data, "patients_evaluated",
                                                   "test_patients_evaluated",
                                                   "validation_patients")})
    mean_fg = summary_value(data, "mean_foreground_dice", "macro_mean_foreground_dice",
                            "foreground_macro_dice")
    if mean_fg is not None:
        rows.append({"version": label, "class": "Mean foreground",
                     "dice": mean_fg, "precision": None, "recall": None,
                     "patients": summary_value(data, "patients_evaluated",
                                               "test_patients_evaluated",
                                               "validation_patients")})
    if not rows:
        rows.append({"version": label, "class": "Raw summary",
                     "summary_json": json.dumps(data, ensure_ascii=False)})
    return pd.DataFrame(rows)


def json_as_rows(path: Path, label: str) -> pd.DataFrame:
    data = read_json(path)
    if data is None:
        return pd.DataFrame([{"source": label, "status": "missing"}])
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append({"source": label, "field": key, "value": value})
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output: {OUTPUT}\n"
            "Rename or back up that file first, then rerun."
        )
    if not ROOT.joinpath("outputs").exists():
        raise SystemExit("Run this script from the NeuroVessel project root.")

    wb = Workbook()
    ws = wb.active
    ws.title = "Executive Summary"

    final_test_path = METRICS / "airrc_v6_1_final_test" / "airrc_v6_1_final_test_summary.json"
    final_test = read_json(final_test_path) or {}
    v6_path = METRICS / "airrc_v6_fullvol" / "airrc_v6_fullvol_summary.json"
    v61_path = METRICS / "airrc_v6_1_fullvol" / "airrc_v6_1_fullvol_summary.json"
    v62_path = METRICS / "airrc_v6_2_fullvol" / "airrc_v6_2_fullvol_summary.json"

    summary_rows = [
        ("Project", "NeuroVessel"),
        ("Current completed task", "Three-class pulmonary artery / pulmonary vein / background segmentation"),
        ("Primary model", "Residual 3D U-Net; V6.1"),
        ("Independent final test patients", 26),
        ("Final test PA Dice (patient macro)", 0.7864475946),
        ("Final test PA precision (patient macro)", 0.7440477591),
        ("Final test PA recall (patient macro)", 0.8352231696),
        ("Final test PV Dice (patient macro)", 0.7130373840),
        ("Final test PV precision (patient macro)", 0.6490702973),
        ("Final test PV recall (patient macro)", 0.7971713135),
        ("Final test mean foreground Dice (patient macro)", 0.7497424893),
        ("V6.1 best checkpoint epoch", summary_value(final_test, "checkpoint_epoch", default=29)),
        ("V6.1 best patch validation Dice", summary_value(final_test, "best_validation_patch_dice", default=0.8108644421)),
        ("Final test role", "Independent final evaluation; not used for model selection or tuning"),
        ("PH prediction", "Not implemented; no validated PH outcome labels established"),
        ("Foundation model integration", "Not implemented; current segmentation model is Residual 3D U-Net"),
        ("Clinical validation", "Not performed"),
        ("Historical workbook", str(HISTORICAL.relative_to(ROOT)) if HISTORICAL.exists() else "Not found"),
        ("Workbook generation", "Existing metrics only; no training or inference performed"),
    ]
    ws.append(["NeuroVessel — Final Research Results"])
    ws.merge_cells("A1:B1")
    ws["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="17365D")
    ws.append(["Metric", "Value"])
    for row in summary_rows:
        ws.append(list(row))
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:B{2 + len(summary_rows)}"

    # Historical PARSE epoch-by-epoch validation summaries.
    hist_parts = [
        epoch_summary(METRICS / "unet_v2_validation.csv", "PARSE V2", "Threshold as recorded in source CSV / experiment configuration"),
        epoch_summary(METRICS / "unet_v3_frangi_validation.csv", "PARSE V3 + Frangi", "Threshold as recorded in source CSV / experiment configuration"),
        epoch_summary(METRICS / "unet_v4_hard_negative_validation.csv", "PARSE V4 hard-negative", "Threshold 0.5 in recorded validation run"),
    ]
    historical_epochs = pd.concat(hist_parts, ignore_index=True)
    add_dataframe(wb, "PARSE Epoch Comparison", historical_epochs,
                  "Per-epoch patient-macro means calculated from per-patient validation CSVs. Do not compare as controlled architecture-only ablations where thresholds/configurations differ.")

    filter_df = read_csv(METRICS / "filter_baseline_PA000005_crop.csv")
    add_dataframe(wb, "Classical Filters", filter_df,
                  "Single PA000005 crop only; exploratory baseline, not a multi-patient benchmark.")

    v42_json = read_json(METRICS / "v42_connected_components" / "v42_summary.json")
    v42_csv = read_csv(METRICS / "v42_connected_components" / "v42_summary.csv")
    add_dataframe(wb, "PARSE V4.2 Ablation", v42_csv,
                  "15 validation patients. Threshold 0.75; selected component cutoff 5000 voxels. Test set was not evaluated.")

    # AirRC binary PA model and component ablation.
    v5b_eval = read_json(METRICS / "airrc_v5b_fullvol" / "evaluation_summary.json")
    add_dataframe(wb, "AirRC V5B Summary", json_as_rows(
        METRICS / "airrc_v5b_fullvol" / "evaluation_summary.json", "AirRC V5B full-volume validation"))
    v5b_ablation = read_csv(METRICS / "airrc_v5b_component_ablation" / "component_ablation_summary.csv")
    add_dataframe(wb, "V5B Component Ablation", v5b_ablation,
                  "Validation-only component filtering experiment; saved predictions unchanged and locked test partition not loaded.")

    # Three-class validation comparison, raw JSON field sheets, and detailed patient CSVs.
    comparison_parts = []
    for label, path in [("V6", v6_path), ("V6.1", v61_path), ("V6.2", v62_path)]:
        df = normalize_fullvol(label, path)
        comparison_parts.append(df)
        add_dataframe(wb, f"{label} Validation Detail", json_as_rows(path, f"{label} full-volume validation"))
    comparison = pd.concat(comparison_parts, ignore_index=True)
    add_dataframe(wb, "V6 Family Comparison", comparison,
                  "AirRC full-volume validation. V6.1 changes sampling and background weighting together; causal attribution to either change alone is not supported.")

    for label, folder, stem in [
        ("V6 Validation Patients", "airrc_v6_fullvol", "airrc_v6_fullvol_per_patient.csv"),
        ("V6.1 Validation Patients", "airrc_v6_1_fullvol", "airrc_v6_1_fullvol_per_patient.csv"),
        ("V6.2 Validation Patients", "airrc_v6_2_fullvol", "airrc_v6_2_fullvol_per_patient.csv"),
    ]:
        add_dataframe(wb, label, read_csv(METRICS / folder / stem))

    add_dataframe(wb, "V6.1 Final Test Summary",
                  json_as_rows(final_test_path, "V6.1 independent final test"),
                  "Locked 26-patient independent test. Do not use for model selection or tuning.")
    add_dataframe(wb, "V6.1 Final Test Patients",
                  read_csv(METRICS / "airrc_v6_1_final_test" / "airrc_v6_1_final_test_per_patient.csv"))
    add_dataframe(wb, "V6.1 Confusion Matrix",
                  read_csv(METRICS / "airrc_v6_1_final_test" / "airrc_v6_1_final_test_confusion_matrix.csv"))

    # Phase 2 analyses: retain source tables and explicit caveats.
    phase2_items = [
        ("Phase2 GT Comparison", METRICS / "phase2_gt_comparison" / "summary.csv",
         "Same V6.1 final-test predictions and labels; secondary analysis, not an independent test."),
        ("Phase2 GT Case Metrics", METRICS / "phase2_gt_comparison" / "case_metrics.csv", None),
        ("Phase2 Fragmentation", METRICS / "phase2_fp_fragmentation" / "summary.csv",
         "Digital connected-component diagnostics; components are not anatomical branches."),
        ("Phase2 Component Detail", METRICS / "phase2_fp_fragmentation" / "predicted_component_diagnostics.csv", None),
        ("Phase2 Case Diagnostics", METRICS / "phase2_fp_fragmentation" / "case_diagnostics.csv", None),
        ("Phase2 Topology", METRICS / "phase2_topology" / "phase2_topology_26cases.csv",
         "Exploratory descriptors of predicted masks; not validated anatomical topology."),
        ("Phase2 Topology Summary", METRICS / "phase2_topology" / "phase2_topology_summary.csv", None),
        ("Phase2 Graph Cases", METRICS / "phase2_graph" / "phase2_graph_26cases.csv",
         "Exploratory graph descriptors on predicted masks; not clinically/anatomically validated."),
        ("Phase2 Graph Branches", METRICS / "phase2_graph" / "phase2_graph_branches.csv", None),
        ("Phase2 Graph Summary", METRICS / "phase2_graph" / "phase2_graph_summary.csv", None),
        ("Phase2 Vascular Features", METRICS / "phase2_vascular_features_26cases.csv",
         "Predicted-mask vascular features; stored intensity is not asserted to be calibrated HU."),
        ("Vascular Feature Summary", METRICS / "phase2_vascular_feature_summary.csv", None),
    ]
    for sheet, path, note in phase2_items:
        add_dataframe(wb, sheet, read_csv(path), note)

    qc_frames = []
    for label, path in [
        ("GT comparison QC", METRICS / "phase2_gt_comparison" / "qc.json"),
        ("Fragmentation QC", METRICS / "phase2_fp_fragmentation" / "qc.json"),
    ]:
        qc_frames.append(json_as_rows(path, label))
    add_dataframe(wb, "Phase2 QC Records", pd.concat(qc_frames, ignore_index=True))

    # Methodology and limitations are deliberately explicit.
    methodology = pd.DataFrame([
        ("Primary task", "Three-class PA/PV/background segmentation on AirRC using Residual 3D U-Net."),
        ("Final model", "V6.1; best checkpoint epoch 29; ROI 96^3; sliding-window overlap 0.5; Gaussian blending."),
        ("Final test split", "26 independent test patients; final test used after model/settings were locked."),
        ("Final metrics", "Patient-macro Dice, precision and recall are headline metrics; voxel accuracy is background-dominated."),
        ("V6.1 vs V6", "Sampling and class weighting both changed; their individual causal effects cannot be separated."),
        ("V5B component ablation", "Validation-only; do not claim improvement on the locked test set."),
        ("PARSE historical comparison", "V2/V3/V4 threshold/configuration differences mean not a controlled architecture-only comparison."),
        ("Classical filters", "Single-crop exploratory baseline; not a full-dataset benchmark."),
        ("Topology/graph", "Predicted-mask digital descriptors; endpoints/junctions are not validated anatomical branch points."),
        ("Skeleton length", "Voxel-count × mean spacing is a coarse proxy, not a validated physical centreline measurement."),
        ("Connected components", "Digital components are not equivalent to anatomical branches."),
        ("Intensity", "Stored CT intensity; do not label as HU without independent calibration verification."),
        ("Confidence analysis", "Selected representative cases only; not a full 26-patient calibration/uncertainty study."),
        ("PH metadata", "Keyword matches are not validated PH outcomes; no PH prediction model was trained."),
        ("Foundation model", "Not implemented in the completed segmentation pipeline."),
        ("Clinical status", "Research-stage segmentation; no clinical validation or diagnostic use claim."),
        ("Reproducibility", "Checkpoint and split-manifest hashes are recorded in the V6.1 final-test summary JSON."),
    ], columns=["Topic", "Verified note"])
    add_dataframe(wb, "Methodology Limitations", methodology)

    # Consistent workbook formatting.
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="D9E2F3")

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if cell.row in (1, 2) and cell.value is not None:
                    if sheet.title == "Executive Summary" and cell.row == 1:
                        continue
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.border = Border(bottom=thin_gray)
        for col in sheet.columns:
            letter = get_column_letter(col[0].column)
            max_len = 0
            for cell in col[:250]:
                if cell.value is not None:
                    max_len = max(max_len, min(len(str(cell.value)), 70))
            sheet.column_dimensions[letter].width = max(12, min(max_len + 2, 42))
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.page_setup.orientation = "landscape"

    # Percent-format obvious metric columns where values are stored as fractions.
    for sheet_name in ["V6 Family Comparison"]:
        if sheet_name in wb.sheetnames:
            sh = wb[sheet_name]
            headers = {c.value: c.column for c in sh[1] if c.value}
            for key in ("dice", "precision", "recall"):
                if key in headers:
                    for r in range(2, sh.max_row + 1):
                        sh.cell(r, headers[key]).number_format = "0.00%"

    # Apply color scale to numeric Dice columns in key result sheets.
    for sheet_name in ["V6 Family Comparison", "V5B Component Ablation", "PARSE V4.2 Ablation"]:
        if sheet_name not in wb.sheetnames:
            continue
        sh = wb[sheet_name]
        headers = {c.value: c.column for c in sh[1] if c.value}
        dice_col = headers.get("dice") or headers.get("mean_dice")
        if dice_col:
            col = get_column_letter(dice_col)
            sh.conditional_formatting.add(
                f"{col}2:{col}{sh.max_row}",
                ColorScaleRule(start_type="min", start_color="F4CCCC",
                               mid_type="percentile", mid_value=50, mid_color="FFF2CC",
                               end_type="max", end_color="D9EAD3")
            )

    wb.save(OUTPUT)

    # Re-open and verify workbook integrity and key sheets.
    check = load_workbook(OUTPUT, read_only=True, data_only=True)
    required = ["Executive Summary", "V6 Family Comparison", "V6.1 Final Test Patients",
                "Methodology Limitations", "Phase2 QC Records"]
    missing = [name for name in required if name not in check.sheetnames]
    if missing:
        raise RuntimeError(f"Workbook verification failed; missing sheets: {missing}")
    print(f"[SUCCESS] Created: {OUTPUT}")
    print(f"[VERIFY] Sheets: {len(check.sheetnames)}")
    print(f"[VERIFY] Required sheets present: {', '.join(required)}")
    print(f"[PRESERVED] Historical workbook untouched: {HISTORICAL.exists()}")
    print("[SAFETY] No training, inference, prediction edits, or checkpoint edits performed.")


if __name__ == "__main__":
    main()
