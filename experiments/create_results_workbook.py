
import csv
import json
import math
import statistics
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
OUTPUT_DIR = ROOT / "outputs" / "reports"
OUTPUT_FILE = OUTPUT_DIR / "NeuroVessel_Experimental_Results.xlsx"

FILES = {
    "v2": METRICS / "unet_v2_validation.csv",
    "v3": METRICS / "unet_v3_frangi_validation.csv",
    "v4": METRICS / "unet_v4_hard_negative_validation.csv",
    "v42": METRICS / "v42_connected_components" / "v42_summary.csv",
    "test": METRICS / "final_test_v4" / "final_test_patient_results.csv",
    "summary": METRICS / "final_test_v4" / "final_test_summary.json",
}


# ============================================================
# VALIDATE AND READ EXISTING FILES
# ============================================================

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def avg(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None

def std(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.stdev(values) if len(values) > 1 else 0

v2 = read_csv(FILES["v2"])
v3 = read_csv(FILES["v3"])
v4 = read_csv(FILES["v4"])
v42 = read_csv(FILES["v42"])
test = read_csv(FILES["test"])

with open(FILES["summary"], "r", encoding="utf-8") as f:
    test_summary = json.load(f)


# ============================================================
# WORKBOOK AND STYLING
# ============================================================

wb = Workbook()

NAVY = "17365D"
BLUE = "D9EAF7"
GREEN = "E2F0D9"
LIGHT_GRAY = "F2F2F2"
WHITE = "FFFFFF"

def setup_sheet(ws, widths=None):
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"

    if widths:
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

def write_table(ws, headers, rows):
    ws.append(headers)

    for row in rows:
        ws.append(row)

    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color=WHITE, bold=True)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    thin = Side(style="thin", color="D9E2F3")

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = Border(bottom=thin)
            cell.alignment = Alignment(vertical="center")

def format_columns(ws, columns):
    for col in columns:
        for row in range(2, ws.max_row + 1):
            ws.cell(row, col).number_format = "0.0000"


# ============================================================
# SHEET 1: EXECUTIVE SUMMARY
# ============================================================

ws = wb.active
ws.title = "Executive Summary"

ws.merge_cells("A1:F1")
ws["A1"] = "NEUROVESSEL | EXPERIMENTAL RESULTS"
ws["A1"].font = Font(size=19, bold=True, color=WHITE)
ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
ws["A1"].alignment = Alignment(vertical="center")
ws.row_dimensions[1].height = 40

ws.merge_cells("A2:F2")
ws["A2"] = (
    "Consolidated research results | Existing experiment records only | "
    "No model training or test rerun"
)
ws["A2"].font = Font(italic=True, color="666666")
ws["A2"].alignment = Alignment(wrap_text=True)

# Calculate macro and micro metrics from patient test results.
dice_values = [number(r["dice"]) for r in test]
precision_values = [number(r["precision"]) for r in test]
recall_values = [number(r["recall"]) for r in test]

tp = sum(number(r["tp"]) or 0 for r in test)
fp = sum(number(r["fp"]) or 0 for r in test)
fn = sum(number(r["fn"]) or 0 for r in test)

micro_dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0
micro_precision = tp / (tp + fp) if (tp + fp) else 0
micro_recall = tp / (tp + fn) if (tp + fn) else 0

summary_rows = [
    ["Final test macro Dice", avg(dice_values), "Mean patient-level Dice"],
    ["Final test macro precision", avg(precision_values), "Mean patient-level precision"],
    ["Final test macro recall", avg(recall_values), "Mean patient-level recall"],
    ["Final test Dice standard deviation", std(dice_values), "Variation across test patients"],
    ["Final test micro Dice", micro_dice, "Pooled voxel-level Dice"],
    ["Final test micro precision", micro_precision, "Pooled voxel-level precision"],
    ["Final test micro recall", micro_recall, "Pooled voxel-level recall"],
    ["Number of test patients", len(test), "Held-out patients evaluated once"],
]

write_table(ws, ["Metric", "Value", "Description"], summary_rows)

for r in range(2, 9):
    ws.cell(r, 2).number_format = "0.0000"
    ws.cell(r, 2).fill = PatternFill("solid", fgColor=GREEN)
    ws.cell(r, 2).font = Font(bold=True, color=NAVY)

ws["A12"] = "IMPORTANT RESEARCH NOTES"
ws["A12"].font = Font(size=13, bold=True, color=NAVY)

notes = [
    "V2, V3 and V4 epoch validation CSVs use threshold 0.5. V4.2 component analysis uses threshold 0.75.",
    "V4.2 threshold and component-size choices were explored using validation patients.",
    "The final held-out test configuration was fixed at threshold 0.75, minimum component size 5000 voxels and 26-connectivity.",
    "The held-out test evaluation has already been completed. Do not rerun it or use it for tuning.",
    "Low recall indicates that the model misses a substantial number of annotated artery voxels.",
    "Connected-component filtering can reduce false positives but may also remove genuine small artery branches.",
    "Component size is measured in voxels, not physical volume.",
    "The current implementation is a 3D U-Net pipeline, not yet a 3D foundation model.",
    "Pulmonary hypertension risk prediction has not yet been implemented.",
    "This is an experimental research project, not a clinically validated diagnostic system."
]

for i, note in enumerate(notes, 13):
    ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=6)
    cell = ws.cell(i, 1, "• " + note)
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[i].height = 32

for col, width in {"A": 36, "B": 20, "C": 55, "D": 16, "E": 16, "F": 16}.items():
    ws.column_dimensions[col].width = width

ws.freeze_panes = "A4"
ws.sheet_view.showGridLines = False


# ============================================================
# SHEET 2: EPOCH-WISE MODEL COMPARISON
# ============================================================

ws = wb.create_sheet("Epoch Comparison")

epoch_rows = []

for model_name, records in [
    ("V2 CT-only", v2),
    ("V3 CT + Frangi", v3),
    ("V4 Hard-negative", v4)
]:
    epochs = sorted({
        int(float(r["epoch"]))
        for r in records
        if r.get("epoch")
    })

    for epoch in epochs:
        group = [
            r for r in records
            if int(float(r["epoch"])) == epoch
        ]

        epoch_rows.append([
            model_name,
            epoch,
            avg([number(r["dice"]) for r in group]),
            std([number(r["dice"]) for r in group]),
            avg([number(r["precision"]) for r in group]),
            avg([number(r["recall"]) for r in group]),
            len(group)
        ])

write_table(
    ws,
    ["Model", "Epoch", "Mean Dice", "Std Dice",
     "Mean Precision", "Mean Recall", "Patients"],
    epoch_rows
)

setup_sheet(ws, {
    "A": 24, "B": 12, "C": 15, "D": 15,
    "E": 19, "F": 16, "G": 12
})
format_columns(ws, [3, 4, 5, 6])

ws.conditional_formatting.add(
    f"C2:C{ws.max_row}",
    ColorScaleRule(
        start_type="min", start_color="F8696B",
        mid_type="percentile", mid_value=50, mid_color="FFEB84",
        end_type="max", end_color="63BE7B"
    )
)

# Chart helper table
helper_col = 9
ws.cell(1, helper_col, "Epoch")
ws.cell(1, helper_col + 1, "V2")
ws.cell(1, helper_col + 2, "V3")
ws.cell(1, helper_col + 3, "V4")

for epoch in range(1, 9):
    ws.cell(epoch + 1, helper_col, epoch)

    for offset, model_name in enumerate(
        ["V2 CT-only", "V3 CT + Frangi", "V4 Hard-negative"], 1
    ):
        value = next(
            (r[2] for r in epoch_rows
             if r[0] == model_name and r[1] == epoch),
            None
        )
        ws.cell(epoch + 1, helper_col + offset, value)

chart = LineChart()
chart.title = "Validation Dice Across Epochs"
chart.y_axis.title = "Mean Dice"
chart.x_axis.title = "Epoch"
chart.style = 13
chart.height = 9
chart.width = 17

data = Reference(ws, min_col=helper_col + 1,
                 max_col=helper_col + 3, min_row=1, max_row=9)
categories = Reference(ws, min_col=helper_col,
                       min_row=2, max_row=9)

chart.add_data(data, titles_from_data=True)
chart.set_categories(categories)
ws.add_chart(chart, "I11")

for col in range(helper_col, helper_col + 4):
    ws.column_dimensions[get_column_letter(col)].hidden = True


# ============================================================
# SHEET 3: V4.2 COMPONENT FILTERING
# ============================================================

ws = wb.create_sheet("V4.2 Component Ablation")

headers = [
    "Minimum Component Size",
    "Mean Dice",
    "Std Dice",
    "Mean Precision",
    "Mean Recall",
    "Mean FP",
    "Mean FN",
    "Remaining Components",
    "Predicted Fraction",
    "Patients"
]

rows = []

for r in v42:
    rows.append([
        int(float(r["minimum_component_size"])),
        number(r["mean_dice"]),
        number(r["std_dice"]),
        number(r["mean_precision"]),
        number(r["mean_recall"]),
        number(r["mean_fp"]),
        number(r["mean_fn"]),
        number(r["mean_remaining_components"]),
        number(r["mean_predicted_fraction"]),
        int(float(r["patients"]))
    ])

write_table(ws, headers, rows)

setup_sheet(ws, {
    "A": 25, "B": 14, "C": 13, "D": 17, "E": 15,
    "F": 16, "G": 16, "H": 24, "I": 20, "J": 12
})

format_columns(ws, [2, 3, 4, 5, 9])

for r in range(2, ws.max_row + 1):
    for c in [6, 7, 8]:
        ws.cell(r, c).number_format = "#,##0.0"

chart = BarChart()
chart.title = "V4.2 Dice vs Minimum Component Size"
chart.y_axis.title = "Mean Dice"
chart.x_axis.title = "Minimum Component Size (voxels)"
chart.style = 10
chart.height = 9
chart.width = 18

chart.add_data(
    Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
    titles_from_data=True
)
chart.set_categories(
    Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
)
ws.add_chart(chart, "L2")


# ============================================================
# SHEET 4: FINAL TEST PATIENT RESULTS
# ============================================================

ws = wb.create_sheet("Final Test Patients")

test_headers = list(test[0].keys())
test_rows = []

for record in test:
    row = []

    for h in test_headers:
        value = record[h]

        if h != "patient_id":
            value = number(value)

            if value is None:
                value = record[h]

        row.append(value)

    test_rows.append(row)

write_table(ws, test_headers, test_rows)

widths = {}
for i, h in enumerate(test_headers, 1):
    widths[get_column_letter(i)] = 19 if h == "patient_id" else 17

setup_sheet(ws, widths)

for i, h in enumerate(test_headers, 1):
    col_letter = get_column_letter(i)

    if h in ["dice", "precision", "recall",
             "predicted_fraction", "target_fraction"]:
        for r in range(2, ws.max_row + 1):
            ws.cell(r, i).number_format = "0.0000"

    elif h in ["inference_seconds"]:
        for r in range(2, ws.max_row + 1):
            ws.cell(r, i).number_format = "0.00"

    elif h == "probability_threshold":
        for r in range(2, ws.max_row + 1):
            ws.cell(r, i).number_format = "0.00"

    if h in ["dice", "precision", "recall"]:
        ws.conditional_formatting.add(
            f"{col_letter}2:{col_letter}{ws.max_row}",
            ColorScaleRule(
                start_type="min", start_color="F8696B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="63BE7B"
            )
        )


# ============================================================
# SHEET 5: METHODOLOGY AND CONFIGURATION
# ============================================================

ws = wb.create_sheet("Methodology Notes")

methodology = [
    ["Dataset split",
     "100 patients: 70 training, 15 validation and 15 test. Patient-level split, seed 42."],

    ["V2 model",
     "Single-channel CT 3D BasicUNet. 64x64x64 patches. Five epochs."],

    ["V3 model",
     "Two-channel CT + Frangi 3D BasicUNet. 64x64x64 patches. Five epochs."],

    ["V4 model",
     "CT + Frangi. Hard-negative-aware sampling: 45% foreground, 30% high-Frangi annotated background and 25% random patches. Eight epochs."],

    ["V4.2 experiment",
     "V4 predictions with threshold 0.75 and 26-connectivity. Component-size sweep on validation patients."],

    ["Final test configuration",
     "V4 best checkpoint, threshold 0.75, minimum component size 5000 voxels and 26-connectivity."],

    ["Macro metrics",
     "Average of patient-level metrics. Every patient contributes equally."],

    ["Micro metrics",
     "Metrics calculated after pooling voxel counts across patients."],

    ["Threshold caveat",
     "V2, V3 and V4 epoch validation records use threshold 0.5. V4.2 uses threshold 0.75. These are not directly controlled comparisons."],

    ["Validation caveat",
     "Threshold and component-size selection used validation data. Validation-selected results are not independent performance estimates."],

    ["Anatomical caveat",
     "Component filtering may remove genuine small artery branches along with false-positive components."],

    ["Research scope",
     "Current project has a 3D U-Net segmentation pipeline. Foundation-model integration and pulmonary hypertension risk prediction remain future work."],

    ["Clinical limitation",
     "Experimental research prototype only. Clinical validity has not been established."]
]

write_table(ws, ["Topic", "Details"], methodology)

setup_sheet(ws, {"A": 28, "B": 115})

for r in range(2, ws.max_row + 1):
    ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[r].height = 42


# ============================================================
# FINAL FORMATTING AND SAVE
# ============================================================

for sheet in wb.worksheets:
    sheet.sheet_view.zoomScale = 90
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wb.save(OUTPUT_FILE)

# Verify workbook can be reopened.
check = load_workbook(OUTPUT_FILE, read_only=True)
sheet_names = check.sheetnames
check.close()

print("\n" + "=" * 55)
print("NEUROVESSEL WORKBOOK CREATED SUCCESSFULLY")
print("=" * 55)
print("Output:", OUTPUT_FILE)
print("Test patients:", len(test))
print("Sheets:", ", ".join(sheet_names))
print("Test macro Dice:", round(avg(dice_values), 4))
print("Test macro Precision:", round(avg(precision_values), 4))
print("Test macro Recall:", round(avg(recall_values), 4))
print("\nExisting experiment files were read only.")
print("No model was trained.")
print("No checkpoint was modified.")
print("Final test evaluation was NOT rerun.")
print("=" * 55)
