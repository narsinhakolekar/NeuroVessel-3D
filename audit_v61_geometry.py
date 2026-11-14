
import sys
from pathlib import Path

sys.path.insert(0, "experiments")

import pandas as pd
import SimpleITK as sitk
import train_airrc_v5b as m

# Load the actual V6.1 split manifest
df = pd.read_csv(m.SPLIT_CSV)

split_col = m.find_column(
    df, ["split", "partition", "data_split", "dataset_split"]
)

uid_col = m.find_column(
    df,
    ["uid", "case_uid", "case_id", "seriesuid",
     "series_instance_uid", "patient_id", "case", "id"]
)

test_rows = df[
    df[split_col].map(m.normalize_split) == "test"
]

prediction_dir = Path(
    "outputs/predictions/airrc_v6_1_final_test"
)

print("\nV6.1 TEST GEOMETRY AUDIT")
print("=" * 110)

header = (
    "PATIENT | CT | PRED | SIZE | SPACING | ORIGIN | DIRECTION"
)
print(header)
print("-" * 110)

issues = 0
passed = 0

for _, row in test_rows.iterrows():

    uid = m.clean_string(row[uid_col])
    patient = row.get("patient_id", uid)

    ct_path = m.resolve_dataset_path(
        None, m.IMAGE_DIR, uid, "_ct.nii.gz"
    )

    pred_path = prediction_dir / (
        uid + "_pred_multiclass.nii.gz"
    )

    if not ct_path.is_file() or not pred_path.is_file():
        print(
            f"{patient} | {ct_path.is_file()} | "
            f"{pred_path.is_file()} | FILE MISSING"
        )
        issues += 1
        continue

    ct = sitk.ReadImage(str(ct_path))
    pred = sitk.ReadImage(str(pred_path))

    checks = [
        ct.GetSize() == pred.GetSize(),
        all(
            abs(a - b) < 1e-5
            for a, b in zip(
                ct.GetSpacing(), pred.GetSpacing()
            )
        ),
        all(
            abs(a - b) < 1e-4
            for a, b in zip(
                ct.GetOrigin(), pred.GetOrigin()
            )
        ),
        all(
            abs(a - b) < 1e-5
            for a, b in zip(
                ct.GetDirection(), pred.GetDirection()
            )
        ),
    ]

    print(
        f"{patient} | True | True | "
        + " | ".join(str(x) for x in checks)
    )

    if all(checks):
        passed += 1
    else:
        issues += 1

print("\n" + "=" * 110)
print("TOTAL TEST CASES:", len(test_rows))
print("GEOMETRY MATCHES:", passed)
print("FILE OR GEOMETRY ISSUES:", issues)
