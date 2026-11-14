
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


DATA_ROOT = Path("data/raw/train")
OUTPUT_PATH = Path("data/splits.json")
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    if not np.isclose(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0
    ):
        raise ValueError("Split ratios must add up to 1")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    cases = sorted(
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("PA")
    )

    if not cases:
        raise RuntimeError("No patient folders found")

    valid_cases = []

    print("Checking patient files...")

    for case_dir in cases:
        case_id = case_dir.name

        image_path = case_dir / "image" / f"{case_id}.nii.gz"
        label_path = case_dir / "label" / f"{case_id}.nii.gz"

        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"Missing image or label for {case_id}"
            )

        image = nib.load(str(image_path))
        label = nib.load(str(label_path))

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: "
                f"{image.shape} vs {label.shape}"
            )

        if not np.allclose(image.affine, label.affine, atol=1e-4):
            raise ValueError(
                f"Affine mismatch for {case_id}"
            )

        labels = np.asarray(label.dataobj)

        if not np.isfinite(labels).all():
            raise ValueError(
                f"Non-finite label values in {case_id}"
            )

        if not np.any(labels != 0):
            raise ValueError(
                f"No foreground labels found for {case_id}"
            )

        valid_cases.append(case_id)

    rng = random.Random(SEED)
    rng.shuffle(valid_cases)

    n = len(valid_cases)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = sorted(valid_cases[:n_train])
    val_ids = sorted(valid_cases[n_train:n_train + n_val])
    test_ids = sorted(valid_cases[n_train + n_val:])

    all_ids = train_ids + val_ids + test_ids

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Patient leakage detected across splits")

    result = {
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "source": str(DATA_ROOT),
        "note": (
            "Internal patient-level split. "
            "Not necessarily the official benchmark split."
        ),
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nPatient-level splits created")
    print("Total patients:", n)
    print("Training:", len(train_ids))
    print("Validation:", len(val_ids))
    print("Test:", len(test_ids))
    print("Random seed:", SEED)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
