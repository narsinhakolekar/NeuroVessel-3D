
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


DATA_ROOT = Path("data/raw/train")
OUTPUT_PATH = Path("outputs/metrics/parse_dataset_audit.csv")


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_ROOT}")

    cases = sorted(
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("PA")
    )

    if not cases:
        raise RuntimeError(f"No patient folders found in {DATA_ROOT}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "image_path",
        "label_path",
        "status",
        "shape",
        "spacing_mm",
        "orientation",
        "foreground_voxels",
        "foreground_fraction",
        "label_values",
        "message",
    ]

    valid = 0
    invalid = 0

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for case_dir in cases:
            case_name = case_dir.name
            image_dir = case_dir / "image"
            label_dir = case_dir / "label"

            image_path = image_dir / f"{case_name}.nii.gz"
            label_path = label_dir / f"{case_name}.nii.gz"

            row = {
                "case": case_name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "status": "ERROR",
                "shape": "",
                "spacing_mm": "",
                "orientation": "",
                "foreground_voxels": "",
                "foreground_fraction": "",
                "label_values": "",
                "message": "",
            }

            try:
                if not image_path.exists():
                    raise FileNotFoundError("CT image missing")

                if not label_path.exists():
                    raise FileNotFoundError("Label file missing")

                image = nib.load(str(image_path))
                label = nib.load(str(label_path))

                if image.shape != label.shape:
                    raise ValueError(
                        f"Shape mismatch: image={image.shape}, "
                        f"label={label.shape}"
                    )

                if not np.allclose(
                    image.affine, label.affine, atol=1e-4
                ):
                    raise ValueError("Image and label affines differ")

                labels = np.asarray(label.dataobj)

                if not np.isfinite(labels).all():
                    raise ValueError("Labels contain non-finite values")

                values, counts = np.unique(
                    labels, return_counts=True
                )

                foreground_voxels = int(
                    counts[values != 0].sum()
                )

                total_voxels = int(labels.size)

                row.update({
                    "status": "OK",
                    "shape": str(tuple(image.shape)),
                    "spacing_mm": str(
                        tuple(round(float(v), 5)
                              for v in image.header.get_zooms()[:3])
                    ),
                    "orientation": str(
                        nib.aff2axcodes(image.affine)
                    ),
                    "foreground_voxels": foreground_voxels,
                    "foreground_fraction": (
                        foreground_voxels / total_voxels
                    ),
                    "label_values": str(values.tolist()),
                    "message": "Geometry and label scan passed",
                })

                valid += 1

                print(
                    f"[OK] {case_name}: shape={image.shape}, "
                    f"foreground={foreground_voxels:,}"
                )

            except Exception as error:
                row["message"] = str(error)
                invalid += 1
                print(f"[ERROR] {case_name}: {error}")

            writer.writerow(row)

    print("\nDataset audit complete")
    print("Patient folders:", len(cases))
    print("Passed:", valid)
    print("Failed:", invalid)
    print("Report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
