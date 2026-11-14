
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path
from src.filters.classical_filters import (
    gaussian_filter_3d,
    median_filter_3d,
    frangi_vesselness_3d,
)


def main():
    image_path = Path(
        "data/raw/train/PA000005/image/PA000005.nii.gz"
    )

    ct_file = nib.load(str(image_path))

    # Load only a limited crop from the CT volume.
    x0, y0, z0 = 192, 192, 125
    x1, y1, z1 = 320, 320, 189

    ct = np.asarray(
        ct_file.dataobj[x0:x1, y0:y1, z0:z1],
        dtype=np.float32,
    )

    if ct.size == 0:
        raise ValueError("The selected CT crop is empty")

    print("Crop shape:", ct.shape)
    print("Original HU range:", float(ct.min()), float(ct.max()))

    # Standard exploratory CT window.
    # Convert Hounsfield units to a stable [0, 1] intensity range.
    ct_windowed = np.clip(ct, -1000, 500)
    ct_windowed = (ct_windowed + 1000) / 1500

    print("\nApplying classical filters...")

    gaussian = gaussian_filter_3d(ct_windowed, sigma=1.0)
    median = median_filter_3d(ct_windowed, size=3)

    vesselness = frangi_vesselness_3d(
        ct_windowed,
        sigmas=(1, 2, 3),
    )

    results = {
        "CT windowed": ct_windowed,
        "Gaussian": gaussian,
        "Median": median,
        "Frangi vesselness": vesselness,
    }

    for name, result in results.items():
        print(
            f"{name}: min={result.min():.4f}, "
            f"max={result.max():.4f}, "
            f"finite={np.isfinite(result).all()}"
        )

    # Show the same axial slice from each processed volume.
    slice_index = ct.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    for ax, (name, volume) in zip(axes, results.items()):
        ax.imshow(
            volume[:, :, slice_index].T,
            cmap="gray",
            origin="lower",
        )
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "PA000005_filter_comparison.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print("\nSaved:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
