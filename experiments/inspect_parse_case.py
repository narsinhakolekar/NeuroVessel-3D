
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    root = Path("data/raw/train/PA000005")

    image_path = root / "image" / "PA000005.nii.gz"
    label_path = root / "label" / "PA000005.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"CT image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    ct_file = nib.load(str(image_path))
    label_file = nib.load(str(label_path))

    print("CT shape:", ct_file.shape)
    print("Label shape:", label_file.shape)
    print("CT voxel spacing:", ct_file.header.get_zooms()[:3])
    print("Label voxel spacing:", label_file.header.get_zooms()[:3])
    print("CT orientation:", nib.aff2axcodes(ct_file.affine))
    print("Label orientation:", nib.aff2axcodes(label_file.affine))

    if ct_file.shape != label_file.shape:
        raise ValueError(
            "CT and label shapes differ. Investigate alignment "
            "before visualization."
        )

    if not np.allclose(ct_file.affine, label_file.affine, atol=1e-4):
        raise ValueError(
            "CT and label affines differ. Investigate alignment "
            "before overlaying the mask."
        )

    ct = np.asarray(ct_file.dataobj, dtype=np.float32)
    labels = np.asarray(label_file.dataobj)

    if not np.isfinite(ct).all():
        raise ValueError("CT contains NaN or infinite values")

    label_values, label_counts = np.unique(
        labels, return_counts=True
    )

    print("\nFull-volume label counts:")
    for value, count in zip(label_values, label_counts):
        print(f"  Label {value}: {count:,} voxels")

    foreground = labels != 0
    counts_per_slice = foreground.sum(axis=(0, 1))
    slice_index = int(np.argmax(counts_per_slice))

    print("\nSelected slice:", slice_index)
    print("Foreground pixels:", int(counts_per_slice[slice_index]))
    print("CT intensity range:", float(ct.min()), float(ct.max()))

    ct_slice = ct[:, :, slice_index]
    mask_slice = foreground[:, :, slice_index]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ct_slice.T, cmap="gray", origin="lower",
                   vmin=-1000, vmax=500)
    axes[0].set_title("CT image")

    axes[1].imshow(mask_slice.T, cmap="gray", origin="lower")
    axes[1].set_title("Segmentation mask")

    axes[2].imshow(ct_slice.T, cmap="gray", origin="lower",
                   vmin=-1000, vmax=500)
    overlay = np.ma.masked_where(
        ~mask_slice.T, mask_slice.T
    )
    axes[2].imshow(
        overlay, cmap="autumn", alpha=0.7, origin="lower",
        vmin=0, vmax=1
    )
    axes[2].set_title("CT + mask overlay")

    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "PA000005_inspection.png"

    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print("\nSaved visualization:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
