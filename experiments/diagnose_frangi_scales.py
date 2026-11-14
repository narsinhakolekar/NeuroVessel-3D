
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.filters import frangi

for patient in ["PA000005", "PA000314"]:
    nii = nib.load(
        f"data/raw/train/{patient}/image/{patient}.nii.gz"
    )

    image = nii.get_fdata(dtype=np.float32)
    spacing = nii.header.get_zooms()[:3]

    image = (np.clip(image, -1000, 500) + 1000) / 1500

    image = zoom(
        image,
        tuple(s / 1.0 for s in spacing),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)

    scales = [1.0, 2.0, 3.0]

    responses = [
        frangi(
            image,
            sigmas=(sigma,),
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=False,
            mode="reflect",
        ).astype(np.float32)
        for sigma in scales
    ]

    stack = np.stack(responses)
    combined = stack.max(axis=0)
    winner = stack.argmax(axis=0)

    print("\nPATIENT:", patient)
    print("ISOTROPIC SHAPE:", image.shape)

    for i, sigma in enumerate(scales):
        response = responses[i]

        print(
            f"\nSigma {sigma} mm"
            f"\nMean: {response.mean():.7f}"
            f"\nStd: {response.std():.7f}"
            f"\nNonzero: {(response > 0).mean():.5f}"
            f"\nP99: {np.percentile(response, 99):.6f}"
            f"\nP99.9: {np.percentile(response, 99.9):.6f}"
            f"\nMaximum: {response.max():.6f}"
            f"\nDominance: {(winner[combined > 0] == i).mean():.4f}"
        )

    print("\nCombined mean:", combined.mean())
    print("Combined maximum:", combined.max())
