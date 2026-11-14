
import numpy as np
from scipy import ndimage
from skimage.filters import frangi


def gaussian_filter_3d(volume, sigma=1.0):
    volume = np.asarray(volume, dtype=np.float32)
    return ndimage.gaussian_filter(volume, sigma=sigma)


def median_filter_3d(volume, size=3):
    volume = np.asarray(volume, dtype=np.float32)

    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    return ndimage.median_filter(volume, size=size)


def frangi_vesselness_3d(
    volume,
    sigmas=(1, 2, 3),
    alpha=0.5,
    beta=0.5,
    gamma=None,
):
    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError("Expected a 3D CT volume")

    if not np.isfinite(volume).all():
        raise ValueError("Volume contains NaN or infinite values")

    volume = volume - volume.min()
    max_value = volume.max()

    if max_value > 0:
        volume = volume / max_value

    response = frangi(
        volume,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
        mode="reflect",
    )

    return np.asarray(response, dtype=np.float32)
