
import numpy as np
import torch

from src.filters.classical_filters import (
    gaussian_filter_3d,
    median_filter_3d,
    frangi_vesselness_3d,
)


def main():
    rng = np.random.default_rng(42)

    volume = np.zeros((64, 64, 64), dtype=np.float32)

    z, y, x = np.indices(volume.shape)
    vessel = (x - 32) ** 2 + (y - 32) ** 2 <= 5 ** 2
    volume[vessel] = 1.0

    noisy_volume = volume + rng.normal(
        0, 0.1, size=volume.shape
    ).astype(np.float32)

    gaussian = gaussian_filter_3d(noisy_volume)
    median = median_filter_3d(noisy_volume)
    vesselness = frangi_vesselness_3d(
        noisy_volume,
        sigmas=(1, 2, 3),
    )

    print("Synthetic 3D volume:", volume.shape)
    print("PyTorch CUDA available:", torch.cuda.is_available())

    for name, result in [
        ("Gaussian", gaussian),
        ("Median", median),
        ("Frangi vesselness", vesselness),
    ]:
        print(
            f"{name}: shape={result.shape}, "
            f"min={result.min():.4f}, "
            f"max={result.max():.4f}, "
            f"finite={np.isfinite(result).all()}"
        )


if __name__ == "__main__":
    main()
