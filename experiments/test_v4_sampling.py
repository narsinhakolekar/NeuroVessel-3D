
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_unet_v4_hard_negative import (
    load_case_v4,
    sample_patch_v4,
    PATCH_SIZE,
    FOREGROUND_PROBABILITY,
    HARD_NEGATIVE_PROBABILITY,
    RANDOM_PROBABILITY,
)

patient_id = "PA000005"

print("Loading patient:", patient_id)

image, frangi, label, threshold = load_case_v4(patient_id)

print("CT shape:", image.shape)
print("Frangi shape:", frangi.shape)
print("Label shape:", label.shape)
print("Hard-negative threshold:", threshold)

rng = np.random.default_rng(42)

print("\nGenerating 20 sample patches...")

for i in range(20):

    x, y = sample_patch_v4(
        image,
        frangi,
        label,
        threshold,
        rng
    )

    assert x.shape == (2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    assert y.shape == (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    print(
        f"Patch {i+1:02d}: "
        f"input={x.shape}, "
        f"label_foreground={y.mean():.5f}, "
        f"Frangi_mean={x[1].mean():.5f}"
    )

print("\nSampling smoke test PASSED.")
print("No model training performed.")
print("Test patients were not accessed.")
