import numpy as np
from train_unet_v3_frangi import load_case, sample_patch

patient_id = "PA000005"

print("Loading patient:", patient_id)
image, frangi, label = load_case(patient_id)

print("\nFull-volume verification:")
print("CT shape:", image.shape)
print("Frangi shape:", frangi.shape)
print("Label shape:", label.shape)

print("CT range:", float(image.min()), float(image.max()))
print("Frangi range:", float(frangi.min()), float(frangi.max()))
print("Label foreground voxels:", int(label.sum()))

assert image.shape == frangi.shape == label.shape
assert np.isfinite(image).all()
assert np.isfinite(frangi).all()
assert label.any()

rng = np.random.default_rng(42)
x_patch, y_patch = sample_patch(image, frangi, label, rng)

print("\nPatch verification:")
print("Input shape:", x_patch.shape)
print("Label shape:", y_patch.shape)
print("CT channel range:", float(x_patch[0].min()), float(x_patch[0].max()))
print("Frangi channel range:", float(x_patch[1].min()), float(x_patch[1].max()))

assert x_patch.shape == (2, 64, 64, 64)
assert y_patch.shape == (64, 64, 64)

print("\nV3 data-loading test PASSED.")