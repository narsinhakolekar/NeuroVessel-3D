"""
NeuroVessel V6.1 — interactive 3D PA/PV segmentation demo.

Uses the selected V6.1 checkpoint and the same preprocessing and
MONAI sliding-window inference settings as the final evaluator.
This app is a research demonstration, not a clinical diagnostic tool.
"""

import gc
import hashlib
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import streamlit as st
import torch
from monai.inferers import sliding_window_inference


# ---------------------------------------------------------------------
# Project paths and model import
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "airrc_v6_1_multiclass_best.pt"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from train_airrc_v6_multiclass import NeuroVesselV6  # noqa: E402


# Exact settings from evaluate_airrc_v6_1_final_test.py
NUM_CLASSES = 3
WINDOW_LOW = -1000.0
WINDOW_HIGH = 1000.0
ROI_SIZE = 96
OVERLAP = 0.5
CLASS_LABELS = {
    0: "Background",
    1: "Pulmonary artery (PA)",
    2: "Pulmonary vein (PV)",
}


st.set_page_config(
    page_title="NeuroVessel | PA/PV Segmentation",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 NeuroVessel")
st.subheader("3D Pulmonary Artery and Vein Segmentation")
st.caption(
    "Research demonstration · Residual 3D U-Net V6.1 · "
    "AirRC-trained checkpoint"
)

st.warning(
    "Research use only. This application is not a medical device and "
    "must not be used for diagnosis, treatment, or clinical decisions. "
    "Predictions may contain false positives, missed vessels, and "
    "PA/PV class confusion."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Loading the selected V6.1 model...")
def load_model(checkpoint_string: str, device_string: str):
    checkpoint = Path(checkpoint_string)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(device_string)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in state:
        raise KeyError("Checkpoint has no model_state_dict.")

    config = state.get("config", {})
    if config.get("output_classes", 3) != 3:
        raise ValueError("Checkpoint is not configured for 3 output classes.")

    epoch = state.get("epoch")
    if epoch != 29:
        raise ValueError(
            f"Expected selected V6.1 epoch 29, but checkpoint says {epoch!r}."
        )

    score = state.get("best_val_mean_foreground_patch_dice")
    if score is None or not np.isclose(
        float(score), 0.8108644421432855, atol=1e-7, rtol=0.0
    ):
        raise ValueError("Checkpoint does not match the selected V6.1 model.")

    model = NeuroVesselV6(
        in_channels=1,
        out_channels=3,
        base=8,
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    metadata = {
        "epoch": int(epoch),
        "best_patch_dice": float(score),
        "sha256": sha256_file(checkpoint),
    }
    return model, device, metadata


def normalize_ct(image: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        image, WINDOW_LOW, WINDOW_HIGH
    ).astype(np.float32, copy=False)
    normalized = (normalized - WINDOW_LOW) / (
        WINDOW_HIGH - WINDOW_LOW
    )
    return normalized.astype(np.float32, copy=False)


def run_inference(model, image: np.ndarray, device: torch.device):
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D CT volume (Z,Y,X); received shape {image.shape}."
        )
    if not np.isfinite(image).all():
        raise ValueError("The uploaded CT contains non-finite voxel values.")

    normalized = normalize_ct(image)
    volume = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    amp_enabled = device.type == "cuda"

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = sliding_window_inference(
                inputs=volume,
                roi_size=(ROI_SIZE, ROI_SIZE, ROI_SIZE),
                sw_batch_size=1,
                predictor=model,
                overlap=OVERLAP,
                mode="gaussian",
                sw_device=device,
                device=torch.device("cpu"),
                progress=False,
            )

    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected 3 output channels, got {logits.shape[1]}."
        )

    prediction = (
        logits.argmax(dim=1)
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
    )
    del volume, logits, normalized
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


def make_overlay(ct_slice: np.ndarray, pred_slice: np.ndarray, alpha: float):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    # PA red; PV cyan. Background remains transparent.
    overlay = np.zeros((*pred_slice.shape, 4), dtype=np.float32)
    overlay[pred_slice == 1] = (1.0, 0.12, 0.12, alpha)
    overlay[pred_slice == 2] = (0.05, 0.85, 1.0, alpha)
    ax.imshow(overlay)
    ax.set_title("CT with predicted PA (red) and PV (cyan)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_3d_plot(prediction: np.ndarray, max_side: int = 160):
    # Downsample only for display; the saved segmentation remains full resolution.
    from skimage.measure import marching_cubes
    import plotly.graph_objects as go

    stride = max(1, int(np.ceil(max(prediction.shape) / max_side)))
    sampled = prediction[::stride, ::stride, ::stride]
    fig = go.Figure()

    surfaces = [
        (1, "Pulmonary artery", "red", 0.85),
        (2, "Pulmonary vein", "cyan", 0.65),
    ]
    added = 0
    for class_id, name, color, opacity in surfaces:
        binary = (sampled == class_id).astype(np.uint8)
        if int(binary.sum()) < 8:
            continue
        try:
            vertices, faces, _, _ = marching_cubes(binary, level=0.5)
        except (ValueError, RuntimeError):
            continue
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 2] * stride,
                y=vertices[:, 1] * stride,
                z=vertices[:, 0] * stride,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=name,
                color=color,
                opacity=opacity,
            )
        )
        added += 1

    fig.update_layout(
        title="3D predicted vascular masks (display downsampled)",
        scene=dict(
            xaxis_title="X voxel",
            yaxis_title="Y voxel",
            zaxis_title="Z voxel",
            aspectmode="data",
        ),
        height=650,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig, added


def save_prediction_nifti(prediction: np.ndarray, reference_image):
    output_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    output_image.CopyInformation(reference_image)
    temp = tempfile.NamedTemporaryFile(
        suffix="_neurovessel_v6_1_prediction.nii.gz",
        delete=False,
    )
    temp_path = Path(temp.name)
    temp.close()
    sitk.WriteImage(output_image, str(temp_path), True)
    return temp_path


# ---------------------------------------------------------------------
# Sidebar: checkpoint and runtime details
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Model configuration")
    st.write("**Model:** Residual 3D U-Net V6.1")
    st.write("**Output classes:** Background / PA / PV")
    st.write(f"**Patch:** {ROI_SIZE}³")
    st.write(f"**Overlap:** {OVERLAP:.0%}")
    st.write("**Blending:** Gaussian")
    st.write("**CT window:** −1000 to +1000 HU")
    st.write("**Normalization:** [0, 1]")
    st.divider()
    st.caption(f"Checkpoint: `{CHECKPOINT_PATH.name}`")
    st.caption("Selected validation checkpoint: epoch 29")


if not CHECKPOINT_PATH.is_file():
    st.error(
        "The selected V6.1 checkpoint was not found. Expected location:\n\n"
        f"`{CHECKPOINT_PATH}`"
    )
    st.stop()

device_name = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, device, model_info = load_model(
        str(CHECKPOINT_PATH), device_name
    )
except Exception as exc:
    st.exception(exc)
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Inference device", "NVIDIA CUDA" if device.type == "cuda" else "CPU")
col2.metric("Selected checkpoint", f"Epoch {model_info['epoch']}")
col3.metric("Best validation patch Dice", f"{model_info['best_patch_dice'] * 100:.2f}%")

with st.expander("Checkpoint verification"):
    st.write("SHA-256")
    st.code(model_info["sha256"])
    st.caption(
        "The validation patch Dice shown here is not the independent "
        "full-volume test Dice."
    )


st.markdown("### 1. Upload a chest CT volume")
uploaded_file = st.file_uploader(
    "Upload a 3D NIfTI CT file (.nii or .nii.gz)",
    type=["nii", "gz"],
    help="Upload a 3D CT volume in NIfTI format. This demo expects CT voxel "
         "values compatible with the training HU window.",
)

if uploaded_file is None:
    st.info(
        "Upload a 3D NIfTI CT volume to run actual V6.1 inference. "
        "The application does not generate simulated predictions."
    )
    st.stop()

if not (
    uploaded_file.name.lower().endswith(".nii")
    or uploaded_file.name.lower().endswith(".nii.gz")
):
    st.error("Please upload a .nii or .nii.gz NIfTI file.")
    st.stop()

if uploaded_file.size > 2 * 1024 * 1024 * 1024:
    st.error("File is larger than the 2 GB demo upload limit.")
    st.stop()

try:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        handle.write(uploaded_file.getbuffer())
        input_path = Path(handle.name)

    reference_image = sitk.ReadImage(str(input_path))
    ct_array = sitk.GetArrayFromImage(reference_image).astype(
        np.float32, copy=False
    )
    if ct_array.ndim != 3:
        raise ValueError(
            f"Uploaded image is not 3D. Array shape: {ct_array.shape}"
        )
    if not np.isfinite(ct_array).all():
        raise ValueError("Uploaded CT contains NaN or infinite values.")
except Exception as exc:
    st.error(f"Could not read the uploaded NIfTI CT: {exc}")
    st.stop()

spacing_xyz = reference_image.GetSpacing()
origin_xyz = reference_image.GetOrigin()
direction = reference_image.GetDirection()

st.markdown("### 2. CT volume information")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volume shape (Z, Y, X)", " × ".join(map(str, ct_array.shape)))
m2.metric("Spacing (mm)", " × ".join(f"{v:.3g}" for v in spacing_xyz))
m3.metric("Voxel range", f"{np.min(ct_array):.0f} to {np.max(ct_array):.0f}")
m4.metric("File size", f"{uploaded_file.size / (1024**2):.1f} MB")

if "nv_prediction" not in st.session_state or st.button(
    "Run V6.1 segmentation", type="primary", use_container_width=True
):
    try:
        with st.spinner(
            "Running full-volume 3D segmentation. This may take several minutes..."
        ):
            prediction = run_inference(model, ct_array, device)
        st.session_state["nv_prediction"] = prediction
        st.session_state["nv_input_name"] = uploaded_file.name
        st.session_state["nv_input_shape"] = tuple(ct_array.shape)
    except Exception as exc:
        st.exception(exc)
        st.stop()

prediction = st.session_state.get("nv_prediction")
if prediction is None:
    st.stop()

if tuple(prediction.shape) != tuple(ct_array.shape):
    st.error("Prediction shape does not match the uploaded CT. Please rerun inference.")
    st.stop()

st.success("V6.1 inference completed.")

# ---------------------------------------------------------------------
# Results and measurements
# ---------------------------------------------------------------------
st.markdown("### 3. Segmentation summary")
pa_voxels = int(np.count_nonzero(prediction == 1))
pv_voxels = int(np.count_nonzero(prediction == 2))
voxel_volume_mm3 = float(np.prod(spacing_xyz))
pa_volume_cm3 = pa_voxels * voxel_volume_mm3 / 1000.0
pv_volume_cm3 = pv_voxels * voxel_volume_mm3 / 1000.0

r1, r2, r3 = st.columns(3)
r1.metric("Predicted PA volume", f"{pa_volume_cm3:.2f} cm³")
r2.metric("Predicted PV volume", f"{pv_volume_cm3:.2f} cm³")
r3.metric("Predicted foreground voxels", f"{pa_voxels + pv_voxels:,}")

st.caption(
    "Volumes are computed from predicted voxel counts and the NIfTI voxel "
    "spacing. They are segmentation-derived measurements, not clinical "
    "measurements or pulmonary-hypertension risk estimates."
)

st.markdown("### 4. 2D slice viewer")
max_z = ct_array.shape[0] - 1
default_z = max_z // 2
slice_index = st.slider(
    "Axial slice (Z index)",
    min_value=0,
    max_value=max_z,
    value=default_z,
)
alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)

left, right = st.columns(2)
with left:
    st.image(
        ct_array[slice_index],
        caption=f"Original CT · axial slice {slice_index}",
        clamp=True,
        use_container_width=True,
    )
with right:
    fig = make_overlay(
        normalize_ct(ct_array[slice_index]),
        prediction[slice_index],
        alpha,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("### 5. 3D vascular reconstruction")
with st.spinner("Preparing downsampled 3D surface for display..."):
    try:
        surface_fig, surface_count = make_3d_plot(prediction)
        if surface_count:
            st.plotly_chart(surface_fig, use_container_width=True)
        else:
            st.info("No sufficiently large PA/PV surface was available to render.")
    except ImportError:
        st.info("3D display dependency is missing. Install it with: pip install scikit-image plotly")
    except Exception as exc:
        st.warning(f"3D surface display could not be generated: {exc}")

st.markdown("### 6. Download prediction")
try:
    output_path = save_prediction_nifti(prediction, reference_image)
    with output_path.open("rb") as handle:
        st.download_button(
            "Download PA/PV segmentation (.nii.gz)",
            data=handle.read(),
            file_name=(
                Path(uploaded_file.name).name.replace(".nii.gz", "")
                .replace(".nii", "")
                + "_neurovessel_v6_1_prediction.nii.gz"
            ),
            mime="application/gzip",
            use_container_width=True,
        )
    st.caption(
        "The exported label map preserves the uploaded CT's spacing, "
        "origin, and direction. Labels: 0 = background, 1 = PA, 2 = PV."
    )
except Exception as exc:
    st.error(f"Could not prepare the NIfTI download: {exc}")

st.divider()
st.caption(
    "NeuroVessel V6.1 · Research prototype · No PH prediction, "
    "clinical diagnosis, or anatomical branch validation is performed."
)
