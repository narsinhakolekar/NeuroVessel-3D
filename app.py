"""
NeuroVessel V6.1 — interactive 3D PA/PV segmentation demo.

The app uses the selected V6.1 checkpoint and reproduces the final
evaluator's preprocessing and MONAI sliding-window inference settings.
Ground-truth comparison is enabled only for matching validation cases
or when a user explicitly supplies a matching label volume.
"""

import gc
import hashlib
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st
import torch
from monai.inferers import sliding_window_inference


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "checkpoints"
    / "airrc_v6_1_multiclass_best.pt"
)
SPLIT_CSV = PROJECT_ROOT / "outputs" / "metrics" / "airrc_v5_splits.csv"
LABEL_DIR = PROJECT_ROOT / "data" / "processed_airrc_v6" / "labels_multiclass"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from train_airrc_v6_multiclass import NeuroVesselV6  # noqa: E402


NUM_CLASSES = 3
WINDOW_LOW = -1000.0
WINDOW_HIGH = 1000.0
ROI_SIZE = 96
OVERLAP = 0.5
CLASS_NAMES = {0: "Background", 1: "PA", 2: "PV"}

st.set_page_config(
    page_title="NeuroVessel | PA/PV Segmentation",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 NeuroVessel")
st.subheader("3D Pulmonary Artery and Vein Segmentation")
st.caption("Research demonstration · Residual 3D U-Net V6.1 · AirRC-trained checkpoint")
st.warning(
    "Research use only. This application is not a medical device and must not "
    "be used for diagnosis, treatment, or clinical decisions. Predictions may "
    "contain false positives, missed vessels, and PA/PV class confusion."
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
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise KeyError("Checkpoint has no model_state_dict.")
    config = state.get("config", {})
    if config.get("output_classes", 3) != 3:
        raise ValueError("Checkpoint is not configured for 3 output classes.")
    epoch = state.get("epoch")
    if epoch != 29:
        raise ValueError(f"Expected V6.1 epoch 29, found {epoch!r}.")
    score = state.get("best_val_mean_foreground_patch_dice")
    if score is None or not np.isclose(
        float(score), 0.8108644421432855, atol=1e-7, rtol=0.0
    ):
        raise ValueError("Checkpoint does not match the selected V6.1 model.")
    model = NeuroVesselV6(in_channels=1, out_channels=3, base=8).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, device, {
        "epoch": int(epoch),
        "best_patch_dice": float(score),
        "sha256": sha256_file(checkpoint),
    }


def normalize_ct(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, WINDOW_LOW, WINDOW_HIGH).astype(np.float32, copy=False)
    return ((image - WINDOW_LOW) / (WINDOW_HIGH - WINDOW_LOW)).astype(
        np.float32, copy=False
    )


def run_inference(model, image: np.ndarray, device: torch.device):
    if image.ndim != 3:
        raise ValueError(f"Expected 3D CT (Z,Y,X); received {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("CT contains non-finite voxel values.")
    volume = torch.from_numpy(normalize_ct(image)).unsqueeze(0).unsqueeze(0)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
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
        raise RuntimeError(f"Expected 3 output channels, got {logits.shape[1]}.")
    prediction = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
    del volume, logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


def geometry_matches(a, b):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def read_label_file(label_path: Path, reference_image):
    label_image = sitk.ReadImage(str(label_path))
    if not geometry_matches(reference_image, label_image):
        raise ValueError(
            "Ground-truth geometry does not match CT "
            "(size/spacing/origin/direction mismatch)."
        )
    labels = sitk.GetArrayFromImage(label_image).astype(np.uint8, copy=False)
    if not set(np.unique(labels).tolist()).issubset({0, 1, 2}):
        raise ValueError(f"Ground truth has unexpected labels: {np.unique(labels)}")
    return labels


def uid_from_ct_filename(filename: str):
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        stem = filename[:-7]
    elif lower.endswith(".nii"):
        stem = filename[:-4]
    else:
        return None
    if stem.endswith("_ct"):
        stem = stem[:-3]
    return stem


def matching_validation_label(uid: str):
    """Return matching GT only when UID is explicitly in validation split."""
    if not uid or not SPLIT_CSV.is_file():
        return None, "Validation manifest not found; no automatic GT lookup."
    split_df = pd.read_csv(SPLIT_CSV)
    required = {"series_instance_uid", "split"}
    if not required.issubset(split_df.columns):
        return None, "Validation manifest is missing required columns."
    rows = split_df[
        (split_df["series_instance_uid"].astype(str) == uid)
        & (split_df["split"].astype(str).str.lower() == "validation")
    ]
    if len(rows) != 1:
        return None, "This uploaded CT is not uniquely identified as a validation case."
    label_path = LABEL_DIR / f"{uid}_multiclass.nii.gz"
    if label_path.is_file():
        return label_path, "Matching validation ground truth found automatically."
    return None, f"Validation case found, but label file is missing: {label_path}"


def dice_precision_recall(pred, gt, class_id):
    p = pred == class_id
    g = gt == class_id
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return dice, precision, recall


def overlay_rgba(mask, class_id, color, alpha):
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask == class_id] = (*color, alpha)
    return rgba


def make_five_panel_figure(ct_slice, gt_slice, pred_slice, slice_index):
    """
    Five-panel layout in the style of the supplied reference figures.
    GT PA=green, GT PV=cyan; prediction PA=red, prediction PV=magenta.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), constrained_layout=True)
    ct_display = normalize_ct(ct_slice)
    titles = [
        "Original CT",
        "Ground Truth: PA + PV",
        "Prediction: PA + PV",
        "PA: GT / Prediction",
        "PV: GT / Prediction",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    axes[0].imshow(ct_display, cmap="gray", vmin=0, vmax=1)

    # Ground truth combined: PA green, PV cyan.
    axes[1].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(overlay_rgba(gt_slice, 1, (0.10, 1.0, 0.12), 0.75))
    axes[1].imshow(overlay_rgba(gt_slice, 2, (0.0, 0.85, 1.0), 0.75))

    # Prediction combined: PA red, PV magenta.
    axes[2].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(overlay_rgba(pred_slice, 1, (1.0, 0.12, 0.10), 0.75))
    axes[2].imshow(overlay_rgba(pred_slice, 2, (1.0, 0.0, 0.85), 0.75))

    # PA comparison: GT yellow, prediction red; overlap becomes orange-like.
    axes[3].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(overlay_rgba(gt_slice, 1, (1.0, 0.9, 0.0), 0.8))
    axes[3].imshow(overlay_rgba(pred_slice, 1, (1.0, 0.12, 0.10), 0.55))

    # PV comparison: GT cyan, prediction magenta.
    axes[4].imshow(ct_display, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(overlay_rgba(gt_slice, 2, (0.0, 0.85, 1.0), 0.8))
    axes[4].imshow(overlay_rgba(pred_slice, 2, (1.0, 0.0, 0.85), 0.55))

    fig.suptitle(
        f"NeuroVessel V6.1 — Axial slice {slice_index}",
        fontsize=13, fontweight="bold"
    )
    return fig


def make_prediction_only_figure(ct_slice, pred_slice, slice_index):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].imshow(normalize_ct(ct_slice), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original CT")
    axes[1].imshow(normalize_ct(ct_slice), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(overlay_rgba(pred_slice, 1, (1.0, 0.12, 0.10), 0.7))
    axes[1].imshow(overlay_rgba(pred_slice, 2, (1.0, 0.0, 0.85), 0.7))
    axes[1].set_title("Prediction: PA (red) + PV (magenta)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"NeuroVessel V6.1 — Axial slice {slice_index}")
    return fig


def make_3d_plot(prediction: np.ndarray, max_side: int = 160):
    from skimage.measure import marching_cubes
    import plotly.graph_objects as go

    stride = max(1, int(np.ceil(max(prediction.shape) / max_side)))
    sampled = prediction[::stride, ::stride, ::stride]
    fig = go.Figure()
    for class_id, name, color, opacity in [
        (1, "Pulmonary artery", "red", 0.82),
        (2, "Pulmonary vein", "magenta", 0.62),
    ]:
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
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=name, color=color, opacity=opacity,
            )
        )
    fig.update_layout(
        title="3D predicted vascular masks (display downsampled)",
        scene=dict(
            xaxis_title="X voxel", yaxis_title="Y voxel",
            zaxis_title="Z voxel", aspectmode="data"
        ),
        height=620, margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def save_prediction_nifti(prediction: np.ndarray, reference_image):
    output_image = sitk.GetImageFromArray(prediction.astype(np.uint8))
    output_image.CopyInformation(reference_image)
    temp = tempfile.NamedTemporaryFile(
        suffix="_neurovessel_v6_1_prediction.nii.gz", delete=False
    )
    path = Path(temp.name)
    temp.close()
    sitk.WriteImage(output_image, str(path), True)
    return path


# Sidebar
with st.sidebar:
    st.header("Model configuration")
    st.write("**Model:** Residual 3D U-Net V6.1")
    st.write("**Output:** Background / PA / PV")
    st.write(f"**Patch:** {ROI_SIZE}³")
    st.write(f"**Overlap:** {OVERLAP:.0%}")
    st.write("**Blending:** Gaussian")
    st.write("**CT window:** −1000 to +1000 HU")
    st.write("**Normalization:** [0, 1]")
    st.divider()
    st.caption(f"Checkpoint: `{CHECKPOINT_PATH.name}`")
    st.caption("Selected checkpoint: epoch 29")

if not CHECKPOINT_PATH.is_file():
    st.error(f"Selected V6.1 checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

device_name = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, device, model_info = load_model(str(CHECKPOINT_PATH), device_name)
except Exception as exc:
    st.exception(exc)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Inference device", "NVIDIA CUDA" if device.type == "cuda" else "CPU")
c2.metric("Selected checkpoint", f"Epoch {model_info['epoch']}")
c3.metric("Best validation patch Dice", f"{model_info['best_patch_dice'] * 100:.2f}%")
with st.expander("Checkpoint verification"):
    st.write("Checkpoint SHA-256")
    st.code(model_info["sha256"])
    st.caption("81.09% is a patch-level validation score, not full-volume test Dice.")

st.markdown("### 1. Upload a chest CT volume")
uploaded_file = st.file_uploader(
    "Upload a 3D NIfTI CT file (.nii or .nii.gz)",
    type=["nii", "gz"],
    help="Use a validation CT for the first demonstration.",
)
if uploaded_file is None:
    st.info("Upload a 3D NIfTI CT volume to run actual V6.1 inference.")
    st.stop()
if uploaded_file.size > 2 * 1024 * 1024 * 1024:
    st.error("File exceeds the 2 GB demo upload limit.")
    st.stop()

try:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        handle.write(uploaded_file.getbuffer())
        ct_path = Path(handle.name)
    reference_image = sitk.ReadImage(str(ct_path))
    ct_array = sitk.GetArrayFromImage(reference_image).astype(np.float32, copy=False)
    if ct_array.ndim != 3 or not np.isfinite(ct_array).all():
        raise ValueError(f"Expected finite 3D CT; got shape {ct_array.shape}.")
except Exception as exc:
    st.error(f"Could not read uploaded CT: {exc}")
    st.stop()

uid = uid_from_ct_filename(uploaded_file.name)
auto_gt_path, auto_gt_message = matching_validation_label(uid)
st.markdown("### 2. Ground-truth comparison")
if auto_gt_path:
    st.success(auto_gt_message)
    gt_array = None
    try:
        gt_array = read_label_file(auto_gt_path, reference_image)
    except Exception as exc:
        st.error(f"Automatic validation label could not be loaded: {exc}")
else:
    st.info(auto_gt_message)
    st.caption(
        "For a five-panel GT comparison, upload the matching multiclass "
        "ground-truth NIfTI. It must have exactly the same geometry as the CT."
    )
    gt_upload = st.file_uploader(
        "Optional matching ground-truth label (.nii or .nii.gz)",
        type=["nii", "gz"],
        key="gt_upload",
    )
    gt_array = None
    if gt_upload is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
                handle.write(gt_upload.getbuffer())
                gt_path = Path(handle.name)
            gt_image = sitk.ReadImage(str(gt_path))
            if not geometry_matches(reference_image, gt_image):
                raise ValueError("Ground-truth geometry does not match CT.")
            gt_array = sitk.GetArrayFromImage(gt_image).astype(np.uint8, copy=False)
            if not set(np.unique(gt_array).tolist()).issubset({0, 1, 2}):
                raise ValueError("Ground-truth labels must be 0, 1, 2.")
            st.success("Matching ground-truth label loaded and geometry verified.")
        except Exception as exc:
            st.error(f"Could not load ground truth: {exc}")
            gt_array = None

spacing_xyz = reference_image.GetSpacing()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volume shape (Z,Y,X)", " × ".join(map(str, ct_array.shape)))
m2.metric("Spacing (mm)", " × ".join(f"{v:.3g}" for v in spacing_xyz))
m3.metric("Voxel range", f"{ct_array.min():.0f} to {ct_array.max():.0f}")
m4.metric("File size", f"{uploaded_file.size / (1024**2):.1f} MB")

# Key predictions to uploaded content to avoid showing a previous scan's mask.
upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
if st.session_state.get("nv_upload_hash") != upload_hash:
    st.session_state.pop("nv_prediction", None)
    st.session_state["nv_upload_hash"] = upload_hash

if st.button("Run V6.1 segmentation", type="primary", use_container_width=True):
    try:
        with st.spinner(
            "Running full-volume 3D inference with 96³ windows and Gaussian blending..."
        ):
            st.session_state["nv_prediction"] = run_inference(model, ct_array, device)
    except Exception as exc:
        st.exception(exc)
        st.stop()

prediction = st.session_state.get("nv_prediction")
if prediction is None:
    st.info("Click **Run V6.1 segmentation** to generate the prediction.")
    st.stop()
if prediction.shape != ct_array.shape:
    st.error("Prediction shape does not match uploaded CT. Please rerun inference.")
    st.stop()

st.success("V6.1 full-volume inference completed.")
pa_voxels = int(np.count_nonzero(prediction == 1))
pv_voxels = int(np.count_nonzero(prediction == 2))
voxel_volume_mm3 = float(np.prod(spacing_xyz))
pa_volume_cm3 = pa_voxels * voxel_volume_mm3 / 1000.0
pv_volume_cm3 = pv_voxels * voxel_volume_mm3 / 1000.0

st.markdown("### 3. Segmentation measurements")
r1, r2, r3 = st.columns(3)
r1.metric("Predicted PA volume", f"{pa_volume_cm3:.2f} cm³")
r2.metric("Predicted PV volume", f"{pv_volume_cm3:.2f} cm³")
r3.metric("Predicted foreground voxels", f"{pa_voxels + pv_voxels:,}")
st.caption(
    "Volumes are computed from predicted masks and NIfTI spacing; they are "
    "not clinical measurements or pulmonary-hypertension risk estimates."
)

if gt_array is not None:
    if gt_array.shape != prediction.shape:
        st.error("Ground truth and prediction array shapes differ; comparison disabled.")
        gt_array = None
    else:
        st.markdown("### 4. Ground-truth quantitative comparison")
        rows = []
        for class_id, class_name in [(1, "PA"), (2, "PV")]:
            dice, precision, recall = dice_precision_recall(
                prediction, gt_array, class_id
            )
            rows.append({
                "Class": class_name,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall,
            })
        metrics_df = pd.DataFrame(rows)
        mean_dice = float(metrics_df["Dice"].mean())
        metric_cols = st.columns(3)
        metric_cols[0].metric("PA Dice", f"{rows[0]['Dice'] * 100:.2f}%")
        metric_cols[1].metric("PV Dice", f"{rows[1]['Dice'] * 100:.2f}%")
        metric_cols[2].metric("Mean foreground Dice", f"{mean_dice * 100:.2f}%")
        display_df = metrics_df.copy()
        for col in ["Dice", "Precision", "Recall"]:
            display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption(
            "These are full-volume metrics for this uploaded validation case, "
            "computed against its matching labels. They are not the patch-level "
            "checkpoint-selection score."
        )

st.markdown("### 5. Five-panel CT / ground-truth / prediction comparison")
max_z = ct_array.shape[0] - 1
slice_index = st.slider(
    "Axial slice",
    min_value=0,
    max_value=max_z,
    value=max_z // 2,
)
if gt_array is not None:
    fig = make_five_panel_figure(
        ct_array[slice_index], gt_array[slice_index],
        prediction[slice_index], slice_index
    )
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Color key: GT PA = green; GT PV = cyan; predicted PA = red; "
        "predicted PV = magenta. The PA/PV comparison panels overlay the "
        "ground truth and prediction for the selected class."
    )
else:
    fig = make_prediction_only_figure(
        ct_array[slice_index], prediction[slice_index], slice_index
    )
    st.pyplot(fig, use_container_width=True)
    st.info(
        "Ground truth is unavailable for this upload, so only CT and prediction "
        "are shown. Upload the matching label to enable all five panels and Dice."
    )
plt.close("all")

st.markdown("### 6. 3D vascular reconstruction")
try:
    from skimage.measure import marching_cubes  # noqa: F401
    import plotly.graph_objects as go  # noqa: F401
    with st.spinner("Preparing downsampled 3D surfaces..."):
        surface_fig = make_3d_plot(prediction)
    st.plotly_chart(surface_fig, use_container_width=True)
except Exception as exc:
    st.warning(f"3D surface display could not be generated: {exc}")

st.markdown("### 7. Download prediction")
try:
    output_path = save_prediction_nifti(prediction, reference_image)
    with output_path.open("rb") as handle:
        st.download_button(
            "Download PA/PV segmentation (.nii.gz)",
            data=handle.read(),
            file_name=(
                Path(uploaded_file.name).name.replace(".nii.gz", "").replace(".nii", "")
                + "_neurovessel_v6_1_prediction.nii.gz"
            ),
            mime="application/gzip",
            use_container_width=True,
        )
    st.caption(
        "Exported labels preserve CT spacing, origin, and direction: "
        "0 = background, 1 = PA, 2 = PV."
    )
except Exception as exc:
    st.error(f"Could not prepare NIfTI download: {exc}")

st.divider()
st.caption(
    "NeuroVessel V6.1 · Research prototype · No PH prediction, clinical "
    "diagnosis, or anatomical branch validation is performed."
)
