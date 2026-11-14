
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader


SEED = 42
PATCH_SIZE = 64
BATCH_SIZE = 1
EPOCHS = 3
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-3

ROOT = Path("data/raw/train")
SPLIT_FILE = Path("data/splits.json")
OUTPUT_DIR = Path("outputs/checkpoints")
METRICS_DIR = Path("outputs/metrics")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PulmonaryArteryPatchDataset(Dataset):
    def __init__(self, patient_ids, patches_per_patient=4):
        self.patient_ids = patient_ids
        self.patches_per_patient = patches_per_patient
        self.records = []

        for patient_id in patient_ids:
            folder = ROOT / patient_id

            image_path = folder / "image" / f"{patient_id}.nii.gz"
            label_path = folder / "label" / f"{patient_id}.nii.gz"

            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing image or label for {patient_id}"
                )

            self.records.append((image_path, label_path, patient_id))

        self.samples = [
            record
            for record in self.records
            for _ in range(self.patches_per_patient)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path, patient_id = self.samples[index]

        image_file = nib.load(str(image_path))
        label_file = nib.load(str(label_path))

        image = np.asarray(image_file.dataobj, dtype=np.float32)
        label = np.asarray(label_file.dataobj) != 0

        if image.shape != label.shape:
            raise ValueError(f"Image/label shape mismatch: {patient_id}")

        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite CT values: {patient_id}")

        # Exploratory HU windowing.
        image = np.clip(image, -1000, 500)
        image = (image + 1000.0) / 1500.0

        sx, sy, sz = image.shape
        p = PATCH_SIZE

        if min(sx, sy, sz) < p:
            raise ValueError(
                f"{patient_id} is smaller than patch size {p}"
            )

        # Half the time, centre a patch near a foreground voxel.
        rng = np.random.default_rng(
            SEED + index + random.randint(0, 1_000_000)
        )

        foreground = np.argwhere(label)

        if foreground.size and rng.random() < 0.5:
            cx, cy, cz = foreground[
                rng.integers(0, len(foreground))
            ]

            x = int(np.clip(cx - p // 2, 0, sx - p))
            y = int(np.clip(cy - p // 2, 0, sy - p))
            z = int(np.clip(cz - p // 2, 0, sz - p))
        else:
            x = int(rng.integers(0, sx - p + 1))
            y = int(rng.integers(0, sy - p + 1))
            z = int(rng.integers(0, sz - p + 1))

        image_patch = image[
            x:x + p, y:y + p, z:z + p
        ].copy()

        label_patch = label[
            x:x + p, y:y + p, z:z + p
        ].astype(np.float32, copy=True)

        # Conv3d expects channel-first tensors.
        image_tensor = torch.from_numpy(
            image_patch[None, ...]
        )
        label_tensor = torch.from_numpy(
            label_patch[None, ...]
        )

        return image_tensor, label_tensor


def calculate_dice(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()

    if denominator == 0:
        return float("nan")

    return 2.0 * intersection / denominator


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the neurovessel environment "
            "and verify your PyTorch installation."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    train_ids = split["train"]
    val_ids = split["validation"]
    test_ids = split["test"]

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Train/validation patient leakage")

    if set(train_ids) & set(test_ids):
        raise RuntimeError("Train/test patient leakage")

    if set(val_ids) & set(test_ids):
        raise RuntimeError("Validation/test patient leakage")

    print(
        f"Patients: train={len(train_ids)}, "
        f"validation={len(val_ids)}, test={len(test_ids)}"
    )

    train_dataset = PulmonaryArteryPatchDataset(
        train_ids,
        patches_per_patient=4,
    )

    # Validation patches are fixed by seed and are not used for gradients.
    val_dataset = PulmonaryArteryPatchDataset(
        val_ids,
        patches_per_patient=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss()

    amp_enabled = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        print(f"\nEpoch {epoch}/{EPOCHS}")

        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(image)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps += 1

            if steps >= STEPS_PER_EPOCH:
                break

            if steps % 20 == 0:
                print(
                    f"  Step {steps}/{STEPS_PER_EPOCH}, "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / max(steps, 1)

        model.eval()
        dice_values = []

        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(image)

                prediction = torch.sigmoid(logits) >= 0.5
                dice = calculate_dice(prediction, label)

                if np.isfinite(dice):
                    dice_values.append(dice)

        val_dice = (
            float(np.mean(dice_values))
            if dice_values else float("nan")
        )

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation patch Dice: {val_dice:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_patch_dice": val_dice,
        })

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_patch_dice": val_dice,
            "seed": SEED,
            "patch_size": PATCH_SIZE,
        }

        torch.save(
            checkpoint,
            OUTPUT_DIR / "unet_baseline_latest.pt",
        )

        if np.isfinite(val_dice) and val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                checkpoint,
                OUTPUT_DIR / "unet_baseline_best.pt",
            )

        (METRICS_DIR / "unet_baseline_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print("Checkpoint saved.")

    print("\nTraining smoke test complete.")
    print("Best validation patch Dice:", best_val_dice)
    print("Checkpoints:", OUTPUT_DIR)
    print("History:", METRICS_DIR / "unet_baseline_history.json")
    print(
        "The test patients were not used. "
        "Validation patch Dice is preliminary, not full-volume performance."
    )


if __name__ == "__main__":
    main()
