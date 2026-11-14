"""
NeuroVessel V6.1 training-curve generator.
Run from the NeuroVessel project root:
    python experiments/plot_airrc_v6_1_training_curves.py

This reads the existing CSV only; it does not train or modify the model.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs/metrics/airrc_v6_1_multiclass/airrc_v6_1_multiclass_history.csv"
OUT_DIR = ROOT / "outputs/visualizations/airrc_v6_1_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
required = ["epoch", "train_loss", "val_loss", "val_pa_patch_dice",
            "val_pv_patch_dice", "val_mean_foreground_patch_dice", "learning_rate"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

best = df.loc[df["val_mean_foreground_patch_dice"].idxmax()]
best_epoch = int(best["epoch"])
best_dice = float(best["val_mean_foreground_patch_dice"])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(df.epoch, df.train_loss, marker="o", markersize=3, label="Training loss")
ax.plot(df.epoch, df.val_loss, marker="s", markersize=3, label="Validation loss")
ax.set(title="NeuroVessel V6.1 — Training and Validation Loss",
       xlabel="Epoch", ylabel="Loss")
ax.set_xticks(df.epoch[::2])
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "v6_1_loss_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(df.epoch, df.val_pa_patch_dice * 100, marker="o", markersize=3, label="PA")
ax.plot(df.epoch, df.val_pv_patch_dice * 100, marker="s", markersize=3, label="PV")
ax.plot(df.epoch, df.val_mean_foreground_patch_dice * 100,
        marker="^", markersize=3, linewidth=2, label="Mean foreground")
ax.scatter([best_epoch], [best_dice * 100], marker="*", s=180, zorder=5,
           label=f"Best: epoch {best_epoch} ({best_dice*100:.2f}%)")
ax.set(title="NeuroVessel V6.1 — Validation Patch Dice",
       xlabel="Epoch", ylabel="Dice score (%)")
ax.set_ylim(0, 100)
ax.set_xticks(df.epoch[::2])
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "v6_1_validation_dice_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df.epoch, df.learning_rate, marker="o", markersize=3)
ax.set(title="NeuroVessel V6.1 — Learning Rate by Epoch",
       xlabel="Epoch", ylabel="Learning rate")
ax.set_xticks(df.epoch[::2])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "v6_1_learning_rate.png", dpi=300, bbox_inches="tight")
plt.close(fig)

summary = [
    "NeuroVessel V6.1 training history summary",
    f"Epochs recorded: {len(df)}",
    f"Best epoch: {best_epoch}",
    f"Best mean foreground patch Dice: {best_dice*100:.2f}%",
    f"PA patch Dice at best epoch: {float(best.val_pa_patch_dice)*100:.2f}%",
    f"PV patch Dice at best epoch: {float(best.val_pv_patch_dice)*100:.2f}%",
    f"Training loss at best epoch: {float(best.train_loss):.6f}",
    f"Validation loss at best epoch: {float(best.val_loss):.6f}",
    f"Final mean foreground patch Dice: {float(df.iloc[-1].val_mean_foreground_patch_dice)*100:.2f}%",
    "Note: these are patch-level validation metrics, not full-volume test metrics."
]
(OUT_DIR / "v6_1_training_summary.txt").write_text("\n".join(summary), encoding="utf-8")

print(f"Saved training graphs and summary in: {OUT_DIR}")
for item in sorted(OUT_DIR.iterdir()):
    print(item.name)
