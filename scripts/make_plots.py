"""Generate presentation figures.

Outputs:
  figures/loss_curve.png         — Baseline vs v2 training-loss curve
  figures/baseline_vs_v2.png     — Final accuracy / overfitting-gap bars
  figures/test_acc_curve.png     — Per-epoch test accuracy (saturation evidence)

Run from project root:  python scripts/make_plots.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── 실측 결과 (scripts/run_experiment.py 출력) ─────────────────────────────
BEST_LOSS = [
    0.2577, 0.1219, 0.0907, 0.0737, 0.0630,
    0.0540, 0.0487, 0.0442, 0.0398, 0.0375,
    0.0257, 0.0213, 0.0195, 0.0175, 0.0154,
]
BASELINE_LOSS = [
    0.2884, 0.1281, 0.0951, 0.0763, 0.0670,
    0.0554, 0.0491, 0.0430, 0.0405, 0.0376,
    0.0351, 0.0307, 0.0306, 0.0270, 0.0252,
]
# Per-epoch test accuracy (recorded inside the train loop via eval_data).
BASELINE_TEST_ACC = [
    96.75, 97.27, 97.94, 97.97, 98.15,
    98.18, 98.13, 98.18, 98.27, 98.36,
    98.39, 98.31, 98.31, 98.27, 98.41,
]
BEST_TEST_ACC = [
    96.84, 97.69, 97.88, 97.78, 98.20,
    98.11, 98.25, 98.31, 98.25, 98.41,
    98.59, 98.60, 98.61, 98.58, 98.62,
]
LR_DECAY_EPOCH = 10  # epoch 10 이후 lr 1e-3 → 1e-4

# ── 그림 1: Loss curve (baseline vs best, lr decay 지점 강조) ──────────────
fig, ax = plt.subplots(figsize=(9, 5))
epochs = np.arange(1, 16)

ax.plot(epochs, BASELINE_LOSS, marker="o", linewidth=2,
        color="#888888", label="Baseline (lr=1e-3 fixed)")
ax.plot(epochs, BEST_LOSS, marker="s", linewidth=2,
        color="#1f77b4", label="v2 (lr step decay)")

ax.axvspan(LR_DECAY_EPOCH + 0.5, 15.5, alpha=0.10, color="#1f77b4",
           label="lr = 1e-4 phase")
ax.annotate(
    "lr 1e-3 → 1e-4\n(0.0375 → 0.0257, -31%)",
    xy=(11, BEST_LOSS[10]),
    xytext=(11.7, 0.10),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Mean training loss", fontsize=12)
ax.set_title("Training Loss Curve — Baseline vs v2", fontsize=13)
ax.set_xticks(epochs)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "loss_curve.png", dpi=140)
plt.close(fig)
print(f"[OK] figures/loss_curve.png")

# ── 그림 2: Baseline vs v2 정확도 비교 ───────────────────────────────────
labels = ["Test acc", "Train acc", "Train/Test gap"]
baseline_vals = [98.41, 99.85, 1.44]
best_vals = [98.62, 99.95, 1.33]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

ax = axes[0]
x = np.arange(2)
width = 0.35
ax.bar(x - width / 2, [baseline_vals[0], baseline_vals[1]], width,
       label="Baseline", color="#888888")
ax.bar(x + width / 2, [best_vals[0], best_vals[1]], width,
       label="v2", color="#1f77b4")
for i, (b, m) in enumerate(zip([baseline_vals[0], baseline_vals[1]],
                                [best_vals[0], best_vals[1]])):
    ax.text(i - width / 2, b + 0.05, f"{b:.2f}%", ha="center", fontsize=9)
    ax.text(i + width / 2, m + 0.05, f"{m:.2f}%", ha="center",
            fontsize=9, fontweight="bold", color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(["Test accuracy", "Train accuracy"])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(98.0, 100.2)
ax.set_title("Accuracy: Baseline vs v2")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar([0, 1], [baseline_vals[2], best_vals[2]],
       color=["#888888", "#1f77b4"], width=0.5)
ax.text(0, baseline_vals[2] + 0.02, f"{baseline_vals[2]:.2f}%p",
        ha="center", fontsize=9)
ax.text(1, best_vals[2] + 0.02, f"{best_vals[2]:.2f}%p",
        ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Baseline", "v2"])
ax.set_ylabel("Train − Test (%p)")
ax.set_ylim(0, 2.0)
ax.set_title("Overfitting Gap (smaller = better)")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(FIG_DIR / "baseline_vs_v2.png", dpi=140)
plt.close(fig)
print(f"[OK] figures/baseline_vs_v2.png")

# ── 그림 3: Per-epoch test accuracy — saturation 증거 ──────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(epochs, BASELINE_TEST_ACC, marker="o", linewidth=2,
        color="#888888", label="Baseline (lr=1e-3 fixed)")
ax.plot(epochs, BEST_TEST_ACC, marker="s", linewidth=2,
        color="#1f77b4", label="v2 (lr step decay)")

ax.axvspan(LR_DECAY_EPOCH + 0.5, 15.5, alpha=0.10, color="#1f77b4",
           label="lr = 1e-4 phase")

# Baseline 후반 saturation 영역 강조 (epoch 11~15에서 98.27~98.41% 진동).
ax.axhspan(98.27, 98.41, xmin=(11 - 0.5) / 15, xmax=1.0,
           alpha=0.10, color="#888888")
ax.annotate(
    "Baseline saturated\n(98.27–98.41%, epoch 11–15)",
    xy=(13, 98.30),
    xytext=(8.5, 97.4),
    fontsize=9,
    color="#555555",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0),
)

# v2 lr decay 직후 점프 + 그 이후 saturation 강조.
ax.annotate(
    "v2 jumps to 98.59% after lr decay,\nthen saturates at 98.58–98.62%",
    xy=(13, 98.61),
    xytext=(2.5, 98.85),
    fontsize=9,
    color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Test accuracy (%)", fontsize=12)
ax.set_title("Per-epoch Test Accuracy — Both Models Saturate by Epoch ~13", fontsize=13)
ax.set_xticks(epochs)
ax.set_ylim(96.5, 99.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "test_acc_curve.png", dpi=140)
plt.close(fig)
print(f"[OK] figures/test_acc_curve.png")
