"""Generate presentation figures (loss curve, baseline vs best comparison).

Outputs:
  figures/loss_curve.png
  figures/baseline_vs_best.png

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
LR_DECAY_EPOCH = 10  # epoch 10 이후 lr 1e-3 → 1e-4

# ── 그림 1: Loss curve (baseline vs best, lr decay 지점 강조) ──────────────
fig, ax = plt.subplots(figsize=(9, 5))
epochs = np.arange(1, 16)

ax.plot(epochs, BASELINE_LOSS, marker="o", linewidth=2,
        color="#888888", label="Baseline (lr=1e-3 fixed)")
ax.plot(epochs, BEST_LOSS, marker="s", linewidth=2,
        color="#1f77b4", label="Best (lr step decay)")

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
ax.set_title("Training Loss Curve — Baseline vs Best", fontsize=13)
ax.set_xticks(epochs)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "loss_curve.png", dpi=140)
plt.close(fig)
print(f"[OK] figures/loss_curve.png")

# ── 그림 2: Baseline vs Best 정확도 비교 ───────────────────────────────────
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
       label="Best", color="#1f77b4")
for i, (b, m) in enumerate(zip([baseline_vals[0], baseline_vals[1]],
                                [best_vals[0], best_vals[1]])):
    ax.text(i - width / 2, b + 0.05, f"{b:.2f}%", ha="center", fontsize=9)
    ax.text(i + width / 2, m + 0.05, f"{m:.2f}%", ha="center",
            fontsize=9, fontweight="bold", color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(["Test accuracy", "Train accuracy"])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(98.0, 100.2)
ax.set_title("Accuracy: Baseline vs Best")
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
ax.set_xticklabels(["Baseline", "Best"])
ax.set_ylabel("Train − Test (%p)")
ax.set_ylim(0, 2.0)
ax.set_title("Overfitting Gap (smaller = better)")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(FIG_DIR / "baseline_vs_best.png", dpi=140)
plt.close(fig)
print(f"[OK] figures/baseline_vs_best.png")
