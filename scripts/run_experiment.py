"""Best-combo experiment: wider hidden + Dropout 0.4 + LR step decay.

Run from project root:  python scripts/run_experiment.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam
from training import train, evaluate

np.random.seed(42)

HIDDEN = (1024, 512)
DROPOUT = 0.4
INIT = "he"
LR_HIGH = 0.001
LR_LOW = 0.0001
EPOCHS_HIGH = 10
EPOCHS_LOW = 5
BATCH = 128

print("[1/5] Loading MNIST...")
(x_train, y_train), (x_test, y_test) = load_mnist(data_dir=str(ROOT / "data"))
print(f"      x_train={x_train.shape}, x_test={x_test.shape}")

print(f"[2/5] Building network (hidden={HIDDEN}, init={INIT}, dropout={DROPOUT})...")
model = NeuralNetwork(
    use_batchnorm=True,
    use_dropout=True,
    dropout_ratio=DROPOUT,
    hidden_sizes=HIDDEN,
    init=INIT,
)
optimizer = Adam(lr=LR_HIGH)
n_params = sum(p.size for p in model.params.values())
print(f"      params={n_params:,}")

print(f"[3/5] Phase 1: {EPOCHS_HIGH} epochs @ lr={LR_HIGH}...")
start = time.time()
history_high = train(model, optimizer, x_train, y_train,
                     epochs=EPOCHS_HIGH, batch_size=BATCH)

print(f"[4/5] Phase 2: {EPOCHS_LOW} epochs @ lr={LR_LOW} (step decay)...")
optimizer.lr = LR_LOW
history_low = train(model, optimizer, x_train, y_train,
                    epochs=EPOCHS_LOW, batch_size=BATCH)
elapsed = time.time() - start
history = history_high + history_low

print("[5/5] Evaluating...")
train_acc, _ = evaluate(model, x_train, y_train)
test_acc, _ = evaluate(model, x_test, y_test)

total_epochs = EPOCHS_HIGH + EPOCHS_LOW
print()
print(f"Elapsed: {elapsed:.1f}s ({elapsed / total_epochs:.1f}s per epoch)")
print(f"Params:  {n_params:,}")
print("Loss curve:")
for i, l in enumerate(history, 1):
    tag = "(lr=1e-3)" if i <= EPOCHS_HIGH else "(lr=1e-4)"
    print(f"  epoch {i:2d}: {l:.4f}  {tag}")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test  accuracy: {test_acc:.2f}%")
print(f"Train/Test gap: {train_acc - test_acc:.2f}%p")
