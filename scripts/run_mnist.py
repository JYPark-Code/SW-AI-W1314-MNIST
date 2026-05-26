"""Quick CLI training entry point.

Run from project root:  python scripts/run_mnist.py
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

print("[1/4] Loading MNIST...")
(x_train, y_train), (x_test, y_test) = load_mnist(data_dir=str(ROOT / "data"))
print(f"      x_train={x_train.shape}, x_test={x_test.shape}")

print("[2/4] Building network (BN=True, Dropout=0.3, Adam lr=0.001)...")
model = NeuralNetwork(use_batchnorm=True, use_dropout=True, dropout_ratio=0.3)
optimizer = Adam(lr=0.001)

print("[3/4] Training 15 epochs @ batch=128...")
start = time.time()
history = train(model, optimizer, x_train, y_train, epochs=15, batch_size=128)
elapsed = time.time() - start

print("[4/4] Evaluating...")
train_acc, n_params = evaluate(model, x_train, y_train)
test_acc, _ = evaluate(model, x_test, y_test)

print()
print(f"Elapsed: {elapsed:.1f}s ({elapsed / 15:.1f}s per epoch)")
print(f"Params:  {n_params:,}")
print("Loss curve:")
for i, l in enumerate(history, 1):
    print(f"  epoch {i:2d}: {l:.4f}")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test  accuracy: {test_acc:.2f}%")
