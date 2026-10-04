"""
generate_figure5.py
----------------------
Figure 5: qualitative reconstruction-oriented JSCC outputs at k=8,
across the SNR range {-5, 0, 10, 20}dB, matching the paper's caption.

Run:
    python generate_figure5.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import ReconstructionJSCC, train_reconstruction

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

K = 8
SNR_VALUES = [-5, 0, 10, 20]
EPOCHS = 80
LR = 0.05
N_EXAMPLES = 6
SEED = 0

digits = load_digits()
X_all = digits.data / 16.0
X_train, X_test, y_train, y_test = train_test_split(
    X_all, digits.target, test_size=0.2, random_state=SEED, stratify=digits.target)

examples_idx = np.arange(N_EXAMPLES)
originals = X_test[examples_idx]

reconstructions = {}
for snr in SNR_VALUES:
    print(f"Training ReconstructionJSCC at k={K}, SNR={snr}dB...")
    rng = np.random.RandomState(SEED)
    model = ReconstructionJSCC(64, K, rng)
    train_reconstruction(model, X_train, snr, rng, epochs=EPOCHS, lr=LR)
    recon = model.forward(originals, snr, rng)
    reconstructions[snr] = recon
    mse = np.mean((recon - originals) ** 2)
    print(f"  MSE: {mse:.4f}")

fig, axes = plt.subplots(len(SNR_VALUES) + 1, N_EXAMPLES, figsize=(N_EXAMPLES * 1.3, (len(SNR_VALUES) + 1) * 1.3))

for j in range(N_EXAMPLES):
    axes[0, j].imshow(originals[j].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
    axes[0, j].axis("off")
axes[0, 0].set_ylabel("Original", rotation=0, labelpad=30, fontsize=9)

for i, snr in enumerate(SNR_VALUES):
    for j in range(N_EXAMPLES):
        axes[i + 1, j].imshow(reconstructions[snr][j].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[i + 1, j].axis("off")
    axes[i + 1, 0].set_ylabel(f"SNR={snr}dB", rotation=0, labelpad=30, fontsize=9)

fig.suptitle(f"Reconstruction-oriented JSCC outputs at k={K}, across channel SNR", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure5_qualitative_reconstructions.png"), dpi=150)
print(f"\nSaved to {FIG_DIR}/figure5_qualitative_reconstructions.png")
