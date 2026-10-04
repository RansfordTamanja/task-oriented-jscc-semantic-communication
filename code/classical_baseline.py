"""
classical_baseline.py
------------------------
Classical separated source-channel coding (SSCC) baseline, the
"conventional" system JSCC is compared against throughout this
literature (Bourtsoulatze et al., 2019; Kurka & Gunduz, 2021): source
coding (compression) and channel transmission are designed and
optimized independently, with no awareness of the downstream task.

Source coding: PCA, the classical, non-learned analogue of the
transform coding (block-DCT) approach underlying real image codecs
such as JPEG (Wallace, 1992); both PCA and DCT are linear orthogonal
transforms that concentrate signal energy into a small number of
coefficients, which is precisely why transform coding is the basis of
classical compression. PCA is used here rather than a literal DCT/JPEG
implementation because it is the more direct, fair point of comparison
at the tiny (8x8) image scale and the very low bit budgets (k as low
as 4 coefficients) this study tests, where block-based JPEG's minimum
8x8 DCT block size and header overhead would dominate the comparison
rather than the underlying source-channel coding trade-off this paper
investigates.

Channel coding: NONE (uncoded transmission) is used as the default
comparison point, exactly as in the original JSCC papers' primary
comparison, since it isolates the source-coding side of the
separation theorem; a repetition-code variant (trading source rate
for channel redundancy within the same total bandwidth budget k) is
also implemented and tested separately as a secondary comparison.

Downstream task: a classifier is trained on CLEAN, uncorrupted
reconstructions and evaluated on noisy ones, reflecting a genuinely
separated design in which the classifier has no knowledge of, or
adaptation to, the channel.
"""
import numpy as np
from nn_layers import MLP, softmax, cross_entropy_loss, one_hot
from channel import normalize_power, awgn_channel


class PCACompressor:
    def __init__(self, k):
        self.k = k
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.k]
        return self

    def encode(self, X):
        return (X - self.mean) @ self.components.T

    def decode(self, Z):
        return Z @ self.components + self.mean


def transmit_uncoded(z, snr_db, rng):
    z_norm, scale = normalize_power(z)
    received_norm = awgn_channel(z_norm, snr_db, rng)
    return received_norm * scale


def transmit_repetition_coded(z, snr_db, rng, k_total):
    k_source = z.shape[1]
    r = max(1, k_total // k_source)
    z_norm, scale = normalize_power(z)
    repeated = np.repeat(z_norm, r, axis=1)
    received = awgn_channel(repeated, snr_db, rng)
    received = received.reshape(z.shape[0], k_source, r)
    averaged = received.mean(axis=2)
    return averaged * scale


def train_classifier_on_clean(X_train_recon, y_train, n_classes, rng, epochs=60, lr=0.1, batch_size=32, hidden=32):
    input_dim = X_train_recon.shape[1]
    model = MLP([input_dim, hidden, n_classes], rng, activations=["relu", "linear"])
    y_onehot = one_hot(y_train, n_classes)
    n = len(X_train_recon)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_recon[idx], y_onehot[idx]
            logits = model.forward(xb)
            probs = softmax(logits)
            d_logits = (probs - yb) / len(idx)
            model.backward(d_logits, lr)
    return model


def classify(model, X):
    logits = model.forward(X)
    probs = softmax(logits)
    return np.argmax(probs, axis=1)
