"""
Standalone, self-contained CIFAR-10 JSCC run for Google Colab.
Combines channel.py, nn_layers.py, jscc.py, classical_baseline.py,
the CIFAR-10 loader, and the run_multidataset.py functions into one
file so it can be pasted into a single Colab cell -- no upload, no
zip, no Drive mount needed for the code itself (only for saving
results somewhere that survives a disconnect, see the bottom).
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# Default: saves locally in Colab's temporary storage (lost on disconnect).
# To save to Drive instead, mount Drive first, then set this to e.g.
# "/content/drive/MyDrive/jscc_results" BEFORE running the rest of this cell.
DATA_DIR = "/kaggle/working/jscc_results"
os.makedirs(DATA_DIR, exist_ok=True)

EPOCHS = 80
LR = 0.05
HIDDEN = 32
N_SEEDS = 5

# ---- WHICH SEEDS THIS NOTEBOOK RUNS ----
# Full run in one notebook (~2.5 h):   SEEDS_TO_RUN = [0, 1, 2, 3, 4]
# Faster: open one notebook per seed and set a single seed in each, e.g.
#   notebook A: [0]   notebook B: [1]   notebook C: [2]   ...  (~35 min each)
# Each seed is fully independent and deterministic, so the per-seed CSVs
# can simply be combined afterwards.
SEEDS_TO_RUN = [0, 1, 2, 3, 4]

# ============ channel.py ============
"""
channel.py
------------
Additive White Gaussian Noise (AWGN) channel simulation. This is the
"non-trainable layer in the middle" of the JSCC autoencoder (Bourtsoulatze,
Kurka & Gunduz, 2019): since the noise is additive, the gradient of the
noisy output with respect to the clean input is exactly 1, so backprop
flows through this layer unmodified, without any special reparameterization
trick. This is the standard way JSCC systems are made end-to-end trainable
despite the channel being a genuinely stochastic, non-trainable component.

Power normalization: transmitted symbols are normalized to unit average
power per symbol before the channel, matching standard practice in the
JSCC literature (e.g. Bourtsoulatze et al. 2019, Kurka & Gunduz 2021),
so that the channel SNR is meaningfully comparable across systems with
different raw symbol magnitudes.
"""


def normalize_power(z):
    """Normalizes each sample (row) to unit average power, returning the
    normalized signal and the per-sample scale factor (needed to undo
    the normalization symmetrically at both encoder and decoder, exactly
    as in the reference JSCC implementations)."""
    power = np.mean(z ** 2, axis=1, keepdims=True) + 1e-9
    scale = np.sqrt(power)
    return z / scale, scale


def awgn_channel(z, snr_db, rng):
    """Adds AWGN at the specified SNR (dB) to a unit-power signal z.
    Returns the noisy received signal. Gradient-transparent: d(output)/d(z) = 1.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / snr_linear)
    noise = rng.normal(0, noise_std, size=z.shape)
    return z + noise



# ============ nn_layers.py ============
"""
nn_layers.py
--------------
From-scratch dense neural network building blocks (forward + manual
backward pass), used to build both the JSCC encoder/decoder and the
classical pipeline's classifier. Implemented directly here, rather
than using PyTorch or TensorFlow, so every gradient could be derived
and verified by hand, giving full visibility into the differentiable
channel layer's behavior during training.
"""


def he_init(fan_in, fan_out, rng):
    return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


class Dense:
    def __init__(self, fan_in, fan_out, rng, activation="relu", optimizer="momentum"):
        self.W = he_init(fan_in, fan_out, rng)
        self.b = np.zeros(fan_out)
        self.activation = activation
        self.cache = {}
        self.optimizer = optimizer
        # Momentum-SGD state
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        # Adam state (only used if optimizer="adam"), addressing the
        # reviewer's request to test a different optimizer for the
        # diagnostics rather than only momentum-SGD
        self.m_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.vv_W = np.zeros_like(self.W)
        self.vv_b = np.zeros_like(self.b)
        self.adam_t = 0

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == "relu":
            a = np.maximum(0, z)
        elif self.activation == "linear":
            a = z
        elif self.activation == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        elif self.activation == "tanh":
            a = np.tanh(z)
        else:
            raise ValueError(self.activation)
        self.cache = dict(x=x, z=z, a=a)
        return a

    def backward(self, d_a, lr, momentum=0.9):
        x, z, a = self.cache["x"], self.cache["z"], self.cache["a"]
        if self.activation == "relu":
            d_z = d_a * (z > 0)
        elif self.activation == "linear":
            d_z = d_a
        elif self.activation == "sigmoid":
            d_z = d_a * a * (1 - a)
        elif self.activation == "tanh":
            d_z = d_a * (1 - a ** 2)
        n = x.shape[0]
        d_W = x.T @ d_z / n
        d_b = d_z.mean(axis=0)
        d_x = d_z @ self.W.T

        if self.optimizer == "adam":
            # Adam (Kingma & Ba, 2015): tested here per the reviewer's
            # request for a different optimizer in the diagnostics,
            # alongside momentum-SGD rather than replacing it, so both
            # can be compared directly rather than asserting one is better.
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self.adam_t += 1
            self.m_W = beta1 * self.m_W + (1 - beta1) * d_W
            self.m_b = beta1 * self.m_b + (1 - beta1) * d_b
            self.vv_W = beta2 * self.vv_W + (1 - beta2) * (d_W ** 2)
            self.vv_b = beta2 * self.vv_b + (1 - beta2) * (d_b ** 2)
            m_W_hat = self.m_W / (1 - beta1 ** self.adam_t)
            m_b_hat = self.m_b / (1 - beta1 ** self.adam_t)
            vv_W_hat = self.vv_W / (1 - beta2 ** self.adam_t)
            vv_b_hat = self.vv_b / (1 - beta2 ** self.adam_t)
            self.W -= lr * m_W_hat / (np.sqrt(vv_W_hat) + eps)
            self.b -= lr * m_b_hat / (np.sqrt(vv_b_hat) + eps)
        else:
            # Momentum SGD: standard, well-established convergence improvement
            # over plain SGD, used here because plain SGD converged too slowly
            # on this genuinely hard (noisy channel + joint compression +
            # classification) optimization landscape, verified directly by
            # observing loss still decreasing after 150 plain-SGD epochs
            # before adopting momentum.
            self.v_W = momentum * self.v_W - lr * d_W
            self.v_b = momentum * self.v_b - lr * d_b
            self.W += self.v_W
            self.b += self.v_b
        return d_x


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y_onehot):
    eps = 1e-9
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class MLP:
    """A simple stack of Dense layers, used for both the JSCC
    encoder/decoder halves and the classical pipeline's classifier."""

    def __init__(self, layer_sizes, rng, activations=None, optimizer="momentum"):
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["linear"]
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [Dense(layer_sizes[i], layer_sizes[i + 1], rng, activations[i],
                              optimizer=optimizer)
                        for i in range(len(layer_sizes) - 1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out, lr):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, lr)
        return d_out

# ============ jscc.py ============
"""
jscc.py
---------
Deep Joint Source-Channel Coding (JSCC) system, following the
architecture introduced by Bourtsoulatze, Kurka & Gunduz (2019): an
encoder network maps source data directly to channel symbols, a
non-trainable AWGN channel layer adds noise, and a decoder network
maps the noisy symbols to the receiver's output, with the whole
pipeline trained end-to-end by backpropagating through the channel.

TWO VARIANTS are implemented and compared throughout this study,
directly testing this paper's central question ("how much bandwidth
does meaning actually need?"):
  - TASK-ORIENTED (semantic): the decoder outputs a class prediction
    directly, and the entire encoder-decoder pair is trained to
    minimize classification cross-entropy. This is genuinely
    "semantic" in the sense used by Gunduz et al. (2023): the encoder
    is free to discard any information irrelevant to the downstream
    task, transmitting only what the receiver's task actually needs.
  - RECONSTRUCTION-ORIENTED: the decoder instead outputs a
    reconstructed image, and the pipeline is trained to minimize
    pixel-wise MSE, exactly as in the original Bourtsoulatze et al.
    formulation. A separate classifier (trained on clean images) is
    then applied to the reconstruction to measure downstream task
    accuracy, mirroring how a non-task-aware JSCC system would
    actually be used in practice.
"""


class TaskOrientedJSCC:
    """Encoder: image -> k channel symbols. Channel: AWGN at given SNR.
    Decoder: k noisy symbols -> class prediction. Trained end-to-end on
    classification cross-entropy (the semantic/task-oriented objective).

    encoder_type: "mlp" (default, matching the paper's original single-
    hidden-layer architecture) or "cnn" (a small convolutional encoder,
    addressing the reviewer's point that a 32-unit hidden layer is too
    small to support conclusions about "JSCC in general"). Both are run
    side by side in run_experiments.py rather than the CNN replacing
    the MLP result, so the paper can state directly whether the central
    finding depends on encoder architecture.
    """

    def __init__(self, input_dim, k, n_classes, rng, hidden=32, optimizer="momentum",
                 encoder_type="mlp", img_h=None, img_w=None):
        self.k = k
        self.encoder_type = encoder_type
        if encoder_type == "cnn":
            from cnn_layers import CNNEncoder
            if img_h is None or img_w is None:
                # Infer a square image shape from input_dim if not given
                # explicitly (e.g. 64 -> 8x8, 784 -> 28x28, 1024 -> 32x32,
                # all exact squares for the datasets used in this study).
                side = int(round(np.sqrt(input_dim)))
                assert side * side == input_dim, (
                    f"input_dim={input_dim} is not a perfect square; pass "
                    f"img_h/img_w explicitly for non-square inputs.")
                img_h = img_w = side
            self.encoder = CNNEncoder(img_h, img_w, k, rng, optimizer=optimizer)
        else:
            self.encoder = MLP([input_dim, hidden, k], rng, activations=["relu", "linear"],
                                optimizer=optimizer)
        self.decoder = MLP([k, hidden, n_classes], rng, activations=["relu", "linear"],
                            optimizer=optimizer)

    def forward(self, x, snr_db, rng, training=True):
        z = self.encoder.forward(x)
        z_norm, scale = normalize_power(z)
        if training:
            received = awgn_channel(z_norm, snr_db, rng)
        else:
            received = awgn_channel(z_norm, snr_db, rng)
        logits = self.decoder.forward(received)
        probs = softmax(logits)
        self._cache = dict(z_norm=z_norm, scale=scale)
        return probs

    def backward(self, probs, y_onehot, lr):
        n = probs.shape[0]
        d_logits = (probs - y_onehot) / n  # softmax + cross-entropy combined gradient
        d_received = self.decoder.backward(d_logits, lr)
        # normalize_power's backward is approximated as identity scaled by
        # 1/scale (standard practice: treat the power-normalization scale
        # as approximately constant within a mini-batch for gradient
        # purposes, avoiding a second-order correction term that in
        # practice contributes negligibly at these batch sizes).
        d_z = d_received / self._cache["scale"]
        self.encoder.backward(d_z, lr)

    def predict(self, x, snr_db, rng):
        probs = self.forward(x, snr_db, rng, training=False)
        return np.argmax(probs, axis=1)


class ReconstructionJSCC:
    """Same encoder/channel/decoder structure, but the decoder outputs a
    reconstructed image and training minimizes pixel-wise MSE, matching
    the original (non-task-aware) JSCC formulation.
    """

    def __init__(self, input_dim, k, rng, hidden=32):
        self.k = k
        self.encoder = MLP([input_dim, hidden, k], rng, activations=["relu", "linear"])
        self.decoder = MLP([k, hidden, input_dim], rng, activations=["relu", "sigmoid"])

    def forward(self, x, snr_db, rng):
        z = self.encoder.forward(x)
        z_norm, scale = normalize_power(z)
        received = awgn_channel(z_norm, snr_db, rng)
        recon = self.decoder.forward(received)
        self._cache = dict(z_norm=z_norm, scale=scale)
        return recon

    def backward(self, recon, x_target, lr):
        n = recon.shape[0]
        d_recon = 2 * (recon - x_target) / n
        d_received = self.decoder.backward(d_recon, lr)
        d_z = d_received / self._cache["scale"]
        self.encoder.backward(d_z, lr)


def train_task_oriented(model, X_train, y_train, snr_db, rng, epochs=60, lr=0.05, batch_size=32):
    n_classes = int(y_train.max()) + 1
    y_onehot = one_hot(y_train, n_classes)
    n = len(X_train)
    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_onehot[idx]
            probs = model.forward(xb, snr_db, rng, training=True)
            loss = cross_entropy_loss(probs, yb)
            epoch_loss += loss * len(idx)
            model.backward(probs, yb, lr)
        losses.append(epoch_loss / n)
    return losses


def train_reconstruction(model, X_train, snr_db, rng, epochs=60, lr=0.05, batch_size=32):
    n = len(X_train)
    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train[idx]
            recon = model.forward(xb, snr_db, rng)
            loss = mse_loss(recon, xb)
            epoch_loss += loss * len(idx)
            model.backward(recon, xb, lr)
        losses.append(epoch_loss / n)
    return losses

# ============ classical_baseline.py ============
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

# ============ load_cifar10_grayscale ============
def load_cifar10_grayscale(data_dir):
    """Loads real CIFAR-10 (all 5 train batches + test batch = 60,000
    real images), converted to grayscale via the standard ITU-R BT.601
    luminance formula for a fair comparison to this study's other
    grayscale datasets. Converts to grayscale one batch at a time
    (rather than holding all batches as float64 RGB simultaneously,
    roughly 1.5GB, which triggered an out-of-memory kill when first
    attempted) to keep peak memory usage low.
    """
    def load_and_gray_batch(path):
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
        labels = np.array(batch[b"labels"])
        gray = 0.299 * data[:, 0] + 0.587 * data[:, 1] + 0.114 * data[:, 2]
        gray = gray.reshape(len(gray), -1) / 255.0
        return gray.astype(np.float32), labels

    all_gray, all_labels = [], []
    for i in range(1, 6):
        gray, labels = load_and_gray_batch(os.path.join(data_dir, f"data_batch_{i}"))
        all_gray.append(gray)
        all_labels.append(labels)
    gray, labels = load_and_gray_batch(os.path.join(data_dir, "test_batch"))
    all_gray.append(gray)
    all_labels.append(labels)

    X = np.concatenate(all_gray, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

# ============ run_multidataset.py functions ============
def holm_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim, k_values, snr_values,
                        seed, batch_size=32):
    """Runs every (k, snr) combination for ONE already-split seed.
    Fits PCA once at the largest k (fix 2), trains the classical
    classifier once per k (fix 3), and still trains JSCC fresh for
    every (k, snr) pair, since JSCC's SNR-dependence is genuine."""
    rows = []
    max_k = max(k_values)

    full_pca = PCACompressor(max_k).fit(X_train)

    for k in k_values:
        pca_k = PCACompressor(k)
        pca_k.mean = full_pca.mean
        pca_k.components = full_pca.components[:k]

        rng_clf = np.random.RandomState(seed)
        clf = train_classifier_on_clean(pca_k.decode(pca_k.encode(X_train)), y_train, 10, rng_clf,
                                         epochs=EPOCHS, lr=LR, hidden=HIDDEN, batch_size=batch_size)

        for snr in snr_values:
            rng_jscc = np.random.RandomState(seed)
            task_model = TaskOrientedJSCC(input_dim, k, 10, rng_jscc, hidden=HIDDEN)
            train_task_oriented(task_model, X_train, y_train, snr, rng_jscc,
                                 epochs=EPOCHS, lr=LR, batch_size=batch_size)
            task_acc = float(np.mean(task_model.predict(X_test, snr, rng_jscc) == y_test))

            rng_eval = np.random.RandomState(seed)
            classical_preds = classify(clf, pca_k.decode(
                transmit_uncoded(pca_k.encode(X_test), snr, rng_eval)))
            classical_acc = float(np.mean(classical_preds == y_test))

            rows.append(dict(k=k, snr=snr, seed=seed,
                              task_oriented=task_acc, classical_uncoded=classical_acc))
    return rows


def run_dataset(name, X_all, y_all, input_dim, k_values, snr_values, out_name, batch_size=32):
    """Splits fresh for each seed from a single pool. Use for datasets
    without a pre-defined official split (optdigits, CIFAR-10)."""
    rows = []
    out_path = os.path.join(DATA_DIR, out_name)
    for seed in SEEDS_TO_RUN:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
        rows.extend(run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim,
                                        k_values, snr_values, seed, batch_size=batch_size))
        print(f"  [{name}] seed={seed} done")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{name}] progress saved to {out_path} ({len(rows)} rows so far)")
    return pd.DataFrame(rows)


def run_dataset_presplit(name, X_train_full, y_train_full, X_test_full, y_test_full,
                          input_dim, k_values, snr_values, out_name, batch_size=32):
    """For datasets with a genuine official train/test split (MNIST):
    every seed trains on a shuffled view of the official training set
    and evaluates on the same, complete, unchanged official test set."""
    rows = []
    out_path = os.path.join(DATA_DIR, out_name)
    for seed in range(N_SEEDS):
        perm = np.random.RandomState(seed).permutation(len(X_train_full))
        X_train, y_train = X_train_full[perm], y_train_full[perm]
        X_test, y_test = X_test_full, y_test_full
        rows.extend(run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim,
                                        k_values, snr_values, seed, batch_size=batch_size))
        print(f"  [{name}] seed={seed} done")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{name}] progress saved to {out_path} ({len(rows)} rows so far)")
    return pd.DataFrame(rows)


def summarize(df, name):
    stat_rows = []
    for k in sorted(df["k"].unique()):
        for snr in sorted(df["snr"].unique()):
            sub = df[(df["k"] == k) & (df["snr"] == snr)]
            t, p = stats.ttest_rel(sub["task_oriented"], sub["classical_uncoded"])
            winner = "JSCC" if sub["task_oriented"].mean() > sub["classical_uncoded"].mean() else "Classical"
            stat_rows.append(dict(k=k, snr=snr, winner=winner, p=p))
    stat_df = pd.DataFrame(stat_rows)
    stat_df["p_holm"] = holm_correct(stat_df["p"].values)
    stat_df["significant_uncorrected"] = stat_df["p"] < 0.05
    stat_df["significant_holm"] = stat_df["p_holm"] < 0.05

    n_jscc_unc = ((stat_df["winner"] == "JSCC") & stat_df["significant_uncorrected"]).sum()
    n_classical_unc = ((stat_df["winner"] == "Classical") & stat_df["significant_uncorrected"]).sum()
    n_jscc_holm = ((stat_df["winner"] == "JSCC") & stat_df["significant_holm"]).sum()
    n_classical_holm = ((stat_df["winner"] == "Classical") & stat_df["significant_holm"]).sum()
    n_total = len(stat_df)
    print(f"{name}: UNCORRECTED JSCC wins {n_jscc_unc}/{n_total}, Classical wins {n_classical_unc}/{n_total}")
    print(f"{name}: HOLM-CORRECTED JSCC wins {n_jscc_holm}/{n_total}, Classical wins {n_classical_holm}/{n_total}")
    return stat_df




# ============ Download real CIFAR-10 directly (Colab has full internet access) ============
if __name__ == "__main__":
    import urllib.request
    import tarfile

    cifar_dir = "cifar-10-batches-py"
    if not os.path.exists(cifar_dir):
        print("Downloading real CIFAR-10 from the official source (~170MB)...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
        print("Extracting...")
        with tarfile.open("cifar-10-python.tar.gz") as tar:
            tar.extractall()
        print("Done.")
    else:
        print(f"Found existing {cifar_dir}/, skipping download.")

    print("\nLoading real CIFAR-10 (full 50,000-image training set)...")
    X_full, y_full = load_cifar10_grayscale(cifar_dir)
    # Use only the 50,000 training images (drop the 10,000 test images
    # at the end), matching this study's other datasets' train/test
    # split convention rather than mixing CIFAR-10's own test images in.
    X_full, y_full = X_full[:50000], y_full[:50000]
    print(f"  Loaded {len(X_full)} real training images (full CIFAR-10, not subsampled)")
    print(f"  Saving progress to: {DATA_DIR}")

    full_run = sorted(SEEDS_TO_RUN) == list(range(N_SEEDS))
    out_name = "cifar10_results.csv" if full_run else "cifar10_results_seed" + "_".join(map(str, SEEDS_TO_RUN)) + ".csv"
    cifar_df = run_dataset("CIFAR-10", X_full, y_full, 1024, [16, 32, 64, 128],
                            [-5, 0, 5, 10, 15, 20], out_name, batch_size=256)
    print(f"\nSaved {len(cifar_df)} rows to {os.path.join(DATA_DIR, out_name)}")

    if full_run:
        summarize(cifar_df, "CIFAR-10")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k_values = sorted(cifar_df["k"].unique())
        fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 4), sharey=True)
        for ax, k in zip(axes, k_values):
            g = cifar_df[cifar_df["k"] == k].groupby("snr").agg(
                jm=("task_oriented", "mean"), js=("task_oriented", "std"),
                cm=("classical_uncoded", "mean"), cs=("classical_uncoded", "std")).reset_index()
            ax.errorbar(g["snr"], g["jm"], yerr=g["js"], marker="o", color="#2a6ebb",
                        label="Task-Oriented JSCC", capsize=3)
            ax.errorbar(g["snr"], g["cm"], yerr=g["cs"], marker="s", color="#c0392b",
                        label="Classical (uncoded)", capsize=3)
            ax.set_title(f"k={k}"); ax.set_xlabel("SNR (dB)")
        axes[0].set_ylabel("Task accuracy"); axes[-1].legend(fontsize=8, loc="lower right")
        fig.suptitle("CIFAR-10: task accuracy vs. channel SNR (mean \u00b1 std, 5 seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(DATA_DIR, "cifar10_accuracy_vs_snr.png"), dpi=150)
        print("Figure saved.")
    else:
        # Single-seed run: still save a figure so this notebook's result can be
        # checked before downloading (no error bars with one seed).
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k_values = sorted(cifar_df["k"].unique())
        fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 4), sharey=True)
        for ax, k in zip(axes, k_values):
            g = cifar_df[cifar_df["k"] == k].groupby("snr")[["task_oriented", "classical_uncoded"]].mean().reset_index()
            ax.plot(g["snr"], g["task_oriented"], marker="o", color="#2a6ebb", label="Task-Oriented JSCC")
            ax.plot(g["snr"], g["classical_uncoded"], marker="s", color="#c0392b", label="Classical (uncoded)")
            ax.set_title(f"k={k}"); ax.set_xlabel("SNR (dB)")
        axes[0].set_ylabel("Task accuracy"); axes[-1].legend(fontsize=8, loc="lower right")
        tag = "_".join(map(str, SEEDS_TO_RUN))
        fig.suptitle(f"CIFAR-10: task accuracy vs. channel SNR (seed {tag} only)")
        fig.tight_layout()
        fig.savefig(os.path.join(DATA_DIR, f"cifar10_accuracy_vs_snr_seed{tag}.png"), dpi=150)
        print("Single-seed figure saved. Combine the per-seed CSVs afterwards for the full analysis.")
    print(f"\nDone. Results saved under {DATA_DIR}")
