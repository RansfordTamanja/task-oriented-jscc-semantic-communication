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
import numpy as np
from nn_layers import MLP, softmax, cross_entropy_loss, mse_loss, one_hot
from channel import normalize_power, awgn_channel


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
