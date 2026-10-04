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
import numpy as np


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


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    z = rng.randn(100, 16) * 3.0  # unnormalized signal
    z_norm, scale = normalize_power(z)
    print("power after normalization (should be ~1.0):", np.mean(z_norm ** 2))
    for snr in [0, 10, 20]:
        received = awgn_channel(z_norm, snr, rng)
        measured_noise_power = np.mean((received - z_norm) ** 2)
        expected_noise_power = 1.0 / (10 ** (snr / 10.0))
        print(f"SNR={snr}dB: measured noise power={measured_noise_power:.4f}, "
              f"expected={expected_noise_power:.4f}")
