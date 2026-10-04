"""
investigate_hidden64_anomaly.py
----------------------------------
Follow-up to run_diagnostics.py's finding that hidden=64 underperforms
hidden=32 (a real, multi-seed-confirmed effect, not noise). This script
asks the obvious next question: is this a genuine capacity/task effect,
or an optimization artifact from using the SAME learning rate (tuned
for hidden=32) on a wider, harder-to-optimize network?

Three checks:
  1. Full loss curves for hidden=32 vs hidden=64 at the same LR --
     if hidden=64 is still decreasing at epoch 80, it's undertrained,
     not at a genuinely worse ceiling.
  2. Gradient norm tracking during training -- exploding or vanishing
     gradients would show up directly here, a concrete signature of
     an optimization problem rather than a capacity ceiling.
  3. An LR sweep SPECIFIC to hidden=64 -- if a different LR closes the
     gap to hidden=32, the original comparison was unfair (same LR
     forced on a network it wasn't tuned for), not evidence hidden=64
     is worse for this task.

Run:
    python investigate_hidden64_anomaly.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import TaskOrientedJSCC
from nn_layers import softmax, cross_entropy_loss, one_hot
from channel import normalize_power, awgn_channel

K = 4
SNR = 0
N_SEEDS = 5

digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target


def train_with_tracking(hidden, lr, seed, epochs=80, batch_size=32, track_grad_norm=False):
    """Like train_task_oriented, but also returns the per-epoch loss
    curve (and, optionally, per-epoch gradient norm) so we can see
    HOW training unfolded, not just the final accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
    rng = np.random.RandomState(seed)
    model = TaskOrientedJSCC(64, K, 10, rng, hidden=hidden)

    n_classes = 10
    y_onehot = one_hot(y_train, n_classes)
    n = len(X_train)
    losses, grad_norms = [], []

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_onehot[idx]
            probs = model.forward(xb, SNR, rng, training=True)
            loss = cross_entropy_loss(probs, yb)
            epoch_loss += loss * len(idx)

            if track_grad_norm:
                # Compute the gradient norm at the decoder's input
                # (post-channel), a direct signal of whether gradients
                # are exploding or vanishing as they flow back through
                # the channel and encoder.
                d_logits = (probs - yb) / len(idx)
                d_received = model.decoder.backward(d_logits, lr=0.0)  # lr=0: measure only, don't update yet
                grad_norm = np.linalg.norm(d_received)
                epoch_grad_norm += grad_norm
                n_batches += 1
                # Now actually do the real update (decoder already
                # "backward"-called above with lr=0, so redo properly)
                probs2 = model.forward(xb, SNR, rng, training=True)
                model.backward(probs2, yb, lr)
            else:
                model.backward(probs, yb, lr)

        losses.append(epoch_loss / n)
        if track_grad_norm:
            grad_norms.append(epoch_grad_norm / max(n_batches, 1))

    preds = model.predict(X_test, SNR, rng)
    acc = float(np.mean(preds == y_test))
    return dict(losses=losses, grad_norms=grad_norms, final_acc=acc)


if __name__ == "__main__":
    print("[1/3] Full loss curves: hidden=32 vs hidden=64, same LR=0.05, 80 epochs, seed=0...")
    r32 = train_with_tracking(32, lr=0.05, seed=0, epochs=80)
    r64 = train_with_tracking(64, lr=0.05, seed=0, epochs=80)
    print(f"  hidden=32: loss epoch 1={r32['losses'][0]:.3f}, epoch 40={r32['losses'][39]:.3f}, "
          f"epoch 80={r32['losses'][-1]:.3f}, final_acc={r32['final_acc']:.3f}")
    print(f"  hidden=64: loss epoch 1={r64['losses'][0]:.3f}, epoch 40={r64['losses'][39]:.3f}, "
          f"epoch 80={r64['losses'][-1]:.3f}, final_acc={r64['final_acc']:.3f}")
    still_improving_64 = r64['losses'][-1] < r64['losses'][-10] * 0.98
    print(f"  hidden=64 still meaningfully improving in the last 10 epochs: {still_improving_64}")
    if still_improving_64:
        print("  -> hidden=64 had NOT converged by epoch 80 -- the original comparison may simply "
              "need more epochs for the wider network, not evidence of a worse ceiling.")
    else:
        print("  -> hidden=64 had plateaued by epoch 80 -- more epochs alone likely would not close the gap.")

    print("\n[2/3] Gradient norm tracking (decoder input, post-channel) across training...")
    r32_grad = train_with_tracking(32, lr=0.05, seed=0, epochs=30, track_grad_norm=True)
    r64_grad = train_with_tracking(64, lr=0.05, seed=0, epochs=30, track_grad_norm=True)
    print(f"  hidden=32: grad norm epoch 1={r32_grad['grad_norms'][0]:.4f}, "
          f"epoch 15={r32_grad['grad_norms'][14]:.4f}, epoch 30={r32_grad['grad_norms'][-1]:.4f}")
    print(f"  hidden=64: grad norm epoch 1={r64_grad['grad_norms'][0]:.4f}, "
          f"epoch 15={r64_grad['grad_norms'][14]:.4f}, epoch 30={r64_grad['grad_norms'][-1]:.4f}")
    ratio = r64_grad['grad_norms'][-1] / max(r32_grad['grad_norms'][-1], 1e-8)
    print(f"  Ratio (hidden64/hidden32) at epoch 30: {ratio:.2f}x")
    if ratio > 3 or ratio < 0.33:
        print("  -> Gradient norms differ substantially between widths -- consistent with an "
              "optimization-stability difference, not just a capacity difference.")
    else:
        print("  -> Gradient norms are comparable -- no obvious explosion/vanishing signature.")

    print(f"\n[3/3] LR sweep specific to hidden=64 ({N_SEEDS} seeds each), to test whether the "
          f"original LR=0.05 (tuned for hidden=32) was simply unfair to hidden=64...")
    for lr in [0.02, 0.05, 0.08, 0.12, 0.2]:
        accs = [train_with_tracking(64, lr=lr, seed=s, epochs=80)['final_acc'] for s in range(N_SEEDS)]
        print(f"  hidden=64, lr={lr}: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}, "
              f"individual={[round(a, 3) for a in accs]}")

    print("\nCompare the best LR found above against hidden=32's original lr=0.05 result "
          "(mean=0.345, std=0.052, from run_diagnostics.py). If any hidden=64 LR setting "
          "matches or beats that, the original 'hidden=64 is worse' finding was an artifact "
          "of forcing one LR on both widths, not a genuine capacity ceiling.")
