"""
investigate_cnn_encoder.py
------------------------------
Follow-up to run_experiments.py's [2a/3] CNN-vs-MLP comparison, which
found CNN underperforming MLP at every representative point -- but
using LR=0.05, epochs=80, both tuned for the MLP encoder specifically,
not the CNN. This script tests whether that comparison was fair,
using the same rigor as investigate_hidden64_anomaly.py: an LR sweep,
an epoch sweep, and loss-curve tracking, specific to the CNN encoder.

Run:
    python investigate_cnn_encoder.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import TaskOrientedJSCC
from nn_layers import cross_entropy_loss, one_hot

digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target

N_SEEDS = 5


def train_cnn_with_tracking(k, snr, lr, seed, epochs, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
    rng = np.random.RandomState(seed)
    model = TaskOrientedJSCC(64, k, 10, rng, encoder_type="cnn")

    y_onehot = one_hot(y_train, 10)
    n = len(X_train)
    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_onehot[idx]
            probs = model.forward(xb, snr, rng, training=True)
            loss = cross_entropy_loss(probs, yb)
            epoch_loss += loss * len(idx)
            model.backward(probs, yb, lr)
        losses.append(epoch_loss / n)

    preds = model.predict(X_test, snr, rng)
    acc = float(np.mean(preds == y_test))
    return dict(losses=losses, final_acc=acc)


if __name__ == "__main__":
    K, SNR = 4, 0
    print(f"[1/3] CNN loss curve at the original LR=0.05, epochs=80, k={K}, SNR={SNR}dB, seed=0...")
    r = train_cnn_with_tracking(K, SNR, lr=0.05, seed=0, epochs=80)
    print(f"  loss epoch 1={r['losses'][0]:.3f}, epoch 40={r['losses'][39]:.3f}, "
          f"epoch 80={r['losses'][-1]:.3f}, final_acc={r['final_acc']:.3f}")
    still_improving = r['losses'][-1] < r['losses'][-10] * 0.98
    print(f"  Still meaningfully improving in the last 10 epochs: {still_improving}")

    print(f"\n[2/3] CNN-specific LR sweep at k={K}, SNR={SNR}dB, {N_SEEDS} seeds each, epochs=80...")
    lr_results = {}
    for lr in [0.01, 0.02, 0.05, 0.08, 0.12]:
        accs = [train_cnn_with_tracking(K, SNR, lr=lr, seed=s, epochs=80)['final_acc']
                for s in range(N_SEEDS)]
        lr_results[lr] = accs
        print(f"  lr={lr}: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}, "
              f"individual={[round(a, 3) for a in accs]}")

    best_lr = max(lr_results, key=lambda lr: np.mean(lr_results[lr]))
    print(f"  Best LR found: {best_lr} (mean={np.mean(lr_results[best_lr]):.3f})")

    print(f"\n[3/3] CNN at best LR ({best_lr}) with more epochs (160 instead of 80), "
          f"{N_SEEDS} seeds, to rule out undertraining at the better LR too...")
    accs_more_epochs = [train_cnn_with_tracking(K, SNR, lr=best_lr, seed=s, epochs=160)['final_acc']
                         for s in range(N_SEEDS)]
    print(f"  epochs=160: mean={np.mean(accs_more_epochs):.3f}, std={np.std(accs_more_epochs):.3f}, "
          f"individual={[round(a, 3) for a in accs_more_epochs]}")

    print(f"\nSummary: original CNN result at (lr=0.05, epochs=80) was mean=0.195 "
          f"(from run_experiments.py's [2a/3]). Original MLP result at the same point "
          f"was mean=0.345. Compare both numbers above against these two baselines: "
          f"if the best-tuned CNN result still falls well short of 0.345, CNN "
          f"underperformance is likely a real property of this task/architecture "
          f"combination, not an artifact of unfair hyperparameters.")
