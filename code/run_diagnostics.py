"""
run_diagnostics.py
---------------------
Optimization diagnostics: momentum vs. plain SGD, a hidden-layer-size
sweep, and a learning-rate sweep, at the most bandwidth- and SNR-
constrained setting tested (k=4, SNR=0dB).

REVIEWER-IDENTIFIED ISSUE THIS VERSION FIXES: the original diagnostics
were run with a single seed each. The reviewer points out that
hidden=64 performing worse than hidden=32 (0.286 vs. 0.406 in the
paper's Table) is much more consistent with high seed-to-seed variance
at this small, hard setting than with a real, converged capacity
effect -- a single seed cannot distinguish these. This version runs
every diagnostic configuration across multiple seeds and reports the
mean and standard deviation, so "hidden=64 is worse" can actually be
assessed for statistical reliability rather than asserted from one
run.

Run:
    python run_diagnostics.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import TaskOrientedJSCC, train_task_oriented

N_DIAG_SEEDS = 5  # multiple seeds per configuration, addressing the
                  # reviewer's variance concern directly
K = 4
SNR = 0
EPOCHS = 80
LR_DEFAULT = 0.05
HIDDEN_DEFAULT = 32

digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target


def run_one(hidden, lr, seed, epochs=EPOCHS, optimizer="momentum"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
    rng = np.random.RandomState(seed)
    model = TaskOrientedJSCC(64, K, 10, rng, hidden=hidden, optimizer=optimizer)
    train_task_oriented(model, X_train, y_train, SNR, rng, epochs=epochs, lr=lr)
    preds = model.predict(X_test, SNR, rng)
    return float(np.mean(preds == y_test))


def run_sweep(label, hidden, lr, epochs=EPOCHS, n_seeds=N_DIAG_SEEDS, optimizer="momentum"):
    accs = [run_one(hidden, lr, seed, epochs=epochs, optimizer=optimizer) for seed in range(n_seeds)]
    mean, std = float(np.mean(accs)), float(np.std(accs))
    print(f"  {label}: mean={mean:.3f}, std={std:.3f}, n_seeds={n_seeds}, "
          f"individual={[round(a, 3) for a in accs]}")
    return dict(label=label, hidden=hidden, lr=lr, epochs=epochs,
                optimizer=optimizer, mean_acc=mean, std_acc=std, n_seeds=n_seeds)


if __name__ == "__main__":
    rows = []

    print(f"[1/5] Hidden-size sweep at k={K}, SNR={SNR}dB, "
          f"{N_DIAG_SEEDS} seeds per configuration...")
    for hidden in [16, 32, 64, 128]:
        rows.append(run_sweep(f"hidden={hidden}", hidden, LR_DEFAULT))

    print(f"\n[2/5] Learning-rate sweep at k={K}, SNR={SNR}dB, hidden={HIDDEN_DEFAULT}, "
          f"{N_DIAG_SEEDS} seeds per configuration...")
    for lr in [0.02, 0.05, 0.1, 0.2]:
        rows.append(run_sweep(f"lr={lr}", HIDDEN_DEFAULT, lr))

    print(f"\n[3/5] Adam vs. momentum-SGD at k={K}, SNR={SNR}dB, hidden={HIDDEN_DEFAULT}, "
          f"{N_DIAG_SEEDS} seeds per configuration (reviewer-requested optimizer test)...")
    momentum_row = run_sweep("optimizer=momentum", HIDDEN_DEFAULT, LR_DEFAULT, optimizer="momentum")
    rows.append(momentum_row)
    adam_row = run_sweep("optimizer=adam", HIDDEN_DEFAULT, lr=0.01, optimizer="adam")
    rows.append(adam_row)
    print(f"  momentum-SGD: {momentum_row['mean_acc']:.3f} +/- {momentum_row['std_acc']:.3f}")
    print(f"  Adam:         {adam_row['mean_acc']:.3f} +/- {adam_row['std_acc']:.3f}")

    print(f"\n[4/5] Longer training (160 epochs, 2x the paper's 80) at k={K}, SNR={SNR}dB, "
          f"hidden={HIDDEN_DEFAULT}, {N_DIAG_SEEDS} seeds per configuration "
          f"(reviewer-requested check)...")
    longer_row = run_sweep("epochs=160", HIDDEN_DEFAULT, LR_DEFAULT, epochs=160)
    rows.append(longer_row)
    baseline_row = [r for r in rows if r["label"] == "hidden=32"][0]
    print(f"  epochs=80 (baseline):  {baseline_row['mean_acc']:.3f} +/- {baseline_row['std_acc']:.3f}")
    print(f"  epochs=160 (longer):   {longer_row['mean_acc']:.3f} +/- {longer_row['std_acc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("diagnostics_multiseed.csv", index=False)

    print("\n[5/5] Checking whether hidden=64's apparent underperformance survives "
          "multiple seeds...")
    h32 = df[df["label"] == "hidden=32"].iloc[0]
    h64 = df[df["label"] == "hidden=64"].iloc[0]
    gap = h32["mean_acc"] - h64["mean_acc"]
    pooled_std = np.sqrt((h32["std_acc"]**2 + h64["std_acc"]**2) / 2)
    print(f"  hidden=32: {h32['mean_acc']:.3f} +/- {h32['std_acc']:.3f}")
    print(f"  hidden=64: {h64['mean_acc']:.3f} +/- {h64['std_acc']:.3f}")
    print(f"  Gap: {gap:.3f}, pooled std: {pooled_std:.3f}")
    if pooled_std > 0 and abs(gap) < pooled_std:
        print("  The gap is smaller than one pooled standard deviation: consistent "
              "with the reviewer's suspicion that the original single-seed result "
              "reflected high variance at this small, hard setting, not a real, "
              "converged capacity effect. Report the multi-seed mean/std in the "
              "paper, not the original single-seed numbers.")
    else:
        print("  The gap exceeds one pooled standard deviation even across "
              f"{N_DIAG_SEEDS} seeds -- this is more consistent with a real effect. "
              "Report the multi-seed mean/std alongside this observation.")

    print("\nSaved to diagnostics_multiseed.csv")
