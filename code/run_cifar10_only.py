"""
run_cifar10_only.py
----------------------
Standalone CIFAR-10 run (full 50,000-image training set). Safe to run
independently of the other datasets. Progress is saved after every seed
and the run RESUMES automatically if interrupted: just run it again.

Run from the code/ directory:
    python run_cifar10_only.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_multidataset import run_dataset, summarize, DATA_DIR
from real_dataset_loaders import load_cifar10_grayscale

if __name__ == "__main__":
    print("Real CIFAR-10, FULL 50,000-image training set...")
    X_full, y_full = load_cifar10_grayscale("real_data_full/cifar-10-batches-py")
    # Keep only the 50,000 official training images (the loader appends the
    # 10,000 test images last), matching the paper's stated protocol.
    X_full, y_full = X_full[:50000], y_full[:50000]
    print(f"  Loaded {len(X_full)} real training images (full CIFAR-10 training set)")
    print(f"  Saving progress to: {DATA_DIR}")
    cifar_df = run_dataset("CIFAR-10", X_full, y_full, 1024, [16, 32, 64, 128],
                           [-5, 0, 5, 10, 15, 20], "cifar10_results.csv", batch_size=256)
    summarize(cifar_df, "CIFAR-10")

    # Figure: accuracy vs. SNR, one panel per bandwidth k (mean +/- std over seeds)
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
        ax.set_title(f"k={k}")
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Task accuracy")
    axes[-1].legend(fontsize=8, loc="lower right")
    n_seeds = cifar_df["seed"].nunique()
    fig.suptitle(f"CIFAR-10: task accuracy vs. channel SNR (mean \u00b1 std, {n_seeds} seeds)")
    fig.tight_layout()
    out = os.path.join(DATA_DIR, "cifar10_accuracy_vs_snr.png")
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")
    print(f"\nDone. Results saved under {DATA_DIR}")
