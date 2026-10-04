"""
generate_figure9.py
----------------------
Figure 9: task-oriented JSCC vs. classical comparison, at a representative
bandwidth, across three real datasets of increasing complexity (optdigits,
MNIST, CIFAR-10). Reads each dataset's actual results CSV (produced by
run_multidataset.py) rather than using placeholder or invented data.

Run (after run_multidataset.py has produced these three files in ../data/):
    python generate_figure9.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# (filename, dataset label, representative k to plot)
DATASETS = [
    ("optdigits_full_results.csv", "Optdigits (k=32)", 32),
    ("mnist_results.csv", "MNIST (k=32)", 32),
    ("cifar10_results.csv", "CIFAR-10 (k=32)", 32),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, (fname, label, rep_k) in zip(axes, DATASETS):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f"{fname}\nnot found --\nrun run_multidataset.py first",
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_title(label)
        continue

    df = pd.read_csv(path)
    sub = df[df["k"] == rep_k]
    grouped = sub.groupby("snr").agg(
        jscc_mean=("task_oriented", "mean"),
        classical_mean=("classical_uncoded", "mean"),
    ).reset_index().sort_values("snr")

    ax.plot(grouped["snr"], grouped["jscc_mean"], marker="o", color="#2a6ebb", label="JSCC")
    ax.plot(grouped["snr"], grouped["classical_mean"], marker="s", color="#c0392b", label="Classical")
    ax.set_xlabel("SNR (dB)")
    ax.set_title(label)

axes[0].set_ylabel("Task accuracy")
axes[0].legend()
fig.suptitle("Task-oriented JSCC vs. classical, representative bandwidth, three real datasets")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure9_three_dataset_curves.png"), dpi=150)
print(f"Saved to {FIG_DIR}/figure9_three_dataset_curves.png")
