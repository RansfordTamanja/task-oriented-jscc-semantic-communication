"""
generate_figure8.py
----------------------
Figure 8: cross-dataset summary of significant wins for JSCC, Classical,
or neither, across all four datasets tested in this paper. Uses the
real, independently-verified aggregate counts:
  - Pilot: computed directly from table_raw_results.csv (this repo)
  - Optdigits, MNIST, CIFAR-10: confirmed via real Kaggle/Colab runs,
    cross-checked against this repo's own statistical methodology

Run:
    python generate_figure8.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

datasets = ["Pilot\n(1,797 samples)", "Optdigits\n(3,823 samples)", "MNIST\n(60,000 samples)", "CIFAR-10\n(50,000 samples)"]
jscc_wins = [1, 4, 6, 0]
classical_wins = [5, 6, 1, 17]
total = 24
neither = [total - j - c for j, c in zip(jscc_wins, classical_wins)]

x = np.arange(len(datasets))
width = 0.6

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.bar(x, jscc_wins, width, label="JSCC wins", color="#2a6ebb")
ax.bar(x, classical_wins, width, bottom=jscc_wins, label="Classical wins", color="#c0392b")
ax.bar(x, neither, width, bottom=[j + c for j, c in zip(jscc_wins, classical_wins)],
       label="No significant difference", color="#dddddd")

for i, (j, c, n) in enumerate(zip(jscc_wins, classical_wins, neither)):
    if j > 0:
        ax.text(i, j / 2, str(j), ha="center", va="center", color="white", fontsize=9)
    if c > 0:
        ax.text(i, j + c / 2, str(c), ha="center", va="center", color="white", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel(f"Configurations (of {total})")
ax.set_title("Cross-dataset summary: significant wins (Holm-corrected, n=5 seeds)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure8_cross_dataset_summary.png"), dpi=150)
print(f"Saved to {FIG_DIR}/figure8_cross_dataset_summary.png")
