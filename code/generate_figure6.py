"""
generate_figure6.py
----------------------
Figure 6: statistical significance summary across the full (k, SNR) grid,
using the real Holm-corrected statistical test results from
run_experiments.py's output (table2_statistical_tests.csv).

Run this AFTER run_experiments.py has produced that file:
    python generate_figure6.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

stats_path = os.path.join(DATA_DIR, "table2_statistical_tests.csv")
if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"{stats_path} not found. Run run_experiments.py first to generate it.")

df = pd.read_csv(stats_path)

# Encode: 0 = no significant difference, 1 = JSCC significant win,
# -1 = Classical significant win (using the Holm-corrected column)
def encode(row):
    if not row["significant_holm"]:
        return 0
    return 1 if row["winner"] == "JSCC" else -1

df["code"] = df.apply(encode, axis=1)

k_values = sorted(df["k"].unique())
snr_values = sorted(df["snr"].unique())
grid = np.zeros((len(k_values), len(snr_values)))
for i, k in enumerate(k_values):
    for j, snr in enumerate(snr_values):
        row = df[(df["k"] == k) & (df["snr"] == snr)]
        grid[i, j] = row["code"].values[0] if len(row) else 0

fig, ax = plt.subplots(figsize=(6, 4.5))
cmap = ListedColormap(["#c0392b", "#dddddd", "#2a6ebb"])  # Classical, none, JSCC
im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(len(snr_values)))
ax.set_xticklabels(snr_values)
ax.set_yticks(range(len(k_values)))
ax.set_yticklabels(k_values)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Bandwidth k")
ax.set_title("Statistical significance summary (Holm-corrected, n=5 seeds)")

for i in range(len(k_values)):
    for j in range(len(snr_values)):
        label = {1: "JSCC", -1: "Classical", 0: ""}[grid[i, j]]
        if label:
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                     color="white" if grid[i, j] != 0 else "black")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure6_significance_summary.png"), dpi=150)
print(f"Saved to {FIG_DIR}/figure6_significance_summary.png")

n_jscc = (df["code"] == 1).sum()
n_classical = (df["code"] == -1).sum()
print(f"Confirmed from real data: JSCC wins {n_jscc}, Classical wins {n_classical}, "
      f"neither {len(df) - n_jscc - n_classical}")
