"""
combine_seed_results.py
--------------------------
Merges per-seed CIFAR-10 result files (cifar10_results_seed0.csv, ...seed1.csv, ...)
produced by running one seed per Kaggle notebook, checks the combined file is
complete (4 k x 6 SNR x 5 seeds = 120 rows), saves data/cifar10_results.csv,
and prints the Holm-corrected summary.

Usage (from code/), after putting the per-seed CSVs in ../data/ :
    python combine_seed_results.py
"""
import glob, os
import pandas as pd
from scipy import stats
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
files = sorted(glob.glob(os.path.join(DATA, "cifar10_results_seed*.csv")))
if not files:
    raise SystemExit("No cifar10_results_seed*.csv files found in data/")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["k", "snr", "seed"]).sort_values(["seed", "k", "snr"])
seeds = sorted(df.seed.unique())
print(f"Found seeds {seeds} from {len(files)} file(s); {len(df)} rows")
if len(df) != 120 or seeds != [0, 1, 2, 3, 4]:
    raise SystemExit("Incomplete: need all 5 seeds (120 rows). Add the missing seed file(s) and rerun.")
out = os.path.join(DATA, "cifar10_results.csv")
df.to_csv(out, index=False)
print("Saved", out)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.empty(m); r = 0
    for i, j in enumerate(o):
        r = max(r, (m - i) * p[j]); a[j] = min(r, 1)
    return a
rows = []
for k in sorted(df.k.unique()):
    for s in sorted(df.snr.unique()):
        sub = df[(df.k == k) & (df.snr == s)]
        _, p = stats.ttest_rel(sub.task_oriented, sub.classical_uncoded)
        w = "JSCC" if sub.task_oriented.mean() > sub.classical_uncoded.mean() else "Classical"
        rows.append(dict(k=k, snr=s, winner=w, p=p))
r = pd.DataFrame(rows); r["ph"] = holm(r.p); r["sig"] = r.ph < 0.05; r["sigu"] = r.p < 0.05
print(f"CIFAR-10: UNCORRECTED JSCC {((r.winner=='JSCC')&r.sigu).sum()}/24, Classical {((r.winner=='Classical')&r.sigu).sum()}/24")
print(f"CIFAR-10: HOLM-CORRECTED JSCC {((r.winner=='JSCC')&r.sig).sum()}/24, Classical {((r.winner=='Classical')&r.sig).sum()}/24")

# ---- Figure: accuracy vs. SNR, one panel per k (mean +/- std over 5 seeds) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = os.path.join(DATA, "..", "figures"); os.makedirs(FIG, exist_ok=True)
ks = sorted(df.k.unique())
fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), sharey=True)
for ax, k in zip(axes, ks):
    g = df[df.k == k].groupby("snr").agg(jm=("task_oriented", "mean"), js=("task_oriented", "std"),
                                           cm=("classical_uncoded", "mean"), cs=("classical_uncoded", "std")).reset_index()
    ax.errorbar(g.snr, g.jm, yerr=g.js, marker="o", color="#2a6ebb", label="Task-Oriented JSCC", capsize=3)
    ax.errorbar(g.snr, g.cm, yerr=g.cs, marker="s", color="#c0392b", label="Classical (uncoded)", capsize=3)
    ax.set_title(f"k={k}"); ax.set_xlabel("SNR (dB)")
axes[0].set_ylabel("Task accuracy"); axes[-1].legend(fontsize=8, loc="lower right")
fig.suptitle("CIFAR-10: task accuracy vs. channel SNR (mean \u00b1 std, 5 seeds)")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cifar10_accuracy_vs_snr.png"), dpi=150)
print("Saved data/cifar10_accuracy_vs_snr.png")

# ---- Regenerate Figure 9 so its CIFAR-10 panel is filled in ----
import subprocess, sys
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_figure9.py")], check=False)
print("Figure 9 regenerated (figures/figure9_three_dataset_curves.png)")
