"""
compute_effect_sizes.py
--------------------------
For each dataset's 24-cell (k, SNR) grid, computes the paired JSCC-minus-classical
accuracy difference with its 95% confidence interval (t-based, n = number of seeds),
the standardized paired effect size d_z, the raw p-value, and the Holm-adjusted
p-value within the dataset's family of 24 tests.

Writes data/effect_sizes_<dataset>.csv (all 24 cells) and prints a per-dataset,
per-direction summary of the Holm-significant cells.

Usage (from code/):  python compute_effect_sizes.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATASETS = [
    ("pilot",     "pilot_table_raw_results.csv"),
    ("optdigits", "optdigits_full_results.csv"),
    ("mnist",     "mnist_results.csv"),
    ("cifar10",   "cifar10_results.csv"),
]


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(running, 1.0)
    return adj


def effect_table(df):
    rows = []
    for k in sorted(df.k.unique()):
        for snr in sorted(df.snr.unique()):
            sub = df[(df.k == k) & (df.snr == snr)]
            x = (sub.task_oriented - sub.classical_uncoded).values
            n = len(x); mean = x.mean(); sd = x.std(ddof=1)
            se = sd / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1)
            _, p = stats.ttest_rel(sub.task_oriented, sub.classical_uncoded)
            rows.append(dict(k=int(k), snr=int(snr), n=n, mean_diff=mean,
                             ci_low=mean - tcrit * se, ci_high=mean + tcrit * se,
                             d_z=mean / sd if sd > 0 else np.nan, p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    t["significant_holm"] = t.p_holm < 0.05
    t["winner"] = np.where(t.mean_diff > 0, "JSCC", "Classical")
    return t


if __name__ == "__main__":
    print(f"{'dataset':10s} {'direction':10s} {'cells':>5s} {'mean diff range':>20s} {'|d_z| range':>14s} {'CIs exclude 0':>14s}")
    for name, fname in DATASETS:
        df = pd.read_csv(os.path.join(DATA, fname))
        t = effect_table(df)
        t.to_csv(os.path.join(DATA, f"effect_sizes_{name}.csv"), index=False)
        for direction in ("JSCC", "Classical"):
            s = t[(t.significant_holm) & (t.winner == direction)]
            if len(s) == 0:
                print(f"{name:10s} {direction:10s} {0:5d}")
                continue
            excl = bool(((s.ci_low > 0) | (s.ci_high < 0)).all())
            print(f"{name:10s} {direction:10s} {len(s):5d} "
                  f"{s.mean_diff.min():+.3f} to {s.mean_diff.max():+.3f}   "
                  f"{s.d_z.abs().min():5.2f} to {s.d_z.abs().max():5.2f}   {str(excl):>10s}")
