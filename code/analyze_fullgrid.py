"""
analyze_fullgrid.py
----------------------
Statistical analysis of the full-grid digital-baseline and CNN-encoder runs
(run_fullgrid_digital_cnn.py). For each comparison, a paired t-test across
seeds is run at each of the 24 (k, SNR) cells and Holm-corrected within that
family of 24; mean differences, 95% CIs, and d_z are reported.

Comparisons:
  digital  vs  analog classical (PCA + uncoded)
  digital  vs  task-oriented JSCC (MLP encoder)
  CNN JSCC vs  analog classical
  CNN JSCC vs  MLP JSCC

Usage (from code/):  python analyze_fullgrid.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(running, 1.0)
    return adj


def compare(df, a, b):
    """Paired comparison of column a minus column b at every (k, snr) cell."""
    rows = []
    for k in sorted(df.k.unique()):
        for snr in sorted(df.snr.unique()):
            sub = df[(df.k == k) & (df.snr == snr)]
            x = (sub[a] - sub[b]).values; n = len(x)
            mean, sd = x.mean(), x.std(ddof=1)
            se = sd / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
            _, p = stats.ttest_rel(sub[a], sub[b])
            rows.append(dict(k=int(k), snr=int(snr), n=n, mean_a=sub[a].mean(), mean_b=sub[b].mean(),
                             mean_diff=mean, ci_low=mean - tc * se, ci_high=mean + tc * se,
                             d_z=mean / sd if sd > 0 else np.nan, p=p))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p.values)
    t["significant_holm"] = t.p_holm < 0.05
    t["winner"] = np.where(t.mean_diff > 0, "A", "B")
    return t


def summarize(t, label, a_name, b_name):
    sig = t[t.significant_holm]
    na = int((sig.winner == "A").sum()); nb = int((sig.winner == "B").sum())
    print(f"{label:38s} {a_name} wins {na:2d}, {b_name} wins {nb:2d}, no significant difference {len(t)-na-nb:2d}   "
          f"(uncorrected: {int(((t.p<.05)&(t.winner=='A')).sum())}/{int(((t.p<.05)&(t.winner=='B')).sum())})")
    return na, nb


if __name__ == "__main__":
    import glob
    raw = pd.read_csv(os.path.join(DATA, "pilot_table_raw_results.csv"))

    def load_part(base):
        """Combined file plus any per-seed files (fullgrid_<base>_seed*.csv), de-duplicated."""
        files = [os.path.join(DATA, f"fullgrid_{base}.csv")] + sorted(glob.glob(os.path.join(DATA, f"fullgrid_{base}_seed*.csv")))
        frames = [pd.read_csv(f) for f in files if os.path.exists(f)]
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["k", "snr", "seed"])
        return df.sort_values(["seed", "k", "snr"]).reset_index(drop=True)

    d_all, c_all = load_part("digital"), load_part("cnn")
    dpath = d_all is not None
    cpath = c_all is not None

    if dpath:
        d = d_all.merge(raw[["k", "snr", "seed", "task_oriented", "classical_uncoded"]],
                                     on=["k", "snr", "seed"])
        print(f"digital baseline: {len(d)} rows ({d.groupby(['k','snr']).size().min()} seeds per cell)")
        t1 = compare(d, "digital_acc", "classical_uncoded"); t1.to_csv(os.path.join(DATA, "fullgrid_digital_vs_analog.csv"), index=False)
        summarize(t1, "digital vs analog classical", "digital", "analog")
        t2 = compare(d, "digital_acc", "task_oriented"); t2.to_csv(os.path.join(DATA, "fullgrid_digital_vs_jscc.csv"), index=False)
        summarize(t2, "digital vs task-oriented JSCC (MLP)", "digital", "JSCC")

    if cpath:
        c = c_all.merge(raw[["k", "snr", "seed", "task_oriented", "classical_uncoded"]],
                                     on=["k", "snr", "seed"])
        full = c.groupby(["k", "snr"]).size().min()
        print(f"\nCNN encoder: {len(c)} rows ({full} seeds per cell)")
        print("MLP column reproduces the main grid's task_oriented exactly:",
              bool((c.mlp_acc - c.task_oriented).abs().max() < 1e-12))
        t3 = compare(c, "cnn_acc", "classical_uncoded"); t3.to_csv(os.path.join(DATA, "fullgrid_cnn_vs_classical.csv"), index=False)
        summarize(t3, "CNN JSCC vs analog classical", "CNN", "classical")
        t4 = compare(c, "cnn_acc", "mlp_acc"); t4.to_csv(os.path.join(DATA, "fullgrid_cnn_vs_mlp.csv"), index=False)
        summarize(t4, "CNN JSCC vs MLP JSCC", "CNN", "MLP")
