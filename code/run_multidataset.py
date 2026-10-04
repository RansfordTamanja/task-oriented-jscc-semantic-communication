"""
run_multidataset.py
----------------------
Multi-dataset validation: UCI optdigits, MNIST, and CIFAR-10, each run
through the identical NumPy JSCC/classical pipeline used in
run_experiments.py.

FOUR ADDITIONAL FIXES IN THIS VERSION, found by an independent code
review and verified directly against the actual code before trusting
any of them:

1. MNIST OFFICIAL TRAIN/TEST SPLIT WAS NOT ACTUALLY PRESERVED. The
   previous version's own comment claimed "use the official test set
   directly as this validation's held-out evaluation set," but the
   code immediately below that comment combined the official 60,000
   train and 10,000 test images into one pool and let
   train_test_split carve out a fresh random 80/20 split -- exactly
   the kind of official-test-set problem the original reviewer
   raised, reintroduced by this codebase's own fix for it. This
   version uses the official split directly for MNIST: every seed
   trains on a shuffled view of the same 60,000 official training
   images and evaluates on the same, complete, fixed 10,000 official
   test images, never mixed into training.
2. PCA WAS REFIT FOR EVERY (k, SNR, SEED) COMBINATION, even though it
   doesn't depend on k or SNR beyond truncation. Verified directly in
   classical_baseline.py: PCACompressor.fit() keeps Vt[:k], and since
   SVD singular values are returned in descending order, the first 16
   rows of Vt are identical whether fit with k=16 or sliced from a
   k=128 fit. This version fits the full SVD once per seed, at the
   largest k needed, and each smaller k just truncates the
   already-computed components.
3. THE CLASSICAL CLASSIFIER WAS RETRAINED ONCE PER SNR, even though it
   doesn't depend on SNR. Verified directly: it trains on the clean
   (noiseless) reconstruction; SNR only enters later, at evaluation,
   via transmit_uncoded. This version trains it once per (seed, k) and
   reuses it across all SNR evaluations. JSCC is unchanged: its SNR
   dependence is baked into training via the channel layer, so it
   genuinely must be retrained per (k, SNR, seed).
4. Progress now saves after every seed, the natural checkpoint for the
   new seed-outer loop structure.

These are efficiency and correctness fixes, not changes to the
statistical comparison itself. Because the exact sequence of random
draws consumed differs from the previous version, results will not be
bit-for-bit identical to a previous run at the same seed -- they
should be statistically equivalent, not identical to the last digit.
The optdigits result already obtained does not need to be re-run
solely because of this; it was not wrong, only slower to obtain.

Requires real_data_full/ populated per README.md for optdigits and
CIFAR-10 (MNIST is fetched automatically via OpenML).

Run:
    python run_multidataset.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from real_dataset_loaders import load_full_optdigits, load_mnist_via_openml, load_cifar10_grayscale
from jscc import TaskOrientedJSCC, train_task_oriented
from classical_baseline import PCACompressor, transmit_uncoded, train_classifier_on_clean, classify

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

EPOCHS = 80
LR = 0.05
HIDDEN = 32
N_SEEDS = 5


def holm_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim, k_values, snr_values,
                        seed, batch_size=32):
    """Runs every (k, snr) combination for ONE already-split seed.
    Fits PCA once at the largest k (fix 2), trains the classical
    classifier once per k (fix 3), and still trains JSCC fresh for
    every (k, snr) pair, since JSCC's SNR-dependence is genuine."""
    rows = []
    max_k = max(k_values)

    full_pca = PCACompressor(max_k).fit(X_train)

    for k in k_values:
        pca_k = PCACompressor(k)
        pca_k.mean = full_pca.mean
        pca_k.components = full_pca.components[:k]

        rng_clf = np.random.RandomState(seed)
        clf = train_classifier_on_clean(pca_k.decode(pca_k.encode(X_train)), y_train, 10, rng_clf,
                                         epochs=EPOCHS, lr=LR, hidden=HIDDEN, batch_size=batch_size)

        for snr in snr_values:
            rng_jscc = np.random.RandomState(seed)
            task_model = TaskOrientedJSCC(input_dim, k, 10, rng_jscc, hidden=HIDDEN)
            train_task_oriented(task_model, X_train, y_train, snr, rng_jscc,
                                 epochs=EPOCHS, lr=LR, batch_size=batch_size)
            task_acc = float(np.mean(task_model.predict(X_test, snr, rng_jscc) == y_test))

            rng_eval = np.random.RandomState(seed)
            classical_preds = classify(clf, pca_k.decode(
                transmit_uncoded(pca_k.encode(X_test), snr, rng_eval)))
            classical_acc = float(np.mean(classical_preds == y_test))

            rows.append(dict(k=k, snr=snr, seed=seed,
                              task_oriented=task_acc, classical_uncoded=classical_acc))
    return rows


def run_dataset(name, X_all, y_all, input_dim, k_values, snr_values, out_name, batch_size=32):
    """Splits fresh for each seed from a single pool. Use for datasets
    without a pre-defined official split (optdigits, CIFAR-10)."""
    rows = []
    out_path = os.path.join(DATA_DIR, out_name)
    done_seeds = set()
    expected_per_seed = len(k_values) * len(snr_values)
    if os.path.exists(out_path):
        # Resume: keep only seeds that were fully written, since every seed's
        # split and random state are determined by the seed index alone.
        prev = pd.read_csv(out_path)
        counts = prev.groupby("seed").size()
        done_seeds = {int(s) for s, n in counts.items() if n == expected_per_seed}
        rows = prev[prev["seed"].isin(done_seeds)].to_dict("records")
        if done_seeds:
            print(f"  [{name}] resuming: seeds {sorted(done_seeds)} already complete in {out_path}")
    for seed in range(N_SEEDS):
        if seed in done_seeds:
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
        rows.extend(run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim,
                                        k_values, snr_values, seed, batch_size=batch_size))
        print(f"  [{name}] seed={seed} done")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{name}] progress saved to {out_path} ({len(rows)} rows so far)")
    return pd.DataFrame(rows)


def run_dataset_presplit(name, X_train_full, y_train_full, X_test_full, y_test_full,
                          input_dim, k_values, snr_values, out_name, batch_size=32):
    """For datasets with a genuine official train/test split (MNIST):
    every seed trains on a shuffled view of the official training set
    and evaluates on the same, complete, unchanged official test set."""
    rows = []
    out_path = os.path.join(DATA_DIR, out_name)
    for seed in range(N_SEEDS):
        perm = np.random.RandomState(seed).permutation(len(X_train_full))
        X_train, y_train = X_train_full[perm], y_train_full[perm]
        X_test, y_test = X_test_full, y_test_full
        rows.extend(run_one_seed_all_k(X_train, X_test, y_train, y_test, input_dim,
                                        k_values, snr_values, seed, batch_size=batch_size))
        print(f"  [{name}] seed={seed} done")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{name}] progress saved to {out_path} ({len(rows)} rows so far)")
    return pd.DataFrame(rows)


def summarize(df, name):
    stat_rows = []
    for k in sorted(df["k"].unique()):
        for snr in sorted(df["snr"].unique()):
            sub = df[(df["k"] == k) & (df["snr"] == snr)]
            t, p = stats.ttest_rel(sub["task_oriented"], sub["classical_uncoded"])
            winner = "JSCC" if sub["task_oriented"].mean() > sub["classical_uncoded"].mean() else "Classical"
            stat_rows.append(dict(k=k, snr=snr, winner=winner, p=p))
    stat_df = pd.DataFrame(stat_rows)
    stat_df["p_holm"] = holm_correct(stat_df["p"].values)
    stat_df["significant_uncorrected"] = stat_df["p"] < 0.05
    stat_df["significant_holm"] = stat_df["p_holm"] < 0.05

    n_jscc_unc = ((stat_df["winner"] == "JSCC") & stat_df["significant_uncorrected"]).sum()
    n_classical_unc = ((stat_df["winner"] == "Classical") & stat_df["significant_uncorrected"]).sum()
    n_jscc_holm = ((stat_df["winner"] == "JSCC") & stat_df["significant_holm"]).sum()
    n_classical_holm = ((stat_df["winner"] == "Classical") & stat_df["significant_holm"]).sum()
    n_total = len(stat_df)
    print(f"{name}: UNCORRECTED JSCC wins {n_jscc_unc}/{n_total}, Classical wins {n_classical_unc}/{n_total}")
    print(f"{name}: HOLM-CORRECTED JSCC wins {n_jscc_holm}/{n_total}, Classical wins {n_classical_holm}/{n_total}")
    return stat_df


if __name__ == "__main__":
    print("[1/3] UCI optdigits, EXCLUDING the sklearn pilot overlap "
          "(.tra portion only, 3,823 real, independent samples)...")
    X, y = load_full_optdigits("real_data_full", exclude_sklearn_overlap=True)
    print(f"  Loaded {len(X)} genuinely independent samples")
    opt_df = run_dataset("optdigits", X, y, 64, [4, 8, 16, 32], [-5, 0, 5, 10, 15, 20],
                          "optdigits_full_results.csv")
    summarize(opt_df, "Optdigits")

    print("\n[2/3] Real MNIST via OpenML, OFFICIAL train/test split genuinely preserved "
          "this time: every seed trains on a shuffled view of the same 60,000 official "
          "training images and evaluates on the same, complete, fixed 10,000 official "
          "test images -- never combined and re-split as the previous version did...")
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist_via_openml()
    print(f"  Loaded official split: {len(X_train_full)} train, {len(X_test_full)} test")
    mnist_df = run_dataset_presplit("MNIST", X_train_full, y_train_full, X_test_full, y_test_full,
                                     784, [16, 32, 64, 128], [-5, 0, 5, 10, 15, 20],
                                     "mnist_results.csv", batch_size=256)
    summarize(mnist_df, "MNIST")

    print("\n[3/3] Real CIFAR-10, FULL 50,000-image training set...")
    X_full, y_full = load_cifar10_grayscale("real_data_full/cifar-10-batches-py")
    X_full, y_full = X_full[:50000], y_full[:50000]  # official training images only
    print(f"  Loaded {len(X_full)} real training images (full CIFAR-10 training set)")
    cifar_df = run_dataset("CIFAR-10", X_full, y_full, 1024, [16, 32, 64, 128], [-5, 0, 5, 10, 15, 20],
                            "cifar10_results.csv", batch_size=256)
    summarize(cifar_df, "CIFAR-10")

    print(f"\nAll multi-dataset results saved under {DATA_DIR}")
