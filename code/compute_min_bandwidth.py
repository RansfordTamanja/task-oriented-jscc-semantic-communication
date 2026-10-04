"""
compute_min_bandwidth.py
---------------------------
Computes the minimum bandwidth k needed to reach a target task accuracy,
at each tested SNR, for both JSCC and the classical baseline. Uses a
finer k grid than the main significance-testing grid specifically for
this analysis (addressing the reviewer's "with only four k values,
half the bandwidth is a single grid step" point), without disrupting
the main grid's already-verified 24-configuration statistical results.

Run:
    python compute_min_bandwidth.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import TaskOrientedJSCC, train_task_oriented
from classical_baseline import PCACompressor, transmit_uncoded, train_classifier_on_clean, classify

MINBANDWIDTH_K_VALUES = [4, 6, 8, 12, 16, 24, 32]
SNR_VALUES = [-5, 0, 5, 10, 15, 20]
N_SEEDS = 3  # fewer than the main grid's 5, since this analysis identifies
             # a threshold crossing in the mean, not a significance test
TARGET_ACC = 0.85
EPOCHS = 80
LR = 0.05
HIDDEN = 32

digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target


def run_point(k, snr, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)

    rng_jscc = np.random.RandomState(seed)
    model = TaskOrientedJSCC(64, k, 10, rng_jscc, hidden=HIDDEN)
    train_task_oriented(model, X_train, y_train, snr, rng_jscc, epochs=EPOCHS, lr=LR)
    jscc_acc = float(np.mean(model.predict(X_test, snr, rng_jscc) == y_test))

    rng_c = np.random.RandomState(seed)
    pca = PCACompressor(k).fit(X_train)
    clf = train_classifier_on_clean(pca.decode(pca.encode(X_train)), y_train, 10, rng_c,
                                     epochs=EPOCHS, lr=LR, hidden=HIDDEN)
    classical_preds = classify(clf, pca.decode(transmit_uncoded(pca.encode(X_test), snr, rng_c)))
    classical_acc = float(np.mean(classical_preds == y_test))

    return jscc_acc, classical_acc


if __name__ == "__main__":
    import sys
    only_snr = None
    if len(sys.argv) > 1:
        only_snr = int(sys.argv[1])

    results = {}
    for snr in SNR_VALUES:
        if only_snr is not None and snr != only_snr:
            continue
        print(f"SNR={snr}dB:")
        for k in MINBANDWIDTH_K_VALUES:
            jscc_accs, classical_accs = [], []
            for seed in range(N_SEEDS):
                j, c = run_point(k, snr, seed)
                jscc_accs.append(j)
                classical_accs.append(c)
            jscc_mean, classical_mean = np.mean(jscc_accs), np.mean(classical_accs)
            print(f"  k={k:3d}: JSCC={jscc_mean:.3f}, Classical={classical_mean:.3f}")
            results[(snr, k)] = (jscc_mean, classical_mean)

    print("\n=== Minimum bandwidth to reach 85% accuracy ===")
    import csv
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "min_bandwidth_results.csv")
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["snr", "jscc_min_k", "classical_min_k"])
        for snr in SNR_VALUES:
            if only_snr is not None and snr != only_snr:
                continue
            jscc_min = next((k for k in MINBANDWIDTH_K_VALUES if results[(snr, k)][0] >= TARGET_ACC), None)
            classical_min = next((k for k in MINBANDWIDTH_K_VALUES if results[(snr, k)][1] >= TARGET_ACC), None)
            print(f"SNR={snr:3d}dB: JSCC min k = {jscc_min}, Classical min k = {classical_min}")
            writer.writerow([snr, jscc_min, classical_min])
    print(f"\nSaved to {out_path}")
