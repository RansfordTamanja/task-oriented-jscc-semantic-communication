"""
run_fullgrid_digital_cnn.py
------------------------------
Extends two comparisons that run_experiments.py performs at four representative
(k, SNR) points to the FULL pilot grid (4 bandwidths x 6 SNRs x 5 seeds):

  * digital separated baseline (4-bit scalar quantization + Huffman + capacity-sized
    repetition code over a BSC) vs. the analog classical baseline (PCA + uncoded)
  * small convolutional JSCC encoder vs. the fully-connected (MLP) encoder

The per-point methodology is identical to run_experiments.py sections [2a] and [2b]
(same seeds, same splits, same random-number consumption), so the four representative
points reproduce the earlier numbers exactly.

Progress is saved after every seed and the run RESUMES automatically: if it is
interrupted, run the same command again.

Usage (from code/):
    python run_fullgrid_digital_cnn.py                 # both parts, all seeds
    python run_fullgrid_digital_cnn.py --part digital  # fast (about 1 minute)
    python run_fullgrid_digital_cnn.py --part cnn      # slow (about 1 hour on one core)
    python run_fullgrid_digital_cnn.py --part cnn --seeds 2   # run only seed 2

Outputs (in ../data/): fullgrid_digital.csv, fullgrid_cnn.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from jscc import TaskOrientedJSCC, train_task_oriented
from classical_baseline import (PCACompressor, transmit_uncoded,
                                train_classifier_on_clean, classify)
from digital_baseline import DigitalSeparatedCoder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

K_VALUES = [4, 8, 16, 32]
SNR_VALUES = [-5, 0, 5, 10, 15, 20]
N_SEEDS = 5
EPOCHS, LR, HIDDEN = 80, 0.05, 32
N_CLASSES, INPUT_DIM = 10, 64

digits = load_digits()
X_all, y_all = digits.data / 16.0, digits.target


def split(seed):
    return train_test_split(X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)


def run_digital(k, snr, seed):
    Xtr, Xte, ytr, yte = split(seed)
    rng = np.random.RandomState(seed)
    pca = PCACompressor(k).fit(Xtr)
    clf = train_classifier_on_clean(pca.decode(pca.encode(Xtr)), ytr, N_CLASSES, rng,
                                    epochs=EPOCHS, lr=LR)
    analog_acc = float(np.mean(classify(clf, pca.decode(transmit_uncoded(pca.encode(Xte), snr, rng))) == yte))
    coder = DigitalSeparatedCoder(pca, n_bits=4).fit(Xtr)
    recon, repeat_factor, ber = coder.transmit_and_decode(Xte, snr, rng)
    digital_acc = float(np.mean(classify(clf, recon) == yte))
    return dict(k=k, snr=snr, seed=seed, analog_acc=analog_acc, digital_acc=digital_acc,
                ber=ber, repeat_factor=repeat_factor)


def run_cnn(k, snr, seed):
    Xtr, Xte, ytr, yte = split(seed)
    rng = np.random.RandomState(seed)
    mlp = TaskOrientedJSCC(INPUT_DIM, k, N_CLASSES, rng, hidden=HIDDEN, encoder_type="mlp")
    train_task_oriented(mlp, Xtr, ytr, snr, rng, epochs=EPOCHS, lr=LR)
    mlp_acc = float(np.mean(mlp.predict(Xte, snr, rng) == yte))
    rng = np.random.RandomState(seed)
    cnn = TaskOrientedJSCC(INPUT_DIM, k, N_CLASSES, rng, hidden=HIDDEN, encoder_type="cnn")
    train_task_oriented(cnn, Xtr, ytr, snr, rng, epochs=EPOCHS, lr=LR)
    cnn_acc = float(np.mean(cnn.predict(Xte, snr, rng) == yte))
    return dict(k=k, snr=snr, seed=seed, mlp_acc=mlp_acc, cnn_acc=cnn_acc)


def run_part(name, fn, out_name, seeds):
    path = os.path.join(DATA_DIR, out_name)
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(int(r["k"]), int(r["snr"]), int(r["seed"])) for r in rows}
    for seed in seeds:
        todo = [(k, s) for k in K_VALUES for s in SNR_VALUES if (k, s, seed) not in done]
        if not todo:
            print(f"  [{name}] seed={seed} already complete, skipping")
            continue
        for k, snr in todo:
            rows.append(fn(k, snr, seed))
            pd.DataFrame(rows).to_csv(path, index=False)   # save after every point
        print(f"  [{name}] seed={seed} done ({len(rows)} rows saved to {path})", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["digital", "cnn", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(N_SEEDS)))
    a = ap.parse_args()
    # When only some seeds are requested (e.g. one seed per parallel process), each
    # process writes its own file so concurrent runs cannot overwrite each other.
    # analyze_fullgrid.py merges fullgrid_<part>.csv and fullgrid_<part>_seed*.csv.
    suffix = "" if sorted(a.seeds) == list(range(N_SEEDS)) else "_seed" + "_".join(map(str, a.seeds))
    if a.part in ("digital", "both"):
        run_part("digital", run_digital, f"fullgrid_digital{suffix}.csv", a.seeds)
    if a.part in ("cnn", "both"):
        run_part("cnn", run_cnn, f"fullgrid_cnn{suffix}.csv", a.seeds)
    print("Done.")
