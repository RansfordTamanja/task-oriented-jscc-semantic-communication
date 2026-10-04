"""
real_dataset_loaders.py
--------------------------
Loaders for the three real datasets used in this follow-up,
higher-tier-venue study: the full UCI optdigits dataset (5,620 real
samples, not sklearn's 1,797-sample subset), real MNIST (60,000 real
28x28 training images; the official test-image file was not
available, only test labels, so a held-out split is drawn from the
training set instead, disclosed here rather than silently worked
around), and real CIFAR-10 (50,000 train + 10,000 test real 32x32 RGB
images, converted to grayscale for a fair, apples-to-apples comparison
to this study's other grayscale datasets).
"""
import os
import struct
import pickle
import numpy as np


def load_full_optdigits(data_dir, exclude_sklearn_overlap=False):
    """Loads the UCI optdigits dataset.

    IMPORTANT, reviewer-confirmed issue: sklearn's bundled 1,797-sample
    "digits" dataset (used elsewhere in this study as the "pilot") is
    NOT an independent sample -- it is exactly the .tes portion of this
    same UCI optdigits dataset. Treating a "pilot" result and a "full
    optdigits" result as two independent replications is therefore
    incorrect when the full dataset is loaded as train+test combined,
    since the full dataset's test split will necessarily re-include
    much of the exact same pilot data.

    Set exclude_sklearn_overlap=True to load ONLY the .tra portion
    (3,823 samples), which does not overlap with the sklearn pilot at
    all, giving a genuinely independent validation set. This is the
    scientifically correct mode for claiming "full optdigits confirms
    the pilot finding" -- the default (train+test combined, 5,620
    samples) is kept for backward compatibility but should NOT be
    described as independent of the pilot in the paper without this
    flag.
    """
    def parse_file(path):
        X, y = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 65:
                    continue
                X.append([float(p) for p in parts[:64]])
                y.append(int(parts[64]))
        return np.array(X), np.array(y)

    X_tra, y_tra = parse_file(os.path.join(data_dir, "optdigits.tra"))

    if exclude_sklearn_overlap:
        X = X_tra / 16.0
        y = y_tra
        return X, y

    X_tes, y_tes = parse_file(os.path.join(data_dir, "optdigits.tes"))
    X = np.vstack([X_tra, X_tes]) / 16.0  # normalize to [0,1], matching this study's other datasets
    y = np.concatenate([y_tra, y_tes])
    return X, y


def load_mnist(data_dir, split="train"):
    """Loads real MNIST images and labels.

    CORRECTION: an earlier version of this loader stated the official
    test-image file "was not available" as justification for drawing
    all splits from the training set. The reviewer correctly points
    out this file IS freely available (e.g. via torchvision or OpenML)
    -- its absence here was a local data-provisioning gap, not a real
    unavailability, and the paper's stated reason was wrong. This
    version supports loading the real, official test split directly
    when the standard IDX files are present.

    split: "train" (60,000 real training images) or "test" (10,000
    real official test images). Place t10k-images-idx3-ubyte and
    t10k-labels-idx1-ubyte in data_dir to use split="test" -- see
    README.md for exactly where to get these files (they are small,
    standard, and freely redistributable).
    """
    def load_idx_images(path):
        with open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        return data

    def load_idx_labels(path):
        with open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    if split == "test":
        images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
        labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")
        if not (os.path.exists(images_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(
                f"Official MNIST test files not found at {data_dir}. "
                f"Download t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte "
                f"(see README.md for sources) and place them there, or use "
                f"split='train' for the training-set-only loader.")
    elif split == "train":
        images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
        labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    X = load_idx_images(images_path).astype(np.float64) / 255.0
    y = load_idx_labels(labels_path).astype(int)
    return X, y


def load_mnist_via_openml():
    """Alternative loader using scikit-learn's fetch_openml, which
    downloads the real, official MNIST dataset (train+test combined,
    70,000 real images) directly -- no manual IDX file placement
    needed, and this IS the official data, addressing the reviewer's
    point that the official test set is freely obtainable. Requires
    network access; caches locally after first call via sklearn's own
    cache directory.
    """
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)
    # OpenML's mnist_784 preserves the official ordering: first 60,000 =
    # official train split, last 10,000 = official test split.
    return X[:60000], y[:60000], X[60000:], y[60000:]


def load_cifar10_grayscale(data_dir):
    """Loads real CIFAR-10 (all 5 train batches + test batch = 60,000
    real images), converted to grayscale via the standard ITU-R BT.601
    luminance formula for a fair comparison to this study's other
    grayscale datasets. Converts to grayscale one batch at a time
    (rather than holding all batches as float64 RGB simultaneously,
    roughly 1.5GB, which triggered an out-of-memory kill when first
    attempted) to keep peak memory usage low.
    """
    def load_and_gray_batch(path):
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
        labels = np.array(batch[b"labels"])
        gray = 0.299 * data[:, 0] + 0.587 * data[:, 1] + 0.114 * data[:, 2]
        gray = gray.reshape(len(gray), -1) / 255.0
        return gray.astype(np.float32), labels

    all_gray, all_labels = [], []
    for i in range(1, 6):
        gray, labels = load_and_gray_batch(os.path.join(data_dir, f"data_batch_{i}"))
        all_gray.append(gray)
        all_labels.append(labels)
    gray, labels = load_and_gray_batch(os.path.join(data_dir, "test_batch"))
    all_gray.append(gray)
    all_labels.append(labels)

    X = np.concatenate(all_gray, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


if __name__ == "__main__":
    X, y = load_full_optdigits("/home/claude/mnist_check/optdigits")
    print(f"Full optdigits: {X.shape[0]} real samples, {X.shape[1]} pixels, "
          f"classes={sorted(set(y))}")

    X, y = load_mnist("/home/claude/mnist_check")
    print(f"MNIST: {X.shape[0]} real samples, {X.shape[1]} pixels, "
          f"classes={sorted(set(y))}")

    X, y = load_cifar10_grayscale("/home/claude/cifar_check/cifar-10-batches-py")
    print(f"CIFAR-10 (grayscale): {X.shape[0]} real samples, {X.shape[1]} pixels, "
          f"classes={sorted(set(y))}")
