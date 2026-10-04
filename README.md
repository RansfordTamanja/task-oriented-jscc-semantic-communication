# Task-Oriented Semantic Communication via Deep JSCC

Code for the paper of the same name, submitted to IEEE ICAST 2026 (Paper ID 40).

## Results

Holm-corrected significant wins across the 24-cell grid (4 bandwidths × 6 SNRs, 5 seeds):

| Dataset | JSCC wins | Classical wins | Data file in `data/` |
|---|---|---|---|
| Pilot (sklearn digits) | 1 | 5 | `pilot_table_raw_results.csv`, `pilot_table1_consolidated.csv` |
| UCI optdigits (.tra), k ∈ {4, 8, 16, 32} | 4 | 6 | `optdigits_full_results.csv` |
| MNIST (official 60k/10k split), k ∈ {16, 32, 64, 128} | 6 | 1 | `mnist_results.csv` |
| CIFAR-10 (full 50k), k ∈ {16, 32, 64, 128} | 0 | 17 | `cifar10_results.csv` |

Each results CSV has one row per (k, SNR, seed) with `task_oriented` and `classical_uncoded` accuracy. A complete file has 120 rows; fewer rows means a partial run.

## Setup

Python 3.10+ is recommended.

    pip install -r requirements.txt

## Reproducing tables and figures (no training)

All results are saved as CSVs in `data/`, so every table, statistic, and figure can be regenerated in seconds:

    cd code
    python compute_effect_sizes.py
    python analyze_fullgrid.py
    python generate_figure5.py
    python generate_figure6.py
    python generate_figure7.py
    python generate_figure8.py
    python generate_figure9.py

Figures 1–4 are produced by `run_experiments.py`. Figure 9 fills in a panel for each dataset whose results CSV is present.

## Re-running training

    python run_all.py                                   # main pilot grid + diagnostics
    cd code
    python compute_min_bandwidth.py                     # minimum-bandwidth table
    python run_fullgrid_digital_cnn.py --part digital   # digital baseline, full grid
    python run_fullgrid_digital_cnn.py --part cnn       # CNN encoder, full grid
    python run_multidataset.py                          # optdigits, MNIST, CIFAR-10
    python run_cifar10_only.py                          # CIFAR-10 alone

Every training script saves progress after each point or seed and resumes automatically if interrupted; just run the same command again.

### CNN grid in parallel (one seed per process)

Each process writes its own `fullgrid_cnn_seedN.csv`, and `analyze_fullgrid.py` merges them.

    # Windows PowerShell, from code/
    0..4 | ForEach-Object { Start-Process python -ArgumentList "run_fullgrid_digital_cnn.py --part cnn --seeds $_" }

    # macOS / Linux, from code/
    for s in 0 1 2 3 4; do python run_fullgrid_digital_cnn.py --part cnn --seeds $s & done; wait

    python analyze_fullgrid.py

### CIFAR-10 on Kaggle (one seed per notebook)

In `kaggle/kaggle_cifar10_standalone.py`, set `SEEDS_TO_RUN` to `[0]`, `[1]`, … `[4]` in up to five notebooks (Internet on, no GPU) and run "Save & Run All (Commit)". Place each `cifar10_results_seedN.csv` in `data/`, then from `code/`:

    python combine_seed_results.py

This writes `data/cifar10_results.csv`, `data/cifar10_accuracy_vs_snr.png` (mean ± std over 5 seeds), and regenerates Figure 9. Seeds are independent and deterministic, so the combined result matches a single full run.

## Outputs

`run_fullgrid_digital_cnn.py` and `analyze_fullgrid.py` write `fullgrid_digital.csv`, `fullgrid_cnn.csv`, and the comparison tables `fullgrid_digital_vs_analog.csv`, `fullgrid_digital_vs_jscc.csv`, `fullgrid_cnn_vs_classical.csv`, and `fullgrid_cnn_vs_mlp.csv` (Holm-corrected paired tests per family of 24).

`compute_effect_sizes.py` writes `effect_sizes_<dataset>.csv` for pilot, optdigits, mnist, and cifar10: for each cell, the paired JSCC-minus-classical mean difference, its 95% CI, d_z, the raw p-value, and the Holm-adjusted p-value.

`compute_min_bandwidth.py` writes `min_bandwidth_results.csv`, using a finer 7-point bandwidth grid (`MINBANDWIDTH_K_VALUES`) separate from the main 4-point grid.

## Approximate run times (one core)

| Step | Time |
|---|---|
| `run_experiments.py` | 20–30 min |
| `run_diagnostics.py` | 5–10 min |
| `run_fullgrid_digital_cnn.py --part digital` | ~1 min |
| `run_fullgrid_digital_cnn.py --part cnn` | ~55 min (~11 min per seed) |
| optdigits | tens of minutes |
| MNIST | ~30–35 min per seed, 2.5–3 h total |
| CIFAR-10 | ~30 min per seed, ~2.5 h total |
| Analysis and figure scripts | seconds |

## Methodology notes

- All main results use 5 seeds with Holm correction, Cohen's d, and 95% confidence intervals.
- Networks are trained with Adam.
- optdigits uses only the `.tra` portion to avoid overlap with the sklearn digits pilot set.
- MNIST training data stays within the official 60,000 training images; evaluation always uses the full official 10,000-image test set.
- PCA is fit once per seed and truncated to each k; the classical classifier is trained once per (seed, k) and reused across SNRs.
- The JPEG/JPEG2000 comparison uses the dataset's measured bit depth of 5 bits/pixel.
- The CNN encoder (`cnn_layers.py`) uses im2col-based Conv2D and MaxPool2D layers, with the Conv2D backward pass verified against numerical gradients.

## Data

`code/real_data_full/` contains the UCI optdigits training file (`optdigits.tra`, 3,823 samples) and the complete CIFAR-10 dataset (`cifar-10-batches-py/`, 60,000 images). MNIST is downloaded automatically from OpenML by `load_mnist_via_openml()` on first run, so a network connection is required.

## Folder structure

```
jscc_pipeline/
├── requirements.txt
├── run_all.py
├── kaggle/
│   └── kaggle_cifar10_standalone.py
├── code/
│   ├── channel.py
│   ├── nn_layers.py                     Adam and momentum-SGD
│   ├── cnn_layers.py                    Conv2D / MaxPool2D
│   ├── jscc.py                          MLP or CNN encoder
│   ├── classical_baseline.py
│   ├── digital_baseline.py              digital separated coding
│   ├── jpeg_comparison.py
│   ├── real_dataset_loaders.py
│   ├── run_experiments.py               main grid, Figures 1–4
│   ├── run_diagnostics.py
│   ├── run_multidataset.py              optdigits, MNIST, CIFAR-10
│   ├── run_cifar10_only.py
│   ├── run_fullgrid_digital_cnn.py
│   ├── combine_seed_results.py
│   ├── compute_min_bandwidth.py
│   ├── compute_effect_sizes.py
│   ├── analyze_fullgrid.py
│   ├── generate_figure5.py … generate_figure9.py
│   ├── investigate_hidden64_anomaly.py
│   ├── investigate_cnn_encoder.py
│   └── real_data_full/
├── data/
└── figures/
```