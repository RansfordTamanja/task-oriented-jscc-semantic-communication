# Task-Oriented Semantic Communication via Deep JSCC

## Quick start: commands to run everything

All commands run from the project root unless noted. Python 3.10+ is recommended.

    pip install -r requirements.txt

**Reproduce every table, statistic, and figure from the saved CSVs (seconds, no training):**

    cd code
    python compute_effect_sizes.py
    python analyze_fullgrid.py
    python generate_figure5.py
    python generate_figure6.py
    python generate_figure7.py
    python generate_figure8.py
    python generate_figure9.py

**Re-run the training (slow; see the timing table below):**

    python run_all.py                                   # main pilot grid + diagnostics
    cd code
    python run_fullgrid_digital_cnn.py --part digital   # about 1 minute
    python run_fullgrid_digital_cnn.py --part cnn       # about 55 minutes
    python run_multidataset.py                          # optdigits, MNIST, CIFAR-10 (hours)
    python run_cifar10_only.py                          # CIFAR-10 alone, resumes if interrupted

**Run the CNN part faster, one seed per process.** Each process writes its own file
(`fullgrid_cnn_seedN.csv`), so parallel runs cannot overwrite each other, and
`analyze_fullgrid.py` merges them automatically.

    # Windows PowerShell, from code/
    0..4 | ForEach-Object { Start-Process python -ArgumentList "run_fullgrid_digital_cnn.py --part cnn --seeds $_" }

    # macOS / Linux, from code/
    for s in 0 1 2 3 4; do python run_fullgrid_digital_cnn.py --part cnn --seeds $s & done; wait

    # then, after all five finish:
    python analyze_fullgrid.py

Every training script saves progress after each point and resumes automatically if interrupted:
just run the same command again.

## Current status of results and data (read this first)

| Dataset | Holm-corrected wins (of 24) | Data file in `data/` |
|---|---|---|
| Pilot (sklearn digits) | JSCC 1, Classical 5 | `pilot_table_raw_results.csv` (per-seed), `pilot_table1_consolidated.csv` (means) |
| UCI optdigits (.tra only), k in {4,8,16,32} | JSCC 4, Classical 6 | `optdigits_full_results.csv` |
| MNIST (official 60k/10k split), k in {16,32,64,128} | JSCC 6, Classical 1 | `mnist_results.csv` |
| CIFAR-10 (full 50k), k in {16,32,64,128} | JSCC 0, Classical 17 | `cifar10_results.csv` (120 rows, 5 seeds) |

Each CSV has one row per (k, snr, seed) with `task_oriented` and `classical_uncoded` accuracy;
a complete 5-seed file has 120 rows. A file with fewer rows is a partial save.

### Running CIFAR-10 locally (about 2.5-3 hours)

    cd code
    python run_cifar10_only.py

Progress is saved after every seed and the run **resumes automatically**: if it is
interrupted (sleep, closed terminal), run the same command again. Outputs:
`data/cifar10_results.csv` and `data/cifar10_accuracy_vs_snr.png`.
The same resume logic applies to `run_multidataset.py`.

### Faster: CIFAR-10 on Kaggle, one seed per notebook (about 35 minutes total)

`kaggle/kaggle_cifar10_standalone.py` has a `SEEDS_TO_RUN` setting. Create up to five
notebooks (Internet on, no GPU), set `SEEDS_TO_RUN = [0]`, `[1]`, ... `[4]` in each, and
"Save & Run All (Commit)". Each notebook also saves a quick `cifar10_accuracy_vs_snr_seedN.png` so you can check it before
downloading. Put each `cifar10_results_seedN.csv` into `data/`, then run
`python combine_seed_results.py` from `code/`: it writes `data/cifar10_results.csv`,
`data/cifar10_accuracy_vs_snr.png` (mean +/- std over 5 seeds), and regenerates Figure 9
with the CIFAR-10 panel filled in. Seeds are independent and
deterministic, so the combined result is identical to a single full run. Setting
`SEEDS_TO_RUN = [0, 1, 2, 3, 4]` in one notebook runs everything (about 2.5 hours) and also
writes the figure.

### Full-grid digital baseline and convolutional encoder (pilot dataset)

`run_fullgrid_digital_cnn.py` runs the digital separated baseline and the small convolutional
JSCC encoder at all 24 (k, SNR) cells x 5 seeds, using exactly the per-point methodology of
`run_experiments.py`. The digital part takes about 1 minute; the convolutional part about 1 hour
on one core. Progress is saved after every point and the run resumes automatically.

    cd code
    python run_fullgrid_digital_cnn.py --part digital
    python run_fullgrid_digital_cnn.py --part cnn
    python analyze_fullgrid.py        # Holm-corrected paired tests per family of 24

Outputs in `data/`: `fullgrid_digital.csv`, `fullgrid_cnn.csv`, and the four comparison tables
`fullgrid_digital_vs_analog.csv`, `fullgrid_digital_vs_jscc.csv`, `fullgrid_cnn_vs_classical.csv`,
`fullgrid_cnn_vs_mlp.csv`.

### Effect sizes for every dataset

`python compute_effect_sizes.py` writes `data/effect_sizes_<dataset>.csv` (pilot, optdigits,
mnist, cifar10): for each of the 24 cells, the paired JSCC-minus-classical mean difference, its
95% confidence interval, d_z, the raw p-value, and the Holm-adjusted p-value.

### Approximate run times (one core)

| Step | Time |
|---|---|
| `run_experiments.py` (main pilot grid) | about 20-30 min (estimate) |
| `run_diagnostics.py` | about 5-10 min (estimate) |
| `run_fullgrid_digital_cnn.py --part digital` | about 1 min (measured) |
| `run_fullgrid_digital_cnn.py --part cnn` | about 55 min (measured; one seed per process is about 11 min) |
| optdigits | tens of minutes (estimate) |
| MNIST | about 30-35 min per seed, 2.5-3 h total |
| CIFAR-10 | about 30 min per seed, about 2.5 h total (see the per-seed Kaggle option) |
| `compute_effect_sizes.py`, `analyze_fullgrid.py`, `generate_figure5-9.py` | seconds |

You do not need to re-run training to reproduce the tables and figures: every result is
saved as a CSV in `data/`, and the analysis scripts regenerate all statistics from those files.

### Regenerating figures 5-9

`generate_figure5.py` through `generate_figure9.py` read the CSVs above
(figure 9 fills in each dataset panel for which a results CSV is present).
Figures 1-4 are produced by `run_experiments.py`.


Code for the paper of the same name, submitted to IEEE ICAST 2026 (Paper ID 40).

## Latest, major gap found: 5 of 9 figures had NO generation code at all

Checking directly: only `figure1_accuracy_vs_snr.png` through
`figure4_semantic_advantage.png` were ever actually generated by
`run_experiments.py`. Figures 5-9, referenced throughout the paper, had
zero corresponding code anywhere in this repository. Built and
genuinely tested all five:

- **`generate_figure5.py`**: real qualitative reconstructions at k=8
  across SNR (-5 to 20dB) -- actually trained and run; MSE decreases
  from 0.073 at SNR=-5 to 0.032 at SNR=20, as expected.
- **`generate_figure6.py`**: significance heatmap, reads the real
  `table2_statistical_tests.csv` -- confirmed it reproduces the exact
  1 JSCC / 5 Classical win pattern now in the paper.
- **`generate_figure7.py`**: minimum-bandwidth curve, reads
  `min_bandwidth_results.csv` -- confirmed matches the paper's
  corrected table exactly.
- **`generate_figure8.py`**: cross-dataset summary bar chart, using
  the four datasets' real, confirmed aggregate win counts.
- **`generate_figure9.py`**: three-dataset accuracy curves, reads each
  dataset's actual results CSV directly -- tested with your real
  optdigits data (plots correctly); gracefully shows a clear
  placeholder message for MNIST/CIFAR-10 if their CSVs aren't present
  yet, rather than crashing or faking data.

Also fixed in the same pass: **`compute_min_bandwidth.py` now saves
results to `min_bandwidth_results.csv`** instead of only printing them,
so `generate_figure7.py` can read real data instead of needing
hardcoded values.

## Previous round: final audit fixes

A final audit of the paper against the code found two real discrepancies,
now fixed:

1. **`run_experiments.py`'s `K_VALUES` had been expanded to 7 points
   `[4,6,8,12,16,24,32]`** for a different purpose earlier, while the paper
   consistently describes a 24-configuration (4 bandwidth x 6 SNR) grid.
   Reverted `K_VALUES` to the original 4-point grid for the main
   significance-testing results, and added a separate
   `MINBANDWIDTH_K_VALUES` (the 7-point grid) used only by the new
   `compute_min_bandwidth.py` script below -- so the finer grid serves its
   intended purpose without silently changing the main grid's
   already-verified configuration count.
2. **`compute_min_bandwidth.py` (NEW)**: the paper's minimum-bandwidth-to-
   target-accuracy table previously had no corresponding script -- it was
   unverifiable. This script actually computes it, run for real to produce
   the paper's current table. One finding changed: the paper's SNR=10dB
   row claimed a tie (JSCC=8, Classical=8); the real computation found
   Classical actually wins there (JSCC=12, Classical=8), not a tie.

## Previous round: real_multidataset.py restructured, two new investigation scripts

An independent code review (verified claim-by-claim against the actual
code before trusting any of it) found four real issues in
`run_multidataset.py`, now fixed:

1. **MNIST's official train/test split was not actually preserved.**
   The previous version's own comment claimed to use the official test
   set directly, but the code combined train+test and re-split
   randomly anyway -- re-introducing the exact problem the original
   reviewer raised. Fixed: `run_dataset_presplit()` now keeps every
   seed's training data within the official 60,000 training images and
   evaluates on the same, complete, fixed 10,000 official test images.
2. **PCA was refit for every (k, SNR, seed) combination** despite not
   depending on k or SNR beyond truncation. Verified directly
   (`Vt[:k]`, SVD sorted descending) and confirmed with an actual test
   that truncating a larger fit gives zero numerical difference from
   refitting. Now fit once per seed.
3. **The classical classifier was retrained once per SNR** despite
   training only on the clean, noiseless reconstruction. Now trained
   once per (seed, k), reused across all SNR evaluations. JSCC is
   unchanged, since its SNR-dependence is genuine.
4. **Progress now saves after every seed**, the natural checkpoint for
   the new seed-outer loop structure.

Combined, measured speedup (not estimated): **~3.9x** at MNIST's real
scale (1,556s/seed down to 399s/seed at k=16), cutting MNIST's
estimated total time from ~8.6 hours to ~2.2 hours.

Two new standalone investigation scripts, both actually run with real
results, not just written:

- **`investigate_hidden64_anomaly.py`**: tests whether hidden=64
  underperforming hidden=32 is a real effect or an LR/undertraining
  artifact. Found it survives both an LR sweep and 2.5x more epochs --
  a genuine effect, not fixable by standard tuning.
- **`investigate_cnn_encoder.py`**: same test for the CNN encoder's
  underperformance. Found tuning closes PART of the gap (0.195->0.25)
  but not all of it -- a partially-unfair original comparison, but a
  real residual gap even after fair tuning.

## Previous round's additions: closing the remaining code-level gaps

Three things were genuinely missing before and are now built, verified,
and wired into the main pipeline:

1. **A real CNN encoder** (`cnn_layers.py`): hand-rolled Conv2D and
   MaxPool2D layers, im2col-based. The Conv2D backward pass was checked
   against a numerical gradient before being trusted (see the file's own
   `__main__` block) -- max relative error came back as 0.000000. Wired
   into `jscc.py` as `encoder_type="cnn"`, a drop-in alternative to the
   original MLP encoder, and actually run end-to-end on real digit data
   (confirmed training reduces loss and reaches above-chance accuracy).
2. **The digital baseline wired into the main experiment script**
   (previously a standalone file only). `run_experiments.py` now runs a
   direct digital-vs-analog comparison at four representative (k, SNR)
   points.
3. **A statistical power check**: re-runs one representative grid point
   at 20 seeds instead of 5, specifically to test whether the Holm-
   corrected null result your last run produced (JSCC 0/42 significant
   wins) is a real null finding or an artifact of insufficient power at
   n=5. Compare the reported p-value and effect size at n=20 against the
   same point's n=5 result in `table2_statistical_tests.csv`.

All three were actually run as part of building this, not just written
-- see the smoke-test output referenced in code comments.

## On the 0/42 result from your last run

This is the most consequential thing that's come out of this whole
update. With `N_SEEDS` correctly restored to 5 (matching the paper's
own claim) and Holm correction properly applied, **JSCC does not have a
single significant win anywhere in the 42-point grid**, while Classical
still wins in 7. This is a real result from real code, not a bug --
verified by rerunning it myself. The new power check (addition 3 above)
is specifically aimed at this: run it, and if the n=20 p-value at the
same point is still far from significant, this is very likely a genuine
null result, not an underpowered one, and the paper's central claim
needs to be reframed around what the data actually shows rather than
patched around it.

## Everything from the previous round, unchanged and still in place

- `N_SEEDS` restored to 5 (was silently reduced to 3 in the code that
  generated the paper's original numbers)
- Finer 7-point bandwidth grid (was 4 points)
- Holm correction, Cohen's d, 95% CIs on every statistical comparison
- Adam optimizer (tested against momentum-SGD: Adam slightly ahead,
  lower variance)
- Longer training tested (160 vs. 80 epochs: only marginal gain, higher
  variance)
- Dataset overlap fix (sklearn digits vs. full optdigits) in
  `real_dataset_loaders.py` and used by default in `run_multidataset.py`
- Real MNIST test set support via OpenML
- Full-scale CIFAR-10 (50,000 images, not a 2,000-image near-chance
  subsample) in `run_multidataset.py`
- Corrected JPEG/JPEG2000 comparison (`jpeg_comparison.py`): the
  dataset's real bit depth is 5 bits/pixel, measured directly, not the
  8-bit assumption the original test used

## Folder structure

```
jscc_pipeline/
├── requirements.txt
├── run_all.py
├── code/
│   ├── channel.py
│   ├── nn_layers.py                    Adam + momentum-SGD
│   ├── cnn_layers.py                     NEW: Conv2D/MaxPool2D, gradient-checked
│   ├── jscc.py                            MLP or CNN encoder, selectable
│   ├── classical_baseline.py
│   ├── digital_baseline.py                 true digital separated coding
│   ├── jpeg_comparison.py
│   ├── real_dataset_loaders.py
│   ├── run_experiments.py                    main grid + CNN/digital/power-check sections
│   ├── run_diagnostics.py
│   ├── run_multidataset.py                     RESTRUCTURED: PCA/classifier once per seed, MNIST split fixed
│   ├── investigate_hidden64_anomaly.py           NEW: confirms the capacity anomaly is real
│   └── investigate_cnn_encoder.py                NEW: fair, independently-tuned CNN comparison
├── real_data_full/                   (real optdigits.tra + complete CIFAR-10, included)
├── data/
└── figures/
```

## Real data for the multi-dataset validation -- now included directly

`code/real_data_full/optdigits.tra` (the genuine, independent 3,823-
sample UCI optdigits training portion) and
`code/real_data_full/cifar-10-batches-py/` (the complete, real 60,000-
image CIFAR-10 dataset, all 6 batches) are both included in this
delivery. Both were verified directly: optdigits loads as 3,823 samples
with pixel values 0-16 and labels 0-9; CIFAR-10 loads as 60,000 images
with the correct 10 real class names (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck). A reduced-scale run of both
through the actual training pipeline completed without error before
this was packaged.

MNIST is not included as a file -- `load_mnist_via_openml()` fetches it
automatically over the network when you run `run_multidataset.py`
yourself (this could not be verified end-to-end from the sandbox this
was built in, since OpenML's API is blocked there; only the
non-network parts of the loader were verified directly).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # main grid + diagnostics
cd code && python run_multidataset.py    # optdigits + MNIST + CIFAR-10, using the real data above
```

`run_experiments.py` now takes noticeably longer than before: the main
42-point grid, plus the CNN-vs-MLP comparison (4 points x 5 seeds x 2
encoders), the digital-vs-analog comparison (4 points x 5 seeds x 2
methods), and the n=20 power check, all in one run.

`run_multidataset.py` will take considerably longer still: full-scale
CIFAR-10 (60,000 images) across a (k, SNR, seed) grid, not the
2,000-image subsample the original version used.
