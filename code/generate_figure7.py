"""
generate_figure7.py
----------------------
Figure 7: minimum bandwidth k to reach 85% task accuracy, by SNR.
Reads real data from min_bandwidth_results.csv, produced by running
compute_min_bandwidth.py for each SNR value first.

Run (after compute_min_bandwidth.py has been run for all 6 SNR values):
    python generate_figure7.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "min_bandwidth_results.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"{csv_path} not found. Run compute_min_bandwidth.py for each SNR "
        f"value first (e.g. `python compute_min_bandwidth.py -5`, then 0, "
        f"5, 10, 15, 20) to generate it.")

results_df = pd.read_csv(csv_path).sort_values("snr")
SNR_VALUES = results_df["snr"].tolist()
JSCC_MIN_K = [None if pd.isna(v) else int(v) for v in results_df["jscc_min_k"]]
CLASSICAL_MIN_K = [None if pd.isna(v) else int(v) for v in results_df["classical_min_k"]]

fig, ax = plt.subplots(figsize=(6, 4.5))

snr_jscc = [s for s, k in zip(SNR_VALUES, JSCC_MIN_K) if k is not None]
k_jscc = [k for k in JSCC_MIN_K if k is not None]
snr_classical = [s for s, k in zip(SNR_VALUES, CLASSICAL_MIN_K) if k is not None]
k_classical = [k for k in CLASSICAL_MIN_K if k is not None]

ax.plot(snr_jscc, k_jscc, marker="o", color="#2a6ebb", label="JSCC")
ax.plot(snr_classical, k_classical, marker="s", color="#c0392b", label="Classical")

# Mark not-reached SNR values explicitly
for s in SNR_VALUES:
    if s not in snr_jscc:
        ax.scatter([s], [0], marker="x", color="gray", s=40)

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Minimum bandwidth k for 85% accuracy")
ax.set_title("Minimum bandwidth to reach 85% task accuracy, by SNR")
ax.legend()
ax.set_ylim(bottom=-2)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "figure7_min_bandwidth.png"), dpi=150)
print(f"Saved to {FIG_DIR}/figure7_min_bandwidth.png")
