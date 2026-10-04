"""
run_all.py

Runs the full JSCC study: the pilot experiment grid (with Holm
correction and effect sizes) and the multi-seed diagnostics.

Usage:
    pip install -r requirements.txt
    python run_all.py

This does NOT include the multi-dataset validation or the full-grid
digital/CNN runs, which are slower and run separately (see README.md):
    python code/run_multidataset.py          (optdigits and CIFAR-10 data are
                                             included; MNIST needs internet)
    python code/run_fullgrid_digital_cnn.py  (digital baseline + CNN encoder)
All results are already saved in ./data, so analysis and figures can be
regenerated in seconds without re-running any training.
"""
import subprocess
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(THIS_DIR, "code")


def run_step(description, script):
    print("=" * 70)
    print(description)
    print("=" * 70)
    subprocess.run([sys.executable, script], check=True, cwd=CODE_DIR)
    print()


run_step("STEP 1/2: Main pilot experiment grid (with Holm correction, "
         "effect sizes, and the finer 7-point bandwidth grid)",
         "run_experiments.py")

run_step("STEP 2/2: Multi-seed optimization diagnostics (hidden-size and "
         "learning-rate sweeps, addressing the reviewer's variance concern)",
         "run_diagnostics.py")

print("Done. See ./data for CSVs/JSON and ./figures for PNGs.")
print("See code/digital_baseline.py separately for the true digital "
      "separated-coding comparison (not yet wired into run_experiments.py "
      "-- see README.md for why and how to integrate it).")
