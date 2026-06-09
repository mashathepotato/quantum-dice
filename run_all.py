#!/usr/bin/env python
"""Reproduce everything: run the test suite, then all five experiments, writing
every figure and table into results/.

Usage (from the repo root, inside the venv):
    python run_all.py            # tests + all experiments
    python run_all.py --no-tests # experiments only
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = [
    "01_penalty_calibration.py",
    "02_slack_bits.py",
    "03_scaling.py",
    "04_diversity.py",
    "05_baseline_comparison.py",
]


def run(cmd, cwd=None):
    print(f"\n=== $ {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)} (exit {r.returncode})")


def main():
    if "--no-tests" not in sys.argv:
        run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    exp_dir = os.path.join(ROOT, "experiments")
    for script in EXPERIMENTS:
        run([sys.executable, script], cwd=exp_dir)
    print("\nAll experiments complete. Artefacts in results/.")


if __name__ == "__main__":
    main()
