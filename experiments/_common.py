"""Shared helpers for the experiment scripts: paths, SA runs, feasibility stats,
and small table/figure utilities. Keeps each experiment script focused on its
own question.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# make `src` importable when running `python experiments/0X_*.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dimod

from src.data import Instance
from src.formulation import Penalties, build_bqm
from src.metrics import soft_objective
from src.solvers import SASampler
from src.validate import validate

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def run_sa(
    instance: Instance,
    penalties: Penalties,
    num_reads: int = 200,
    seed: int = 0,
    num_sweeps: int = 1000,
    slack_bits: Optional[Dict[int, int]] = None,
) -> Tuple[dimod.SampleSet, float]:
    """Build the BQM and sample it with SA. Returns (sampleset, wall_seconds)."""
    bqm = build_bqm(instance, penalties, slack_bits=slack_bits)
    sampler = SASampler(seed=seed, num_sweeps=num_sweeps)
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=num_reads)
    return ss, time.perf_counter() - t0


def feasibility_rate(instance: Instance, ss: dimod.SampleSet) -> float:
    """Fraction of reads (weighted by occurrence) that satisfy the hard constraints."""
    total = 0
    feasible = 0
    for datum in ss.data(fields=["sample", "num_occurrences"]):
        n = int(datum.num_occurrences)
        total += n
        if validate(instance, datum.sample).feasible:
            feasible += n
    return feasible / total if total else 0.0


def best_feasible(instance: Instance, ss: dimod.SampleSet, P_C: float) -> Optional[Tuple[List[int], float]]:
    """Lowest soft-objective FEASIBLE assignment in the sample set, or None."""
    best = None
    best_obj = float("inf")
    for datum in ss.data(fields=["sample"]):
        f = validate(instance, datum.sample)
        if not f.feasible:
            continue
        obj = soft_objective(instance, f.assignment, P_C)
        if obj is not None and obj < best_obj:
            best_obj, best = obj, f.assignment
    return (best, best_obj) if best is not None else None


def distinct_feasible_assignments(instance: Instance, ss: dimod.SampleSet, P_C: float):
    """Map {tuple(assignment): soft_objective} over distinct feasible samples."""
    out: Dict[tuple, float] = {}
    for datum in ss.data(fields=["sample"]):
        f = validate(instance, datum.sample)
        if not f.feasible:
            continue
        key = tuple(f.assignment)
        if key not in out:
            out[key] = soft_objective(instance, f.assignment, P_C)
    return out


def write_table(path: str, header: List[str], rows: List[List]) -> None:
    """Write a GitHub-flavoured markdown table."""
    with open(path, "w") as fh:
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c) for c in r) + " |\n")


def matplotlib_noninteractive():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
