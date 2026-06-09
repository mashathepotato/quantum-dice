"""Experiment 04 — Solution diversity and re-sampling.

The honest value proposition of a probabilistic sampler (SA now, p-bits at
Stage 3) is NOT that it beats an exact solver on a single instance. It is that a
single run returns a DISTRIBUTION of low-energy solutions: many distinct, valid,
near-optimal routing plans the Supervisor can choose among on preferences not
encoded in Q (latency, provider mix, fairness across agents), and that it can be
re-sampled cheaply every dispatch cycle as costs and the budget drift.

This experiment shows, on a fixed instance:
  (1) the energy distribution over reads (not a single point),
  (2) how many DISTINCT feasible plans fall within x% of the optimum, and
  (3) that those near-optimal plans are genuinely different routings (they
      escalate different jobs), quantified by mean pairwise Hamming distance.

We also confirm the optimum the sampler clusters around matches the ILP optimum.

Outputs: results/04_diversity_energy.png, results/04_diversity_count.png,
         results/04_diversity.md
"""

import os
from itertools import combinations
import numpy as np

from _common import (
    RESULTS, run_sa, distinct_feasible_assignments, best_feasible,
    write_table, matplotlib_noninteractive,
)
from src.data import generate_instance
from src.formulation import Penalties, objective_coeff_max
from src.baselines import solve_ilp

plt = matplotlib_noninteractive()

SEED = 5
P_C = 0.2
NUM_READS = 2000
NUM_SWEEPS = 2000

inst = generate_instance(N=14, A=2, D=3, seed=SEED, credit_unit=2_000.0, cap_tightness=0.35)
Smax = objective_coeff_max(inst)
pen = Penalties(P_A=10 * Smax, P_B=10 * Smax, P_C=P_C)
print(inst.summary())

ilp = solve_ilp(inst, P_C=P_C)
opt = ilp.soft_objective
print(f"ILP optimum (soft) = {opt:,.1f}")

ss, _ = run_sa(inst, pen, num_reads=NUM_READS, seed=SEED, num_sweeps=NUM_SWEEPS)
bf = best_feasible(inst, ss, P_C)
print(f"best feasible SA soft objective = {bf[1]:,.1f}  (gap {100*(bf[1]-opt)/abs(opt):+.2f}%)")

# distinct feasible plans and their objectives
distinct = distinct_feasible_assignments(inst, ss, P_C)
objs = np.array(sorted(distinct.values()))
print(f"distinct FEASIBLE plans sampled: {len(distinct)}")

# energies of all reads (for the distribution figure)
energies = ss.record.energy

# --- (1) energy distribution -----------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.hist(energies, bins=60, color="tab:blue", alpha=0.8)
ax.set_xlabel("BQM energy of sampled read")
ax.set_ylabel("count")
ax.set_title(f"[SYNTHETIC] SA returns a DISTRIBUTION over {NUM_READS} reads\n{inst.name}")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "04_diversity_energy.png"), dpi=120)

# --- (2) #distinct near-optimal feasible plans vs tolerance ----------------
tols = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
counts = []
for t in tols:
    thr = opt + abs(opt) * t
    counts.append(int(np.sum(objs <= thr + 1e-9)))
fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
ax2.bar([f"{int(t*100)}%" for t in tols], counts, color="tab:green", alpha=0.85)
ax2.set_xlabel("tolerance above ILP optimum")
ax2.set_ylabel("# distinct feasible plans")
ax2.set_title(f"[SYNTHETIC] Distinct near-optimal feasible routings\n(one SA run, {NUM_READS} reads)")
for i, c in enumerate(counts):
    ax2.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
ax2.grid(alpha=0.3, axis="y")
fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS, "04_diversity_count.png"), dpi=120)

# --- (3) are the near-optimal plans genuinely different? -------------------
within5 = [np.array(a) for a, o in distinct.items() if o <= opt + abs(opt) * 0.05 + 1e-9]
if len(within5) >= 2:
    dists = [np.sum(a != b) for a, b in combinations(within5, 2)]
    mean_h = float(np.mean(dists))
    # which jobs vary in tier across these near-optimal plans?
    stacked = np.stack(within5)
    varying = int(np.sum(stacked.min(axis=0) != stacked.max(axis=0)))
else:
    mean_h, varying = 0.0, 0
print(f"plans within 5% of optimum: {len(within5)}; mean pairwise Hamming "
      f"distance = {mean_h:.2f} jobs; {varying}/{inst.N} jobs differ in tier across them")

rows = [
    ["ILP optimum (soft)", f"{opt:,.1f}"],
    ["best feasible SA (soft)", f"{bf[1]:,.1f}"],
    ["SA gap to optimum", f"{100*(bf[1]-opt)/abs(opt):+.2f}%"],
    ["distinct feasible plans sampled", len(distinct)],
    ["distinct plans within 1% of optimum", counts[tols.index(0.01)]],
    ["distinct plans within 5% of optimum", counts[tols.index(0.05)]],
    ["mean pairwise Hamming dist (within 5%)", f"{mean_h:.2f} jobs"],
    ["jobs that vary in tier (within 5%)", f"{varying}/{inst.N}"],
]
write_table(os.path.join(RESULTS, "04_diversity.md"), ["metric", "value"], rows)
print("\nWrote results/04_diversity_energy.png, 04_diversity_count.png, 04_diversity.md")
