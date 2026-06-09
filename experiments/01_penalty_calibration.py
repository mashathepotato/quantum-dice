"""Experiment 01 — Penalty calibration.

Question: how large must the HARD penalties P_A (one-hot) and P_B (daily cap) be,
relative to the objective scale, for SA to return feasible AND near-optimal plans?

We sweep P_A = alpha · S and P_B = beta · S, where S = objective_scale(instance)
(the sum of |c - v|, an upper bound on the objective an infeasible solution could
"buy" by cheating a constraint). For each (alpha, beta) we measure:
  * feasibility rate over all SA reads, and
  * optimality gap of the best feasible SA solution vs the ILP optimum.

Expected story:
  * alpha/beta too LOW  -> cheating a constraint is cheaper than paying the penalty
    -> infeasible solutions dominate (low feasibility rate).
  * alpha/beta high enough -> feasible regime: high feasibility, small gap.
  * alpha/beta too HIGH -> penalty terms dwarf the objective; the energy landscape
    is dominated by huge penalty ridges, sampling gets harder and the best feasible
    objective degrades (gap grows) even though feasibility stays high.

Outputs: results/01_penalty_feasibility.png, results/01_penalty_gap.png,
         results/01_penalty_calibration.md
"""

import numpy as np

from _common import (
    RESULTS, run_sa, feasibility_rate, best_feasible, write_table,
    matplotlib_noninteractive,
)
import os

from src.data import generate_instance
from src.formulation import Penalties, objective_scale
from src.baselines import solve_ilp
from src.metrics import optimality_gap

plt = matplotlib_noninteractive()

SEED = 7
NUM_READS = 150
P_C = 0.2  # soft coupling, fixed throughout this sweep

inst = generate_instance(N=12, A=2, D=3, seed=SEED, credit_unit=2_000.0)
S = objective_scale(inst)
print(inst.summary())
print(f"objective_scale S = {S:,.0f} (penalties reported as multiples of S)")

# ILP optimum as the reference for the optimality gap.
ilp = solve_ilp(inst, P_C=P_C)
assert ilp.feasible, "ILP infeasible — instance misconfigured"
ref = ilp.soft_objective
print(f"ILP optimum (soft objective) = {ref:,.1f}")

alphas = [0.02, 0.1, 0.3, 1.0, 3.0, 10.0]   # P_A multipliers
betas = [0.02, 0.1, 0.3, 1.0, 3.0, 10.0]    # P_B multipliers

feas = np.zeros((len(alphas), len(betas)))
gap = np.full((len(alphas), len(betas)), np.nan)

rows = []
for ia, alpha in enumerate(alphas):
    for ib, beta in enumerate(betas):
        pen = Penalties(P_A=alpha * S, P_B=beta * S, P_C=P_C)
        ss, _ = run_sa(inst, pen, num_reads=NUM_READS, seed=SEED)
        fr = feasibility_rate(inst, ss)
        feas[ia, ib] = fr
        bf = best_feasible(inst, ss, P_C)
        g = optimality_gap(bf[1], ref) if bf else np.nan
        gap[ia, ib] = g
        rows.append([f"{alpha:g}", f"{beta:g}", f"{fr:.2f}",
                     "—" if bf is None else f"{g:+.3f}"])

# --- figures ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(feas, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
ax.set_xlabel(r"$P_B / S$ (daily-cap penalty)")
ax.set_ylabel(r"$P_A / S$ (one-hot penalty)")
ax.set_title(f"[SYNTHETIC] Feasibility rate over SA reads\n{inst.name}")
for ia in range(len(alphas)):
    for ib in range(len(betas)):
        ax.text(ib, ia, f"{feas[ia, ib]:.2f}", ha="center", va="center",
                color="white" if feas[ia, ib] < 0.6 else "black", fontsize=8)
fig.colorbar(im, label="feasibility rate")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "01_penalty_feasibility.png"), dpi=120)

fig2, ax2 = plt.subplots(figsize=(6, 5))
masked = np.ma.masked_invalid(gap)
im2 = ax2.imshow(masked, origin="lower", cmap="magma_r", aspect="auto")
ax2.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
ax2.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
ax2.set_xlabel(r"$P_B / S$ (daily-cap penalty)")
ax2.set_ylabel(r"$P_A / S$ (one-hot penalty)")
ax2.set_title(f"[SYNTHETIC] Optimality gap of best feasible SA soln\n(vs ILP optimum; blank = no feasible read)")
for ia in range(len(alphas)):
    for ib in range(len(betas)):
        if not np.ma.is_masked(masked[ia, ib]):
            ax2.text(ib, ia, f"{gap[ia, ib]:+.2f}",
                     ha="center", va="center", fontsize=8)
fig2.colorbar(im2, label="relative optimality gap")
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS, "01_penalty_gap.png"), dpi=120)

# --- table + regime summary ------------------------------------------------
write_table(os.path.join(RESULTS, "01_penalty_calibration.md"),
            ["P_A/S", "P_B/S", "feasibility", "opt. gap"], rows)

# Identify the feasible regime: cells with feasibility >= 0.5 and small gap.
good = [(alphas[ia], betas[ib], feas[ia, ib], gap[ia, ib])
        for ia in range(len(alphas)) for ib in range(len(betas))
        if feas[ia, ib] >= 0.5 and np.isfinite(gap[ia, ib]) and gap[ia, ib] <= 0.05]
print("\nFeasible regime (feasibility>=0.5, gap<=5%):")
for a, b, fr, g in good:
    print(f"  P_A={a:g}·S, P_B={b:g}·S  -> feas={fr:.2f}, gap={g:+.3f}")
print(f"\nWrote results/01_penalty_feasibility.png, 01_penalty_gap.png, "
      f"01_penalty_calibration.md")
