"""Experiment 01 — Penalty calibration and the cap-encoding iteration.

Two questions:

(A) How large must the HARD penalties P_A (one-hot) and P_B (daily cap) be,
    relative to the objective scale, for SA to return feasible AND near-optimal
    plans? We sweep P_A = alpha · S_max and P_B = beta · S_max, where
    S_max = max_{i,a} |c - v| is the largest single objective coefficient — the
    Lagrangian bound a hard penalty must beat (a single constraint-cheating flip
    can buy at most ~S_max of objective). For each (alpha, beta) we report the
    feasibility rate over SA reads and the optimality gap of the best feasible
    SA solution vs the ILP optimum.

    Expected: too LOW -> cheating is cheaper than the penalty -> infeasible reads
    dominate; adequate -> high feasibility, ~0 gap; too HIGH -> penalty ridges
    dwarf the objective, sampling degrades (gap creeps up / feasibility wobbles).

(B) The cap-encoding iteration. Our v1 encoded the literal cap Σ c̃·x + slack = B̃
    ("absolute"); SA satisfied it only ~5% of the time because every job on a day
    — cheap or not — sits in one big quadratic clique with a large constant
    baseline. v2 ("shifted") encodes Σ Δ̃·x + slack = headroom, so cheap-tier
    choices drop out of the clique. Identical at feasible points, but far easier
    to sample. We plot feasibility rate vs penalty for both encodings on the same
    instance to show the improvement.

Outputs: results/01_penalty_feasibility.png, results/01_penalty_gap.png,
         results/01_encoding_comparison.png, results/01_penalty_calibration.md
"""

import os
import numpy as np

from _common import (
    RESULTS, run_sa, feasibility_rate, best_feasible, write_table,
    matplotlib_noninteractive,
)
from src.data import generate_instance
from src.formulation import Penalties, objective_coeff_max
from src.baselines import solve_ilp
from src.metrics import optimality_gap

plt = matplotlib_noninteractive()

SEED = 7
NUM_READS = 200
NUM_SWEEPS = 2000
P_C = 0.2  # soft coupling, fixed throughout this sweep

inst = generate_instance(N=14, A=2, D=3, seed=SEED, credit_unit=2_000.0)
Smax = objective_coeff_max(inst)
print(inst.summary())
print(f"S_max = max|c - v| = {Smax:,.0f} (penalties reported as multiples of S_max)")

ilp = solve_ilp(inst, P_C=P_C)
assert ilp.feasible, "ILP infeasible — instance misconfigured"
ref = ilp.soft_objective
print(f"ILP optimum (soft objective) = {ref:,.1f}")

# --- (A) penalty sweep (shifted encoding) ----------------------------------
alphas = [0.25, 0.5, 1.0, 3.0, 10.0, 30.0]   # P_A multipliers
betas = [0.25, 0.5, 1.0, 3.0, 10.0, 30.0]    # P_B multipliers
feas = np.zeros((len(alphas), len(betas)))
gap = np.full((len(alphas), len(betas)), np.nan)
rows = []
for ia, alpha in enumerate(alphas):
    for ib, beta in enumerate(betas):
        pen = Penalties(P_A=alpha * Smax, P_B=beta * Smax, P_C=P_C)
        ss, _ = run_sa(inst, pen, num_reads=NUM_READS, seed=SEED, num_sweeps=NUM_SWEEPS)
        fr = feasibility_rate(inst, ss)
        feas[ia, ib] = fr
        bf = best_feasible(inst, ss, P_C)
        g = optimality_gap(bf[1], ref) if bf else np.nan
        gap[ia, ib] = g
        rows.append([f"{alpha:g}", f"{beta:g}", f"{fr:.2f}",
                     "—" if bf is None else f"{g:+.3f}"])


def _heatmap(mat, title, fname, cbar_label, fmt, cmap):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto",
                   vmin=0 if cmap == "viridis" else None,
                   vmax=1 if cmap == "viridis" else None)
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel(r"$P_B / S_{\max}$ (daily-cap penalty)")
    ax.set_ylabel(r"$P_A / S_{\max}$ (one-hot penalty)")
    ax.set_title(title)
    for ia in range(len(alphas)):
        for ib in range(len(betas)):
            if not np.ma.is_masked(masked[ia, ib]):
                ax.text(ib, ia, fmt.format(mat[ia, ib]), ha="center", va="center",
                        fontsize=8, color="white" if (cmap == "viridis" and mat[ia, ib] < 0.5) else "black")
    fig.colorbar(im, label=cbar_label)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, fname), dpi=120)


_heatmap(feas, f"[SYNTHETIC] Feasibility rate over SA reads\n{inst.name} (shifted cap encoding)",
         "01_penalty_feasibility.png", "feasibility rate", "{:.2f}", "viridis")
_heatmap(gap, "[SYNTHETIC] Optimality gap of best feasible SA soln\n(vs ILP optimum; blank = no feasible read)",
         "01_penalty_gap.png", "relative optimality gap", "{:+.2f}", "magma_r")

# --- (B) encoding comparison: shifted vs absolute --------------------------
mults = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
feas_shifted, feas_absolute = [], []
for m in mults:
    pen = Penalties(P_A=m * Smax, P_B=m * Smax, P_C=P_C)
    ss_s = run_sa(inst, pen, num_reads=NUM_READS, seed=SEED, num_sweeps=NUM_SWEEPS)[0]
    feas_shifted.append(feasibility_rate(inst, ss_s))
    # absolute encoding via a fresh BQM build
    from src.formulation import build_bqm
    from src.solvers import SASampler
    bqm_abs = build_bqm(inst, pen, cap_mode="absolute")
    ss_a = SASampler(seed=SEED, num_sweeps=NUM_SWEEPS).sample(bqm_abs, num_reads=NUM_READS)
    feas_absolute.append(feasibility_rate(inst, ss_a))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(mults, feas_shifted, "o-", label="shifted  (Σ Δ·x + slack = headroom)  [v2]")
ax.plot(mults, feas_absolute, "s--", label="absolute (Σ c·x + slack = cap)  [v1]")
ax.set_xscale("log")
ax.set_xlabel(r"penalty multiple $P_A=P_B = m\cdot S_{\max}$")
ax.set_ylabel("feasibility rate over SA reads")
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"[SYNTHETIC] Cap-encoding iteration (v1 -> v2)\n{inst.name}, {NUM_SWEEPS} sweeps, {NUM_READS} reads")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "01_encoding_comparison.png"), dpi=120)

# --- table + regime summary ------------------------------------------------
write_table(os.path.join(RESULTS, "01_penalty_calibration.md"),
            ["P_A/Smax", "P_B/Smax", "feasibility", "opt. gap"], rows)

good = [(alphas[ia], betas[ib], feas[ia, ib], gap[ia, ib])
        for ia in range(len(alphas)) for ib in range(len(betas))
        if feas[ia, ib] >= 0.5 and np.isfinite(gap[ia, ib]) and gap[ia, ib] <= 0.05]
print("\nFeasible regime (feasibility>=0.5, gap<=5%):")
for a, b, fr, g in good:
    print(f"  P_A={a:g}·Smax, P_B={b:g}·Smax -> feas={fr:.2f}, gap={g:+.3f}")
print(f"\nEncoding comparison feasibility (shifted vs absolute):")
for m, fs, fa in zip(mults, feas_shifted, feas_absolute):
    print(f"  m={m:5g}: shifted={fs:.2f}  absolute={fa:.2f}")
print("\nWrote results/01_penalty_feasibility.png, 01_penalty_gap.png, "
      "01_encoding_comparison.png, 01_penalty_calibration.md")
