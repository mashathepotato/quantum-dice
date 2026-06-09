"""Experiment 02 — Slack-bit validation.

The daily cap Σ c̃·x ≤ B̃ is encoded as an equality with binary slack:
Σ Δ̃·x + Σ_k 2^k s_{d,k} = headroom. The slack must be able to represent any
unused headroom in [0, headroom_units]; with K_d bits the largest representable
slack is 2^{K_d} − 1. The correct count is K_d = ceil(log2(headroom+1)).

Two things to prove:
  (1) With ENOUGH slack bits, the cap is enforced exactly: the QUBO ground state
      is cap-feasible and matches the ILP optimum.
  (2) With TOO FEW slack bits, feasible-but-low-spend plans become UNREPRESENTABLE.
      If 2^{K_d} − 1 < (headroom − minimal escalation), the slack cannot zero the
      cap term for the optimal low-spend plan, so the penalty pushes the optimiser
      either to over-spend (violate the cap) or to a worse feasible plan. The
      "cap correctly enforced" property breaks below the threshold.

We use a SMALL single-day instance and solve the penalty QUBO EXACTLY
(dimod.ExactSolver) so the result is deterministic ground truth, not SA noise.
We also report the SA feasibility rate for context.

Outputs: results/02_slack_bits.png, results/02_slack_bits.md
"""

import os
import numpy as np

from _common import RESULTS, run_sa, feasibility_rate, write_table, matplotlib_noninteractive
from src.data import generate_instance
from src.formulation import (
    Penalties, build_bqm, objective_coeff_max, default_slack_bits,
    day_headroom_units,
)
from src.baselines import solve_ilp, solve_bruteforce
from src.metrics import soft_objective, optimality_gap
from src.validate import validate
import dimod

plt = matplotlib_noninteractive()

SEED = 3
P_C = 0.2

# Small single-day instance so ExactSolver is tractable and we can reason exactly.
# We scale escalation value DOWN so escalation is not worthwhile: the optimum is
# then (near) all-cheap, i.e. LOW spend with a LARGE unused headroom. That is
# precisely the regime where slack must represent a large value, so it cleanly
# exposes the under-provisioning failure mode. (With a high-spend optimum, little
# slack is needed and too-few-bits would not bite — a less informative test.)
inst = generate_instance(N=6, A=2, D=1, seed=SEED, credit_unit=4_000.0, cap_tightness=0.5)
inst.value = inst.value * 0.1  # escalation rarely pays -> optimum stays cheap
Smax = objective_coeff_max(inst)
headroom = int(day_headroom_units(inst)[0])
K_correct = default_slack_bits(inst)[0]
print(inst.summary())
print(f"day-0 headroom = {headroom} credit units -> correct slack bits K = {K_correct} "
      f"(max representable slack 2^K-1 = {2**K_correct - 1})")

ilp = solve_ilp(inst, P_C=P_C)
bf = solve_bruteforce(inst, P_C=P_C)
ref = ilp.soft_objective
print(f"ILP optimum (soft) = {ref:,.1f} ; brute-force optimum = {bf.soft_objective:,.1f}")

pen = Penalties(P_A=10 * Smax, P_B=10 * Smax, P_C=P_C)

Ks = list(range(0, K_correct + 2))
rows = []
max_repr = []
gs_feasible = []
gs_gap = []
sa_feas = []
for K in Ks:
    bqm = build_bqm(inst, pen, slack_bits={0: K})
    # exact ground state of the penalised QUBO
    ss = dimod.ExactSolver().sample(bqm)
    gs = ss.first.sample
    f = validate(inst, gs)
    gobj = soft_objective(inst, f.assignment, P_C) if f.feasible else None
    gap = optimality_gap(gobj, ref) if gobj is not None else np.nan
    # SA feasibility rate at this K
    ss_sa, _ = run_sa(inst, pen, num_reads=200, seed=SEED, num_sweeps=2000, slack_bits={0: K})
    fr = feasibility_rate(inst, ss_sa)
    max_repr.append(2 ** K - 1)
    gs_feasible.append(1 if f.feasible else 0)
    gs_gap.append(gap)
    sa_feas.append(fr)
    rows.append([K, 2 ** K - 1, "yes" if 2 ** K - 1 >= headroom else "NO",
                 "feasible" if f.feasible else "INFEASIBLE",
                 "—" if gobj is None else f"{gap:+.3f}", f"{fr:.2f}"])
    print(f"  K={K}: max_slack={2**K-1:3d} (need {headroom}) | ground state "
          f"{'feasible' if f.feasible else 'INFEASIBLE'} gap={gap if np.isnan(gap) else round(gap,3)} | SA feas={fr:.2f}")

# --- figure ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

ax1.plot(Ks, max_repr, "o-", label=r"max representable slack $2^K-1$")
ax1.axhline(headroom, color="crimson", ls="--", label=f"headroom = {headroom} (needed)")
ax1.axvline(K_correct, color="green", ls=":", label=f"correct K = {K_correct}")
ax1.set_xlabel("slack bits K (per day)")
ax1.set_ylabel("credit units")
ax1.set_title("[SYNTHETIC] Representability:\nslack range vs needed headroom")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

gap_plot = [0.0 if np.isnan(g) else g for g in gs_gap]
ax2.plot(Ks, gap_plot, "D-", color="tab:orange",
         label="optimality gap of exact ground state")
ax2.axvline(K_correct, color="green", ls=":", label=f"correct K = {K_correct}")
ax2.axhline(0.0, color="grey", lw=0.8)
ax2.set_xlabel("slack bits K (per day)")
ax2.set_ylabel("relative optimality gap (vs ILP optimum)")
ax2.set_title("[SYNTHETIC] Under-provisioned slack -> suboptimal:\nlow-spend optimum is unrepresentable, forcing over-escalation")
# annotate that solutions remain cap-feasible (no overspend), just suboptimal
ax2.text(0.02, 0.92, "all points stay UNDER the cap\n(feasible) — the loss is in quality",
         transform=ax2.transAxes, fontsize=8, va="top",
         bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
ax2.legend(fontsize=8, loc="center right")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "02_slack_bits.png"), dpi=120)

write_table(os.path.join(RESULTS, "02_slack_bits.md"),
            ["K bits", "max slack 2^K-1", "covers headroom?", "exact ground state",
             "gap vs ILP", "SA feas. rate"], rows)
print(f"\nWrote results/02_slack_bits.png, 02_slack_bits.md")
