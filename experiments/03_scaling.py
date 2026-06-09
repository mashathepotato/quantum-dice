"""Experiment 03 — Scaling.

How do SA cost and quality scale with the instance size, and where does SA start
to struggle relative to the exact ILP reference?

We vary the number of jobs N (days scale as ~N/5), keep A=2, and for each size
measure, averaged over several seeds:
  * SA wall time (for a fixed read/sweep budget) and the ILP (CBC) solve time,
  * SA feasibility rate over reads,
  * optimality gap of the best feasible SA plan vs the ILP optimum.

A secondary sweep varies the number of tiers A at fixed N (the QUBO has N·A
decision bits plus slack), to show tier scaling.

Honest framing: this is NOT a speed claim for SA over the ILP — CBC solves these
small/medium instances to proven optimality very fast. The point is to locate the
size where SA's feasibility/quality begins to degrade under a fixed sampling
budget, which is the regime where more sampling effort (or, at Stage 3, p-bit
hardware) would be needed.

Outputs: results/03_scaling.png, results/03_scaling_tiers.png,
         results/03_scaling.md
"""

import os
import time
import numpy as np

from _common import RESULTS, run_sa, feasibility_rate, best_feasible, write_table, matplotlib_noninteractive
from src.data import generate_instance
from src.formulation import Penalties, objective_coeff_max, build_bqm
from src.baselines import solve_ilp
from src.metrics import optimality_gap

plt = matplotlib_noninteractive()

P_C = 0.2
SEEDS = [1, 2, 3]
NUM_READS = 200
NUM_SWEEPS = 2000
PEN_MULT = 10.0  # P_A = P_B = 10 * S_max (well inside the feasible regime per exp 01)

# --- (A) scale N -----------------------------------------------------------
Ns = [6, 10, 16, 24, 34, 48]
rowsN = []
sa_time, ilp_time, sa_gap, sa_feas, nbits = [], [], [], [], []
for N in Ns:
    D = max(2, N // 5)
    t_sa, t_ilp, gaps, frs = [], [], [], []
    nb = None
    for seed in SEEDS:
        inst = generate_instance(N=N, A=2, D=D, seed=seed, credit_unit=2_000.0, cap_tightness=0.3)
        Smax = objective_coeff_max(inst)
        pen = Penalties(P_A=PEN_MULT * Smax, P_B=PEN_MULT * Smax, P_C=P_C)
        nb = build_bqm(inst, pen).num_variables
        t0 = time.perf_counter()
        ilp = solve_ilp(inst, P_C=P_C)
        t_ilp.append(time.perf_counter() - t0)
        ss, dt = run_sa(inst, pen, num_reads=NUM_READS, seed=seed, num_sweeps=NUM_SWEEPS)
        t_sa.append(dt)
        frs.append(feasibility_rate(inst, ss))
        bf = best_feasible(inst, ss, P_C)
        if bf and ilp.feasible:
            gaps.append(optimality_gap(bf[1], ilp.soft_objective))
    sa_time.append(np.mean(t_sa))
    ilp_time.append(np.mean(t_ilp))
    sa_feas.append(np.mean(frs))
    g = np.mean(gaps) if gaps else np.nan
    sa_gap.append(g)
    nbits.append(nb)
    rowsN.append([N, D, nb, f"{np.mean(frs):.2f}",
                  "—" if np.isnan(g) else f"{g:+.3f}",
                  f"{np.mean(t_sa)*1000:.0f}", f"{np.mean(t_ilp)*1000:.0f}"])
    print(f"N={N:3d} D={D} bits={nb:3d} feas={np.mean(frs):.2f} "
          f"gap={'nan' if np.isnan(g) else round(g,3)} "
          f"SA={np.mean(t_sa)*1000:.0f}ms ILP={np.mean(t_ilp)*1000:.0f}ms")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(Ns, np.array(sa_time) * 1000, "o-", label="SA (200 reads)")
axes[0].plot(Ns, np.array(ilp_time) * 1000, "s-", label="ILP (CBC, exact)")
axes[0].set_yscale("log"); axes[0].set_xlabel("N jobs"); axes[0].set_ylabel("wall time (ms)")
axes[0].set_title("[SYNTHETIC] Wall time vs N"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(Ns, sa_feas, "o-", color="tab:blue")
axes[1].set_xlabel("N jobs"); axes[1].set_ylabel("SA feasibility rate"); axes[1].set_ylim(-0.05, 1.05)
axes[1].set_title("[SYNTHETIC] SA feasibility vs N\n(fixed 200 reads / 2000 sweeps)"); axes[1].grid(alpha=0.3)

axes[2].plot(Ns, [0 if np.isnan(g) else g for g in sa_gap], "o-", color="tab:red")
axes[2].set_xlabel("N jobs"); axes[2].set_ylabel("opt. gap of best feasible SA")
axes[2].set_title("[SYNTHETIC] SA quality vs N\n(best feasible vs ILP optimum)"); axes[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "03_scaling.png"), dpi=120)

# --- (B) scale tiers A -----------------------------------------------------
As = [2, 3, 4, 5]
rowsA = []
feasA, gapA, bitsA = [], [], []
for A in As:
    frs, gaps, nb = [], [], None
    for seed in SEEDS:
        inst = generate_instance(N=12, A=A, D=3, seed=seed, credit_unit=2_000.0, cap_tightness=0.3)
        Smax = objective_coeff_max(inst)
        pen = Penalties(P_A=PEN_MULT * Smax, P_B=PEN_MULT * Smax, P_C=P_C)
        nb = build_bqm(inst, pen).num_variables
        ilp = solve_ilp(inst, P_C=P_C)
        ss, _ = run_sa(inst, pen, num_reads=NUM_READS, seed=seed, num_sweeps=NUM_SWEEPS)
        frs.append(feasibility_rate(inst, ss))
        bf = best_feasible(inst, ss, P_C)
        if bf and ilp.feasible:
            gaps.append(optimality_gap(bf[1], ilp.soft_objective))
    feasA.append(np.mean(frs)); gapA.append(np.mean(gaps) if gaps else np.nan); bitsA.append(nb)
    rowsA.append([A, nb, f"{np.mean(frs):.2f}", "—" if not gaps else f"{np.mean(gaps):+.3f}"])
    print(f"A={A} bits={nb} feas={np.mean(frs):.2f} gap={round(np.mean(gaps),3) if gaps else 'nan'}")

fig2, ax = plt.subplots(figsize=(6.2, 4.3))
ax.plot(As, feasA, "o-", label="feasibility rate")
ax.plot(As, [0 if np.isnan(g) else g for g in gapA], "s-", label="opt. gap (best feasible)")
ax.set_xlabel("number of tiers A (N=12, D=3)"); ax.set_xticks(As)
ax.set_title("[SYNTHETIC] Scaling in tiers A")
ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS, "03_scaling_tiers.png"), dpi=120)

write_table(os.path.join(RESULTS, "03_scaling.md"),
            ["N", "D", "QUBO bits", "SA feas.", "SA gap", "SA ms", "ILP ms"], rowsN)
with open(os.path.join(RESULTS, "03_scaling.md"), "a") as fh:
    fh.write("\n\n### Tier scaling (N=12, D=3)\n\n")
    fh.write("| A tiers | QUBO bits | SA feas. | SA gap |\n| --- | --- | --- | --- |\n")
    for r in rowsA:
        fh.write("| " + " | ".join(str(c) for c in r) + " |\n")
print("\nWrote results/03_scaling.png, 03_scaling_tiers.png, 03_scaling.md")
