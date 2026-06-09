"""Experiment 05 — Baseline comparison.

Compare the SA sampler against classical references on the SAME problem:
  * ExactSolver  — exact ground state of the penalty QUBO (tiny instances only),
  * brute force  — exact constrained optimum over all A^N one-hot plans (small),
  * ILP (CBC)    — proven optimum of the constrained problem (all sizes here),
  * greedy       — cheap heuristic, for context.

Metrics, averaged over seeds at each size:
  * feasibility (does the method return a cap-respecting one-hot plan?),
  * optimality gap vs the ILP optimum (the common reference),
  * agreement: does the QUBO ground state / SA best match the ILP optimum value?

Honest framing: on these sizes CBC returns the proven optimum essentially
instantly; SA is a stochastic sampler and is expected to match or slightly trail
it. The value of SA/p-bits is the sampled distribution (experiment 04) and cheap
online re-sampling, not beating CBC here. ExactSolver also confirms the
penalty-encoded QUBO has the SAME optimum as the constrained problem — i.e. the
formulation is faithful.

Outputs: results/05_baseline_comparison.png, results/05_baseline_comparison.md
"""

import os
import numpy as np

from _common import (
    RESULTS, run_sa, best_feasible, feasibility_rate, write_table,
    matplotlib_noninteractive,
)
from src.data import generate_instance
from src.formulation import Penalties, objective_coeff_max
from src.baselines import solve_ilp, solve_exact, solve_bruteforce, solve_greedy
from src.metrics import optimality_gap

plt = matplotlib_noninteractive()

P_C = 0.2
SEEDS = [1, 2, 3, 4, 5]
NUM_READS = 500
NUM_SWEEPS = 2000
PEN_MULT = 10.0

# Sizes: small enough that brute force / ExactSolver also run on the lower end.
SIZES = [(6, 1), (8, 2), (10, 2), (14, 3), (20, 4)]

rows = []
labels = []
sa_gaps, greedy_gaps = [], []
sa_feasr, exact_agree = [], []
for (N, D) in SIZES:
    g_sa, g_gr, f_sa, agree, n_agree = [], [], [], 0, 0
    bf_agree, bf_n = 0, 0  # ILP vs brute-force constrained optimum
    for seed in SEEDS:
        inst = generate_instance(N=N, A=2, D=D, seed=seed, credit_unit=2_000.0, cap_tightness=0.3)
        Smax = objective_coeff_max(inst)
        pen = Penalties(P_A=PEN_MULT * Smax, P_B=PEN_MULT * Smax, P_C=P_C)
        ilp = solve_ilp(inst, P_C=P_C)
        if not ilp.feasible:
            continue
        ref = ilp.soft_objective

        ss, _ = run_sa(inst, pen, num_reads=NUM_READS, seed=seed, num_sweeps=NUM_SWEEPS)
        bf = best_feasible(inst, ss, P_C)
        f_sa.append(feasibility_rate(inst, ss))
        if bf:
            g_sa.append(optimality_gap(bf[1], ref))

        gr = solve_greedy(inst, P_C=P_C)
        if gr.feasible:
            g_gr.append(optimality_gap(gr.soft_objective, ref))

        # ExactSolver agreement (tiny only): does the penalty QUBO ground state
        # match the ILP optimum? Confirms the formulation is faithful.
        ex = solve_exact(inst, pen, max_vars=20)
        if ex.feasible and ex.soft_objective is not None:
            agree += 1
            if abs(ex.soft_objective - ref) < 1e-6:
                n_agree += 1

        # Brute-force constrained optimum (scales further than ExactSolver):
        # confirms the ILP returns the true optimum of the actual problem.
        bfo = solve_bruteforce(inst, P_C=P_C, max_combos=1 << 15)
        if bfo.feasible and bfo.soft_objective is not None:
            bf_n += 1
            if abs(bfo.soft_objective - ref) < 1e-6:
                bf_agree += 1

    labels.append(f"N{N}/D{D}")
    sa_gaps.append(np.mean(g_sa) if g_sa else np.nan)
    greedy_gaps.append(np.mean(g_gr) if g_gr else np.nan)
    sa_feasr.append(np.mean(f_sa) if f_sa else np.nan)
    exact_agree.append((n_agree, agree))
    rows.append([
        f"N{N}/D{D}",
        f"{np.mean(f_sa):.2f}" if f_sa else "—",
        f"{np.mean(g_sa):+.3f}" if g_sa else "—",
        f"{np.mean(g_gr):+.3f}" if g_gr else "—",
        f"{n_agree}/{agree}" if agree else "n/a (too big)",
        f"{bf_agree}/{bf_n}" if bf_n else "n/a (too big)",
    ])
    print(f"N{N}/D{D}: SA feas={np.mean(f_sa):.2f} SA gap={np.mean(g_sa):+.3f} "
          f"greedy gap={np.mean(g_gr):+.3f} exact==ILP {n_agree}/{agree} "
          f"ILP==bruteforce {bf_agree}/{bf_n}")

# --- figure: gaps vs size --------------------------------------------------
x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
w = 0.38
ax1.bar(x - w/2, [0 if np.isnan(g) else g for g in sa_gaps], w, label="SA (best feasible)")
ax1.bar(x + w/2, [0 if np.isnan(g) else g for g in greedy_gaps], w, label="greedy")
ax1.set_xticks(x, labels); ax1.set_ylabel("optimality gap vs ILP optimum")
ax1.set_title("[SYNTHETIC] Optimality gap vs ILP (lower = better)")
ax1.axhline(0, color="grey", lw=0.8); ax1.legend(); ax1.grid(alpha=0.3, axis="y")

ax2.plot(x, sa_feasr, "o-", color="tab:blue")
ax2.set_xticks(x, labels); ax2.set_ylabel("SA feasibility rate"); ax2.set_ylim(-0.05, 1.05)
ax2.set_title(f"[SYNTHETIC] SA feasibility ({NUM_READS} reads)")
ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "05_baseline_comparison.png"), dpi=120)

write_table(os.path.join(RESULTS, "05_baseline_comparison.md"),
            ["size", "SA feas. rate", "SA gap vs ILP", "greedy gap vs ILP",
             "ExactSolver == ILP", "ILP == brute force"], rows)
print("\nNote: ExactSolver agreement confirms the penalty QUBO shares the ILP's "
      "optimum (faithful formulation). Blank ExactSolver = instance too large to "
      "enumerate 2^vars.")
print("Wrote results/05_baseline_comparison.png, 05_baseline_comparison.md")
