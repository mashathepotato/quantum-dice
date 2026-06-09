"""Classical baselines: a true optimum / feasibility reference for the QUBO.

Three references, in increasing scale:

* :func:`solve_bruteforce` — enumerate every one-hot assignment (A^N), check the
  hard daily cap directly in raw tokens, and minimise the *soft objective*
  (linear cost-minus-value + P_C·wasted-escalation). Penalty-free and slack-free,
  so it is the unambiguous ground-truth optimum for small N.
* :func:`solve_exact` — ground state of the actual BQM via ``dimod.ExactSolver``
  (enumerates 2^num_vars, slack bits included). Confirms the penalty-encoded QUBO
  has the *same* optimum as the constrained problem when penalties are calibrated.
* :func:`solve_ilp` — the same constrained problem stated directly as a 0/1 ILP
  and solved to optimality by CBC (via PuLP). The coupling product is linearised
  exactly. This is the scalable optimum/feasibility reference for the experiments.

Plus :func:`solve_greedy`, a cheap heuristic for context.

"Soft objective" everywhere == linear objective + P_C · Σ_wasted w, matching what
the QUBO minimises subject to the hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional

import dimod
import numpy as np
import pulp

from .data import Instance
from .formulation import Penalties, build_bqm, xvar
from .metrics import assignment_true_objective, soft_objective
from .validate import validate


@dataclass
class BaselineResult:
    method: str
    assignment: Optional[List[int]]    # tier per job, or None if infeasible/failed
    soft_objective: Optional[float]    # linear + P_C * wasted weight
    true_objective: Optional[float]    # linear only
    feasible: bool
    status: str = "ok"


def _soft_objective(instance: Instance, assignment: List[int], P_C: float) -> float:
    """linear objective + P_C * wasted-escalation weight (delegates to metrics)."""
    val = soft_objective(instance, assignment, P_C)
    assert val is not None
    return val


def _assignment_feasible(instance: Instance, assignment: List[int]) -> bool:
    """Daily cap feasibility (one-hot is guaranteed by construction here)."""
    spend = {d: 0.0 for d in range(instance.D)}
    for i, a in enumerate(assignment):
        spend[int(instance.day[i])] += float(instance.cost[i, a])
    return all(spend[d] <= instance.caps[d] + 1e-9 for d in range(instance.D))


def solve_bruteforce(instance: Instance, P_C: float = 1.0, max_combos: int = 1 << 20) -> BaselineResult:
    """True optimum by enumerating all A^N one-hot assignments. Small N only."""
    combos = instance.A ** instance.N
    if combos > max_combos:
        return BaselineResult("bruteforce", None, None, None, False,
                              status=f"skipped: A^N={combos} > {max_combos}")
    best = None
    best_obj = float("inf")
    for assignment in product(range(instance.A), repeat=instance.N):
        a = list(assignment)
        if not _assignment_feasible(instance, a):
            continue
        obj = _soft_objective(instance, a, P_C)
        if obj < best_obj:
            best_obj, best = obj, a
    if best is None:
        return BaselineResult("bruteforce", None, None, None, False, status="no feasible assignment")
    return BaselineResult("bruteforce", best, best_obj,
                          assignment_true_objective(instance, best), True)


def solve_exact(instance: Instance, penalties: Penalties, max_vars: int = 18) -> BaselineResult:
    """Ground state of the penalty-encoded BQM (dimod.ExactSolver). Tiny only."""
    bqm = build_bqm(instance, penalties)
    if bqm.num_variables > max_vars:
        return BaselineResult("exact_bqm", None, None, None, False,
                              status=f"skipped: {bqm.num_variables} vars > {max_vars}")
    ss = dimod.ExactSolver().sample(bqm)
    best = ss.first.sample
    feas = validate(instance, best)
    assign = feas.assignment if feas.onehot_ok else None
    if assign is None or not feas.feasible:
        return BaselineResult("exact_bqm", assign, None,
                              assignment_true_objective(instance, assign) if assign else None,
                              feas.feasible, status="ground state infeasible (penalties too low)")
    return BaselineResult("exact_bqm", assign,
                          _soft_objective(instance, assign, penalties.P_C),
                          assignment_true_objective(instance, assign), True)


def solve_ilp(instance: Instance, P_C: float = 1.0, msg: bool = False) -> BaselineResult:
    """Solve the exact constrained problem as a 0/1 ILP with CBC (PuLP).

    Variables x[i,a] in {0,1}; one-hot per job; per-day cap in raw tokens;
    coupling linearised with z_e = e_j·(1 − e_i) for each edge. Objective =
    Σ (c − v) x + P_C Σ_e w_e z_e. CBC returns the proven optimum.
    """
    prob = pulp.LpProblem("routing", pulp.LpMinimize)
    x = {(i, a): pulp.LpVariable(f"x_{i}_{a}", cat="Binary")
         for i in range(instance.N) for a in range(instance.A)}

    def esc(i):
        # escalation indicator e_i = Σ_{a>=1} x[i,a]
        return pulp.lpSum(x[(i, a)] for a in range(1, instance.A))

    # objective: linear part
    obj = pulp.lpSum(
        float(instance.cost[i, a] - instance.value[i, a]) * x[(i, a)]
        for i in range(instance.N) for a in range(instance.A)
    )

    # coupling: z_e >= e_j - e_i, z_e >= 0 ; minimisation forces z_e = e_j(1-e_i)
    z = {}
    for idx, (i, j, w) in enumerate(instance.edges):
        if w == 0:
            continue
        ze = pulp.LpVariable(f"z_{idx}", lowBound=0, upBound=1, cat="Continuous")
        prob += ze >= esc(j) - esc(i)
        prob += ze <= esc(j)
        prob += ze <= 1 - esc(i)
        z[idx] = (ze, w)
    obj = obj + pulp.lpSum(P_C * w * ze for (ze, w) in z.values())
    prob += obj

    # one-hot
    for i in range(instance.N):
        prob += pulp.lpSum(x[(i, a)] for a in range(instance.A)) == 1

    # per-day cap (raw tokens)
    for d in range(instance.D):
        jobs = instance.jobs_on_day(d)
        if not jobs:
            continue
        prob += pulp.lpSum(
            float(instance.cost[i, a]) * x[(i, a)] for i in jobs for a in range(instance.A)
        ) <= float(instance.caps[d])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status_str = pulp.LpStatus[status]
    if status_str != "Optimal":
        return BaselineResult("ilp", None, None, None, False, status=status_str)

    assign = [-1] * instance.N
    for i in range(instance.N):
        for a in range(instance.A):
            if x[(i, a)].value() is not None and round(x[(i, a)].value()) == 1:
                assign[i] = a
    return BaselineResult("ilp", assign,
                          _soft_objective(instance, assign, P_C),
                          assignment_true_objective(instance, assign), True, status="Optimal")


def solve_greedy(instance: Instance, P_C: float = 1.0) -> BaselineResult:
    """Greedy heuristic: start all-cheap, escalate jobs that most improve the
    soft objective while keeping every daily cap feasible. Context baseline."""
    assign = [0] * instance.N
    if not _assignment_feasible(instance, assign):
        # even all-cheap violates a cap: no heuristic recovery here
        return BaselineResult("greedy", None, None, None, False, status="all-cheap infeasible")

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_move = None
        cur = _soft_objective(instance, assign, P_C)
        for i in range(instance.N):
            for a in range(instance.A):
                if a == assign[i]:
                    continue
                trial = list(assign)
                trial[i] = a
                if not _assignment_feasible(instance, trial):
                    continue
                gain = cur - _soft_objective(instance, trial, P_C)
                if gain > best_gain + 1e-9:
                    best_gain, best_move = gain, (i, a)
        if best_move is not None:
            i, a = best_move
            assign[i] = a
            improved = True

    return BaselineResult("greedy", assign,
                          _soft_objective(instance, assign, P_C),
                          assignment_true_objective(instance, assign), True)
