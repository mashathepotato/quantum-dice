"""Solution-quality metrics, computed from the decoded assignment.

These are the numbers the experiments report. The "true objective" here is the
*linear* routing objective Σ_i (c_{i,a_i} − v_{i,a_i}) on the decoded assignment
— the thing we actually want to minimise — NOT the QUBO energy (which folds in
penalty terms). Optimality gaps compare like-for-like true objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .data import Instance
from .validate import Feasibility, validate


@dataclass
class SolutionMetrics:
    feasible: bool
    true_objective: float          # Σ (c - v) over chosen tiers (lower is better)
    total_spend: float             # Σ c over chosen tiers (tokens)
    total_value: float             # Σ v over chosen tiers (tokens-equivalent)
    num_escalated: int             # jobs on a tier >= 1
    num_jobs: int
    wasted_escalation_weight: float
    spend_per_day: Dict[int, float]
    assignment: List[int]


def assignment_true_objective(instance: Instance, assignment: List[int]) -> Optional[float]:
    """Linear routing objective of a fully-assigned plan, or None if any job
    is unassigned (-1)."""
    if any(a == -1 for a in assignment):
        return None
    total = 0.0
    for i, a in enumerate(assignment):
        total += float(instance.cost[i, a] - instance.value[i, a])
    return total


def compute_metrics(instance: Instance, sample: Dict) -> SolutionMetrics:
    feas = validate(instance, sample)
    assign = feas.assignment
    obj = assignment_true_objective(instance, assign)
    total_spend = 0.0
    total_value = 0.0
    num_esc = 0
    for i, a in enumerate(assign):
        if a == -1:
            continue
        total_spend += float(instance.cost[i, a])
        total_value += float(instance.value[i, a])
        if a >= 1:
            num_esc += 1
    return SolutionMetrics(
        feasible=feas.feasible,
        true_objective=obj if obj is not None else float("inf"),
        total_spend=total_spend,
        total_value=total_value,
        num_escalated=num_esc,
        num_jobs=instance.N,
        wasted_escalation_weight=feas.wasted_escalation_weight,
        spend_per_day=feas.spend_per_day,
        assignment=assign,
    )


def soft_objective(instance: Instance, assignment: List[int], P_C: float) -> Optional[float]:
    """Full minimised quantity: linear objective + P_C · wasted-escalation weight.

    This is what the QUBO minimises subject to the hard constraints, so SA and the
    ILP must be compared on *this* number, not on the linear part alone. Returns
    None if the assignment is not fully one-hot.
    """
    lin = assignment_true_objective(instance, assignment)
    if lin is None:
        return None
    wasted = 0.0
    for (i, j, w) in instance.edges:
        if assignment[j] >= 1 and assignment[i] == 0:
            wasted += w
    return lin + P_C * wasted


def optimality_gap(value: float, reference: float) -> float:
    """Relative gap (value − reference)/|reference|. 0 == matches reference.

    Objectives can be negative (value-dominated escalation), so we normalise by
    |reference| and guard the zero case.
    """
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - reference) / abs(reference)
