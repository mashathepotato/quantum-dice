"""Decode a sample and check the hard constraints of the routing problem.

Validation is done on the *decoded* assignment in raw token units — independent
of the QUBO penalty machinery — so it is a genuine, trustworthy check that a
sampled bitstring is a feasible routing plan, not a restatement of the energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .data import Instance
from .formulation import decode_assignment


@dataclass
class Feasibility:
    """Result of validating one sample against the problem's hard constraints."""
    assignment: List[int]                 # tier per job, -1 = one-hot violation
    onehot_ok: bool
    onehot_violations: List[int]          # job indices not assigned exactly once
    cap_ok: bool
    spend_per_day: Dict[int, float]       # raw token spend per day
    cap_violations: Dict[int, float]      # day -> overspend (tokens) where > 0
    wasted_escalation_edges: List[tuple]  # (i, j, w) edges with wasted escalation
    wasted_escalation_weight: float       # Σ w over those edges (soft, informational)

    @property
    def feasible(self) -> bool:
        """Feasible == all HARD constraints satisfied (one-hot + daily cap).

        The escalation coupling is SOFT and never gates feasibility.
        """
        return self.onehot_ok and self.cap_ok


def validate(instance: Instance, sample: Dict) -> Feasibility:
    """Validate a sample dict (var -> 0/1) against the hard constraints."""
    assign = decode_assignment(instance, sample)

    onehot_violations = [i for i, a in enumerate(assign) if a == -1]
    onehot_ok = len(onehot_violations) == 0

    # spend per day uses raw token costs of the chosen tier (cheap=0 if violation)
    spend_per_day: Dict[int, float] = {d: 0.0 for d in range(instance.D)}
    for i, a in enumerate(assign):
        if a == -1:
            continue
        spend_per_day[int(instance.day[i])] += float(instance.cost[i, a])

    cap_violations: Dict[int, float] = {}
    for d in range(instance.D):
        over = spend_per_day[d] - float(instance.caps[d])
        if over > 1e-9:
            cap_violations[d] = over
    cap_ok = len(cap_violations) == 0

    # wasted escalation: edge i->j where j escalated (a>=1) and i not (a==0).
    wasted = []
    wasted_w = 0.0
    for (i, j, w) in instance.edges:
        ai, aj = assign[i], assign[j]
        if ai == -1 or aj == -1:
            continue
        if aj >= 1 and ai == 0:
            wasted.append((i, j, w))
            wasted_w += w

    return Feasibility(
        assignment=assign,
        onehot_ok=onehot_ok,
        onehot_violations=onehot_violations,
        cap_ok=cap_ok,
        spend_per_day=spend_per_day,
        cap_violations=cap_violations,
        wasted_escalation_edges=wasted,
        wasted_escalation_weight=wasted_w,
    )
