"""Validation / decoding correctness: detect one-hot and cap violations,
pass genuinely feasible plans, and count wasted escalation correctly."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import Instance
from src.formulation import xvar
from src.metrics import compute_metrics
from src.validate import validate


def inst():
    # N=3, A=2, D=2. job0,1 on day0; job2 on day1. edges 0->2, 1->2.
    return Instance(
        N=3, A=2,
        cost=np.array([[10.0, 30.0], [20.0, 60.0], [15.0, 40.0]]),
        value=np.array([[0.0, 20.0], [0.0, 25.0], [0.0, 30.0]]),
        day=np.array([0, 0, 1]),
        caps=np.array([45.0, 100.0]),  # day0 cap 45 binds: 30+20 ok, 30+60 not
        edges=[(0, 2, 3.0), (1, 2, 4.0)],
        roles=["Researcher", "Researcher", "Analyst"],
        credit_unit=5.0,
        name="vtest",
    )


def test_feasible_plan_passes():
    # all cheap: day0 spend = 10+20 = 30 <= 45 ; day1 = 15 <= 100.
    s = {xvar(0, 0): 1, xvar(0, 1): 0,
         xvar(1, 0): 1, xvar(1, 1): 0,
         xvar(2, 0): 1, xvar(2, 1): 0}
    f = validate(inst(), s)
    assert f.feasible
    assert f.onehot_ok and f.cap_ok
    assert f.spend_per_day[0] == 30.0
    assert f.cap_violations == {}


def test_cap_violation_detected():
    # job0 escalates on day0: 30 (esc) + 20 (cheap) = 50 > 45.
    s = {xvar(0, 0): 0, xvar(0, 1): 1,
         xvar(1, 0): 1, xvar(1, 1): 0,
         xvar(2, 0): 1, xvar(2, 1): 0}
    f = validate(inst(), s)
    assert not f.feasible
    assert not f.cap_ok
    assert f.cap_violations[0] == 5.0  # 50 - 45
    assert f.onehot_ok


def test_onehot_violation_detected():
    s = {xvar(0, 0): 1, xvar(0, 1): 1,  # both
         xvar(1, 0): 0, xvar(1, 1): 0,  # neither
         xvar(2, 0): 1, xvar(2, 1): 0}
    f = validate(inst(), s)
    assert not f.feasible
    assert not f.onehot_ok
    assert set(f.onehot_violations) == {0, 1}


def test_wasted_escalation_counted():
    # job2 escalates (day1), upstreams job0/job1 cheap -> both edges wasted.
    s = {xvar(0, 0): 1, xvar(0, 1): 0,
         xvar(1, 0): 1, xvar(1, 1): 0,
         xvar(2, 0): 0, xvar(2, 1): 1}
    f = validate(inst(), s)
    assert f.feasible  # day1 spend 40 <= 100, day0 30 <= 45
    assert len(f.wasted_escalation_edges) == 2
    assert f.wasted_escalation_weight == 7.0  # 3 + 4

    # if an upstream is also escalated, that edge is no longer wasted.
    s2 = dict(s)
    s2[xvar(0, 0)], s2[xvar(0, 1)] = 0, 1  # job0 escalates; day0 = 30+20=50 > 45
    f2 = validate(inst(), s2)
    assert not f2.cap_ok  # now infeasible, but coupling count still correct:
    assert len(f2.wasted_escalation_edges) == 1  # only edge 1->2 wasted now
    assert f2.wasted_escalation_weight == 4.0


def test_metrics_objective_matches_assignment():
    # all cheap: obj = Σ (c - v) cheap = 10 + 20 + 15 = 45 (value cheap = 0)
    s = {xvar(0, 0): 1, xvar(0, 1): 0,
         xvar(1, 0): 1, xvar(1, 1): 0,
         xvar(2, 0): 1, xvar(2, 1): 0}
    m = compute_metrics(inst(), s)
    assert m.feasible
    assert m.true_objective == 45.0
    assert m.total_spend == 45.0
    assert m.total_value == 0.0
    assert m.num_escalated == 0
