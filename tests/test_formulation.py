"""BQM-construction correctness: hand-computed energy on a tiny instance.

The whole formulation is only trustworthy if the BQM energy of a known
assignment matches algebra done by hand. This file pins that down.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dimod

from src.data import Instance
from src.formulation import (
    Penalties,
    add_squared_penalty,
    build_bqm,
    decode_assignment,
    slack_bits_for,
    slack_values,
    xvar,
    svar,
)


def tiny_instance():
    """N=2, A=2, D=1, credit_unit=10. All numbers chosen for hand algebra.

    cost  = [[10, 30], [20, 50]]   value = [[0, 15], [0, 40]]
    => objective coef (c - v) = [[10, 15], [20, 10]]
    credit units: c̃ = [[1, 3], [2, 5]],  cap = 100 -> B̃ = 10
    one DAG edge 0 -> 1 with weight 5.
    """
    return Instance(
        N=2, A=2,
        cost=np.array([[10.0, 30.0], [20.0, 50.0]]),
        value=np.array([[0.0, 15.0], [0.0, 40.0]]),
        day=np.array([0, 0]),
        caps=np.array([100.0]),
        edges=[(0, 1, 5.0)],
        roles=["Researcher", "Executor"],
        credit_unit=10.0,
        is_synthetic=True,
        name="tiny",
    )


def test_slack_bits_count():
    # cap units = 10 -> ceil(log2(11)) = 4 bits, max representable slack 15 >= 10
    assert slack_bits_for(10) == 4
    assert (2 ** slack_bits_for(10)) - 1 >= 10
    assert slack_bits_for(0) == 0
    assert slack_bits_for(1) == 1  # ceil(log2(2)) = 1


def test_add_squared_penalty_matches_algebra():
    # penalty = 2 * (2a - b + 3)^2 ; check at several points by brute force.
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    bqm.add_variable("a", 0.0)
    bqm.add_variable("b", 0.0)
    add_squared_penalty(bqm, {"a": 2.0, "b": -1.0}, constant=3.0, strength=2.0)
    for a in (0, 1):
        for b in (0, 1):
            expected = 2.0 * (2 * a - b + 3) ** 2
            assert bqm.energy({"a": a, "b": b}) == pytest.approx(expected)


def test_hand_computed_energy():
    """Assignment: job0->cheap (x00=1), job1->escalate (x11=1).

    Spend (units) = c̃00 + c̃11 = 1 + 5 = 6; pick slack = 4 (bit s_{0,2}) so the
    cap term is exactly satisfied: 6 + 4 - 10 = 0.

    H_obj    = 10 (x00) + 10 (x11)                       = 20
    H_onehot = 0 (each job assigned exactly once)        -> P_A * 0
    H_cap    = (6 + 4 - 10)^2 = 0                         -> P_B * 0
    H_couple = w * e_j(1 - e_i) = 5 * 1 * (1 - 0) = 5     -> P_C * 5
    Total (P_A=10, P_B=10, P_C=1) = 20 + 0 + 0 + 5 = 25
    """
    inst = tiny_instance()
    bqm = build_bqm(inst, Penalties(P_A=10, P_B=10, P_C=1))
    sample = {
        xvar(0, 0): 1, xvar(0, 1): 0,
        xvar(1, 0): 0, xvar(1, 1): 1,
        svar(0, 0): 0, svar(0, 1): 0, svar(0, 2): 1, svar(0, 3): 0,
    }
    assert bqm.energy(sample) == pytest.approx(25.0)


def test_onehot_violation_costs_P_A():
    """Assigning job0 to BOTH tiers adds P_A*(1-2)^2 = P_A to the energy,
    relative to a clean one-hot, plus the extra objective of the second tier."""
    inst = tiny_instance()
    P_A = 1000.0
    bqm = build_bqm(inst, Penalties(P_A=P_A, P_B=10, P_C=1))
    # job0 on both tiers, job1 cheap; pick slack to satisfy cap loosely.
    # spend units = c̃00 + c̃01 + c̃10 = 1 + 3 + 2 = 6 -> slack 4 again.
    base_sample = {
        xvar(0, 0): 1, xvar(0, 1): 1,
        xvar(1, 0): 1, xvar(1, 1): 0,
        svar(0, 0): 0, svar(0, 1): 0, svar(0, 2): 1, svar(0, 3): 0,
    }
    e = bqm.energy(base_sample)
    # one-hot penalty for job0: (1 - 2)^2 = 1 -> contributes P_A; job1 fine (0).
    # H_obj = 10 + 15 (job0 both) + 20 (job1 cheap) = 45 ; H_cap = 0 ; H_couple:
    #   e_j=e_1=0 (job1 cheap) -> 0.
    expected = 45.0 + P_A * 1.0
    assert e == pytest.approx(expected)


def test_coupling_zero_when_upstream_also_escalated():
    """Wasted-escalation penalty vanishes when the upstream job is escalated too."""
    inst = tiny_instance()
    bqm = build_bqm(inst, Penalties(P_A=10, P_B=10, P_C=1))
    # both escalate: spend units = c̃01 + c̃11 = 3 + 5 = 8 -> slack 2 (bit s_{0,1}).
    sample = {
        xvar(0, 0): 0, xvar(0, 1): 1,
        xvar(1, 0): 0, xvar(1, 1): 1,
        svar(0, 0): 0, svar(0, 1): 1, svar(0, 2): 0, svar(0, 3): 0,
    }
    # H_obj = 15 (x01) + 10 (x11) = 25 ; one-hot 0 ; cap (8+2-10)^2=0 ;
    # couple: e_j=1, e_i=1 -> 1*(1-1)=0.
    assert bqm.energy(sample) == pytest.approx(25.0)


def test_decode_assignment_and_slack():
    inst = tiny_instance()
    sample = {
        xvar(0, 0): 1, xvar(0, 1): 0,
        xvar(1, 0): 0, xvar(1, 1): 1,
        svar(0, 0): 0, svar(0, 1): 0, svar(0, 2): 1, svar(0, 3): 0,
    }
    assert decode_assignment(inst, sample) == [0, 1]
    assert slack_values(inst, sample) == {0: 4}


def test_decode_detects_onehot_violation():
    inst = tiny_instance()
    # job1 assigned to neither tier -> -1
    sample = {xvar(0, 0): 1, xvar(0, 1): 0, xvar(1, 0): 0, xvar(1, 1): 0}
    assert decode_assignment(inst, sample) == [0, -1]
    # job0 assigned to both -> -1
    sample2 = {xvar(0, 0): 1, xvar(0, 1): 1, xvar(1, 0): 1, xvar(1, 1): 0}
    assert decode_assignment(inst, sample2) == [-1, 0]
