"""Loader correctness: the example JSON export parses into a consistent Instance
that the formulation, validator and baselines all accept, and is labelled REAL
(is_synthetic=False) so outputs never imply synthetic data is real."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_instance
from src.formulation import Penalties, build_bqm, objective_coeff_max
from src.baselines import solve_ilp

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "example_run_export.json")


def test_example_export_loads_consistently():
    inst = load_instance(EXAMPLE)
    assert inst.N == 5 and inst.A == 2 and inst.D == 2
    assert not inst.is_synthetic           # tagged REAL
    assert inst.data_kind == "REAL"
    # costs/values shaped (N, A); escalation strictly dearer than cheap
    assert inst.cost.shape == (5, 2) and inst.value.shape == (5, 2)
    assert (inst.cost[:, 1] > inst.cost[:, 0]).all()
    assert (inst.value[:, 0] == 0).all()    # cheap tier is the value baseline
    # edges remapped to indices and within range
    assert len(inst.edges) == 4
    for (i, j, w) in inst.edges:
        assert 0 <= i < inst.N and 0 <= j < inst.N and w > 0


def test_loaded_instance_is_solvable():
    inst = load_instance(EXAMPLE)
    # the formulation builds and the ILP returns a feasible optimum
    bqm = build_bqm(inst, Penalties(P_A=10 * objective_coeff_max(inst),
                                    P_B=10 * objective_coeff_max(inst), P_C=0.2))
    assert bqm.num_variables >= inst.num_vars
    ilp = solve_ilp(inst, P_C=0.2)
    assert ilp.feasible
    assert ilp.assignment is not None and len(ilp.assignment) == inst.N
