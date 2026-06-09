"""Baseline correctness: known-optimum instances where brute force, the ILP and
the penalty-encoded ExactSolver must all agree.

Test instances use credit_unit=1 and integer costs so there is NO cost rounding;
the QUBO cap and the ILP cap are then identical and the three optima must match
exactly (up to ties, which we compare on objective value, not bit pattern).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import Instance
from src.formulation import Penalties
from src.baselines import solve_bruteforce, solve_exact, solve_ilp, solve_greedy


def loose_instance():
    """No edges, loose cap. Optimum: job0 cheap, job1 escalate (value 30 >> cost 10)."""
    return Instance(
        N=2, A=2,
        cost=np.array([[10.0, 20.0], [10.0, 20.0]]),
        value=np.array([[0.0, 5.0], [0.0, 30.0]]),
        day=np.array([0, 0]),
        caps=np.array([100.0]),
        edges=[],
        credit_unit=1.0, name="loose",
    )


def tight_instance():
    """Same costs/values but cap=25 forbids any escalation (each esc plan spends 30)."""
    inst = loose_instance()
    inst.caps = np.array([25.0])
    inst.name = "tight"
    return inst


def coupled_instance():
    """Edge 0->1; escalating job1 while job0 is cheap incurs a wasted-escalation
    penalty large enough to suppress the otherwise-worthwhile escalation."""
    return Instance(
        N=2, A=2,
        cost=np.array([[10.0, 20.0], [10.0, 20.0]]),
        value=np.array([[0.0, 0.0], [0.0, 12.0]]),  # job1 esc net gain -2 before coupling
        day=np.array([0, 0]),
        caps=np.array([100.0]),
        edges=[(0, 1, 50.0)],  # heavy wasted-escalation penalty
        credit_unit=1.0, name="coupled",
    )


def test_loose_known_optimum():
    inst = loose_instance()
    bf = solve_bruteforce(inst, P_C=1.0)
    assert bf.feasible
    assert bf.assignment == [0, 1]          # job1 escalates, job0 cheap
    assert bf.true_objective == 0.0          # 10 + (20 - 30)


def test_tight_cap_forbids_escalation():
    inst = tight_instance()
    bf = solve_bruteforce(inst, P_C=1.0)
    assert bf.feasible
    assert bf.assignment == [0, 0]           # cap forces all cheap
    assert bf.true_objective == 20.0


def test_bruteforce_ilp_exact_agree_loose():
    inst = loose_instance()
    bf = solve_bruteforce(inst, P_C=1.0)
    ilp = solve_ilp(inst, P_C=1.0)
    ex = solve_exact(inst, Penalties(P_A=1e4, P_B=1e4, P_C=1.0))
    assert ilp.feasible and ex.feasible
    assert ilp.soft_objective == bf.soft_objective
    assert ex.soft_objective == bf.soft_objective


def test_bruteforce_ilp_exact_agree_tight():
    inst = tight_instance()
    bf = solve_bruteforce(inst, P_C=1.0)
    ilp = solve_ilp(inst, P_C=1.0)
    ex = solve_exact(inst, Penalties(P_A=1e4, P_B=1e4, P_C=1.0))
    assert ilp.feasible and ex.feasible
    assert ilp.soft_objective == bf.soft_objective == 20.0
    assert ex.soft_objective == 20.0


def test_coupling_suppresses_escalation():
    """With a heavy wasted-escalation penalty the optimum keeps job1 cheap, even
    though escalating job1 in isolation would look attractive on cost/value."""
    inst = coupled_instance()
    bf = solve_bruteforce(inst, P_C=1.0)
    ilp = solve_ilp(inst, P_C=1.0)
    assert bf.assignment == [0, 0]          # escalation suppressed by coupling
    assert ilp.soft_objective == bf.soft_objective
    # sanity: ignoring coupling (P_C=0), escalating job1 becomes worthwhile
    bf0 = solve_bruteforce(inst, P_C=0.0)
    assert bf0.assignment[1] == 1


def test_ilp_matches_bruteforce_on_random_small():
    """Cross-check on several seeded synthetic instances small enough to brute force."""
    from src.data import generate_instance
    for seed in range(6):
        inst = generate_instance(N=8, A=2, D=2, seed=seed, credit_unit=1.0)
        bf = solve_bruteforce(inst, P_C=1.0)
        ilp = solve_ilp(inst, P_C=1.0)
        assert bf.feasible == ilp.feasible
        if bf.feasible:
            assert ilp.soft_objective == bf.soft_objective, f"seed {seed}"


def test_greedy_is_feasible_and_not_better_than_optimum():
    from src.data import generate_instance
    for seed in range(4):
        inst = generate_instance(N=8, A=2, D=2, seed=seed, credit_unit=1.0)
        bf = solve_bruteforce(inst, P_C=1.0)
        gr = solve_greedy(inst, P_C=1.0)
        if gr.feasible and bf.feasible:
            # greedy can only do as well or worse than the true optimum
            assert gr.soft_objective >= bf.soft_objective - 1e-9
