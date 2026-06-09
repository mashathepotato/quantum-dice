"""Sampler-interface contract: SA returns a usable SampleSet over the BQM's
variables; the ORBIT stub raises a clear NotImplementedError (the Stage-3 swap is
a one-line change at the call site, not silent breakage)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dimod

from src.data import generate_instance
from src.formulation import Penalties, build_bqm, objective_coeff_max
from src.solvers import Sampler, SASampler, OrbitAdapter


def test_sa_sampler_returns_sampleset_over_bqm_vars():
    inst = generate_instance(N=6, A=2, D=2, seed=0)
    bqm = build_bqm(inst, Penalties(P_A=10 * objective_coeff_max(inst),
                                    P_B=10 * objective_coeff_max(inst), P_C=0.2))
    ss = SASampler(seed=0).sample(bqm, num_reads=20)
    assert isinstance(ss, dimod.SampleSet)
    assert len(ss) > 0
    assert set(ss.variables) == set(bqm.variables)
    # energies are real and the reported energy matches the BQM
    best = ss.first
    assert best.energy == pytest.approx(bqm.energy(best.sample))


def test_sa_sampler_is_seeded_reproducible():
    inst = generate_instance(N=6, A=2, D=2, seed=0)
    bqm = build_bqm(inst, Penalties(P_A=1000, P_B=1000, P_C=0.2))
    e1 = SASampler(seed=42).sample(bqm, num_reads=20).first.energy
    e2 = SASampler(seed=42).sample(bqm, num_reads=20).first.energy
    assert e1 == e2


def test_orbit_adapter_is_a_sampler_but_not_implemented():
    assert issubclass(OrbitAdapter, Sampler)
    inst = generate_instance(N=4, A=2, D=1, seed=0)
    bqm = build_bqm(inst, Penalties(P_A=1000, P_B=1000, P_C=0.2))
    with pytest.raises(NotImplementedError, match="Stage-3"):
        OrbitAdapter().sample(bqm, num_reads=10)
