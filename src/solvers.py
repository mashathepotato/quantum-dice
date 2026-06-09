"""Sampler interface and implementations.

The whole point of this thin layer is that swapping the Stage-2 simulated-annealing
stand-in for the Stage-3 ORBIT p-bit hardware is a **one-line change** at the call
site::

    sampler = SASampler()          # Stage 2 (now)
    sampler = OrbitAdapter(...)    # Stage 3 (from 12 June, when ORBIT is available)

Both satisfy the same ``Sampler.sample(bqm, num_reads) -> dimod.SampleSet`` contract,
so every experiment, baseline comparison and validator works unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import dimod
from dwave.samplers import SimulatedAnnealingSampler


class Sampler(ABC):
    """Minimal sampler contract used throughout Stage 2.

    Implementations return a :class:`dimod.SampleSet` of ``num_reads`` low-energy
    samples for the given BQM. The SampleSet carries energies and occurrence
    counts, which the diversity experiment relies on.
    """

    name: str = "sampler"

    @abstractmethod
    def sample(self, bqm: dimod.BinaryQuadraticModel, num_reads: int = 100, **kwargs) -> dimod.SampleSet:
        ...


class SASampler(Sampler):
    """Simulated annealing via ``dwave-samplers``, the p-bit stand-in for Stage 2.

    SA is a classical heuristic that, like a p-bit network, *samples* a
    distribution of low-energy states rather than proving an optimum. It is the
    honest Stage-2 proxy for ORBIT: same QUBO in, a distribution of bitstrings out.
    """

    name = "SA"

    def __init__(self, seed: Optional[int] = None, num_sweeps: int = 1000):
        self._sampler = SimulatedAnnealingSampler()
        self.seed = seed
        self.num_sweeps = num_sweeps

    def sample(self, bqm: dimod.BinaryQuadraticModel, num_reads: int = 100, **kwargs) -> dimod.SampleSet:
        params = dict(num_reads=num_reads, num_sweeps=self.num_sweeps)
        if self.seed is not None:
            params["seed"] = self.seed
        params.update(kwargs)
        return self._sampler.sample(bqm, **params)


class OrbitAdapter(Sampler):
    """Stage-3 ORBIT p-bit sampler — NOT YET AVAILABLE (hardware lands 12 June).

    Deliberately a stub: it raises with a clear TODO so nothing silently depends
    on hardware that is not here yet. When ORBIT's SDK is available, implement
    ``sample`` to (1) embed/upload the BQM, (2) run the p-bit network for
    ``num_reads`` shots, and (3) return the shots as a ``dimod.SampleSet`` with
    energies recomputed via ``bqm.energy``. The call site changes by one line.
    """

    name = "ORBIT"

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def sample(self, bqm: dimod.BinaryQuadraticModel, num_reads: int = 100, **kwargs) -> dimod.SampleSet:
        raise NotImplementedError(
            "OrbitAdapter is a Stage-3 stub. ORBIT hardware is unavailable until "
            "12 June 2026. TODO(stage3): upload `bqm`, run the p-bit network for "
            "`num_reads` shots, wrap the returned bitstrings in a dimod.SampleSet "
            "(dimod.SampleSet.from_samples_bqm(shots, bqm)). Until then use SASampler()."
        )
