"""QUBO/BQM construction for agentic job-to-tier routing.

Builds a :class:`dimod.BinaryQuadraticModel` from an :class:`~src.data.Instance`
according to the energy in ``NOTES.md`` section 2::

    H = H_obj + P_A·H_onehot + P_B·H_cap + P_C·H_couple

    H_obj    = Σ_{i,a} (c_{i,a} − v_{i,a}) x_{i,a}                          (linear)
    H_onehot = Σ_i (1 − Σ_a x_{i,a})^2                                     (hard)
    H_cap    = Σ_d (Σ_{i∈d,a} c̃_{i,a} x_{i,a} + Σ_k 2^k s_{d,k} − B̃[d])^2 (hard, slack)
    H_couple = Σ_{i→j} w_{ij} · e_j · (1 − e_i),   e_i = Σ_{a≥1} x_{i,a}    (soft)

All squared terms are expanded to QUBO linear/quadratic/offset by the single
helper :func:`add_squared_penalty`, which is unit-tested against hand algebra.

Variable labels (hashable tuples, used everywhere):
    ("x", i, a) — job i assigned to tier a
    ("s", d, k) — slack bit k for the day-d cap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import dimod
import numpy as np

from .data import Instance


def xvar(i: int, a: int) -> Tuple[str, int, int]:
    return ("x", i, a)


def svar(d: int, k: int) -> Tuple[str, int, int]:
    return ("s", d, k)


@dataclass
class Penalties:
    """Penalty weights for the QUBO. P_A, P_B hard; P_C soft."""
    P_A: float = 1.0
    P_B: float = 1.0
    P_C: float = 1.0


def add_squared_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms: Dict[Tuple, float],
    constant: float,
    strength: float,
) -> None:
    """Add ``strength · (Σ_v terms[v]·v + constant)^2`` to *bqm* (binary vars).

    Expansion for binary variables (x^2 = x)::

        (Σ a_i x_i + C)^2 = Σ_i (a_i^2 + 2 C a_i) x_i
                            + Σ_{i<j} 2 a_i a_j  x_i x_j
                            + C^2

    This is the single audited place where squared penalties become QUBO
    coefficients; both the one-hot and the cap penalties go through it.
    """
    items = list(terms.items())
    # linear: a_i^2 + 2*C*a_i, all scaled by strength
    for v, a in items:
        bqm.add_linear(v, strength * (a * a + 2.0 * constant * a))
    # quadratic: 2 a_i a_j
    for idx in range(len(items)):
        vi, ai = items[idx]
        for jdx in range(idx + 1, len(items)):
            vj, aj = items[jdx]
            bqm.add_quadratic(vi, vj, strength * (2.0 * ai * aj))
    # offset: C^2
    bqm.offset += strength * (constant * constant)


def slack_bits_for(slack_range_units: int) -> int:
    """Number of slack bits to represent unused budget in [0, slack_range_units].

    K = ceil(log2(range + 1)) gives max representable slack 2^K - 1 >= range, so
    every feasible spend level has an exact slack complement and the equality
    Σ c̃x + slack = B̃ is satisfiable iff Σ c̃x ≤ B̃ (within the representable range).

    NOTE: ``slack_range_units`` is the span the slack must cover, which is
    ``cap_units − min_spend_units`` (see :func:`day_slack_range`), NOT the full
    cap. Because every job runs on at least the cheap tier, daily spend can never
    fall below the all-cheap spend, so the slack never needs to reach the cap.
    Sizing slack to this tighter range is both correct and far cheaper in bits —
    it shrinks the per-day cap clique that simulated annealing must navigate.
    """
    if slack_range_units <= 0:
        return 0
    return int(math.ceil(math.log2(slack_range_units + 1)))


def day_min_spend_units(instance: Instance) -> np.ndarray:
    """Per-day minimum achievable spend in credit units (cheapest tier per job)."""
    c_units = rounded_costs(instance)
    mins = np.zeros(instance.D, dtype=int)
    for d in range(instance.D):
        mins[d] = int(sum(c_units[i].min() for i in instance.jobs_on_day(d)))
    return mins


def day_slack_range(instance: Instance) -> np.ndarray:
    """Per-day slack span to encode: max(0, cap_units − min_spend_units)."""
    b_units = rounded_caps(instance)
    min_units = day_min_spend_units(instance)
    return np.maximum(0, b_units - min_units)


def default_slack_bits(instance: Instance) -> Dict[int, int]:
    """Correct slack-bit count per day, sized to the slack *range* (not the cap)."""
    rng = day_slack_range(instance)
    return {d: slack_bits_for(int(rng[d])) for d in range(instance.D)}


def rounded_costs(instance: Instance) -> np.ndarray:
    """Costs rounded to integer credit units (used by the cap encoding)."""
    return np.rint(instance.cost / instance.credit_unit).astype(int)


def rounded_caps(instance: Instance) -> np.ndarray:
    """Caps rounded to integer credit units."""
    return np.rint(instance.caps / instance.credit_unit).astype(int)


def rounded_deltas(instance: Instance) -> np.ndarray:
    """Per-tier *extra* cost over the cheap tier, in credit units.

    ``delta[i, a] = c̃[i, a] − c̃[i, 0] >= 0`` with ``delta[:, 0] = 0``. Used by the
    baseline-shifted cap encoding: cheap-tier choices contribute 0 and therefore
    drop out of the cap clique entirely (only escalation choices remain coupled),
    which is far better conditioned for simulated annealing.
    """
    c = rounded_costs(instance)
    return c - c[:, [0]]


def day_cheap_units(instance: Instance) -> np.ndarray:
    """Per-day all-cheap spend in credit units (the unavoidable baseline)."""
    c = rounded_costs(instance)
    out = np.zeros(instance.D, dtype=int)
    for d in range(instance.D):
        out[d] = int(sum(c[i, 0] for i in instance.jobs_on_day(d)))
    return out


def day_headroom_units(instance: Instance) -> np.ndarray:
    """Per-day escalation headroom in credit units: cap − all-cheap spend."""
    return rounded_caps(instance) - day_cheap_units(instance)


def build_bqm(
    instance: Instance,
    penalties: Penalties,
    slack_bits: Optional[Dict[int, int]] = None,
    include_cap: bool = True,
    cap_mode: str = "shifted",
) -> dimod.BinaryQuadraticModel:
    """Construct the routing BQM.

    Parameters
    ----------
    instance : the problem.
    penalties : P_A (one-hot, hard), P_B (cap, hard), P_C (coupling, soft).
    slack_bits : optional ``{day: K_d}`` override. Defaults to the correct
        range-sized count per day. Deliberately under-provisioning K_d is the
        failure mode probed in experiment 02.
    include_cap : if False, omit the daily-cap penalty entirely (used to isolate
        terms in tests/experiments).
    cap_mode : ``"shifted"`` (default) encodes the cap as
        ``Σ Δ̃·x + slack = headroom`` (escalation extra over the all-cheap
        baseline); cheap-tier vars drop out of the cap clique. ``"absolute"``
        encodes the literal ``Σ c̃·x + slack = B̃``. The two are identical at
        one-hot-feasible points but ``"shifted"`` is much easier for SA — see
        experiment 01. Use ``"absolute"`` only to reproduce the v1 failure mode.

    Returns
    -------
    dimod.BinaryQuadraticModel (BINARY vartype). The objective is in raw token
    units; the cap penalty operates in rounded credit units.
    """
    if cap_mode not in ("shifted", "absolute"):
        raise ValueError(f"cap_mode must be 'shifted' or 'absolute', got {cap_mode!r}")
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    N, A = instance.N, instance.A

    # Ensure all x variables exist even if a coefficient is 0.
    for i in range(N):
        for a in range(A):
            bqm.add_variable(xvar(i, a), 0.0)

    # --- H_obj: linear objective (raw token units) --------------------------
    for i in range(N):
        for a in range(A):
            bqm.add_linear(xvar(i, a), float(instance.cost[i, a] - instance.value[i, a]))

    # --- H_onehot: Σ_i (1 − Σ_a x_{i,a})^2 ----------------------------------
    for i in range(N):
        terms = {xvar(i, a): -1.0 for a in range(A)}
        add_squared_penalty(bqm, terms, constant=1.0, strength=penalties.P_A)

    # --- H_cap: per-day squared-slack cap (rounded credit units) ------------
    if include_cap:
        c_units = rounded_costs(instance)
        d_units = rounded_deltas(instance)
        b_units = rounded_caps(instance)
        head_units = day_headroom_units(instance)
        default_bits = default_slack_bits(instance)
        for d in range(instance.D):
            jobs = instance.jobs_on_day(d)
            if not jobs:
                continue
            Kd = default_bits[d] if slack_bits is None else slack_bits.get(d, default_bits[d])
            terms: Dict[Tuple, float] = {}
            if cap_mode == "shifted":
                # Σ Δ̃·x + slack = headroom ; cheap-tier (Δ=0) vars drop out.
                for i in jobs:
                    for a in range(A):
                        if d_units[i, a] != 0:
                            terms[xvar(i, a)] = float(d_units[i, a])
                target = float(head_units[d])
            else:  # "absolute": literal Σ c̃·x + slack = B̃
                for i in jobs:
                    for a in range(A):
                        terms[xvar(i, a)] = float(c_units[i, a])
                target = float(b_units[d])
            for k in range(Kd):
                bqm.add_variable(svar(d, k), 0.0)
                terms[svar(d, k)] = float(2 ** k)
            # penalty = P_B * (Σ terms − target)^2
            add_squared_penalty(bqm, terms, constant=-target, strength=penalties.P_B)

    # --- H_couple: Σ_{i→j} w_{ij} e_j (1 − e_i) -----------------------------
    # e_i = Σ_{a≥1} x_{i,a};  e_j(1−e_i) = e_j − e_j·e_i
    for (i, j, w) in instance.edges:
        if w == 0:
            continue
        esc_j = [xvar(j, a) for a in range(1, A)]
        esc_i = [xvar(i, a) for a in range(1, A)]
        for vj in esc_j:
            bqm.add_linear(vj, penalties.P_C * w)          # +w · e_j
        for vj in esc_j:
            for vi in esc_i:
                bqm.add_quadratic(vj, vi, -penalties.P_C * w)  # −w · e_j·e_i

    return bqm


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def decode_assignment(instance: Instance, sample: Dict) -> List[int]:
    """Decode a sample to a per-job tier assignment.

    Returns a length-N list where entry i is the tier a with x_{i,a}=1, or -1 if
    job i is not assigned to exactly one tier (a one-hot violation). The
    validator interprets -1 as infeasible; this function does not raise.
    """
    assign = [-1] * instance.N
    for i in range(instance.N):
        ones = [a for a in range(instance.A) if sample.get(xvar(i, a), 0) == 1]
        assign[i] = ones[0] if len(ones) == 1 else -1
    return assign


def slack_values(instance: Instance, sample: Dict, slack_bits: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    """Decode the slack integer Σ_k 2^k s_{d,k} for each day present in *sample*."""
    default_bits = default_slack_bits(instance)
    out: Dict[int, int] = {}
    for d in range(instance.D):
        Kd = default_bits[d] if slack_bits is None else slack_bits.get(d, default_bits[d])
        total = 0
        for k in range(Kd):
            total += (2 ** k) * int(sample.get(svar(d, k), 0))
        out[d] = total
    return out


def objective_scale(instance: Instance) -> float:
    """A representative magnitude of the linear objective, for penalty calibration.

    Returns the sum of absolute per-assignment objective coefficients — a loose
    upper bound on the total objective swing.
    """
    return float(np.abs(instance.cost - instance.value).sum())


def objective_coeff_max(instance: Instance) -> float:
    """Largest single objective coefficient magnitude, max_{i,a} |c_{i,a} − v_{i,a}|.

    This is the Lagrangian-style bound that matters for penalty calibration: a hard
    penalty must exceed the objective gain a *single* constraint-cheating flip can
    buy, which is bounded by this — NOT by the sum over all jobs. Scaling penalties
    to this (rather than the much larger sum) keeps the QUBO coefficients in a
    narrow dynamic range, which simulated annealing handles far better. Penalty
    weights are reported as multiples of this in experiment 01.
    """
    return float(np.abs(instance.cost - instance.value).max())
