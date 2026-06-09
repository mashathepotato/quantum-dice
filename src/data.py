"""Problem instances for the agentic job-to-tier routing QUBO.

An :class:`Instance` is the single source of truth consumed by the formulation,
the baselines and the validator. We provide:

* a seeded, reproducible synthetic generator that mimics a realistic agentic
  research-lab workload (Researcher / Executor / Analyst roles, a DAG of job
  dependencies laid out over days, per-tier token costs and escalation values),
  and
* a loader that ingests a *real* run-database export (CSV or JSON) if one is
  dropped into ``data/``.

Costs and values are expressed in **tokens**. The formulation rounds them to a
coarse ``credit_unit`` for the slack-bit cap encoding (see ``formulation.py``);
the rounding lives there, not here, so this module stays a faithful description
of the workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import csv
import json

import numpy as np

# Agent roles in the lab. The dependency chain runs Researcher -> Executor ->
# Analyst (proposals -> runs -> digests), matching the proposal's role order.
ROLES = ("Researcher", "Executor", "Analyst")
ROLE_ORDER = {r: i for i, r in enumerate(ROLES)}


@dataclass
class Instance:
    """A single dispatch-cycle routing instance.

    Attributes
    ----------
    N : number of jobs.
    A : number of model tiers (tier 0 = cheap default; tiers >= 1 = escalation).
    cost : (N, A) array, ``cost[i, a]`` = expected token spend of job i on tier a.
    value : (N, A) array, ``value[i, a]`` = value (in token-equivalent credits)
        gained by running job i on tier a. ``value[:, 0] == 0`` by convention
        (the cheap tier is the baseline against which escalation value is measured).
    day : (N,) int array, the day each job runs on.
    caps : (D,) array, hard spend cap (tokens) for each day.
    edges : list of (i, j, w) DAG dependency edges (i -> j) with coupling weight w >= 0.
    roles : (N,) list of role names (metadata / generation provenance).
    credit_unit : tokens per coarse credit unit used by the cap encoding.
    is_synthetic : True for generated instances, False for loaded real data.
    name : human-readable label, surfaced in every output.
    """

    N: int
    A: int
    cost: np.ndarray
    value: np.ndarray
    day: np.ndarray
    caps: np.ndarray
    edges: List[Tuple[int, int, float]]
    roles: List[str] = field(default_factory=list)
    credit_unit: float = 1.0
    is_synthetic: bool = True
    name: str = "instance"

    @property
    def D(self) -> int:
        """Number of days."""
        return len(self.caps)

    @property
    def num_vars(self) -> int:
        """Number of decision (x) variables, excluding slack bits."""
        return self.N * self.A

    def jobs_on_day(self, d: int) -> List[int]:
        return [i for i in range(self.N) if self.day[i] == d]

    @property
    def data_kind(self) -> str:
        return "SYNTHETIC" if self.is_synthetic else "REAL"

    def summary(self) -> str:
        return (
            f"[{self.data_kind}] {self.name}: N={self.N} jobs, A={self.A} tiers, "
            f"D={self.D} days, {len(self.edges)} DAG edges, "
            f"credit_unit={self.credit_unit:g} tokens"
        )


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #

# Plausible per-role base token volumes for the cheap tier. Researcher proposals
# are short; Executor runs are the heavy hitters; Analyst digests are medium.
# (median tokens, lognormal sigma)
_ROLE_TOKEN_PROFILE = {
    "Researcher": (4_000, 0.5),
    "Executor": (40_000, 0.7),
    "Analyst": (12_000, 0.5),
}


def generate_instance(
    N: int = 12,
    A: int = 2,
    D: int = 3,
    seed: int = 0,
    escalation_multiplier: Tuple[float, float] = (3.0, 8.0),
    cap_tightness: float = 0.35,
    edge_density: float = 0.4,
    credit_unit: float = 2_000.0,
    name: Optional[str] = None,
) -> Instance:
    """Generate a reproducible synthetic agentic-workload instance.

    Parameters
    ----------
    N, A, D : jobs, tiers, days.
    seed : RNG seed (reproducible).
    escalation_multiplier : (lo, hi) range; the most expensive tier costs this
        many times the cheap tier. Intermediate tiers are interpolated.
    cap_tightness : daily cap = all-cheap baseline spend + ``cap_tightness`` ·
        (escalation headroom), where headroom = Σ (expensive − cheap) on that day.
        So the all-cheap plan is ALWAYS feasible (the budget never starves the
        baseline), and ``cap_tightness`` directly controls how much of the full
        escalation the budget permits: 0 -> no escalation affordable, 1 -> all
        escalation affordable. Values ~0.2-0.5 make the escalation decision bind.
    edge_density : probability of a forward DAG edge between eligible job pairs.
    credit_unit : tokens per coarse credit unit (passed through to Instance).
    name : label.

    Returns
    -------
    Instance with ``is_synthetic=True``.
    """
    if A < 2:
        raise ValueError("A must be >= 2 (need a cheap tier and at least one escalation tier)")
    rng = np.random.default_rng(seed)

    # --- roles: bias the role mix toward the natural pipeline shape ----------
    roles = list(rng.choice(ROLES, size=N, p=[0.4, 0.35, 0.25]))

    # --- days: assign roughly by role order so the DAG can be made acyclic ---
    # Researchers tend to run earlier in the multi-day window, Analysts later.
    day = np.empty(N, dtype=int)
    for i, r in enumerate(roles):
        # bias the day toward the role's position in the pipeline
        center = ROLE_ORDER[r] / max(1, len(ROLES) - 1) * (D - 1)
        d = int(np.clip(round(rng.normal(center, 0.8)), 0, D - 1))
        day[i] = d

    # --- costs: cheap-tier base from role profile; escalate by multiplier ----
    cost = np.zeros((N, A))
    lo, hi = escalation_multiplier
    tier_mult = np.linspace(1.0, 1.0, A)
    if A > 1:
        tier_mult = np.concatenate([[1.0], np.linspace(lo, hi, A - 1)])
    for i, r in enumerate(roles):
        median, sigma = _ROLE_TOKEN_PROFILE[r]
        base = float(rng.lognormal(mean=np.log(median), sigma=sigma))
        cost[i, :] = base * tier_mult

    # --- escalation value: gain from escalating, in token-equivalent credits -
    # Some jobs benefit a lot from a stronger model (hard reasoning), others
    # barely. We model value[:, a] as a fraction of the *marginal* cost of that
    # tier scaled by a per-job "difficulty" in [0, 1]; difficulty>~0.5 means
    # escalation pays off. value[:, 0] = 0 (cheap tier is the baseline).
    value = np.zeros((N, A))
    difficulty = rng.beta(2.0, 2.0, size=N)  # centred ~0.5, full support
    for i in range(N):
        for a in range(1, A):
            marginal = cost[i, a] - cost[i, 0]
            # value scales with difficulty: at difficulty=1 escalation returns
            # up to ~1.6x its marginal cost (clearly worth it); at 0, ~0.4x.
            value[i, a] = marginal * (0.4 + 1.2 * difficulty[i])

    # --- DAG edges: forward in (day, role-order, index); respect acyclicity --
    order = sorted(range(N), key=lambda i: (day[i], ROLE_ORDER[roles[i]], i))
    pos = {i: k for k, i in enumerate(order)}
    edges: List[Tuple[int, int, float]] = []
    for a_idx in range(N):
        for b_idx in range(N):
            i, j = order[a_idx], order[b_idx]
            if pos[i] >= pos[j]:
                continue
            # only connect across a role boundary or same-day adjacency, and never
            # backwards in time
            if day[j] < day[i]:
                continue
            if rng.random() < edge_density and ROLE_ORDER[roles[j]] >= ROLE_ORDER[roles[i]]:
                # coupling weight ~ marginal cost of the downstream escalation:
                # wasting escalation on a big downstream job is more costly.
                w = float(cost[j, A - 1] - cost[j, 0]) * rng.uniform(0.3, 0.7)
                edges.append((i, j, w))

    # keep the DAG sparse/realistic: cap out-degree implicitly via density; dedupe
    # (already unique by construction).

    # --- daily caps: all-cheap baseline + a fraction of escalation headroom --
    # cap[d] = Σ cheap + cap_tightness · Σ (most-expensive − cheap). The all-cheap
    # plan always fits; cap_tightness controls how much escalation is affordable.
    caps = np.zeros(D)
    for d in range(D):
        jobs = [i for i in range(N) if day[i] == d]
        cheap = sum(cost[i, 0] for i in jobs)
        headroom = sum(cost[i, A - 1] - cost[i, 0] for i in jobs)
        caps[d] = max(1.0, cheap + cap_tightness * headroom)

    if name is None:
        name = f"synthetic_N{N}_A{A}_D{D}_seed{seed}"

    return Instance(
        N=N, A=A, cost=cost, value=value, day=day, caps=caps,
        edges=edges, roles=roles, credit_unit=credit_unit,
        is_synthetic=True, name=name,
    )


# --------------------------------------------------------------------------- #
# Real run-database loader
# --------------------------------------------------------------------------- #

def load_instance(path: str, credit_unit: float = 2_000.0, name: Optional[str] = None) -> Instance:
    """Load a real run-database export (CSV or JSON) into an Instance.

    Expected schema (documented in README). JSON form::

        {
          "tiers": ["cheap", "gpt-x"],            # length A; index 0 = cheap default
          "caps": [120000, 120000, 90000],        # length D, tokens/day
          "credit_unit": 2000,
          "jobs": [
            {"id": 0, "role": "Executor", "day": 0,
             "cost": [38000, 190000], "value": [0, 90000]},
            ...
          ],
          "edges": [[0, 3, 1500], ...]            # [i, j, w]  (i -> j, weight tokens)
        }

    CSV form: one ``jobs.csv`` with columns
    ``id, role, day, cost_0..cost_{A-1}, value_0..value_{A-1}`` plus sibling
    ``edges.csv`` (``i,j,w``) and ``caps.csv`` (``day,cap``). Use the JSON form
    unless you have a reason not to; it is unambiguous.

    Real instances are tagged ``is_synthetic=False`` and labelled ``[REAL]`` in
    every output. No real export ships with the repo.
    """
    if path.endswith(".json"):
        return _load_json(path, credit_unit, name)
    raise ValueError(
        "CSV multi-file loading is supported via load_instance_csv(dir); "
        "pass a .json file here. See README for the schema."
    )


def _load_json(path: str, credit_unit_default: float, name: Optional[str]) -> Instance:
    with open(path) as fh:
        d = json.load(fh)
    jobs = d["jobs"]
    N = len(jobs)
    A = len(d["tiers"])
    cost = np.zeros((N, A))
    value = np.zeros((N, A))
    day = np.zeros(N, dtype=int)
    roles = ["?"] * N
    by_id = {j["id"]: k for k, j in enumerate(jobs)}
    for k, j in enumerate(jobs):
        cost[k, :] = j["cost"]
        value[k, :] = j.get("value", [0.0] * A)
        day[k] = int(j["day"])
        roles[k] = j.get("role", "?")
    caps = np.asarray(d["caps"], dtype=float)
    edges = [(by_id[i], by_id[j], float(w)) for i, j, w in d.get("edges", [])]
    return Instance(
        N=N, A=A, cost=cost, value=value, day=day, caps=caps, edges=edges,
        roles=roles, credit_unit=float(d.get("credit_unit", credit_unit_default)),
        is_synthetic=False, name=name or f"real:{path}",
    )


def load_instance_csv(directory: str, credit_unit: float = 2_000.0, name: Optional[str] = None) -> Instance:
    """Load a real instance from ``jobs.csv`` / ``edges.csv`` / ``caps.csv`` in *directory*."""
    import os

    jobs_rows = list(csv.DictReader(open(os.path.join(directory, "jobs.csv"))))
    N = len(jobs_rows)
    # infer A from cost_* columns
    cost_cols = sorted(c for c in jobs_rows[0] if c.startswith("cost_"))
    A = len(cost_cols)
    value_cols = sorted(c for c in jobs_rows[0] if c.startswith("value_"))
    cost = np.zeros((N, A))
    value = np.zeros((N, A))
    day = np.zeros(N, dtype=int)
    roles = ["?"] * N
    id_to_idx = {}
    for k, row in enumerate(jobs_rows):
        id_to_idx[int(row["id"])] = k
        day[k] = int(row["day"])
        roles[k] = row.get("role", "?")
        for a, c in enumerate(cost_cols):
            cost[k, a] = float(row[c])
        for a, c in enumerate(value_cols):
            value[k, a] = float(row[c])

    caps_rows = list(csv.DictReader(open(os.path.join(directory, "caps.csv"))))
    caps = np.zeros(len(caps_rows))
    for row in caps_rows:
        caps[int(row["day"])] = float(row["cap"])

    edges = []
    edges_path = os.path.join(directory, "edges.csv")
    try:
        for row in csv.DictReader(open(edges_path)):
            edges.append((id_to_idx[int(row["i"])], id_to_idx[int(row["j"])], float(row["w"])))
    except FileNotFoundError:
        pass

    return Instance(
        N=N, A=A, cost=cost, value=value, day=day, caps=caps, edges=edges,
        roles=roles, credit_unit=credit_unit, is_synthetic=False,
        name=name or f"real:{directory}",
    )
