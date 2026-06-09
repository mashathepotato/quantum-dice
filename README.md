# Quantum Dice Trinity Challenge — Stage 2

**Agentic job-to-tier routing as a QUBO**, sampled with simulated annealing (a
stand-in for ORBIT p-bit hardware), and benchmarked against exact classical
solvers.

A single autonomous research lab runs a queue of jobs (Researcher / Executor /
Analyst) with DAG dependencies. Each job is routed to a **cheap default tier** or
an **expensive escalation tier**, under a **hard daily spend cap** inside a fixed
multi-week credit budget. The decision: *which jobs to escalate, and when
escalation pays off, without blowing the daily cap.*

See **[STAGE2_REPORT.md](STAGE2_REPORT.md)** for the full story (problem →
formulation → implementation → results) and the mapping to the Stage 2 criteria,
**[NOTES.md](NOTES.md)** for the formulation and how it reconciles with the Stage 1
proposal (`proposal/`), and the **[Formulation](#formulation)** section below for
the energy at a glance.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pinned versions (Python 3.9, macOS): dimod 0.12.21, dwave-samplers 1.6.0,
PuLP 3.3.1 (CBC), numpy 2.0.2, matplotlib 3.9.4, networkx 3.2.1, pytest 8.4.2.

## Run

```bash
pytest                       # 23 tests: BQM energy by hand, validation, baselines, loader
python run_all.py            # tests + all 5 experiments -> results/  (~25 s)
python run_all.py --no-tests # experiments only
```

Each experiment is also runnable on its own and writes its own figures/tables:

```bash
cd experiments
python 01_penalty_calibration.py   # -> results/01_*  penalty regime + cap-encoding v1 vs v2
python 02_slack_bits.py            # -> results/02_*  slack-bit sufficiency / failure mode
python 03_scaling.py               # -> results/03_*  scaling in jobs N and tiers A
python 04_diversity.py             # -> results/04_*  distribution of near-optimal plans
python 05_baseline_comparison.py   # -> results/05_*  SA vs ExactSolver vs ILP vs greedy
```

### Reproducing every figure

`python run_all.py` regenerates every artefact in `results/` from scratch
(everything is seeded). Figure ↔ script map:

| Figure / table | Produced by |
| --- | --- |
| `01_penalty_feasibility.png`, `01_penalty_gap.png`, `01_encoding_comparison.png`, `01_penalty_calibration.md` | `experiments/01_penalty_calibration.py` |
| `02_slack_bits.png`, `02_slack_bits.md` | `experiments/02_slack_bits.py` |
| `03_scaling.png`, `03_scaling_tiers.png`, `03_scaling.md` | `experiments/03_scaling.py` |
| `04_diversity_energy.png`, `04_diversity_count.png`, `04_diversity.md` | `experiments/04_diversity.py` |
| `05_baseline_comparison.png`, `05_baseline_comparison.md` | `experiments/05_baseline_comparison.py` |

## Repository layout

```
src/
  data.py         Instance dataclass; seeded synthetic agentic-workload generator; real-data loader
  formulation.py  build_bqm(); single audited squared-penalty helper; decode; slack sizing
  solvers.py      Sampler interface (SA implementation + ORBIT stub) — ORBIT swap is one line
  baselines.py    brute force, dimod.ExactSolver, PuLP ILP (CBC), greedy heuristic
  validate.py     decode + hard-constraint checks (one-hot, daily cap), wasted-escalation count
  metrics.py      objective, spend, optimality gap, soft objective
experiments/      01..05 + _common.py
tests/            test_formulation, test_validate, test_baselines, test_data_loader
results/          generated figures + tables
proposal/         Stage 1 submission (ground truth)
NOTES.md          plan + proposal reconciliation + implementation iteration log
STAGE2_REPORT.md  the writeup
```

## Formulation

Decision `x[i,a] ∈ {0,1}` = job `i` on tier `a` (0 = cheap, ≥1 = escalation).
Minimise

```
H = Σ_{i,a} (c_{i,a} − v_{i,a}) x_{i,a}                                   objective (escalate where value beats marginal cost)
  + P_A Σ_i (1 − Σ_a x_{i,a})²                                           one-hot: assign each job once          (hard)
  + P_B Σ_d (Σ_{i∈d,a} Δ̃_{i,a} x_{i,a} + Σ_k 2^k s_{d,k} − headroom_d)²  daily cap via binary slack             (hard)
  + P_C Σ_{i→j} w_{ij} · e_j · (1 − e_i),   e_i = Σ_{a≥1} x_{i,a}        wasted-escalation coupling over DAG    (soft)
```

The coupling term `e_j(1−e_i)` penalises escalating a downstream job whose upstream
input was produced cheaply — the genuinely **quadratic** structure that makes this
a QUBO rather than a separable knapsack. `Δ̃` and `headroom` are in coarse credit
units (cost rounding is a documented realism-vs-tractability trade-off). Full
derivation and the proposal reconciliation are in [NOTES.md](NOTES.md).

## Swapping in ORBIT at Stage 3

`src/solvers.py` defines a `Sampler` interface with one method,
`sample(bqm, num_reads) -> dimod.SampleSet`. Stage 2 uses `SASampler`
(dwave-samplers simulated annealing). `OrbitAdapter` is a stub that raises with a
TODO until the ORBIT hardware lands (12 June 2026). Switching is a **one-line
change** at the call site:

```python
sampler = SASampler()          # Stage 2 (now)
sampler = OrbitAdapter(...)     # Stage 3 (when available)
```

Everything downstream (experiments, baselines, validation) is sampler-agnostic.

## Real data

All shipped results are on **synthetic** data, clearly labelled `[SYNTHETIC]` in
every figure title and console line. To route a real workload, drop a run-database
export into `data/` and load it with `src.data.load_instance("data/your.json")`
(or `load_instance_csv`). The schema is documented in
`data/example_run_export.json` (an illustrative example — **not** real data) and in
the `load_instance` docstring. Loaded instances are tagged `[REAL]`.
