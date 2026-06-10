# NOTES — Stage 2 design, plan, and reconciliation with the Stage 1 proposal

This file records (a) the file-by-file build plan and (b) every place where the
Stage 2 implementation deviates from the Stage 1 proposal
(`proposal/neurips_2026.tex`) or from the Stage 2 scaffold, with justification.
The proposal is the authoritative *problem* specification; where the scaffold
refines a term in a domain-justified way, the deviation is documented here.

---

## 1. Reconciliation: proposal ↔ scaffold ↔ implementation

### 1.1 Objective: pure cost (proposal) → cost minus escalation value (implemented)

* **Proposal** (Early QUBO mapping): `min Σ_{i,a} c_{i,a} x_{i,a}`, a pure token-cost
  minimisation, with a capability mask `c_{i,a} → ∞` where a tier cannot serve a job.
* **Problem with the literal proposal objective:** with a pure-cost objective and
  `c_{i,1} > c_{i,0}`, the cheap tier is *always* weakly optimal, so the model would
  never escalate unless a capability mask *forces* it. The interesting Stage 2
  decision — *"which jobs is it worth escalating, and when does escalation pay off?"* —
  then disappears.
* **Implemented:** linear objective
  `Σ_i ( c_{i,0} x_{i,0} + (c_{i,1} − v_i) x_{i,1} )`, generalised to A tiers as
  `Σ_{i,a} (c_{i,a} − v_{i,a}) x_{i,a}` with `v_{i,0}=0`. Here `v_i` is the **value of
  escalating job i** (e.g. expected quality/throughput gain expressed in the same
  credit units as cost). Escalation is chosen only where its value beats its marginal
  cost `c_{i,1} − c_{i,0}`; the daily cap then forces trade-offs across competing jobs.
* **Why this is faithful to the proposal:** the proposal explicitly says "the marginal
  value of escalation is not separable across jobs" — i.e. escalation *has* a value.
  The capability mask is the degenerate case `v_i → +∞` (must escalate) or
  `c_{i,0} → ∞` (cannot use cheap tier). The value term is the natural Stage 2
  generalisation, and the capability mask is still supported via large costs/values.

### 1.2 Coupling term C3: shared-context affinity (proposal) → DAG "wasted escalation" (implemented)

* **Proposal C3:** `− P_esc Σ_{i<j} w_{ij} x_{i,hi} x_{j,hi}` — rewards sending *related*
  jobs (shared context/hypothesis) to the expensive tier *together*. Pairwise over an
  undirected "relatedness" graph.
* **Scaffold C3:** `+ P_C Σ_{i→j} w_{ij} x_{j,1}(1 − x_{i,1})` over **DAG edges** —
  penalises *wasted escalation*: paying for an expensive downstream job `j` whose
  upstream input `i` was produced on the cheap tier (low-quality input bottlenecks the
  expensive downstream step). The scaffold explicitly **drops** shared-context affinity
  because **prompt caching already absorbs most shared-context cost**.
* **Decision: adopt the scaffold's DAG term.** Rationale:
  1. The proposal *itself* defers DAG precedence to "Stage 2–3" and treats within-cycle
     jobs as independent in Stage 1. Adding the DAG is precisely the promised Stage 2
     evolution, so this is not a contradiction of the proposal but its continuation.
  2. The prompt-caching argument is a real, defensible domain fact: shared static
     context is cached at the provider, so co-escalation no longer yields the cost
     synergy the proposal's affinity term assumed. The wasted-escalation coupling
     captures a *different* and still-genuine non-separability: downstream escalation
     value depends on the upstream tier choice.
  3. Both terms are genuinely quadratic (`x_j·x_i` product), so C3 still makes this a
     true QUBO rather than a separable knapsack — the property the proposal cared about.
* **Generality:** for A>2 tiers the "escalated" indicator is `e_i = Σ_{a≥1} x_{i,a}`,
  and the term is `w_{ij} e_j (1 − e_i)` — still quadratic (product of two linear forms).
* **Sign / direction:** `w_{ij} ≥ 0` (a penalty), matching the scaffold. The proposal's
  reward-direction (`w<0`) is representable too but is not the default.

### 1.3 Single budget B (proposal) → per-day hard caps B[d] (implemented)

* **Proposal:** one spend cap `Σ c x ≤ B`.
* **Implemented:** each job runs on a day `d(i)`; a **hard daily cap** `B[d]` per day,
  `Σ_{i: d(i)=d, a} c_{i,a} x_{i,a} ≤ B[d]`, encoded with squared-slack penalties per day.
  This is the "fixed multi-week credit budget enforced as a HARD daily spend cap"
  described in the Stage 2 brief and is the multi-day generalisation of the proposal's
  single cap. Days are assigned consistently with the DAG (an edge `i→j` never has
  `d(j) < d(i)`).

### 1.4 Cost rounding to coarse credit units (realism vs. tractability)

Token costs are large integers (thousands–millions). The squared-slack cap encoding
needs `K_d = ceil(log2(B̃[d]+1))` slack bits *per day*, so we **round costs and caps to a
coarse "credit unit"** (`credit_unit` tokens per unit) before building the cap penalty.
This is an explicit realism-vs-tractability trade-off: coarser units → fewer slack bits
→ smaller/easier QUBO, but the cap is enforced only to ±1 credit unit. Documented and
swept in `experiments/02_slack_bits.py`.

---

## 2. Formulation (final energy)

```
H(x,s) = H_obj  +  P_A · H_onehot  +  P_B · H_cap  +  P_C · H_couple

H_obj     = Σ_{i,a} (c_{i,a} − v_{i,a}) · x_{i,a}                         (linear objective)
H_onehot  = Σ_i ( 1 − Σ_a x_{i,a} )^2                                     (hard: assign once)
H_cap     = Σ_d ( Σ_{i∈d,a} c̃_{i,a} x_{i,a} + Σ_k 2^k s_{d,k} − B̃[d] )^2 (hard: daily cap, slack)
H_couple  = Σ_{i→j} w_{ij} · e_j · (1 − e_i),   e_i = Σ_{a≥1} x_{i,a}     (soft: wasted escalation)
```

* `P_A, P_B` large (hard); `P_C` moderate (soft). The hard penalties must exceed the
  largest objective gain achievable by violating a constraint — finding that regime is
  experiment 01.
* `c̃, B̃` are costs/caps in rounded credit units (see 1.4).
* All squared penalties are expanded to QUBO linear+quadratic+offset terms by one
  audited helper `add_squared_penalty` (single source of truth, unit-tested).

---

## 3. File-by-file plan

```
src/
  data.py         Instance dataclass; seeded synthetic agentic-workload generator;
                  CSV/JSON loader for a real run-DB export (data/). Labels synthetic vs real.
  formulation.py  build_bqm(instance, penalties) -> dimod.BQM; add_squared_penalty helper;
                  variable labelling x(i,a)/s(d,k); decode(sample)->assignment.
  solvers.py      Sampler ABC (sample(bqm,num_reads)->SampleSet); SASampler (dwave-samplers);
                  OrbitAdapter stub (NotImplementedError + TODO). One-line swap at Stage 3.
  baselines.py    ExactSolver / brute-force (≤~20 bits); PuLP ILP (same problem, coupling
                  linearised) for true optimum/feasibility; greedy heuristic.
  validate.py     decode + check one-hot, per-day cap, count wasted-escalation edges;
                  Feasibility report dataclass.
  metrics.py      objective, spend-per-day, #escalated, energy, optimality gap, agreement.

experiments/
  01_penalty_calibration.py   sweep P_A,P_B vs objective scale -> feasible regime (heatmap).
  02_slack_bits.py            vary K_d; show cap broken when too few bits; correct when enough.
  03_scaling.py               vary N/days/tiers; SA time-to-feasible & quality vs ILP.
  04_diversity.py             num_reads -> distribution of distinct near-optimal solutions.
  05_baseline_comparison.py   SA vs ExactSolver vs PuLP: gap, feasibility, agreement.

tests/
  test_formulation.py  hand-computed energy on a tiny instance; squared-penalty helper.
  test_validate.py     one-hot & cap violation detection; feasible passes.
  test_baselines.py    tiny instance w/ known optimum: ExactSolver==PuLP==bruteforce.

results/   generated figures (.png) + tables (.md/.csv)
README.md  install / run / reproduce-every-figure
STAGE2_REPORT.md  problem→formulation→implementation→results + criteria mapping
```

## 4. Build order (TDD where correctness matters)
1. data.py (instance + generator).
2. formulation.py + test_formulation.py (energy hand-check before trusting the BQM).
3. validate.py / metrics.py + test_validate.py.
4. solvers.py.
5. baselines.py + test_baselines.py (known-optimum agreement).
6. Run pytest green.
7. experiments 01–05, write artefacts to results/.
8. STAGE2_REPORT.md, README.md.

Each step is a small git commit.

---

## 5. Implementation iteration log (what changed during build, and why)

These are deviations from the *scaffold* discovered while making the model
actually work under simulated annealing. They are the core "iteration" evidence.

1. **Slack sized to the slack *range*, not the cap.** The scaffold suggested
   `K_d = ceil(log2(B[d]+1))`. But every job runs on at least the cheap tier, so
   daily spend never falls below the all-cheap baseline; the slack only needs to
   cover `[0, cap − all-cheap]`. Sizing to this range cut slack bits substantially
   and shrank the per-day cap clique. (`slack_bits_for`, `default_slack_bits`.)

2. **Baseline-shifted cap encoding (the big one).** The literal cap penalty
   `(Σ c̃·x + slack − B̃)²` ("absolute") put *every* job on a day — cheap or not —
   into one fully-connected quadratic clique with a large constant baseline; SA
   satisfied it only ~5% of the time. Rewriting it as
   `(Σ Δ̃·x + slack − headroom)²` ("shifted"), where `Δ̃ = c̃ − c̃[:,0]`, makes the
   cheap-tier variables drop out of the clique entirely. Identical at one-hot
   feasible points, but SA feasibility jumped from ~5% to ~80%. This is the
   default; `cap_mode="absolute"` is retained to reproduce the v1 failure
   (experiment 01, `01_encoding_comparison.png`).

3. **Penalty calibration reference = max single coefficient, not the sum.** Hard
   penalties must beat the objective a *single* constraint-cheating flip can buy,
   bounded by `S_max = max|c−v|`, not the sum over all jobs. Scaling penalties to
   `S_max` keeps the QUBO's coefficient dynamic range narrow, which SA needs.
   (`objective_coeff_max`; experiment 01 finds the feasible regime ≳ `1·S_max`.)

4. **Cap generation = all-cheap baseline + tightness·headroom.** The initial
   `tightness·mean-tier` caps were so tight the feasible region was nearly the
   single all-cheap point. The new form always admits the all-cheap plan and lets
   `cap_tightness` directly control how much escalation the budget permits — a
   cleaner, more interpretable knob and a more realistic budget.

All four are documented in code docstrings at the point of definition.

---

## 6. Next steps (see STAGE2_REPORT §6)

Lead Stage 3 direction: **context-freshness as a routing dimension**. Agents
reason best in roughly the first ~50k tokens of a context and degrade as it fills,
so the most intellectually intensive jobs should be deployed into a *fresh* context
rather than appended to a long-running one. This is complementary to the dropped
shared-context affinity term: prompt caching makes a long shared context *cheap*
(cost), but it degrades output *quality* (value) — so for high-value jobs it can
pay to start fresh. Encode as a state-dependent escalation value `v_i` discounted
by tokens already accumulated in the job's context slot (start with a static
per-job freshness discount — no new variables — then full slot-assignment with
measured qubit cost). Other extensions: real run-DB data, soft per-provider rate
limits, and the ORBIT hardware swap.
