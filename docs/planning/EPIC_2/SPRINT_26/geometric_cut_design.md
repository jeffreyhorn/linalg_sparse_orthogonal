# Sprint 26 Day 9 — Geometric Grid-Cut Design + Rejection Decision

## Decision

**Reject Item 6 (geometric grid-cut) for Sprint 26.**  Day 9's
empirical investigation found the PLAN.md grid-detection heuristic's
core premise — "Pres_Poisson is a regular 2D grid with vertex
degrees clustered at 4 (interior), 3 (edges), 2 (corners), nnz/n ≈
5" — is **false**.  Pres_Poisson is a **high-order FE-mesh
discretization** with mean vertex degree 47.3 and nnz/n = 47.29.
The PLAN's proposed heuristic would never fire on Pres_Poisson.

Re-allocating Day 10's budget: pull Item 7 (per-vertex separator
scoring) forward from Day 11 to Day 10.  Day 9 ships only the
design doc (this) + rejection rationale; no source changes.

## Empirical investigation (PLAN.md task 1)

Day 9 ran a quick degree-distribution probe across the full
Sprint 25 corpus (`/tmp/grid_check.c` + `/tmp/grid_probe.c`):

| fixture | n | nnz/n | min deg | mean deg | max deg | CV (σ/μ) | % in {3,4,5} |
|---|---|---|---|---|---|---|---|
| nos4 | 100 | 4.94 | 1 | 4.9 | 6 | 0.293 | 39 % |
| bcsstk04 | 132 | 26.64 | 14 | 26.6 | 45 | 0.404 | 0 % |
| Kuu | 7 102 | 46.90 | 22 | 46.9 | 97 | 0.425 | 0 % |
| bcsstk14 | 1 806 | 34.14 | 0 | 34.1 | 47 | 0.280 | 0 % |
| s3rmt3m3 | 5 357 | 37.66 | 6 | 37.7 | 47 | 0.188 | 0 % |
| **Pres_Poisson** | **14 822** | **47.29** | **17** | **47.3** | **49** | **0.108** | **0 %** |

PLAN.md item 6 design assumed the detection heuristic:
> (a) ≥ 90 % of vertices have degree ∈ {3, 4, 5}, (b) nnz/n ∈ [4.5, 5.5]

This heuristic:
- Would fire on **nos4 only** (the small synthetic 10×10-style grid with mean
  degree 4.9; 39 % in {3,4,5} — fails the 90 % gate but is the
  only fixture in the right vicinity).
- **Would not fire on Pres_Poisson** (0 % in {3,4,5}; nnz/n = 47.29
  ≫ 5.5).
- Would not fire on any irregular fixture (Kuu, bcsstk04, etc.) —
  the rejection criterion is robust there.

The headline-fixture rejection invalidates the original Item 6
design.  Whatever Pres_Poisson's structure is, it's not "regular
2D grid by vertex-adjacency signature".

## Why Pres_Poisson has degree ~47 — structural diagnosis

Pres_Poisson is described in `algorithm.md` as "the canonical 2D-PDE
benchmark."  The actual matrix represents a **high-order finite-
element discretization** of Poisson's equation on a 2D mesh.  Each
P2 (or higher) FE element couples ~9-12 nodes; each interior node
belongs to ~4-6 elements; total cumulative degree per node = ~30-50.

Pres_Poisson's degree distribution confirms this:
- 88.8 % of vertices in bin [40, 49] (13 166 of 14 822)
- CV = 0.108 — the *lowest* coefficient of variation in the corpus
  (Pres_Poisson is the *most regular* fixture by adjacency variance,
  just regular at a different scale than the PLAN assumed)

The matrix is regular AT A DIFFERENT SCALE than the PLAN expected.
Vertices have similar high-degree adjacency (CV < 0.15 = very tight
band), but the per-vertex degree is high (~47) rather than low (~4).

## Two redesign options (and why both rejected for Sprint 26)

### (i) High-mean-degree-regular detection + Fiedler-axis cut

**Detection criterion**: CV ≤ 0.15 + mean degree ≥ 20 → "this is a
regular FE mesh."  Detected fixtures: Pres_Poisson (CV=0.108, mean=47);
s3rmt3m3 (CV=0.188 — borderline; under 0.20).

**Cut algorithm**: project the graph onto the Fiedler vector (the
eigenvector of the second-smallest Laplacian eigenvalue), bisect at
the median value, lift the boundary edge as separator.  Reuses
Sprint 25 Day 6-8's `graph_bisect_coarsest_spectral` infrastructure
but at the ROOT level (pre-empting multilevel coarsening) rather
than the coarsest level.

**Why rejected for Sprint 26**: Sprint 25 Day 8 measured spectral
bisection AT THE COARSEST LEVEL on Pres_Poisson and found it
**neutral on nnz_L** (0.953× vs default 0.952× — essentially zero
movement).  Day 9's hypothesis is that running spectral at the
ROOT level (full graph, n=14 822) would produce different cuts than
running at the coarsest level (n ≈ 50).  But:
- Lanczos on n=14 822 is much slower than on n ≈ 50 (Sprint 21
  shift-invert measurements: Lanczos at n=14 822 takes ~5-10 s, vs
  ~10 ms at n=50).  Day 9-10 budget would be consumed by getting
  the eigensolver to scale.
- Sprint 25's spectral bisection's neutral nnz_L outcome at the
  coarsest level suggests the FIEDLER AXIS itself doesn't add
  fill-quality value on Pres_Poisson — it produces cuts of similar
  quality to GGGP + uncoarsening + FM.  Running it at root level
  changes WHEN the cut is decided, not WHAT cut it produces.
- The fundamental issue: Pres_Poisson's FM landscape (explored
  Sprint 22-26 across 200+ measurements) doesn't have a "missing"
  structural cut that Fiedler-axis would discover.

This redesigned variant might marginally help if root-level
spectral combined with leaf-AMD splice produces tighter overall
ordering, but it's a 2-day bet with no prior evidence.  Sprint 27+
budget is more appropriate.

### (ii) Original Item 6 ships for synthetic regular grids only

**Detection criterion**: PLAN.md's original (degree ∈ {3,4,5}, nnz/n
≈ 5).  Detected fixtures: nos4 only (in current corpus); also
synthetic `make_grid_2d(n, m)` test fixtures.

**Cut algorithm**: median-row-or-column cut as PLAN proposed.

**Why rejected for Sprint 26**: works for synthetic test fixtures
but doesn't help Pres_Poisson — Day 8's escalation point.  Sprint
26's headline gate is Pres_Poisson 0.85×; shipping a feature that
helps make_grid_2d-style synthetic fixtures while leaving the
headline open consumes 2 days of budget for zero headline impact.
Sprint 27+ could ship this as an opt-in feature for callers using
literal regular grids; not a Sprint-26 priority.

## Re-allocating Day 9-10 budget

Day 9-10 originally allotted 24 hours for grid-cut design +
implementation + Pres_Poisson validation.  With Item 6 rejected:

- **Day 9 (today, ~6 hours actual)**: this design doc + rejection
  rationale.  No source changes; doc-only commit.  No env-var
  skeleton + no test stubs — they would be dead code given the
  rejection.
- **Day 10 (12 hours)**: pull Item 7 (per-vertex separator scoring)
  design + implementation forward from Day 11.  This gives Item 7
  a 3-day window (Days 10-12) instead of 2 (Days 11-12), making the
  per-vertex sub-axis sweep more thorough (more weight combinations
  + fixture-targeted experiments).

The PLAN.md Day-13 cross-corpus re-bench + Day-14 retrospective
remain on schedule.

## Sprint 27+ routing for the redesigned Item 6

If Sprint 27 inherits the Pres_Poisson 0.85× target (likely, since
Item 7 alone has historically been a ≤ 1pp lever per Sprint 24 Day 6's
balanced_boundary at +0.1pp on Pres_Poisson), the **redesigned
Item 6 (high-mean-degree-regular detection + Fiedler-axis cut at
root level)** is a 4-day candidate:

- Day A: extend Sprint 25's spectral bisection from coarsest to
  root-level invocation; measure Lanczos wall on n=14 822 graphs.
- Day B: implement the regular-mesh detection heuristic
  (`SPARSE_ND_REGULAR_MESH_DETECT={off,on,auto}`); measure detection
  accuracy on Sprint 26 corpus.
- Day C: cross-corpus sweep of root-level Fiedler-cut + decide
  flip-or-stay.
- Day D: tests + docs + decision.

This deferred Item 6 has a higher prior than Sprint 26's
originally-designed Item 6 because it leverages Sprint 25's
already-validated spectral bisection infrastructure.

## What ships in Sprint 26 Day 9

- `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md` (this
  doc) — empirical investigation, original-design rejection,
  redesign options + Sprint 27 routing.
- No source changes.  Default code path bit-identical.
- All quality checks clean.

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 9 (the originally-
  proposed grid-detection heuristic this doc rejects)
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md` "Escalation
  to Day 9-10 (geometric grid-cut)" — the Day-8 routing point this
  Day-9 work was supposed to address
- `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_decision.md` —
  Sprint 25's spectral-bisection-at-coarsest-level finding
  (neutral on Pres_Poisson nnz_L) that informs Day 9's redesign-(i)
  rejection
- `/tmp/grid_check.c` + `/tmp/grid_probe.c` — Day 9's empirical
  investigation programs (intermediate; not committed)
- Pres_Poisson SuiteSparse dataset — see `tests/data/suitesparse/`
- Sprint 22-26 historical attempts at the Pres_Poisson 0.85× literal
  target (cumulative: 4 sprints, 5+ algorithmic axes, ~200
  measurements, gap remains 7.2pp)
