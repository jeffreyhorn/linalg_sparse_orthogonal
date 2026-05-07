# Sprint 26 Day 10 — Per-Vertex Separator Scoring Design + Quick-Look

## Decision (preliminary; Day 11-12 finalises)

**Per-vertex sep scoring SHIPS as advisory only**; the Day-10
quick-look on Pres_Poisson shows a **+29pp regression** (catastrophic
on the headline fixture).  No Sprint 26 default flip.  Day 11 runs
the full corpus sweep to confirm + finalise the decision.

The implementation is correct (deterministic; produces measurably
different separators; default code path bit-identical).  The
empirical result tells us per-vertex tie-break-style separator
extraction picks a **worse** separator than the side-then-lift
heuristics on regular-mesh fixtures like Pres_Poisson.

## Implementation (PLAN.md task 1-3)

### Scoring formula

```
score(v) = 2 * cross_deg(v) + balance_bonus(v)
```

where:
- `cross_deg(v)`: number of v's neighbours on the OTHER side
  (high = "deep boundary" — lifting v removes many cross-edges per
  separator vertex)
- `balance_bonus(v)`: 1 if v's side is the LARGER side; 0 otherwise
  (preferentially lifts from the larger side to improve balance)

The 2× multiplier on `cross_deg` ensures cross-degree dominates the
ranking; `balance_bonus` is effectively a tie-break in the integer
score.  Single integer score keeps the `qsort` comparator simple.

### Selection algorithm (top-K with 70/30 balance)

1. Compute `is_boundary[v]` for every vertex (Sprint 22 Day 4
   convention; reuse existing pass).
2. For each boundary vertex, compute `score(v)` per the formula
   above.  O(|E|) total.
3. Sort all boundary vertices by score DESCENDING (ties broken by
   lower vertex id, deterministic).  O(B log B) where B = total
   boundary count.
4. Greedily lift one-by-one.  Track running w0/w1 as if each
   picked vertex is removed from its original side.  Stop when
   adding the next vertex would push max(w0, w1) / (w0 + w1)
   past 0.70 (the Sprint 24 Day 6 70/30 ceiling).
5. If lifted_count > 0: use the per-vertex mask for the lift.
   If lifted_count == 0 (rare; would only fire if even the
   highest-score vertex violates 70/30): fall back to
   smaller_weight side-lift via the existing default path.

### Env-var dispatch

`SPARSE_ND_SEP_LIFT_STRATEGY` extends Sprint 24 Day 6's existing
two-value parser to recognize three values:
- `smaller_weight` (default; Sprint 22)
- `balanced_boundary` (Sprint 24 Day 6 advisory)
- `per_vertex` (Sprint 26 Day 10 — new)

Out-of-range / non-numeric / missing → `smaller_weight` (default).
Refactored from the inline `if (env && strcmp(...) == 0)` chain into
a `parse_sep_lift_strategy()` helper + `sep_lift_strategy_t` enum
matching Sprint 25's parsing patterns.

### Code location

`src/sparse_graph.c::graph_edge_separator_to_vertex_separator` lines
~2146-2310.  The per-vertex code path is gated by
`if (strategy == SEP_LIFT_PER_VERTEX)` after the Sprint 22 / 24
strategy selection.  Default (`smaller_weight`) and `balanced_boundary`
paths are unchanged; bit-identical.

Net code delta: ~80 new lines (per_vertex_score_t struct + comparator
+ scoring + greedy lift loop).  ~10 lines changed in the existing
strategy-dispatch block (refactored env-var read into the new
`parse_sep_lift_strategy()` helper).

## Day 10 quick-look (PLAN.md tasks 4-5)

### Pres_Poisson under per_vertex (`bench_day10_per_vertex_pres_poisson.txt`)

| setting | nnz_L | ND/AMD | wall ms | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline (smaller_weight) | 2 536 427 | 0.9504× | 11 843 | — |
| balanced_boundary alone | 2 542 064 | 0.9525× | 10 593 | +0.05pp noise |
| **per_vertex alone** | **3 312 433** | **1.241×** | 7 776 | **+29pp REGRESS** |
| per_vertex + setting 13 (HCC + ratio=200) | 3 366 643 | 1.262× | 7 430 | +31pp regress |

**Per_vertex catastrophically regresses Pres_Poisson** by +29pp on
nnz_L.  Combining with Sprint 25's setting 13 actively makes it
WORSE (+31pp).

Why: Pres_Poisson's regular FE-mesh structure (mean degree 47.3,
CV 0.108 per Day 9 investigation) means high-cross-degree
vertices are NOT a sparse "wing" — they form a thick band along
the multilevel-pipeline-discovered boundary.  Lifting all of them
+ maintaining 70/30 balance produces a thick separator that
contains 30 % of the original side, defeating the separator-last
fill-quality argument.  The side-then-lift heuristics
(smaller_weight / balanced_boundary) lift at most ~50 % of one
side's boundary, naturally producing a thinner separator on
regular-mesh fixtures.

Note: per_vertex's wall time is the LOWEST among the four (7 776 ms
vs 11 843 ms baseline).  This is a side effect of the larger
separator: more vertices peel off the recursion at depth 0, so
fewer vertices recurse + leaf-AMD picks them up.  But this wall
"win" is dwarfed by the +29pp fill regression.

### Kuu under per_vertex (`bench_day10_per_vertex_kuu.txt`)

| setting | nnz_L | ND/AMD | wall ms | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline (smaller_weight) | 881 177 | 2.169× | 6 384 | — |
| **per_vertex alone** | **691 880** | **1.703×** | 4 703 | **-21pp WIN** |
| balanced_boundary alone | 527 884 | 1.299× | 3 822 | -40pp WIN (better) |

Per_vertex on Kuu lands -21pp from baseline — between
smaller_weight and balanced_boundary.  But **balanced_boundary at
-40pp is still strictly better**; per_vertex is not the optimal
advisory for Kuu.

## Per-fixture advisory (preliminary; Day 11-12 will validate)

| workload | recommended setting | notes |
|---|---|---|
| Pres_Poisson | `SPARSE_ND_COARSENING=hcc SPARSE_ND_COARSEN_FLOOR_RATIO=200` (Sprint 25 setting 13; unchanged) | per_vertex would regress this fixture by +29pp; AVOID |
| Kuu / irregular SPDs | `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (Sprint 24 advisory; unchanged) | per_vertex provides -21pp vs baseline but balanced_boundary's -40pp is better |
| Default (no env vars) | baseline | unchanged |

## Hypothesis falsification: per_vertex doesn't help

Day 4's per-recursion-depth profile + Day 8's FIFO falsification +
Day 10's per_vertex quick-look together establish a clear pattern:

**Tie-break-and-scoring-style interventions don't move
Pres_Poisson nnz_L.**  Sprint 26's empirical findings:
- Day 7-8 (FIFO bucket-tie-break): Pres_Poisson +3pp regress
- Day 10 (per-vertex separator scoring): Pres_Poisson +29pp regress
- Day 9 (geometric grid-cut): not even applicable (Pres_Poisson is
  not a regular 2D grid)

The Pres_Poisson 0.85× literal target requires a structural
change to the multilevel pipeline itself (e.g. different coarsening
that tightens cuts at the COARSEST level, then projects back), not
a tie-break tweak at the FM-cascade or separator-extraction level.
Sprint 25's setting 13 (HCC + ratio=200) is the closest the
codebase has come — 0.9217× — and it operates at the coarsening
floor level.

## Day 11-12 plan (consolidated)

Per Day 9's re-allocation, Day 10 pulled Item 7 implementation
forward.  Day 11-12 collapse to:

- **Day 11**: full cross-corpus sweep of per_vertex (8 settings ×
  6 fixtures matrix matching Sprint 26 Day 8's structure).  Apply
  the flip rule.  Document the no-flip outcome + per-fixture
  advisory.  Save sweep + decision docs to
  `SPRINT_26/per_vertex_sep_sweep.txt` + `per_vertex_sep_decision.md`.
- **Day 12**: pulled-forward Day 13 work (cross-corpus re-bench +
  test bound tightening) starts here.  Item 7 closes; Sprint 26's
  three algorithmic-axis attempts (Items 5, 6, 7) all closed.
- **Day 13**: original cross-corpus re-bench + production-default
  decisions + test-bound tightening.
- **Day 14**: original soak + retro + PR.

Net: 1 day pulled forward (Day 12 starts what Day 13 was scheduled
for), preserving the Day 14 retro slot.

## Sprint 27 inputs

If Sprint 27 inherits the Pres_Poisson 0.85× target (now likely,
given Items 5-7 all fall short), the per_vertex implementation
ships behind the env var and could be combined with future
algorithmic axes:

- **per_vertex + new coarsening strategies**: a Sprint 27
  HCC-improved coarsening might produce small-boundary cuts where
  per_vertex's "lift entire boundary" approach DOES land tighter
  separators.  Sprint 26's per_vertex regression on Pres_Poisson
  reflects the existing HEM/HCC coarsening's "thick boundary" output;
  a different coarsening could change the answer.
- **per_vertex + tunable scoring weights**: Day 10's score formula
  is `2 * cross_deg + balance_bonus`.  Sprint 27 could sweep
  alternative scoring weights (existing-separator-adjacency,
  coarsening-cmap-stability, vertex-degree as a softening factor).
  PLAN.md task 1 lists these but Day 10 implemented only the
  simplest two-feature formula.

## What ships in Sprint 26 Day 10

- `src/sparse_graph.c`:
  - New `sep_lift_strategy_t` enum + `parse_sep_lift_strategy()`
    helper (refactored from inline env-var check)
  - `per_vertex_score_t` + comparator
  - `SEP_LIFT_PER_VERTEX` code path in
    `graph_edge_separator_to_vertex_separator` — ~80 new lines
- `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_design.md` (this
  doc) — design + Day 10 quick-look findings + Day 11-12 routing
- `docs/planning/EPIC_2/SPRINT_26/bench_day10_per_vertex_pres_poisson.txt`
- `docs/planning/EPIC_2/SPRINT_26/bench_day10_per_vertex_kuu.txt`
- All quality checks clean.  Default code path bit-identical
  (`SEP_LIFT_SMALLER_WEIGHT` is the default; existing tests pass).

## References

- `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 10 + Day 11
- `docs/planning/EPIC_2/SPRINT_26/geometric_cut_design.md` — Day 9's
  rejection that pulled this work forward
- `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md` — Day 8's
  FIFO escalation point
- `docs/planning/EPIC_2/SPRINT_24/nd_sep_strategy_decision.md` —
  Sprint 24 Day 6's `balanced_boundary` decision (the prior art
  that per_vertex extends)
- `src/sparse_graph.c::graph_edge_separator_to_vertex_separator` —
  Day 10 implementation site
