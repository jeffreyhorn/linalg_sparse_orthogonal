# Annealing FM at the Finest Level — Design (Sprint 27 Day 5)

## Background

Sprint 26 Day 6's `finest_fm_design.md` evaluated three FINEST-FM sub-axes:

- **(a) Annealing acceptance** — accept worsening moves with `P = exp(-Δgain / T)`; T decreases per pass.
- **(b) Bucket tie-break (FIFO)** — change the order in which equal-gain moves are processed.
- **(c) Thick-restart with global rollback** — track best cut across passes; restart each pass from saved anchor with random perturbation.

Sprint 26 Day 6 picked sub-axis (b) FIFO for Day 7 implementation (lowest risk; smallest movement).  Sub-axes (a) and (c) were **rejected on cost grounds**:
- Annealing: estimated 20-50 % wall expansion (every pass becomes randomised; no early termination on gain saturation).
- Thick-restart: estimated 200-300 % wall expansion (each pass re-runs FM from scratch from a saved anchor).

Sprint 27 PLAN.md Day 5 revisits annealing under the new wall budget:
- Sprint 26 Day 5 flipped `nd_base_threshold` 32 → 96, cutting Pres_Poisson ND wall 38.1 s → 12.2 s (-67.9 %).
- Sprint 27 Day 2 flipped `SPARSE_ND_COARSENING` heavy_edge → HCC + Kuu-safe, dropping Pres_Poisson ND wall to 8.8 s.
- Sprint 27 Day 3 flipped `nd_base_threshold` 96 → 128, dropping Pres_Poisson ND wall to 7.1 s.

**Cumulative -81 % wall vs Sprint 25 baseline.**  Annealing's projected 20-50 % wall expansion now lands at 8.5-10.6 s on Pres_Poisson — well under the `wall-check` 1.5× ceiling of 47 s × 1.5 = 70.5 s.  Affordable.

This document captures Day 5's design + temperature-schedule selection + Day 6 implementation plan.

## Annealing Acceptance Contract

Standard simulated-annealing acceptance (Kirkpatrick-1983 §3, Cherkassky-1995 §4.2):

```
For each candidate move v with gain g:
  if g >= 0:
    accept (same as baseline FM)
  else:
    accept with probability P = exp(g / T)    (g < 0; -g/T < 0; exp(-Δgain/T))
    reject with probability 1 - P
```

Where `T` is the per-pass temperature.  When `T → ∞`, the algorithm becomes random (P → 1 for all g); when `T → 0`, it becomes greedy (P → 0 for g < 0, matching baseline FM).  The temperature schedule cools `T` from a high initial value `T_0` toward 0 across passes.

### Key design choices

1. **Worsening-move acceptance is per-vertex** at pop time, not per-step (a single FM pass typically pops O(n) vertices).  Each pop's acceptance check pays one `exp` evaluation.
2. **Best-cut tracking is unchanged** — Sprint 23 Day 11's "track the best cut seen across the pass; restore at end of pass" is preserved.  Annealing produces a noisier *trajectory* but the saved best-cut still floors the result at baseline-or-better.
3. **Determinism**: requires a seeded RNG for reproducibility.  Sprint 27 Day 5 reuses the existing graph_uncoarsen seed (passed through to graph_refine_fm via thread-local + per-pass increment).  Day 6 implementation: a per-call `xorshift32` seeded from the level + pass number gives reproducible randomness.
4. **Determinism contract under default-off**: when `SPARSE_FM_FINEST_STRATEGY != annealing`, no randomness is introduced — the baseline path stays bit-identical to current master.

## Temperature Schedule Selection

Sprint 27 PLAN.md Day 5 task 3 names three plausible variants.

### (a) Linear: T_k = T_0 × (1 − k/K)

- Simplest formulation.  Predictable cutoff at pass K (T → 0).
- Linear decrease means the middle passes get medium-temperature acceptance — annealing-typical, but no sharp transition.
- Risk: too-slow cooling wastes wall budget on already-converged states; too-fast cooling is no different from baseline.

### (b) Exponential: T_k = T_0 × α^k where α ≈ 0.5

- **Classical Kirkpatrick-1983 formulation.**  Halve T each pass; pass 0 = T_0, pass 1 = T_0/2, pass 2 = T_0/4, etc.
- Cheapest to compute (one multiplication per pass).
- Front-loads worsening-acceptance: passes 0-1 have the most variability, passes 4-5 are essentially greedy.
- Risk: premature freeze if α is too low (acceptance rate drops to ~0 before exploring enough of the cut landscape).

### (c) Cosine: T_k = T_0/2 × (1 + cos(πk/K))

- Popular in modern training schedules (deep learning warmup-then-decay).
- Slow start (high T for the first 1-2 passes), fast end (T → 0 by pass K).
- More worsening-acceptance early; sharper convergence late.
- Risk: more `exp` evaluations per pass on average; needs `cos` evaluation per pass (cheap but extra).

### Decision: Default Exponential (Variant b)

Reasons:

1. **Classical SA formulation.**  Kirkpatrick-1983 + Cherkassky-1995 both pick exponential as the canonical schedule.  Easy to reason about; predictable behaviour.
2. **Cheapest to compute.**  One multiplication per pass; no transcendental at schedule-update time.
3. **Acceptance probability `exp` evaluation cost is the dominant per-step overhead anyway.**  The schedule itself is amortised across O(n) pops per pass; cosine's extra `cos` per pass is noise vs the `exp` per pop.
4. **Day 6 implementation can sweep all three** under `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential (default), cosine}` to validate the choice empirically.  Day 5 wires the parser; Day 6 implements all three branches.

### Sweep Dimensions for Day 6/7

- **Schedule**: linear / exponential (default) / cosine.
- **Initial temperature `T_0`**: max gain in pass 1 (default; matches Cherkassky-1995's "T_0 = max gain such that 80 % of moves are accepted").  Could also try `T_0 = mean |gain|` or fixed constants.
- **Cooling rate `α`** (exponential only): default 0.5 (halve per pass).  Could try 0.3 (faster cool) or 0.7 (slower cool).
- **Cutoff**: T < 1 (move accepted greedily — same as baseline).  Could parameterise.
- **Number of passes**: inherits `SPARSE_FM_FINEST_PASSES` (default 3 per Sprint 23 Day 11).  Annealing may need more passes (5-7) to fully explore the landscape — Day 7 measurement decides.

## Implementation Plan

### Day 5 (this commit) — Skeleton

1. Add `_Thread_local int fm_use_annealing` (parallel to `fm_pop_use_tail`); set by graph_uncoarsen at the finest level under `SPARSE_FM_FINEST_STRATEGY=annealing`; restored after.
2. Add `_Thread_local fm_anneal_schedule_t fm_anneal_schedule` (default `EXPONENTIAL`); set from `SPARSE_FM_ANNEALING_SCHEDULE` env var.
3. Add `parse_fm_anneal_schedule()` parsing `{linear, exponential, cosine}` with default exponential.
4. **No graph_refine_fm changes.**  Annealing thread-locals are set but not read; Day 5's dispatch is no-op.  Default-off path stays bit-identical to current master.
5. Stub `tests/test_graph.c::test_finest_fm_annealing_accepts_worsening` — pins the differs-from-baseline contract on a 30×30 grid.  RUN_TEST commented out for Day 5 (test fails today; Day 6 lights it up).

### Day 6 — Implementation

1. Add `graph_refine_fm_annealing` entry point — clones the graph_refine_fm pop-eval-accept loop with annealing-acceptance overlay.  Or: add an inline branch in graph_refine_fm reading `fm_use_annealing` (smaller diff, easier to reason about).  Pick the latter for surgical change.
2. Track per-pass max |gain| seen for T_0 selection.
3. For each pop, if gain < 0 AND fm_use_annealing AND T > 1: roll a per-call deterministic random number ∈ [0, 1); accept iff random < `exp(gain / T)`.
4. Implement all three temperature schedules behind `fm_anneal_schedule`.
5. Emit per-pass annealing stats under `SPARSE_FM_ANNEALING_DEBUG=1` (worsening-move count + acceptance rate + T_k).
6. Light up `test_finest_fm_annealing_accepts_worsening`: assert worsening-move count > 0 (or partitions differ).
7. Capture interim Pres_Poisson ND nnz_L under `SPARSE_FM_FINEST_STRATEGY=annealing` × 3 schedules.

### Day 7 — Sweep + Decision

1. Full corpus sweep under `annealing` × 3 schedules (18 measurements).
2. Apply flip rule for `SPARSE_FM_FINEST_STRATEGY` default: flip to `annealing` (best schedule) if (a) Pres_Poisson lands ≤ 0.85× of AMD nnz_L AND (b) no smaller-fixture regress past 5pp.
3. Document in `annealing_fm_decision.md`.
4. If miss: kick off Item 5 (root-level spectral) design Days 7-9.

## Annealing's Hypothesised Pres_Poisson Win

Pres_Poisson under Sprint 27 Day 3 default lands at ND/AMD = 0.923× (2 462 201 / 2 668 793).  Sprint 25 Day 5 measured the FM saturation point at 0.952× (passes ≥ 5 don't move the needle past 4.8pp away from the 0.85× target).  Annealing's value proposition: escape the local minimum that 5+ passes converge to.

If annealing's worsening-acceptance lets the FM walk explore cuts that baseline-greedy can't reach, the resulting cut landscape includes lower-cut points the saved best-cut tracking can capture.  The hypothesis is that Pres_Poisson's regular FE-mesh structure has many near-optimal local minima within the multilevel pipeline's reach, and annealing's stochastic acceptance lets the FM cross between them.

Quantitative target: Pres_Poisson nnz_L ratio ≤ 0.85× = 2 268 474.  Current default 2 462 201 → need −7.9 % to hit target.

If annealing doesn't get there alone, the combination with item 5 (root-level spectral) — which targets a different stage of the pipeline (root vs finest) — may close the remainder.  If both fall short, item 6 (thick-restart) fires in Days 10-12.

## Files Generated Day 5

- `docs/planning/EPIC_2/SPRINT_27/annealing_fm_design.md` — this document
- `src/sparse_graph.c` — `_Thread_local int fm_use_annealing`; `fm_anneal_schedule_t` enum + `_Thread_local fm_anneal_schedule`; `parse_fm_anneal_schedule()`; dispatch wiring in `graph_uncoarsen` (sets thread-locals at finest level under `SPARSE_FM_FINEST_STRATEGY=annealing`; restored after)
- `tests/test_graph.c` — `test_finest_fm_annealing_accepts_worsening` stub (RUN_TEST commented out for Day 5; Day 6 enables)

## Headline Status After Day 5

- Annealing dispatch skeleton lands; default behaviour bit-identical to current master.
- `SPARSE_FM_ANNEALING_SCHEDULE` env var stub recognised (3 values; default exponential).
- Pres_Poisson ratio unchanged (Day 5 is design + skeleton; Day 6 implementation moves the needle).
- Quality gates clean (format, lint, test, wall-check).
