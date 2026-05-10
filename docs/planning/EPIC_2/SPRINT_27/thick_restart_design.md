# Thick-Restart FM at the Finest Level — Design (Sprint 27 Day 10)

## Background

Sprint 26 Day 6's `finest_fm_design.md` rejected thick-restart on cost grounds — estimated 2-3× wall expansion (each pass re-runs FM from scratch from a saved anchor with random perturbation).  Under the Sprint 25 baseline (Pres_Poisson ND wall = 38.1 s), thick-restart would push to 76-114 s, breaching the 1.5× wall-check ceiling of ~57 s.

Sprint 27 Days 2 + 3 cumulatively dropped Pres_Poisson ND wall to 7.1 s (-81 % vs Sprint 25 baseline).  Thick-restart's projected 2-3× expansion now lands at 14-21 s on Pres_Poisson — well under the 70.5 s wall-check 1.5× ceiling.  **Affordable.**

This document captures Day 10's design + perturbation sub-axis selection + Days 11-12 implementation/decision plan.

## Why Sprint 27 Day 10 (Conditional Fallback) Fires

Sprint 27 Day 9 verdict (`root_spectral_decision.md`):
- Item 4 (annealing FM): **MISSED** Pres_Poisson 0.85× target (regress 2.2-3.1pp under all 3 schedules).
- Item 5 (root-spectral): **MISSED** Pres_Poisson 0.85× target (regress 2.3pp).
- Items 4 + 5 combined: **MISSED** (lands 0.947× = +2.4pp).

Per Sprint 27 PLAN.md Day 10 task 1: "If items 4-5 missed: proceed with tasks 2-7."  Item 6 (thick-restart) is now Sprint 27's **last 0.85× candidate**.  If item 6 also misses, the sprint closes with the literal target unmet — fifth consecutive sprint (Sprints 23, 24, 25, 26, 27).

## Thick-Restart Algorithm Contract

**Standard formulation** (Karypis-Kumar 1998 §6 references; tracked across multilevel-partition literature):

```
best_part ← initial 2-way partition (after coarsening + bisection + uncoarsening)
best_cut  ← compute_cut_weight(G, best_part)

for pass in 1..K:
  if pass > 1:
    perturb best_part → cur_part   # randomise some vertices' assignments
  else:
    cur_part ← best_part            # first pass: baseline FM run

  cur_part ← graph_refine_fm(G, cur_part)   # standard FM walk (with rollback-to-best-cut floor)
  cur_cut ← compute_cut_weight(G, cur_part)

  if cur_cut < best_cut:
    best_part ← cur_part
    best_cut ← cur_cut

return best_part
```

Key contract: `best_part` and `best_cut` are tracked GLOBALLY across all passes (vs Sprint 23 Day 11's per-pass rollback-to-best-cut, which floors each pass at its OWN best).  Each pass restarts from the GLOBAL anchor with a perturbation, exploring a different region of the cut landscape.

**Hypothesis**: baseline FM converges to a SUBOPTIMAL local minimum on Pres_Poisson — different from annealing's hypothesis (which assumed baseline reached the global minimum and annealing's stochasticity disrupted that).  Thick-restart's mechanism is fundamentally different: it doesn't trust the baseline trajectory, it actively re-explores from the saved anchor.

## Perturbation Sub-Axis Selection

Sprint 27 PLAN.md Day 10 task 4 names three plausible variants.

### (a) Random-vertex flip (k vertices)

Flip k randomly-selected vertices' partition assignments (0 → 1 or 1 → 0) before each pass except the first.  k = 1 % of n (rounded up; minimum 1).

- **Pros**: simplest formulation; cheapest per-pass cost (O(k) flips); deterministic with a seeded RNG; doesn't require boundary-vertex computation.
- **Cons**: random flips may break far from the cut (interior-interior swap is wasteful); may take more passes to converge.

### (b) Boundary-vertex shuffle

Among the boundary vertices (vertices with at least one cross-edge to the other side), shuffle their partition assignments uniformly.  k_eff = boundary_count.

- **Pros**: structurally targeted (only perturbs vertices that matter for the cut); larger per-pass deviation than random-flip.
- **Cons**: requires per-pass boundary identification (one O(|E|) pass); biases toward already-cut-influencing vertices, possibly missing global-restructuring opportunities.

### (c) Gaussian noise on gain estimates

Perturb the initial-pass gain values with N(0, σ) noise; σ = max gain / 4.  FM's bucket structure then operates on noisy gains, picking different vertices in different orders.

- **Pros**: smoothest perturbation (continuous rather than discrete state changes); the perturbed gains drive natural FM exploration.
- **Cons**: most expensive (one Gaussian sample per vertex per pass; needs Box-Muller or similar); noise injection point isn't the partition state directly so the trajectory deviates more subtly.

### Decision: Default Random-Vertex Flip (Variant a)

Reasons:

1. **Simplest implementation** for a Day-11 budget.  Days 5-7's annealing was the more invasive change (per-pop acceptance overlay); thick-restart's perturbation can be a small pre-pass loop.
2. **Cheapest per-pass cost.**  Boundary-shuffle adds O(|E|) per pass; gauss-noise adds O(n) Box-Muller samples.  Random-flip is O(k) = O(n/100).
3. **Determinism via per-pass-seeded RNG.**  Same approach as Days 5-6's annealing xorshift32.
4. **Day 11 implementation can sweep all three** under `SPARSE_FM_THICK_RESTART_PERTURB={random_flip (default), boundary_shuffle, gauss_noise}` to validate the choice empirically.  Day 10 wires the parser; Day 11 implements all three branches.

### Sweep Dimensions for Day 11

- **Perturbation**: random_flip (default) / boundary_shuffle / gauss_noise.
- **Perturbation magnitude**: default 1 % × n flipped (k_default).  Could try 0.5 %, 2 %, 5 % to see whether more aggressive perturbation helps escape the baseline minimum.
- **Pass count**: inherits `SPARSE_FM_FINEST_PASSES` (default 3).  Thick-restart needs more passes to amortise the per-pass restart cost; Day 11 may bump default to 5 if measurements support.
- **Best-cut-anchor scope**: this design uses GLOBAL anchor (across all passes).  Sprint 23 Day 11's existing rollback uses LOCAL (per-pass).  Day 11 implementation maintains both — global-anchor for thick-restart, per-pass for baseline FM.

## Hypothesised Pres_Poisson Outcome

Pres_Poisson under Sprint 27 Day 3 default lands at **0.923×** of AMD.  Sprint 25 Day 5 measured FM saturation at 0.952× (passes ≥ 5 don't improve).  Annealing's hypothesis (escape the local minimum at 0.952× to find the global) failed empirically — annealing produced WORSE cuts.

Thick-restart's hypothesis: baseline FM is reaching a near-global cut on Pres_Poisson (0.923× under default), but there might exist near-equivalent cuts in the local neighborhood that lead to better downstream recursive partitions.  Thick-restart's perturbation samples those neighborhoods.  Probability of closing the 7.3pp gap to 0.85×: **modest**.  The mechanism is plausible but Pres_Poisson's regular FE-mesh structure has produced unusually flat cut landscapes in our previous explorations (annealing + spectral both regressed).

If thick-restart also misses, Sprint 27 closes with the literal target unmet (fifth consecutive sprint); Sprint 28+ pivots to non-pipeline-level interventions (e.g. domain-decomposition recursion at the geometric level, METIS-style coarsening with multiple matchings per level).

## Implementation Plan

### Day 10 (this commit) — Skeleton

1. Add `_Thread_local int fm_use_thick_restart` (parallel to `fm_use_annealing`); set by graph_uncoarsen at the finest level under `SPARSE_FM_FINEST_STRATEGY=thick_restart`; restored after.
2. Add `_Thread_local fm_thick_restart_perturb_t fm_thick_restart_perturb` (default `RANDOM_FLIP`); set from `SPARSE_FM_THICK_RESTART_PERTURB` env var.
3. Add `parse_fm_thick_restart_perturb()` parsing `{random_flip, boundary_shuffle, gauss_noise}` with default random_flip.
4. **No graph_refine_fm changes.**  Thick-restart thread-locals are set but not read; Day 10's dispatch is no-op.  Default-off path stays bit-identical to Sprint 27 Day 9.
5. Stub `tests/test_reorder_nd.c::test_finest_fm_thick_restart_returns_to_anchor` — pins the differs-from-baseline contract on bcsstk14 (same fixture pattern as Day-6 annealing test).  RUN_TEST commented out for Day 10 (test fails today; Day 11 lights it up).

### Day 11 — Implementation

1. **Add a "global anchor" mode to graph_refine_fm**: when `fm_use_thick_restart == 1`, the function reads/writes a thread-local `best_part_anchor[]` + `best_cut_anchor` that persist across calls (cleared by graph_uncoarsen on entry to the finest-level loop).
2. **Per-pass perturbation in graph_uncoarsen** (not graph_refine_fm): before each pass except the first, copy anchor → working partition + apply perturbation per the chosen `fm_thick_restart_perturb` mode.
3. **All three perturbation branches**:
   - `RANDOM_FLIP`: pick k random vertex IDs via xorshift32; flip their part assignments.
   - `BOUNDARY_SHUFFLE`: identify boundary vertices (one O(|E|) pass); flip their part assignments uniformly.
   - `GAUSS_NOISE`: defer to the gain-init point inside graph_refine_fm; perturb gain[v] += N(0, σ) once per pass.
4. **End-of-passes reduction**: after the last pass, the anchor holds the global-best partition; copy it to part_io.
5. Emit per-pass anchor stats under `SPARSE_FM_THICK_RESTART_DEBUG=1` (cur_cut vs best_cut, perturbation count).
6. Light up `test_finest_fm_thick_restart_returns_to_anchor`.

### Day 12 — Sweep + Decision

1. Full corpus sweep under `thick_restart` × 3 perturbation variants.
2. Apply flip rule for `SPARSE_FM_FINEST_STRATEGY` default: flip to `thick_restart` (best perturbation) if (a) Pres_Poisson lands ≤ 0.85× of AMD AND (b) no smaller-fixture regress past 5pp.
3. Document in `thick_restart_decision.md`.
4. **Likely outcome**: STAY at default; thick-restart ships as advisory.  Sprint 27 closes with literal 0.85× target unmet.

## Files Generated Day 10

- `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md` — this document
- `src/sparse_graph.c` — `_Thread_local int fm_use_thick_restart`; `fm_thick_restart_perturb_t` enum + thread-local; `parse_fm_thick_restart_perturb()`; dispatch wiring in `graph_uncoarsen` (sets thread-locals at finest level under `SPARSE_FM_FINEST_STRATEGY=thick_restart`; restored after)
- `tests/test_reorder_nd.c` — `test_finest_fm_thick_restart_returns_to_anchor` stub (RUN_TEST commented out for Day 10; Day 11 enables)

## Headline Status After Day 10

- Thick-restart dispatch skeleton lands; default behaviour bit-identical to Sprint 27 Day 9 (4 fixtures unchanged).
- `SPARSE_FM_THICK_RESTART_PERTURB` env var stub recognised (3 values; default random_flip).
- Pres_Poisson default unchanged (Day 10 is design + skeleton; Day 11 implementation moves the needle).
- Quality gates clean (format, lint, test, wall-check).
