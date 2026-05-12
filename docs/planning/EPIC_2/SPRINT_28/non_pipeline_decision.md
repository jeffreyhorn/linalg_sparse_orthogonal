# Supernodal-Etree Reordering — Flip-or-Stay Decision (Sprint 28 Day 10)

## Background

Sprint 28 Day 1 (`pivot_decision_day1.md`) picked **(c) supernodal-etree reordering** as Item 4's non-pipeline-level pivot, rejecting (a) METIS-style multi-matching coarsening (still pipeline-level) and (b) geometric domain decomposition (structurally blocked on Pres_Poisson — corpus lacks coordinates).  The (c) approach lands a Liu 1990 / Davis 2006 §6.5 post-pass that composes the elimination-tree postorder into `analysis->perm`, then rebuilds B + recomputes etree+postorder so colcount + symbolic Cholesky run on the final ordering.

Sprint 28 PLAN.md Day 10 task 1 specifies the flip-rule outcome classes:

> - **Closed:** Pres_Poisson ≤ 0.85× of AMD; flip the env-var default to on...
> - **Partial:** Pres_Poisson moved measurably (≥ 1pp better than Sprint 27 default 0.923×) but didn't reach 0.85×; advisory only...
> - **Missed:** Pres_Poisson didn't move (≤ 1pp delta from Sprint 27 default); advisory only or removed depending on corpus safety...

Day 6 landed the env-var scaffolding (`SPARSE_ND_SUPERNODAL_POSTORDER={off, on}`) + failing-as-expected test stub.  Day 7 lit up the core composition algorithm in `src/sparse_analysis.c::apply_supernodal_postorder`.  Day 8 added corpus-safety contracts (residual unchanged, nnz_L invariant, deterministic, n=1 edge case, supernode-count corpus safety).  Day 9 ran the 24-cell sweep ({AMD, ND} × {env off, env on} × 6 fixtures).  This document records the Day-10 flip-rule application + decision.

## Reproducer

```
# Default-off path (Sprint 27 behaviour bit-identical):
build/bench_reorder --only Pres_Poisson --skip-factor

# Env-var-on path requires sparse_analyze with reorder enum
# (bench_reorder's pre-applied-perm + REORDER_NONE path doesn't fire
# the dispatch — analysis->perm is NULL).  See /tmp/bench_day9.c
# pattern used to capture the Day-9 sweep.
SPARSE_ND_SUPERNODAL_POSTORDER=on <call sparse_analyze with reorder=AMD or ND>
```

Library state at Day 10: HCC + Kuu-safe coarsening default (Sprint 27 Day 2); `nd_base_threshold = 128` (Sprint 27 Day 3); `SPARSE_FM_FINEST_STRATEGY=baseline` default (Sprint 27 + Sprint 28 Days 2-5 advisory only); `SPARSE_ND_SUPERNODAL_POSTORDER` default `off` (this decision).

## Sprint 28 Day 9 Sweep Summary

(Full per-cell capture in `non_pipeline_sweep.txt`; condensed here.)

### Pres_Poisson (headline fixture, n = 14 822)

| Reorder | env | nnz_L | ratio vs AMD | super_count | total_grouped | analyze_ms |
|---|---|---:|---:|---:|---:|---:|
| AMD | off | 2 668 793 | 1.0000 | 565 | 11 360 | 4 472 |
| AMD | on  | 2 668 793 | 1.0000 | 566 | 11 360 | 4 768 (+6.6%) |
| ND  | off | 2 462 201 | 0.9226 | 571 | 10 753 | 7 078 |
| ND  | on  | 2 462 201 | **0.9226** | 573 | 10 756 | 6 725 (-5%, noise) |

### Per-fixture nnz_L delta env-on vs env-off (across 12 cells: 6 fixtures × 2 reorders)

**0 delta on every cell.**  Symmetric permutation preserves symbolic Cholesky fill by construction; the supernodal-etree post-pass reorders columns within the existing fill pattern but cannot eliminate fill.

### Per-fixture supernode-count delta env-on vs env-off (min_size = 4)

| Fixture | AMD off / on (super_count, total_grouped) | ND off / on (super_count, total_grouped) |
|---|---|---|
| nos4 | (1, 9) / (1, 6) | (1, 9) / (1, 6) |
| bcsstk04 | (10, 51) / (8, 39) | (7, 75) / (7, 73) |
| Kuu | (534, 3748) / (535, 3748) | (477, 5782) / (474, 5786) |
| bcsstk14 | (107, 1246) / (107, 1246) | (91, 838) / (94, 837) |
| s3rmt3m3 | (522, 4960) / (523, 4960) | (537, 5049) / (535, 5050) |
| **Pres_Poisson** | (565, 11 360) / (566, 11 360) | (571, 10 753) / (573, 10 756) |

Pattern: on fixtures with n > 1000, super_count differs by ±3 and total_grouped is identical or differs by ≤ 4 columns.  On the smaller fixtures (nos4, bcsstk04) total_grouped slightly DECREASES under env-on — AMD's perm produces more compact contiguous supernodes than the strict etree-postorder on small graphs.

## Flip-Rule Application

Flip rule (from Sprint 28 PLAN.md Day 10 task 1):

| Outcome | Trigger | Sprint 28 measurement |
|---|---|---|
| **Closed** | Pres_Poisson ratio ≤ 0.85× | 0.9226× ✗ |
| **Partial** | Pres_Poisson delta ≥ 1pp better than Sprint 27 default 0.923× | 0.0pp delta ✗ |
| **Missed** | Pres_Poisson delta ≤ 1pp from Sprint 27 default | **0.0pp delta ✓** |

**Verdict: MISSED.**  Pres_Poisson ND under env-on lands at exactly 0.9226× of AMD — bit-identical to env-off, identical to Sprint 27 Day 13's measurement.  No movement on the headline metric.

## Decision: Stay At Default `off`; Ship Supernodal-Etree Post-Pass As Advisory Infrastructure

The supernodal-etree post-pass is structurally correct (Day-7 + Day-8 tests pin: residual ≤ 1e-15 on bcsstk04, nnz_L invariant across the corpus, deterministic, no corpus-safety regression).  But it does **not move the fill-reduction metric** because symmetric permutation preserves the symbolic Cholesky fill pattern by construction — the algorithm reorders columns within the fill, it doesn't eliminate fill.

### Why this isn't surprising in retrospect

Sprint 28 Day 1's pivot_decision_day1.md candidate-(c) dossier explicitly flagged this:

> "Pres_Poisson upside per literature: Liu 1990 + Davis 2006 §6.5 cite ≤ 5% nnz reduction from etree postorder.  Supernodal-etree reordering's PRIMARY value is numeric-factor performance (dense-block kernels' efficiency on contiguous supernodes via batched BLAS calls), NOT fill-reduction.  **Cannot close 7.3pp gap on Pres_Poisson alone.**"

The empirical measurement matches the literature prior tighter than predicted: 0.0pp nnz_L delta on the corpus, well below the cited ≤ 5% literature ceiling.  The headline-target gap of 7.3pp was always ~1.5× the literature-cited maximum — Day 9's measurement just confirms that the achievable upside on this codebase + corpus is at the low end of the literature range, not the high end.

### Where supernodal-etree would help

The post-pass ships infrastructure that becomes valuable when the numeric-factor path exploits supernodal kernels.  The existing `src/sparse_chol_csc.c::chol_csc_eliminate_supernodal` detects supernodes but still runs the scalar Day-5 elimination kernel — the batched supernodal cmod + dense factor + panel solve is a future-sprint deliverable (per `src/sparse_chol_csc_internal.h` line 614-628).  When that future sprint lands, the env-on path becomes the natural input ordering for the supernodal kernels: contiguous supernodes from etree-postorder give the dense BLAS kernels their cache-friendly column blocks.

Today's value of `SPARSE_ND_SUPERNODAL_POSTORDER=on`: zero on this codebase.  Tomorrow's value: directly enables supernodal-kernel speedups if/when Sprint 29+ wires them.

### Production-default rationale

| Criterion | Outcome |
|---|---|
| Pres_Poisson nnz_L improvement | 0.0 % (invariant) |
| Corpus-wide nnz_L delta | 0.0 % (invariant by construction) |
| Supernode-count delta | ±1-3 supernodes; trivial (Day-8 measurement) |
| Analyze wall cost | +6-15 % on n > 1000 fixtures |
| Numeric-factor wall benefit | 0 % (current chol_csc path is scalar) |
| Corpus safety | Clean (nnz_L invariant on all 12 cells) |

**Verdict: default stays OFF.**  The +6-15 % analyze wall cost has no offsetting benefit today.  The env-on path remains as opt-in advisory infrastructure for future sprints.

## Per-Fixture-Class Advisory

| Fixture class | Supernodal-postorder verdict | Default behaviour |
|---|---|---|
| Tiny (n ≤ 200; nos4, bcsstk04) | Bit-stable nnz_L; total_grouped slight decrease | Stay default |
| Mid (Kuu, s3rmt3m3) | Bit-stable nnz_L; ±1 supernode | Stay default |
| Large irregular (bcsstk14) | Bit-stable nnz_L; supernode structure exactly equal | Stay default |
| **Headline (Pres_Poisson)** | **Bit-stable nnz_L; ±2 supernode** | **Stay default** |
| Future supernodal-kernel workloads | Opt-in via `SPARSE_ND_SUPERNODAL_POSTORDER=on` | Document advisory |

## Literal 0.85× Target — Formal Retirement

After 6 consecutive sprints (Sprints 23-28 inclusive; Sprint 22's 1.063× pre-dated the ND-beats-AMD framing of the target) + ~200 pipeline-level measurements + Sprint 27 Day-13's 24-setting matrix + Sprint 28 Day-9's 24-cell non-pipeline sweep:

| Sprint | Pres_Poisson ND/AMD ratio (default) | Sprint hypothesis | Outcome |
|---|---:|---|---|
| 22 | 1.063× | Multilevel ND beats AMD | Miss (regress) |
| 23 | 0.952× | Leaf-AMD splice closes gap | Miss |
| 24 | 0.942× (best opt-in) | qg-AMD tuning + threshold sweep | Miss |
| 25 | 0.922× (best opt-in) | Spectral coarsest + per-vertex lift | Miss |
| 26 | 0.9217× (best opt-in) | HCC matching | Miss |
| 27 | 0.9226× (default) | HCC default flip + FINEST FM axes | Miss; advisory-only verdict |
| **28** | **0.9226× (default)** | **Non-pipeline supernodal-etree post-pass** | **Miss; nnz_L invariant by construction** |

The literal 0.85× target on Pres_Poisson is **formally retired** with Sprint 28's empirical evidence.  Future sprints may revisit only if fundamentally different machinery becomes available:

- **METIS C library interop** (vs the in-house multilevel pipeline)
- **Geometric mesh-aware ordering with first-class coordinate API** (vs synthesising coordinates from the Laplacian spectrum, which Sprint 27 Day 9 already rejected at +2.3pp regress)
- **Hybrid AMD-then-ND-on-separators** (Sprint 27 evidence shows AMD already finds near-optimal cuts on Pres_Poisson; ND adds marginal value only at the separator levels)

None of these are in the Sprint 29 budget; routed to long-term parking-lot.

## Items Deferred (route to Sprint 29+)

1. **Supernodal numeric-factor kernels** — the natural follow-up that gives the Sprint-28 env-on path measurable production value.  Day-1 dossier estimated 5-15 % numeric-factor wall reduction on supernodal-heavy fixtures.  Out of Sprint 28 scope (Item 4's 60 hrs was for the post-pass infrastructure, not the kernels).

2. **Pres_Poisson literal 0.85× target** — formally retired per the table above.  Sprint 29+ revisit only with a fundamentally different approach.

3. **Bench_reorder env-var integration** — the Day-7 measurement noted that `bench_reorder.c` pre-applies the perm and calls `sparse_analyze` with `REORDER_NONE`, which doesn't fire the supernodal-postorder dispatch.  Adding a `--reorder-via-analyze` flag would let the standard bench harness exercise the env var.  Low priority; the Day-9 ad-hoc helper is sufficient for the current verdict.

## Test Bound Calibration

`tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` Sprint 27 bound: **0.94×**.  Sprint 28 Day 9 measurement: 0.9226× under both env settings.

**Bound stays at 0.94×.**  Sprint 27's 2pp safety margin above the 0.923× default is preserved; the literal 0.85× target retirement is documented in this decision doc and will land in `headline_summary.md` + `docs/algorithm.md` per the Day 13-14 plan.

## Day 10 Test Scaffolding

`tests/test_reorder_nd.c::test_non_pipeline_pres_poisson_close_to_target` lands as a failing-as-expected stub with `RUN_TEST` commented out — mirrors the Sprint 27 Day-12 pattern for `test_finest_fm_annealing_pres_poisson_close_to_target` + `test_nd_root_spectral_pres_poisson_close_to_target` (both still failing-as-expected after Sprint 27 closed).

Contract: under `SPARSE_SUPERNODAL_POSTORDER=on` (canonical name post-PR-#36 review; legacy alias `SPARSE_ND_SUPERNODAL_POSTORDER` still accepted), Pres_Poisson ND nnz_L ≤ 0.87× of AMD (= 0.85× target + 2pp tolerance).  Sprint 28 measurement: 0.9226× → ratio gap +7.26pp, fails-as-expected.  Comment block cites the Sprint 28 retirement of the literal target after **6 consecutive sprints** of misses (Sprints 23-28 inclusive); documents that uncommenting the RUN_TEST requires a fundamentally different machinery per Sprint 29+ routing.

## Files Generated

- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_sweep.txt` — Day 9 24-cell sweep capture
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day9.md` — Day 9 decision direction
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_supernode_count_day8.txt` — Day 8 supernode-count corpus measurement
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_interim_day7.txt` — Day 7 nnz_L invariance + analyze wall measurement
- `docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md` — this document
- `docs/planning/EPIC_2/SPRINT_28/pivot_decision_day1.md` — Day 1 pivot selection (candidate-(c) dossier)

## Files NOT Modified

- `src/sparse_analysis.c::parse_supernodal_postorder()` — default stays `SUPERNODAL_POSTORDER_OFF` (no flip).  Sprint-28 Day-10 originally named these `parse_nd_supernodal_postorder()` / `ND_SUPERNODAL_POSTORDER_OFF`; renamed per PR #36 review (the post-pass is reorder-agnostic so the `ND_` prefix was misleading).
- `tests/test_reorder_nd.c::test_nd_pres_poisson_fill_with_leaf_amd` — bound stays `≤ 0.94×` (Sprint 27 ratio + 2pp; no tightening — Sprint 28 didn't close the literal 0.85× target)

## Headline Status After Day 10 Item-4 Close

- **Pres_Poisson default unchanged: 0.9226× of AMD** (env-on path bit-identical to default by symmetric-permutation invariance).
- **No production default flip.**  `SPARSE_SUPERNODAL_POSTORDER` ships as opt-in advisory infrastructure (canonical name post-PR-#36 review; legacy alias `SPARSE_ND_SUPERNODAL_POSTORDER` still accepted for back-compat with Sprint 28 captures + advisory recipes that shipped under the original name).
- **Literal 0.85× target formally retired** with 6-consecutive-sprint (Sprints 23-28 inclusive) + non-pipeline-pivot empirical evidence.
- Sprint 29+ routing for supernodal numeric-factor kernels (the natural follow-up that gives the Sprint-28 infrastructure measurable production value).
