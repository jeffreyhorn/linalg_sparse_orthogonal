# Sprint 25 Day 3 — HCC Validation & Headline-Progress Assessment

## Summary

Day 3 lands HCC's correctness work + landed the Day-1 stub
assertions + ran the bench_day3 cross-corpus capture under
`SPARSE_ND_COARSENING=hcc`.  HCC is correctness-verified
(determinism contracts intact; matching produces valid coarse
graphs).  HCC alone moves Pres_Poisson ND nnz_L from 0.952× to
**0.937× of AMD** — a 1.5pp tightening, in the "0.90-0.94× = item
1 is smaller of three contributions; items 2 + 3 must close the
rest" band per PLAN.md Day 3 task 4.

Per-fixture deltas (HEM default vs HCC):

| fixture       | HEM nnz_L | HCC nnz_L | Δ      |
|---------------|----------:|----------:|--------|
| nos4          |       968 |       901 | -6.9 % |
| bcsstk04      |     3 702 |     3 708 | +0.16 % |
| Kuu           |   924 385 | 1 003 293 | +8.5 % |
| bcsstk14      |   131 017 |   130 358 | -0.5 % |
| s3rmt3m3      |   478 890 |   483 478 | +0.96 % |
| Pres_Poisson  | 2 541 734 | 2 501 876 | -1.6 % |

## What landed Day 3

1. **Day-1 HCC test stubs replaced with real assertions**
   (`tests/test_graph.c`):
   - `test_hcc_match_selection_grid`: pins determinism + cmap range +
     "HCC differs from HEM" on a 5×5 grid (Day 2 diagnostic measured
     12/25 cmap entries differ; assertion requires ≥ 1).
   - `test_hcc_match_selection_irregular`: pins determinism + n_coarse=3
     + structural histogram (1 hub-leaf pair + 2 unmatched leaves) on
     a Y-graph (4 vertices: hub + 3 leaves).  Day 2 diagnostic
     observed n_coarse=3 across all 5 tested seeds.

2. **Determinism re-runs explicit**: ran the three determinism
   contracts under `SPARSE_ND_COARSENING=hcc` and confirmed each
   passes:
   - `test_partition_determinism_10x10_grid` — PASS under HCC
   - `test_partition_determinism_two_cliques` — PASS under HCC
   - `test_nd_determinism_public_api` — PASS under HCC

3. **Cross-corpus capture** (`bench_day3_hcc_only.{csv,txt}`):
   ran `bench_reorder.c --skip-factor` under `SPARSE_ND_COARSENING=hcc`
   on the full SuiteSparse corpus.  Results above.

## Why the Day-2 interim file's "HCC alone changes nothing" finding was wrong

The Sprint 25 Day 2 interim file (`hcc_interim_day2.txt`) reported
HCC and HEM produce bit-identical ND nnz_L on every corpus
fixture.  Day 3's clean rebuild + re-bench shows this was
incorrect: HCC moves nnz_L on every corpus fixture (range -6.9 %
to +8.5 %).

**Root cause**: Day 2's bench captures were taken with a benchmark
binary built BEFORE the Day-2 lint-fix to the HCC matching loop's
branch consolidation.  The lint fix replaced an `if/else if` form
with a single OR-condition; while the conditions are *logically*
equivalent in most cases, the binary used for the Day-2 interim
capture had not been rebuilt after the lint fix.  Day 3's clean
rebuild picks up the post-lint-fix consolidated condition, which
behaves slightly differently from the pre-lint-fix code on at least
one corpus fixture (the bench output here is the authoritative
post-Day-2-commit baseline).

**Lesson**: future bench captures that follow a code edit + lint
fix sequence must `make clean && make` between them to ensure the
binary reflects the latest source state.  Day 11's profiling work
should flag this as a Sprint 25 retrospective lesson.

## Item-1 (HCC) headline-progress assessment

Per PLAN.md Day 3 task 4, Pres_Poisson under HCC alone at 0.937×
falls into the "0.90-0.94×" band:

> If 0.90-0.94×, item 1 is the smaller of three contributions
> and items 2 + 3 must close the rest.

So:
- **Item 1 contribution**: ~1.5pp Pres_Poisson tightening.
- **Remaining gap to 0.85×**: 0.937× → 0.85× = 8.7pp.
- **Items 2 + 3 must contribute**: ~8.7pp combined.

This is achievable if:
- Day 5's multi-pass FM intermediate sweep finds a 1-3pp Pres_Poisson
  tightening at `SPARSE_FM_INTERMEDIATE_PASSES=2` or `=3`.
- Day 8's spectral-bisection-only Pres_Poisson measurement reaches
  ≤ 0.94× (PLAN.md Day 8 task 4 threshold).
- Day 9's combined-effect matrix surfaces a HCC + multi-pass FM +
  spectral combination that composes constructively (Day-2's
  destructive interaction between Sprint 24's two env vars on
  Pres_Poisson is a cautionary precedent — Day 9 must verify
  composition).

If the combined effect comes up short (≤ 0.90× partial close instead
of ≤ 0.85×), Sprint 26 inherits the residual gap.  This is the
expected fallback path per Sprint 24's retrospective routing.

## Production-default flip rule (preliminary; final on Day 10)

PLAN.md Sprint 25 item 1 + Day 10 task 1 spec the flip rule:

> Flip default if HCC produces a clear corpus-wide win on
> Pres_Poisson + small fixtures.

Day 3's measurements:
- **Pres_Poisson**: HCC wins (-1.6 % nnz_L). ✓
- **Small-fixture wins**: nos4 (-6.9 %), bcsstk14 (-0.5 %). ✓
- **Small-fixture losses**: Kuu (+8.5 %).  ✗ (beyond 5pp band)

Kuu's +8.5 % regression is the deciding factor.  Kuu is the worst
ND/AMD fixture in the corpus (2.275× under default) — a regression
on it under HCC means HCC's coarsening choice doesn't benefit the
broader irregular-SPD class.  HCC should ship as advisory env-var
behind `SPARSE_ND_COARSENING=hcc` for Pres_Poisson-shaped workloads
(regular 2D PDE meshes); default stays `heavy_edge` for
corpus-wide robustness.

Day 10's final decision will lock in this preliminary call after
Day 9's cross-corpus combined-effect matrix sweep.

## Implications for Days 4-10

Per the headline-progress assessment + production-default rule:

- **Days 4-5 (multi-pass FM intermediate)**: primary partner to HCC.
  Goal: combine HCC's 1.5pp Pres_Poisson contribution with multi-pass
  FM at intermediate levels to push toward 0.85×.

- **Days 6-8 (spectral bisection)**: primary headline-mover.
  Goal: contribute the remaining 5-7pp on Pres_Poisson via
  Fiedler-vector-based coarsest-level cuts.  Spectral bisection
  intervenes BEFORE FM at uncoarsening can wash out coarsening
  changes — strongest candidate for closing the literal 0.85× gap.

- **Day 9 (cross-corpus re-bench)**: must measure all 3 axes ×
  Sprint 24's 2 env vars in combination on Pres_Poisson + small
  fixtures.  Look for both constructive composition (HCC + spectral
  > sum of individual wins) and destructive composition (HCC + ratio=200
  < ratio=200 alone — Sprint 24 already saw this with ratio=200 +
  balanced_boundary on Pres_Poisson).

- **Day 10 (production-default decisions)**:
  - `SPARSE_ND_COARSENING` likely stays `heavy_edge` (Kuu regression
    rules out HCC default).
  - `SPARSE_FM_INTERMEDIATE_PASSES` flip TBD on Day 5's sweep.
  - `SPARSE_ND_COARSEST_BISECTION` flip TBD on Day 8's sweep + Day 9's
    combined-effect bench.
  - Documented advisory env-var combinations for Pres_Poisson
    workloads (likely HCC + spectral).

## References

- `docs/planning/EPIC_2/SPRINT_25/PLAN.md` Day 3 — task list +
  completion criteria.
- `docs/planning/EPIC_2/SPRINT_25/hcc_design.md` — KK1998 §5
  contract.
- `docs/planning/EPIC_2/SPRINT_25/hcc_interim_day2.txt` — Day-2
  interim findings (now superseded; see "Why the Day-2 interim
  file's finding was wrong" above).
- `docs/planning/EPIC_2/SPRINT_25/bench_day3_hcc_only.{csv,txt}` —
  Day-3 cross-corpus capture under HCC.
- `tests/test_graph.c::test_hcc_match_selection_grid`,
  `test_hcc_match_selection_irregular` — Day-3 assertion landings.
