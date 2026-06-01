# Sprint 51 Retrospective

**Sprint:** 51 — Public Direct-Solver Lifecycle API Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 51 baseline and scope captured from the Sprint 50 contract
- [x] reviewed validation/truthfulness baseline rechecked before code landing
- [x] public direct header surface mapped from design to implementation
- [x] first public direct lifecycle header batch landed
- [x] LU lifecycle integration landed through the bounded shared path where the default option surface fit cleanly
- [x] Cholesky lifecycle integration landed through the shared analysis/factor path
- [x] LDL^T lifecycle integration landed through the shared analysis/factor path
- [x] one-shot wrapper preservation batch completed for the touched direct families
- [x] focused regression expansion completed for the public lifecycle contract
- [x] example and benchmark adoption/docs batch completed
- [x] post-landing compatibility audit completed
- [x] full validation sweep completed
- [x] Sprint 51 closeout and Sprint 52 handoff completed from the validated baseline

## What Went Well

1. **Sprint 51 converted the Sprint 50 design package into a real implementation without reopening the contract.**
   The sprint stayed aligned to the Sprint 50 boundary:
   - shared analysis/factors-centric repeated-run story
   - one-shot direct APIs remain first-class
   - no generic direct-handle redesign
   - no raw internal storage exposure
   That kept the work implementation-shaped instead of drifting back into API
   brainstorming.

2. **The shared direct lifecycle story is now user-visible in the public headers.**
   Sprint 51 made the repeated-run direct path explicit across:
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   The key public contract now reads coherently as:
   - zero/init
   - analyze once
   - factor / solve
   - refactor / solve many
   - free

3. **Cholesky and LDL^T integrated cleanly through the shared lifecycle path.**
   The shared `sparse_analyze(...)` plus `sparse_factor_numeric(...)` path now
   routes through the normal family implementations for:
   - Cholesky
   - LDL^T
   That means the public repeated-run path uses the same real backend dispatch
   and factor behavior as the family-local one-shot entries instead of acting
   like a second-class side channel.

4. **LU advanced materially without pretending the harder option cases were solved.**
   Sprint 51 still landed a meaningful LU improvement:
   - the bounded default `sparse_lu_factor_opts(...)` path now routes through
     the shared analysis/factor path
   - the one-shot `sparse_lu_factor(...)` / `sparse_lu_solve(...)` caller story
     stayed intact
   Just as importantly, the sprint did not fake completeness for the option
   cases that do not yet map cleanly:
   - custom pivot/tolerance
   - progress/cancellation callback
   - non-original matrix state

5. **The wrapper-preservation pass found and respected a real LU recursion seam instead of papering over it.**
   Day 8 did the right thing technically:
   - Cholesky and LDL^T simple wrappers were normalized through their bounded
     default-options seams
   - LU was intentionally left on the family-local one-shot wrapper path after
     the attempt exposed real recursion through `sparse_factor_numeric(...)`
   That is a better result than forcing superficial symmetry and introducing a
   latent bug.

6. **The regression story is stronger now and is aimed at the public contract that actually matters.**
   Sprint 51 added focused public lifecycle coverage in
   `tests/test_integration.c` for:
   - wrapper/default-options parity
   - explicit lifecycle parity vs one-shot family routes
   - solve rejection on zeroed/unfactored `sparse_factors_t`
   - first-use acceptance for zeroed `sparse_factors_t` in
     `sparse_refactor_numeric(...)`
   That is higher-value coverage than only proving internal helper behavior.

7. **The adoption/docs surfaces were corrected where they actually mattered.**
   Sprint 51 fixed the two concrete caller-surface drifts identified in Sprint
   50:
   - `examples/README.md` now includes `example_analysis`
   - `benchmarks/README.md` now describes `bench_refactor` and
     `bench_refactor_csc` in terms that match the live Cholesky
     analyze-once/refactor-many path
   That gives the repeated-run direct story a better public teaching surface.

8. **The sprint closed from a fully validated baseline, not from partial smoke coverage.**
   The full gate passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   And the maintained truthfulness anchors stayed exact:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - reviewed CMake `ctest` `53 / 53`
   - `Total Test time (real) = 500.45 sec`

## What Didn't Go Well

1. **LU only reached a bounded Phase 1 integration, not a fully uniform lifecycle route.**
   That is an acceptable Sprint 51 result, but it means the LU family still has
   more compatibility-shaped complexity than Cholesky and LDL^T because some
   option cases remain intentionally on the older path.

2. **The wrapper story is not perfectly symmetrical across the direct families.**
   Cholesky and LDL^T simple wrappers now delegate through their bounded default
   options seams, while LU intentionally does not because of the recursion seam.
   That asymmetry is justified, but it is still a small maintainability debt.

3. **The repeated-run performance story is credible on the heavier CSC path, but not universally faster on every small/moderate one-shot comparison.**
   Day 13 preserved strong CSC repeated-run wins:
   - `bcsstk14 5.36x`
   - `s3rmt3m3 7.87x`
   - `Kuu 6.33x`
   - `Pres_Poisson 12.14x`
   But the lighter `bench_refactor` one-shot-vs-analyze comparison still showed
   some near-parity or loss cases, including:
   - `nos4 0.76x`
   That is not a correctness problem, but it means the lifecycle story should
   stay framed as a stable-pattern repeated-run path, not as a blanket
   speedup promise.

4. **The sprint intentionally did not broaden adoption into every example/tutorial surface.**
   That kept scope healthy, but it also means the repeated-run direct story is
   still concentrated in the strongest example/benchmark surfaces rather than
   uniformly present across every teaching asset.

## Final Metrics

### Validated implementation baseline

| Metric | Sprint 51 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity contract | `53 vs 53` |
| full reviewed CMake `ctest` result | `53 / 53` |
| reviewed CMake total test time | `500.45 sec` |

### Sprint 51 artifact package

| Metric | Sprint 51 close state |
|---|---:|
| total artifact files under `SPRINT_51/artifacts/` | `15` |
| baseline/setup artifacts (Days 1-2) | `3` |
| implementation/design/validation artifacts (Days 3-10) | `8` |
| adoption/audit/closeout artifacts (Days 11-14) | `4` |

### Direct lifecycle implementation outputs

| Metric | Sprint 51 close state |
|---|---:|
| direct solver families advanced in Phase 1 | `3` |
| shared/family public headers refreshed | `4` |
| touched source/header/test files in the landed Phase 1 package | `9` |
| focused lifecycle regression home | `1` |
| adopted caller-facing README surfaces | `2` |
| targeted direct-lifecycle follow-on binaries in Day 13 sweep | `8` |

Notes:

- direct solver families advanced in Phase 1:
  - LU
  - Cholesky
  - LDL^T
- shared/family public headers refreshed:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- touched source/header/test files in the landed Phase 1 package:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_lu.c`
  - `src/sparse_analysis.c`
  - `src/sparse_cholesky.c`
  - `src/sparse_ldlt.c`
  - `tests/test_integration.c`
- focused lifecycle regression home:
  - `tests/test_integration.c`
- adopted caller-facing README surfaces:
  - `examples/README.md`
  - `benchmarks/README.md`
- targeted direct-lifecycle follow-on binaries in Day 13 sweep:
  - `./build/example_analysis`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

## Residual Deferred Debt

Sprint 51 intentionally stopped at a bounded Phase 1 public lifecycle landing.
The main open work it hands forward is:

- deeper LU lifecycle integration beyond the bounded default-options path
- any later cleanup needed to resolve the LU recursion seam more uniformly
- broader repeated-run direct adoption beyond the highest-signal example and
  benchmark surfaces
- any later direct-family expansions intentionally deferred from Sprint 51
- further public direct-solver workflow cleanup built on the now-live
  analysis/factor/refactor contract

Not carried forward as unresolved Sprint 51 debt:

- missing public direct header contract
- missing shared Cholesky lifecycle routing
- missing shared LDL^T lifecycle routing
- missing focused lifecycle regression coverage
- missing example/benchmark adoption fixups for the known Sprint 50 drifts
- missing full validation sweep

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day4-public-direct-lifecycle-header-batch.md](./artifacts/day4-public-direct-lifecycle-header-batch.md)
- [day5-lu-lifecycle-integration-batch.md](./artifacts/day5-lu-lifecycle-integration-batch.md)
- [day6-cholesky-lifecycle-integration-batch.md](./artifacts/day6-cholesky-lifecycle-integration-batch.md)
- [day7-ldlt-lifecycle-integration-batch.md](./artifacts/day7-ldlt-lifecycle-integration-batch.md)
- [day8-wrapper-preservation-batch.md](./artifacts/day8-wrapper-preservation-batch.md)
- [day10-focused-regression-expansion-batch.md](./artifacts/day10-focused-regression-expansion-batch.md)
- [day11-example-and-benchmark-adoption-batch.md](./artifacts/day11-example-and-benchmark-adoption-batch.md)
- [day12-post-landing-compatibility-audit.md](./artifacts/day12-post-landing-compatibility-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 51 achieved its goal:

- the public direct-solver repeated-run story is now implemented, not only
  designed
- Cholesky and LDL^T now route cleanly through the shared public
  analysis/factor path
- LU advanced through a bounded lifecycle integration without breaking its
  compatibility-facing one-shot story
- one-shot direct APIs remain first-class peer entry points
- the public lifecycle regression story is stronger and aimed at the real
  contract surface
- the sprint closed from a fully validated reviewed baseline with exact
  preserved truthfulness anchors

Sprint 52 can now build on a live, validated Phase 1 direct-lifecycle surface
instead of starting from a design package or needing to re-prove the basic
analysis/factor/refactor public model.
