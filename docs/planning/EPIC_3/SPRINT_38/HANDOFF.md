# Sprint 38 Handoff

**Source sprint:** 38  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 38's regression-proofing, gate-expansion, and
readiness work into explicit starting constraints for Sprint 39 final audit and
Epic 3 closeout work.

## Starting State For Sprint 39

Sprint 38 does **not** hand off a broken reviewed baseline, an unresolved
compile-db exclusion queue, or ambiguous coverage/readiness wording.

Authoritative validated close state at Sprint 38 close:

- `make format`: passed, `real 3.05`
- `make lint`: passed, `real 239.91`
- `make test`: passed, `real 71.18`
- `make quality-review-full`: passed, `real 485.93`
- `ctest -N --test-dir build/quality-review-cmake`: `53`
- full reviewed CMake `ctest`: `53 / 53` passed
- authoritative serial `make deadcode-report`: passed, `real 0.33`
- authoritative serial `make deadcode-check`: passed, `real 0.52`

## Highest-Value Shipped Sprint 38 Results

Sprint 38 closed or clarified the main regression-proofing queue it inherited:

- coverage-honesty drift reduced:
  - README no longer overstates test counts or coverage policy
  - `SPARSE_TEST_LARGE=1 make test` is now part of the documented live opt-in
    surface
- dead-code compile-db exclusion list closed:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- dead-code report/check output is clearer in the zero-gap state
- `make quality-review-full` now names the strongest local reviewed baseline
- `README.md` now includes a concise quality-readiness checklist

## Maintained Contract Now In Force

### Local reviewed baseline

The strongest local reviewed baseline is now explicit:

- `make quality-review-full`

That wrapper runs:

- `make quality-review`
- `make quality-review-cmake`

### Dead-code contract

The dead-code contract remains:

- report/check completeness gate
- no current benchmark/example compile-db coverage gap
- no current definitely-unused internal cleanup batch
- public rows are audited keeps
- `cppcheck` secondary/noise rows remain supporting or explanatory data only

Still **not** the intended claim:

- zero findings
- zero static-analysis noise
- concurrent-safe execution

### Coverage/readiness contract

The README now states the maintained truth:

- current reviewed CMake suite size is `53`
- coverage is supplemental
- Linux coverage enforcement remains `80%` line coverage on `src/`

## Residual Deferred Queue

Sprint 38 closes without a new cleanup backlog, but several bounded residual
items remain for Sprint 39 final audit/closeout work.

Carried forward:

- dead-code shared-path serialized execution remains open
- residual dead-code buckets remain:
  - `public-surface-review = 4`
  - `secondary-candidate-signal = 35`
  - `non-deadcode-static-analysis-noise = 6`
- macOS dead-code remains staged
- Windows local Makefile reviewed-wrapper parity remains staged
- Windows dead-code remains excluded

Not carried forward as new Sprint 38 debt:

- dead-code compile-db exclusion list: closed
- stale coverage wording: closed in the main public surfaces
- missing strongest local reviewed baseline command: closed
- missing concise readiness checklist: closed

## Suggested First-Fix Queue For Sprint 39

Sprint 39 should start from final audit/closeout, not from reopening Sprint 38
implementation work.

Immediate later-sprint emphasis belongs here instead:

- final dead-code audit over the residual buckets
- final cross-platform audit over the still-staged/excluded surfaces
- standards/documentation closeout using the new README readiness surface
- final validation grounded in the current `53`-test reviewed parity baseline

## Reproduction Commands

Use these commands before and after Sprint 39 final-audit work:

1. `make format`
2. `make lint`
3. `make test`
4. `make quality-review-full`
5. `make deadcode-report`
6. `make deadcode-check`

Expected stable comparison targets at Sprint 38 close:

- `53` reviewed CTest tests
- full reviewed CMake `ctest`: `53 / 53` passing
- dead-code compile-db coverage gap: `0`
- no current definitely-unused internal cleanup batch

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day6-compile-only-regression-batch1.md](./artifacts/day6-compile-only-regression-batch1.md)
- [day8-deadcode-workflow-maturation-batch1.md](./artifacts/day8-deadcode-workflow-maturation-batch1.md)
- [day10-quality-gate-expansion-batch1.md](./artifacts/day10-quality-gate-expansion-batch1.md)
- [day12-readiness-checklist-and-reporting-polish.md](./artifacts/day12-readiness-checklist-and-reporting-polish.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)

