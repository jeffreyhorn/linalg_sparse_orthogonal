# Sprint 51 Day 14 - Closeout and handoff

Date: 2026-06-01
Branch: `sprint-51`

## Summary

Sprint 51 closes the first implemented public direct-solver lifecycle API phase
for the main direct families. The sprint started from the Sprint 50 design
contract and ended with a validated implementation package rather than a second
design-only handoff.

## Delivered package

Sprint 51 leaves behind one coherent package:

- refreshed shared/family direct-solver header contract in:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- bounded LU lifecycle integration where the default one-shot option surface
  already mapped cleanly onto the shared analysis/factor path
- shared Cholesky lifecycle routing through the existing public
  analysis/factor path
- shared LDL^T lifecycle routing through the existing public analysis/factor
  path
- preserved one-shot wrapper posture across the touched families
- focused lifecycle regression additions in `tests/test_integration.c`
- aligned adoption/docs updates in:
  - `examples/README.md`
  - `benchmarks/README.md`

## Preserved contract

Sprint 51 preserved the Sprint 50 compatibility fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class supported peer entry
  points
- one-shot usage remains the simple/default path for one-off solves
- repeated direct runs remain analysis/factors-centric around:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation/setup state, not old numeric factor
  contents
- no raw internal CSC/native storage layout was exposed
- no generic direct-handle redesign was introduced

## Validation close state

Sprint 51 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- reviewed CMake parity = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 500.45 sec`

Targeted direct-lifecycle follow-ons also passed:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` kept residuals at `4.44e-16`
- `bench_refactor_csc` preserved the heavier repeated-run CSC wins:
  - `nos4 2.34x`
  - `bcsstk04 2.81x`
  - `bcsstk14 5.36x`
  - `s3rmt3m3 7.87x`
  - `Kuu 6.33x`
  - `Pres_Poisson 12.14x`

## Handoff to Sprint 52

Sprint 52 no longer needs to design the public direct repeated-run story. It
inherits a real implemented Phase 1 surface plus a validated compatibility
record.

The next sprint can therefore focus on later bounded adoption, expansion, or
cleanup work that builds on the shared analysis/factor/refactor contract
without reopening the basic public model.

## Conclusion

Sprint 51 is complete. It hands off an implemented and validated public
direct-solver lifecycle Phase 1 package with preserved one-shot compatibility
and stable reviewed-baseline truthfulness anchors.
