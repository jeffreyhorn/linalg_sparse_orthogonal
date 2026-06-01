# Sprint 52 Day 14 - Closeout and handoff

Date: 2026-06-01
Branch: `sprint-52`

## Summary

Sprint 52 closes the second direct-solver lifecycle phase from a measured
validated baseline rather than from design intent or partial implementation.

The sprint started from the Sprint 51 Phase 1 public lifecycle package and
ended with a stronger shared analysis/factor/refactor path, tighter bounded
refactor semantics, refreshed factor-many benchmark proof, aligned caller-
facing adoption surfaces, and expanded public repeated-run regression proof.

## Delivered package

Sprint 52 leaves behind one coherent Phase 2 package:

- stronger shared repeated-run direct integration in:
  - `src/sparse_analysis.c`
  - `include/sparse_analysis.h`
- deeper numeric reuse on the highest-value shared direct paths:
  - shared Cholesky CSC repeated-run path reuses caller analysis directly
  - shared LDL^T CSC repeated-run path reuses caller analysis directly when
    the scalar pivot pre-pass does not introduce extra swaps
- tighter shared refactor boundary:
  - zero-init first-factorization support preserved
  - family/dimension/payload mismatch rejection tightened
  - cheap gross-structure drift rejection added via analyzed `nnz` tracking
- refreshed factor-many benchmark proof in:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/README.md`
- aligned caller-facing adoption updates in:
  - `README.md`
  - `examples/example_analysis.c`
- expanded public repeated-run regression proof in:
  - `tests/test_integration.c`

## Preserved contract

Sprint 52 preserved the Sprint 50-51 compatibility fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- repeated direct runs remain analysis/factors-centric around:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation setup, not old numeric factor contents
- repeated-run structure validation remains a cheap boundary check rather than
  a full structural-pattern verifier
- LU remains the strongest intentionally family-local special-case seam
- no raw internal CSC/native storage layout was exposed
- no generic direct-handle redesign was introduced

## Validation close state

Sprint 52 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained truthfulness anchors:

- reviewed CMake parity = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 200.43 sec`

Targeted Sprint 52 follow-ons also passed:

- `./build/test_integration`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` kept residuals at `4.44e-16`
- `bench_refactor` kept the repeated-run direct path ahead:
  - `tridiag-200 4.81x`
  - `tridiag-500 5.28x`
  - `bcsstk04 2.45x`
  - `nos4 2.72x`
- `bench_refactor_csc nos4` kept the CSC repeated-run path ahead:
  - `speedup_refactor = 1.52x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`

## Handoff to Sprint 53

Sprint 53 no longer needs to prove that the shared public direct lifecycle is
real or validated for the main Phase 2 paths.

The next bounded queue can therefore focus on real post-Sprint-52 work such
as:

- later direct-solver lifecycle depth that goes beyond the Sprint 52 fence
- stronger or broader structure-compatibility validation if a later sprint
  decides to pay that complexity cost
- any later LU-specific follow-on that should remain family-local rather than
  reopening the shared direct contract
- future caller-surface or benchmark expansion that builds on the now-
  validated Phase 2 package

## Project-plan impact

Sprint 52 does not require a `PROJECT_PLAN.md` update.

Reason:

- the sprint closed from the planned Day 13 validated baseline
- the delivered package still matches the Epic 5 Sprint 52 intent
- no blocker or replanning queue surfaced during closeout

## Conclusion

Sprint 52 is complete. It hands off a validated Phase 2 direct-solver
lifecycle package with stronger shared integration, preserved first-class
one-shot family entries, honestly bounded reuse/refactor semantics, measured
factor-many evidence, aligned high-signal caller surfaces, and stable reviewed-
baseline truthfulness anchors.
