# Sprint 52 Day 11: Regression Expansion Batch

## Purpose

Day 11 adds the smallest missing high-signal public-lifecycle regression proof
for the stronger Sprint 52 Phase 2 direct repeated-run path.

The goal is not broader churn. The goal is to prove that
`sparse_factor_solve(...)` enforces the ownership pairing between
`sparse_analysis_t` and `sparse_factors_t` and does not damage good factor
state after a rejected mismatched call.

## Main Day 11 Conclusion

Sprint 52 now has direct public proof for the most obvious remaining solve-time
ownership seam:

- mismatched analysis/factors family pairing is rejected
- mismatched analysis/factors dimension pairing is rejected
- rejected mismatched solves do not corrupt the original good factors
- the batch stays entirely in regression coverage, not implementation churn

## Touched Surface

### `tests/test_integration.c`

Day 11 adds one focused regression:

- `test_public_lifecycle_solve_rejects_mismatched_analysis_and_preserves_factors`

That test:

1. builds a good Cholesky repeated-run state on a 4x4 SPD matrix
2. builds a mismatched LU analysis on a same-size unsymmetric matrix
3. builds a mismatched Cholesky analysis on a different-size SPD matrix
4. verifies:
   - `sparse_factor_solve(&factors, &lu_analysis, ...)` returns
     `SPARSE_ERR_BADARG`
   - `sparse_factor_solve(&factors, &other_n_analysis, ...)` returns
     `SPARSE_ERR_SHAPE`
   - the original `factors` still solve correctly with `good_analysis`

This is the narrowest public regression that closes the solve-time pairing gap
without widening Sprint 52 into broader contract or implementation work.

## Explicit Non-Landings

Day 11 intentionally does **not** do these:

- change any `src/` implementation file
- redesign the public repeated-run direct contract
- reopen LU routing or wrapper posture
- retouch README / example / benchmark wording
- widen the batch into a larger compatibility or parity rewrite

## Validation

Because `tests/test_integration.c` changed, the full required code-day gate
was run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 156.92 sec`

## Focused Follow-On

The direct public repeated-run follow-ons also stayed clean:

- `./build/test_integration`
  - `33 / 33` passed
  - includes the new solve-time mismatch regression
- `./build/example_analysis`
  - solve residual remained `4.44e-16`
- `./build/bench_refactor`
  - repeated-run path remained ahead:
    - `tridiag-200 4.78x`
    - `tridiag-500 5.24x`
    - `bcsstk04 2.48x`
    - `nos4 2.81x`

## Day 11 Operational Result

Sprint 52 now has a better-balanced public-lifecycle regression floor:

1. zeroed/unfactored solve rejection is covered
2. zero-init first-factorization and refactor mismatch paths are covered
3. solve-time analysis/factors mismatch rejection and state preservation are
   now covered too

That closes the main missing public repeated-run regression seam cleanly enough
for Day 12 to focus on the post-landing compatibility audit instead of
reopening regression proof gaps.
