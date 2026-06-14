# Sprint 68 Day 9: Large-`n` Cholesky Public-Path Oracle/Parity Batch

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Land the Day 8 oracle design as one bounded integration-owner batch so Sprint
68 strengthens the hardest retained large-`n` CSC-backed Cholesky public path
without widening into implementation work or unrelated giant-test seams.

## Landed Surface

Required owner, as designed:

- `tests/test_integration.c`

Support not needed after landing:

- `tests/test_chol_csc.c`

The batch stayed inside that exact fence.

## What Changed

The landed batch strengthens:

- `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky(...)`

The strengthened test now carries a full staged oracle across three same-pattern
SPD states on the large-`n` CSC side:

1. baseline matrix
2. first same-pattern refactor matrix
3. second same-pattern refactor matrix

For each stage, the test now does all of the following:

- solves through the explicit repeated-run public lifecycle
- solves through the one-shot public Cholesky wrapper on a peer matrix
- checks the repeated-run result against the fixed exact solution
- checks the one-shot result against the fixed exact solution
- checks repeated-run and one-shot agreement directly

The baseline stage was the missing additive proof. The repo already had
baseline and later parity evidence in separate places, but not one continuous
public-path oracle story across baseline plus multiple same-pattern refactors in
the same owner.

## Explicit CSC-Side Routing Proof

The landed batch also tightened the route-publication contract where it is
available:

- `ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD)`
- baseline one-shot Cholesky asserts `used_csc_path == 1`
- refactor stage 1 one-shot Cholesky asserts `used_csc_path == 1`
- refactor stage 2 one-shot Cholesky asserts `used_csc_path == 1`

That keeps the test honest about which public path it is actually checking.

## Numerical Contract

The landed tolerance/oracle contract stayed exactly within the Day 8 design:

- exact-solution agreement at `1e-12`
- one-shot versus explicit repeated-run agreement at `1e-12`
- explicit CSC-side routing assertion when publication state is observed

This is still a public success-path oracle batch only. It is not:

- a failure-preservation batch
- a family-local helper-route batch
- a backend or implementation batch
- a benchmark/performance batch

## Non-Widening Fence Preserved

The landing did not widen into:

- `tests/test_chol_csc.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- implementation `src/` files
- benchmark/docs truth surfaces

## Validation

Because `*.c` changed, the required validation ran:

- `make format`
- `make lint`
- `make test`

Because this is a substantial assurance batch, the stronger reviewed baseline
also ran:

- `make quality-review-full`

All passed. The reviewed CMake parity anchor remained exact at `53`.
The maintained reviewed anchors also stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 452.07 sec`

## Exit State

Sprint 68 Day 9 closes with one stronger large-`n` CSC-backed public-path
oracle:

1. owner stayed:
   - `tests/test_integration.c`
2. the public-path parity lane now covers:
   - baseline
   - same-pattern refactor stage 1
   - same-pattern refactor stage 2
3. each stage now proves:
   - exact-solution agreement
   - one-shot versus repeated-run agreement
   - CSC-side routing when published
4. the batch stayed bounded:
   - no implementation widening
   - no extra test-owner churn
   - no benchmark/docs churn

That gives Day 10 a much cleaner rerank question:

- what is the strongest remaining giant-test or assurance seam after the staged
  public-path CSC-backed Cholesky oracle is now closed?
