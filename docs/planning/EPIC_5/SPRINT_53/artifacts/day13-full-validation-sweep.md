# Sprint 53 Day 13: Full Validation Sweep

## Purpose

Day 13 runs the full validated closeout from the landed Sprint 53 state.

The goal is not another design or compatibility pass. The goal is to confirm
that the full required gate, the reviewed Makefile/CMake truthfulness anchors,
and the targeted Sprint 53 CSC/direct-solver follow-ons all pass together from
the same branch state.

## Main Day 13 Conclusion

Sprint 53 has a real measured validation close state:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `make quality-review-full` reviewed CMake total time = `124.22 sec`

## Required Gate

### `make format`

- passed

### `make lint`

- passed

### `make test`

- passed

### `make quality-review-full`

- passed

This included:

- reviewed Makefile path
- dead-code report-completeness closeout
- reviewed CMake rebuild/parity path
- full reviewed CMake `ctest`

## Truthfulness Anchors

The maintained reviewed truthfulness anchors remained exact:

- reviewed CMake test discovery:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity:
  - `53 vs 53`
- reviewed CMake execution through `make quality-review-full`:
  - `53 / 53`
- reviewed CMake total time from `make quality-review-full`:
  - `124.22 sec`

An earlier direct `ctest` invocation transiently reported a missing
`test_reorder_amd_qg` executable. Day 13 reran both:

- `ctest --test-dir build/quality-review-cmake -R test_reorder_amd_qg -V`
- `ctest --test-dir build/quality-review-cmake --output-on-failure`

Both passed cleanly, so the validated close state is the rerun result above,
not the transient false alarm.

## Targeted Sprint 53 Follow-Ons

### Public repeated-run CSC proof

- `./build/test_integration`
  - `37 / 37` passed
  - includes the Sprint 53 public repeated-run indefinite KKT lifecycle and
    `nnz`-drift preservation regressions

### Main repeated-run direct example

- `./build/example_analysis`
  - solve residual remained `4.44e-16`
  - runtime output still states:
    - reused state is symbolic/permutation setup only
    - stale numeric factor contents are not reused

### CSC factor-many benchmark proof

- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `workflow = chol_spd`
  - `speedup_refactor = 1.64x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
  - `workflow = ldlt_kkt`
  - `speedup_refactor = 1.36x`
  - `res_public = 2.96e-16`
  - `res_csc = 2.96e-16`

### Direct family/regression reruns

- `./build/test_cholesky`
  - `21 / 21`
- `./build/test_ldlt`
  - `84 / 84`
- `./build/test_etree`
  - `97 / 97`
- `./build/test_chol_csc`
  - `137 / 137`
- `./build/test_ldlt_csc`
  - `96 / 96`

These reruns keep the direct-family, symbolic-analysis, and CSC-completion
surfaces green alongside the shared public repeated-run path.

## Operational Result

Sprint 53 now closes from a validated measured baseline instead of from
inference:

1. the full required gate passed
2. the reviewed Makefile/CMake truthfulness anchors remained exact
3. the targeted CSC follow-through, indefinite repeated-run, and direct-family
   follow-ons all stayed green

No new reconciliation queue surfaced during validation. Day 14 can therefore
focus on closeout and Sprint 54 handoff rather than post-validation repair.
