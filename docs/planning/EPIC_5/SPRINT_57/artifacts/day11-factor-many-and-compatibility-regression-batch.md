# Sprint 57 Day 11 - factor-many and compatibility regression batch

Date: 2026-06-06 19:26:07 CDT  
Branch: `sprint-57`

## Goal

Land one bounded factor-many / compatibility regression expansion in the
highest-signal remaining public direct seam:

- same-pattern refactor-many parity with the one-shot Cholesky path
- benchmark-facing analyze-once / refactor-many truthfulness

This batch must preserve the Sprint 50-56 direct lifecycle contract exactly
and avoid widening into new solver behavior.

## Files landed

- `tests/test_integration.c`

## Landed regression seam

Added:

- `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky`

The new test owns one narrow but important public story:

1. analyze once on an SPD matrix
2. factor once on the shared direct lifecycle path
3. refactor twice on fresh same-pattern value-updated matrices
4. solve after each refactor through the public repeated-run path
5. factor matching fresh matrices through the one-shot Cholesky compatibility
   path
6. prove the repeated-run and one-shot paths recover the same solution on both
   updates

This is the right Day 11 seam because it directly proves the same-pattern
factor-many assumption used by the benchmark and caller-facing repeated-run
story, while still respecting the one-shot compatibility fence.

## Preserved fence

The landing stayed inside the Day 11 boundary:

- no new test target
- no `Makefile` changes
- no `CMakeLists.txt` changes
- no direct-solver API changes
- no benchmark-driver changes
- no new repeated-run semantics beyond the already documented contract

This was a regression-proof expansion, not a solver behavior change.

## Validation

### Required gate

- `make format`
- `make lint`
- `make test`

All passed.

### Focused touched-surface follow-ons

- `./build/test_integration` -> `39 / 39`
- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained outputs:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor`
  - `tridiag-200` = `1.72x`
  - `tridiag-500` = `1.39x`
  - `bcsstk04` = `1.52x`
  - `nos4` = `1.49x`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 1.08x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

### Reviewed baseline

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 202.80 sec`

## Conclusion

Sprint 57 Day 11 delivered one bounded but high-signal factor-many proof gain:

- the shared public refactor-many path now has direct parity coverage against
  the one-shot Cholesky compatibility path
- the benchmark-facing analyze-once / refactor-many contract is now explicit in
  the regression surface
- the Sprint 50-56 direct lifecycle fence stayed exact

That leaves the remaining Sprint 57 queue smaller and more focused on later
compatibility/factor-many cleanup rather than on still-implicit one-shot
versus repeated-run caller expectations.
