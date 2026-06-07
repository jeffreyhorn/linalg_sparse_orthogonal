# Sprint 57 Day 10 - lifecycle regression expansion batch 1

Date: 2026-06-06 18:56:28 CDT  
Branch: `sprint-57`

## Goal

Land a bounded lifecycle regression expansion in the strongest remaining
public direct caller-story seam:

- repeated solve reuse on one analyzed/factored path
- explicit free-to-zero proof for `sparse_factors_t`
- explicit free-to-zero proof for `sparse_analysis_t`

The batch must preserve the Sprint 50-56 direct lifecycle contract exactly and
avoid broadening into new public behavior.

## Files landed

- `tests/test_integration.c`

## Landed regression seam

Added:

- `test_public_lifecycle_repeated_solve_and_free_zeroed`

The new test owns one narrow public lifecycle story:

1. analyze once
2. factor once
3. solve two distinct right-hand sides through the same
   `sparse_analysis_t` + `sparse_factors_t`
4. free `sparse_factors_t` and verify the public state is zeroed
5. free `sparse_analysis_t` and verify the public state is zeroed
6. call both free entry points again on the zeroed state to prove no-op safety

This is the right Day 10 seam because it strengthens the public steady-state
caller contract without widening into family-local numeric behavior.

## Preserved fence

The landing stayed inside the Day 10 boundary:

- no new test target
- no `Makefile` changes
- no `CMakeLists.txt` changes
- no direct-solver API changes
- no repeated-run semantic expansion beyond the already documented contract
- no support-boundary drift across LU / Cholesky / LDL^T

This was a regression-proof expansion, not a solver behavior change.

## Validation

### Required gate

- `make format`
- `make lint`
- `make test`

All passed.

### Focused touched-surface follow-ons

- `./build/test_integration` -> `38 / 38`
- `./build/example_analysis`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained outputs:

- `example_analysis`
  - solve residual = `4.44e-16`
- `bench_refactor_csc nos4`
  - `speedup_refactor = 2.52x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

### Reviewed baseline

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 203.98 sec`

## Conclusion

Sprint 57 Day 10 delivered one bounded but high-signal lifecycle proof gain:

- repeated public direct solve reuse is now explicit in the regression surface
- public free-to-zero behavior is now explicit for both lifecycle structs
- the Sprint 50-56 public lifecycle fence stayed exact

That leaves the remaining Sprint 57 lifecycle queue smaller and more focused on
later factor-many / compatibility proof rather than on still-implicit steady-
state caller behavior.
