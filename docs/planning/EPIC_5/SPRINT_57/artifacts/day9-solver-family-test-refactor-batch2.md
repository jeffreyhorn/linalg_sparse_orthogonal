# Sprint 57 Day 9 - solver-family test refactor batch 2

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Land the second bounded solver-family maintainability improvement by extracting
the public repeated-run iterative handle proof cluster out of
`tests/test_iterative.c` while preserving the `test_iterative` binary shape and
the final supported repeated-run handle boundary.

## Re-audit result

Post-Day-8 solver-family shape:

- `tests/test_svd.c` is now materially smaller and intentionally proof-dense
- `tests/test_iterative.c` is the strongest next solver-family seam because
  its public repeated-run handle cluster is contiguous and behaviorally
  cohesive
- `tests/test_qr.c` remains intentionally deferred

That made the Day 9 highest-value remaining solver-family seam:

- `tests/test_iterative.c`
- public repeated-run handle proof cluster:
  - `test_cg_public_handle_validation_reuse_and_on_demand`
  - `test_gmres_public_handle_prepare_reuse_and_growth`
  - `test_minres_public_handle_prepare_reuse_and_growth`

## Files landed

- `tests/test_iterative.c`
- `tests/test_iterative_handle_helpers.h`

## Ownership change

### New owned seam

`tests/test_iterative_handle_helpers.h` now owns the public repeated-run
iterative handle family:

- `CG` repeated-run handle proof
- `GMRES` repeated-run handle proof
- `MINRES` repeated-run handle proof

This is a real support-boundary seam because it captures the final public
handle set without widening into the larger one-shot, preconditioned, or
matrix-free solver families.

### Retained in `tests/test_iterative.c`

- CG one-shot and preconditioned proof
- GMRES one-shot, restart, and preconditioning proof
- GMRES/CG matrix-free proof
- `main()` and current `RUN_TEST(...)` ordering

## Preserved fence

The landing stayed inside the Day 9 boundary:

- no new test target
- no `Makefile` changes
- no `CMakeLists.txt` changes
- same `test_iterative` binary shape
- same `main()` ownership in `tests/test_iterative.c`
- same `RUN_TEST(...)` ordering
- same supported iterative repeated-run handle set:
  - `CG`
  - `GMRES`
  - `MINRES`

This was an ownership/readability change, not an iterative-solver behavior
change.

## Measured reduction

Line counts after landing:

- `tests/test_iterative.c` = `2802`
- `tests/test_iterative_handle_helpers.h` = `197`

Against the Sprint 57 baseline:

- `tests/test_iterative.c`: `2993 -> 2802`

That gives Sprint 57 a second real solver-family giant-test reduction while
keeping the test runner intact.

## Validation

### Required gate

- `make format`
- `make lint`
- `make test`

All passed.

### Focused touched-surface follow-ons

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/bench_iterative_reuse`
- `./build/example_ic_minres`

Representative retained outputs:

- `bench_iterative_reuse`
  - `cg-tridiag-300` = `1.08x`
  - `gmres-unsym-220` = `1.11x`
  - `minres-kkt-42` = `1.27x`
- `example_ic_minres`
  - `MINRES` on KKT `42x42` = `39` iterations
  - `Jacobi-MINRES` = `26` iterations
  - speedup = `1.5x`

### Reviewed baseline

- `make quality-review-full`

Passed with maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 207.54 sec`

## Deferred density

The remaining solver-family density is now intentional rather than accidental:

- the larger CG / GMRES / matrix-free proof surface in `tests/test_iterative.c`
  remains dense because it is still the main one-shot/front-door iterative
  behavior story
- `tests/test_qr.c` remains explicitly deferred

## Conclusion

Sprint 57 Day 9 delivered the second solver-family maintainability landing as a
clean owned seam:

- the public repeated-run iterative handle cluster is now explicitly separated
- `tests/test_iterative.c` is materially smaller
- the build/test surface stayed stable
- the final iterative repeated-run support boundary stayed exact

That leaves the remaining solver-family giant-test density as an intentional
coverage choice rather than a missed obvious split.
