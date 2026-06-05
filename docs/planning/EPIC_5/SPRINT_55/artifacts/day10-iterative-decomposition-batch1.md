# Sprint 55 Day 10 - iterative decomposition batch 1

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Land the first bounded `src/sparse_iterative.c` extraction by moving the
scalar/handle `MINRES` family into its own permanent source file while keeping
the shared block-wrapper scaffolding in the retained main iterative file and
preserving the public repeated-run solver contract.

## Landed implementation

The new owned iterative file is:

- `src/sparse_iterative_minres.c`

Moved implementation set:

- `sparse_solve_minres_with_workspace_internal(...)`
- `sparse_solve_minres(...)`
- `sparse_solve_minres_with_handle(...)`

Retained in `src/sparse_iterative.c`:

- public handle init/free and growth helpers
- shared staging / residual-history / reporting helpers
- `CG`
- `GMRES`
- shared block-column scaffolding
- `solve_block_minres_column(...)`
- `sparse_minres_solve_block(...)`
- `BiCGSTAB`

This kept the first iterative batch narrower than the original Day 8 sketch and
matched the Day 9 design.

## Internal declaration strategy used

The extraction reused the existing iterative private headers:

- `src/sparse_iterative_internal.h`
- `src/sparse_iterative_workspace_internal.h`

No new private `MINRES` header was introduced.

The only real declaration widening was to make the split-safe shared helper
surface explicit in `src/sparse_iterative_internal.h`:

- `s29_iter_now_s(...)`
- `s49_iter_handle_ensure(...)`
- `stag_tracker_t`
- `stag_init(...)`
- `stag_free(...)`
- `stag_record(...)`
- `stag_check(...)`
- `reshist_t`
- `reshist_make(...)`
- `reshist_record(...)`
- `iter_report(...)`
- `sparse_solve_minres_with_workspace_internal(...)`

This kept the batch ownership-focused rather than turning it into a private
header taxonomy redesign.

## Touched permanent files

- `CMakeLists.txt`
- `Makefile`
- `src/sparse_iterative.c`
- `src/sparse_iterative_internal.h`
- `src/sparse_iterative_minres.c` (new)

## Ownership/result summary

Current post-Day-10 line counts:

- `src/sparse_iterative.c` = `1985`
- `src/sparse_iterative_minres.c` = `308`
- `src/sparse_iterative_internal.h` = `79`

Relative to the pre-Day-10 state:

- `src/sparse_iterative.c`: `2377` -> `1985`

This is a real decomposition improvement:

- the retained main iterative file is materially smaller
- the scalar/handle `MINRES` family now owns its own implementation body
- the public iterative header/API surface did not need to change

## Comment cleanup performed

The moved `MINRES` ownership band no longer carries the stale
`Sprint 29 Day 7` progress/cancel narration inside its extracted backend body.

The cleanup stayed bounded:

- durable algorithm commentary was preserved
- only the moved ownership band was normalized
- the batch did not turn into a whole-file comment rewrite

## Validation

Required code-day validation passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

Reviewed truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 244.60 sec`

## Focused follow-ons

The strongest iterative parity surfaces also passed:

- `./build/test_iterative` -> `79 / 79`
- `./build/test_minres` -> `43 / 43`
- `./build/example_ic_minres`
- `./build/bench_iterative_reuse`

Representative retained behavior:

- `example_ic_minres`:
  - `MINRES` on the `42x42` KKT system still converged in `39` iterations
  - Jacobi-`MINRES` still converged in `26` iterations
- `bench_iterative_reuse`:
  - `cg-tridiag-300` -> `1.05x`
  - `gmres-unsym-220` -> `1.04x`
  - `minres-kkt-42` -> `1.11x`
  - one-shot vs handle-path iteration counts and residuals stayed identical

## Conclusion

Sprint 55 Day 10 successfully landed the first bounded iterative extraction:

- `MINRES` scalar/handle ownership now lives in `src/sparse_iterative_minres.c`
- `src/sparse_iterative.c` is smaller and more orchestration-focused
- the existing internal header strategy was sufficient after one bounded shared
  helper widening
- block wrappers stayed in the main file as planned
- public repeated-run behavior, one-shot compatibility, examples, and
  benchmarks remained stable under full validation

That closes the planned Batch 1 iterative implementation step without
reopening the Sprint 54 repeated-run solver support boundary.
