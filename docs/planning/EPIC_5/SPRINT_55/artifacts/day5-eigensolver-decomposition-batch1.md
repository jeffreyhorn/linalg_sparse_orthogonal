# Sprint 55 Day 5 - eigensolver decomposition batch 1

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Land the first bounded `src/sparse_eigs.c` decomposition batch by moving the
LOBPCG backend into its own permanent source file while preserving the public
one-shot/handle contract, backend routing/reporting semantics, and the main
proof/adoption surfaces.

## Landed implementation

The Day 5 extraction moved the LOBPCG backend into:

- `src/sparse_eigs_lobpcg.c`

The extracted function set is:

- `s21_lobpcg_orthonormalize_block(...)`
- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`
- `s21_lobpcg_init_X(...)`

The retained `src/sparse_eigs.c` role is now more explicit:

- public one-shot and handle entry points
- backend AUTO/explicit selection
- shared validation/result setup
- generic Lanczos helpers
- grow-m Lanczos path
- thick-restart path and restart-state machinery
- top-level backend dispatch/orchestration

## Internal declaration strategy used

The extraction reused the existing:

- `src/sparse_eigs_internal.h`

No new private LOBPCG-specific header was introduced in Batch 1.

This keeps the first batch ownership-focused:

- source extraction landed
- private-header taxonomy redesign stayed deferred

## Touched permanent files

- `CMakeLists.txt`
- `Makefile`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_lobpcg.c` (new)

## Ownership/result summary

The largest eigensolver source split moved from:

- `src/sparse_eigs.c` = `3233` lines at Sprint 55 Day 1 baseline

to:

- `src/sparse_eigs.c` = `2660`
- `src/sparse_eigs_lobpcg.c` = `401`

This is a real decomposition improvement rather than a cosmetic comment pass:

- the public orchestration file is materially smaller
- the extracted backend now owns its own implementation body
- the public header/API surface did not need to change

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
- `Total Test time (real) = 248.97 sec`

## Focused follow-ons

The strongest touched eigensolver surfaces stayed green:

- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative retained behavior:

- `example_eigs`:
  - explicit `LOBPCG` on `bcsstk04` still converged `3 / 3` smallest pairs in
    `62` outer iterations
  - reported residual stayed `8.808e-09`
- `bench_eigs_reuse`:
  - `growm-nos4-k5` -> `1.10x`
  - `thick-bcsstk14-k5` -> `0.99x`
  - `lobpcg-diag40-k3` -> `1.02x`
  - all retained exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

## Conclusion

Sprint 55 Day 5 successfully landed the first bounded eigensolver extraction:

- LOBPCG now lives in `src/sparse_eigs_lobpcg.c`
- `src/sparse_eigs.c` is smaller and more orchestration-focused
- the existing internal header strategy was sufficient for the move
- public repeated-run behavior, backend reporting, benchmarks, and shipped
  examples remained stable under full validation

That gives Sprint 55 a solid first decomposition result without reopening the
solver API boundary.
