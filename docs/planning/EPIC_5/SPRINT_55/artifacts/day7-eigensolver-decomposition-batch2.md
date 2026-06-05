# Sprint 55 Day 7 - eigensolver decomposition batch 2

Date: 2026-06-04
Branch: `sprint-55`

## Scope

Land the second bounded `src/sparse_eigs.c` decomposition batch by moving the
thick-restart restart-state / arrowhead / bounded-memory outer-loop backend
cluster into its own permanent source file while preserving the public
eigensolver contract, backend routing/reporting semantics, and the existing
proof/adoption surfaces.

## Landed implementation

The Day 7 extraction moved the thick-restart backend into:

- `src/sparse_eigs_thick_restart.c`

The extracted implementation set is:

- `lanczos_restart_state_free(...)`
- `s21_arrowhead_to_tridiag(...)`
- `lanczos_restart_pick_locked(...)`
- `lanczos_restart_state_assemble(...)`
- `lanczos_thick_restart_iterate(...)`
- `s21_build_dense_arrowhead(...)`
- `s21_recompute_residual(...)`
- `s21_thick_restart_outer_loop(...)`

The retained `src/sparse_eigs.c` role is now more explicit:

- public one-shot and handle entry points
- shared validation and result setup
- backend AUTO/explicit selection
- generic Lanczos helpers
- grow-m Lanczos path
- shared dense Jacobi helper
- shift-invert/shared operator composition
- top-level backend dispatch/orchestration

## Internal declaration strategy used

The extraction reused the existing:

- `src/sparse_eigs_internal.h`

No new thick-restart-specific private header was introduced in Batch 2.

The only declaration widening needed was to expose the shared helper surface
used by both the retained orchestration file and the extracted thick-restart
backend:

- `s20_lanczos_starting_vector(...)`
- `s20_spectrum_scale(...)`
- `s20_lift_ritz_vectors(...)`
- `s21_thick_restart_outer_loop(...)`

This keeps the batch ownership-focused:

- source extraction landed
- private-header taxonomy redesign stayed deferred

## Touched permanent files

- `CMakeLists.txt`
- `Makefile`
- `src/sparse_eigs.c`
- `src/sparse_eigs_internal.h`
- `src/sparse_eigs_thick_restart.c` (new)

## Ownership/result summary

Current post-Day-7 line counts:

- `src/sparse_eigs.c` = `1727`
- `src/sparse_eigs_thick_restart.c` = `934`

Relative to the post-Day-5 baseline:

- `src/sparse_eigs.c`: `2660` -> `1727`

This is a real decomposition improvement rather than a comment-only pass:

- the retained shared eigensolver file is materially smaller
- the thick-restart backend now owns its own implementation body
- the public header/API surface did not need to change

## Extraction fixes during landing

The first splice exposed two real cleanup items before validation completed:

- the new `src/sparse_eigs_thick_restart.c` still ended with a dangling section
  banner fragment
- the retained `src/sparse_eigs.c` lost the opening `/*` for the LOBPCG banner

Both were corrected before the full gate. No algorithm-level follow-up change
was needed after those splice fixes.

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
- `Total Test time (real) = 248.71 sec`

## Focused follow-ons

The strongest eigensolver surfaces stayed green:

- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_thick_restart` -> `20 / 20`
- `./build/test_eigs_lobpcg` -> `26 / 26`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative retained behavior:

- `example_eigs`:
  - explicit `LOBPCG` on `bcsstk04` still converged `3 / 3` smallest pairs in
    `62` outer iterations
  - residual stayed `8.808e-09`
- `bench_eigs_reuse`:
  - `growm-nos4-k5` -> `1.02x`
  - `thick-bcsstk14-k5` -> `0.97x`
  - `lobpcg-diag40-k3` -> `0.96x`
  - all retained exact eigenvalue parity:
    - `|lambda|max diff = 0.000e+00`

## Conclusion

Sprint 55 Day 7 successfully landed the second bounded eigensolver extraction:

- thick-restart now lives in `src/sparse_eigs_thick_restart.c`
- `src/sparse_eigs.c` is smaller and more orchestration-focused
- the existing internal header strategy was sufficient for the move
- public repeated-run behavior, backend routing, benchmarks, and shipped
  examples remained stable under full validation

That closes the planned Phase 1 eigensolver decomposition pair without
reopening the solver API boundary.
