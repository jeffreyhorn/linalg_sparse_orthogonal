# Sprint 54 Day 11 - regression and example adoption batch II

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Close the last high-value explicit repeated-run proof gap and clean up the last
stale high-signal wording seams before Sprint 54 moves into the compatibility
audit and final validation phases.

## Landed scope

The Day 11 batch stayed narrowly bounded to:

- `tests/test_eigs.c`
- `README.md`
- `docs/tutorial.md`

No API expansion, tutorial rewrite, or example-source churn was introduced.

## What changed

### 1. Direct public repeated-run eigensolver proof now covers all three supported backend branches explicitly

Before Day 11, the direct public repeated-run eigensolver tests already covered:

- generic repeated-run prepare/reuse
- zero-init on-demand growth
- explicit thick-restart
- explicit `LOBPCG`

The remaining high-value gap was explicit grow-m under the public handle
surface.

Day 11 added:

- `test_public_handle_growm_prepare_reuse_and_growth`

That regression now proves:

- explicit `SPARSE_EIGS_BACKEND_LANCZOS`
- explicit prepare on a smaller problem
- repeated reuse on the same prepared shape
- later on-demand growth to a larger problem and larger `k`
- preserved `backend_used == SPARSE_EIGS_BACKEND_LANCZOS`

That makes the explicit direct proof set match the final supported repeated-run
eigensolver handle surface exactly:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

### 2. The last stale high-signal README/tutorial lines now match the landed support state

Day 11 also cleaned up the last small summary lines that still lagged behind
the landed state:

- `README.md` project-structure line for `sparse_iterative.h`
  - now names repeated-run handles for `CG` / `GMRES` / `MINRES`
- `docs/tutorial.md` top include comment
  - now says `CG, GMRES, MINRES iterative solvers`

These are small edits, but they matter because they are high-visibility summary
surfaces that readers scan early.

## Validation

### Required Day 11 gates

- `make format`
- `make lint`
- `make test`

All passed.

### Focused follow-ons

- `./build/test_eigs`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`
- `rg` sanity checks over the touched summary wording

All passed.

## Representative results

### Direct proof surface

`test_eigs`:

- `30 / 30`
- now explicitly includes:
  - `test_public_handle_growm_prepare_reuse_and_growth`
  - `test_public_handle_thick_restart_prepare_reuse_and_growth`
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`

### Reuse benchmark surface

`bench_eigs_reuse`:

- `growm-nos4-k5`
  - `1.07x`
  - `|lambda|max diff = 0.000e+00`
- `thick-bcsstk14-k5`
  - `1.01x`
  - `|lambda|max diff = 0.000e+00`
- `lobpcg-diag40-k3`
  - `0.99x`
  - `|lambda|max diff = 0.000e+00`

### Example stability check

`example_eigs`:

- explicit `LOBPCG` on `bcsstk04`
- `3 / 3` smallest eigenpairs
- `62` outer iterations
- `backend_used = LOBPCG`
- `residual_norm = 8.808e-09`

## Conclusion

Day 11 closes the remaining high-value Sprint 54 proof/docs gaps without
reopening scope:

- direct public repeated-run eigensolver proof now explicitly covers all three
  supported backend branches
- the last high-signal README/tutorial summary lines now match the landed
  repeated-run solver support surface

That leaves the branch ready for the Day 12 compatibility audit and the final
validation/closeout path.
