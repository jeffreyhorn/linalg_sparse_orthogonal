# Sprint 54 Day 10 - regression and example adoption batch I

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Land the first bounded adoption/proof batch after the Day 9 benchmark
alignment:

- close the highest-value remaining direct public-handle proof gap
- align the strongest README/example entry points with the final Sprint 54
  repeated-run solver support boundary

## Landed scope

The Day 10 batch stayed bounded to:

- `tests/test_eigs.c`
- `README.md`
- `examples/README.md`

It did not broaden tutorial scope, add new solver families, or convert the
shipped examples into dedicated public-handle demos.

## What changed

### 1. Direct public-handle proof now covers the full intended eigensolver backend set

Before Day 10, the direct public-handle eigensolver tests already covered:

- generic repeated-run prepare/reuse
- zero-init on-demand growth
- explicit `LOBPCG`

The highest-value missing branch was explicit thick-restart under the public
handle surface.

Day 10 added:

- `test_public_handle_thick_restart_prepare_reuse_and_growth`

That regression now proves:

- explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
- explicit prepare on a smaller problem
- repeated reuse on the same prepared shape
- later on-demand growth to a larger problem and larger `k`
- preserved `backend_used == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`

That makes the direct proof set align with the final supported eigensolver
handle surface:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

### 2. The top-level README now states the final repeated-run iterative boundary honestly

The repeated-run lifecycle section in `README.md` now matches the landed Sprint
54 state:

- one-shot APIs remain first-class
- the iterative repeated-run public handle surface now lists:
  - `sparse_iter_handle_prepare_cg(...)`
  - `sparse_iter_handle_prepare_gmres(...)`
  - `sparse_iter_handle_prepare_minres(...)`
  - `sparse_solve_cg_with_handle(...)`
  - `sparse_solve_gmres_with_handle(...)`
  - `sparse_solve_minres_with_handle(...)`
- the final supported iterative repeated-run-handle families are named
  explicitly:
  - `CG`
  - `GMRES`
  - `MINRES`
- the bounded exclusions are named explicitly:
  - `BiCGSTAB`
  - block iterative workflows

The summary table and key-functions section were also updated so they no longer
lag the repeated-run section.

### 3. `examples/README.md` now matches the final example-support boundary

The examples README now states the real contract more directly:

- the shipped examples stay intentionally one-shot-first
- explicit repeated-run public surfaces are opt-in and bounded
- iterative repeated-run-handle support is:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver repeated-run-handle support is:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- Sprint 54 does not broaden the public repeated-run handle surface to:
  - `BiCGSTAB`
  - block iterative workflows

The per-example descriptions were also tightened:

- `example_iterative` remains a one-shot GMRES teaching surface by design
- `example_eigs` remains a one-shot eigensolver teaching surface by design,
  while the public repeated-run handle path is described separately

## Validation

### Required Day 10 gates

- `make format`
- `make lint`
- `make test`

All passed.

### Focused follow-ons

- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/bench_eigs_reuse`

All passed.

## Representative results

### Direct public-handle proof

`test_eigs`:

- `29 / 29`
- now includes:
  - `test_public_handle_prepare_and_reuse`
  - `test_public_handle_thick_restart_prepare_reuse_and_growth`
  - `test_public_handle_lobpcg_prepare_reuse_and_growth`

`test_eigs_lobpcg`:

- `26 / 26`

### Reuse benchmark alignment remains stable

`bench_eigs_reuse`:

- `growm-nos4-k5`
  - `1.00x`
  - `|lambda|max diff = 0.000e+00`
- `thick-bcsstk14-k5`
  - `1.05x`
  - `|lambda|max diff = 0.000e+00`
- `lobpcg-diag40-k3`
  - `1.05x`
  - `|lambda|max diff = 0.000e+00`

### Example stability checks

`example_eigs`:

- explicit `LOBPCG` on `bcsstk04`
- `3 / 3` smallest eigenpairs
- `62` outer iterations
- `backend_used = LOBPCG`
- `residual_norm = 8.808e-09`

`example_iterative`:

- one-shot GMRES: `25` iterations, `9.56e-11`
- ILU(0)-GMRES: `9` iterations, `3.14e-11`

`example_ic_minres`:

- `MINRES`: `39` iterations, `3.87e-11`
- `Jacobi-MINRES`: `26` iterations, `4.16e-11`

## Conclusion

Day 10 closes the highest-value remaining first adoption/proof drift without
reopening scope:

- direct public-handle proof now explicitly covers the full intended
  eigensolver backend set
- the strongest user-facing README/example surfaces now describe the final
  repeated-run support boundary honestly

That leaves Sprint 54 positioned for any final residual proof/docs sweep,
followed by the compatibility audit and validation closeout.
