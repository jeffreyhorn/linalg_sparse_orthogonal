# Sprint 54 Day 9 - public reuse benchmark alignment batch

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Close the narrow benchmark support-set drift identified on Day 8 so the public
repeated-run benchmark surfaces match Sprint 54's final supported solver
lifecycle boundary.

## Landed scope

The Day 9 batch stayed tightly bounded to three touched surfaces:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`

No new public API surface was introduced, and the benchmark framework/CLI was
not redesigned.

## What changed

### 1. `bench_iterative_reuse.c` now covers the full supported iterative public-handle set

The iterative reuse benchmark previously covered only:

- `CG`
- `GMRES`

Day 9 added one bounded `MINRES` repeated-run case:

- generated symmetric-indefinite KKT fixture
- `42x42`
- one-shot `sparse_solve_minres(...)` path
- explicit public-handle `sparse_solve_minres_with_handle(...)` path

This keeps the benchmark honest about what it is proving:

- repeated-run comparison on the public handle path
- no claim of public repeated-run-handle support for `BiCGSTAB`
- no claim of public block-iterative repeated-run-handle support

### 2. `bench_eigs_reuse.c` now covers the full supported eigensolver public-handle set

The eigensolver reuse benchmark previously covered only:

- grow-m Lanczos
- thick-restart Lanczos

Day 9 added one bounded explicit `LOBPCG` repeated-run case:

- generated diagonal SPD fixture
- `diag40`
- `k = 3`
- explicit `SPARSE_EIGS_BACKEND_LOBPCG`
- one-shot vs explicit public-handle repeated-run comparison

The refactor stayed small:

- existing file-backed cases were preserved
- one helper now runs arbitrary matrix-backed cases
- one helper continues to load the file-backed cases

### 3. `benchmarks/README.md` now names the reuse proof surfaces explicitly

The benchmark README now documents:

- `bench_iterative_reuse`
- `bench_eigs_reuse`

It also states their intended interpretation explicitly:

- narrow public repeated-run handle proof surfaces
- not broad solver bake-offs
- not replacements for `bench_eigs`

## Validation

### Required Day 9 gates

- `make format`
- `make lint`
- `make test`

All passed.

### Focused follow-ons

- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`
- `./build/example_ic_minres`
- `./build/example_eigs`

All passed.

## Representative measured results

### Iterative public reuse proof

`bench_iterative_reuse`:

- `cg-tridiag-300`
  - one-shot `23.3770 ms`
  - reuse `23.4230 ms`
  - `1.00x`
- `gmres-unsym-220`
  - one-shot `16.5780 ms`
  - reuse `15.7790 ms`
  - `1.05x`
- `minres-kkt-42`
  - one-shot `4.6260 ms`
  - reuse `4.5670 ms`
  - `1.01x`
  - one-shot / reuse both:
    - `39` iterations
    - `3.870e-11` relative residual
    - converged

### Eigensolver public reuse proof

`bench_eigs_reuse`:

- `growm-nos4-k5`
  - one-shot `1.1390 ms`
  - reuse `1.1380 ms`
  - `1.00x`
- `thick-bcsstk14-k5`
  - one-shot `39.0170 ms`
  - reuse `39.9380 ms`
  - `0.98x`
- `lobpcg-diag40-k3`
  - one-shot `0.8340 ms`
  - reuse `0.8330 ms`
  - `1.00x`
  - one-shot / reuse both:
    - `45` iterations
    - `6.696e-11` reported residual
    - `|lambda|max diff = 0.000e+00`
    - backend `LOBPCG`

### Example stability checks

- `example_ic_minres`
  - `MINRES`: `39` iterations, `3.87e-11`
  - `Jacobi-MINRES`: `26` iterations, `4.16e-11`
- `example_eigs`
  - explicit `LOBPCG` on `bcsstk04`
  - `3 / 3` smallest eigenpairs
  - `62` outer iterations
  - `residual_norm = 8.808e-09`

## Conclusion

Day 9 closed the benchmark support-set completeness gap without reopening API
design or benchmark-framework scope:

- iterative repeated-run public-handle proof now covers:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver repeated-run public-handle proof now covers:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

That leaves Sprint 54 in a better position for the remaining example/README
support-boundary adoption work and final closeout.
