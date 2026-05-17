# Sprint 31 Day 12: Documentation Refresh

## Goal

Refresh the benchmark-facing and public-facing docs so they match the
Sprint 31 benchmark/tooling behavior already implemented on this branch:

- benchmark reorder-mode coverage
- compile-only tooling gate workflow
- designated-initializer guidance in public examples

## Files Updated

- `benchmarks/README.md`
- `README.md`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`
- `docs/algorithm.md`

## What Changed

### Benchmark-facing docs

`benchmarks/README.md` now documents the intentionally different reorder
surfaces across the benchmark entry points:

- `bench_main` accepts `none`, `rcm`, `amd`, and `nd`
- `bench_reorder` compares `none`, `rcm`, `amd`, `colamd`, and `nd`
- `bench_colamd` / `example_colamd` are QR/COLAMD comparison tools
- `bench_chol_csc` / `bench_ldlt_csc` remain fixed-reorder backend
  comparison tools rather than general reorder sweeps

This closes the last benchmark-doc gap left after the Day 3, Day 4, and
Day 9 behavior cleanup.

### Compile-only gate docs

The root `README.md` now includes:

- `make tooling-build`
- `make lint`

with the Sprint 31 note that `make lint` includes the compile-only
benchmark/example gate introduced on Day 11.

### Public-facing example style

The public LU, Cholesky, and analysis examples now use designated
initializers instead of brittle positional option-struct initialization.

The most important contract clarification landed in
`include/sparse_analysis.h`: the documented reorder set now matches the
real analysis API (`NONE`, `RCM`, `AMD`, `COLAMD`, `ND`), with the
existing note preserved that `sparse_analyze()` applies `COLAMD`
symmetrically and that QR remains the column-only `COLAMD` path.

## Validation

Day 12 is docs-only, so no `make format`, `make lint`, or `make test`
run was needed.

Validation was limited to a targeted doc sanity sweep:

- inspect the diffs for the touched documentation files
- grep for the new reorder-contract and designated-initializer strings

## End State

The Sprint 31 docs now match the shipped benchmark/tooling behavior on
this branch instead of leaving maintainers to infer it from the code.
