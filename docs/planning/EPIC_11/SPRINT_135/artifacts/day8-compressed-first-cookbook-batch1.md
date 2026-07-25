# Sprint 135 Day 8 - Compressed-First Cookbook Batch 1

## Purpose

Implement the first cookbook batch from the Day 7 design: direct, iterative,
and Matrix Market compressed-first adoption paths. This batch makes the
workflow discoverable from first-use docs without changing public APIs,
examples, benchmarks, package support, or platform claims.

## Public Documentation Changes

### `docs/cookbook.md`

Created the compressed-first cookbook with these Day 8 sections:

- `Start From Your Data`
  - maps CSR arrays, CSC arrays, Matrix Market files, and small hand-written
    matrices to the first public entry point
  - states that CSR/CSC constructors validate and copy caller-owned arrays
  - distinguishes `sparse_create_from_*` from `sparse_from_*` diagnostic paths
- `Direct Solves From Compressed Input`
  - routes imported CSR/CSC data into LU, Cholesky, LDL^T, or QR by problem
    shape
  - keeps the one-shot in-place factorization copy rule visible
  - links `example_compressed_input`, `example_basic_solve`, and
    `example_analysis`
  - keeps repeated direct reuse scoped to same-pattern analyze/factor/refactor
- `Iterative Solves From Compressed Input`
  - routes imported CSR/CSC data into CG, GMRES, MINRES, or one-shot BiCGSTAB
  - preserves the repeated-run handle boundary for CG, GMRES, and MINRES only
  - keeps IC(0), ILU(0), and ILUT tied to their matrix/solver assumptions
  - separates matrix-free callback workflows from compressed-input workflows
- `Matrix Market Load/Use`
  - documents `sparse_load_mm(...)` as the first public load step
  - routes loaded matrices through the same public matrix-shell workflow
  - links `docs/matrix_market.md` and `example_matrix_market`
- `Next Steps`
  - points to solver selection, tutorial, examples, and benchmark docs without
    making the cookbook a benchmark manual

### Inbound Links

Added cookbook navigation from:

- `README.md`
  - Start Here entry for CSR, CSC, and Matrix Market data
  - Documentation index entry
- `docs/tutorial.md`
  - getting-started handoff for compressed-first data
- `docs/solver_selection.md`
  - front-matter link before API and benchmark references
- `examples/README.md`
  - CSR/CSC and Matrix Market start-here bullets

## Maintained Example Handoff

The cookbook links to maintained examples rather than duplicating complete
source:

- `examples/example_compressed_input.c`
- `examples/example_basic_solve.c`
- `examples/example_analysis.c`
- `examples/example_iterative.c`
- `examples/example_ic_minres.c`
- `examples/example_matrix_free.c`
- `examples/example_matrix_market.c`

## Claim Boundary Review

This batch did not widen product claims:

- no package-manager availability claim added
- no shared-library or dynamic-ABI claim added
- no portable performance claim added
- no platform support-tier claim changed
- no benchmark pass/fail timing claim added
- no claim that CSR/CSC constructors adopt caller arrays
- no claim that repeated-run handles cover BiCGSTAB or block iterative
  workflows
- no Matrix Market builder/module claim added

## Day 9 Residual Queue

Day 9 should extend `docs/cookbook.md` with:

- compressed-first SVD and low-rank route
- symmetric eigensolver route from imported/loaded sparse data
- benchmark/report handoff after API workflow selection
- refreshed claim-boundary scan after those sections land

## Validation Plan

Documentation-only validation for this batch:

- `git diff --check`
- trailing-whitespace scan on touched docs and Sprint 135 artifacts
- existence check for `docs/cookbook.md` and linked maintained examples
- cookbook link scan from README, tutorial, solver selection, and examples
- unsupported-claim scan for package, ABI, platform, and performance wording
- `git diff --name-only -- '*.c' '*.h'` to confirm no code-day quality gate is
  required

## Completion Criteria

- direct, iterative, and Matrix Market compressed-first paths are discoverable
  from first-use docs
- cookbook text stays concise and links to detailed references where needed
- maintained examples remain the executable detail source
- package, platform, ABI, and performance claim boundaries are preserved
