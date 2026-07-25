# Sprint 135 Day 9 - Compressed-First Cookbook Batch 2

## Purpose

Complete the compressed-first cookbook implementation by adding SVD,
symmetric eigensolver, and benchmark/report handoff paths. This batch keeps
the cookbook as concise adoption guidance while preserving examples, headers,
and benchmark docs as the detailed authorities.

## Public Documentation Changes

### `docs/cookbook.md`

Added `SVD and Low-Rank Workflows`:

- routes CSR, CSC, and Matrix Market inputs through the normal public matrix
  shell before SVD calls
- keeps the original unfactored/unreordered matrix view rule visible
- maps output needs to public SVD routes:
  - full SVD
  - partial SVD
  - numerical rank
  - condition estimate
  - pseudoinverse
  - dense low-rank approximation
  - sparse dropped low-rank approximation
- links `examples/example_svd_lowrank.c` and `include/sparse_svd.h`
- states caller ownership for dense buffers and `SparseMatrix *` ownership for
  sparse low-rank output

Added `Symmetric Eigensolver Workflows`:

- states the symmetric-input requirement before routing into
  `sparse_eigs_sym(...)`
- keeps CSR/CSC/Matrix Market import as the first step before solver choice
- recommends default backend behavior for first use
- keeps shift-invert, preconditioning, explicit backend selection, and
  repeated-run handles framed as advanced paths
- links `examples/example_eigs.c` and `include/sparse_eigs.h`
- explicitly avoids a nonsymmetric eigensolver workflow claim

Added `Measure After Choosing the API Workflow`:

- maps chosen API workflows to benchmark handoffs:
  - `bench_main`
  - `bench_refactor`
  - `bench_refactor_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
  - `bench_svd`
  - reorder/fill benchmark surfaces
  - `make bench-canonical-report`
  - `make performance-sentinels`
- links `benchmarks/README.md` as the command, CSV, report-artifact, and
  interpretation authority
- states benchmark rows are local evidence tied to environment, matrix corpus,
  build options, backend selection, and thread settings

### `README.md`

Updated the cookbook documentation-index description so it reflects the
complete compressed-first scope:

- direct
- iterative
- Matrix Market
- SVD
- eigensolver
- benchmark handoff

## Maintained Example and Authority Handoff

The cookbook now links these maintained examples or authority surfaces:

- `examples/example_compressed_input.c`
- `examples/example_basic_solve.c`
- `examples/example_analysis.c`
- `examples/example_iterative.c`
- `examples/example_ic_minres.c`
- `examples/example_matrix_free.c`
- `examples/example_matrix_market.c`
- `examples/example_svd_lowrank.c`
- `examples/example_eigs.c`
- `docs/matrix_market.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`

## Claim Boundary Review

This batch preserved the Day 7 claim fences:

- no package-manager availability claim added
- no shared-library or dynamic-ABI claim added
- no platform support-tier claim changed
- no state-of-the-art parity claim added
- no nonsymmetric eigensolver support claim added
- no portable performance guarantee added
- no claim that benchmark reports are broad pass/fail timing gates
- no claim that compressed constructors adopt caller arrays
- no claim that repeated-run handles cover BiCGSTAB or block iterative
  workflows
- no claim that benchmark rows replace examples, tests, or API assumptions

## Residual Queue

Remaining Sprint 135 documentation work should focus on:

- Day 10 report-index adoption language, especially generated `manifest.txt`
  and `index.tsv` discoverability
- Day 11 navigation alignment after the cookbook and algorithm split have both
  landed
- Day 12 validation and any residual isolated historical anecdotes in
  `docs/algorithm.md`

## Validation Plan

Documentation-only validation for this batch:

- `git diff --check`
- trailing-whitespace scan on touched docs and Sprint 135 artifacts
- existence check for cookbook, linked examples, benchmark README, and linked
  public headers
- cookbook section scan for SVD, eigensolver, and benchmark headings
- benchmark and unsupported-claim scan
- `git diff --name-only -- '*.c' '*.h'` to confirm no code-day quality gate is
  required

## Completion Criteria

- all requested compressed-first workflow families now have concise adoption
  paths
- benchmark guidance points to measurement interpretation rather than broad
  performance claims
- maintained examples remain the executable detail source
- benchmark docs remain the command/report interpretation authority
