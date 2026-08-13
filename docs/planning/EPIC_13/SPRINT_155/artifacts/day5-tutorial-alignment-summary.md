# Sprint 155 Day 5 Tutorial Alignment Summary

## Purpose

Day 5 finishes the Sprint 155 tutorial alignment pass by adding workflow-local
diagnostics, advanced handoffs, symmetric eigensolver coverage, and claim-safe
language for preconditioning and partial-SVD evidence.

## Files Changed

| File | Change Role |
| --- | --- |
| `docs/tutorial.md` | Finished tutorial diagnostics, advanced handoffs, eigensolver coverage, partial-SVD evidence delegation, and preconditioning claim cleanup. |
| `docs/planning/EPIC_13/SPRINT_155/WORKING_NOTES.md` | Recorded Day 5 summary and Day 6 header-selection handoff. |
| `docs/planning/EPIC_13/SPRINT_155/artifacts/day5-tutorial-alignment-summary.md` | Captures this Day 5 alignment record. |

No `.c` or public `.h` files were modified.

## Tutorial Changes

### Diagnostics

Added a workflow-local diagnostics handoff that routes:

- CSR/CSC construction to `NULL` or explicit `sparse_err_t`;
- Matrix Market input to `sparse_errno()` after `SPARSE_ERR_IO`;
- one-shot direct solves to factor/solve return codes and local residuals;
- repeated direct lifecycle to analyze/factor/refactor return codes and the
  same-pattern invariant;
- iterative solves to convergence, residual, iteration, stagnation, and
  breakdown fields;
- QR to rank, residual, nullity/nullspace, and minimum-norm outputs;
- SVD and partial-SVD to rank, condition, triplet residuals, convergence, and
  fail-closed status;
- symmetric eigensolvers to Ritz residual, convergence count, selected
  backend, peak basis size, and shift-invert/preconditioner status;
- benchmark/report interpretation to matrix, compiler, backend, thread
  settings, generated index, and manifest context.

The tutorial links diagnostics to `docs/solver_selection.md`, examples README,
and benchmarks README as the maintained owner surfaces.

### Install And Advanced Handoffs

Added an advanced handoff section that delegates:

- runtime/backend controls to README and solver-selection;
- benchmark/report commands and caveats to benchmarks README;
- installed downstream consumers and static-first support to `INSTALL.md`;
- exact declarations, options, result structs, ownership rules, and
  return-code contracts to public headers and generated API reference;
- maintainer evidence, report freshness, package/ABI, and support-tier
  interpretation to the maintainer guide.

The tutorial does not duplicate package-contract or support-tier tables.

### Symmetric Eigensolver Coverage

Added a compact symmetric eigensolver section for `sparse_eigs_sym(...)`:

- starts with AUTO backend behavior;
- links to `examples/example_eigs.c`;
- identifies convergence count, Ritz residual, selected backend, and peak
  basis size as diagnostics;
- delegates exact options/results/handles to `include/sparse_eigs.h`;
- rejects nonsymmetric eigensolver and portable state-of-the-art claims.

### Preconditioning Wording

Replaced the overbroad "ILU preconditioning dramatically reduces iteration
counts" text with local/workload-dependent wording:

- preconditioning can reduce iteration counts when the preconditioner matches
  the matrix assumptions;
- the result is local diagnostic evidence;
- it is not a portable performance guarantee.

### Partial-SVD Evidence

Replaced the stale single-fixture partial-SVD evidence paragraph with a short
current handoff:

- names the clustered/repeated 8x6 lane;
- names the Sprint 151 rank-deficient projector, sparse low-rank output, and
  fail-closed recovery rows;
- delegates the detailed evidence boundary to
  `docs/solver_selection.md#svd-and-low-rank-workflows`;
- preserves non-claims for broad correctness, raw singular-vector identity,
  external-library parity, performance, package/platform/ABI support, and
  state-of-the-art behavior.

## Cross-Link Reconciliation

The tutorial now has maintained links to:

- `README.md`;
- `INSTALL.md`;
- `examples/README.md`;
- `docs/cookbook.md`;
- `docs/solver_selection.md`;
- `docs/matrix_market.md`;
- `benchmarks/README.md`;
- public headers under `include/`;
- generated API reference surface;
- `docs/maintainer_guide.md`.

## Claim Boundary Check

Day 5 preserved these boundaries:

- no unqualified state-of-the-art sparse linear algebra claim;
- no broad QR, SVD, partial-SVD, or eigensolver correctness claim beyond
  maintained owner surfaces;
- no external-library parity claim;
- no portable performance or backend-superiority claim;
- no package-manager support claim;
- no shared-library or dynamic ABI support claim;
- no runtime-loader compatibility claim;
- no broad Windows parity, Windows Makefile parity, or Windows `pkg-config`
  parity claim;
- no generated report freshness claim from tutorial prose.

## Day 6 Handoff

Day 6 should select public headers for cleanup using the tutorial alignment
results:

1. prioritize headers whose comments support the updated tutorial pathways;
2. keep previously cleaned Sprint 145 headers as references rather than
   default rework targets;
3. favor high-impact headers not yet cleaned, especially those that map to
   current tutorial gaps around LDLT, eigensolver, IC/preconditioning,
   analysis lifecycle, Matrix Market/I/O, and public error contracts;
4. require declaration-preservation scans before and after any public header
   cleanup.

## Day 5 Completion Check

- Tutorial diagnostics guidance exists.
- Install and downstream-consumer references are delegated to `INSTALL.md`.
- Advanced-control, benchmark/report, API reference, and maintainer handoffs
  exist.
- Preconditioning wording is claim-safe.
- Partial-SVD evidence is current or delegated to the current owner surface.
- Unsupported package, ABI, platform, performance, external-parity,
  generated-report, and state-of-the-art claims were not introduced.
