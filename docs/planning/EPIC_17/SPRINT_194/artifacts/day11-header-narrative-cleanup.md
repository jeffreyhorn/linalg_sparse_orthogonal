# Sprint 194 Day 11: Header Narrative Cleanup

## Objective

Move selected tutorial-style, workflow-routing, and evidence-summary prose out
of public headers while preserving declaration-adjacent API contracts.

## Scope

Edited selected comment blocks in:

- `include/sparse_matrix.h`
- `include/sparse_csr.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`

No declarations, typedefs, enum values, macro names, struct fields, function
signatures, ownership contracts, status-code mappings, callback contracts, or
public constants were intentionally changed.

## Cleanup Summary

- Replaced broad first-use and workflow-routing paragraphs with concise header
  ownership statements in matrix, iterative, QR, SVD, and eigensolver headers.
- Shortened compressed-format introduction text in `sparse_csr.h` without
  changing physical-index-space or ownership semantics.
- Trimmed Cholesky CSC threshold prose in `sparse_matrix.h` to keep the local
  dispatch policy and override behavior while removing benchmark-corpus
  narrative from the declaration header.
- Removed QR reorder recommendations and QR evidence-boundary routing while
  keeping accepted reorder enum behavior and rank-diagnostic semantics.
- Shortened partial-SVD and low-rank sparse approximation evidence prose while
  preserving compute mode, output-shape, non-convergence, and cleanup
  contracts.
- Replaced the large eigensolver usage example with the API-local buffer and
  return-code contract, leaving runnable workflow examples in documentation.
- Shortened eigensolver backend dispatch examples and repeated-run handle
  positioning while retaining AUTO dispatch rules, backend meanings, memory
  class descriptions, and handle lifecycle/ownership notes.

## Declaration Preservation Notes

- The cleanup is documentation-only inside public headers.
- Doxygen file briefs, parameter docs, return docs, notes, warnings, ownership
  language, and lifecycle cleanup requirements remain present for edited API
  surfaces.
- The public documentation set already owns the fuller workflow path:
  `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`,
  `docs/solver_selection.md`, `docs/api_reference.md`, and
  `examples/README.md`.

## Validation

Because public headers changed, Day 11 requires the full C/header quality gate:

- `make format`
- `make lint`
- `make test`

API documentation validation should also be run because generated API inputs
changed:

- `make api-docs-validate`

Validation result: passed.
