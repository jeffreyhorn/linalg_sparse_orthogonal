# Sprint 145 Day 10 Public Header Cleanup

## Purpose

Day 10 executed the Day 9 public-header cleanup plan for the smallest
high-impact adoption batch:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`

The cleanup reduces tutorial-scale and maintainer-history comments in public
headers while preserving public API declarations, ownership rules, error
semantics, backend boundaries, and bounded evidence claims.

## Header Changes

| Header | Cleanup Applied | Contract Preserved |
| --- | --- | --- |
| `include/sparse_matrix.h` | Replaced long file-level positioning with a compact matrix-shell, allocator, compatibility, and scalar-buffer boundary summary. Shortened the CSC threshold note. | Matrix ownership, construction/import/export scope, one-shot compatibility shell, repeated-workspace routing, `SPARSE_IDX_BITS`, `double` scalar boundary, and local Cholesky CSC threshold override semantics. |
| `include/sparse_iterative.h` | Replaced long CG/GMRES/preconditioner examples with concise solver-family routing. Shortened breakdown and repeated-run handle descriptions. | Solver assumptions, caller-supplied preconditioner ownership, `sparse_iter_result_t` diagnostics, breakdown/convergence distinction, drop-tolerance threshold, and workspace reuse limits. |
| `include/sparse_qr.h` | Replaced tutorial-style overview with compact QR API contract and documentation handoff. Shortened rank-threshold guidance. | Least-squares/rank-revealing scope, identity-permutation preconditions, output buffer sizes, residual/rank diagnostics, `sparse_qr_free()`, and QR evidence boundary links. |
| `include/sparse_svd.h` | Replaced tutorial-style overview with compact SVD API contract and documentation handoff. Shortened partial-SVD corpus and low-rank outer-product notes. | SVD rank/condition scope, output ownership and sorted singular values, fixture-local partial-SVD evidence, non-claims for broad parity/performance, and default-off low-rank accumulator behavior. |

## API And ABI Preservation

- No declarations, typedefs, enums, struct fields, macros, includes, or
  function signatures were intentionally changed.
- Header edits are documentation/comment changes only.
- Ownership/freeing rules remain present for matrix, QR, and SVD objects.
- NULL/error behavior and solver diagnostic boundaries remain documented.
- Backend dispatch, threshold, and environment-variable controls remain framed
  as local policy knobs rather than portable performance guarantees.
- QR and partial-SVD evidence remains fixture-local and explicitly bounded.
- Static-first package and ABI boundaries are not expanded.

## Claim Boundary Review

The cleanup intentionally avoids adding or strengthening unsupported claims:

- no state-of-the-art performance claim
- no external-library parity claim
- no broad repeated-spectrum partial-SVD claim
- no platform parity claim
- no shared-library or ABI promise
- no package-manager support claim

Remaining mentions of broad parity/performance are explicit non-claims in the
header text.

## Validation

| Check | Result |
| --- | --- |
| Header changed-file scan | Passed; changed C/header paths are limited to the four selected public headers. |
| Declaration-change scan | Passed; no signature/declaration-like added or removed lines were found in the header diff. |
| Unsupported-claim scan | Passed; matches are limited to explicit non-claims. |
| `git diff --check` | Passed. |
| Header trailing-whitespace scan | Passed. |
| `make format` | Passed. |
| `make lint` | Passed. |
| `make test` | Passed. |

## Completion Criteria

- Selected public headers are clearer for first-use readers.
- Public API contracts, ownership rules, error semantics, backend boundaries,
  and bounded evidence claims are preserved.
- Header edits do not introduce API or ABI changes.
- Required C/header quality gates pass.

## Day 11 Handoff

Day 11 should run the cross-surface coherence pass across README, INSTALL,
examples, cookbook, solver-selection guidance, public headers, report rows, and
Sprint 139-144 evidence to confirm that the now-cleaner headers still align
with the front-door adoption route.
