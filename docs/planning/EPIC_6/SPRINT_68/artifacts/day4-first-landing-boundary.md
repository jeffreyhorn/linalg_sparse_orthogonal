# Sprint 68 Day 4: First-Landing Boundary

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Convert the Day 3 hotspot ranking into one exact first implementation fence so
Sprint 68 starts from a bounded giant-test refactor instead of a generic
multi-file cleanup set.

## Exact First Landing

The exact first landing is now fixed to:

- `tests/test_chol_csc.c`

Why this is the right first batch:

- it is the largest remaining giant test in the live tree
- it still combines multiple permanent proof roles in one file:
  - family-local CSC factorization behavior
  - dense primitive checks
  - supernodal extract/writeback and helper plumbing
  - backend-contract and dispatch proof
  - large corpus and regression lanes

So the first Sprint 68 landing should reduce the densest mixed-owner test file,
not spread evenly across every large test surface.

## Support Context, Not First-Batch Center

The first landing may rely on the already-existing family-local support surface:

- `tests/test_chol_csc_supernodal_helpers.h`

Why it stays support-only unless the design proves otherwise:

- it already reads as a local Cholesky CSC helper seam
- it allows a bounded split without widening into a generic test-framework
  abstraction
- widening beyond that immediately would blur whether Sprint 68 is still doing
  a family-local refactor or a broad testing-architecture rewrite

## Strongest Second Target, Explicitly Deferred

The strongest second target is now fixed to:

- `tests/test_reorder_nd.c`

Why it is not first:

- its strongest pressure is chronology and proof-layer density
- it has less obvious first-batch helper extraction value than
  `tests/test_chol_csc.c`
- it is better treated as the next bounded refactor lane after the first
  Cholesky CSC split is designed and landed

## Oracle Lane, Not First Refactor Lane

The shared assurance owner remains:

- `tests/test_integration.c`

Why it stays out of the first landing:

- its main value is public-path parity and oracle follow-through
- the first Sprint 68 move is family-local test-architecture reduction
- mixing the oracle owner into the first landing would blur refactor-first work
  with second-layer assurance expansion

## Explicit Non-Touch Set

The following stay outside the first landing fence:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_graph.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- `tests/test_integration.c`
- implementation `src/` files
- benchmark and maintained docs surfaces

## Ranked Order After Day 4

Sprint 68 now has one explicit implementation order:

1. exact first landing:
   - `tests/test_chol_csc.c`
2. support only if needed:
   - `tests/test_chol_csc_supernodal_helpers.h`
3. strongest second target:
   - `tests/test_reorder_nd.c`
4. explicit oracle/assurance lane:
   - `tests/test_integration.c`
5. later/deferred:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_graph.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `tests/test_svd.c`

## Exit State

Sprint 68 Day 4 closes with one exact first landing boundary:

- `tests/test_chol_csc.c` first
- `tests/test_chol_csc_supernodal_helpers.h` support only if needed
- `tests/test_reorder_nd.c` explicitly deferred to the second batch
- `tests/test_integration.c` held in the oracle lane instead of the first
  refactor lane

That gives Day 5 one exact job:

- define the bounded ownership and helper-extraction contract inside
  `tests/test_chol_csc.c`
