# Sprint 75 Day 6 Artifact: Design Freeze & Proof Map

Date: 2026-06-17
Branch: sprint-75

## Purpose

Freeze the exact Day 7 implementation and proof ownership map before the first
Sprint 75 backend edits begin.

## Main Result

Sprint 75 now has one exact Day 7 touch set:

- implementation center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- first regression/fallback proof owner:
  - `tests/test_chol_csc.c`
- likely public-path support proof only if truly forced:
  - `tests/test_integration.c`
- benchmark proof owner:
  - `benchmarks/bench_chol_csc.c`
- support-only follow-through:
  - `include/sparse_cholesky.h`
  - `docs/maintainer_guide.md`

## Ownership Map

### Implementation owners

- `src/sparse_dense.c`
  - dense-kernel descriptor and shipped default implementation
- `src/sparse_chol_csc_supernodal.c`
  - supernodal consumption of the dense-kernel seam
  - narrow `SPARSE_ERR_BACKEND_CONTRACT` boundary
- `src/sparse_chol_csc.c`
  - CSC dispatch, orchestration, and compatibility-shell publication

### Proof owners

- `tests/test_chol_csc.c`
  - family-local backend, fallback, and dense-kernel contract proof
- `tests/test_integration.c`
  - only if a caller-facing public-path guarantee actually changes
- `benchmarks/bench_chol_csc.c`
  - benchmark-visible path identity and dense-kernel descriptor identity

## Day 7 Fence

The Day 7 batch should:

- stay inside the three-file implementation center
- add or update proof first in `tests/test_chol_csc.c`
- touch `tests/test_integration.c` only if the public-path contract truly
  changes
- touch `benchmarks/bench_chol_csc.c` only if the landed backend behavior
  needs new benchmark-visible measurement
- avoid widening into eigs, QR, SVD, or broad docs/governance spill

## Support-Only Reading

The first landing keeps these surfaces support-only unless the code forces
movement:

- `include/sparse_cholesky.h`
  - only for local public truth changes
- `docs/maintainer_guide.md`
  - only for bounded policy/ownership clarification

## Day 8 Audit Rubric

After the Day 7 landing, the post-landing audit should answer:

- did the first batch close the strongest dense-kernel/backend-owner seam
- is the strongest remaining seam now:
  - callback/runtime parity
  - benchmark proof refresh
  - residual support-surface drift
- did any support-only surface actually need to move
- does eigs still rank as the strongest second landing

## Exit State

Day 7 can now proceed from:

- one exact implementation center
- one fixed benchmark and regression/fallback proof map
- one support-only follow-through list
- one explicit Day 8 rerank rubric
