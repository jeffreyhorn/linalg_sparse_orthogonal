# Sprint 64 Day 7: Kernel Integration Landing Design

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Convert the Sprint 64 backend abstraction and build/options design into the
exact touched-file and proof plan for the first code batch.

## First Code-Batch Center

### 1. Day 8 should stay centered on the Cholesky CSC supernodal kernel file

The first code batch should keep one clear implementation center:

- `src/sparse_chol_csc_supernodal.c`

That preserves the Sprint 64 Phase 1 boundary:

- selected hot path
- bounded touched surface
- internal-first architecture change
- preserved default self-contained path

### 2. Support seams are conditional, not automatic

The first batch may touch:

- `src/sparse_dense.c`
- `src/sparse_chol_csc.c`

But only if the landed kernel path proves they are actually needed:

- `src/sparse_dense.c` only for bounded helper support
- `src/sparse_chol_csc.c` only for a minimal dispatch/contract bridge

The first batch should not widen beyond that just to make the sprint look more
architectural.

## Proof Split

### 3. Family-local correctness belongs in `tests/test_chol_csc.c`

The first backend-aware kernel proof should live primarily in:

- `tests/test_chol_csc.c`

That is the natural home for:

- kernel-local equivalence
- fallback-preserve behavior
- supernodal error-path behavior
- helper-level contract tightening if the selected batch changes it

### 4. Public non-regression belongs in `tests/test_integration.c` only if needed

The public proof surface remains:

- `tests/test_integration.c`

But the first landing should only use it for the smallest required public
contract proof if the landed semantics actually cross the family-local
boundary.

This avoids turning the first kernel batch into a broad public lifecycle
rewrite.

### 5. Benchmark proof belongs in `benchmarks/bench_chol_csc.c`

The benchmark proof surface is already good enough for the first landing:

- linked-list baseline
- CSC scalar comparison lane
- CSC supernodal comparison lane
- comparable factor/solve timing columns
- residual checks

Therefore Day 8 does not need a new benchmark harness or benchmark format.

## Minimum Viable Fallback Contract

### 6. The first landing only needs bounded fallback preservation

The first backend-aware landing should preserve:

- the authoritative self-contained default path
- scalar CSC and existing supernodal correctness where the same public workflow
  applies
- explicit failure behavior rather than silent drift

This means the first batch should prefer:

- bounded equivalence checks
- bounded fallback-preserve checks
- bounded error-path checks

It should not try to solve every future backend-selection question in Sprint
64 Day 8.

## Day 8 Fence

### 7. Exact first implementation fence

Required implementation seam:

- `src/sparse_chol_csc_supernodal.c`

Optional bounded support seam only if the code proves it necessary:

- `src/sparse_dense.c`

Optional dispatch/bridge seam only if required:

- `src/sparse_chol_csc.c`

Required proof surface:

- `tests/test_chol_csc.c`

Optional bounded public proof only if needed:

- `tests/test_integration.c`

## Day 9-12 Follow-Through Queue

### 8. The later queue is now bounded before code moves

Day 9:

- post-landing safety audit
- remaining selection/fallback/error-path proof rerank

Day 10:

- smallest required build/dispatch follow-through only if Day 8 proves it
  necessary

Day 11:

- bounded benchmark proof refresh in `benchmarks/bench_chol_csc.c`

Day 12:

- docs/maintainer truth follow-through only after landed semantics are real

## Explicit Non-Goals

The first code batch should not widen into:

- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- public header widening
- broad benchmark README rewriting
- packaging/platform work
- threading-policy generalization

## Exit State

Sprint 64 now has an exact first code-batch plan:

- Day 8 centers on `src/sparse_chol_csc_supernodal.c`
- `src/sparse_dense.c` and `src/sparse_chol_csc.c` are conditional only
- regression proof and benchmark proof are explicitly separated
- the minimum viable fallback-preserve contract is fixed
- the Day 9-12 follow-through queue is bounded before implementation begins
