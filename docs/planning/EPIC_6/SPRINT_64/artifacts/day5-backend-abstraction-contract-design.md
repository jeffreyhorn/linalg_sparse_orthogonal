# Sprint 64 Day 5: Backend Abstraction Contract Design

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Define the bounded backend abstraction contract for Sprint 64’s selected first
hot path so the implementation batch starts from a real kernel boundary,
fallback contract, and touched-file fence instead of a generic “performance
backend” aspiration.

## Abstraction Decision

### 1. The first abstraction stays local to the Cholesky CSC supernodal lane

The first Sprint 64 backend-aware landing should stay centered on:

- `src/sparse_chol_csc_supernodal.c`

That lane is already the selected first target because it combines:

- bounded touched surface
- strong existing proof homes
- real runtime relevance
- explicit scalar/fallback neighbors

The live code also shows that the hottest local kernels are still lane-owned:

- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`

That means the first abstraction should be:

- local to the Cholesky CSC supernodal path
- internal
- explicitly bounded

It should not attempt to introduce a repository-wide universal dense backend
layer in Sprint 64.

## Ownership Split

### 2. Local kernel ownership remains in the supernodal Cholesky lane

The selected lane should continue to own:

- supernode extract
- diagonal-block factor
- panel solve
- writeback
- local fallback and error semantics tied to the supernodal CSC path

This keeps the first landing aligned with the actual runtime hotspot rather
than prematurely forcing the repository into a general backend framework.

### 3. `src/sparse_dense.c` is a bounded support seam, not the new global backend center

The generic dense helper layer belongs inside the first landing only as bounded
support.

Current live generic helpers:

- `dense_gemm(...)`
- `dense_gemv(...)`

Current live selected-kernel ownership:

- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`

Therefore the first landing contract is:

- touch `src/sparse_dense.c` only if it materially helps the selected
  Cholesky CSC kernel abstraction
- do not treat Sprint 64 as a repo-wide relocation of dense math into one file
- do not widen into QR/SVD-wide dense unification

## Compatibility and Fallback Contract

### 4. The self-contained default build remains authoritative

The first backend-aware landing must preserve:

- the self-contained default build as the authoritative path
- current fallback correctness as the truth surface
- scalar CSC and existing supernodal semantics as the comparison anchors

Optional backend-aware acceleration may be added only if it remains:

- bounded
- internal-first
- correctness-preserving
- easy to disable or leave unused in the default build

### 5. Public option widening is not justified by default

Sprint 64’s first landing does not automatically justify a new public control
surface.

The preferred control order is:

1. existing default path remains authoritative
2. internal dispatch or build-time enablement if needed
3. public option widening only if the selected kernel path truly requires it

This preserves the Epic 6 constraint that backend work should not create fake
product maturity or unnecessary API expansion.

## Proof and Telemetry Contract

### 6. The proof home is already exact enough

The first implementation batch should prove itself through the existing
surfaces:

- family-local correctness:
  - `tests/test_chol_csc.c`
- public lifecycle/non-regression:
  - `tests/test_integration.c`
- benchmark proof:
  - `benchmarks/bench_chol_csc.c`

This means Sprint 64 does not need:

- a new backend test harness
- a new benchmark governance framework
- broader proof widening just to start the first landing

## Day 6-10 Touched-File Fence

### 7. Exact first implementation fence

The Day 6-10 first implementation fence is now:

- required implementation seam:
  - `src/sparse_chol_csc_supernodal.c`
- likely bounded support seam:
  - `src/sparse_dense.c`
- likely CSC wrapper/dispatch seam only if required by the landed contract:
  - `src/sparse_chol_csc.c`
- required proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
- likely build/options surfaces only if the selected abstraction truly needs
  them:
  - `CMakeLists.txt`
  - `Makefile`

## Explicit Non-Goals

The first Sprint 64 landing should not widen into:

- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- default public API/header widening
- packaging/platform work
- broad benchmark-governance redesign
- repository-wide dense backend generalization
- threading-policy generalization beyond the selected kernel path

## Exit State

Sprint 64 now has an explicit backend abstraction contract before code moves:

- the first abstraction stays local to the Cholesky CSC supernodal lane
- `src/sparse_dense.c` is only a bounded support seam
- default-build and fallback correctness remain authoritative
- the first proof home is fixed to `test_chol_csc`, `test_integration`, and
  `bench_chol_csc`
- the Day 6-10 touched-file fence is explicit before implementation begins
