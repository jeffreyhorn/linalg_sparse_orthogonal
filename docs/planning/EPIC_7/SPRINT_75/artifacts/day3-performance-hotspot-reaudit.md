# Sprint 75 Day 3 Artifact: Performance Hotspot Re-audit

Date: 2026-06-17
Branch: sprint-75

## Purpose

Re-rank the live backend/performance hotspots by actual user value,
implementation leverage, and proof cost so Sprint 75 starts from the strongest
bounded second backend-aware landing rather than from a generic architecture
wishlist.

## Inputs Reviewed

- `README.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`
- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `include/sparse_qr.h`
- `src/sparse_qr.c`
- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/bench_svd.c`
- `tests/test_chol_csc.c`
- `tests/test_eigs.c`
- `tests/test_qr.c`
- `tests/test_svd.c`

## Main Result

Sprint 75's backend/performance pressure is no longer a generic "dense kernels
and more backends" problem. It is now one ranked contradiction map:

- strongest first target:
  - CSC supernodal Cholesky dense-kernel/runtime ownership
- strongest second target:
  - eigensolver backend/runtime parity
- strongest later target:
  - QR and SVD backend-aware follow-through
- strongest cross-cutting support seam:
  - callback/runtime truth across maintained families

## Ranked Findings

### 1. CSC supernodal Cholesky is still the strongest first landing

The strongest current backend-aware seam remains concentrated in:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `tests/test_chol_csc.c`

This lane stays first because it already has the full bounded architecture
shape Sprint 75 needs:

- one concrete dense-kernel owner
- one shipped runtime descriptor (`builtin`)
- one maintained benchmark-side proof surface
- one family-local regression owner
- one already-documented truthfulness fence

That makes it the strongest current implementation-leverage seam, not just the
 most mature benchmark lane.

### 2. Eigs is the strongest second backend/runtime lane

The eigs lane is now the strongest second seam across:

- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `tests/test_eigs.c`
- `benchmarks/bench_eigs_reuse.c`

It is already a real backend-aware public/runtime surface:

- the public header exposes a backend selector
- the implementation owns shared backend selection and orchestration
- the tests already prove real backend routing and parity
- the benchmark surface already emits retained backend-aware reporting

It stays second rather than first because it is more of a runtime/callback
parity seam than a dense-kernel-ownership seam, and the CSC Cholesky lane is
 still the stronger bounded architecture center.

### 3. QR and SVD are later lanes, not the first batch center

QR and SVD still matter, but they are not the best first landing:

- `include/sparse_qr.h`
- `src/sparse_qr.c`
- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `benchmarks/bench_svd.c`
- `tests/test_qr.c`
- `tests/test_svd.c`

The useful separation is:

- QR and SVD are large and important
- they are not yet the strongest bounded backend-aware product seam
- `bench_svd.c` remains more exploratory/profiling-oriented than the canonical
  maintained benchmark owners
- the public headers still do not expose the same backend-aware surface shape
  the eigs lane already has

So QR and SVD remain later Sprint 75 targets unless the landing-boundary pass
shows that the strongest second batch should be callback/runtime cleanup
instead.

### 4. The strongest cross-cutting seam is callback/runtime truth

The strongest cross-cutting secondary seam is callback/runtime truth, not
broad benchmark governance.

This matters because:

- callback behavior is still family-local
- progress/cancellation semantics are not uniform across all maintained
  backend-aware families
- a Sprint 75 landing must preserve those asymmetries honestly rather than
  flattening them into one generic backend story

The strongest proof and wording pressure is therefore in:

- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `include/sparse_qr.h`
- `README.md`
- `docs/maintainer_guide.md`

## Day 4 Implication

The next boundary pass should treat the Sprint 75 first landing as:

- required first center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- likely proof homes:
  - `tests/test_chol_csc.c`
  - `benchmarks/bench_chol_csc.c`
- strongest second batch candidate:
  - `include/sparse_eigs.h`
  - `src/sparse_eigs.c`
  - `tests/test_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`

## Exit State

Sprint 75 now has one explicit Day 3 hotspot rerank:

- start from the strongest shipped backend-aware dense-kernel lane
- treat eigs as the strongest second runtime/backend lane
- defer QR/SVD until the first boundary is fixed
- preserve callback/runtime asymmetry truth rather than smoothing it into a
  generic backend-governance rewrite
