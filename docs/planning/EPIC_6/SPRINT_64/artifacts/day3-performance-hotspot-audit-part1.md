# Sprint 64 Day 3: Performance Hotspot Audit, Part 1

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Reduce the broad Sprint 64 “performance backend architecture” claim to a
ranked live seam map before choosing the first bounded implementation target.

## Ranked Audit

### 1. The Cholesky CSC supernodal dense-kernel lane is the strongest first target

The live Cholesky CSC supernodal path now carries the strongest first-phase
backend leverage:

- `src/sparse_chol_csc_supernodal.c` owns the full batched supernodal flow:
  - extract
  - diagonal-block factor
  - panel solve
  - writeback
- the densest hot computations already sit behind compact local helpers:
  - `chol_dense_factor`
  - `chol_dense_solve_lower`
- the path already has strong proof and measurement surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`

Why this ranks first:

- it combines real runtime payoff with a bounded touched surface
- the scalar CSC and linked-list neighbors already provide nearby fallback
  baselines
- the benchmark/proof story is mature enough to support a Phase 1 backend-aware
  landing without new framework work

### 2. LDL^T supernodal follow-through is the strongest second target

The LDL^T CSC supernodal path is also backend-worthy, but it is more complex
and more correctness-sensitive as a first landing:

- `src/sparse_ldlt_csc_supernodal.c` mirrors the same extracted dense-panel
  strategy
- but it couples the dense block path to:
  - Bunch-Kaufman pivot structure
  - `D` / `D_offdiag` / `pivot_size` ownership
  - stricter writeback and threshold semantics
- it already has strong family-local proof and benchmark support:
  - `tests/test_ldlt_csc.c`
  - `benchmarks/bench_ldlt_csc.c`

Why this ranks second:

- it is still a strong backend seam
- but its pivot-state complexity and proof burden are both higher than the
  Cholesky lane
- it should follow a successful first abstraction landing rather than define
  that landing

### 3. `src/sparse_dense.c` is an important internal seam, but not the whole Phase 1 story

The generic dense helper layer is real architecture, but it is currently
narrower than the supernodal hotspot story:

- it owns:
  - `dense_gemm`
  - `dense_gemv`
- those helpers already have clear low-level proof in `tests/test_dense.c`
- the strongest supernodal hot logic still lives locally in the CSC kernels

Why this ranks third:

- it matters as an internal seam for a bounded backend layer
- but the right first move is not turning it into a repo-wide universal backend
  hub
- broad dense-kernel unification across QR/SVD would widen the sprint too
  early

### 4. Build and threading seams are real, but should remain subordinate to the selected kernel path

The live repo shows build and threading sensitivity mainly through:

- `CMakeLists.txt`
- `Makefile`
- existing `SPARSE_OPENMP` build-time switches
- benchmark and README wording around backend and dispatch behavior

Why this does not rank first:

- these seams matter, but they should follow the selected kernel path
- starting from OpenMP or a generic parallel-backend layer would widen the
  sprint too early
- the self-contained default-build contract still needs to stay authoritative

### 5. QR and SVD remain later backend candidates

The live QR and SVD sources still expose dense-kernel opportunities:

- `src/sparse_qr.c`
- `src/sparse_svd.c`

Why they rank later:

- the first benchmark/proof home is less tightly focused than the CSC
  supernodal lane
- broad dense-kernel unification there would immediately widen the abstraction
  surface
- fallback and public-story consequences are broader than the first CSC landing

## Proof Surface Ranking

The existing proof burden already has a natural home:

1. `tests/test_chol_csc.c`
2. `benchmarks/bench_chol_csc.c`
3. `tests/test_ldlt_csc.c`
4. `benchmarks/bench_ldlt_csc.c`
5. `tests/test_dense.c`
6. `tests/test_integration.c`

Implication:

- Sprint 64 does not need a new backend test framework
- the first landing can be proved through existing CSC family-local tests plus
  the maintained benchmark surfaces
- `tests/test_dense.c` is the natural low-level proof home if the bounded
  abstraction touches generic dense helpers

## Day 4 Target

The exact Day 4 rerank target is now fixed:

1. confirm the Cholesky CSC supernodal lane as the first selected landing
2. keep LDL^T supernodal follow-through as the strongest second target
3. separate build/options wiring from the first kernel choice
4. keep QR/SVD in the later lane unless the rerank reveals a lower-risk seam

## Exit State

Sprint 64 now has a ranked live hotspot map instead of a generic backend
architecture backlog:

- the Cholesky CSC supernodal dense-kernel lane is the strongest first target
- LDL^T supernodal follow-through is the strongest second target
- `src/sparse_dense.c` is an important internal seam, but not a universal
  backend hub yet
- build/threading work is real but should follow the selected kernel path
- QR/SVD remain later backend candidates rather than first-phase targets
