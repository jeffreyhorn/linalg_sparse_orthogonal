# Sprint 64 Day 9: Post-Landing Safety Audit and Proof Re-Rank

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Re-rank the remaining Sprint 64 backend queue from the landed Day 8 branch
state instead of the pre-landing design, and fix the exact Day 10-12
follow-through from the live dense-kernel integration seam.

## Reviewed Surfaces

Implementation and proof surfaces reviewed:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`

Truth surfaces rechecked:

- `include/sparse_cholesky.h`
- `README.md`
- `docs/maintainer_guide.md`

## Main Finding

After the Day 8 landing, Sprint 64 no longer has a broad “introduce a bounded
backend abstraction” problem.

The strongest remaining queue now reduces to:

1. internal fallback/error-path truthfulness on the new dense-kernel seam
2. family-local proof expansion for that seam
3. conditional benchmark observability follow-through only if the Day 10
   landing creates a real output distinction
4. conditional docs/maintainer truth follow-through only if the Day 10
   contract materially sharpens the current story

## What Changed Since the Day 7 Design

### 1. The first backend-aware integration seam is already real

The landed Day 8 branch already has:

- one bounded internal descriptor:
  - `chol_dense_kernels_t`
  - `chol_csc_supernodal_dense_kernels()`
- one authoritative builtin implementation in `src/sparse_dense.c`
- one selected consumer in `src/sparse_chol_csc_supernodal.c`
- one family-local contract proof in `tests/test_chol_csc.c`

That means the opening Sprint 64 problem is no longer “how do we introduce a
backend-aware seam without widening the public surface?”

That part is already done.

### 2. The strongest remaining gap is the internal fallback/error-path contract

The landed implementation now contains an explicit error path for:

- missing dense-kernel descriptor
- missing dense-kernel function pointer

But the current branch still carries one truthfulness mismatch:

- the Day 8 notes/artifact already treat that path as a distinct
  backend-contract failure lane
- the live code still returns `SPARSE_ERR_BADARG`

This is now the highest-value remaining Sprint 64 seam because it sits exactly
on the new integration point and determines how the library classifies failure
when the internal backend contract is violated.

### 3. Benchmark proof is not yet the next blocker

`benchmarks/bench_chol_csc.c` still already gives a useful first benchmark
proof surface:

- linked-list baseline
- CSC scalar lane
- CSC supernodal lane
- timing columns
- residual checks

What it does not yet report is the internal dense-kernel descriptor name.

That is real future observability work, but it is not yet the strongest Day 10
target because the current branch still has only one authoritative builtin
descriptor and no external selection surface.

### 4. Public docs/header follow-through is now conditional, not automatic

The Day 8 landing did not create a new public contradiction:

- no public backend selector moved
- no public lifecycle rule changed
- no public benchmark/docs claim currently depends on the dense-kernel
  descriptor name

So the next follow-through should stay internal-first until the final error and
fallback contract lands.

## Exact Day 10 Target

The next implementation batch should answer one bounded question:

- what is the final shipped error/fallback contract when the supernodal
  Cholesky lane cannot resolve the Day 8 dense-kernel descriptor it now
  depends on?

Exact Day 10 touched-file fence:

- required:
  - `src/sparse_chol_csc_supernodal.c`
  - `tests/test_chol_csc.c`
- likely support:
  - `src/sparse_chol_csc_internal.h`
  - `src/sparse_dense.c`
- explicitly not required unless the code proves otherwise:
  - `CMakeLists.txt`
  - `Makefile`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - public headers
  - top-level docs

## Intended Day 10 Proof Shape

The next proof should stay family-local:

- keep the main proof burden in `tests/test_chol_csc.c`
- add a bounded override seam only if necessary to simulate:
  - missing descriptor
  - missing `factor` pointer
  - missing `solve_lower` pointer
- prove the supernodal path returns the final intended internal error code
  explicitly rather than relying on the builtin descriptor always being present

## Explicit Non-Targets

- no public runtime/backend selector growth
- no build-option widening unless the internal proof seam truly forces it
- no LDL^T/QR/SVD spillover
- no benchmark-governance redesign
- no broad docs cleanup while the remaining internal contract still moves

## Exit State

Sprint 64 Day 9 closes with a materially smaller and more concrete queue:

- the abstraction problem is already solved
- the strongest remaining seam is now the internal fallback/error-path
  contract on the new dense-kernel lane
- benchmark and docs follow-through are conditional rather than automatic
- Day 10 can proceed from an exact touched-file fence and a consciously
  smaller proof queue
