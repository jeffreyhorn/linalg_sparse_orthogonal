# Sprint 75 Day 8 Artifact: Post-Landing Audit & Rerank

Date: 2026-06-17
Branch: sprint-75

## Purpose

Re-rank the remaining Sprint 75 queue after the Day 7 kernel landing and fix
the exact Day 9 design center.

## Main Result

The Day 7 landing closed the strongest dense-kernel/backend-owner seam.

The strongest remaining Sprint 75 seam is now:

- CSC callback/runtime parity for Cholesky

The exact Day 9 design center is:

- `include/sparse_cholesky.h`
- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

## Why The Rerank Changed

### Closed by Day 7

These are no longer the strongest contradiction center:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`

Reason:

- the dense-kernel descriptor now owns a real batched panel-solve callback
- the supernodal consumer now uses that callback directly
- the missing-callback failure path is locally proven and aligned to
  `SPARSE_ERR_BACKEND_CONTRACT`

### Now strongest

The strongest remaining seam is now the runtime/observability split across:

- `include/sparse_cholesky.h`
- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

Reason:

- the public header still says only the linked-list backend emits progress
- the CSC lane already owns `used_csc_path` and bounded backend-contract truth
- after the Day 7 kernel landing, runtime parity is the highest-value
  remaining follow-through in the same family

## Support-Only Drift

These surfaces remain support-only rather than the next batch center:

- `benchmarks/bench_chol_csc.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Notes:

- `benchmarks/bench_chol_csc.c` now has real comment drift because it still
  describes the older row-by-row panel-solve reading, but that is weaker than
  the runtime seam
- `tests/test_integration.c` should move only if the public-path runtime
  contract itself widens
- `docs/maintainer_guide.md` remains broadly truthful at the policy layer

## Explicit Non-Centers

Not the next Sprint 75 batch center:

- eigensolver backend/runtime parity
- QR backend-aware follow-through
- SVD backend-aware follow-through
- another dense-kernel descriptor expansion in `src/sparse_dense.c`

## Exit State

Day 8 closes with:

- one explicit confirmation that the first backend-aware landing closed the
  strongest dense-kernel seam
- one rerank to CSC callback/runtime parity
- one exact Day 9 design center
- one bounded support-only follow-through list
