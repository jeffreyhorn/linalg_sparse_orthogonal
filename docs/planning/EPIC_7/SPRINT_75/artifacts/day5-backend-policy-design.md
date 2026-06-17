# Sprint 75 Day 5 Artifact: Backend / Policy Design

Date: 2026-06-17
Branch: sprint-75

## Purpose

Define the bounded implementation contract for Sprint 75's first backend-aware
landing before any code edits begin.

## Main Result

Sprint 75 now has one explicit first implementation contract:

- required implementation center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if the first batch truly forces it:
  - `include/sparse_cholesky.h`
  - `benchmarks/bench_chol_csc.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`

## Ownership Split

### Dense-kernel owner

- `src/sparse_dense.c`

Owns:

- the concrete dense-kernel descriptor
- the shipped self-contained default implementation
- the local override seam used only for proof

### Supernodal batch owner

- `src/sparse_chol_csc_supernodal.c`

Owns:

- consumption of the dense-kernel descriptor during batched supernodal
  elimination
- the local `SPARSE_ERR_BACKEND_CONTRACT` boundary when a required dense
  callback or descriptor is unavailable

### CSC-lane orchestration owner

- `src/sparse_chol_csc.c`

Owns:

- dispatch into the supernodal path
- CSC-lane runtime and publish-back truth
- compatibility-shell publication back to the caller-owned `SparseMatrix`

## Preserved Guarantees

The first batch must preserve:

- the self-contained default build remains the main product path
- the default shipped dense-kernel descriptor remains explicit and measurable
- linked-list, CSC scalar, and CSC supernodal benchmark truth stays
  like-for-like
- runtime/backend observability becomes clearer, not broader
- benchmark surfaces remain reporting/proof surfaces, not timing gates
- the one-shot Cholesky compatibility shell and publish-back story remain
  truthful and bounded

## Support-Surface Reading

Support surfaces move only if the implementation actually forces them:

- `include/sparse_cholesky.h`
  - only if the batch changes local public truth around dense-kernel
    descriptors, publish-back, or callback/runtime interpretation
- `benchmarks/bench_chol_csc.c`
  - only if the batch changes what must be benchmark-visible
- `tests/test_chol_csc.c`
  - only if backend/fallback correctness proof must move
- `tests/test_integration.c`
  - only if a caller-facing public-path guarantee truly changes
- `docs/maintainer_guide.md`
  - only if the bounded backend contract itself becomes clearer in a way the
    policy surface should capture

## Non-Touch Set

The first Sprint 75 batch explicitly does not include:

- eigensolver backend/runtime work:
  - `include/sparse_eigs.h`
  - `src/sparse_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `tests/test_eigs.c`
- QR work:
  - `include/sparse_qr.h`
  - `src/sparse_qr.c`
  - `tests/test_qr.c`
- SVD work:
  - `include/sparse_svd.h`
  - `src/sparse_svd.c`
  - `tests/test_svd.c`
  - `benchmarks/bench_svd.c`
- broad docs/governance/platform spill:
  - `README.md`
  - `benchmarks/README.md`
  - `INSTALL.md`
  - packaging or reviewed-platform workflow files
- capability-surface reopening from Sprint 74

## Day 6 Implication

The proof-map pass should now start from:

- exact Day 7 implementation center:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- support only if truly forced:
  - `include/sparse_cholesky.h`
  - `benchmarks/bench_chol_csc.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
- explicitly deferred:
  - eigs lane
  - QR lane
  - SVD lane
  - broad benchmark refresh
  - broad docs/platform spill
