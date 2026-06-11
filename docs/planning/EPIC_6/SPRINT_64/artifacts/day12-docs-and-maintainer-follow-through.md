# Sprint 64 Day 12: Docs and Maintainer Follow-Through

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Close the remaining bounded public-header, README, and maintainer-story gaps
after the Sprint 64 backend-aware Cholesky CSC landing.

This batch exists to make three already-landed facts read coherently across
maintained surfaces:

- the benchmark proof surface identifies the active dense-kernel descriptor
- the public error taxonomy now includes `SPARSE_ERR_BACKEND_CONTRACT`
- the first backend-aware lane is still intentionally narrow and default-safe

## Landed Surfaces

Public/API-local wording:

- `include/sparse_cholesky.h`

Top-level user-facing interpretation:

- `README.md`

Maintainer policy home:

- `docs/maintainer_guide.md`

## Main Result

The public Cholesky header now documents the actual Sprint 64 contract:

- `sparse_cholesky_factor_opts(...)` can return
  `SPARSE_ERR_BACKEND_CONTRACT`
- that code is reserved for the CSC supernodal dense-kernel seam when a
  required internal descriptor or callback cannot be resolved

The top-level docs and maintainer surface now align with that contract:

- `README.md` ties the benchmark proof surface and the backend-aware lane
  together where the Cholesky CSC dispatch story is already taught
- `docs/maintainer_guide.md` now owns the bounded interpretation:
  - Sprint 64 is not a general backend framework
  - `bench_chol_csc` is the maintained proof surface for this lane
  - `SPARSE_ERR_BACKEND_CONTRACT` should stay narrow and truthful

## Why This Was the Right Day 12 Slice

By the start of Day 12, the repo already had:

- the backend-aware dense-kernel descriptor seam
- family-local proof for missing-descriptor and missing-callback failures
- benchmark-side path measurability for `scalar`, `supernodal`, and `builtin`

The remaining risk was interpretive drift:

- the public error code existed
- the benchmark proof existed
- but the public header and maintainer-facing policy surface did not yet tell
  readers how those pieces fit together

So the right follow-through was:

- header-local truth
- bounded README alignment
- explicit maintainer ownership

Not:

- more implementation work
- more benchmark modes
- broader backend marketing

## Validation

Ran:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Exit State

Sprint 64 Day 12 now leaves:

- one coherent API-local explanation of `SPARSE_ERR_BACKEND_CONTRACT` on the
  Cholesky CSC supernodal lane
- one coherent top-level and maintainer-facing explanation of the current
  backend-aware proof surface
- a smaller residual queue for Day 13 validation and Day 14 closeout
