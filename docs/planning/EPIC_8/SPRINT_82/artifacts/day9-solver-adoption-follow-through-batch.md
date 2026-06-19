# Sprint 82 Day 9 - Solver Adoption Follow-Through Batch

Date: 2026-06-19  
Branch: sprint-82

## Purpose

Widen the first optional dense-backend seam beyond the Cholesky CSC lane by
landing one bounded LDL^T backend/runtime follow-through batch, while
preserving the builtin backend as the default product path and keeping the
existing scalar-prepass / supernodal fallback story intact.

## Main Result

The bounded Day 9 backend-adoption batch landed in:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_internal.h`
- `src/sparse_ldlt_csc_supernodal.c`
- `tests/test_ldlt.c`

The main implementation result is now explicit:

- the shipped builtin LDL^T dense block factor remains the default path
- the LDL^T dense-factor owner now recognizes one bounded runtime selection
  knob:
  - `SPARSE_LDLT_DENSE_BACKEND=accelerate`
- on Darwin only, that knob can activate an optional Accelerate-backed dense
  block-factor path for the LDL^T CSC supernodal lane
- the widened runtime seam stays bounded and fallback-safe:
  - if the optional backend is unavailable, the builtin backend remains active
  - if the optional backend cannot preserve the existing BK pivot/block
    contract, it returns `SPARSE_ERR_PIVOT_REJECTED`
  - the LDL^T CSC lane therefore preserves the resolved scalar-prepass /
    supernodal fallback story instead of publishing mismatched pivot metadata

## Landed Ownership

### Dense-factor owner

- `src/sparse_chol_csc.c`
  - still owns the shipped builtin `ldlt_dense_factor(...)`
  - now owns the bounded runtime selector for the LDL^T dense block-factor
    seam
  - now publishes:
    - `ldlt_dense_factor_selected(...)`
    - `ldlt_dense_factor_backend_name()`
  - now exposes one bounded Darwin-only Accelerate-backed LDL^T dense-factor
    path when the runtime probe succeeds

### Internal contract owner

- `src/sparse_chol_csc_internal.h`
  - now documents the runtime-selected LDL^T dense-factor contract
  - now makes the backend-name query explicit for family-local proof

### LDL^T CSC consumer owner

- `src/sparse_ldlt_csc_supernodal.c`
  - now consumes the runtime-selected LDL^T dense-factor path instead of the
    builtin-only helper

### Proof owner

- `tests/test_ldlt.c`
  - owns builtin env-selection proof
  - owns accelerate env-selection proof
  - owns small dense correctness checks through the selected LDL^T factor path
  - owns one public forced-CSC LDL^T factor/solve proof that the widened
    backend seam preserves the solver-visible contract

## Contract Reconciliation

The Day 8 design estimated `src/sparse_ldlt.c` as part of the required Day 9
center, but the landed implementation stayed narrower than that estimate:

- no public LDL^T wrapper dispatch change was actually needed
- the widened runtime selector could be inserted below the public dispatch
  layer
- this kept the batch bounded to the dense-factor owner, the LDL^T CSC
  supernodal consumer, and the family-local proof surface

That is still within the Day 8 fence because:

- the target seam was LDL^T backend/runtime parity
- the public reading did not widen enough to force header or docs churn
- the batch preserved the builtin default path and the existing fallback story

## Preserved Fence

The Day 8 fence held:

- no QR or SVD widening occurred
- no package/platform convergence reopened
- no shared-library maturity or platform-parity claim widened
- no benchmark threshold/gate work was added
- no broad backend framework rewrite was introduced
- no benchmark or docs spill was needed beyond the sprint record and proof

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 448.79 sec`

Focused proof stayed clean inside the broader pass:

- `test_ldlt` passed the new builtin env-selection contract
- `test_ldlt` passed the new accelerate env-selection contract
- the public forced-CSC LDL^T solve proof preserved residual correctness under
  the widened selector seam

## Exit State

- Sprint 82 now has one real optional accelerated dense-factor slice on the
  LDL^T CSC lane, not only on the Cholesky lane.
- The builtin backend remains the default shipped path.
- Optional runtime selection is now bounded, proof-backed, and fallback-safe
  across both direct-family lanes that currently matter most.
