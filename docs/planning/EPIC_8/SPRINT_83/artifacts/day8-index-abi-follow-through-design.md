# Sprint 83 Day 8: Index / ABI Follow-Through Design

## Purpose

Fix the exact shared-vocabulary and touched-path ABI follow-through contract so
Day 9 can reconcile the post-Day-6 scalar/index reading without reopening
matrix-shell implementation work or widening prematurely into later
algorithm-family capability work.

## Main Result

Sprint 83 now has one exact second implementation contract:

- required Day 9 center:
  - `include/sparse_types.h`
- strongest support-only proof if the header wording truly forces movement:
  - `tests/test_sparse_matrix.c`
- strongest support-only wording if the contract truly forces movement:
  - `README.md`
  - `docs/maintainer_guide.md`
- lower-value non-touch surfaces for this batch:
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

## Why This Is The Exact Next Seam

The exact residual contradiction is now explicit:

- `include/sparse_types.h` still describes `sparse_scalar_t` primarily as the
  iterative/eigs dense-scalar seam
- Day 6 already widened the shared matrix-shell helper seam to the same shared
  vocabulary owner
- that makes the shared vocabulary header the strongest stale touched public
  contract left behind by the first landing

This is stronger than another code batch because:

- `include/sparse_matrix.h` already says the matrix-shell helper seam routes
  through `sparse_scalar_t`
- `src/sparse_matrix.c` already preserves the intended compatibility behavior
- the remaining mismatch is interpretation and public contract ownership, not
  numeric behavior

## Preserved Day 9 Fence

Day 9 should preserve:

- the shipped scalar contract as real-only `double`
- the existing matrix-shell implementation behavior
- the current proof-owner split unless the header wording truly forces proof
  movement
- unchanged package/install/export mechanics

Day 9 should not widen into:

- QR or SVD algorithm-surface work
- Cholesky or LDL^T family-local public rewrites
- matrix-shell implementation churn
- broader package/platform maturity claims
- complex or mixed-precision claims

## Strongest Clarification

The useful Day 8 clarification is explicit:

- Day 9 is a shared-vocabulary reconciliation batch, not another capability
  implementation batch
- support surfaces only move if the `include/sparse_types.h` contract change
  truly leaves them stale
- algorithm-family widening remains the later Sprint 83 seam, not the next
  one

## Exit State

- Sprint 83 now has one exact index / ABI follow-through design contract.
- Day 9 can land one bounded shared-vocabulary reconciliation batch without
  reopening broader capability work.
- The remaining support-only and non-touch surfaces are explicitly separated
  from the required implementation center.
