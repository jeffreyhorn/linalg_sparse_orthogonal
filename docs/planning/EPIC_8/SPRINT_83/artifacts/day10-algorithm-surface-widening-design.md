# Sprint 83 Day 10: Algorithm-Surface Widening Design

## Goal

Fix the exact solver-family or algorithm-surface widening seam justified by
the landed Sprint 83 scalar/index work, while preserving the bounded
real-only capability reading and avoiding broad family drift.

## Inputs Rechecked

- `docs/planning/EPIC_8/PROJECT_PLAN.md` Sprint 83 section
- `docs/planning/EPIC_8/SPRINT_83/PLAN.md`
- Sprint 83 Day 6 and Day 9 artifacts
- `include/sparse_matrix.h`
- `include/sparse_types.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `tests/test_qr.c`
- `tests/test_svd.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `README.md`
- `docs/maintainer_guide.md`

## Design Outcome

Sprint 83 now has one exact Day 11 algorithm-surface widening contract:

- required Day 11 center:
  - `include/sparse_qr.h`
- strongest support-only proof if the header wording truly forces movement:
  - `tests/test_qr.c`
- strongest support-only wording if the contract truly forces movement:
  - `README.md`
  - `docs/maintainer_guide.md`
- strongest explicit non-touch surfaces for this batch:
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `src/sparse_qr.c`

## Why QR Is The Next Seam

- The Day 6 landing widened the shared matrix-shell helper seam onto
  `sparse_scalar_t`.
- The Day 9 landing reconciled `include/sparse_types.h` so the shared
  vocabulary owner reads consistently across the matrix-shell and
  iterative/eigs public scalar seams.
- After those two landings, `include/sparse_qr.h` is the strongest remaining
  high-value public algorithm header that still exposes caller-owned numeric
  buffers and helper outputs entirely in raw `double` terms.
- QR is the highest-value next family because it is both:
  - a direct public solve lane
  - the clearest bounded algorithm follow-through after the shared-owner work

## Why Not SVD Or Direct-Family Work First

- `include/sparse_svd.h` remains a narrower real-only surface, but it is
  lower-value than QR for the next bounded move because it is not the first
  caller-facing solve lane that naturally follows the Day 6 / Day 9 owner
  widening.
- `include/sparse_cholesky.h` and `include/sparse_ldlt.h` remain non-touch
  because Sprint 83 has not yet widened the direct-family public numeric story
  in a way that forces their public contract to move now.
- Reopening direct-family wording here would create family drift without
  stronger product value than the QR lane.

## Day 11 Fence

- The intended Day 11 move is a bounded QR public-header interpretation batch.
- The first target is caller-facing QR header wording and type exposure, not
  an implementation rewrite in `src/sparse_qr.c`.
- Support surfaces move only if the QR header contract truly forces them.
- No SVD, Cholesky, LDL^T, package/install/export, complex-scalar, or
  mixed-precision spill should be implied by this batch.

## Validation

This was a design-only Day 10 pass. I re-read the live touched public-owner
surfaces, public family headers, and strongest proof-owner/support-only
surfaces. No code or support-surface validation rerun was required.
