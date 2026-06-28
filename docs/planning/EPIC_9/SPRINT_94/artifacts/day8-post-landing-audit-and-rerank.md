# Sprint 94 Day 8 - Post-Landing Audit and Re-rank

## Scope
- Assess the validated Day 7 scalar landing against the live Sprint 94
  contradiction order
- Decide whether a second scalar landing is still justified
- Freeze the exact Day 9 design center and directly forced support surfaces

## Main Result

The Day 7 scalar landing closed the strongest first Sprint 94 contradiction:

- the shared matrix-shell helper seam no longer reads as scalar preparation
  without a matching implementation owner
- the touched matrix-shell storage/build path now follows `sparse_scalar_t`
- the first scalar capability step is now real on the highest-value touched
  owner

That means a second immediate scalar landing is not the strongest next move.

The strongest remaining contradiction is now the touched 64-bit and ABI
maturity seam:

- `SPARSE_IDX_BITS`, `idx_t`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` are already
  real public contract surfaces
- the strongest remaining work is touched-path maturity and consumer
  interpretation, especially on matrix-shell save/load and debug/inspection
  surfaces
- this is a smaller and more defensible next batch than reopening deeper
  solver-family numeric owners immediately

## Updated Contradiction Order

- strongest next target:
  - touched 64-bit and ABI maturity
- strongest later target:
  - bounded solver-family breadth only where the widened scalar/index contract
    truly needs it
- strongest support-only later target:
  - proof/docs/package wording only where the touched capability claim truly
    moves

## Exact Day 9 Design Center

- required Day 9 center:
  - `src/sparse_matrix.c`

- directly forced support-only follow-through only if the Day 9 contract truly
  needs it:
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`

## Deferred From The Next Batch

- a second immediate scalar widening on broader solver-family owners
- fake broad complex or mixed-precision maturity
- generic family-wide numeric rewriting
- broader maintainer or package wording churn detached from the touched
  index/ABI seam

## Interpretation

Sprint 94 should now move from the first scalar landing to stronger touched
index and ABI maturity on the same matrix-shell family. That keeps the sprint
inside the bounded scalar/index contract while avoiding early solver-family
claim widening.
