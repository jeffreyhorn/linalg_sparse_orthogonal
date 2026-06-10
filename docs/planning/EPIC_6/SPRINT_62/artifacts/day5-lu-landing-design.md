# Sprint 62 Day 5: LU Landing Design

Date: 2026-06-10
Branch: `sprint-62`


## Purpose

Turn the Day 4 lifecycle/wrapper contract into an exact touched-file and
API/implementation boundary plan so the Day 6-7 LU hardening batch stays
bounded and does not expand into a broad direct-solver rewrite.

## Minimum Viable Public Surface

### Public surfaces to touch first

- `include/sparse_lu.h`

### Public surfaces to keep untouched in Batch 1

- `include/sparse_analysis.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`

### Public design rule

The first batch should normalize the LU one-shot contract in place rather than
widen the shared lifecycle header or normalize every direct-family header at
once.

## Internal Bridge Design

### Core bridge decision

The smallest viable implementation lane stays inside:

- `src/sparse_lu.c`

with helper/state support allowed only if the landed behavior proves it is
necessary:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_state_internal.h`

Why this matters:

- the highest-value Sprint 62 LU risk is wrapper/lifecycle crossover plus
  reorder/cancel state invalidation
- those seams already live in the LU wrapper path and factor-state helpers
- widening into `src/sparse_analysis.c` on the first batch would turn a
  usability sprint into a lifecycle-core rewrite too early

## Proof-Surface Plan

### Required proof home

- `tests/test_integration.c`

### Optional only if the landed path forces it

- `tests/test_sparse_lu.c`

### Proof design rule

The first LU batch should stay integration-led because the main risks already
show up there:

- cancellation
- reorder invalidation
- one-shot versus explicit lifecycle parity

Do not widen the batch into a mandatory LU unit-test expansion unless the
landed behavior proves an integration-only proof is too weak.

## Day 6 vs Day 7 Split

### Day 6: First LU wrapper/lifecycle hardening slice

Touch:

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

Focus:

- public wrapper wording normalization where needed
- first bounded LU wrapper/lifecycle hardening
- first bounded regression additions for the exact landed path

### Day 7: Cleanup/state-preservation follow-through only if needed

Potential touch set:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`
- small touched-file LU wording follow-through only

Focus:

- cleanup/state-preservation tightening if the Day 6 landing proves it is
  needed
- helper hardening only where it directly supports the LU batch
- regression expansion only where the Day 6 behavior exposed a real proof gap

## Exact Touched-File Plan

### Expected Day 6-7 touched files

- public:
  - `include/sparse_lu.h`
- implementation:
  - `src/sparse_lu.c`
- optional helper/state support:
  - `src/sparse_factor_state_internal.c`
  - `src/sparse_matrix_state_internal.h`
- required proof:
  - `tests/test_integration.c`
- optional proof:
  - `tests/test_sparse_lu.c`
- likely later docs follow-through after code lands:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`

### Exact non-touch set for the first landing

- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`
- broad docs simplification
- benchmark/example edits
- packaging/platform work
- configuration-surface work

## Operational Non-Goals

Do not widen the first code landing into:

- a shared lifecycle API redesign
- hidden copy semantics for LU one-shot wrappers
- Cholesky or LDL^T cleanup in the same patch
- QR expectation cleanup in the same patch
- benchmark/example modernization
- platform or packaging work

## Day 5 Exit State

Sprint 62 now has a precise LU-first implementation boundary:

- the minimum viable public surface is fixed
- the primary implementation seam is fixed
- the required proof home is fixed
- the helper/state support lane is bounded and optional
- the Day 6 versus Day 7 split is fixed
- the non-touch set is fixed before implementation edits begin
