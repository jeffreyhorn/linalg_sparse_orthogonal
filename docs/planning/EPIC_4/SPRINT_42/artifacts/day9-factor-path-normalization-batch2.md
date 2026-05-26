# Sprint 42 Day 9 Artifact: Factor-Path Normalization Batch II

## Scope

Day 9 implemented the bounded bridge normalization batch chosen on Day 7:

- analyze-once bridge cleanup in `src/sparse_analysis.c`
- small LDLT bridge normalization where it directly supported that cleanup
- no public installed-header changes

The batch stayed intentionally narrow:

- normalize bridge-owned payload setup
- normalize bridge-owned factor handoff
- normalize bridge-side LDLT solve-view reconstruction
- bring touched working-copy permutation cleanup onto the private factor-state
  seam
- preserve the current public `sparse_factors_t` shape and solve behavior

This is implementation-side bridge cleanup, not public lifecycle redesign.

## Delivered Code Changes

### 1. Working-copy sanitation now uses the private permutation-ownership seam

Touched file:

- `src/sparse_analysis.c`

Day 9 change:

- `sanitize_working_copy(...)` now clears reorder-permutation ownership through
  `sparse_factor_state_replace_reorder_perm(...)` instead of directly freeing
  and nulling `reorder_perm`

Result:

- the analyze-once bridge now uses the same touched permutation-ownership seam
  already established for direct LU / Cholesky lifecycle publication work

### 2. Added private bridge helpers for payload setup and handoff

Touched file:

- `src/sparse_analysis.c`

New private helpers:

- `sparse_factors_init_payload(...)`
- `sparse_factors_take_matrix_factor(...)`
- `sparse_factors_take_ldlt_factor(...)`
- `sparse_factors_make_ldlt_view(...)`

These helpers now centralize the touched bridge chores that were previously
open-coded inside `sparse_factor_numeric(...)` and `sparse_factor_solve(...)`:

- initialize bridge payload state
- transfer LU / Cholesky matrix-factor ownership
- transfer LDLT ownership
- reconstruct a temporary LDLT solve view from the bridge payload

### 3. LU / Cholesky analyze-once handoff is now centralized

Touched file:

- `src/sparse_analysis.c`

Day 9 changes:

- `sparse_factor_numeric(...)` now starts bridge payload setup through
  `sparse_factors_init_payload(...)`
- LU and Cholesky factor handoff now routes through
  `sparse_factors_take_matrix_factor(...)`

Result:

- the bridge no longer repeats small matrix-factor ownership/factor-norm
  packaging logic in each direct-factor case
- the touched LU / Cholesky analyze-once bridge path now reads more like one
  internal family

### 4. LDLT bridge transfer and solve-view reconstruction are now centralized

Touched file:

- `src/sparse_analysis.c`

Day 9 changes:

- LDLT bridge ownership transfer now routes through
  `sparse_factors_take_ldlt_factor(...)`
- `sparse_factor_solve(...)` now rebuilds the temporary LDLT view through
  `sparse_factors_make_ldlt_view(...)`

Result:

- LDLT remains compatibility-preserving in public shape
- the bridge implementation no longer spreads the touched LDLT packaging and
  solve-view reconstruction across multiple field-by-field blocks

## What Stayed Intentionally Unchanged

Day 9 did **not** attempt:

- public `sparse_factors_t` redesign
- installed-header changes in `include/sparse_analysis.h`
- broader LDLT API changes
- QR / SVD ownership changes
- cancellation / mutation contract rewriting

That keeps the batch aligned with the Day 7 landing order:

- Day 8 = direct LU / Cholesky normalization
- Day 9 = bounded analyze-once bridge normalization
- Day 10 = cancellation / mutation contract cleanup

## Compatibility Boundary Preserved

The important Sprint 40 / Sprint 42 rules still hold:

- public `sparse_factors_t` shape is unchanged
- public analyze-once entry points are unchanged
- solve-side behavior is unchanged
- LDLT still uses the same compatibility bridge object
- `SparseMatrix` remains the caller-facing wrapper surface for direct matrix
  families

The Day 9 change is that the touched bridge ownership and reconstruction path
is now more uniform and less ad hoc.

## Validation

Because `*.c` changed, the required full gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Day 9 Outcome

Sprint 42 now has a cleaner analyze-once bridge path without widening into a
public redesign:

- working-copy permutation cleanup now routes through the private factor-state
  seam
- LU / Cholesky bridge payload setup is centralized
- LDLT bridge ownership transfer is centralized
- LDLT solve-view reconstruction is centralized

That is the intended Day 9 handoff into Day 10:

- normalize cancellation and mutation semantics across the touched direct and
  bridge paths
