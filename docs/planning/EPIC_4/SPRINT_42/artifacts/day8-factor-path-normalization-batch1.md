# Sprint 42 Day 8 Artifact: Factor-Path Normalization Batch I

## Scope

Day 8 implemented the first direct normalization batch chosen on Day 7:

- LU one-shot matrix path
- Cholesky one-shot matrix path
- bounded Cholesky CSC writeback/publication alignment

The batch stayed intentionally narrow:

- strengthen the private factor-state seam as the authoritative internal
  publication path
- remove remaining small compatibility-publication drift in the touched LU /
  Cholesky paths
- bring the CSC Cholesky writeback path onto the same publication seam
- preserve the current public matrix API and solve behavior

This is internal seam normalization, not a public lifecycle redesign.

## Delivered Code Changes

### 1. Expanded private factor-state helper layer

Added the next small helper slice to the private seam in:

- `src/sparse_matrix_internal.h`
- `src/sparse_factor_state_internal.c`

New helpers:

- `sparse_factor_state_begin_lu(...)`
- `sparse_factor_state_begin_cholesky(...)`
- `sparse_factor_state_replace_reorder_perm(...)`
- `sparse_factor_state_publish_factored(...)`

These helpers now centralize the touched ownership/publication chores that were
previously split across several factor-entry and writeback paths:

- bind the right private factor-state kind
- clear stale factored state at factor-entry
- replace the owned reorder permutation safely
- publish final factored-state + cached factor norm through one seam

### 2. LU direct-path normalization

Touched file:

- `src/sparse_lu.c`

Day 8 changes:

- LU factor entry now starts through `sparse_factor_state_begin_lu(...)`
- touched reorder-permutation replacement now uses
  `sparse_factor_state_replace_reorder_perm(...)`

Result:

- LU's private factor-state seam is now the clear internal start/publish path
  for the touched entry logic
- direct compatibility-field management drift was reduced without changing the
  public one-shot LU API

### 3. Cholesky direct-path normalization

Touched file:

- `src/sparse_cholesky.c`

Day 8 changes:

- linked-list Cholesky factor entry now starts through
  `sparse_factor_state_begin_cholesky(...)`
- touched reorder-permutation replacement now uses
  `sparse_factor_state_replace_reorder_perm(...)`
- the CSC dispatch path now also binds/resets the private Cholesky factor-state
  seam before symbolic/numeric CSC work begins

Result:

- both linked-list and CSC Cholesky routes now enter the touched lifecycle
  publication path through the same private seam
- this closes the most visible Day 7 mismatch between the direct Cholesky path
  and the CSC writeback path

### 4. Cholesky CSC writeback alignment

Touched file:

- `src/sparse_chol_csc.c`

Day 8 changes:

- writeback precondition now uses the shared Day 6 original-state helper
  instead of a separate handwritten factored/permutation check
- empty and non-empty writeback completion now publish through
  `sparse_factor_state_publish_factored(...)` instead of writing:
  - `reorder_perm`
  - `factor_norm`
  - `factored`
  individually by hand

Result:

- the CSC Cholesky backend no longer bypasses the Day 5 internal factor-state
  seam on its final publication step
- writeback still preserves the same external matrix result format and solve
  behavior

## What Stayed Intentionally Unchanged

Day 8 did **not** attempt:

- public API changes in `include/sparse_lu.h` or `include/sparse_cholesky.h`
- broader QR / SVD / LDLT ownership refactors
- `sparse_factors_t` bridge normalization
- cancellation-semantics rewriting
- wider CSC backend redesign

That keeps the batch aligned with the Day 7 landing order:

- Day 8 = direct LU / Cholesky normalization
- Day 9 = bridge normalization
- Day 10 = cancellation / mutation contract cleanup

## Compatibility Boundary Preserved

The important Sprint 40 / Sprint 42 rules still hold:

- public one-shot LU / Cholesky APIs are unchanged
- solve-side behavior is unchanged
- reorder/unpermute behavior is unchanged
- matrix compatibility fields remain present
- `SparseMatrix` is still the caller-facing wrapper surface

The Day 8 change is that the touched internal ownership/publication path is now
more consistent and less ad hoc.

## Validation

Because `*.c` / `*.h` changed, the required full gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

### Validation note

The first Day 8 gate attempt failed during compile, not runtime:

- `sparse_cholesky_factor_opts(...)` reused `payload_err` in the CSC path
  without a local declaration

That was fixed immediately, and the full required gate was rerun from the top.
The authoritative rerun passed completely.

## Day 8 Outcome

Sprint 42 now has a cleaner direct LU / Cholesky lifecycle publication path:

- LU and Cholesky factor entry both start through dedicated private seam
  helpers
- touched reorder-permutation ownership now routes through one private helper
- the CSC Cholesky writeback path now publishes factored state through the same
  seam as the direct path

That is the intended Day 8 handoff into Day 9:

- bounded `sparse_factors_t` bridge normalization
- only small LDLT / CSC follow-ons if they directly support that bridge work
