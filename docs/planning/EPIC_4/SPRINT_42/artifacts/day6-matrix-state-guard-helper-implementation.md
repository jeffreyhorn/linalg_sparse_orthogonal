# Sprint 42 Day 6 Artifact: Matrix-State Guard Helper Implementation

## Scope

Day 6 implemented the shared matrix-state guard seam designed on Day 4 and
landed the first live adoption set. The batch stayed deliberately narrow:

- add one private helper header for shared lifecycle-state checks
- migrate the first low-risk original-state and factored-state guard users
- preserve current user-visible error semantics
- leave algorithm-specific validation local

This is implementation normalization, not a public lifecycle-contract rewrite.

## Delivered Code Changes

### 1. New private guard-helper seam

Added:

- `src/sparse_matrix_state_internal.h`

The helper seam currently provides:

- `sparse_matrix_has_identity_row_col_perms(...)`
- `sparse_matrix_has_identity_perms(...)`
- `sparse_matrix_require_original_row_col_state(...)`
- `sparse_matrix_require_original_state(...)`
- `sparse_matrix_require_factored_state(...)`

The helpers are intentionally header-only and private. Day 6 did not introduce
new public APIs or new public lifecycle terminology.

### 2. First original-state adoption set

The new shared original-state helpers now cover the first bounded adoption set:

- `src/sparse_analysis.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_bidiag.c`
- `src/sparse_ilu.c`
- `src/sparse_ic.c`

This removed repeated bespoke checks for combinations of:

- not already factored
- identity row permutations
- identity column permutations
- in selected cases, identity inverse permutations

### 3. First factored-state adoption set

The shared factored-state helper now covers the touched solve-side checks in:

- `src/sparse_cholesky.c`
- `src/sparse_lu.c`

This keeps the solve/condest/block-solve entry checks consistent with the new
private factor-state seam introduced on Day 5.

### 4. Cholesky original-state normalization

Day 6 also normalized the linked-list Cholesky factor entry path onto:

- `sparse_matrix_require_original_state(...)`

This is slightly stricter than the row/column-only helper because that path
still depends on the full original-matrix permutation state, including the
inverse-permutation arrays.

## Shared vs Local Boundary Preserved

Day 6 intentionally **did not** turn all validation into generic helpers.

The new shared seam owns:

- original-state required
- identity row/column permutations required
- factored-state required

Algorithm-specific checks remain local, including:

- symmetry / SPD requirements
- shape and dimension checks
- reorder-mode checks
- tolerance / numerical-threshold logic
- symbolic or storage-structure assumptions

This matches the Day 4 design boundary and avoids creating a vague
catch-all validation layer.

## User-Visible Semantics Preserved

The important compatibility rule held:

- bad lifecycle state still returns `SPARSE_ERR_BADARG`

Day 6 did not change:

- public function signatures
- documented lifecycle requirements
- caller cleanup rules
- factorization algorithms
- cancellation behavior

The batch only removes guard drift and routes the touched families through one
private interpretation of the existing lifecycle contract.

## Validation

Because `*.c` / `*.h` changed, the required full gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

Unlike Day 5, this batch did not require an authoritative clean-tree rerun.
The standard gate passed directly after the Day 6 implementation batch.

## Day 6 Outcome

Sprint 42 now has a real shared matrix-state validation seam in live code:

- original-state checks are no longer duplicated across the first-wave
  lifecycle-sensitive families
- factored-state checks now align better with the new Day 5 internal
  factor-state ownership seam
- the Sprint 40 compatibility rule is preserved:
  - internal-first
  - no public API churn
  - stable user-visible error semantics

That is the intended handoff into the next Sprint 42 work:

- broader factor-path landing decisions
- cancellation-contract normalization
- compatibility bridge planning around `sparse_factors_t`
