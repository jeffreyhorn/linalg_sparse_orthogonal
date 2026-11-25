# Sprint 227 Day 2: Frozen Arithmetic and Permutation Contract

## Decision

Sprint 227 selects **pre-mutation rejection**, not logical-permutation arithmetic. `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` will accept only original-state operands: the matrix must not report a factorized state, and row, inverse-row, column, and inverse-column maps must all be identity. Any selected input outside that state returns `SPARSE_ERR_BADARG` after pointer and shape checks but before output allocation, cache/factor invalidation, scaling, insertion, or other mutation.

This is the compatible safety policy because the public header already says these APIs operate in physical index space and warns callers not to use nonidentity permutations. The change converts an unenforced precondition that currently produces misleading success into the repository's established invalid-argument result. It does not add logical arithmetic, reset maps automatically, reconstruct pre-factorization values, or broaden support for factored matrices.

The contract is frozen for Days 3–14. A change requires an explicit amendment to this artifact, the fixture matrix, and affected estimates before implementation continues.

## Scope and terminology

An **original-state matrix** satisfies all of these conditions:

1. `sparse_factor_state_is_factored(mat)` is false;
2. every `row_perm[i]` and `inv_row_perm[i]` equals `i`;
3. every `col_perm[j]` and `inv_col_perm[j]` equals `j`.

The implementation should use the existing `sparse_matrix_require_original_state` seam from `src/sparse_matrix_state_internal.h`, which checks factor state and all four maps. Day 3 must add explicit inverse-map-only and factored-with-identity-map fixtures so a weaker row/column-only guard cannot pass accidentally.

`reorder_perm` metadata alone is not a logical coordinate map and is not a separate rejection condition when factor state is false and all four maps are identity. Corrupted internal matrices, missing internal arrays, and manually inconsistent permutations are outside the public construction contract; tests may mutate internals only to create valid map pairs or narrowly prove each guard branch.

## Error precedence

### `sparse_add`

1. If A, B, or `C_out` is NULL, return `SPARSE_ERR_NULL`. Because all three pointers are validated together, a valid `C_out` storage location is unchanged when A or B is NULL.
2. After all pointers validate, set `*C_out=NULL`.
3. If dimensions differ, return `SPARSE_ERR_SHAPE`.
4. If either input is not original-state, return `SPARSE_ERR_BADARG`.
5. Only then allocate and compute the result; allocation failure returns `SPARSE_ERR_ALLOC` with `*C_out=NULL`.

### `sparse_add_inplace`

1. If A or B is NULL, return `SPARSE_ERR_NULL` without mutation.
2. If dimensions differ, return `SPARSE_ERR_SHAPE` without mutation.
3. If either input is not original-state, return `SPARSE_ERR_BADARG` without changing values, maps, cached norm, or factor state.
4. If `A == B`, return `SPARSE_ERR_BADARG` with A unchanged. Snapshot alias semantics are not silently inferred from an API that mutates its first operand while borrowing its second.
5. Only after all validation may A's cache/factor state be invalidated and arithmetic begin. An allocation/insertion failure after that point may leave A partially updated under the existing public contract, but A must remain structurally valid, reusable, and destroyable; B remains unchanged.

### `sparse_matmul`

1. If C is NULL, return `SPARSE_ERR_NULL`.
2. Set `*C=NULL` before validating A and B, preserving the existing output contract.
3. If A or B is NULL, return `SPARSE_ERR_NULL`.
4. If inner dimensions differ, return `SPARSE_ERR_SHAPE`.
5. If either input is not original-state, return `SPARSE_ERR_BADARG`.
6. Only then allocate workspaces/output; any allocation failure returns `SPARSE_ERR_ALLOC` with `*C=NULL`.

Thus pointer errors precede shape errors, and shape errors precede original-state errors. Coefficients do not weaken the precondition: `alpha == 0` or `beta == 0` does not permit a nonidentity/factored operand, because validation must not vary with arithmetic shortcuts.

## Success, ownership, and output state

| API | Input ownership | Success output | Maps and factor state | Cache |
| --- | --- | --- | --- | --- |
| `sparse_add` | A and B borrowed and unchanged; `A == B` is supported | Fresh caller-owned C; caller frees with `sparse_free` | Identity row/inverse-row/column/inverse-column; unfactored | Invalid/uncomputed (`cached_norm < 0`) |
| `sparse_add_inplace` | Distinct B borrowed and unchanged; A caller-owned destination | A contains the result | Identity maps retained; prior factor state cleared after validation | Invalidated after validation |
| `sparse_matmul` | A and B borrowed and unchanged; square `A == B` is supported | Fresh caller-owned C; caller frees with `sparse_free` | Identity row/inverse-row/column/inverse-column; unfactored | Invalid/uncomputed (`cached_norm < 0`) |

Output parameters are storage locations, not garbage collectors. If a caller passes a variable already holding another matrix, clearing that variable does not free the prior matrix. Tests must retain a second owner when checking stale-output behavior and prove that the old matrix remains untouched.

## Numerical and threshold boundary

For finite stored values and finite scalars, the logical oracle is:

- add: `C(i,j) = alpha*A(i,j) + beta*B(i,j)`;
- in-place add: `A_after(i,j) = alpha*A_before(i,j) + beta*B(i,j)` with distinct A and B;
- matmul: `C(i,j) = sum_k A(i,k)*B(k,j)` in the implementation's documented accumulation order.

Computed entries with magnitude below `1e-15` are omitted. Magnitude exactly `1e-15` is retained. Inputs are structurally sparse; an absent entry is exact zero. Duplicate coordinates are not expected inside a valid `SparseMatrix`.

The Sprint 227 conformance claim is limited to finite inputs/scalars. These APIs currently expose no uniform NaN/Inf status contract, and matmul already has separate nonfinite regression behavior. This sprint will not claim generalized nonfinite arithmetic semantics or add a new numerical error code. Existing nonfinite tests must continue to pass, and the sorted-add refactor must not accidentally broaden the finite-domain claim.

## Alias contract

- `sparse_add(A,A,alpha,beta,&C)` is supported because both inputs are read-only and output is separate.
- `sparse_matmul(A,A,&C)` is supported for square original-state A because output is separate.
- `sparse_add_inplace(A,A,alpha,beta)` is rejected with `SPARSE_ERR_BADARG` before cache/factor invalidation or value changes.
- The output pointer variable must not be the caller's only retained owner of an input matrix. The function does not and cannot recover a pointee overwritten by its documented output initialization.

## In-place failure contract

Sprint 227 preserves the existing post-validation partial-update allowance rather than promising transactional rollback. This is bounded as follows:

1. NULL, shape, original-state, and self-alias rejection are failure-atomic and leave A unchanged.
2. After validation, A may have been scaled and may contain a prefix of B updates when insertion allocation fails.
3. On such failure, A's row and column links, `nnz`, maps, cache marker, and factor-state fields must remain internally consistent; A must support lookup, destruction, copy, and a documented retry scenario.
4. B must remain byte/logically unchanged.

Day 10 may strengthen this to transactional behavior only if implementation and allocation evidence justify it; the sprint cannot claim atomicity while this frozen minimum is in force.

## Guard implementation design

Day 4 should include `sparse_matrix_state_internal.h` in `src/sparse_matrix.c` and call the existing `sparse_matrix_require_original_state` helper for each operand. No new public symbol or ABI field is required.

Pseudocode placement:

```text
sparse_add:
    validate A, B, C_out together
    *C_out = NULL
    validate equal shape
    require original state for A and B
    allocate and merge

sparse_add_inplace:
    validate A and B
    validate equal shape
    require original state for A and B
    reject A == B
    invalidate A and mutate

sparse_matmul:
    validate C; set *C = NULL
    validate A and B
    validate inner shape
    require original state for A and B
    allocate and multiply
```

One helper invocation per operand is intentional. It keeps the policy consistent with ILU, IC, bidiagonalization, selected SVD, QR, and analysis paths that already reject original-state violations with `SPARSE_ERR_BADARG`.

## Frozen fixture outcomes

The [Day 2 contract fixture matrix](day2-contract-fixture-matrix.tsv) maps pointer, shape, state, alias, allocation, and success cases to exact status and mutation/output expectations. The Day 1 numerical fixtures resolve as follows:

| Day 1 fixture class | Frozen outcome |
| --- | --- |
| Identity, finite, shape-compatible add/in-place/matmul rows | `SPARSE_OK`; match logical dense oracle and threshold rules. |
| Any row/column/inverse-map permutation on either operand | `SPARSE_ERR_BADARG`; out-of-place output NULL after valid pointers; in-place A unchanged. |
| In-place self alias | `SPARSE_ERR_BADARG`; A unchanged. |
| Empty stored operand with positive dimensions | `SPARSE_OK`; dense zero behavior. |
| Wide seeded pair | Identity variant succeeds against oracle; any selected valid map permutation is rejected unchanged. |

## Acceptance criteria for Days 3–5

1. Tests construct valid row-only, column-only, combined, inverse-only, and factor-state violations on the left, right, and both operands.
2. Every rejected out-of-place call returns `SPARSE_ERR_BADARG`, allocates nothing observable, leaves inputs unchanged, and follows the API-specific output-clearing rule.
3. Every rejected in-place call preserves values, maps, cache/factor metadata, `nnz`, and links.
4. Shape-plus-state cases return `SPARSE_ERR_SHAPE`; null cases retain their current API-specific output behavior.
5. Identity cases continue to pass existing tests; add and matmul same-input out-of-place calls remain supported; in-place self alias is rejected.
6. Public docs replace warning-only wording with the original-state precondition and `SPARSE_ERR_BADARG` result without claiming logical-permutation support.

## Claim boundary

This contract closes one caller-safety gap for three matrix arithmetic entry points. It does not validate arbitrary corrupted internals, implement logical arithmetic over permutation maps, restore an original matrix from factors, make in-place failures transactional, define universal NaN/Inf behavior, or establish performance improvement. Those boundaries remain visible in tests and public wording.
