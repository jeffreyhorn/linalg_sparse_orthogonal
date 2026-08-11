# Sprint 151 Day 8: Proof-Owner Test Design

## Purpose

Design focused executable proof-owner coverage for the Sprint 151 partial-SVD
corpus metadata and oracle rows before Day 9 changes C test files.

The design maps every selected expected-result row to a concrete assertion
while preserving the sprint non-claims: no raw singular-vector identity, no
sign/orientation/phase parity, no arbitrary basis-order parity, no broad
partial-SVD correctness, no broad sparse-output optimality, and no convergence
rate or portable iteration-count claim.

## Current Test Surface

| File | Current Role | Day 8 Decision |
| --- | --- | --- |
| `tests/test_svd_partial_corpus.c` | Focused maintained partial-SVD corpus proof owner for the Sprint 140 clustered/repeated fixture. | Extend this file with the three Sprint 151 selected fixture families. |
| `tests/test_svd_partial_shared_helpers.h` | Shared partial-SVD residual and projector helpers. | Keep as the shared helper location; add only small reusable helpers if Day 9 duplication becomes hard to read. |
| `tests/test_svd_partial_helpers.h` | Broader partial-SVD unit/helper tests, including rank-deficient projector and fail-closed seeds. | Leave broad tests in place; use their proven assertion patterns in the focused corpus owner. |
| `tests/test_svd.c` | Broader SVD tests, including sparse low-rank dense/sparse consistency. | Leave broad test in place; promote a fixture-keyed focused variant into the corpus owner. |

Day 9 should extend `tests/test_svd_partial_corpus.c` instead of creating a
new test binary. This keeps maintained corpus proof ownership discoverable in
one place and avoids a new CMake/Makefile registration change.

## Fixture-Key Diagnostics

Every new Day 9 test should print a diagnostic line that includes the fixture
key and the observed metrics it owns. Use stable, grep-friendly prefixes:

- `partial_svd_rankdef_diag6x4_k2_range_projector_v1:`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1:`
- `partial_svd_fail_closed_diag6_k2_v1:`

Diagnostics should report numeric residuals and statuses, but they must not
print or compare raw singular-vector entries as expected evidence.

## Helper Cleanup Plan

Day 9 should keep helper changes narrow:

1. Add fixture-local constants in `tests/test_svd_partial_corpus.c` for rows,
   columns, `k`, tolerances, and fixture keys.
2. Add small local matrix builders for the three selected diagonal fixtures
   unless the file can reuse existing local builders cleanly.
3. Add a local sorted top-k sigma-error helper that accepts expected values,
   or generalize the existing clustered helper to avoid duplicating the sort
   loop.
4. Reuse `partial_svd_u_coordinate_range_projector_error`,
   `partial_svd_v_coordinate_range_projector_error`, and
   `partial_svd_max_triplet_residuals` from
   `tests/test_svd_partial_shared_helpers.h`.
5. Reuse `tf_dense_column_orthogonality_error`,
   `tf_svd_vt_row_orthogonality_error`,
   `tf_svd_dense_lowrank_frobenius_error`, and
   `tf_svd_sparse_dense_frobenius_diff` from existing SVD helper surfaces.

Do not extract a broad fixture framework on Day 9. The selected fixtures are
small enough that a few local builders and assertion helpers are clearer than
a generic test DSL.

## Test Design

### Rank-Deficient Rectangular Range Projector

Add one or two focused tests for:

```text
partial_svd_rankdef_diag6x4_k2_range_projector_v1
```

Recommended Day 9 shape:

- `test_partial_svd_corpus_rankdef_diag6x4_k2_metadata_and_values`
- `test_partial_svd_corpus_rankdef_diag6x4_k2_projectors_and_residuals`

Assertions:

- generated matrix shape `6x4`;
- `sparse_svd_rank(A, 1e-8, &rank)` returns `SPARSE_OK` and rank `2`;
- `sparse_svd_partial(A, 2, compute_uv/economy/default-budget)` returns
  `SPARSE_OK`;
- result dimensions are `m=6`, `n=4`, `k=2`;
- `sigma`, `U`, and `Vt` are non-null;
- sorted top-2 singular values match `9,6` within `1e-8`;
- left coordinate-range projector distance is at most `1e-8`;
- right coordinate-range projector distance is at most `1e-8`;
- max `A v ~= sigma u` and `A^T u ~= sigma v` residuals are at most `1e-8`;
- U and V selected-column orthogonality residuals are at most `1e-8`.

This test proves selected subspaces and triplet consistency, not raw U/V
identity or basis orientation.

### Sparse Low-Rank Output

Add one focused test for:

```text
partial_svd_lowrank_rect5x7_k3_sparse_output_v1
```

Recommended Day 9 name:

```text
test_partial_svd_corpus_lowrank_rect5x7_k3_sparse_output
```

Assertions:

- generated matrix shape `5x7`;
- `sparse_svd_lowrank(A, 3, &dense)` returns `SPARSE_OK`;
- `sparse_svd_lowrank_sparse(A, 3, 0.0, &sp)` returns `SPARSE_OK`;
- dense and sparse outputs are non-null;
- sparse output shape is `5x7`;
- retained sparse-output nonzero count is exactly `3` if a stable public or
  local `nnz` accessor is available; otherwise assert the selected retained
  and zeroed coordinates and record that `sparse_nnz` remains oracle-owned
  until a stable accessor is added;
- selected coordinates satisfy `(0,0)=8`, `(1,1)=4`, `(2,2)=2`, `(3,3)=0`
  within `1e-10`;
- dense low-rank Frobenius error equals `1.0` within `1e-10`;
- sparse-vs-dense Frobenius difference is `0.0` within `1e-10`.

This test proves the named diagonal fixture at `drop_tol=0` only. It does not
claim storage optimality, broad sparse-output correctness, drop-tolerance
optimality, or sparse-output performance.

### Non-Repeated Fail-Closed Convergence

Add one focused test for:

```text
partial_svd_fail_closed_diag6_k2_v1
```

Recommended Day 9 name:

```text
test_partial_svd_corpus_fail_closed_diag6_k2_recovery
```

Assertions:

- tight-budget `sparse_svd_partial(A, 2, max_iter=1)` returns
  `SPARSE_ERR_NOT_CONVERGED`;
- failure result retains `m=6`, `n=6`, and `k=2`;
- failure result publishes no `sigma`, `U`, or `Vt` arrays;
- `sparse_svd_free()` safely handles the failed result;
- a default-budget `compute_uv/economy` run returns `SPARSE_OK` after the
  failure attempt;
- default-budget `sigma`, `U`, and `Vt` are non-null;
- sorted top-2 singular values match `9,6` within `1e-8`;
- max triplet residuals are at most `1e-8`.

This test proves fail-closed behavior and recovery for the named fixture. It
does not claim convergence rate, portable iteration count, or useful partial
outputs after non-convergence.

## Expected-Row To Assertion Map

### Rank-Deficient Rectangular

| Expected Row | Executable Assertion |
| --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_default_status` | `sparse_svd_partial()` returns `SPARSE_OK`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_singular_values` | Sorted top-2 sigma error against `9,6` is `<= 1e-8`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_rank` | `sparse_svd_rank(A, 1e-8, &rank)` returns rank `2`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_left_subspace` | `partial_svd_u_coordinate_range_projector_error(&svd, 2) <= 1e-8`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_right_subspace` | `partial_svd_v_coordinate_range_projector_error(&svd, 2) <= 1e-8`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_vector_residuals` | Max Av and AtU residuals from `partial_svd_max_triplet_residuals()` are `<= 1e-8`. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_orthogonality` | U and V selected-column orthogonality residuals are `<= 1e-8`. |

### Sparse Low-Rank Output

| Expected Row | Executable Assertion |
| --- | --- |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_status` | `sparse_svd_lowrank_sparse()` returns `SPARSE_OK`. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_shape` | `sparse_rows(sp) == 5` and `sparse_cols(sp) == 7`. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_nnz` | Prefer exact sparse-output nnz `3`; if no stable accessor exists, verify selected retained/zeroed coordinates and keep nnz as generated-oracle-only until an accessor is added. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values` | `sparse_get()` for `(0,0)`, `(1,1)`, `(2,2)`, and `(3,3)` matches `8,4,2,0` within `1e-10`. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_dense_frobenius_error` | `tf_svd_dense_lowrank_frobenius_error()` equals `1.0` within `1e-10`. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_dense_frobenius_diff` | `tf_svd_sparse_dense_frobenius_diff()` equals `0.0` within `1e-10`. |

### Fail-Closed Convergence

| Expected Row | Executable Assertion |
| --- | --- |
| `partial_svd_fail_closed_diag6_k2_v1_tight_budget_status` | Tight-budget partial SVD returns `SPARSE_ERR_NOT_CONVERGED`. |
| `partial_svd_fail_closed_diag6_k2_v1_tight_budget_no_partial_arrays` | Failed result has `sigma == NULL`, `U == NULL`, and `Vt == NULL`. |
| `partial_svd_fail_closed_diag6_k2_v1_recovery_status` | Default-budget partial SVD after failure returns `SPARSE_OK`. |
| `partial_svd_fail_closed_diag6_k2_v1_default_singular_values` | Sorted top-2 sigma error against `9,6` is `<= 1e-8`. |
| `partial_svd_fail_closed_diag6_k2_v1_default_vector_residuals` | Max Av and AtU residuals from `partial_svd_max_triplet_residuals()` are `<= 1e-8`. |

## Affected-Test Validation Plan

Day 9 should run focused and affected tests first:

```sh
make build/test_svd_partial_corpus
./build/test_svd_partial_corpus
make build/test_svd
./build/test_svd
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Because Day 9 will change `.c` and possibly `.h` files, it must then run the
required full C gate:

```sh
make format && make lint && make test
```

If the focused corpus proof owner becomes unstable, Day 9 should stop and
either reduce the affected row to generated-reference-only evidence or ask for
direction before weakening tolerances.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Every selected claim has a focused executable proof design. | Complete | Expected-row-to-assertion maps cover all 18 Sprint 151 expected rows. |
| Raw-vector identity and sign/orientation parity remain non-claims. | Complete | Projector, residual, orthogonality, selected-coordinate, status, and diagnostic assertions avoid raw U/V equality. |
| Implementation can proceed without broad monolithic test expansion. | Complete | Day 9 extends the existing focused corpus proof owner and reuses narrow shared helpers. |
