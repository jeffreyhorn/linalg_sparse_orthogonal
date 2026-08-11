# Sprint 151 Day 4: Comparison Contract Design

## Purpose

Define the comparison semantics for the Sprint 151 partial-SVD corpus families
before fixture metadata, expected rows, proof-owner tests, and oracle/report
integration expand the maintained corpus surface.

This contract keeps the sprint claim fixture-local and rejects raw singular
vector identity, sign parity, orientation parity, phase parity, and arbitrary
basis-order parity as evidence.

## Existing Oracle Conventions

The current corpus oracle already supports these comparison paths:

| Comparison Kind | Expected Tolerance | Current Use | Sprint 151 Use |
| --- | --- | --- | --- |
| `rank` | `exact` with `0` | QR rank-deficient corpus rows | Rank-deficient partial-SVD fixture rank. |
| `nullity` | `exact` with `0` | QR rank-deficient corpus rows | Not selected for Day 4 rows; rank-deficient fixture may mention nullity only as fixture metadata. |
| `value` | `absolute` | `top_k`, solution norms, solution values | Partial-SVD `top_k` singular-value rows. |
| `subspace_distance` | `projector` | QR nullspace and partial-SVD repeated-spectrum projector rows | Rank-deficient left/right range projectors. |
| `residual_norm` | `absolute` | QR residuals and partial-SVD triplet/orthogonality rows | Triplet residuals, orthogonality, Frobenius-error scalars. |
| `status` | `status_only` with empty tolerance | Solver success and non-convergence rows | Default success, sparse-output success, tight-budget non-convergence, recovery success. |
| `diagnostic` | `not_applicable` with empty tolerance | No partial arrays on failure | Tight-budget no-partial-array fail-closed row. |

The sparse-output fixture needs a small oracle extension later in the sprint
because current `value` comparison only handles `top_k`, `solution_norm`, and
`solution_values`. Day 4 defines that extension as fixture-local sparse-output
fields, not as broad sparse-output correctness.

## Selected Family Row Contract

### Rank-Deficient Rectangular Range Projector

Fixture key:

```text
partial_svd_rankdef_diag6x4_k2_range_projector_v1
```

Contract rows:

| Row Suffix | Operation | Comparison | Expected Result | Tolerance | Claim |
| --- | --- | --- | --- | --- | --- |
| `default_status` | `partial_svd_default` | `status` / `status` | `SPARSE_SUCCESS` | `status_only` / empty | Default partial-SVD succeeds for the named fixture. |
| `singular_values` | `singular_values` | `value` / `value` | `top_k=9,6` | `absolute` / `1e-8` | Top-2 singular values match after descending sort. |
| `rank` | `rank_info` | `rank` / `rank` | `2` | `exact` / `0` | Fixture rank is exactly 2 at the corpus rank tolerance. |
| `left_subspace` | `singular_subspace` | `subspace_distance` / `subspace_distance` | `left_projector_distance<=1e-8` | `projector` / `1e-8` | Computed left selected subspace matches the coordinate range projector. |
| `right_subspace` | `singular_subspace` | `subspace_distance` / `subspace_distance` | `right_projector_distance<=1e-8` | `projector` / `1e-8` | Computed right selected subspace matches the coordinate range projector. |
| `vector_residuals` | `vector_residuals` | `residual_norm` / `residual_norm` | `max_triplet_residual<=1e-8` | `absolute` / `1e-8` | Selected triplets satisfy `A v ~= sigma u` and `A^T u ~= sigma v`. |
| `orthogonality` | `orthogonality` | `residual_norm` / `residual_norm` | `max_orthogonality_residual<=1e-8` | `absolute` / `1e-8` | Selected U and V columns remain orthonormal. |

This family does not compare raw U or V entries. The projector rows compare
subspace membership only, so the contract is stable under sign changes and
valid orthonormal-basis rotations.

### Sparse Low-Rank Output

Fixture key:

```text
partial_svd_lowrank_rect5x7_k3_sparse_output_v1
```

Contract rows:

| Row Suffix | Operation | Comparison | Expected Result | Tolerance | Claim |
| --- | --- | --- | --- | --- | --- |
| `sparse_status` | `sparse_lowrank` | `status` / `status` | `SPARSE_SUCCESS` | `status_only` / empty | Sparse low-rank output succeeds for the named fixture at `drop_tol=0`. |
| `sparse_shape` | `sparse_lowrank` | `diagnostic` / `diagnostic` | `shape=5x7` | `not_applicable` / empty | Output matrix shape is retained exactly. |
| `sparse_nnz` | `sparse_lowrank` | `rank` / `rank` | `3` | `exact` / `0` | Fixture-local retained nonzero count is 3 at `drop_tol=0`. |
| `sparse_selected_values` | `sparse_lowrank` | `value` / `value` | `selected_values=8,4,2,0` | `absolute` / `1e-10` | Coordinates `(0,0)`, `(1,1)`, `(2,2)`, and `(3,3)` match expected retained/zeroed values. |
| `dense_frobenius_error` | `lowrank_reconstruction` | `residual_norm` / `residual_norm` | `dense_frobenius_abs_error<=1e-10` | `absolute` / `1e-10` | Dense rank-3 low-rank reconstruction error equals the omitted tail norm for this fixture. |
| `sparse_dense_frobenius_diff` | `sparse_lowrank_consistency` | `residual_norm` / `residual_norm` | `sparse_dense_frobenius_diff<=1e-10` | `absolute` / `1e-10` | Sparse output matches dense low-rank output for this fixture at `drop_tol=0`. |

The `sparse_shape` row intentionally uses diagnostic equality because shape is
not a floating-point comparison. The `sparse_selected_values` row requires a
Sprint 151 oracle extension to parse `selected_values` as a vector, analogous
to existing `solution_values` handling. That extension must remain generic
enough for this row but must not introduce broad sparse-output claims.

The `dense_frobenius_error` expected result uses an absolute-error residual
threshold so the existing residual comparator can compare the observed scalar:
the observed value must be `abs(measured_dense_error - 1.0)`.

### Non-Repeated Convergence Fail-Closed

Fixture key:

```text
partial_svd_fail_closed_diag6_k2_v1
```

Contract rows:

| Row Suffix | Operation | Comparison | Expected Result | Tolerance | Claim |
| --- | --- | --- | --- | --- | --- |
| `tight_budget_status` | `convergence_budget` | `status` / `status` | `SPARSE_ERR_NOT_CONVERGED` | `status_only` / empty | `max_iter=1` fails closed for the named non-repeated fixture. |
| `tight_budget_no_partial_arrays` | `diagnostic` | `diagnostic` / `diagnostic` | `no_partial_sigma_u_vt_on_failure` | `not_applicable` / empty | Tight-budget failure does not publish partial `sigma`, `U`, or `Vt` arrays. |
| `recovery_status` | `convergence_budget` | `status` / `status` | `SPARSE_SUCCESS` | `status_only` / empty | A default-budget run succeeds after a prior tight-budget failure attempt. |
| `default_singular_values` | `singular_values` | `value` / `value` | `top_k=9,6` | `absolute` / `1e-8` | Default-budget top-2 singular values match after descending sort. |
| `default_vector_residuals` | `vector_residuals` | `residual_norm` / `residual_norm` | `max_triplet_residual<=1e-8` | `absolute` / `1e-8` | Default-budget selected triplets satisfy residual bounds. |

This family does not claim convergence rate, portable iteration counts, or
useful partial outputs after non-convergence. It proves only the selected
fixture's tight-budget failure mode and recovery path.

## Singular-Value Semantics

Singular-value rows compare selected top-k singular values as finite doubles
with absolute tolerance. The expected and observed `top_k` vectors are sorted
descending before comparison, matching current oracle behavior and avoiding
tie-order claims.

Selected Sprint 151 singular-value tolerances:

| Fixture | Expected `top_k` | Tolerance |
| --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | `9,6` | `1e-8` |
| `partial_svd_fail_closed_diag6_k2_v1` | `9,6` | `1e-8` |

The sparse-output fixture does not use a `top_k` singular-value row because
its claim is about low-rank output behavior rather than raw partial-SVD output
ordering.

## Projector And Subspace Semantics

Projector comparisons use subspace distance, not vector equality. For the
rank-deficient rectangular fixture:

- the expected left selected subspace is the coordinate range spanning rows
  `0` and `1`;
- the expected right selected subspace is the coordinate range spanning
  columns `0` and `1`;
- each projector distance must be at most `1e-8`.

This permits valid sign flips and orthonormal-basis changes while still
proving the selected singular subspaces are correct for the fixture.

## Vector Residual And Orthogonality Semantics

Vector-residual rows compare helper-defined maximum triplet residuals:

- `A v_i ~= sigma_i u_i`;
- `A^T u_i ~= sigma_i v_i`;
- maximum over selected `i < k`.

Orthogonality rows compare the maximum selected-column orthogonality residual
for U and V. The Sprint 151 bound for triplet residuals and orthogonality is
`1e-8`, matching the existing partial-SVD corpus lane and owner-local tests.

These rows reject raw-vector identity. They prove numerical consistency of
selected triplets and selected bases, not stable sign, orientation, or basis
order.

## Sparse-Output Semantics

Sparse-output rows are fixture-local and apply only to
`partial_svd_lowrank_rect5x7_k3_sparse_output_v1` at `drop_tol=0`.

The expected diagonal fixture has values `8,4,2,1,0` and rank target `k=3`.
The maintained comparison must prove:

- sparse low-rank construction returns `SPARSE_SUCCESS`;
- output shape remains `5x7`;
- retained nonzero count is exactly `3`;
- selected output coordinates are `(0,0)=8`, `(1,1)=4`, `(2,2)=2`, and
  `(3,3)=0`;
- dense low-rank reconstruction error equals `1.0` within `1e-10`, encoded
  as `dense_frobenius_abs_error<=1e-10`;
- sparse output matches dense output with Frobenius difference at most
  `1e-10`.

The row does not claim storage optimality, drop-tolerance optimality,
performance, broad rectangular low-rank behavior, or broad sparse-output
correctness.

## Convergence And Fail-Closed Semantics

Convergence rows are status-only except for the default-budget residual proof.
The fail-closed fixture must prove:

- tight budget `max_iter=1` returns `SPARSE_ERR_NOT_CONVERGED`;
- tight-budget failure does not publish partial `sigma`, `U`, or `Vt` arrays;
- a default-budget run can recover to `SPARSE_SUCCESS`;
- default-budget singular values and triplet residuals match the declared
  tolerance.

The comparison contract does not infer convergence rate, deterministic
iteration count, platform-independent iteration budgets, or any useful partial
result after non-convergence.

## Required Oracle Extension

Day 5 and Day 11 should keep most rows within current oracle semantics. The
only required comparator extension from this Day 4 contract is:

| Field | Extension |
| --- | --- |
| `selected_values` | Treat as a comma-separated numeric vector under `comparison_kind=value`, parallel to existing `solution_values` behavior, with optional `max_abs_error` validation when the observed row reports it. |

No extension is required for status, rank, projector, residual, or diagnostic
rows.

## Completion Check

Day 4 is complete when:

- each selected family has concrete expected-result kinds and tolerances;
- singular-value comparisons use sorted top-k values and avoid tie-order
  overclaims;
- projector comparisons use subspace distance instead of raw vector equality;
- vector residual and orthogonality rows prove numerical consistency without
  sign or orientation assumptions;
- sparse-output rows are bounded to selected fixture coordinates and
  dense/sparse consistency;
- convergence rows prove tight-budget fail-closed behavior without convergence
  rate or partial-result claims.
