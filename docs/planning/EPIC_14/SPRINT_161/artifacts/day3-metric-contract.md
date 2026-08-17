# Day 3 Metric Contract

## Summary

Day 3 finalizes the metric contract for the selected Sprint 161 comparison
family, `partial_svd_diag6_k2`. The contract is narrow by design: it publishes
fixture-local top-k singular-value agreement plus bounded residual,
orthogonality, and diagonal-safe projector diagnostics without claiming raw
singular-vector identity or external-library parity.

## Selected Target Contract

| Field | Value |
| --- | --- |
| Target name | `partial-svd-diag6-k2` |
| Fixture key | `partial_svd_diag6_k2` |
| Matrix shape | 6x6 diagonal |
| Requested rank | 2 |
| Reference helper | `tests/svd_external_dense_reference.py` |
| Reference top-k values | `9`, `6` |
| Output directory | `build/comparison/partial_svd_diag6_k2` |
| Study path | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| Report family | `comparison` |
| Subfamily | `partial_svd_diag6_k2` |
| Operation | `partial_svd` |
| Support tier | `local_only` |
| Claim scope | fixture-local partial-SVD diagonal top-k comparison only |

## Required Selected Rows

These row IDs are the selected generated comparison set for this family. Day 4
and later implementation work should preserve the names unless the contract is
explicitly revised.

| Row ID | Metric | Expected/Baseline | Tolerance | Status Rule | Claim Role |
| --- | --- | --- | --- | --- | --- |
| `comparison_partial_svd_diag6_k2_project_status_v1` | `project_status` | `pass` | exact | `pass` only | Claim-bearing readiness row |
| `comparison_partial_svd_diag6_k2_baseline_status_v1` | `baseline_status` | `pass` | exact | `pass` only | Claim-bearing reference row |
| `comparison_partial_svd_diag6_k2_singular_value_0_v1` | `singular_value_0` | `9` | `1e-10` absolute | `pass` only | Claim-bearing value row |
| `comparison_partial_svd_diag6_k2_singular_value_1_v1` | `singular_value_1` | `6` | `1e-10` absolute | `pass` only | Claim-bearing value row |
| `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1` | `singular_values_max_abs_delta` | `0` | `1e-10` absolute max | `pass` only | Claim-bearing aggregate row |
| `comparison_partial_svd_diag6_k2_residual_norm_v1` | `residual_norm` | `0` ideal | `1e-10` upper bound | `pass` only | Claim-bearing residual row |
| `comparison_partial_svd_diag6_k2_u_orthogonality_v1` | `u_orthogonality` | `0` ideal | `1e-10` upper bound | `pass` only | Diagnostic selected row |
| `comparison_partial_svd_diag6_k2_v_orthogonality_v1` | `v_orthogonality` | `0` ideal | `1e-10` upper bound | `pass` only | Diagnostic selected row |
| `comparison_partial_svd_diag6_k2_u_projector_diag_v1` | `u_projector_diag` | `0` ideal | `1e-10` upper bound | `pass` only | Diagnostic selected row |
| `comparison_partial_svd_diag6_k2_v_projector_diag_v1` | `v_projector_diag` | `0` ideal | `1e-10` upper bound | `pass` only | Diagnostic selected row |

The selected family therefore contributes `10` required generated comparison
rows. The normalizer should reject freshness for this family if any row is
missing, stale, duplicate, unexpected, malformed, deferred, skipped, or
non-passing.

## Metric Semantics

| Metric | Semantics |
| --- | --- |
| `project_status` | The local project partial-SVD probe built and ran successfully for the selected fixture. |
| `baseline_status` | The source-controlled dense helper produced exactly two top-k reference singular values. |
| `singular_value_0` | Project largest singular value compared with helper value `9`. |
| `singular_value_1` | Project second singular value compared with helper value `6`. |
| `singular_values_max_abs_delta` | Maximum absolute delta across the selected two singular values. |
| `residual_norm` | Maximum selected-vector residual norm for the computed top-k result, expressed as a scalar upper-bound metric. |
| `u_orthogonality` | Left selected-vector orthogonality diagnostic. This checks bounded orthogonality, not vector identity. |
| `v_orthogonality` | Right selected-vector orthogonality diagnostic. This checks bounded orthogonality, not vector identity. |
| `u_projector_diag` | Diagonal-fixture projector diagnostic for the left selected subspace. This is not a broad subspace claim. |
| `v_projector_diag` | Diagonal-fixture projector diagnostic for the right selected subspace. This is not a broad subspace claim. |

## Tolerance Contract

| Tolerance Class | Value | Applies To |
| --- | --- | --- |
| Status equality | exact string match | `project_status`, `baseline_status` |
| Singular-value absolute tolerance | `1e-10` | `singular_value_0`, `singular_value_1`, `singular_values_max_abs_delta` |
| Residual upper bound | `1e-10` | `residual_norm` |
| Orthogonality upper bound | `1e-10` | `u_orthogonality`, `v_orthogonality` |
| Diagonal projector upper bound | `1e-10` | `u_projector_diag`, `v_projector_diag` |
| Freshness | current source commit and selected row set | generated comparison rows in `study.tsv` |
| Row identity | exact `comparison_row_id` match | all selected rows |

Day 5 implementation can choose the exact emitted numeric formatting, but the
normalizer and tests should compare parsed numeric values rather than relying
on string identity for floating-point fields.

## Row-State Semantics

| State | Interpretation |
| --- | --- |
| `pass` | Required for every selected row. |
| `fail` | Fails selected freshness and blocks the claim. |
| `defer` | Non-proof context only; fails selected freshness if emitted for a selected row. |
| `skip` | Non-proof context only; fails selected freshness if emitted for a selected row. |
| missing row | Fails selected freshness and should name the missing row ID. |
| unexpected row | Fails selected freshness for the selected family until metadata and tests are updated. |
| duplicate row | Fails selected freshness, even if duplicate rows are passing. |
| stale row | Fails selected freshness when source commit, generated timestamp policy, or manifest freshness does not match the selected run contract. |
| malformed row | Fails selected freshness and should report the parse or schema field issue. |

Optional dependency rows may be emitted as separate context rows if the runner
keeps the Sprint 160 dependency-status pattern, but they are not selected
comparison rows and cannot substitute for the ten required rows.

## Claim-Bearing Versus Diagnostic Rows

Claim-bearing rows:

- `project_status`
- `baseline_status`
- `singular_value_0`
- `singular_value_1`
- `singular_values_max_abs_delta`
- `residual_norm`

Diagnostic selected rows:

- `u_orthogonality`
- `v_orthogonality`
- `u_projector_diag`
- `v_projector_diag`

The diagnostic rows are still required for freshness once selected. They
support bounded fixture-local interpretation, but public documentation should
avoid converting them into broad subspace, vector identity, or repeated-spectrum
claims.

## Convergence, Fail-Closed, And Recovery

`partial_svd_diag6_k2` is selected as the first passing comparison family.
Sprint 161 should not use this family to publish:

- convergence-rate superiority;
- fail-closed behavior;
- recovery after tight iteration budgets;
- partial-result guarantees after failure;
- repeated-spectrum ordering;
- sparse-output or drop-tolerance optimality.

Those behaviors remain covered by existing corpus and C proof-owner tests where
applicable, and broader generated comparison publication should be deferred
until the first passing comparison family is stable.

## Raw-Vector And Ordering Non-Claims

The metric contract intentionally avoids rows that compare raw singular-vector
components against a dense reference. It also avoids any claim that vector signs,
orientations, or basis ordering match another implementation. Top-k singular
values are ordered by singular value only; this does not create a raw vector
ordering claim.

## Implementation Handoff

Day 4 should design the runner and metadata extension around this fixed row
set. Implementation should stop if a required metric cannot be produced without
changing the claim boundary, or if the comparison would require raw
singular-vector identity or external-package parity.

## Validation

Day 3 is documentation-only. Validation is limited to Markdown hygiene checks
for Sprint 161 planning files.
