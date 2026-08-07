# Day 6 Partial-SVD Comparison Semantics

## Purpose

Day 6 defines how the Day 5 partial-SVD expected rows should be compared when
Day 7 adds oracle output. The design uses the existing oracle row schema and
keeps all evidence fixture-local to
`partial_svd_clustered_repeated_diag8x6_k3_v1`.

No source-controlled generated oracle rows are introduced by this artifact.
Generated rows must remain under ignored `build/` paths.

## Row Mapping

| Oracle row ID | Solver family | Operation | Comparison kind | Pass condition |
| --- | --- | --- | --- | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values` | `partial_svd` | `singular_values` | `value` | Returned top-k singular values sorted descending match `{10.0, 10.0, 9.999999}` with max absolute error `<= 1e-8`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_left_subspace` | `partial_svd` | `singular_subspace` | `subspace_distance` | Left top-k projector distance is `<= 1e-8`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_right_subspace` | `partial_svd` | `singular_subspace` | `subspace_distance` | Right top-k projector distance is `<= 1e-8`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_vector_residual` | `partial_svd` | `vector_residuals` | `residual_norm` | Maximum triplet residual across returned vectors is `<= 1e-8`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_orthogonality` | `partial_svd` | `orthogonality` | `residual_norm` | Maximum U/V orthogonality residual is `<= 1e-8`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_default_status` | `partial_svd` | `convergence_budget` | `status` | Default-budget run returns `SPARSE_SUCCESS`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_status` | `partial_svd` | `convergence_budget` | `status` | Tight-budget run returns `SPARSE_ERR_NOT_CONVERGED`. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_no_partial_arrays` | `partial_svd` | `diagnostic` | `diagnostic` | Tight-budget failure publishes no `sigma`, `U`, or `Vt` arrays. |

## Observed Result Formats

Observed results should use stable scalar or key/value text so report rows are
readable and future validators can parse them.

| Comparison kind | Expected-result format | Observed-result format | Parser rule |
| --- | --- | --- | --- |
| `value` | `top_k=10,10,9.999999` | `top_k=<s1>,<s2>,<s3>;max_abs_error=<err>` | Parse both vectors, require equal length, compare sorted descending absolute error to tolerance. |
| `subspace_distance` | `left_projector_distance<=1e-8` or `right_projector_distance<=1e-8` | `<side>_projector_distance=<distance>` | Parse the scalar distance and compare to `tolerance_value`. |
| `residual_norm` | `max_triplet_residual<=1e-8` or `max_orthogonality_residual<=1e-8` | `<metric>=<residual>` | Parse the scalar residual and compare to `tolerance_value`. |
| `status` | `SPARSE_SUCCESS` or `SPARSE_ERR_NOT_CONVERGED` | Exact status token | Require string equality; `tolerance_kind=status_only` and empty `tolerance_value`. |
| `diagnostic` | `no_partial_sigma_u_vt_on_failure` | `no_partial_sigma_u_vt_on_failure` or failure key/value details | Require exact success token when the diagnostic passes; otherwise fail with details. |

The existing QR comparison path may keep using scalar observed results for
`rank`, `nullity`, and `residual_norm`. Day 7 should avoid changing QR output
formats unless needed for shared helper extraction.

## Singular-Value Ordering

- The solver may return singular vectors with arbitrary sign and may rotate
  bases inside the repeated leading singular-value block.
- The singular-value comparison should sort observed values descending before
  comparing against the expected top-k values.
- The expected sequence remains `{10.0, 10.0, 9.999999}`. The third value is
  intentionally distinct from the repeated pair, so the comparison must not
  collapse the fixture into an unordered triple-repeat check.
- Extra singular values beyond `k=3` are out of scope for this row.
- Missing or non-finite values must fail as malformed or mismatched output.

## Subspace And Vector Semantics

Subspace comparisons should use projectors, not raw basis columns.

For returned U and V bases:

1. Build the returned top-k projector for each side.
2. Build the exact expected projector for the first three diagonal coordinates.
3. Compare the Frobenius norm or another explicitly documented projector
   distance to `1e-8`.
4. Treat sign flips and rotations inside the repeated singular-value subspace
   as valid when projector and residual rows pass.

Vector residuals should be computed from returned triplets:

```text
max_i(max(||A*v_i - sigma_i*u_i||, ||A^T*u_i - sigma_i*v_i||))
```

The residual row complements the projector rows. It catches bad triplet
pairing or scaling without requiring raw vector identity.

## Orthogonality Semantics

The orthogonality row should compute the maximum deviation from identity across
the returned U and V bases for the requested top-k factors. A passing row means
the selected fixture's returned partial-SVD bases are orthonormal within
`1e-8`; it does not prove broad basis quality for other spectra.

## Convergence-Budget Policy

| Run | Required status | Factor arrays | Evidence interpretation |
| --- | --- | --- | --- |
| Default budget | `SPARSE_SUCCESS` | `sigma`, `U`, and `Vt` must be present for value, subspace, residual, and orthogonality rows. | Counts only as fixture-local partial-SVD pass evidence if all associated rows pass. |
| Tight budget | `SPARSE_ERR_NOT_CONVERGED` | `sigma`, `U`, and `Vt` must not be published. | Counts only as fail-closed diagnostic evidence. |

If the tight-budget run unexpectedly succeeds, the tight-budget status row must
fail. If it returns a different error, the row must fail. If it returns
`SPARSE_ERR_NOT_CONVERGED` but leaves any partial factor array visible, the
diagnostic row must fail.

Non-converged output must never be used to populate singular-value, subspace,
residual, or orthogonality pass rows.

## Failure Mapping

| Condition | Comparison status | Failure class |
| --- | --- | --- |
| Numeric value, residual, or subspace metric exceeds tolerance | `fail` | `fail_oracle_mismatch` |
| Status token differs from expected status | `fail` | `fail_oracle_mismatch` |
| Diagnostic token differs or partial arrays are visible on failure | `fail` | `fail_oracle_mismatch` |
| Missing expected row, malformed observed result, non-finite metric, wrong vector length, or TSV width mismatch | `fail` | `fail_malformed_row` |
| Generated fixture hash does not match manifest metadata | `fail` | `fail_generator_mismatch` |
| Source commit, command, platform, compiler, configuration, or generated timestamp is stale for report interpretation | `fail` | `fail_report_stale` |
| Optional data unavailable | Not applicable for this fixture | No optional-data row should be emitted for the primary generated fixture. |

## Support Tier And Report Interpretation

- Initial solver-backed rows should use `support_tier=local_only` unless a
  later reviewed CI lane promotes them.
- Passing rows support only the expected row's fixture-local `claim_scope`.
- Report rows under `build/corpus-reports/` are generated evidence and must not
  be source-controlled during Sprint 140.
- A report is stale if the fixture row, generator row, expected TSV, oracle
  runner, proof owner, source commit, or command changes after generation.
- The report must preserve the Day 5 non-claims verbatim or with a strictly
  narrower boundary.

## Non-Claims

These semantics do not establish:

- broad partial-SVD correctness;
- raw singular-vector identity;
- broad repeated or clustered spectrum coverage;
- broad rectangular, nonsymmetric, rank-deficient, or near-zero behavior;
- low-rank product guarantees;
- partial-result availability after non-convergence;
- convergence-rate or performance claims;
- LAPACK, NumPy, SciPy, SuiteSparse, or broad external-library parity;
- platform, package, ABI, or state-of-the-art claims.

## Day 7 Handoff

Day 7 should implement a focused partial-SVD oracle path that can emit and
compare the eight Day 5 expected rows. The implementation should keep QR
behavior backward-compatible, keep generated outputs under `build/`, and run:

```sh
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py
```

If Day 7 touches `.c` or `.h` files, it must also run the full C quality gate.
