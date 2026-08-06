# Day 5 Partial-SVD Fixture Implementation

## Summary

Day 5 implements the source-controlled corpus skeleton for
`partial_svd_clustered_repeated_diag8x6_k3_v1`. The implementation adds the
generated fixture registry entry, fixture manifest row, generator manifest row,
and expected-result rows needed by later Sprint 140 oracle and proof-owner
work.

This is not solver pass evidence yet. The validation proves schema integrity,
known-generator hashes, row references, tolerance fields, and non-claim
boundaries.

## Implemented Files

| File | Change |
| --- | --- |
| `scripts/validate_corpus_schema.py` | Added the deterministic partial-SVD generated diagonal fixture and made known-generator parameter validation data-driven. |
| `tests/corpus/manifests/fixtures.tsv` | Added `partial_svd_clustered_repeated_diag8x6_k3_v1`. |
| `tests/corpus/manifests/generators.tsv` | Added `partial_svd_clustered_repeated_diag8x6_generator_v1` with canonical hashes. |
| `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` | Added eight expected-result rows for values, subspaces, residuals, orthogonality, and budget diagnostics. |

## Fixture Metadata

| Field | Value |
| --- | --- |
| Fixture key | `partial_svd_clustered_repeated_diag8x6_k3_v1` |
| Generator key | `partial_svd_clustered_repeated_diag8x6_generator_v1` |
| Dimensions | 8 x 6 |
| Nonzeros | 5 |
| Requested rank for future solver proof | `k=3` |
| Diagonal entries | `10`, `10`, `9.999999`, `4`, `1`, structural zero |
| Expected matrix rank | 5 |
| Nullity | 1 |
| Support tier | `local_only` |
| Current validation command | `python3 scripts/validate_corpus_schema.py` |

## Expected Rows

| Row family | Rows |
| --- | --- |
| Singular values | `partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values` |
| Subspace projectors | `partial_svd_clustered_repeated_diag8x6_k3_v1_left_subspace`, `partial_svd_clustered_repeated_diag8x6_k3_v1_right_subspace` |
| Residual quality | `partial_svd_clustered_repeated_diag8x6_k3_v1_vector_residual`, `partial_svd_clustered_repeated_diag8x6_k3_v1_orthogonality` |
| Budget behavior | `partial_svd_clustered_repeated_diag8x6_k3_v1_default_status`, `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_status`, `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_no_partial_arrays` |

## Claim Boundary

The expected rows support only fixture-local partial-SVD evidence for the
generated 8x6 clustered/repeated diagonal matrix with `k=3`.

They do not support broad partial-SVD correctness, raw singular-vector
identity, broad repeated-spectrum coverage, external-library parity,
performance claims, or partial-result guarantees.

## Validation Evidence

Required Day 5 validation:

```sh
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/validate_corpus_schema.py
```

The full C quality gate is not required for Day 5 unless later edits touch
`.c` or `.h` files.

## Day 6 Handoff

Day 6 should define the oracle comparison behavior for the new expected rows.
The current `scripts/run_corpus_oracle.py` comparison function only handles the
first QR lane's `rank`, `nullity`, and `residual_norm` comparisons. The
partial-SVD lane needs comparison design for `value`, `subspace_distance`,
`status`, and `diagnostic` rows before solver-backed oracle output should be
implemented.
