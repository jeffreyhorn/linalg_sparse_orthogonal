# Sprint 151 Day 6: Metadata Batch

## Purpose

Implement the source-controlled partial-SVD corpus metadata batch designed on
Day 5: deterministic generator validation, fixture rows, generator rows,
expected-result rows, claim scopes, support tiers, non-claims, and validation
evidence.

## Implemented Files

| Surface | File | Day 6 Change |
| --- | --- | --- |
| Generator validation | `scripts/validate_corpus_schema.py` | Added deterministic entry builders and generator contracts for the three Sprint 151 partial-SVD fixtures. |
| Fixture manifest | `tests/corpus/manifests/fixtures.tsv` | Added three source-controlled partial-SVD fixture rows. |
| Generator manifest | `tests/corpus/manifests/generators.tsv` | Added three deterministic generator rows with computed structure/value hashes. |
| Rank-deficient expected rows | `tests/corpus/expected/partial_svd_rankdef_diag6x4_k2_range_projector_v1.tsv` | Added seven expected-result rows. |
| Sparse-output expected rows | `tests/corpus/expected/partial_svd_lowrank_rect5x7_k3_sparse_output_v1.tsv` | Added six expected-result rows. |
| Fail-closed expected rows | `tests/corpus/expected/partial_svd_fail_closed_diag6_k2_v1.tsv` | Added five expected-result rows. |

## Fixture Rows

| Fixture Key | Family | Shape / NNZ | Rank / Nullity | Expected Behavior | Support Tier |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | `partial_svd_rankdef_projector` | `6x4`, `2` | `2 / 2` | `success` | `local_only` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | `partial_svd_sparse_lowrank_output` | `5x7`, `4` | `4 / 3` | `success` | `local_only` |
| `partial_svd_fail_closed_diag6_k2_v1` | `partial_svd_fail_closed` | `6x6`, `6` | `6 / 0` | `non_convergence` | `local_only` |

All three rows use:

- `storage_kind=generated`;
- blank `matrix_path`;
- `symmetry=none`;
- `conditioning_class=moderate`;
- `scale_class=scaled`;
- `sparsity_class=diagonal`;
- `rhs_policy=none`;
- `validation_command=python3 scripts/run_corpus_oracle.py --include-partial-svd`;
- `owner=Sprint 151`;
- `introduced_in=Sprint 151 Day 6`.

## Generator Rows

| Generator Key | Structure Hash | Value Hash |
| --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1` | `5e38ed6d9d818205dde64c282aed13388a57f44f6c96276bfc79882a5f7666f0` | `189c5d8eac23eb42ff57872ec1e799b44336ae5586d8eb0850a720df09683287` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1` | `e06892510eefe6477c674bbe9740f95ef9232bc587af4699e7b0195caac10c31` | `1fb0d55f44b7afac9d4e23415992958c7974e7400ecd70bb5d6a8d61ef6a56d9` |
| `partial_svd_fail_closed_diag6_k2_generator_v1` | `e5b22b33f24e83a3cd111e325ebefb14fea56e89f2845c195decb2fb3bf825a5` | `17e580624f29101093724802e4621ea3e1f811f171ce49d286445566ad943a57` |

Hashes were computed from the validator's canonical generated matrix text using
`coo_zero_based_row_col_value_f64_text_v1` for values and
`coo_zero_based_row_col_text_v1` for structure.

## Expected-Result Rows

### Rank-Deficient Rectangular Range Projector

`partial_svd_rankdef_diag6x4_k2_range_projector_v1.tsv` contains:

- default `SPARSE_SUCCESS`;
- `top_k=9,6` singular values with absolute tolerance `1e-8`;
- exact rank `2`;
- left and right projector distances bounded by `1e-8`;
- max triplet residual bounded by `1e-8`;
- max orthogonality residual bounded by `1e-8`.

### Sparse Low-Rank Output

`partial_svd_lowrank_rect5x7_k3_sparse_output_v1.tsv` contains:

- sparse low-rank `SPARSE_SUCCESS`;
- exact shape diagnostic `shape=5x7`;
- exact retained sparse-output nonzero count `3`;
- selected values `8,4,2,0` with absolute tolerance `1e-10`;
- dense Frobenius absolute-error bound `1e-10`;
- sparse-vs-dense Frobenius-difference bound `1e-10`.

The selected-values row remains the known Day 7/Day 11 oracle follow-up: the
comparator must parse `selected_values` as a vector or the row must be replaced
with scalar supported rows before generated oracle output is claimed.

### Non-Repeated Fail-Closed

`partial_svd_fail_closed_diag6_k2_v1.tsv` contains:

- tight-budget `SPARSE_ERR_NOT_CONVERGED`;
- fail-closed diagnostic `no_partial_sigma_u_vt_on_failure`;
- default-budget recovery `SPARSE_SUCCESS`;
- default `top_k=9,6` singular values with absolute tolerance `1e-8`;
- default max triplet residual bounded by `1e-8`.

## Claim Boundaries

All new metadata rows are fixture-local and `local_only`. They do not claim:

- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness;
- storage or drop-tolerance optimality;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.

## Validation

Commands run:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Results:

- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`
- `normalize-report-index: 102 rows ok`

No `.c` or `.h` files changed on Day 6, so the C quality gate was not
required.

## Day 7 Handoff

Day 7 should implement or refine deterministic partial-SVD oracle inputs for
the new fixtures. Specific handoff items:

1. Teach `scripts/run_corpus_oracle.py --include-partial-svd` to load and emit
   observed rows for all three Sprint 151 expected-result files.
2. Add the narrow `selected_values` comparison support or replace that row
   with scalar rows using currently supported comparison semantics.
3. Confirm generated observed rows keep `support_tier=local_only`,
   solver-family `partial_svd`, explicit command metadata, and fixture-local
   non-claims.
4. Keep generated oracle rows under ignored `build/corpus/` paths unless a
   later sprint explicitly promotes an artifact.
