# Sprint 151 Day 5: Metadata Design

## Purpose

Translate the Day 4 partial-SVD comparison contract into concrete fixture,
generator, expected-result, claim-scope, support-tier, non-claim, and
validation-command metadata before source-controlled corpus rows are edited.

Day 5 is a design pass. It intentionally does not invent generator hashes or
mark generated oracle output as pass evidence.

## Schema Compatibility

The planned Sprint 151 metadata mostly fits existing corpus schemas:

- `tests/corpus/manifests/fixtures.tsv` can represent all three selected
  fixture families with `storage_kind=generated`, fixture-local claim scopes,
  `support_tier=local_only`, and validation commands.
- `tests/corpus/manifests/generators.tsv` can represent all three
  deterministic diagonal generator contracts with `seed=none`, versioned
  generator keys, canonical hash fields, regeneration commands, and change
  policy.
- `tests/corpus/expected/<fixture_key>.tsv` can represent status, rank,
  projector, residual, and diagnostic rows with existing expected-result
  fields.
- `tests/corpus/manifests/report_families.tsv` already has source-controlled
  corpus rows and generated-local oracle rows for
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`;
  no report-family row is required for Day 6 metadata.

One narrow oracle comparator extension remains required later in the sprint:
`comparison_kind=value` must accept `selected_values` as a comma-separated
numeric vector, parallel to current `solution_values` handling.

## Fixture Row Design

### Fixture Batch

| Fixture Key | Family | Shape / NNZ | Rank / Nullity | Generator Key | Expected Behavior | Support Tier |
| --- | --- | --- | --- | --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | `partial_svd_rankdef_projector` | `6x4`, `2` | `2 / 2` | `partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1` | `success` | `local_only` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | `partial_svd_sparse_lowrank_output` | `5x7`, `4` | `4 / 3` | `partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1` | `success` | `local_only` |
| `partial_svd_fail_closed_diag6_k2_v1` | `partial_svd_fail_closed` | `6x6`, `6` | `6 / 0` | `partial_svd_fail_closed_diag6_k2_generator_v1` | `non_convergence` | `local_only` |

Common fixture fields:

- `storage_kind=generated`
- blank `matrix_path`
- `symmetry=none`
- `definiteness=rectangular` for the `6x4` and `5x7` fixtures
- `definiteness=singular` is not appropriate for the fail-closed fixture
  because it is full rank; use `definiteness=unknown` for the nonsymmetric
  square diagonal fixture unless the validator accepts a more precise
  diagonal nonsymmetric/full-rank class
- `rank_status=rank_deficient` for the `6x4` and `5x7` fixtures
- `rank_status=full_rank` for the fail-closed fixture
- `conditioning_class=moderate`
- `scale_class=scaled`
- `sparsity_class=diagonal`
- `rhs_policy=none`
- `validation_command=python3 scripts/run_corpus_oracle.py --include-partial-svd`
- `owner=Sprint 151`
- `introduced_in=Sprint 151 Day 6`

### Fixture Claim Scopes

| Fixture Key | Claim Scope |
| --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Fixture-local partial-SVD rank-deficient rectangular evidence for top-2 singular values, rank, left/right coordinate-range projectors, triplet residuals, and orthogonality. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | Fixture-local sparse low-rank output evidence for shape, retained nonzero count, selected coordinate values, dense low-rank Frobenius error, and dense/sparse output agreement at `drop_tol=0`. |
| `partial_svd_fail_closed_diag6_k2_v1` | Fixture-local partial-SVD convergence-budget evidence for tight-budget non-convergence, no partial arrays on failure, default-budget recovery, default singular values, and default triplet residuals. |

### Fixture Non-Claims

Use semicolon-separated wording in the manifest rows. Each row should include
the common non-claims below plus the fixture-specific additions.

Common non-claims:

- no broad partial-SVD correctness
- no raw singular-vector identity
- no sign/orientation/phase/basis-order parity
- no external-library parity
- no platform/package/ABI/performance/state-of-the-art claim

Fixture-specific non-claims:

| Fixture Key | Additional Non-Claims |
| --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | no broad rank-deficient behavior; no broad null-space behavior; no pseudoinverse or minimum-norm claim |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | no broad low-rank optimality; no broad sparse-output correctness; no storage optimality; no drop-tolerance optimality; no sparse-output performance claim |
| `partial_svd_fail_closed_diag6_k2_v1` | no convergence-rate claim; no portable iteration-count claim; no useful partial-result guarantee after non-convergence |

## Generator Row Design

Day 6 should add deterministic generator builders and compute hashes from the
same canonical format used by `scripts/validate_corpus_schema.py`. Day 5 does
not invent `expected_structure_hash` or `expected_value_hash`.

| Generator Key | Algorithm | Parameters | Floating Policy |
| --- | --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1` | `fixed_partial_svd_rankdef_diag6x4_k2_range_projector` | `rows=6;cols=4;k=2;diag=9,6,0,0;expected_rank=2;nullity=2` | exact generated coordinates and values; top-k singular values tolerance `1e-8`; projector/residual/orthogonality tolerance `1e-8`; rank exact |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1` | `fixed_partial_svd_lowrank_rect5x7_k3_sparse_output` | `rows=5;cols=7;k=3;diag=8,4,2,1,0;drop_tol=0;expected_rank=4;nullity=3` | exact generated coordinates and values; selected sparse-output values tolerance `1e-10`; Frobenius consistency tolerance `1e-10` |
| `partial_svd_fail_closed_diag6_k2_generator_v1` | `fixed_partial_svd_fail_closed_diag6_k2` | `rows=6;cols=6;k=2;diag=9,6,3,1,0.5,0.25;tight_max_iter=1;expected_rank=6;nullity=0` | exact generated coordinates and values; status rows exact; top-k singular values and residual tolerance `1e-8`; no partial arrays on failure |

Common generator fields:

- `generator_version=1`
- `seed=none`
- `canonical_format=coo_zero_based_row_col_value_f64_text_v1`
- `regeneration_command=python3 scripts/run_corpus_oracle.py --include-partial-svd`
- `change_policy=update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together`

Day 6 implementation should add or extend deterministic generator support in
`scripts/validate_corpus_schema.py`, run the validator, copy the computed
hashes into `tests/corpus/manifests/generators.tsv`, and rerun validation.

## Expected-Result Row Design

Each selected fixture should get one source-controlled expected-result TSV:

- `tests/corpus/expected/partial_svd_rankdef_diag6x4_k2_range_projector_v1.tsv`
- `tests/corpus/expected/partial_svd_lowrank_rect5x7_k3_sparse_output_v1.tsv`
- `tests/corpus/expected/partial_svd_fail_closed_diag6_k2_v1.tsv`

All expected rows use `status=ready_for_oracle`.

### Rank-Deficient Rectangular Rows

| Row ID | Operation | Comparison Kind | Expected Result Kind | Expected Result | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_default_status` | `partial_svd_default` | `status` | `status` | `SPARSE_SUCCESS` | `status_only`, empty |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_singular_values` | `singular_values` | `value` | `value` | `top_k=9,6` | `absolute`, `1e-8` |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_rank` | `rank_info` | `rank` | `rank` | `2` | `exact`, `0` |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_left_subspace` | `singular_subspace` | `subspace_distance` | `subspace_distance` | `left_projector_distance<=1e-8` | `projector`, `1e-8` |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_right_subspace` | `singular_subspace` | `subspace_distance` | `subspace_distance` | `right_projector_distance<=1e-8` | `projector`, `1e-8` |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_vector_residuals` | `vector_residuals` | `residual_norm` | `residual_norm` | `max_triplet_residual<=1e-8` | `absolute`, `1e-8` |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1_orthogonality` | `orthogonality` | `residual_norm` | `residual_norm` | `max_orthogonality_residual<=1e-8` | `absolute`, `1e-8` |

### Sparse Low-Rank Output Rows

| Row ID | Operation | Comparison Kind | Expected Result Kind | Expected Result | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_status` | `sparse_lowrank` | `status` | `status` | `SPARSE_SUCCESS` | `status_only`, empty |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_shape` | `sparse_lowrank` | `diagnostic` | `diagnostic` | `shape=5x7` | `not_applicable`, empty |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_nnz` | `sparse_lowrank` | `rank` | `rank` | `3` | `exact`, `0` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values` | `sparse_lowrank` | `value` | `value` | `selected_values=8,4,2,0` | `absolute`, `1e-10` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_dense_frobenius_error` | `lowrank_reconstruction` | `residual_norm` | `residual_norm` | `dense_frobenius_abs_error<=1e-10` | `absolute`, `1e-10` |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_dense_frobenius_diff` | `sparse_lowrank_consistency` | `residual_norm` | `residual_norm` | `sparse_dense_frobenius_diff<=1e-10` | `absolute`, `1e-10` |

The `sparse_selected_values` row is the only expected row that requires a
comparator extension. If that extension becomes risky, Day 6 should split the
row into supported scalar residual rows instead of widening the comparator.

### Fail-Closed Rows

| Row ID | Operation | Comparison Kind | Expected Result Kind | Expected Result | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_fail_closed_diag6_k2_v1_tight_budget_status` | `convergence_budget` | `status` | `status` | `SPARSE_ERR_NOT_CONVERGED` | `status_only`, empty |
| `partial_svd_fail_closed_diag6_k2_v1_tight_budget_no_partial_arrays` | `diagnostic` | `diagnostic` | `diagnostic` | `no_partial_sigma_u_vt_on_failure` | `not_applicable`, empty |
| `partial_svd_fail_closed_diag6_k2_v1_recovery_status` | `convergence_budget` | `status` | `status` | `SPARSE_SUCCESS` | `status_only`, empty |
| `partial_svd_fail_closed_diag6_k2_v1_default_singular_values` | `singular_values` | `value` | `value` | `top_k=9,6` | `absolute`, `1e-8` |
| `partial_svd_fail_closed_diag6_k2_v1_default_vector_residuals` | `vector_residuals` | `residual_norm` | `residual_norm` | `max_triplet_residual<=1e-8` | `absolute`, `1e-8` |

## Support-Tier And Report Design

All Sprint 151 metadata rows should start as `support_tier=local_only`.
Promotion beyond local-only is out of scope for this sprint because the
selected evidence is generated-local and fixture-local.

Report integration should reuse existing rows:

| Report Family Row | Use In Sprint 151 |
| --- | --- |
| `corpus/fixtures` | Source-controlled fixture manifest rows. |
| `corpus/generators` | Source-controlled deterministic generator metadata. |
| `corpus/expected` | Source-controlled expected-result rows. |
| `oracle/solver_backed` | Generated-local solver-backed partial-SVD rows from `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`. |

No new `tests/corpus/manifests/report_families.tsv` row is planned for Day 6.
If Day 11 discovers that partial-SVD sparse-output rows need a distinct
subfamily for report indexing, that should be added with explicit generated
local-only non-claims and no hosted-platform proof wording.

## Validation And Regeneration Commands

Planned metadata validation sequence:

```sh
python3 scripts/validate_corpus_schema.py
```

Planned local oracle generation sequence after Day 11 implementation:

```sh
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

If Day 6 or later changes `.c` or `.h` files, run:

```sh
make format && make lint && make test
```

Day 5 is documentation-only and does not require the C quality gate.

## Day 6 Implementation Checklist

1. Add deterministic generator builders for the three selected fixtures to
   `scripts/validate_corpus_schema.py`.
2. Compute and populate generator structure/value hashes from the validator.
3. Add the three fixture rows to `tests/corpus/manifests/fixtures.tsv`.
4. Add the three generator rows to `tests/corpus/manifests/generators.tsv`.
5. Add the three expected-result TSV files under `tests/corpus/expected/`.
6. Add only the minimal schema or comparator TODO markers needed for the
   `selected_values` expected row; actual oracle emission can wait for Day 11
   unless Day 6 validation requires it.
7. Run `python3 scripts/validate_corpus_schema.py`.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Metadata design fits existing corpus schemas or names required changes. | Complete | Fixture, generator, expected, support-tier, and report rows fit existing columns; only `selected_values` needs a later narrow comparator extension. |
| Each selected family has complete planned metadata coverage. | Complete | All three selected families have fixture row, generator row, expected-result row, claim-scope, support-tier, non-claim, and validation-command designs. |
| Regeneration and validation commands are identified before rows are edited. | Complete | Validator, partial-SVD oracle, combined oracle, report normalization, freshness, and C gate commands are listed. |
