# Day 4 Partial-SVD Fixture Batch Design

## Purpose

Day 4 turns the selected closure contract into source-controlled corpus row
design. The fixture remains unimplemented until Day 5; this artifact defines
the exact metadata, expected rows, hash policy, and claim boundaries that the
implementation should use.

## Fixture Batch

Sprint 140 uses one primary fixture. Additional partial-SVD residuals stay
deferred so the sprint can close one behavior family completely.

| Fixture key | Role | Status |
| --- | --- | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Primary selected partial-SVD clustered/repeated top-k subspace and convergence-budget fixture | Implement on Day 5 |
| `partial_svd_rankdef_range_projector_budget_v1` | Backup rank-deficient range-projector promotion | Deferred unless the primary fixture is blocked |

## Planned Fixture Manifest Row

The Day 5 implementation should append this fixture row to
`tests/corpus/manifests/fixtures.tsv`.

| Field | Value |
| --- | --- |
| `fixture_key` | `partial_svd_clustered_repeated_diag8x6_k3_v1` |
| `fixture_family` | `partial_svd_clustered_repeated` |
| `storage_kind` | `generated` |
| `matrix_path` | empty |
| `generator_key` | `partial_svd_clustered_repeated_diag8x6_generator_v1` |
| `rows` | `8` |
| `cols` | `6` |
| `nnz` | `5` |
| `symmetry` | `none` |
| `definiteness` | `rectangular` |
| `rank_status` | `full_rank` |
| `expected_rank` | `5` |
| `nullity` | `1` |
| `conditioning_class` | `moderate` |
| `scale_class` | `scaled` |
| `sparsity_class` | `diagonal` |
| `rhs_policy` | `none` |
| `expected_behavior` | `success` |
| `claim_scope` | Fixture-local partial-SVD clustered/repeated top-k subspace and budget behavior. |
| `non_claims` | no broad partial-SVD correctness; no raw singular-vector identity; no broad repeated-spectrum coverage; no external-library parity; no performance claim |
| `support_tier` | `local_only` until promoted by validation |
| `validation_command` | `python3 scripts/validate_corpus_schema.py` initially; solver-backed command to be added when implemented |
| `owner` | `Sprint 140` |
| `introduced_in` | `Sprint 140 Day 5` |

The fixture uses `rank_status=full_rank` relative to its six columns? No. The
matrix has rank 5 and nullity 1, so Day 5 should use `rank_deficient` if the
validator treats rank status as matrix-column rank. The implementation should
choose `rank_deficient` for consistency with `expected_rank=5` and `nullity=1`.

Day 5 final value:

```text
rank_status=rank_deficient
```

## Planned Generator Manifest Row

The Day 5 implementation should append this generator row to
`tests/corpus/manifests/generators.tsv`.

| Field | Value |
| --- | --- |
| `generator_key` | `partial_svd_clustered_repeated_diag8x6_generator_v1` |
| `generator_version` | `1` |
| `algorithm` | `fixed_diagonal_clustered_repeated_partial_svd` |
| `seed` | `none` |
| `parameters` | `rows=8;cols=6;k=3;diag=10,10,9.999999,4,1,0;expected_rank=5;nullity=1` |
| `expected_structure_hash` | `d454c7476f8a444b1c7785bf19beae7bdff1915e3e3b32872b198513e90f8adb` |
| `expected_value_hash` | `bcc1fe2e1ee7be5339204c16a8a0c1facc26a70962acf51d642550fb16def6b1` |
| `canonical_format` | `coo_zero_based_row_col_value_f64_text_v1` |
| `floating_policy` | exact generated coordinates and values; top-k singular values absolute tolerance `1e-8`; projector/residual/orthogonality tolerance `1e-8`; status rows exact |
| `regeneration_command` | `python3 scripts/validate_corpus_schema.py` initially; partial-SVD oracle command once implemented |
| `change_policy` | update generator version, fixture metadata, expected rows, oracle rows, validation command, and docs together |

## Canonical Text

Structure hash input:

```text
format coo_zero_based_row_col_text_v1
rows 8
cols 6
nnz 5
0 0
1 1
2 2
3 3
4 4
```

Value hash input:

```text
format coo_zero_based_row_col_value_f64_text_v1
rows 8
cols 6
nnz 5
0 0 10.0000000000000000
1 1 10.0000000000000000
2 2 9.9999990000000007
3 3 4.0000000000000000
4 4 1.0000000000000000
```

The final zero singular value is structural. It is represented by the missing
sixth diagonal entry rather than by an explicit zero-valued nonzero.

## Planned Expected Rows

The Day 5 implementation should add
`tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` with
these row IDs and semantics.

| Oracle row ID | Operation | Comparison kind | Expected result kind | Expected result | Tolerance kind | Tolerance value | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values` | `singular_values` | `value` | `value` | `top_k=10,10,9.999999` | `absolute` | `1e-8` | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_left_subspace` | `singular_subspace` | `subspace_distance` | `subspace_distance` | `left_projector_distance<=1e-8` | `projector` | `1e-8` | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_right_subspace` | `singular_subspace` | `subspace_distance` | `subspace_distance` | `right_projector_distance<=1e-8` | `projector` | `1e-8` | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_vector_residual` | `vector_residuals` | `residual_norm` | `residual_norm` | `max_triplet_residual<=1e-8` | `absolute` | `1e-8` | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_orthogonality` | `orthogonality` | `residual_norm` | `residual_norm` | `max_orthogonality_residual<=1e-8` | `absolute` | `1e-8` | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_default_status` | `convergence_budget` | `status` | `status` | `SPARSE_SUCCESS` | `status_only` | empty | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_status` | `convergence_budget` | `status` | `status` | `SPARSE_ERR_NOT_CONVERGED` | `status_only` | empty | `ready_for_oracle` |
| `partial_svd_clustered_repeated_diag8x6_k3_v1_tight_budget_no_partial_arrays` | `diagnostic` | `diagnostic` | `diagnostic` | `no_partial_sigma_u_vt_on_failure` | `not_applicable` | empty | `ready_for_oracle` |

## Claim Wording

Use this claim scope on expected rows unless a specific row needs narrower
wording:

```text
Fixture-local partial-SVD evidence for a generated 8x6 clustered/repeated
diagonal matrix with k=3.
```

Use this non-claim boundary on expected rows:

```text
no broad partial-SVD correctness; no raw singular-vector identity; no broad
repeated-spectrum coverage; no external-library parity; no performance claim;
no partial-result guarantee
```

## Skip And Defer Rows

No optional external-data skip row is needed for the primary fixture because it
is generated locally. The following residuals remain intentionally deferred:

| Deferred residual | Reason |
| --- | --- |
| `partial_svd_rankdef_range_projector_budget_v1` | Backup only; current helper evidence already covers much of the behavior. |
| optional SuiteSparse partial-SVD fixture | Requires support-tier promotion before pass evidence is meaningful. |
| near-zero threshold fixture family | Requires broader rank/tolerance policy work. |
| low-rank sparse approximation fixture family | Exceeds the selected residual and risks product-level overclaiming. |

## Portability And Ambiguity Review

- The fixture is generated from exact coordinate/value literals and has no
  random seed or external data dependency.
- The top-k subspace is separated from the fourth singular value by a large gap,
  so projector comparisons have a stable target.
- The repeated leading pair permits valid basis rotations, so raw vector-column
  equality is forbidden.
- The clustered third singular value prevents the selected lane from collapsing
  into a pure repeated-value smoke test.
- The missing sixth diagonal entry creates a nullity-one rectangular fixture
  without introducing near-zero threshold policy.
- Tight-budget failure rows are diagnostics only and do not imply partial
  result availability, convergence rate, or performance behavior.

## Day 5 Implementation Checklist

1. Add the generated fixture function to the corpus validator/oracle generator
   registry.
2. Append the fixture row with `rank_status=rank_deficient`.
3. Append the generator row with the hashes recorded above.
4. Add the expected-result TSV with the eight planned rows.
5. Update schema documentation only if the current comparison validation cannot
   represent `value`, `subspace_distance`, `status`, or `diagnostic` rows.
6. Run `python3 scripts/validate_corpus_schema.py`.
7. Keep generated oracle/report files under `build/` and uncommitted.
