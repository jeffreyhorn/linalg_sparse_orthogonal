# Sprint 151 Day 10: Report Integration Design

## Purpose

Design the normalized report-index handling for the Sprint 151 partial-SVD
corpus expansion before Day 11 changes report-index implementation or tests.

Day 10 keeps the report layer as a navigation and freshness aid. It does not
turn generated-local rows into hosted CI proof, external-library parity, broad
partial-SVD correctness, package support, performance evidence, or
state-of-the-art support.

## Current Report Surface

The current generated command:

```sh
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

emits:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

The normalized index command:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle
```

currently produces `105` rows when generated local outputs are present. The
selected Sprint 151 fixtures appear as source-controlled fixture, generator,
and expected-result rows, plus generated-local oracle rows under the
`oracle/solver_backed` report-family contract because their oracle rows include
`solver_family=partial_svd`.

## Required Normalized Row Coverage

Day 11 should preserve this coverage matrix for the three Sprint 151 selected
fixtures:

| Fixture | Fixture Rows | Generator Rows | Expected Rows | Generated Oracle Rows | Freshness Expectation |
| --- | ---: | ---: | ---: | ---: | --- |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | 1 | 1 | 7 | 7 | source-controlled metadata plus generated-local oracle rows |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | 1 | 1 | 6 | 6 | source-controlled metadata plus generated-local oracle rows |
| `partial_svd_fail_closed_diag6_k2_v1` | 1 | 1 | 5 | 5 | source-controlled metadata plus generated-local oracle rows |

The generated oracle rows must retain:

- `report_family=oracle`;
- `subfamily=solver_backed`;
- `row_origin=generated_local`;
- `support_tier=local_only`;
- `status=pass` only when the generated comparison passed;
- `configuration` fields including `solver_family=partial_svd`,
  `fixture_key=...`, `proof_owner=generated_partial_svd_reference`,
  `solver_execution=none`, generator hashes, and tolerance settings;
- fixture-local `claim_scope`;
- non-claims excluding broad correctness, raw singular-vector identity,
  arbitrary basis parity, broad fixture-family coverage, external-library
  parity, platform/package/ABI proof, performance, and state-of-the-art
  claims.

## Expected Row Groups

### Rank-Deficient Rectangular

The normalized index should expose these generated oracle row subjects:

- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_default_status`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_singular_values`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_rank`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_left_subspace`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_right_subspace`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_vector_residuals`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1_orthogonality`

### Sparse Low-Rank Output

The normalized index should expose these generated oracle row subjects:

- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_status`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_shape`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_nnz`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_selected_values`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_dense_frobenius_error`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1_sparse_dense_frobenius_diff`

### Non-Repeated Fail-Closed Convergence

The normalized index should expose these generated oracle row subjects:

- `partial_svd_fail_closed_diag6_k2_v1_tight_budget_status`
- `partial_svd_fail_closed_diag6_k2_v1_tight_budget_no_partial_arrays`
- `partial_svd_fail_closed_diag6_k2_v1_recovery_status`
- `partial_svd_fail_closed_diag6_k2_v1_default_singular_values`
- `partial_svd_fail_closed_diag6_k2_v1_default_vector_residuals`

## Freshness Rules

The report-index layer should distinguish three states:

| State | Required Behavior |
| --- | --- |
| Generated rows absent | Emit `not_generated` rows for selected generated families; do not create pass evidence. |
| Generated rows present for current commit | Preserve generated rows with local-only status and a freshness diagnostic that does not imply hosted proof. |
| Generated rows present for an older commit | Report stale generated rows; default freshness may warn, while `--strict-generated` or `--require-generated oracle` should make stale strict oracle evidence fail. |

For Sprint 151 partial-SVD oracle rows, strict freshness should remain stricter
than benchmark/advisory report freshness because the rows compare maintained
expected values. However, even fresh rows remain generated-local evidence only.

## Day 11 Implementation Checklist

Day 11 should:

1. Extend `tests/test_normalize_report_index.py` so generated oracle tests
   assert all three Sprint 151 partial-SVD fixture families, not only the
   existing clustered/repeated Sprint 140 fixture.
2. Assert expected generated row counts for the three selected families:
   `7`, `6`, and `5`.
3. Assert normalized generated rows preserve `solver_family=partial_svd`,
   `fixture_key=...`, `proof_owner=generated_partial_svd_reference`, and
   `solver_execution=none`.
4. Assert stale partial-SVD generated oracle rows warn by default and fail
   under strict generated freshness.
5. Keep generated-local report rows under `local_only` support tier and avoid
   changing report-family wording into hosted or release proof.
6. Run Python syntax, normalizer tests, corpus schema validation, generated
   oracle, normalized report-index, freshness, and whitespace checks.

## Non-Claim Register

Day 10 report integration design does not claim:

- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness;
- low-rank storage or drop-tolerance optimality;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- hosted CI proof;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.
