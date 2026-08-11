# Sprint 150 Day 4: Fixture Metadata Design

## Purpose

Design the source-controlled metadata rows Sprint 150 will add for the selected
QR corpus families before editing corpus manifests, expected-result files, or
oracle code. The design keeps the row set small enough to close completely
within the sprint while preserving explicit claim and non-claim boundaries.

## Schema Compatibility

Sprint 150 does not need new corpus columns. The planned rows fit the existing
schemas:

- `tests/corpus/manifests/fixtures.tsv` owns fixture identity, family, storage,
  shape, rank/nullity metadata, RHS policy, expected behavior, support tier,
  validation command, claim scope, and non-claims.
- `tests/corpus/manifests/generators.tsv` owns deterministic generator identity,
  generator version, algorithm label, parameters, expected structure/value
  hashes, canonical format, floating policy, regeneration command, and change
  policy.
- `tests/corpus/expected/*.tsv` owns oracle row identity, operation,
  comparison kind, expected-result kind, expected value, tolerance, claim scope,
  non-claims, and readiness status.
- `tests/corpus/manifests/report_families.tsv` already has source-controlled
  fixture, generator, expected-result, and generated-local oracle/report rows.
  Day 4 does not require a report-family schema change.

The existing enumerations in `scripts/validate_corpus_schema.py` support the
planned expected rows through `rank`, `nullity`, `residual_norm`, `value`,
`subspace_distance`, and `status` comparison/result kinds. Projector-oriented
rows should use `comparison_kind=subspace_distance` and
`expected_result_kind=subspace_distance` rather than adding a new comparison
kind.

## Fixture Batch

### Rank-Deficient Rectangular QR

Sprint 150 will keep the existing seed fixture and add two new rank-deficient
rectangular fixtures. The wide nullspace-subspace fixture remains a stretch row
because nullity `3` projector semantics are useful but would increase the Day
5-9 implementation surface.

| Fixture Key | Action | Family | Shape / NNZ | Rank / Nullity | Generator Key | RHS Policy | Support Tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1` | Keep existing row | `qr_rank_deficient` | `6x4`, `14` | `3 / 1` | `qr_rank_deficient_6x4_nullspace_generator_v1` | `generated_rhs` | `local_only` |
| `qr_rankdef_duplicate_5x4_v1` | Add row | `qr_rank_deficient_rectangular` | `5x4`, `14` | `3 / 1` | `qr_rankdef_duplicate_5x4_generator_v1` | `generated_rhs` | `local_only` |
| `qr_rankdef_dependent_row_4x3_v1` | Add row | `qr_rank_deficient_rectangular` | `4x3`, `9` | `2 / 1` | `qr_rankdef_dependent_row_4x3_generator_v1` | `generated_rhs` | `local_only` |

Common fixture fields:

- `storage_kind=generated`
- blank `matrix_path`
- `symmetry=none`
- `definiteness=rectangular`
- `rank_status=rank_deficient`
- `conditioning_class=moderate`
- `scale_class=unit`
- `sparsity_class=structured_sparse`
- `expected_behavior=success`
- `validation_command=python3 scripts/run_corpus_oracle.py --include-solver-qr`
- `owner=Sprint 150`
- `introduced_in=Sprint 150 Day 5`

Rank-deficient claim scope:

- fixture-local shape, `nnz`, rank, and nullity metadata;
- QR factorization success for the selected fixture;
- rank/nullity agreement with expected rows;
- normalized solver-produced nullspace residual;
- subspace-safe projector distance where an expected row owns it.

Rank-deficient non-claims:

- no raw Q/R basis equality;
- no Q-sign, orientation, scale, or column-order parity;
- no global rank-threshold policy;
- no broad rank-deficient QR correctness;
- no broad least-squares guarantee;
- no external-library parity;
- no platform, package, ABI, performance, or state-of-the-art claim.

### Underdetermined Minimum-Norm QR

Sprint 150 will add three deterministic underdetermined minimum-norm fixtures
with exact small-system expectations. Rank-deficient and zero-row minimum-norm
variants remain deferred so the first maintained family can close cleanly.

| Fixture Key | Action | Family | Shape / NNZ | Rank / Nullity | Generator Key | RHS Policy | Support Tier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | Add row | `qr_minnorm_underdetermined` | `2x4`, `4` | `2 / 2` | `qr_underdetermined_minnorm_2x4_generator_v1` | `explicit_rhs` | `local_only` |
| `qr_minnorm_3x6_exact_values` | Add row | `qr_minnorm_underdetermined` | `3x6`, `6` | `3 / 3` | `qr_minnorm_3x6_exact_values_generator_v1` | `explicit_rhs` | `local_only` |
| `qr_minnorm_5x10_exact_values` | Add row | `qr_minnorm_underdetermined` | `5x10`, `10` | `5 / 5` | `qr_minnorm_5x10_exact_values_generator_v1` | `explicit_rhs` | `local_only` |

Common fixture fields:

- `storage_kind=generated`
- blank `matrix_path`
- `symmetry=none`
- `definiteness=rectangular`
- `rank_status=full_row_rank`
- `conditioning_class=moderate`
- `scale_class=unit`
- `sparsity_class=structured_sparse`
- `expected_behavior=success`
- `validation_command=python3 scripts/run_corpus_oracle.py --include-solver-qr`
- `owner=Sprint 150`
- `introduced_in=Sprint 150 Day 5`

Minimum-norm claim scope:

- fixture-local shape, `nnz`, rank, nullity, and RHS metadata;
- `sparse_qr_solve_minnorm()` success for selected consistent fixtures;
- residual `||Ax-b||` within fixture tolerance;
- solution norm agreement with expected rows;
- exact solution entries only for rows that explicitly own exact values.

Minimum-norm non-claims:

- no global minimum-norm guarantee beyond selected fixtures and tolerances;
- no SVD pseudoinverse global-oracle claim;
- no broad rank-deficient recovery claim;
- no broad inconsistent-system behavior claim;
- no exact-vector identity for fixtures without explicit exact-value rows;
- no external-library parity;
- no platform, package, ABI, performance, or state-of-the-art claim.

## Generator Rows

Day 5 should add generator rows with exact computed hashes. Day 4 intentionally
does not invent hashes; the hash values must come from the same deterministic
entry order used by `scripts/validate_corpus_schema.py`.

| Generator Key | Algorithm | Parameters | Floating Policy |
| --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_generator_v1` | `fixed_rankdef_duplicate_5x4` | `rows=5;cols=4;expected_rank=3;nullity=1;duplicate_column=c3-c1` | exact integer structure/values; exact rank/nullity; residual tolerance `1e-10`; subspace tolerance `1e-8` |
| `qr_rankdef_dependent_row_4x3_generator_v1` | `fixed_rankdef_dependent_row_4x3` | `rows=4;cols=3;expected_rank=2;nullity=1;dependent_row=r2-r0-r1` | exact integer structure/values; exact rank/nullity; residual tolerance `1e-10`; subspace tolerance `1e-8` |
| `qr_underdetermined_minnorm_2x4_generator_v1` | `fixed_underdetermined_minnorm_2x4` | `rows=2;cols=4;expected_rank=2;nullity=2;rhs=1,1;expected_norm=1.0` | exact integer structure/values/RHS; residual tolerance `1e-10`; exact-value tolerance `1e-10` |
| `qr_minnorm_3x6_exact_values_generator_v1` | `fixed_minnorm_3x6_exact_values` | `rows=3;cols=6;expected_rank=3;nullity=3;expected_norm=sqrt(8.4)` | exact integer structure/values/RHS; residual tolerance `1e-10`; exact-value tolerance `1e-10` |
| `qr_minnorm_5x10_exact_values_generator_v1` | `fixed_minnorm_5x10_exact_values` | `rows=5;cols=10;expected_rank=5;nullity=5;expected_norm=sqrt(11.0)` | exact integer structure/values/RHS; residual tolerance `1e-10`; exact-value tolerance `1e-10` |

Common generator fields:

- `generator_version=1`
- `seed=none`
- `canonical_format=coo_zero_based_row_col_value_f64_text_v1`
- `regeneration_command=python3 scripts/run_corpus_oracle.py --include-solver-qr`
- `change_policy=update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together`

Day 5 implementation note: add deterministic entry builders to
`scripts/validate_corpus_schema.py` before committing manifest rows, then use
the validator's hash policy to populate `expected_structure_hash` and
`expected_value_hash`.

## Expected-Result Rows

Each new fixture should get its own expected-result TSV file under
`tests/corpus/expected/`.

### Rank-Deficient Expected Rows

For each added rank-deficient rectangular fixture:

| Row Suffix | Operation | Comparison Kind | Expected Result Kind | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| `_rank` | `rank_info` | `rank` | `rank` | `exact`, `0` | `ready_for_oracle` |
| `_nullity` | `rank_info` | `nullity` | `nullity` | `exact`, `0` | `ready_for_oracle` |
| `_nullspace_residual` | `nullspace` | `residual_norm` | `residual_norm` | `absolute`, `1e-10` | `ready_for_oracle` |
| `_nullspace_subspace` | `nullspace` | `subspace_distance` | `subspace_distance` | `projector`, `1e-8` | `ready_for_oracle` |

Expected values:

- `qr_rankdef_duplicate_5x4_v1_rank`: `3`
- `qr_rankdef_duplicate_5x4_v1_nullity`: `1`
- `qr_rankdef_duplicate_5x4_v1_nullspace_residual`:
  `normalized_null_vector_residual<=1e-10`
- `qr_rankdef_duplicate_5x4_v1_nullspace_subspace`:
  `projector_distance<=1e-8`
- `qr_rankdef_dependent_row_4x3_v1_rank`: `2`
- `qr_rankdef_dependent_row_4x3_v1_nullity`: `1`
- `qr_rankdef_dependent_row_4x3_v1_nullspace_residual`:
  `normalized_null_vector_residual<=1e-10`
- `qr_rankdef_dependent_row_4x3_v1_nullspace_subspace`:
  `projector_distance<=1e-8`

### Minimum-Norm Expected Rows

For each minimum-norm fixture:

| Row Suffix | Operation | Comparison Kind | Expected Result Kind | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| `_status` | `minnorm_solve` | `status` | `status` | `status_only`, `0` | `ready_for_oracle` |
| `_residual` | `minnorm_solve` | `residual_norm` | `residual_norm` | `absolute`, `1e-10` | `ready_for_oracle` |
| `_solution_norm` | `minnorm_solve` | `value` | `value` | `absolute`, `1e-10` | `ready_for_oracle` |
| `_solution_values` | `minnorm_solve` | `value` | `value` | `absolute`, `1e-10` | `ready_for_oracle` |

Expected values:

- `qr_underdetermined_minnorm_2x4_status`: `success`
- `qr_underdetermined_minnorm_2x4_residual`: `residual_norm<=1e-10`
- `qr_underdetermined_minnorm_2x4_solution_norm`: `1.0`
- `qr_underdetermined_minnorm_2x4_solution_values`:
  `0.5,0.5,0.5,0.5`
- `qr_minnorm_3x6_exact_values_status`: `success`
- `qr_minnorm_3x6_exact_values_residual`: `residual_norm<=1e-10`
- `qr_minnorm_3x6_exact_values_solution_norm`: `sqrt(8.4)`
- `qr_minnorm_3x6_exact_values_solution_values`:
  `1.2,1.2,1.0,0.6,0.4,2.0`
- `qr_minnorm_5x10_exact_values_status`: `success`
- `qr_minnorm_5x10_exact_values_residual`: `residual_norm<=1e-10`
- `qr_minnorm_5x10_exact_values_solution_norm`: `sqrt(11.0)`
- `qr_minnorm_5x10_exact_values_solution_values`:
  `0.4,0.8,1.2,1.6,2.0,0.2,0.4,0.6,0.8,1.0`

Minimum-norm exact-value rows must be removed or downgraded to residual/norm
rows if Day 8-9 proof-owner tests show platform-sensitive instability.

## Deferred Rows

These rows remain outside the Day 5 metadata batch unless later days find
unused capacity:

- `qr_rankdef_wide_3x5_nullspace_subspace_v1`: valuable nullity `3` subspace
  fixture, but projector/subspace semantics need more oracle and test budget.
- `qr_minnorm_rankdef_2x4`: useful rank-deficient minimum-norm fixture, but it
  would expand the claim from full-row-rank underdetermined systems.
- `qr_minnorm_zero_row_2x4`: useful edge case, but it is better paired with
  the rank-deficient minimum-norm row in a later residual-closure batch.
- Reorder/COLAMD QR rows: still deferred from Day 3.

## Day 5 Implementation Checklist

1. Add deterministic generator builders and expected hashes to
   `scripts/validate_corpus_schema.py`.
2. Add two rank-deficient rectangular fixture rows and three minimum-norm
   fixture rows to `tests/corpus/manifests/fixtures.tsv`.
3. Add five generator rows to `tests/corpus/manifests/generators.tsv`.
4. Add expected-result TSV files for the five new fixtures.
5. Run `python3 scripts/validate_corpus_schema.py`.
6. Keep solver pass evidence out of documentation until Days 8-11 add focused
   tests and generated oracle/report rows.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Metadata design fits existing corpus schemas. | Complete | No new columns are required; planned rows use existing fixture, generator, expected, and report-family schemas. |
| Each selected family has complete planned metadata coverage. | Complete | Rank-deficient rectangular and underdetermined minimum-norm rows have fixture, generator, expected, tolerance, claim-scope, and non-claim designs. |
| Regeneration and validation commands are identified before rows are edited. | Complete | Planned rows use `python3 scripts/run_corpus_oracle.py --include-solver-qr`; Day 5 must also pass `python3 scripts/validate_corpus_schema.py`. |
