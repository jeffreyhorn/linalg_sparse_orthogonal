# Sprint 150 Day 5: Fixture Metadata Batch

## Purpose

Implement the Day 4 QR fixture metadata design as source-controlled corpus
rows, generator rows, and expected-result skeletons. These rows define
fixture-local expected-result contracts only; they are not observed solver pass
evidence until the focused proof-owner tests and generated oracle/report rows
land on Days 7-11.

## Implemented Fixture Rows

Added five generated QR fixtures to
`tests/corpus/manifests/fixtures.tsv`.

| Fixture Key | Family | Shape / NNZ | Rank / Nullity | Claim Boundary |
| --- | --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_v1` | `qr_rank_deficient_rectangular` | `5x4`, `14` | `3 / 1` | Rank/nullity, nullspace residual, and subspace-distance metadata only. |
| `qr_rankdef_dependent_row_4x3_v1` | `qr_rank_deficient_rectangular` | `4x3`, `9` | `2 / 1` | Rank/nullity, nullspace residual, and subspace-distance metadata only. |
| `qr_underdetermined_minnorm_2x4` | `qr_minnorm_underdetermined` | `2x4`, `4` | `2 / 2` | Minimum-norm residual, solution norm, and exact entries for this fixture only. |
| `qr_minnorm_3x6_exact_values` | `qr_minnorm_underdetermined` | `3x6`, `6` | `3 / 3` | Minimum-norm residual, solution norm, and exact entries for this fixture only. |
| `qr_minnorm_5x10_exact_values` | `qr_minnorm_underdetermined` | `5x10`, `10` | `5 / 5` | Minimum-norm residual, solution norm, and exact entries for this fixture only. |

All new rows use:

- `storage_kind=generated`
- `support_tier=local_only`
- `expected_behavior=success`
- `validation_command=python3 scripts/run_corpus_oracle.py --include-solver-qr`
- explicit non-claims excluding broad QR correctness, raw-basis parity,
  external-library parity, platform/package/ABI, performance, and
  state-of-the-art claims.

## Implemented Generator Rows

Added five deterministic generator rows to
`tests/corpus/manifests/generators.tsv` and added matching builder functions to
`scripts/validate_corpus_schema.py`.

| Generator Key | Algorithm | Structure Hash | Value Hash |
| --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_generator_v1` | `fixed_rankdef_duplicate_5x4` | `b14f80521b05893959db4ce00baf53831bd08588bbdc91d5785a045f4f544e66` | `af7b44a174a5dafc9c100b9472585dcdea36666faa2e4bf09c8128552e1547e8` |
| `qr_rankdef_dependent_row_4x3_generator_v1` | `fixed_rankdef_dependent_row_4x3` | `80e7f8520e7546cc4dc9ed961d8fc170be4a0aa4f7c9947bd47e35f336c9150e` | `642b740fa8bda01eb4fcd0ac9c38ffffe7ca1d68de94754b93ab8bd3582b3c9f` |
| `qr_underdetermined_minnorm_2x4_generator_v1` | `fixed_underdetermined_minnorm_2x4` | `467ffc3589d30ab560bf55d1516e66197b975cd1be0b4499527443ada0bb76ee` | `8d7b1dab76b212fe89fe2deb1dac6e2b579978f87312e4fc305e595af59bc9db` |
| `qr_minnorm_3x6_exact_values_generator_v1` | `fixed_minnorm_3x6_exact_values` | `ad2f3da1c402411545e1eb224952efa7bde2ee01e2cfb9bede3f1684d581b436` | `f7d8385abd88a28428a74bc30e6097ed0d244bc8aa59cdbd4ce663a5a7cc306a` |
| `qr_minnorm_5x10_exact_values_generator_v1` | `fixed_minnorm_5x10_exact_values` | `3889ecec38a3272dbcf78e22fccc28bc6aec3d620e2dbd23f6119ce47818b789` | `483bc47b26078139be8a685445036f26ccabd95264ab0023551ba22090ee600a` |

Hash provenance:

- hashes were computed from `scripts/validate_corpus_schema.py` using the
  existing canonical `coo_zero_based_row_col_value_f64_text_v1` value text and
  `coo_zero_based_row_col_text_v1` structure text;
- the validator now checks the new generator algorithms, parameters, row
  dimensions, rank/nullity metadata, `nnz`, and expected hashes.

## Implemented Expected-Result Rows

Added expected-result skeleton TSVs under `tests/corpus/expected/`:

- `qr_rankdef_duplicate_5x4_v1.tsv`
- `qr_rankdef_dependent_row_4x3_v1.tsv`
- `qr_underdetermined_minnorm_2x4.tsv`
- `qr_minnorm_3x6_exact_values.tsv`
- `qr_minnorm_5x10_exact_values.tsv`

Rank-deficient files define:

- exact rank rows;
- exact nullity rows;
- `normalized_null_vector_residual<=1e-10`;
- `projector_distance<=1e-8` using `subspace_distance` and `projector`
  tolerance semantics.

Minimum-norm files define:

- `SPARSE_SUCCESS` status rows;
- `residual_norm<=1e-10`;
- solution norm rows for `1.0`, `sqrt(8.4)`, and `sqrt(11.0)`;
- exact solution entry rows only for the selected deterministic fixtures.

## Validation

Command run:

```sh
python3 scripts/validate_corpus_schema.py
```

Result:

```text
validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok
```

## Report-Index Status

No report-family row changes were required on Day 5. Existing
source-controlled fixture, generator, expected-result, and local oracle/report
families already cover the metadata category. Day 10-11 will decide whether
new generated-local QR family rows are needed after oracle semantics and
proof-owner tests exist.

## Day 6 Handoff

Day 6 should define concrete oracle semantics for:

1. rank/nullity checks for the two new rank-deficient rectangular fixtures;
2. normalized nullspace residual calculation;
3. projector/subspace-distance calculation without raw-basis identity;
4. minimum-norm residual, status, solution-norm, and exact-value comparisons;
5. downgrade rules for exact-value rows if proof-owner tests expose
   platform-sensitive instability.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Selected QR families have complete source-controlled metadata. | Complete | Five fixture rows, five generator rows, and five expected-result TSV files were added. |
| Corpus schema validation passes. | Complete | `python3 scripts/validate_corpus_schema.py` passed after the row additions. |
| Report-index normalization remains stable or planned updates are explicit. | Complete | No Day 5 report-family row change was needed; Day 10-11 will handle generated-local report integration. |
