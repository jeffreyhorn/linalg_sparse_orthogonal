# Day 11: Report Integration Implementation

## Scope

Day 11 promoted the Sprint 154 QR minimum-norm comparison output from
artifact-only local evidence into an explicit normalized report-index family.

The integration remains narrow:

- one report family: `comparison`;
- one subfamily: `qr_minnorm`;
- one generated artifact: `build/comparison/qr_minnorm/study.tsv`;
- one selected fixture: `qr_underdetermined_minnorm_2x4`;
- six required selected rows;
- local-only support tier.

## Implementation

### Report-Family Metadata

Added one source-controlled report-family contract row to
`tests/corpus/manifests/report_families.tsv`:

- `report_family=comparison`;
- `subfamily=qr_minnorm`;
- `row_meaning=external_process_dense_reference_comparison`;
- `row_origin=generated_local`;
- `status=unknown`;
- `support_tier=local_only`;
- `freshness_policy=generated_compare_inputs`;
- `generator_command=python3 scripts/run_external_comparison.py --target qr-minnorm`;
- `artifact_pattern=build/comparison/qr_minnorm/study.tsv`.

### Schema Validation

Updated `scripts/validate_corpus_schema.py` so the maintained manifest accepts
the new row meaning:

- `external_process_dense_reference_comparison`.

Contract rows still cannot be `pass` evidence, and the non-claim field remains
required.

### Normalized Report Index

Updated `scripts/normalize_report_index.py` to:

- load `build/comparison/qr_minnorm/study.tsv`;
- map study rows into normalized report-index fields;
- preserve source commit, source branch, generated timestamp, platform,
  compiler, configuration, artifact path, support tier, claim scope, and
  non-claims;
- keep generated comparison rows under `freshness_policy=generated_compare_inputs`;
- add selected comparison row diagnostics;
- fail required comparison freshness on missing, duplicate, unexpected, or
  non-pass selected rows;
- treat `skip` and `defer` comparison rows as visible non-proof states.

### Maintained Command

Added:

```sh
make report-index-comparison-freshness
```

The target:

1. regenerates the selected local comparison output with
   `python3 scripts/run_external_comparison.py --target qr-minnorm`;
2. checks comparison freshness with
   `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`;
3. reports the lane as local-only generated comparison freshness.

## Selected Rows

The required selected comparison row set is:

1. `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`;
2. `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`;
3. `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`;
4. `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`;
5. `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`;
6. `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`.

Required comparison freshness fails closed if this row set is incomplete,
duplicated, has unexpected generated comparison rows, or contains non-pass
selected rows.

## Validation

Ran:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family comparison --check
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
make report-index-comparison-freshness
python3 scripts/run_external_comparison.py --self-check
python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check
git diff --check
```

Results:

- comparison harness regenerated the selected output and all selected rows
  passed;
- corpus schema validation passed;
- comparison report-index normalization produced `7` rows: one contract row
  plus six generated selected rows;
- required comparison freshness passed with generated rows present;
- broader corpus/oracle/comparison report-index check produced `85` rows;
- comparison self-check passed.

Negative required-generated smoke check:

```sh
python3 scripts/normalize_report_index.py \
  --family comparison \
  --build-root <empty-temp-dir> \
  --require-generated comparison \
  --check-freshness
```

Expected result: non-zero exit with
`required generated family missing: comparison`.

## Non-Claims

The comparison report family does not claim:

- broad QR parity;
- NumPy parity;
- SciPy parity;
- LAPACK parity;
- SuiteSparse parity;
- Eigen parity;
- external-library ecosystem parity;
- hosted CI proof;
- release proof;
- platform portability proof;
- package-manager proof;
- shared-library or ABI proof;
- performance superiority;
- state-of-the-art status.

## Day 12 Handoff

Day 12 should align maintainer and public documentation with:

- `make report-index-comparison-freshness`;
- `comparison/qr_minnorm` normalized report rows;
- the six selected-row requirement;
- local-only support-tier interpretation;
- optional dependency `defer` not counting as proof;
- the non-claim register above.
