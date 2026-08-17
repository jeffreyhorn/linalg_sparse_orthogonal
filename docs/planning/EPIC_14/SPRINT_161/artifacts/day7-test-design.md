# Day 7 Focused Proof-Owner Test Design

## Summary

Day 7 defines the remaining focused test scope before selected partial-SVD
comparison freshness is promoted. The existing runner test now owns target
dispatch and metadata. Day 8 should add normalizer coverage for the expanded
selected comparison row set and should avoid C proof-owner changes unless
solver behavior changes.

## Test Ownership Decision

| Surface | Owner Test | Decision |
| --- | --- | --- |
| CLI target dispatch | `tests/test_run_external_comparison.py` | Already expanded on Day 5/Day 6 for `partial-svd-diag6-k2`. |
| Generated output files | `tests/test_run_external_comparison.py` | Already verifies project, baseline, dependency, study, summary, and manifest files. |
| Generated row IDs and metrics | `tests/test_run_external_comparison.py` | Already verifies the ten selected partial-SVD row IDs and metric names. |
| Source-controlled report metadata | `tests/test_run_external_comparison.py`; `scripts/validate_corpus_schema.py` | Already verifies generator command, artifact pattern, support tier, and non-claim boundaries. |
| Selected comparison freshness | `tests/test_normalize_report_index.py`; `scripts/normalize_report_index.py` | Day 8 should expand selected comparison row set from QR-only to QR plus partial-SVD. |
| C solver proof owners | `tests/test_svd.c`; `tests/test_svd_partial_corpus.c`; helper headers | No new C tests for Day 8 unless implementation behavior changes. |

## Runner Test Scope

The runner test should continue to assert for every selected target:

- supported target appears in unsupported-target diagnostics;
- required output files are generated;
- manifest `target`, `fixture_key`, and `study_path` match the selected
  target;
- row count equals the per-target expected metric set;
- `comparison_row_id`, `metric`, `report_family`, `subfamily`, `fixture_key`,
  `operation`, `support_tier`, and `artifact_path` match expectations;
- all selected rows are `pass`;
- required source-controlled helper dependency is present and passing;
- optional `numpy` and `scipy` rows remain `defer` context with
  `deferred rows are not pass evidence`;
- report-family metadata preserves `generated_local`, `unknown`, `local_only`,
  `generated_compare_inputs`, parity non-claims, and state-of-the-art
  non-claims.

No Day 8 runner test expansion is required unless implementation changes alter
target metadata or emitted rows.

## Normalizer Test Scope

Day 8 should update the normalizer's selected comparison constants:

- add the ten `comparison_partial_svd_diag6_k2_*_v1` row IDs to
  `SELECTED_COMPARISON_ROW_IDS`;
- add `build/comparison/partial_svd_diag6_k2/study.tsv` to
  `SELECTED_COMPARISON_ARTIFACTS`;
- update the artifact diagnostic string to include the new partial-SVD study;
- keep existing QR selected rows intact.

Day 8 should update `tests/test_normalize_report_index.py` helper data to emit
all selected QR and partial-SVD comparison rows, then verify the expanded row
set is strict.

## Required Row-State Cases

| Case | Expected Result |
| --- | --- |
| Valid complete row set | `--require-generated comparison --check-freshness` succeeds with no selected comparison warning or error. |
| Missing partial-SVD row | Freshness fails with `comparison_selected_rows`, `row_set_mismatch`, the missing partial-SVD row ID, and the expanded artifact diagnostic. |
| Unexpected partial-SVD row | Freshness fails with `unexpected=...` and the expanded artifact diagnostic. |
| Duplicate partial-SVD row | Normalization fails with duplicate normalized row ID diagnostics. |
| Stale partial-SVD row | Freshness fails with stale source-commit diagnostics and `run make report-index-comparison-freshness`. |
| Failing partial-SVD row | Freshness fails with `generated comparison row reports fail` and `comparison_selected_status`. |
| Deferred partial-SVD row | Freshness fails; `defer` must not satisfy selected comparison evidence. |
| Skipped partial-SVD row | Freshness fails; `skip` must not satisfy selected comparison evidence. |
| Malformed partial-SVD row | Normalizer should fail or report a schema/parse issue before treating the row as selected evidence. |
| QR-only row set | Freshness fails after promotion because the partial-SVD selected rows are missing. |

## C Proof-Owner Decision

No C or header proof-owner tests are required for Day 8 because:

- Day 5 only added a temporary generated C probe inside a Python script;
- public partial-SVD APIs and fixture helpers were not changed;
- existing C tests already cover partial-SVD singular values, vector
  residuals, orthogonality, projector behavior, fail-closed behavior, and
  corpus fixtures;
- adding a C test for the same diagonal target would duplicate existing solver
  proof rather than protect the generated comparison plumbing.

If a later day modifies `.c` or `.h` files, the required quality gate becomes:

```sh
make format && make lint && make test
```

## Validation Command Matrix

| Changed Surface | Required Focused Validation |
| --- | --- |
| `scripts/run_external_comparison.py` | `python3 -m py_compile scripts/run_external_comparison.py`; `python3 scripts/run_external_comparison.py --self-check`; `python3 tests/test_run_external_comparison.py` |
| `tests/test_run_external_comparison.py` | `python3 -m py_compile tests/test_run_external_comparison.py`; `python3 tests/test_run_external_comparison.py` |
| `tests/corpus/manifests/report_families.tsv` | `python3 scripts/validate_corpus_schema.py` |
| `scripts/normalize_report_index.py` | `python3 -m py_compile scripts/normalize_report_index.py`; `python3 tests/test_normalize_report_index.py` |
| `tests/test_normalize_report_index.py` | `python3 -m py_compile tests/test_normalize_report_index.py`; `python3 tests/test_normalize_report_index.py` |
| `Makefile` comparison freshness target | `make report-index-comparison-freshness` after integration |
| `.c` or `.h` files | `make format && make lint && make test` |

## Day 8 Handoff

Day 8 should implement the normalizer selected-comparison expansion and focused
row-state tests. It should not add C tests unless new implementation behavior
is introduced.
