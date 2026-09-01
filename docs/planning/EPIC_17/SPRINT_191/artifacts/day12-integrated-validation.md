# Sprint 191 Day 12: Integrated Local Validation

## Summary

Day 12 validated the selected `qr-incompatible-ls` comparison family end to
end.

The direct generator, target-specific freshness command, full selected
comparison freshness gate, focused Python tests, docs guard, manifest/schema
checks, workflow guard, and affected QR solve C test all passed.

## Generated Artifacts

`python3 scripts/run_external_comparison.py --target qr-incompatible-ls`
regenerated:

| Artifact | Path |
| --- | --- |
| Project observations | `build/comparison/qr_incompatible_ls/project_observations.tsv` |
| Baseline observations | `build/comparison/qr_incompatible_ls/baseline_observations.tsv` |
| Dependency status | `build/comparison/qr_incompatible_ls/dependency_status.tsv` |
| Study rows | `build/comparison/qr_incompatible_ls/study.tsv` |
| Summary | `build/comparison/qr_incompatible_ls/summary.md` |
| Manifest | `build/comparison/qr_incompatible_ls/manifest.tsv` |

## Study Inspection

The regenerated study contains six rows and every row reports `pass`.

| Metric | Expected | Project | Baseline | Delta | Status |
| --- | --- | --- | --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | `SPARSE_SUCCESS` | | | `pass` |
| `baseline_status` | `success` | | `success` | | `pass` |
| `residual_norm` | `1.7320508075688772` | `1.7320508075688772` | `1.7320508075688772` | `0` | `pass` |
| `solution_norm` | `2.2360679774997898` | `2.2360679774997894` | `2.2360679774997894` | `0` | `pass` |
| `solution_values` | `2,-1` | `1.9999999999999996,-1.0000000000000002` | `1.9999999999999998,-1` | `2.2204460492503131e-16` | `pass` |
| `project_vs_baseline_max_abs_delta` | `<=1e-10` | `1.9999999999999996,-1.0000000000000002` | `1.9999999999999998,-1` | `2.2204460492503131e-16` | `pass` |

## Manifest Inspection

The regenerated manifest records:

- `target=qr-incompatible-ls`;
- `fixture_key=qr_overdetermined_incompatible_4x2`;
- `baseline_helper_path=tests/qr_external_dense_reference.py`;
- `baseline_type=external-process-source-controlled-helper`;
- `configuration=stage=sprint191_day8_comparison_logic;baseline_status=integrated_and_compared;support_tier=local_only`;
- `source_branch=sprint-191`;
- `worktree_state=dirty`;
- `study_path=build/comparison/qr_incompatible_ls/study.tsv`.

The dirty worktree state is expected during sprint-local validation because
the generated artifacts are local ignored outputs produced before the branch
changes are committed.

## Dependency Inspection

`dependency_status.tsv` records required dependencies as pass evidence and
optional external packages as deferred context:

| Dependency | Status | Required | Interpretation |
| --- | --- | --- | --- |
| `python3` | `pass` | `yes` | Current interpreter is available. |
| `tests/qr_external_dense_reference.py` | `pass` | `yes` | Required source-controlled dense QR helper is available. |
| `numpy` | `defer` | `no` | Optional package baseline is not selected. |
| `scipy` | `defer` | `no` | Optional package baseline is not selected. |

## Validation Results

| Command | Result |
| --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Pass |
| `make build/test_qr_solve` | Pass |
| `./build/test_qr_solve` | Pass, 19 tests, 0 failed, 0 skipped, 1104 assertions |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Pass, 46 rows |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `bash scripts/check_qr_header_docs_guard.sh` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `make report-index-comparison-freshness` | Pass, 46 rows |
| `python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py` | Pass |

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 12.

## Remaining Risk

The new family remains Linux/macOS selected comparison freshness only. It does
not promote Windows selected report freshness, package-manager proof, ABI
proof, external-library parity, performance proof, release proof, or
state-of-the-art status.

## Day 13 Handoff

Day 13 should reduce review surface by checking for duplicated constants,
stale wording, overbroad docs, unnecessary workflow churn, and brittle
assertions before closeout.
