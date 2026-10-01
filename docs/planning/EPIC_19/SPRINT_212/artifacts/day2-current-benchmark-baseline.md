# Sprint 212 Day 2: Current Benchmark Baseline

## Summary

Day 2 captures the current selected benchmark workflow, metadata, guard
behavior, and threshold blockers before the Sprint 212 policy decision. The
baseline confirms that the current selected benchmark evidence is fresh and
well-described, but intentionally threshold-free.

## Commands Run

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_performance_docs.py` | Passed: `test-selected-performance-docs: ok`. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed: `test-selected-report-targets-manifest: ok`. |
| `make bench-canonical-report-freshness` | Passed: generated `build/bench-reports/canonical/` and validated the selected local `bench_refactor_csc` row. |

The generated benchmark files are ignored build artifacts and were not added
to the branch. They were inspected only to capture the current local baseline.

## Selected Target Baseline

| Manifest field | Current value |
| --- | --- |
| `target_id` | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| `family` | `benchmark` |
| `subfamily` | `canonical` |
| `target_key` | `bench_refactor_csc` |
| `row_meaning` | selected canonical benchmark freshness for nos4 repeat-one workload |
| `selection_scope` | `hosted_selected` |
| `support_tier` | `hosted_selected` |
| `freshness_policy` | `generated_local_advisory` |
| `generator_command` | `make bench-canonical-report-freshness` |
| `artifact_pattern` | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| `required_files` | `bench_refactor_csc.csv;index.tsv;manifest.txt` |
| `expected_rows` | `1` |
| `expected_row_ids` | `bench_refactor_csc` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `hosted-performance-freshness;selected-performance-freshness` |
| `workflow_artifact` | `sprint168-selected-performance-freshness;sprint202-macos-selected-performance-freshness` |
| `workflow_platforms` | `linux;macos` |
| `claim_scope` | Selected canonical benchmark report metadata is fresh for `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1` with threshold-free methodology fields on reviewed Linux and macOS hosted lanes. |
| `non_claims` | No portable performance claim; no release benchmark claim; no algorithmic superiority claim; no platform parity; no state-of-the-art claim; no package or ABI support claim; no broad package-manager distribution claim; no Windows selected benchmark freshness. |

## Local Generated Metadata Snapshot

The local Day 2 freshness run produced four canonical report rows and exactly
one selected `bench_refactor_csc` row.

| Field | Observed local value |
| --- | --- |
| `report_label` | `unlabeled` |
| `runner_context` | `local` |
| `build_flags` | `not_recorded` |
| `cpu_model` | `unknown` |
| `build_mode` | `serial` |
| `omp_num_threads` | `unset` |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `status` | `measurement` |
| `support_tier` | `local_only` |
| `claim_boundary` | `local_threshold_free` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |
| `backend_context` | `n/a` |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |

The selected CSV row is a local measurement sample only. Its timing columns
are not evidence for a portable threshold because the generated metadata
explicitly records no warmup, no variance computation, no baseline, and no
threshold.

## Hosted Workflow Baseline

| Platform | Job | Key metadata | Interpretation |
| --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml::hosted-performance-freshness` | `sprint-168-hosted-performance`, `hosted_selected`, `hosted_selected_threshold_free`, `github-actions-ubuntu-latest`, `default_make_flags`, `serial`, CPU model from `/proc/cpuinfo`. | Reviewed hosted freshness for the exact selected row only. |
| macOS | `.github/workflows/macos-ci.yml::selected-performance-freshness` | `sprint202-macos-hosted-performance`, `hosted_selected`, `hosted_selected_threshold_free`, `github-actions-macos-latest`, `default_make_flags`, `serial`, CPU model from `sysctl`. | Reviewed hosted freshness for the exact selected row only. |

Both hosted jobs run `make bench-canonical-report`, then run
`scripts/check_bench_canonical_freshness.py --mode hosted`, then upload only
the selected freshness artifacts. Their comments explicitly exclude timing
thresholds, portable performance claims, broad benchmark-family publication,
package/ABI claims, and state-of-the-art claims.

## Documentation Baseline

| Surface | Baseline wording |
| --- | --- |
| `README.md` | Hosted selected-performance lanes check artifact presence, selected row identity, methodology metadata, manifest agreement, and `hosted_selected_threshold_free` claim boundaries; they do not compare timing values or set a regression threshold. |
| `INSTALL.md` | Linux/macOS selected performance freshness is hosted evidence with no portable performance, timing threshold, release benchmark, platform parity, package/ABI proof, broad package-manager distribution, or state-of-the-art claim. |
| `benchmarks/README.md` | Canonical selected freshness is threshold-free with `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`; local sentinel thresholds are separate. |
| `docs/maintainer_guide.md` | The selected freshness lane should remain `baseline=n/a`, `threshold=n/a`, and `status=measurement` until a future sprint records hosted-runner baseline, variance model, tolerance, and same-machine comparison policy. |
| `tests/corpus/README.md` | Selected performance target is threshold-free methodology/freshness evidence only. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index schema documents the selected benchmark target as measurement status with no baseline or threshold. |

## Threshold Eligibility Gaps

| Gap | Current evidence | Threshold impact |
| --- | --- | --- |
| Repeat count | Selected row uses `configured_repeat_1`. | Single-sample evidence cannot support variance-aware thresholding. |
| Warmup | `warmup=none_configured`. | No warmup-controlled timing series exists. |
| Variance | `variance=not_computed_single_sample`. | No statistical threshold can be justified from current selected evidence. |
| Baseline | `baseline=n/a`. | There is no selected canonical baseline to compare against. |
| Threshold | `threshold=n/a`. | The current policy intentionally rejects a selected canonical timing threshold. |
| Runner stability | Hosted jobs capture CPU model, but maintainer docs state GitHub-hosted CPU assignment can vary. | Hosted thresholding needs a same-machine, machine-class, or smoke-ceiling policy before implementation. |
| Existing local threshold separation | S5/S6 local sentinels are separate from hosted canonical selected freshness. | Existing local thresholds cannot be promoted as hosted selected freshness without a Sprint 212 decision and guard updates. |

## Day 2 Outcome

Item 212.1 now has evidence-backed current-state data. The baseline clearly
distinguishes local generated evidence from Linux/macOS hosted selected
freshness evidence, and it records the metadata gaps that currently block a
normal timing threshold. Day 3 should deepen runner and variance analysis
before Day 4 criteria and the Day 5 policy decision.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
