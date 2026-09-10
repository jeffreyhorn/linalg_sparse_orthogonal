# Sprint 202 Day 5: Validator And Manifest Design

## Summary

Day 5 defines the minimal implementation design for the selected macOS hosted
benchmark freshness lane. The design reuses the existing selected benchmark row
and freshness checker instead of adding a new benchmark row or broadening
canonical benchmark publication.

## Selected Implementation Path

| Surface | Planned change |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Update the existing `SRT-BENCH-REFACTOR-CSC-NOS4` row to include macOS workflow file, job, artifact, and platform metadata. |
| `.github/workflows/macos-ci.yml` | Add one reviewed hosted selected performance freshness job. |
| `scripts/check_bench_canonical_freshness.py` | Keep as the hard artifact and metadata freshness authority; avoid workflow-layout coupling here unless required by tests. |
| `tests/test_selected_comparison_workflow.py` | Extend selected workflow coverage to validate the macOS selected performance lane and selected-only upload. |
| `tests/test_bench_canonical_freshness.py` | Add focused macOS hosted metadata or path-drift coverage only where existing hosted tests do not cover the contract. |
| `tests/test_selected_performance_docs.py` | Update claim guards after public docs mention Linux and macOS selected benchmark freshness. |

## Manifest Design

Keep one selected benchmark row:

- `target_id=SRT-BENCH-REFACTOR-CSC-NOS4`;
- `target_key=bench_refactor_csc`;
- `expected_row_ids=bench_refactor_csc`;
- `artifact_pattern=build/bench-reports/canonical/bench_refactor_csc.csv`;
- `required_files=bench_refactor_csc.csv;index.tsv;manifest.txt`.

Change only hosted workflow metadata:

| Field | Planned value |
| --- | --- |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `hosted-performance-freshness;selected-performance-freshness` |
| `workflow_artifact` | `sprint168-selected-performance-freshness;sprint202-macos-selected-performance-freshness` |
| `workflow_platforms` | `linux;macos` |

## Workflow Design

The macOS job should:

- run on `macos-latest`;
- set hosted selected threshold-free metadata through environment variables;
- capture CPU model with `sysctl -n machdep.cpu.brand_string`, falling back to
  `unknown`;
- run `make bench-canonical-report`;
- run `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted`;
- summarize the selected `bench_refactor_csc` row and uploaded paths;
- upload only `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`.

## Validator Design

The existing benchmark freshness checker already validates the selected
artifact contents and threshold-free metadata. It should remain focused on
generated artifact correctness:

- required files exist;
- `index.tsv` schema is valid;
- exactly one selected `bench_refactor_csc` row exists;
- hosted metadata is non-empty and not local placeholders;
- selected row values match the manifest-derived contract;
- selected CSV contents match the selected row;
- `manifest.txt` agrees with `index.tsv`;
- unselected rows remain local-only.

Workflow YAML structure belongs in `tests/test_selected_comparison_workflow.py`,
not in the artifact checker.

## Path Normalization Decision

For the selected macOS lane, keep `relative_path=bench_refactor_csc.csv`.
Because the generator emits basename-only relative paths and macOS uses the
same POSIX-style report directory as Linux, no Windows-style path normalization
is required for Day 6.

Backslash handling remains deferred with Windows selected benchmark freshness.

## Regression Matrix

| Case | Owner |
| --- | --- |
| macOS selected performance job exists and runs the hosted checker | `tests/test_selected_comparison_workflow.py` |
| macOS selected upload contains exactly selected files | `tests/test_selected_comparison_workflow.py` |
| Broad macOS benchmark upload is rejected | `tests/test_selected_comparison_workflow.py` |
| Unselected macOS benchmark upload is rejected | `tests/test_selected_comparison_workflow.py` |
| Manifest lists Linux and macOS metadata for the selected benchmark row | `tests/test_selected_comparison_workflow.py` and selected manifest tests |
| Hosted macOS-style metadata passes selected benchmark checker | `tests/test_bench_canonical_freshness.py` |
| Selected `relative_path` drift is rejected | `tests/test_bench_canonical_freshness.py` |
| Docs mention Linux and macOS only, without portable performance claims | `tests/test_selected_performance_docs.py` |

## Non-Claims Preserved

The design does not claim:

- Windows selected benchmark freshness;
- timing thresholds or regression baselines;
- portable Linux/macOS performance;
- benchmark superiority;
- broad benchmark publication;
- package-manager, ABI, runtime-loader, OpenMP speedup, backend superiority, or
  state-of-the-art status.
