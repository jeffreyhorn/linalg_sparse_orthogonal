# Sprint 181 Day 10: Workflow Scope Checks

## Purpose

Day 10 refactors workflow scope guards so selected hosted report lanes consume
workflow expectations from `tests/corpus/manifests/selected_report_targets.tsv`
without widening YAML scans or upload scopes.

The implementation keeps exact job and upload-block checks in the workflow
guard while moving selected target identity, expected rows, workflow job
membership, workflow artifact names, workflow platforms, and generated command
expectations behind the manifest.

## Guard Refactor

Updated `tests/test_selected_comparison_workflow.py` from a comparison-only
guard into a selected report freshness workflow guard.

Manifest-backed workflow fields now include:

| Manifest field | Guard usage |
| --- | --- |
| `workflow_file` | Selects rows relevant to Linux and macOS workflow files. |
| `workflow_job` | Selects rows relevant to exact workflow job IDs. |
| `workflow_artifact` | Drives expected upload artifact names, including platform-specific comparison artifacts. |
| `workflow_platforms` | Maps Linux and macOS comparison rows to the correct upload artifact. |
| `target_key` | Drives selected comparison target tuple checks. |
| `artifact_pattern` | Derives selected comparison directories and selected benchmark artifact path. |
| `required_files` | Drives exact selected comparison upload path checks. |
| `expected_rows` | Drives selected comparison summary tuple checks. |
| `expected_row_ids` | Drives selected benchmark row identity checks. |
| `generator_command` | Drives selected oracle command checks. |

## Scoped Workflow Coverage

The guard now validates these hosted lanes:

| Workflow | Job | Selected families |
| --- | --- | --- |
| `.github/workflows/ci.yml` | `generated-report-freshness` | oracle and comparison |
| `.github/workflows/ci.yml` | `hosted-performance-freshness` | benchmark |
| `.github/workflows/macos-ci.yml` | `selected-comparison-freshness` | comparison |

The guard also validates `.github/workflows/windows-ci.yml` remains outside
selected report freshness. Windows must not run selected oracle, comparison,
or benchmark freshness commands or upload selected freshness artifacts.

## Exact Block Checks

The test now extracts exact job blocks before checking report commands and
upload steps. Upload checks remain scoped to the specific
`actions/upload-artifact@v4` block for the manifest-owned artifact name.

Checked fail-closed behavior includes:

- required `if-no-files-found: error`;
- exact selected comparison paths for every manifest-owned required file;
- no broad `build/comparison/**` upload path;
- exact selected oracle upload block and no broad generated-report upload;
- exact selected benchmark artifact plus current context files and no broad
  benchmark-report upload;
- Linux workflow guard remains in `build-and-test`, outside the validated
  `generated-report-freshness` job.

## Preserved Boundaries

The guard preserves these non-claims:

- no Windows selected report freshness;
- no broad generated report upload;
- no broad comparison or benchmark publication;
- no package/ABI support from selected report lanes;
- no external-library parity claim;
- no performance superiority or state-of-the-art claim.

Workflow YAML structure remains guard-owned. The manifest identifies selected
rows, exact workflow jobs, platforms, and upload artifact names; the test still
owns job-boundary parsing, upload-action placement, fail-closed upload
settings, and broad-upload rejection.

## Validation

Validation run:

- `python3 tests/test_selected_comparison_workflow.py`
- `python3 -m py_compile tests/test_selected_comparison_workflow.py`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Workflow guards use manifest-owned expectations without widening YAML scan scope. | Complete | The guard loads selected target rows by `workflow_file` and `workflow_job`, then checks exact job and upload blocks. |
| Missing or broadened workflow report lanes fail clearly. | Complete | Missing jobs, upload artifact names, required paths, commands, and broad upload globs raise targeted assertion messages. |
| Selected hosted/local report boundaries remain explicit. | Complete | Linux/macOS selected lanes are manifest-backed; Windows selected report freshness remains rejected. |
