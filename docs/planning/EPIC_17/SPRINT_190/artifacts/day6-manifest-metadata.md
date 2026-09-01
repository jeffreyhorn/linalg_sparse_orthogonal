# Sprint 190 Day 6: Manifest Metadata

## Purpose

Define how selected report manifest metadata should represent the Sprint 190
Windows Cholesky decision, while avoiding premature `windows` metadata before
the Windows-safe generator and workflow evidence exist.

## Day 6 Decision

Do not edit `tests/corpus/manifests/selected_report_targets.tsv` yet.

The selected Cholesky row remains Linux/macOS only because the branch has not
implemented:

- Windows-safe generated comparison probe build/link;
- `.exe`-aware probe execution;
- hosted Windows selected comparison workflow job;
- exact Windows artifact upload;
- hosted Windows freshness evidence.

Adding `windows` to the manifest before those pieces exist would make the
structured metadata stronger than the evidence.

## Current Cholesky Metadata

| Field | Current value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `required_files` | `project_observations.tsv;baseline_observations.tsv;dependency_status.tsv;study.tsv;summary.md;manifest.tsv` |
| `expected_rows` | `6` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |

## Future Metadata Patch

When the Windows lane is executable, mutate only
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`:

| Field | Required future value |
| --- | --- |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml;.github/workflows/windows-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness;sprint190-windows-selected-comparison-cholesky` |
| `workflow_platforms` | `linux;macos;windows` |

Retain these fields unless the generator output contract changes:

- `expected_rows=6`;
- all six `expected_row_ids`;
- `artifact_pattern=build/comparison/cholesky_spd_tridiag_5/study.tsv`;
- the six required files;
- narrow claim scope and non-claims.

No other selected row may gain `windows` in Sprint 190.

## Schema Findings

Current schema validation already catches general metadata drift:

- missing selected artifact patterns;
- invalid expected row counts;
- missing row IDs for countable selected targets;
- missing generator commands or required files for generated selected targets;
- missing hosted metadata groups;
- mismatched artifact/platform counts;
- duplicate hosted artifact keys spanning report families.

The schema does not currently encode a one-row Windows allowlist. That is
acceptable on Day 6 because the deferral is still active and the no-Windows
policy is enforced by focused tests and Windows workflow guards.

## Test Update Contract

When promotion becomes live, update manifest tests to:

1. Require a Sprint 190 decision marker before Windows selected metadata is
   allowed.
2. Allow `windows` only for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.
3. Require `.github/workflows/windows-ci.yml`,
   `selected-comparison-freshness`, and
   `sprint190-windows-selected-comparison-cholesky` to appear in the same
   platform position.
4. Reject `windows` on QR, LU, partial-SVD, oracle, and benchmark selected
   rows.
5. Reject broad or reused Windows artifact names.
6. Keep deferral tests active for every unselected Windows report freshness
   lane.

## Day 6 Outcome

The manifest remains unchanged and valid. The future metadata patch is exact
and reviewable, but it is blocked until the generator and workflow produce
real Windows evidence.

## Validation

Commands run:

- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 scripts/validate_corpus_schema.py`
- targeted reads of the Cholesky manifest row and schema/test guard code

Both validation commands passed.

Day 6 changed only planning documentation. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.
