# Sprint 190 Day 4: Decision Record Draft

## Purpose

Draft the Sprint 190 Windows selected report freshness decision and define the
exact implementation contract, manifest changes, guard behavior, failure
diagnostics, and fallback deferral path.

## Draft Decision

Sprint 190 selects a provisional promotion path for exactly one Windows
selected report freshness lane:

`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`

This is not yet an accepted promotion. The promotion becomes accepted only if
Days 5 through 13 implement the Windows-safe generator/probe path, workflow
job, manifest metadata, freshness guards, docs, and hosted Windows evidence.

Until those gates are complete, the Sprint 182 Windows report freshness
deferral remains active.

## Selected Lane Contract

| Field | Value |
| --- | --- |
| Target ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Target key | `cholesky-spd-tridiag-5` |
| Family | `comparison` |
| Subfamily | `cholesky_spd_tridiag_5` |
| Generator | `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5` with a reviewed Windows-safe probe mode or equivalent explicit Windows probe arguments |
| Workflow file | `.github/workflows/windows-ci.yml` |
| Workflow job | `selected-comparison-freshness` |
| Workflow platform | `windows` |
| Workflow artifact | `sprint190-windows-selected-comparison-cholesky` |
| Expected rows | `6` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Artifact root | `build/comparison/cholesky_spd_tridiag_5/` |
| Timeout | A bounded hosted Windows job, target cap `20` minutes or less |
| Upload behavior | Upload exact required files only, with `if-no-files-found: error` |
| Claim scope | Hosted Windows freshness for one fixture-local Cholesky SPD tridiagonal solve comparison |

## Retained Non-Claims

Even if the Cholesky lane is promoted, Sprint 190 must still not claim:

- broad Windows report freshness;
- Windows oracle freshness;
- Windows benchmark freshness;
- Windows selected comparison freshness beyond the one Cholesky row;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library support;
- dynamic ABI support;
- DLL/import-library or runtime-loader support;
- portable performance, performance superiority, or state-of-the-art status.

## Manifest Update Checklist

Only `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` may change.

Required manifest updates if promotion lands:

- add `.github/workflows/windows-ci.yml` to `workflow_file`;
- add `selected-comparison-freshness` to `workflow_job`;
- add `sprint190-windows-selected-comparison-cholesky` to
  `workflow_artifact`;
- add `windows` to `workflow_platforms`;
- keep `expected_rows=6`;
- keep expected row IDs exact and unchanged;
- keep required files exact and unchanged;
- keep non-claims explicit.

No QR, LU, partial-SVD, oracle, or benchmark selected row may gain `windows` in
Sprint 190.

## Workflow Contract

The Windows workflow may add a selected comparison freshness job only if it:

- runs on `windows-2022`;
- builds through CMake/MSVC;
- runs only the Cholesky selected comparison target;
- avoids Makefile and Bash requirements;
- avoids Unix `.a`, `-lm`, and extensionless executable assumptions;
- validates exactly six expected rows;
- uploads only the six Cholesky comparison bundle files;
- uses artifact name `sprint190-windows-selected-comparison-cholesky`;
- uses `if-no-files-found: error`;
- keeps PowerShell validation ownership separate from report freshness.

## Guard Contract

Current guards are broad no-Windows deferral guards. Promotion requires
converting them to exact allowlists:

| Guard | Required conversion |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Permit only the selected Cholesky Windows job/artifact and keep all other selected Windows freshness blocked. |
| `tests/test_validate_windows_powershell.py` | Test the positive Cholesky allowlist and negative unselected-lane drift. |
| `tests/test_selected_comparison_workflow.py` | Verify exact Windows job, target command, artifact name, required files, row count, and non-claims. |
| `tests/test_selected_report_targets_manifest.py` | Allow `windows` only for the Cholesky row when this Sprint 190 decision exists. |
| `scripts/validate_corpus_schema.py` | Retain metadata cardinality checks and reject mismatched artifact/platform lists. Add schema-level Windows allowlist only if needed. |
| `scripts/normalize_report_index.py` | Validate exact selected Cholesky row IDs, required files, source commit freshness, and artifact path once Windows output exists. |

## Failure Diagnostics

The implemented path must fail if:

- the Windows job is missing or runs on the wrong runner;
- the command runs an unselected target or broad freshness wrapper;
- the generator falls back to `make`;
- the project probe links a Unix archive or uses `-lm`;
- the temporary executable path is not `.exe` aware;
- the artifact upload path is broad;
- upload omits `if-no-files-found: error`;
- any required file is absent;
- row IDs or row count differ from the manifest;
- generated `source_commit` is stale when freshness is required;
- the manifest adds `windows` to any row other than the selected Cholesky row;
- docs imply broader Windows support than the evidence proves.

## Fallback Deferral Draft

If a Windows-safe Cholesky path is not implemented, Sprint 190 will renew the
formal deferral. The renewed record should state:

- selected comparison is still blocked by Unix `cc`, `make`, `.a`, `-lm`, and
  extensionless probe execution assumptions;
- no reviewed CMake/MSVC generated-probe helper exists yet;
- no hosted Windows selected comparison artifact upload has passed;
- current guards intentionally keep Windows selected report freshness
  unpromoted;
- revisit criteria should focus on the Cholesky candidate first.

## Day 4 Decision

Proceed to Day 5 with an exact provisional promotion contract for
`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`.

Do not update the manifest, public docs, or workflow to claim Windows report
freshness until the Windows-safe probe mode and allowlist guards are ready.

## Validation

Day 4 changed only planning documentation. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

Run `git diff --check` after this artifact is added.
