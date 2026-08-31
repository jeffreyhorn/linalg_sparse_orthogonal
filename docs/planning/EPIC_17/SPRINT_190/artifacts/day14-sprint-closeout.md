# Sprint 190 Day 14: Sprint Closeout

## Purpose

Close Sprint 190 with the selected outcome, implementation summary, validation
evidence, residual decision, and review handoff for the Windows selected report
freshness decision.

## Sprint Outcome

Sprint 190 ends with **bounded workflow path implemented, residual narrowed**.

The sprint selected `cholesky-spd-tridiag-5` as the smallest credible Windows
selected report freshness candidate and wired a hosted Windows workflow path
for that exact selected comparison lane. The sprint does not claim broad
Windows report freshness and does not promote the selected target manifest to
`windows` metadata until hosted Windows evidence is reviewed.

## Implemented Surfaces

| Surface | Closeout Status |
| --- | --- |
| Windows workflow | Added `selected-comparison-freshness` on `windows-2022` with a 20-minute timeout. |
| Generator path | Added/validated CMake probe support for selected Cholesky comparison generation. |
| Freshness guard | Added target-specific selected comparison freshness validation with `--selected-target cholesky-spd-tridiag-5`. |
| Artifact policy | Uploads only the six required selected Cholesky comparison files as `sprint190-windows-selected-comparison-cholesky`. |
| Windows PowerShell validation | Allows only the bounded selected Cholesky workflow path while keeping other selected report freshness claims blocked. |
| Manifest tests | Preserve current no-`windows` source metadata and test the future exact Cholesky-only promotion shape. |
| Public docs | State one bounded workflow path and retain non-claims for broad Windows report freshness, selected oracle freshness, selected benchmark freshness, package-manager support, shared-library support, dynamic ABI, runtime-loader behavior, broad platform parity, performance superiority, and state-of-the-art status. |
| Residual queue | Renewed and narrowed `R186-WIN-REPORT-FRESHNESS` to hosted evidence plus manifest-promotion review. |

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Workflow contract and negative selected-lane drift checks pass. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Manifest invariants and future exact Windows Cholesky metadata allowlist tests pass. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus schema and report-index field docs validate. |
| `python3 tests/test_validate_windows_powershell.py` | Pass | PowerShell validation unit coverage passes, including unavailable and `--require-pwsh` cases. |
| `python3 tests/test_normalize_report_index.py` | Pass | Report normalization and selected-target freshness regression coverage passes. |
| `python3 tests/test_run_external_comparison.py` | Pass | External comparison and CMake probe regression coverage passes. |
| `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake` | Pass | Local CMake-probe selected Cholesky comparison bundle regenerated. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Pass | Six generated selected Cholesky rows are fresh against current `HEAD`. |
| `make windows-powershell-validate` | Unavailable | Structural checks passed; local `pwsh` is unavailable, so the target exited 2 by design. |
| `git diff --check` | Pass | No whitespace errors in changed files. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No C or header files changed. |
| `git status --short --ignored build/comparison/cholesky_spd_tridiag_5` | Pass | Generated report evidence remains ignored as `build/` output. |

No `.c` or `.h` files changed during Sprint 190, so the full
`make format && make lint && make test` C gate is not required.

## Final Residual Status

`R186-WIN-REPORT-FRESHNESS` remains open as a narrowed residual.

Closed within Sprint 190:

- candidate selection for a bounded Windows selected freshness lane;
- hosted workflow wiring for one selected Cholesky comparison lane;
- Windows-safe CMake probe command shape;
- target-specific freshness validation;
- selected workflow, manifest, and PowerShell guard coverage;
- public and maintainer claim calibration.

Still pending:

- hosted `windows-2022` pass evidence for `selected-comparison-freshness`;
- review of the hosted artifact
  `sprint190-windows-selected-comparison-cholesky`;
- selected manifest promotion to include `windows` only for
  `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, if hosted evidence supports it.

## Review Handoff

Review should focus on:

1. whether the hosted Windows workflow command is sufficiently bounded and
   portable;
2. whether the artifact upload list is narrow enough;
3. whether local unavailable `pwsh` semantics are clear;
4. whether docs avoid claiming reviewed Windows selected freshness before CI
   evidence exists;
5. whether the residual queue correctly keeps manifest promotion separate from
   workflow wiring.

## Retrospective Inputs

- Outcome: bounded workflow path implemented, residual narrowed.
- Primary remaining blocker: hosted Windows evidence and manifest promotion.
- Validation status: all feasible local checks pass; local PowerShell Make
  wrapper reports expected unavailable semantics.
- Generated evidence policy: local `build/` output remains ignored and is not
  committed.
- Next sprint candidate: review hosted Windows evidence and either promote the
  exact selected Cholesky row or retain the staged residual.

