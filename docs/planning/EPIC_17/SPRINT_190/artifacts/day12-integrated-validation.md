# Sprint 190 Day 12: Integrated Validation

## Purpose

Run the integrated Sprint 190 validation surface across workflow guards,
selected report manifest checks, Windows PowerShell ownership, generated report
freshness, claim-boundary documentation, and source hygiene before the final
audit.

## Validation Results

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Verified the Windows workflow exposes only the bounded selected Cholesky comparison freshness lane and retains guard coverage for unsupported selected report surfaces. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Verified selected manifest references, row counts, and continued absence of `windows` in source manifest platform metadata. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Validated corpus schema and report-index field documentation. |
| `python3 tests/test_validate_windows_powershell.py` | Pass | Verified hosted Windows PowerShell wiring, claim boundaries, parse behavior, unavailable local PowerShell semantics, and strict `--require-pwsh` failure behavior. |
| `make windows-powershell-validate` | Unavailable | Structural checks passed, then the target returned exit 2 because `pwsh` is not installed locally. This is explicit unavailable evidence, not hosted pass evidence. |
| `python3 tests/test_normalize_report_index.py` | Pass | Verified target-specific freshness, stale-output failures, row-set checks, and source-controlled advisory handling. |
| `python3 tests/test_run_external_comparison.py` | Pass | Verified external comparison generation, CMake probe metadata, dependency handling, and failure cases. |
| `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake` | Pass | Regenerated the selected Cholesky comparison bundle locally through the CMake probe path. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Pass | Reported six generated selected Cholesky rows as fresh against current `HEAD`. |
| `git diff --check` | Pass | No whitespace errors in changed source, documentation, workflow, or planning files. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No changed C or header files were present, so the full C quality gate was not required. |
| `git status --short --ignored build/comparison/cholesky_spd_tridiag_5` | Pass | Confirmed generated comparison evidence remains ignored as `build/` output. |

## Source Change Scope

Changed source surfaces remain limited to:

- Windows CI workflow wiring;
- install, README, maintainer, corpus, and schema documentation;
- report-index normalization;
- external comparison generation;
- Windows PowerShell validation;
- Python tests for selected workflows, manifests, freshness, generator output,
  and PowerShell validation.

No `.c` or `.h` files changed during Sprint 190 Day 12.

## Integrated Findings

The integrated checks confirm that Sprint 190 has a coherent bounded workflow
implementation:

- `selected-comparison-freshness` is present on `windows-2022`;
- the workflow builds through CMake/MSVC before running the selected generator;
- only `cholesky-spd-tridiag-5` is allowed in the Windows selected comparison
  freshness path;
- the uploaded artifact name remains
  `sprint190-windows-selected-comparison-cholesky`;
- the source manifest still does not list `windows` for selected report
  targets;
- unsupported broad Windows report freshness, selected oracle freshness, and
  selected benchmark freshness claims remain guarded;
- local PowerShell unavailability is reported distinctly from hosted pass
  evidence.

## Residual Risks for Day 13

- Hosted `windows-2022` execution has not been observed from this local
  validation pass.
- Manifest promotion remains staged because the source manifest still omits
  `windows`.
- Local CMake-probe generated evidence is macOS evidence and is not a
  substitute for hosted Windows artifact evidence.
- Documentation should continue to describe the lane as a bounded workflow
  path until hosted CI evidence and manifest promotion are reviewed together.

