# Sprint 199 Day 13 Integrated Validation

## Purpose

Day 13 runs the applicable Sprint 199 validation gates before closeout and
records the quality-gate decision for the changed file set.

## Changed-Surface Review

Tracked source changes at Day 13 are limited to documentation plus Python
normalizer/validator tests and implementation:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_18/PROJECT_PLAN.md`
- `scripts/normalize_report_index.py`
- `tests/corpus/README.md`
- `tests/test_normalize_report_index.py`
- `tests/test_validate_windows_powershell.py`

No `.c` or `.h` files are modified, so `make format && make lint && make test`
is not required by the sprint quality-check rule for Day 13. Focused Python,
docs, report freshness, package/static deferral, and whitespace gates are the
applicable local checks.

## Command Results

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest invariants still agree with Windows re-deferral. |
| `python3 tests/test_normalize_report_index.py` | Passed | Selected-target filtering, Windows path normalization, missing/stale diagnostics, and CLI misuse coverage pass. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison runner coverage passes after selected Cholesky gate work. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Workflow markers, claim boundaries, fake PowerShell pass paths, local unavailable behavior, and hosted-required fail-closed behavior pass. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Windows selected Cholesky workflow remains target/artifact/path scoped. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Passed | Six selected Cholesky generated rows were fresh to current `HEAD`; output remained advisory and selected-target scoped. |
| `make report-index-comparison-freshness` | Passed | Regenerated selected local comparison outputs and ended with `passed (local-only generated comparison freshness)`. |
| `make docs-check` | Passed | Doxygen generation and API docs coverage completed. |
| `make windows-powershell-validate` | Exit `2` | Structural checks passed; local `pwsh` is unavailable, which is recorded as unavailable local evidence, not pass evidence. Hosted `--require-pwsh` remains the pass/fail owner. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager public non-claims and selected Homebrew local proof boundary remain intact. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static package deferral and Windows package non-claim wording remain intact. |
| `git diff --check` | Passed | No whitespace errors. |

## Generated Artifact Review

`make report-index-comparison-freshness` regenerated ignored local comparison
outputs under `build/`. `make docs-check` regenerated ignored Doxygen output
under `docs/api/`.

`git status --short --ignored build docs/api/html` reported both generated
trees as ignored:

```text
!! build/
!! docs/api/
```

`git ls-files --others --exclude-standard` reported only Sprint 199 planning
files and artifacts as untracked source-controlled candidates. No generated
report, Doxygen HTML, workflow cache, or proof output is intended for staging.

## Residuals

- Selected Windows Cholesky freshness remains re-deferred in the manifest.
- Generated comparison rows still carry `support_tier=local_only`.
- Generated summary/non-claim wording still does not promote Windows.
- Local `make windows-powershell-validate` cannot provide pass evidence without
  `pwsh`; hosted Windows `--require-pwsh` remains the authoritative pass/fail
  path for PowerShell parseability.
- Day 14 still owns final artifact consistency review and retrospective input
  preparation.
