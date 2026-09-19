# Sprint 205 Day 13 Integrated Validation

## Scope

Day 13 validated the Sprint 205 support matrix, adoption quick-reference,
diagnostics vocabulary, package/static install non-claims, Windows claim
boundaries, selected-performance docs, generated API routing, and formatting
state for the current branch.

No `.c` or `.h` files were changed, so the full C gate
(`make format && make lint && make test`) was not required for this
documentation and guard-only validation pass.

## Changed-File Inventory

Tracked modified files:

- `Makefile`
- `README.md`
- `benchmarks/README.md`
- `docs/api_reference.md`
- `docs/cookbook.md`
- `docs/maintainer_guide.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`
- `examples/README.md`
- `scripts/check_api_docs_routing.py`
- `scripts/package_manager_deferral_check.sh`
- `scripts/static_package_deferral_check.sh`
- `scripts/validate_windows_powershell.py`
- `tests/test_api_docs_routing.py`
- `tests/test_validate_windows_powershell.py`

Untracked Sprint 205/support guard files:

- `docs/planning/EPIC_18/SPRINT_205/PLAN.md`
- `docs/planning/EPIC_18/SPRINT_205/WORKING_NOTES.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day1-support-intake.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day2-public-doc-audit.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day3-maintainer-report-audit.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day4-quick-reference-design.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day5-support-truth-architecture.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day6-quick-reference-implementation.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day7-support-truth-consolidation.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day8-example-workflow-routing.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day9-diagnostics-vocabulary-design.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day10-diagnostics-vocabulary-implementation.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day11-claim-guard-design.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day12-claim-guard-implementation.md`
- `docs/planning/EPIC_18/SPRINT_205/artifacts/day13-integrated-validation.md`
- `tests/test_support_quick_reference_docs.py`

Generated-output tracking state:

- `docs/api/` exists locally after Doxygen generation.
- `git status --ignored --short docs/api` reports `!! docs/api/`.
- No generated API output is staged or tracked.

## Validation Commands

| Command | Result | Notes |
| --- | --- | --- |
| `git diff --name-only -- '*.c' '*.h'` | Passed | No changed C or header files. |
| `git ls-files --others --exclude-standard -- '*.c' '*.h'` | Passed | No untracked C or header files. |
| `git diff --check` | Passed | No whitespace errors. |
| `python3 tests/test_support_quick_reference_docs.py` | Passed | Sprint 205 support/quick-reference markers and overclaim regressions passed. |
| `make support-docs-guard` | Passed | Makefile wrapper for the Sprint 205 guard passed. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Selected-performance wording remained bounded. |
| `python3 tests/test_api_docs_routing.py` | Passed | API routing regression suite passed. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Local-only generated API guard regressions passed. |
| `make api-docs-freshness` | Passed | Doxygen generation, coverage, local-only, and routing checks passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and local Homebrew proof boundary passed after marker alignment. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static package/shared-library deferral and Windows package non-claims passed after marker alignment. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Windows claim-boundary and selected Cholesky workflow guards passed; local `pwsh` remained unavailable and was not counted as pass evidence. |

## Fixes Made During Validation

- Re-anchored `scripts/package_manager_deferral_check.sh` to the Sprint 205
  README wording for `package-manager distribution` and the developer-mode
  local static source Homebrew formula proof.
- Re-anchored `scripts/static_package_deferral_check.sh` to the Sprint 205
  README wording for shared-library/dynamic ABI deferral and Windows
  Makefile/`pkg-config` non-claims.
- Re-anchored `scripts/validate_windows_powershell.py` and its regression
  fixture to the current README Windows selected-freshness and QR deferral
  wording.
- Restored the Sprint 170 shared-library/ABI product-decision link in the
  README static package summary.

## Completion Criteria

- Item 205.6 has current integrated validation evidence.
- Required docs, support, package, static, Windows, selected-performance, and
  generated API checks passed.
- The full C gate was correctly skipped because no C or header files changed.
- Generated Doxygen output remained ignored local output.
