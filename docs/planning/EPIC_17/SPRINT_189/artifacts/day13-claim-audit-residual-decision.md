# Sprint 189 Day 13: Claim Audit and Residual Decision

## Purpose

Decide the Sprint 189 final claim state by comparing the implemented Windows
PowerShell validation owner, hosted wiring, guard coverage, documentation, and
validation evidence against the Sprint 187 Windows acceptance gates.

## Sprint 187 Gate Comparison

| Requirement | Evidence | Day 13 decision |
| --- | --- | --- |
| PowerShell surface inventory | Day 1 and Day 2 artifacts inventory Windows workflow `shell: pwsh` steps, selected report metadata, artifact names, and claim boundaries. | Complete. |
| Validation command | `scripts/validate_windows_powershell.py` parses selected PowerShell snippets when `pwsh` is available and returns explicit unavailable evidence when absent. | Complete. |
| Hosted ownership | `.github/workflows/windows-ci.yml` has `powershell-validation` on `windows-2022` running `python scripts/validate_windows_powershell.py --require-pwsh`. | Source-controlled wiring complete; hosted pass evidence awaits PR CI execution. |
| Local skip semantics | Local default command exits `2` with `local unavailable PowerShell is not pass evidence` when `pwsh` is absent. | Complete. |
| Report freshness boundary | Validator and workflow guards keep selected report generators, selected artifact uploads, and manifest `windows` platforms out of Sprint 189. | Complete. |
| Docs alignment | README, INSTALL, maintainer guide, corpus README, workflow comments, and validator claim-boundary markers separate validation ownership from report freshness. | Complete. |

## Final Sprint 189 Claim State

Sprint 189 closes source-controlled PowerShell validation ownership.

The closed claim is narrow: selected Windows workflow PowerShell snippets now
have an owned validator, Make entry point, fake-`pwsh` local test coverage,
hosted fail-closed workflow wiring, documentation, and drift guards.

The remaining hosted evidence item is operational, not a source-code residual:
the new `powershell-validation` job must pass in PR CI after the branch is
pushed. Until that hosted run passes, do not cite hosted Windows PowerShell
parse success as observed evidence.

## Retained Residuals

| Residual | Status | Owner |
| --- | --- | --- |
| Local `pwsh` not installed on this machine | Retained environment residual; default validator exits `2`. | Local developer environment. |
| Hosted Windows PowerShell validation run evidence | Pending PR CI after branch push. | Sprint 189 PR workflow. |
| Windows report freshness | Still formally deferred by Sprint 182. | Sprint 190. |
| Selected Windows report artifact publication | Not implemented and explicitly guarded against. | Sprint 190 only if promotion is selected. |
| Windows report generator execution | Not implemented. | Sprint 190 promotion or renewed-deferral decision. |
| Broad Windows parity, Windows Makefile parity, Windows `pkg-config` execution parity | Not claimed. | Out of Sprint 189 scope. |
| Package-manager, shared-library, dynamic ABI, DLL/import-library, runtime-loader support | Not claimed. | Out of Sprint 189 scope. |

## Claim Audit Results

| Surface | Audit result |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Adds only validation ownership; no selected report generator or upload promotion. |
| `README.md` | Mentions hosted PowerShell validation ownership and states it is workflow validation only. |
| `INSTALL.md` | Adds hosted PowerShell validation ownership to the Windows row and keeps report freshness/non-parity exclusions. |
| `docs/maintainer_guide.md` | Documents local/hosted command semantics, exit codes, and retained non-claims. |
| `tests/corpus/README.md` | States the hosted Windows PowerShell lane is not selected report freshness or artifact publication evidence. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Still has no selected `windows` workflow platform rows. |

## Unsupported Claim Scan

Day 13 scanned touched Windows/report surfaces for unsupported promotion
phrases covering Windows report freshness support/promotion/closure,
PowerShell validation proving freshness, local unavailable PowerShell as pass
evidence, and selected Windows report artifact publication.

Result: no unsupported promotion phrase was found.

## Fresh Validation Evidence

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Guard coverage for ownership, hosted wiring, claim boundaries, fake PowerShell, and unavailable wording passes. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| Unsupported claim `rg` scan | No matches | Touched Windows/report surfaces do not contain the scanned unsupported promotion phrases. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected report workflow guard still rejects Windows selected freshness commands/uploads. |

Day 12 remains the integrated validation record for the broader schema,
report-index, docs, and hygiene checks.

## Sprint 190 Handoff

Sprint 190 starts from a cleaner boundary: PowerShell validation ownership is
implemented, while Windows report freshness remains a separate product
decision. Sprint 190 must either promote exactly one Windows-safe selected
freshness lane with manifest/artifact/freshness/docs guards, or renew the
formal deferral with stronger blockers and revisit criteria.

## PR Summary Inputs

- Added `make windows-powershell-validate`.
- Added `scripts/validate_windows_powershell.py`.
- Added `tests/test_validate_windows_powershell.py`.
- Added hosted `powershell-validation` job in `.github/workflows/windows-ci.yml`.
- Updated README, INSTALL, maintainer guide, and corpus README with bounded
  Windows PowerShell validation wording.
- Preserved Windows report freshness deferral and selected artifact
  non-publication.

## Validation Scope

Day 13 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
