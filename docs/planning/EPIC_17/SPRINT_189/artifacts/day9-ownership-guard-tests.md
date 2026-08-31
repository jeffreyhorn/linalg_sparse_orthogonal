# Sprint 189 Day 9: Ownership Guard Tests

## Purpose

Add guard coverage that fails on stale PowerShell validation ownership,
unsupported Windows report artifact publication, local unavailable wording
drift, or documentation claims that treat PowerShell validation as Windows
report freshness evidence.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added a claim-boundary guard over README, INSTALL, maintainer guide, and corpus README Windows/PowerShell non-claim anchors. |
| `scripts/validate_windows_powershell.py` | Added unsupported Windows/PowerShell promotion phrase detection. |
| `scripts/validate_windows_powershell.py` | Added a Windows workflow artifact-publication guard while selected Windows report evidence remains absent. |
| `tests/test_validate_windows_powershell.py` | Added tests for claim markers, promotion wording, Windows upload-artifact drift, and unavailable non-pass wording. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 9 guard ownership and expected failure modes. |

## Guarded Surfaces

The owned validator now checks:

| Surface | Guard |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Must not add `actions/upload-artifact` while selected Windows report evidence is absent. |
| `README.md` | Must retain Windows support non-claims and Sprint 182 Windows report freshness deferral wording. |
| `INSTALL.md` | Must retain Windows Makefile, `pkg-config`, runtime-loader, and broad Windows parity non-claims. |
| `docs/maintainer_guide.md` | Must retain Sprint 182 deferral wording and local unavailable PowerShell non-pass interpretation. |
| `tests/corpus/README.md` | Must retain selected-target, Windows deferral, and unavailable local PowerShell non-pass interpretation. |

## Drift Tests

| Test | Failure protected |
| --- | --- |
| `test_windows_upload_artifact_fails_while_selected_evidence_absent` | Windows workflow artifact publication without selected Windows report evidence. |
| `test_claim_boundaries_validate_current_docs` | Current claim-boundary anchors across maintained docs. |
| `test_claim_boundary_missing_marker_fails_clearly` | Removal of the README Windows report freshness deferral marker. |
| `test_claim_boundary_promotion_wording_fails_clearly` | Documentation that says PowerShell validation proves Windows report freshness. |
| `test_unavailable_output_keeps_non_pass_evidence_wording` | Local unavailable output losing the required non-pass evidence wording. |

Existing Day 5 through Day 8 tests still guard selected PowerShell step
ownership, hosted fail-closed wiring, selected artifact name blockers, fake
PowerShell parse behavior, and local unavailable exit codes.

## Claim Boundary

PowerShell validation ownership can prove that selected Windows workflow
PowerShell snippets parse under `pwsh` and that the Windows workflow retains
the reviewed structural contract. It still cannot prove:

- generated Windows report freshness;
- selected Windows report artifact publication;
- Windows-safe report generator execution;
- broad Windows parity;
- Windows Makefile or `pkg-config` execution parity;
- package-manager support;
- shared-library, dynamic ABI, DLL/import-library, or runtime-loader support.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Ownership, claim-boundary, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point reports unavailable evidence locally. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 10 Handoff

Day 10 should update maintainer-facing documentation to explain the new
`make windows-powershell-validate` command, the hosted
`--require-pwsh` lane, the local unavailable exit `2`, and the retained
non-claims now guarded by Day 9.

## Validation Scope

Day 9 changed Python tests, a Python script, and planning documentation. No
`.c` or `.h` files were modified, so `make format && make lint && make test`
is not required.
