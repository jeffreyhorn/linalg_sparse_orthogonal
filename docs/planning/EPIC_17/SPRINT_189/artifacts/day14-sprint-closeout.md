# Sprint 189 Day 14: Sprint Closeout

## Purpose

Package Sprint 189 evidence, retrospective inputs, PR-ready notes, validation
results, and retained non-goals after completing PowerShell validation
ownership for Windows workflow material.

## Final Sprint State

Sprint 189 closes source-controlled PowerShell validation ownership.

The completed scope is intentionally narrow:

- selected Windows workflow PowerShell snippets are inventoried and owned;
- `scripts/validate_windows_powershell.py` validates workflow structure,
  selected report manifest references, claim boundaries, hosted wiring, and
  local/hosted `pwsh` behavior;
- `make windows-powershell-validate` is the stable maintainer entry point;
- `.github/workflows/windows-ci.yml` has a hosted `powershell-validation` job
  running `python scripts/validate_windows_powershell.py --require-pwsh`;
- focused tests cover drift, fake local PowerShell availability, parse
  failure, hosted fail-closed wiring, docs claim anchors, and unavailable
  non-pass wording;
- README, INSTALL, maintainer guide, and corpus README explain the bounded
  validation owner without promoting Windows report freshness.

## Item Completion

| Item | Planned scope | Closeout disposition |
| --- | --- | --- |
| 189.1 | PowerShell Surface Audit | Complete. Day 1 and Day 2 inventory Windows workflow snippets, selected report surfaces, artifact names, docs, and non-claims. |
| 189.2 | Validation Command Design | Complete. Day 3 defines the command, exit codes, local/hosted behavior, and non-promotion rules. |
| 189.3 | Hosted CI Wiring | Complete in source. Day 8 adds the hosted Windows `powershell-validation` job with `--require-pwsh`; hosted pass evidence waits for PR CI. |
| 189.4 | Guard Tests | Complete. Days 5 through 9 add workflow, manifest, artifact, claim-boundary, hosted wiring, fake PowerShell, and unavailable-output drift tests. |
| 189.5 | Documentation Update | Complete. Days 10 and 11 update maintainer, public, and report-facing docs with bounded support wording. |
| 189.6 | Validation | Complete. Day 12 records integrated validation; Day 13 and Day 14 add fresh audit/hygiene checks. |

## PR-Ready Change Summary

| Area | Summary |
| --- | --- |
| Workflow | Added a dedicated hosted Windows PowerShell validation ownership job without adding report generation or artifact publication. |
| Makefile | Added `windows-powershell-validate`. |
| Validator | Added an owned Python validator for selected Windows PowerShell workflow material, hosted wiring, selected report non-promotion, manifest references, claim boundaries, and local/hosted `pwsh` exit semantics. |
| Tests | Added focused validator tests for drift, fake `pwsh`, hosted fail-closed behavior, claim boundaries, and unavailable wording. |
| Docs | Updated README, INSTALL, maintainer guide, corpus README, and Sprint 189 artifacts/notes. |

## Retained Non-Goals

Sprint 189 does not claim:

- Windows report freshness;
- selected Windows report artifact publication;
- Windows report generator execution;
- broad Windows parity;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library support;
- dynamic ABI support;
- DLL/import-library support;
- runtime-loader support;
- portable performance or state-of-the-art evidence from Windows reports.

## Residuals and Handoff

| Residual | Disposition |
| --- | --- |
| Local `pwsh` absent on this machine | Expected environment residual. Local default validation exits `2` after structural checks and says unavailable PowerShell is not pass evidence. |
| Hosted Windows pass evidence | Pending PR CI after branch push. The source-controlled workflow is wired to fail closed with `--require-pwsh`. |
| Windows report freshness | Still formally deferred by Sprint 182 and handed to Sprint 190 for promotion or renewed deferral. |
| Selected Windows report artifacts | Not published and guarded against until a reviewed Sprint 190 decision changes the selected scope. |

## Final Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Ownership, hosted wiring, claim-boundary, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point preserves local unavailable semantics. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected report target manifest remains valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing selected report workflow guard remains valid. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalized report-index semantics and non-claims remain valid. |
| `python3 scripts/normalize_report_index.py --check` | Passed, `112` rows | Current normalized report index remains valid without Windows freshness promotion. |
| `make docs-check` | Passed | Doxygen/API docs coverage remains valid. |
| Unsupported Windows/report claim scan | No matches | Touched Windows/report surfaces do not contain scanned unsupported promotion phrases. |
| Stale marker scan | No open blockers | Hits are existing explanatory language in docs/plan text, not unresolved Sprint 189 work. |
| `git diff --check` | Passed | Patch whitespace is valid. |
| Sprint 189 markdown link check | Passed | Sprint-local markdown links resolve. |

## C Gate Decision

No `.c` or `.h` files were modified in Sprint 189. The requested full C gate,
`make format && make lint && make test`, is not required.

## Retrospective Inputs

What worked:

- Owning the Windows PowerShell surface through one Python validator kept
  workflow, manifest, docs, and claim-boundary checks together.
- Fake-`pwsh` tests made the available path testable on machines without
  PowerShell.
- The hosted job uses `--require-pwsh`, so hosted Windows cannot silently pass
  without PowerShell.
- Claim-boundary markers make documentation drift visible before review.

What remains:

- Hosted pass evidence must come from PR CI after branch push.
- Windows report freshness remains a separate Sprint 190 decision.
- Future Windows report promotion must update workflow, selected target
  manifest metadata, artifact scope, freshness guards, and docs together.

Review notes:

- Local exit `2` is expected on this machine because `pwsh` is not installed.
- That unavailable state is intentionally not pass evidence.
- The new workflow job validates PowerShell ownership only and does not upload
  report artifacts.

## Branch Readiness

The branch is ready for retrospective creation and PR preparation once the
Sprint 189 retrospective is requested. Source-controlled PowerShell validation
ownership is complete, with hosted proof pending PR CI execution.
