# Day 9: Focused Validation And Fixes

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Run focused closeout checks and fix claim, routing, manifest, or
documentation regressions before broad gates.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 9 executed the focused validation matrix defined on Day 8 for the current
Sprint 206 documentation and planning diff. The changed surfaces remain public
docs, maintainer/API docs, Epic 18 project-plan status, and Sprint 206
planning artifacts.

No `.c` or `.h` files, workflow files, guard scripts, manifests, schemas,
Makefile rules, CMake files, benchmark sources, examples, or tests were
changed on Day 9.

## Focused Validation Results

| Command | Result | Notes |
| --- | --- | --- |
| `git diff --check` | Passed | Patch whitespace is clean. |
| `make support-docs-guard` | Passed | Reported `test-support-quick-reference-docs: ok`. |
| `bash scripts/package_manager_deferral_check.sh` | Failed initially, then passed after fix | Initial failure: `README no longer records current local Homebrew proof status`. The README now keeps the guard-required Homebrew proof status phrase contiguous while preserving the non-claim boundary. Rerun passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static package, shared-library, dynamic ABI, Windows package, and workflow non-claim checks passed. |
| `make api-docs-freshness` | Passed | Doxygen generation, API docs coverage, local-only generated-output checks, and routing checks passed. |
| Stale `PROJECT_PLAN.md` wording search | Passed | The old Sprint 197 interim heading, old Sprint 206 `SPRINT_197`-only status row, and old `In progress with numbering caveat` project-plan row are absent. |
| `git status --ignored --short docs/api` | Passed | Existing generated API output remains ignored as `!! docs/api/`. |

## Fix Applied

`README.md` was reflowed so the exact package-manager guard marker remains on
one physical line:

```text
Homebrew proof is a developer-mode local static source formula proof
```

This is a guard compatibility fix only. It does not change the support claim:
the Sprint 198 Homebrew proof remains a developer-mode local static source
formula proof, not a user-facing Homebrew install path, Homebrew/core proof,
bottle proof, Linuxbrew proof, public tap proof, or broad package-manager
distribution proof.

## Non-Triggered Validation

| Gate | Day 9 disposition |
| --- | --- |
| `make format && make lint && make test` | Not required. No `.c` or `.h` files are changed through Day 9. |
| `make windows-powershell-guard` | Not required by the Day 9 diff. No Windows workflow or Windows claim text changed on Day 9. |
| `make report-index-comparison-freshness` and selected manifest tests | Not required by the Day 9 diff. No selected report target manifest, comparison output, benchmark metadata, or corpus schema changed. |
| `make qr-header-docs-guard` | Not required by the Day 9 diff. No QR public-header documentation changed. |

## Claim Boundaries Preserved

Day 9 does not promote:

- broad package-manager distribution, Homebrew/core readiness, bottles,
  Linuxbrew, public tap maintenance, or binary package support;
- broad Windows support, selected Windows Cholesky promotion, or selected
  Windows QR incompatible promotion;
- broad allocation-failure guarantees;
- repository-wide review-surface cleanup;
- hosted generated API publication, retained generated-doc artifacts, or
  committed generated HTML;
- shared-library or dynamic ABI support;
- portable performance, release readiness, external-library parity, or
  state-of-the-art status.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Focused checks pass or have documented blockers. | Met. All focused Day 9 checks passed after the README marker reflow. |
| No known claim-boundary, routing, manifest, or planning-status regression remains open. | Met for the focused Day 9 scope. |
| Item 206.4 has focused validation evidence. | Met. This artifact records command results and the only fix needed. |

## Day 9 Disposition

Day 9 is complete. Day 10 should run the broad documentation/generated-output
gates from the Day 8 matrix and confirm no generated API output is staged.
