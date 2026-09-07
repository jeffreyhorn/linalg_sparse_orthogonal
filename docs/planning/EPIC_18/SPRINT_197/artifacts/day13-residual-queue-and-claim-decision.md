# Sprint 197 Day 13 Residual Queue and Claim Decision

## Purpose

Day 13 completes final-validation item 206.6 for the current branch by
publishing a prioritized Epic 18 residual queue and recording the final claim
decision before Day 14 closeout review.

## Deliverables

| Deliverable | Path |
| --- | --- |
| Prioritized residual queue | `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` |
| Long-horizon deferral list | `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md#long-horizon-deferrals` |
| Final claim decision table | This artifact and `EPIC_18_RESIDUAL_QUEUE.md#final-claim-decision` |
| Day 14 review checklist | This artifact |

## Prioritized Residual Queue Summary

| Priority | Residual ID | Theme | Closure criteria |
| ---: | --- | --- | --- |
| 1 | E18-RQ-001 | Homebrew/package-manager support blocker | Approved license metadata, selected Homebrew proof success, guard updates, install checks, docs calibration, and retained non-claims. |
| 2 | E18-RQ-002 | Selected Windows Cholesky freshness promotion | Hosted Windows artifact review, manifest metadata decision, path/normalizer tests, workflow/PowerShell guards, and calibrated docs. |
| 3 | E18-RQ-003 | Additional allocation-failure owner proof | One selected owner, invariant record, deterministic failure/retry tests, focused gate, docs, and full C validation when implementation changes. |
| 4 | E18-RQ-004 | Additional review-surface reduction | One selected cluster, behavior-preserving extraction, ownership guard, focused tests, source/CMake validation as applicable, and no behavior claims. |
| 5 | E18-RQ-005 | Additional hosted selected benchmark freshness | One exact platform/row, methodology metadata, hosted artifact evidence, freshness tests, docs, and no portable performance claim. |
| 6 | E18-RQ-006 | Windows QR incompatible comparison promotion | MSVC/CMake proof, Windows-safe generation/path behavior, exact manifest metadata, selected tests, hosted evidence review, and calibrated docs. |
| 7 | E18-RQ-007 | Generated API publication policy | Product decision, matching publication/local-only guards, freshness/link/staging checks, and docs. |
| 8 | E18-RQ-008 | Adoption and diagnostics simplification | Quick reference, support truth consolidation, diagnostics vocabulary, claim guards, and docs validation. |
| 9 | E18-RQ-009 | Release, shared-library, and dynamic ABI readiness | Release criteria, ABI policy, shared-library metadata, loader validation, package selectors, and public claim review. |
| 10 | E18-RQ-010 | State-of-the-art evidence program | External baselines, methodology, platform matrix, reliability semantics, package provenance, thresholds, and reviewed hosted evidence. |

## Final Claim Decision Table

| Claim area | Day 13 decision | Evidence |
| --- | --- | --- |
| Sprint 197 final-validation governance | Earned as interim branch evidence | `SPRINT_197/PLAN.md`, `WORKING_NOTES.md`, Day 1-13 artifacts, `PROJECT_PLAN.md` interim snapshot. |
| Evidence reconciliation | Earned as partial final-validation evidence | Day 1 closeout intake, Day 2 outcome ledger, Day 3 evidence conflict review. |
| Public claim calibration | Earned as no-promotion evidence | Day 4 public audit and Day 6 public recalibration no-op. |
| Maintainer/API claim calibration | Earned as no-promotion evidence | Day 5 maintainer/API audit and Day 7 maintainer/API recalibration no-op. |
| Project-plan status | Earned as interim status evidence | Day 8 item-level ledger and `PROJECT_PLAN.md` snapshot. |
| Focused/full validation | Earned for current docs/planning diff | Day 9 matrix, Day 10 focused validation log, Day 11 full-gate decision log. |
| Epic retrospective | Drafted and Day 14 final review complete for the current branch state | Day 12 retrospective draft, Day 13 residual updates, and Day 14 closeout review. |
| Residual queue | Published for current closeout state | `EPIC_18_RESIDUAL_QUEUE.md`. |
| Homebrew/package-manager support | Not earned | No approved license metadata or successful proof output on this branch. |
| Selected Windows Cholesky freshness promotion | Not earned | No hosted evidence review or manifest promotion on this branch. |
| Additional allocation-failure owner proof | Not earned | No owner selection, harness extension, regression, focused gate, or C validation evidence on this branch. |
| Additional review-surface reduction | Not earned | No selected extraction, guard, focused regression, or full validation evidence on this branch. |
| Additional hosted benchmark freshness | Not earned | No platform/row selection, hosted artifact, methodology metadata, or freshness-test update on this branch. |
| Windows QR incompatible comparison promotion | Not earned | No MSVC/CMake proof, generator fix, hosted artifact review, or manifest promotion on this branch. |
| Generated API publication | Not earned | Existing policy remains local-only; no publication product decision or hosted/artifact output was added. |
| Adoption/support simplification | Not earned | Public docs were audited but not consolidated or simplified. |
| Release readiness | Not earned | No release criteria, packaging, ABI, or full product-readiness evidence was added. |
| State-of-the-art status | Not earned | No broad external baseline, portable performance, platform matrix, package provenance, ABI policy, or research-quality comparative evidence was added. |

## Long-Horizon Deferrals

- Shared-library packaging and dynamic ABI compatibility.
- Runtime-loader behavior and platform-specific binary packaging.
- Release readiness and release benchmark policy.
- Broad Windows parity.
- Portable performance and hosted timing thresholds.
- Broad external-library parity.
- Hosted generated API publication if product policy selects it.
- Broad allocation-failure, OS OOM, and concurrency semantics.
- Unqualified state-of-the-art sparse linear algebra positioning.

## Day 14 Review Checklist

| Check | Day 14 expectation |
| --- | --- |
| Artifact completeness | Confirm `PLAN.md`, `WORKING_NOTES.md`, Day 1-13 artifacts, retrospective, residual queue, and project-plan snapshot are present. |
| Internal consistency | Confirm numbering caveat, pending Sprint 198-205 status, no-promotion decisions, and residual queue agree across artifacts. |
| Claim calibration | Confirm public, maintainer, retrospective, residual, and project-plan text does not promote unsupported package, Windows, benchmark, API, release, ABI, or state-of-the-art claims. |
| Validation currency | Re-run lightweight checks after Day 13 edits and record results. |
| C/header trigger | Confirm whether any `*.c` or `*.h` files changed; run `make format && make lint && make test` only if triggered. |
| Generated artifact noise | Confirm generated API/build/cache artifacts remain ignored and untracked. |
| PR summary inputs | Prepare summary of changed docs, validation evidence, non-claims, residual queue, and numbering caveat. |

## Validation Notes

- Created `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`.
- Updated the retrospective from Day 12 draft wording to reference the
  published residual queue.
- Updated Sprint 197 working notes with Day 13 claim decisions.
- Updated only planning documentation.
