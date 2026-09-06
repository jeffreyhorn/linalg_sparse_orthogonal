# Project Plan: Epic 18 - Selected Productization, Reliability & Evidence Promotion

## Overview

Epic 18 starts from the completed Epic 17 closeout. The project has broad
solver coverage, strong selected evidence, static-first install validation,
guarded Windows workflow ownership, selected generated comparison freshness,
one selected hosted benchmark lane, improved adoption/API guidance, and a
published residual queue.

The project is still not defensibly state of the art. Remaining gaps include
package-manager/Homebrew support, selected Windows freshness promotion,
additional allocation-failure proof, large review surfaces, cross-platform
benchmark freshness, Windows QR comparison promotion, generated API
publication policy, shared-library/dynamic ABI support, release readiness,
portable performance evidence, and broad external-library parity.

Epic 18 focuses on closing fewer gaps completely:

- complete the selected Homebrew/package-manager proof if the license
  metadata decision is available;
- promote the selected Windows Cholesky freshness lane or formally close its
  deferral state with stronger guards;
- add one additional selected allocation-failure proof owner;
- reduce one additional high-risk review surface;
- add one hosted selected benchmark freshness lane outside the existing Linux
  proof;
- promote Windows QR incompatible comparison only if MSVC/CMake evidence is
  sufficient;
- decide generated API HTML publication scope;
- finish with calibrated release and state-of-the-art non-claims.

The epic deliberately avoids broad state-of-the-art, portable performance,
external ecosystem parity, release, dynamic ABI, shared-library, broad
package-manager, and broad Windows parity claims unless exact evidence is
added.

## Sprint 197 Day 8 Interim Status Snapshot

This snapshot was added from
`SPRINT_197/artifacts/day8-project-plan-status.md`. It is an interim
evidence-ledger update, not a final Epic 18 closeout. The requested
`SPRINT_197` branch artifacts are executing the cited final-validation scope
from this file's Sprint 206 section, while the source project plan still labels
Sprint 197 as the baseline and closure-selection sprint. To avoid converting
future plan intent into support claims, Sprints 198 through 205 remain pending
future execution until their own artifacts, validation records, and PR evidence
exist.

| Sprint | Current disposition | Evidence |
| --- | --- | --- |
| 197 | In progress with numbering caveat | `SPRINT_197/PLAN.md`; `SPRINT_197/WORKING_NOTES.md`; `SPRINT_197/artifacts/day1-closeout-intake.md`; the branch currently contains final-validation scaffold and reconciliation artifacts, not completed baseline sprint closeout evidence. |
| 198 | Pending future execution | No Sprint 198 artifact directory exists in this branch; Homebrew/package-manager proof, license metadata decision, formula proof, guard promotion, documentation promotion, and validation remain future work. |
| 199 | Pending future execution | No Sprint 199 artifact directory exists in this branch; selected Windows Cholesky promotion evidence, manifest decision, normalizer hardening, workflow guard updates, docs calibration, and validation remain future work. |
| 200 | Pending future execution | No Sprint 200 artifact directory exists in this branch; additional allocation-failure owner selection, invariant record, harness work, regression tests, focused gate, docs, and validation remain future work. |
| 201 | Pending future execution | No Sprint 201 artifact directory exists in this branch; additional review-surface candidate ranking, extraction, guard, regression, docs, and validation remain future work. |
| 202 | Pending future execution | No Sprint 202 artifact directory exists in this branch; additional hosted selected benchmark platform/row evidence, methodology metadata, workflow lane, freshness tests, docs, and validation remain future work. |
| 203 | Pending future execution | No Sprint 203 artifact directory exists in this branch; Windows QR incompatible comparison proof or re-deferral, generator fixes, manifest decision, tests, docs, and validation remain future work. |
| 204 | Pending future execution | No Sprint 204 artifact directory exists in this branch; generated API publication/local-only decision, guard implementation, freshness/link checks, API routing docs, claim guard, and validation remain future work. |
| 205 | Pending future execution | No Sprint 205 artifact directory exists in this branch; support matrix/adoption quick-reference consolidation, diagnostics vocabulary, claim guards, docs, and validation remain future work. |
| 206 | Requested final-validation evidence recorded through `SPRINT_197` artifacts | `SPRINT_197/artifacts/day2-outcome-ledger.md`; `SPRINT_197/artifacts/day3-evidence-conflicts.md`; `SPRINT_197/artifacts/day6-public-recalibration.md`; `SPRINT_197/artifacts/day7-maintainer-api-recalibration.md`; `SPRINT_197/artifacts/day8-project-plan-status.md`; `SPRINT_197/artifacts/day10-focused-validation-log.md`; `SPRINT_197/artifacts/day11-full-quality-gate-log.md`; `SPRINT_197/artifacts/day12-retrospective-draft.md`; `SPRINT_197/artifacts/day13-residual-queue-and-claim-decision.md`; `SPRINT_197/artifacts/day14-final-closeout-review.md`; `EPIC_18_RETROSPECTIVE.md`; `EPIC_18_RESIDUAL_QUEUE.md`; item 206.1-206.6 evidence is recorded for the current branch state. |

The full item-level Day 8 disposition ledger is
`SPRINT_197/artifacts/day8-project-plan-status.md`.

## Sprint 197: Epic 18 Baseline, Residual Ledger & Closure Selection

**Duration:** 14 days, approximately 166 hours

**Goal:** Convert the 2026-09-04 review and Epic 17 residual queue into an
Epic 18 closure ledger, select complete closures, and define acceptance gates.

### Prerequisites from Previous Sprints

- Epic 17 retrospective and residual queue are merged on `master`.
- `docs/planning/EPIC_18/reviews/review-codex-2026-09-04.md` exists.
- `docs/planning/EPIC_18/reviews/todo-codex-2026-09-04.md` exists.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 197.1 | Review Intake | Convert the 2026-09-04 Codex review into a source-controlled Epic 18 gap ledger with owner files and claim risks. | 24 |
| 197.2 | Residual Deduplication | Reconcile Epic 17 residuals with the new review and merge duplicate package, Windows, benchmark, API, reliability, and review-surface gaps. | 28 |
| 197.3 | Closure Selection | Select the exact gaps Epic 18 will fully close and record non-goals for broad state-of-the-art, ABI, package, platform, release, and performance claims. | 30 |
| 197.4 | Acceptance Gate Map | Define local/hosted validation commands, artifact expectations, support-tier wording, and failure semantics for each selected closure. | 34 |
| 197.5 | Claim Surface Map | Map public and maintainer docs that must change for each closure and identify guard scripts that must prevent overclaiming. | 24 |
| 197.6 | Sprint Handoff | Create Sprint 197 artifacts, working notes, and handoff records for package and Windows promotion work. | 26 |

### Deliverables

- Epic 18 gap ledger.
- Deduplicated residual queue.
- Closure selection and non-goals.
- Acceptance-gate and claim-surface maps.

### Total Estimate

166 hours

## Sprint 198: Homebrew License Metadata and Formula Proof Closure

**Duration:** 14 days, approximately 168 hours

**Goal:** Close the package-manager/Homebrew blocker by adding approved
license metadata, proving the selected local formula workflow, and promoting
only the earned support claim.

### Prerequisites from Previous Sprints

- Sprint 197 selects package-manager/Homebrew proof as a closure.
- An approved standalone root license metadata decision is available.
- Existing Homebrew proof material and package guards are present.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 198.1 | License Metadata Decision | Record the approved root license metadata and exact Homebrew formula license identifier. | 24 |
| 198.2 | Metadata Implementation | Add or update root license files and formula metadata according to the approved decision. | 28 |
| 198.3 | Formula Proof Execution | Run and harden the local Homebrew proof through archive/checksum, render, install, `brew test`, uninstall, and cleanup. | 38 |
| 198.4 | Guard Promotion | Update package-manager and static-package guards so support wording depends on proof success and exact metadata. | 28 |
| 198.5 | Documentation Promotion | Update README, INSTALL, Homebrew README, and maintainer guidance with the exact earned support tier and retained non-claims. | 26 |
| 198.6 | Validation | Run Homebrew proof, package guards, install checks, docs checks, and full C gate if `.c` or `.h` files changed. | 24 |

### Deliverables

- Approved license metadata.
- Passing selected Homebrew local formula proof.
- Updated package guards and claim-safe docs.

### Total Estimate

168 hours

## Sprint 199: Selected Windows Cholesky Freshness Promotion

**Duration:** 14 days, approximately 166 hours

**Goal:** Promote the guarded selected Windows Cholesky comparison freshness
lane only if hosted evidence, manifest metadata, and docs agree.

### Prerequisites from Previous Sprints

- Sprint 197 accepts selected Windows freshness promotion criteria.
- Sprint 190 guarded Windows Cholesky workflow path exists.
- PowerShell validation ownership remains available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 199.1 | Hosted Evidence Review | Inspect hosted Windows artifacts for `cholesky-spd-tridiag-5`, row IDs, artifact paths, and workflow metadata. | 26 |
| 199.2 | Manifest Promotion Decision | Promote selected target metadata only if hosted evidence proves the exact target and platform scope. | 28 |
| 199.3 | Normalizer Hardening | Add or update tests for Windows path normalization, selected-target filtering, missing rows, and stale artifacts. | 30 |
| 199.4 | Workflow Guard Update | Align Windows workflow, PowerShell validation, artifact names, and selected freshness commands. | 28 |
| 199.5 | Documentation Calibration | Update README, INSTALL, corpus docs, and maintainer guide with the promoted or re-deferred selected Windows claim. | 26 |
| 199.6 | Validation | Run selected manifest, workflow, PowerShell, normalizer, freshness, docs, and applicable C quality gates. | 28 |

### Deliverables

- Reviewed hosted Windows Cholesky freshness evidence.
- Promoted or explicitly re-deferred selected manifest state.
- Updated guards and claim-safe docs.

### Total Estimate

166 hours

## Sprint 200: Additional Allocation-Failure Owner Proof

**Duration:** 14 days, approximately 168 hours

**Goal:** Prove one additional selected allocation-failure owner with
deterministic cleanup, stale-output suppression, and retry evidence.

### Prerequisites from Previous Sprints

- Sprint 197 ranks reliability candidates.
- Sprint 195 symbolic Cholesky proof pattern is available.
- The selected owner is not already covered by existing allocation-failure
  gates.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 200.1 | Owner Selection | Rank candidates and select exactly one owner, such as symbolic LU, analyze lifecycle, or direct-solver output publication. | 24 |
| 200.2 | Invariant Record | Document cleanup, publication, stale-output, retry, caller-owned input, and unsupported-breadth invariants before code edits. | 24 |
| 200.3 | Harness Integration | Extend deterministic failure injection only as needed to reach the selected owner. | 34 |
| 200.4 | Regression Tests | Add failed-allocation, cleanup, stale-output, retry, and caller-owned input tests for the selected owner. | 36 |
| 200.5 | Focused Gate | Add Make/CTest labels and registration guards for the selected reliability proof. | 24 |
| 200.6 | Docs And Validation | Update claim docs and run focused gate, source-list checks, docs checks, and `make format && make lint && make test`. | 26 |

### Deliverables

- One additional selected allocation-failure owner proof.
- Focused reliability gate and registration guard.
- Claim-safe reliability documentation.

### Total Estimate

168 hours

## Sprint 201: Additional Review-Surface Reduction

**Duration:** 14 days, approximately 166 hours

**Goal:** Reduce one large QR, LDLT, SVD, etree, integration, or direct-solver
review surface without changing behavior.

### Prerequisites from Previous Sprints

- Sprint 197 ranks large source and test surfaces.
- Existing source-list, CMake parity, and helper guard patterns are available.
- The selected cluster has focused tests that can prove behavior preservation.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 201.1 | Candidate Ranking | Rank current >2000-line source/test surfaces by review risk, ownership clarity, and extraction feasibility. | 24 |
| 201.2 | Cluster Selection | Select one cluster and record behavior-preservation invariants and no-public-API-change boundaries. | 24 |
| 201.3 | Helper Or Module Extraction | Extract helpers or source module boundaries only where reviewability improves. | 38 |
| 201.4 | Ownership Guard | Add or update guard scripts/tests to keep the extracted surface registered and reviewable. | 26 |
| 201.5 | Focused Regression | Run focused tests and add narrow regressions only for extraction safety. | 26 |
| 201.6 | Validation And Docs | Run source-list, CMake parity if registration changed, full C quality gate, and maintainer docs updates. | 28 |

### Deliverables

- One reduced large review surface.
- Behavior-preservation artifact.
- Focused guard and validation record.

### Total Estimate

166 hours

## Sprint 202: Hosted Selected Benchmark Freshness on One Additional Platform

**Duration:** 14 days, approximately 166 hours

**Goal:** Add one hosted selected benchmark freshness lane outside the current
Linux-only selected performance proof, without claiming portable performance.

### Prerequisites from Previous Sprints

- Sprint 197 selects the platform and benchmark row.
- Sprint 192 selected benchmark methodology and freshness checks are present.
- Hosted workflow runtime budget is acceptable.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 202.1 | Platform And Row Selection | Select exactly one platform/row pair and record why it closes the highest-value evidence gap. | 24 |
| 202.2 | Methodology Metadata | Define platform, compiler, build flags, repeat policy, artifact scope, and threshold-free interpretation. | 26 |
| 202.3 | Workflow Lane | Add hosted workflow generation, selected artifact upload, and freshness check for the selected row. | 34 |
| 202.4 | Freshness Tests | Add tests for missing, stale, duplicate, malformed, deferred, and path-normalized benchmark artifacts. | 30 |
| 202.5 | Docs Calibration | Update benchmark docs, README/INSTALL support matrix, and maintainer guide without portable performance claims. | 24 |
| 202.6 | Validation | Run benchmark freshness tests, selected manifest tests, docs checks, hosted evidence review, and full C gate if needed. | 28 |

### Deliverables

- One additional hosted selected benchmark freshness lane.
- Methodology-bound benchmark artifact contract.
- Claim-safe performance docs.

### Total Estimate

166 hours

## Sprint 203: Windows QR Incompatible Comparison Promotion

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote the QR incompatible comparison target on Windows only if
MSVC/CMake generation, artifacts, manifest metadata, and docs support it.

### Prerequisites from Previous Sprints

- Sprint 199 Windows Cholesky promotion lessons are available.
- Sprint 191 QR incompatible comparison family exists locally.
- PowerShell and Windows workflow validation are owned.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 203.1 | MSVC Probe | Run or reproduce the QR incompatible comparison under MSVC/CMake and record failures or proof output. | 28 |
| 203.2 | Generator Fixes | Fix selected Windows generator, path, or CMake issues without broadening the comparison surface. | 34 |
| 203.3 | Manifest Promotion | Promote selected metadata only if hosted Windows evidence proves the exact target. | 26 |
| 203.4 | Normalizer And Workflow Tests | Add tests for Windows QR artifact paths, selected row filtering, dependency status, and stale output. | 30 |
| 203.5 | Docs Calibration | Update README, INSTALL, corpus docs, and maintainer guide with exact selected QR Windows boundaries. | 24 |
| 203.6 | Validation | Run selected comparison workflow, normalizer, manifest, QR focused tests, Windows guards, docs checks, and full C gate if needed. | 26 |

### Deliverables

- Promoted or explicitly re-deferred Windows QR incompatible comparison.
- Hardened Windows comparison generator/normalizer behavior.
- Claim-safe selected comparison docs.

### Total Estimate

168 hours

## Sprint 204: Generated API Publication Decision

**Duration:** 14 days, approximately 166 hours

**Goal:** Decide and implement either hosted generated API publication or a
stronger local-only generated API policy.

### Prerequisites from Previous Sprints

- Sprint 197 identifies generated API publication as a closure candidate.
- Current `make docs-check` and `make api-docs-freshness` pass locally.
- Public header and API reference ownership are clear.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 204.1 | Product Decision | Decide hosted publication, retained artifact publication, committed generated output, or strengthened local-only policy. | 28 |
| 204.2 | Publication Or Guard Implementation | Implement the chosen policy with workflow, artifact, Pages, ignore, or staging guard changes. | 36 |
| 204.3 | Freshness And Link Checks | Add freshness, generated-page coverage, staging, and Markdown/link validation needed by the chosen policy. | 30 |
| 204.4 | API Routing Docs | Update `docs/api_reference.md`, README, INSTALL, and maintainer guide with source-of-truth and generated-output semantics. | 24 |
| 204.5 | Claim Boundary Guard | Ensure generated API docs do not imply ABI, package-manager, hosted publication, or completeness claims beyond the selected policy. | 22 |
| 204.6 | Validation | Run docs/API freshness, docs checks, link checks if added, workflow checks, and full C gate if headers changed. | 26 |

### Deliverables

- Explicit generated API publication decision.
- Implemented publication or strengthened local-only guard.
- Updated API docs and validation record.

### Total Estimate

166 hours

## Sprint 205: Support Matrix and Adoption Quick-Reference Consolidation

**Duration:** 14 days, approximately 164 hours

**Goal:** Reduce public documentation friction by centralizing support truth
and adding a compact problem-shape quick reference without weakening claim
boundaries.

### Prerequisites from Previous Sprints

- Package, Windows, benchmark, comparison, and API publication decisions from
  Sprints 198-204 are known.
- Existing README, INSTALL, solver-selection, cookbook, and maintainer docs
  remain claim-calibrated.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 205.1 | Public Doc Audit | Re-audit README, INSTALL, tutorial, cookbook, solver-selection, examples, benchmark, and API reference for duplicate caveats and adoption friction. | 26 |
| 205.2 | Quick Reference Design | Define a compact problem-shape to workflow table for common local and installed use cases. | 24 |
| 205.3 | Support Truth Consolidation | Centralize support/readiness truth and convert repeated caveats into links where safe. | 32 |
| 205.4 | Diagnostics Vocabulary | Normalize result/status/residual/convergence wording across direct, iterative, QR/SVD, eigensolver, and report docs. | 28 |
| 205.5 | Claim Guard Updates | Update or add docs guards so simplified wording does not broaden support, package, ABI, platform, performance, or state-of-the-art claims. | 26 |
| 205.6 | Validation | Run docs checks, claim guards, install docs checks, generated API checks, and full C gate if headers changed. | 28 |

### Deliverables

- Compact adoption quick reference.
- Consolidated support/readiness routing.
- Updated diagnostics vocabulary and claim guards.

### Total Estimate

164 hours

## Sprint 206: Epic 18 Final Validation, Claim Calibration & Closeout

**Duration:** 14 days, approximately 166 hours

**Goal:** Reconcile Epic 18 outcomes, run final validation, calibrate claims,
publish the retrospective and residual queue, and decide whether any stronger
support claims are earned.

### Prerequisites from Previous Sprints

- Sprints 197-205 are complete or have explicit deferred/residual outcomes.
- Package, Windows, benchmark, comparison, API, review-surface, adoption, and
  reliability evidence records are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 206.1 | Evidence Reconciliation | Reconcile Sprint 197-205 outcomes, validation records, decisions, and residuals. | 26 |
| 206.2 | Claim Recalibration | Update README, INSTALL, maintainer guide, benchmark docs, API docs, corpus docs, and planning docs so claims match earned evidence. | 34 |
| 206.3 | Project Plan Status | Mark Epic 18 sprint items complete, narrowed, deferred, residualized, or superseded with evidence links. | 24 |
| 206.4 | Integrated Validation | Run final focused gates plus full quality gates required by changed surfaces. | 34 |
| 206.5 | Epic Retrospective | Create `EPIC_18_RETROSPECTIVE.md` with outcomes, evidence, non-claims, residuals, and state-of-the-art assessment. | 26 |
| 206.6 | Residual Queue | Publish a prioritized next-epic residual queue with exact closure targets and long-horizon deferrals. | 22 |

### Deliverables

- Final claim-recalibrated documentation.
- Integrated validation record.
- Epic 18 retrospective.
- Prioritized residual queue and state-of-the-art assessment.

### Total Estimate

166 hours
