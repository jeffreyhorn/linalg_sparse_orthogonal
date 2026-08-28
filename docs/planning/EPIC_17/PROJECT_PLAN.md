# Project Plan: Epic 17 - Product Proof, Platform Freshness & Reviewability

## Overview

Epic 17 starts from the completed Epic 16 closeout. The project has broad
solver coverage, strong local tests, substantial CI, static-first package
validation, selected generated report freshness, and disciplined non-claims.
It is not yet defensibly state of the art because package-manager proof,
Windows report freshness, broad external comparison evidence, portable
performance evidence, shared-library ABI policy, and several maintainability
surfaces remain incomplete or explicitly deferred.

Epic 17 focuses on closing fewer gaps completely:

- complete the selected Homebrew local proof path by resolving the license
  metadata blocker;
- promote or formally close one Windows PowerShell/report freshness lane;
- add one bounded external evidence lane with hosted freshness;
- harden one selected methodology-bound performance lane;
- reduce one high-risk implementation/test review surface;
- simplify user adoption and API guidance without expanding unsupported
  claims;
- add one selected reliability/failure-path proof;
- recalibrate final state-of-the-art and support-tier claims.

The epic deliberately avoids broad state-of-the-art, portable performance,
external ecosystem parity, dynamic ABI, shared-library, broad package-manager,
or broad Windows parity claims unless the corresponding evidence is added.

## Sprint 187: Epic 17 Baseline, Gap Ledger & Acceptance Gates

**Duration:** 14 days, approximately 166 hours

**Goal:** Freeze the post-Epic-16 baseline, convert the Codex review into a
source-controlled gap ledger, and select the exact Epic 17 closures.

### Prerequisites from Previous Sprints

- Epic 16 retrospective and residual queue are merged on `master`.
- `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` and
  `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` exist.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 187.1 | Review Intake | Convert Codex review findings into a prioritized Epic 17 gap ledger with owner files and claim risks. | 24 |
| 187.2 | Residual Reconciliation | Reconcile Epic 16 residuals with new review findings and deduplicate package, Windows, comparison, performance, and review-surface gaps. | 28 |
| 187.3 | Closure Selection | Select the complete gaps Epic 17 will close and record non-goals for broad state-of-the-art, ABI, platform, package, and performance claims. | 28 |
| 187.4 | Acceptance Gates | Define validation commands, hosted/local ownership, artifact expectations, and support-tier wording for each selected closure. | 34 |
| 187.5 | Quality Surface Map | Map required checks for docs-only, script, workflow, package, generated-report, benchmark, and C/header changes. | 24 |
| 187.6 | Sprint Handoff | Create Sprint 187 artifacts, working notes, and handoff records for package and Windows work. | 28 |

### Deliverables

- Epic 17 gap ledger.
- Deduplicated residual selection.
- Acceptance gates and quality surface map.
- Sprint handoffs for package and Windows validation work.

### Total Estimate

166 hours

## Sprint 188: Homebrew Proof Completion

**Duration:** 14 days, approximately 168 hours

**Goal:** Close the selected Homebrew local proof blocker by resolving
standalone license metadata and proving the full local formula workflow.

### Prerequisites from Previous Sprints

- Sprint 187 selected package proof completion as an Epic 17 closure.
- Sprint 180 Homebrew formula template and proof script are available.
- Epic 16 residual `R186-PKG-LICENSE` is the owner residual.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 188.1 | License Strategy Decision | Decide approved standalone license metadata or a documented alternate formula license strategy. | 24 |
| 188.2 | Metadata Implementation | Add or update root license metadata and Homebrew formula metadata according to the selected strategy. | 28 |
| 188.3 | Proof Script Hardening | Ensure `scripts/homebrew_local_formula_proof.sh` validates render, archive, checksum, install, test, uninstall, and cleanup with clear failures. | 34 |
| 188.4 | Package Guards | Update package-manager and static-package guards so Homebrew support wording depends on the proven local proof state. | 28 |
| 188.5 | Documentation Calibration | Update README, INSTALL, Homebrew README, and maintainer guidance with the exact earned support level and retained non-claims. | 28 |
| 188.6 | Validation | Run Homebrew proof, package guards, install checks selected by the change, and required docs/C quality gates. | 26 |

### Deliverables

- Resolved Homebrew license metadata blocker.
- Passing local Homebrew formula proof.
- Updated package guards and claim-safe docs.

### Total Estimate

168 hours

## Sprint 189: PowerShell Validation Ownership

**Duration:** 14 days, approximately 166 hours

**Goal:** Close the PowerShell validation environment gap by adding an
owned local/hosted validation command for Windows report workflow material.

### Prerequisites from Previous Sprints

- Sprint 187 acceptance gates identify Windows validation owners.
- Sprint 182 Windows report freshness deferral is available.
- Windows CI remains CMake-first and static-first.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 189.1 | PowerShell Surface Audit | Inventory PowerShell workflow snippets, Windows report scripts, artifact names, and selected report guard assumptions. | 24 |
| 189.2 | Validation Command Design | Define a command that parses or dry-runs selected PowerShell report workflow material locally when `pwsh` exists and in hosted Windows CI. | 28 |
| 189.3 | Hosted CI Wiring | Add the hosted Windows validation lane without promoting report freshness or artifact publication yet. | 34 |
| 189.4 | Guard Tests | Add tests or scripts that fail on stale PowerShell validation ownership, unsupported artifact names, or accidental report freshness promotion. | 30 |
| 189.5 | Documentation Update | Update maintainer and Windows support docs to explain the new PowerShell validation owner and remaining non-claims. | 24 |
| 189.6 | Validation | Run Windows-adjacent guards, manifest/schema tests, docs checks, and full C gate if C/header files changed. | 26 |

### Deliverables

- Owned PowerShell validation command.
- Hosted Windows validation wiring.
- Guarded distinction between PowerShell validation and report freshness.

### Total Estimate

166 hours

## Sprint 190: Windows Selected Report Freshness Decision

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote one Windows-safe selected report freshness lane or renew the
formal deferral with stronger guard evidence.

### Prerequisites from Previous Sprints

- Sprint 189 PowerShell validation ownership is complete.
- Selected report target manifest and report normalizer tests are available.
- Epic 16 residual `R186-WIN-REPORT-FRESHNESS` is still open unless closed.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 190.1 | Freshness Candidate Selection | Select one Windows-safe report freshness lane, including exact generator, manifest row, expected rows, artifact scope, and runtime limit. | 26 |
| 190.2 | Workflow Implementation | Wire the selected Windows lane or publish a refreshed deferral artifact if promotion is rejected. | 38 |
| 190.3 | Manifest And Schema Updates | Add selected metadata fields and schema checks needed for the Windows lane or renewed deferral. | 28 |
| 190.4 | Freshness Guard | Add or update freshness checks and artifact-name guards for the selected Windows scope. | 32 |
| 190.5 | Docs And Claim Calibration | Update README, maintainer guide, report docs, and support-tier wording for the selected outcome. | 22 |
| 190.6 | Validation | Run selected workflow tests, schema tests, report freshness tests, docs checks, and required C quality gates if needed. | 22 |

### Deliverables

- Windows report freshness decision.
- Implemented selected lane or stronger formal deferral.
- Updated report manifest, guards, and claim-safe docs.

### Total Estimate

168 hours

## Sprint 191: Bounded External Comparison Family

**Duration:** 14 days, approximately 168 hours

**Goal:** Add one bounded external comparison family that improves numerical
credibility without claiming broad ecosystem parity.

### Prerequisites from Previous Sprints

- Sprint 187 selects the comparison target and acceptance tolerances.
- Existing selected comparison workflow and manifest infrastructure are
  available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 191.1 | Family Selection | Choose exactly one solver family, fixture, reference implementation, metrics, tolerances, and dependency policy. | 24 |
| 191.2 | Fixture And Reference | Add source-controlled fixture/generator material and a bounded external reference path. | 34 |
| 191.3 | Runner Integration | Extend comparison runner output, dependency-status rows, study rows, and manifest metadata for the selected family. | 34 |
| 191.4 | Focused Tests | Add parser, dependency, tolerance, pass/fail, and stale-output tests for the selected comparison family. | 34 |
| 191.5 | Docs And Non-Claims | Update maintainer/report/solver docs with exact claim wording and retained broad-parity non-claims. | 20 |
| 191.6 | Validation | Run selected comparison workflow, freshness, focused C tests, and full C gate if C/header files changed. | 22 |

### Deliverables

- One new bounded external comparison family.
- Source-controlled fixtures, reports, manifest rows, and tests.
- Claim-safe documentation.

### Total Estimate

168 hours

## Sprint 192: Methodology-Bound Performance Evidence Lane

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote one performance lane from local threshold-free context to a
methodology-bound hosted evidence lane with explicit limits.

### Prerequisites from Previous Sprints

- Sprint 187 selected the performance target and evidence standard.
- Existing benchmark/report infrastructure is available.
- Sprint 191 comparison work has not expanded performance claims.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 192.1 | Lane Selection | Select one benchmark family, fixture, backend policy, platform, repeat count, runtime budget, and acceptance meaning. | 24 |
| 192.2 | Methodology Metadata | Add compiler, flags, CPU, thread count, warmup, repeats, variance, timestamp, branch, commit, and matrix metadata. | 34 |
| 192.3 | Hosted Freshness | Add a hosted CI freshness lane with artifact upload and bounded runtime. | 34 |
| 192.4 | Regression Policy | Define one conservative regression sentinel or explicitly keep rows threshold-free with reviewed rationale. | 28 |
| 192.5 | Docs And Report Index | Update benchmark docs, maintainer guide, report index normalization, and claim boundaries. | 24 |
| 192.6 | Validation | Run benchmark freshness, report normalizer tests, docs checks, and required C quality gates if needed. | 24 |

### Deliverables

- One methodology-bound hosted performance evidence lane.
- Stable report metadata and artifact publication.
- Clear performance claim and non-claim language.

### Total Estimate

168 hours

## Sprint 193: Selected Large Review-Surface Reduction

**Duration:** 14 days, approximately 168 hours

**Goal:** Reduce one high-risk implementation/test review surface while
preserving behavior and validation.

### Prerequisites from Previous Sprints

- Sprint 187 review-surface ranking is complete.
- Sprint 185 helper extraction pattern and guard model are available.
- Source-list and full C quality gates are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 193.1 | Candidate Ranking | Rank large source/test candidates by size, algorithm risk, helper ownership, current tests, and user-facing importance. | 22 |
| 193.2 | Cluster Selection | Select exactly one cluster and record no-behavior-change invariants. | 20 |
| 193.3 | Extraction Design | Design helper/source boundaries, cleanup ownership, override restoration, and registration guards. | 30 |
| 193.4 | Implementation | Extract the selected helper/source cluster with minimal behavior-preserving edits. | 42 |
| 193.5 | Guard And Docs | Add a cluster-specific ownership guard and maintainer documentation for the new boundary. | 24 |
| 193.6 | Validation | Run focused tests, source-list checks, `make format && make lint && make test`, and CMake parity if source lists changed. | 30 |

### Deliverables

- One reduced review surface.
- Focused helper/source ownership guard.
- Full behavior-preserving validation record.

### Total Estimate

168 hours

## Sprint 194: Adoption And API Coherence Simplification

**Duration:** 14 days, approximately 166 hours

**Goal:** Make the project easier to adopt by simplifying user-facing
workflow guidance and consolidating support/readiness truth.

### Prerequisites from Previous Sprints

- Sprint 188 package proof and Sprint 190 Windows decision are reflected in
  docs.
- Sprint 191 and Sprint 192 evidence lanes have claim-safe documentation.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 194.1 | Adoption Audit | Audit README, INSTALL, tutorial, cookbook, solver selection, API reference, examples, and maintainer guide for duplication and friction. | 24 |
| 194.2 | Support Matrix | Add a compact production-readiness/support matrix separating user truth from historical sprint evidence. | 28 |
| 194.3 | Installed Consumer Tutorial | Add or improve a minimal external installed-consumer tutorial for Make/pkg-config and CMake paths. | 30 |
| 194.4 | Diagnostics Coherence | Normalize result/status/residual diagnostics wording across direct, iterative, QR/SVD, and eigensolver docs. | 34 |
| 194.5 | Header Narrative Cleanup | Move narrative workflow guidance out of selected public headers where possible while preserving exact declarations and Doxygen coverage. | 26 |
| 194.6 | Validation | Run docs link checks, Doxygen checks, examples build, install checks, and full C gate if headers changed. | 24 |

### Deliverables

- Simpler adoption path.
- Compact support/readiness matrix.
- Improved installed-consumer tutorial.
- Calibrated API/header documentation.

### Total Estimate

166 hours

## Sprint 195: Selected Reliability And Failure-Path Proof

**Duration:** 14 days, approximately 168 hours

**Goal:** Add deterministic reliability evidence for one selected
allocation-heavy or failure-prone owner beyond prior proof lanes.

### Prerequisites from Previous Sprints

- Sprint 187 reliability candidate list is available.
- Existing deterministic allocation-failure harness patterns are available.
- Sprint 193 review-surface changes are merged or explicitly out of scope.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 195.1 | Owner Selection | Select one reliability owner by allocation density, cleanup complexity, user impact, and testability. | 22 |
| 195.2 | Invariant Record | Document cleanup invariants, publication points, retry semantics, and unsupported breadth. | 26 |
| 195.3 | Harness Extension | Extend deterministic failure injection or add owner-local fail-at-count control. | 36 |
| 195.4 | Regression Tests | Add failed allocation, cleanup, stale-output suppression, and successful retry tests. | 40 |
| 195.5 | Focused Gate And Docs | Add a focused Make/CTest target and update README/maintainer wording with exact claim boundaries. | 22 |
| 195.6 | Validation | Run focused gate, source-list check, `make format && make lint && make test`, and docs checks. | 22 |

### Deliverables

- Deterministic proof for one selected failure-path owner.
- Focused validation gate.
- Claim-safe reliability documentation.

### Total Estimate

168 hours

## Sprint 196: Epic 17 Final Validation, Claim Calibration & Closeout

**Duration:** 14 days, approximately 166 hours

**Goal:** Reconcile all Epic 17 work, run final validation, calibrate public
claims, and publish the Epic 17 retrospective and residual queue.

### Prerequisites from Previous Sprints

- Sprints 187-195 are complete or have explicit deferred/residual outcomes.
- Package, Windows, comparison, performance, maintainability, adoption, and
  reliability evidence records are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 196.1 | Evidence Reconciliation | Reconcile Sprint 187-195 outcomes, validation records, decisions, and residuals. | 26 |
| 196.2 | Claim Recalibration | Update README, INSTALL, maintainer guide, benchmark docs, API docs, and planning docs so claims match earned evidence. | 34 |
| 196.3 | Project Plan Status | Mark Epic 17 sprint items complete, narrowed, deferred, residualized, or superseded with evidence links. | 24 |
| 196.4 | Integrated Validation | Run final focused gates plus full quality gates required by changed surfaces. | 34 |
| 196.5 | Epic Retrospective | Create `EPIC_17_RETROSPECTIVE.md` with outcomes, evidence, non-claims, residuals, and state-of-the-art assessment. | 26 |
| 196.6 | Residual Queue | Publish a prioritized next-epic residual queue with exact closure targets and long-horizon deferrals. | 22 |

### Deliverables

- Final claim-recalibrated documentation.
- Integrated validation record.
- Epic 17 retrospective.
- Prioritized residual queue and state-of-the-art assessment.

### Total Estimate

166 hours

