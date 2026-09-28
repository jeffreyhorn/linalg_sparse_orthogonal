# Project Plan: Epic 19 - Productization, Platform Proof & Evidence Maturity

## Overview

Epic 19 starts from the completed Epic 18 closeout. The project has broad
solver coverage, strong selected local evidence, static-first install
validation, selected hosted Linux/macOS/Windows proof lanes, claim-boundary
guards, and a current residual queue.

The project is still not defensibly state of the art. The largest remaining
gaps are productization and evidence maturity rather than simple feature count:
package distribution beyond local proof, selected Windows freshness promotion,
broader allocation-failure reliability, large review surfaces, benchmark
methodology, generated API publication policy, shared-library/ABI/release
readiness, and external baseline methodology.

Epic 19 focuses on closing fewer gaps completely. Each sprint must either
promote one narrow earned claim with exact validation or close a deferral with
stronger guards and documentation. Broad package, Windows, ABI, performance,
release, ecosystem parity, and state-of-the-art claims remain unearned unless
the sprint explicitly records exact evidence.

## Epic 19 Goals

- Promote or deliberately close one user-facing package distribution path.
- Promote or deliberately re-close selected Windows Cholesky freshness with
  manifest, docs, and guard alignment.
- Promote or deliberately re-close selected Windows QR incompatible freshness.
- Add one additional deterministic allocation-failure owner proof.
- Reduce one additional large review surface with ownership guards.
- Decide one selected benchmark threshold/methodology policy.
- Decide generated API publication policy and implement the selected path or
  stronger local-only closure.
- Produce a concrete ABI/release readiness design or one narrow shared-library
  proof.
- Produce an executable state-of-the-art evidence blueprint without
  overclaiming.
- Finish with a coherent Epic 19 retrospective and residual queue.

## Non-Goals

Epic 19 does not assume or pre-claim:

- Homebrew/core readiness, bottles, Linuxbrew, public tap maintenance, vcpkg,
  Conan, pkgsrc, distro/system packages, binary packages, or broad
  package-manager distribution;
- broad Windows parity, Windows Makefile parity, or Windows `pkg-config`
  execution parity;
- shared-library packaging or dynamic ABI compatibility beyond an explicitly
  selected proof path;
- hosted generated API publication unless the publication sprint selects and
  proves it;
- portable performance, timing superiority, backend superiority, release
  benchmark readiness, broad external-library parity, or state-of-the-art
  sparse linear algebra status.

## Source Inputs

- `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md`
- `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`
- `docs/planning/EPIC_19/reviews/review-codex-2026-09-20.md`
- `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md`

## Epic 19 Current Status Snapshot

This snapshot starts with Sprint 207 as the first executed Epic 19 sprint.
Sprint 207 closes the package distribution support decision by selecting
continued package-provider deferral with stronger guards. Sprint 208 closes
the selected Windows Cholesky freshness decision as continued re-deferral with
stronger manifest, workflow/PowerShell, normalizer, public-doc, maintainer,
corpus, schema, and planning guard coverage. Sprint 209 closes the Windows QR
incompatible promotion decision as continued re-deferral with stronger
workflow, PowerShell, artifact-inspection, manifest, public-doc, maintainer,
corpus, schema, and planning guards. Sprint 210 closes one
additional selected allocation-failure owner proof for no-reorder linked-list
LDLT while retaining broad allocation-failure, CSC LDLT, reordered LDLT,
package, ABI, platform, performance, release, external-library parity, and
state-of-the-art non-claims. Sprints 211 through 216 remain pending future
execution.

| Sprint | Current disposition | Evidence |
| --- | --- | --- |
| 207 | Closed with continued package-provider deferral and stronger guards | `SPRINT_207/PLAN.md`; `SPRINT_207/WORKING_NOTES.md`; `SPRINT_207/artifacts/day1-package-intake.md`; `SPRINT_207/artifacts/day2-provider-scope-options.md`; `SPRINT_207/artifacts/day3-formula-metadata-baseline.md`; `SPRINT_207/artifacts/day4-environment-proof-baseline.md`; `SPRINT_207/artifacts/day5-provider-decision.md`; `SPRINT_207/artifacts/day6-proof-deferral-design.md`; `SPRINT_207/artifacts/day7-deferral-guard-implementation.md`; `SPRINT_207/artifacts/day8-proof-regression-cleanup.md`; `SPRINT_207/artifacts/day9-package-guard-alignment.md`; `SPRINT_207/artifacts/day10-user-package-docs.md`; `SPRINT_207/artifacts/day11-maintainer-package-docs.md`; `SPRINT_207/artifacts/day12-integrated-validation.md`; `SPRINT_207/artifacts/day13-review-hardening.md`; `SPRINT_207/artifacts/day14-closeout-review.md`; Sprint 207 retained the Sprint 198 developer-mode local Homebrew static source formula proof as local evidence only, added stronger package-provider overclaim guards and regression fixtures, updated user and maintainer docs, and revalidated local proof, package guards, docs/support guards, and install checks. Public tap, Homebrew/core readiness, bottles, Linuxbrew, vcpkg, Conan, pkgsrc, distro/system packages, binary packages, release packages, package-manager release readiness, shared-library packages, dynamic ABI behavior, and broad package-manager distribution remain unclaimed. |
| 208 | Closed with continued selected Windows Cholesky re-deferral and stronger guards | `SPRINT_208/PLAN.md`; `SPRINT_208/WORKING_NOTES.md`; `SPRINT_208/RETROSPECTIVE.md`; `SPRINT_208/artifacts/day1-windows-cholesky-intake.md`; `SPRINT_208/artifacts/day2-hosted-artifact-inventory.md`; `SPRINT_208/artifacts/day3-row-path-traceability.md`; `SPRINT_208/artifacts/day4-promotion-criteria.md`; `SPRINT_208/artifacts/day5-promotion-decision.md`; `SPRINT_208/artifacts/day6-manifest-metadata-design.md`; `SPRINT_208/artifacts/day7-manifest-guard-implementation.md`; `SPRINT_208/artifacts/day8-workflow-powershell-guard-alignment.md`; `SPRINT_208/artifacts/day9-normalizer-regression-design.md`; `SPRINT_208/artifacts/day10-normalizer-regression-implementation.md`; `SPRINT_208/artifacts/day11-public-docs-calibration.md`; `SPRINT_208/artifacts/day12-maintainer-corpus-docs.md`; `SPRINT_208/artifacts/day13-integrated-validation.md`; `SPRINT_208/artifacts/day14-closeout-review.md`; current branch reviewed hosted Windows Cholesky evidence and kept selected Windows freshness re-deferred while strengthening manifest, workflow/PowerShell, normalizer, public-doc, maintainer, corpus, schema, and planning guard surfaces. Generated support tier and non-claim wording still block positive selected Windows freshness promotion; broad Windows freshness, package support, ABI support, performance, release readiness, external-library parity, and state-of-the-art claims remain unearned. |
| 209 | Closed with selected Windows QR re-deferral and stronger guards | `SPRINT_209/PLAN.md`; `SPRINT_209/WORKING_NOTES.md`; `SPRINT_209/RETROSPECTIVE.md`; `SPRINT_209/artifacts/day1-windows-qr-intake.md`; `SPRINT_209/artifacts/day2-msvc-probe-design.md`; `SPRINT_209/artifacts/day3-hosted-evidence-inventory.md`; `SPRINT_209/artifacts/day4-workflow-implementation-design.md`; `SPRINT_209/artifacts/day5-workflow-implementation.md`; `SPRINT_209/artifacts/day6-artifact-inspection-tests.md`; `SPRINT_209/artifacts/day7-manifest-decision-criteria.md`; `SPRINT_209/artifacts/day8-manifest-decision.md`; `SPRINT_209/artifacts/day9-guard-integration.md`; `SPRINT_209/artifacts/day10-public-docs-calibration.md`; `SPRINT_209/artifacts/day11-maintainer-corpus-docs.md`; `SPRINT_209/artifacts/day12-focused-validation.md`; `SPRINT_209/artifacts/day13-integrated-validation.md`; `SPRINT_209/artifacts/day14-closeout-review.md`; current branch has no hosted Windows CI run artifact for QR incompatible promotion, so `SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only and `local_only`; the Sprint 209 Windows QR lane is evidence-collection only, with workflow/PowerShell, normalizer, manifest, public-doc, maintainer, corpus, schema, and planning guards preserving the non-claim boundary. |
| 210 | Closed with selected linked-list LDLT allocation-failure owner proof | `SPRINT_210/PLAN.md`; `SPRINT_210/WORKING_NOTES.md`; `SPRINT_210/artifacts/day1-allocation-proof-intake.md`; `SPRINT_210/artifacts/day2-owner-ranking.md`; `SPRINT_210/artifacts/day3-lifecycle-baseline.md`; `SPRINT_210/artifacts/day4-harness-design.md`; `SPRINT_210/artifacts/day5-harness-implementation.md`; `SPRINT_210/artifacts/day6-failure-sweep.md`; `SPRINT_210/artifacts/day7-cleanup-proof.md`; `SPRINT_210/artifacts/day8-stale-output-preservation.md`; `SPRINT_210/artifacts/day9-retry-proof.md`; `SPRINT_210/artifacts/day10-focused-gate.md`; `SPRINT_210/artifacts/day11-documentation-calibration.md`; `SPRINT_210/artifacts/day12-integrated-validation.md`; `SPRINT_210/artifacts/day13-review-hardening.md`; `SPRINT_210/artifacts/day14-closeout-review.md`; Sprint 210 selects no-reorder linked-list LDLT numeric factorization as one additional allocation-failure owner, adds deterministic 25-site failure injection, cleanup, stale-output, caller-input preservation, retry, active-registration guard, focused Make/CTest gate, user/maintainer docs, and integrated validation. CSC LDLT, reordered LDLT, Cholesky, broad direct solver, QR/SVD/eigensolver, matrix construction/conversion/IO, package/install, generated tooling, OS OOM, concurrent hook, hosted, platform, ABI, performance, release, external-library parity, and state-of-the-art reliability claims remain unearned. |
| 211-216 | Pending future execution | No Epic 19 branch-local execution artifacts are present yet for these sprint sections. |

## Sprint 207: Package Distribution Support Decision

**Duration:** 14 days, approximately 166 hours

**Goal:** Promote one exact package distribution path, or close the package
distribution residual with stronger deferral guards and no user-facing package
claim.

### Prerequisites from Previous Sprints

- Sprint 198 local Homebrew static source proof is merged.
- Package-manager and static-package deferral guards are present.
- Root MIT license metadata exists.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 207.1 | Provider Scope Decision | Decide exact package target: public Homebrew tap/source formula, Homebrew/core readiness, or explicit continued deferral. | 24 |
| 207.2 | Formula And Metadata Audit | Audit `packaging/homebrew/`, root metadata, source archive/checksum behavior, license wording, and installed static surface. | 24 |
| 207.3 | Proof Path Implementation | Add or update the selected proof command, formula checks, environment gates, and cleanup behavior. | 34 |
| 207.4 | Package Guard Alignment | Update package-manager and static-package guards to enforce the selected support tier and retained non-claims. | 28 |
| 207.5 | User And Maintainer Docs | Update README, INSTALL, packaging docs, and maintainer guide with only the earned provider status. | 28 |
| 207.6 | Validation And Closeout | Run package proof, install tests, docs guards, static package guard, and full C gate if `.c` or `.h` changed. | 28 |

### Deliverables

- Provider-scope decision artifact.
- Updated package proof or strengthened deferral guard.
- Claim-safe package documentation.
- Sprint 207 retrospective and residuals.

### Total Estimate

166 hours

## Sprint 208: Selected Windows Cholesky Freshness Promotion

**Duration:** 14 days, approximately 166 hours

**Goal:** Fully promote or deliberately re-defer the selected Windows Cholesky
freshness lane using hosted evidence, manifest metadata, docs, and guards.

### Prerequisites from Previous Sprints

- Sprint 199 re-deferral evidence is merged.
- Windows selected Cholesky workflow exists.
- Selected target manifest and PowerShell guards are current.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 208.1 | Hosted Artifact Intake | Fetch and inspect latest hosted Windows `cholesky-spd-tridiag-5` evidence, row IDs, paths, and artifact membership. | 26 |
| 208.2 | Manifest Promotion Decision | Decide whether the manifest earns selected Windows freshness metadata or remains re-deferred. | 28 |
| 208.3 | Metadata And Guard Implementation | Update manifest, workflow metadata, or absence guards according to the decision. | 32 |
| 208.4 | Normalizer Regression Coverage | Add tests for Windows path normalization, selected target filtering, missing rows, stale rows, and artifact mismatch. | 30 |
| 208.5 | Documentation Calibration | Update README, INSTALL, corpus docs, and maintainer guide with promoted or re-deferred wording. | 24 |
| 208.6 | Validation And Closeout | Run selected manifest, workflow, PowerShell, normalizer, freshness, docs, and applicable C quality gates. | 26 |

### Deliverables

- Windows Cholesky promotion or re-deferral decision.
- Manifest/workflow/guard updates.
- Hosted evidence ledger.
- Claim-safe docs.

### Total Estimate

166 hours

## Sprint 209: Windows QR Incompatible Promotion Decision

**Duration:** 14 days, approximately 168 hours

**Goal:** Add hosted Windows/MSVC proof for `qr-incompatible-ls` and promote
selected metadata only if the exact evidence supports it.

### Prerequisites from Previous Sprints

- Sprint 203 local QR incompatible proof and re-deferral guards are merged.
- Windows CMake/MSVC workflow is available.
- Selected comparison generation and normalization tests are current.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 209.1 | MSVC Probe Design | Define hosted MSVC/CMake QR incompatible proof command, artifact layout, and expected rows. | 24 |
| 209.2 | Workflow Implementation | Add or update hosted Windows QR proof workflow steps without broad Windows promotion. | 34 |
| 209.3 | Artifact Inspection Tests | Add tests for Windows-style artifact paths, row filtering, generated rows, and stale/missing artifacts. | 32 |
| 209.4 | Manifest Decision | Promote or re-defer selected QR incompatible Windows metadata based on hosted proof. | 28 |
| 209.5 | Docs And Claim Guards | Update README, INSTALL, corpus docs, maintainer guide, and Windows/manifest guards. | 24 |
| 209.6 | Validation And Closeout | Run QR generator/freshness, normalizer, manifest, workflow, PowerShell, QR-focused, docs, and full C gates if required. | 26 |

### Deliverables

- Hosted QR incompatible Windows proof or explicit re-deferral.
- Artifact and manifest regression tests.
- Updated Windows QR claim documentation.

### Total Estimate

168 hours

## Sprint 210: Additional Allocation-Failure Owner Proof

**Duration:** 14 days, approximately 168 hours

**Goal:** Add deterministic allocation-failure proof for one new high-value
owner outside the already closed selected symbolic LU path.

### Prerequisites from Previous Sprints

- Sprint 200 symbolic LU allocation proof is merged.
- Existing allocation hook infrastructure and focused gates are available.
- Candidate owner ranking from Epic 19 review is accepted.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 210.1 | Owner Selection | Rank and select one owner from matrix import/export, QR workspace, LDLT/Cholesky, eigensolver, or SVD workspace paths. | 22 |
| 210.2 | Lifecycle Invariant Record | Document status, cleanup, stale-output, caller-input, retry, and partial-publication invariants. | 24 |
| 210.3 | Harness Extension | Extend deterministic allocation failure hooks and focused fixtures for the selected owner. | 38 |
| 210.4 | Regression Tests | Add failed-allocation, cleanup, stale-output, preservation, and retry tests. | 34 |
| 210.5 | Gate And Documentation | Add focused Make/CTest gate, registration guard if needed, and claim-safe documentation. | 24 |
| 210.6 | Validation And Closeout | Run focused gate, family tests, source-list checks, docs checks, and full C quality gate. | 26 |

### Deliverables

- One new selected allocation-failure owner proof.
- Focused gate and tests.
- Updated reliability claim docs.

### Total Estimate

168 hours

## Sprint 211: Large Review-Surface Reduction

**Duration:** 14 days, approximately 166 hours

**Goal:** Reduce one large high-risk implementation, test, or tooling surface
with behavior-preserving extraction and ownership guards.

### Prerequisites from Previous Sprints

- Sprint 201 selected SVD helper reduction is merged.
- Large-surface ranking from Epic 19 review is available.
- Full validation can run if source/header files change.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 211.1 | Candidate Ranking | Rank large C tests, C implementations, and Python tools by risk, churn, and review value. | 22 |
| 211.2 | Cluster Boundary | Select one cluster and record no-behavior-change invariants, ownership, and non-goals. | 24 |
| 211.3 | Extraction Design | Design helper/module split, registration impact, include dependencies, and guard strategy. | 26 |
| 211.4 | Extraction Implementation | Move selected code into helper/module ownership without changing behavior. | 38 |
| 211.5 | Guard And Test Coverage | Add or update helper ownership, registration, order, source-list, and focused regression tests. | 30 |
| 211.6 | Validation And Closeout | Run focused tests, relevant guards, source-list/CMake parity, docs checks, and full C gate if source/header changed. | 26 |

### Deliverables

- One reduced review surface.
- Behavior-preservation artifact.
- Ownership guard and focused regression coverage.

### Total Estimate

166 hours

## Sprint 212: Benchmark Methodology And Threshold Policy

**Duration:** 14 days, approximately 166 hours

**Goal:** Decide and implement one selected benchmark methodology policy:
thresholded gate for one row or stronger threshold-free deferral proof.

### Prerequisites from Previous Sprints

- Sprint 202 macOS hosted selected benchmark freshness is merged.
- Benchmark freshness scripts and selected target manifest are current.
- Benchmark docs preserve non-portable performance wording.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 212.1 | Benchmark Evidence Inventory | Inventory selected hosted benchmark lanes, report fields, runner metadata, and non-claims. | 22 |
| 212.2 | Threshold Decision | Decide whether to add one selected threshold gate or close threshold deferral with stronger methodology docs. | 28 |
| 212.3 | Methodology Implementation | Implement threshold metadata with runner class, compiler, repeats, warmup, variance rule, and allowed regression threshold, or implement threshold-free guard improvements for the selected decision. | 34 |
| 212.4 | Regression Tests | Add benchmark freshness, manifest, methodology, missing metadata, and docs guard tests. | 30 |
| 212.5 | Documentation Calibration | Update benchmark README, README, INSTALL, maintainer guide, and selected manifest wording. | 26 |
| 212.6 | Validation And Closeout | Run benchmark freshness, selected performance docs, manifest tests, docs checks, and relevant CI evidence review. | 26 |

### Deliverables

- Selected benchmark threshold or threshold-free methodology decision.
- Updated benchmark tooling/tests/docs.
- No portable performance overclaim.

### Total Estimate

166 hours

## Sprint 213: Generated API Publication Decision

**Duration:** 14 days, approximately 166 hours

**Goal:** Decide whether generated API HTML remains local-only or is published,
then implement the selected policy with matching automation.

### Prerequisites from Previous Sprints

- Sprint 204 stronger local-only generated API policy is merged.
- `make api-docs-freshness` and API routing/local-only guards are current.
- Doxygen public header coverage is stable.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 213.1 | Publication Option Review | Compare local-only, hosted Pages, retained artifact, and committed generated HTML policies. | 24 |
| 213.2 | Policy Decision | Select one generated API policy and record support, retention, routing, and claim implications. | 24 |
| 213.3 | Automation Implementation | Implement workflow, retention, route, staging, or strengthened local-only guard behavior. | 38 |
| 213.4 | Routing And Guard Tests | Add tests for publication links, generated-output staging, workflow references, and source-controlled route coverage. | 32 |
| 213.5 | User And Maintainer Docs | Update API reference, README, INSTALL, maintainer guide, and generated API evidence docs. | 22 |
| 213.6 | Validation And Closeout | Run docs-check, api-docs-freshness, routing/local-only tests, workflow checks, and full C gate if headers changed. | 26 |

### Deliverables

- Final generated API publication policy for Epic 19.
- Matching automation and documentation.
- Updated residual status.

### Total Estimate

166 hours

## Sprint 214: ABI, Shared-Library And Release Readiness Decision

**Duration:** 14 days, approximately 168 hours

**Goal:** Either implement one narrow shared-library/ABI proof path or close a
complete release/ABI design with acceptance gates and deferral guards.

### Prerequisites from Previous Sprints

- Static-first install/export proof is current.
- Shared-library and dynamic ABI support remain explicitly deferred.
- Package distribution decision from Sprint 207 is available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 214.1 | ABI/Release Scope Decision | Decide whether Epic 19 implements a narrow shared proof or only a complete design/deferral closure. | 24 |
| 214.2 | ABI Policy Design | Define symbol visibility, versioning, compatibility, loader, and platform metadata expectations. | 30 |
| 214.3 | Build/Install Proof Or Guard | Add narrow shared-library proof path or strengthen configure/docs guards for deferral. | 38 |
| 214.4 | Downstream Consumer Evidence | Add installed shared consumer tests if implemented, or exact future acceptance tests if deferred. | 28 |
| 214.5 | Release Checklist | Add changelog, tag, reproducible source archive requirement, artifact, package provenance, and CI evidence checklist. | 24 |
| 214.6 | Validation And Closeout | Run install/export tests, static/shared guards, docs checks, and full C/build quality gates. | 24 |

### Deliverables

- ABI/release decision record.
- Shared proof or strengthened deferral guard.
- Release readiness checklist.

### Total Estimate

168 hours

## Sprint 215: External Baseline And State-Of-The-Art Evidence Blueprint

**Duration:** 14 days, approximately 166 hours

**Goal:** Create an executable state-of-the-art evidence blueprint and, if
feasible, one selected external-baseline pilot without broad parity claims.

### Prerequisites from Previous Sprints

- Benchmark methodology decision from Sprint 212 is available.
- Selected comparison/report tooling is current.
- Public docs already avoid broad state-of-the-art claims.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 215.1 | Baseline Selection | Select exact external baselines, versions, solver families, and matrix-suite subset for future parity work. | 26 |
| 215.2 | Methodology Specification | Define correctness metrics, tolerances, performance metrics, platforms, compilers, package provenance, and reporting fields. | 34 |
| 215.3 | Report Schema Design | Extend or design report schema fields for external baseline identity, methodology, and evidence status. | 28 |
| 215.4 | Selected Pilot | Implement one small external-baseline pilot if feasible, otherwise record blocker and executable design. | 34 |
| 215.5 | Claim Documentation | Update docs to distinguish blueprint, selected pilot, and unearned broad state-of-the-art claims. | 20 |
| 215.6 | Validation And Closeout | Run schema, report, comparison, docs, and focused pilot validation; full C gate if source/header changed. | 24 |

### Deliverables

- State-of-the-art evidence blueprint.
- Optional selected external-baseline pilot.
- Updated claim-boundary docs and residual queue.

### Total Estimate

166 hours

## Sprint 216: Epic 19 Final Validation, Claim Calibration & Closeout

**Duration:** 14 days, approximately 166 hours

**Goal:** Reconcile Sprint 207-215 outcomes, calibrate claims, publish the
Epic 19 retrospective and residual queue, and decide whether any stronger
support or state-of-the-art claim is earned.

### Prerequisites from Previous Sprints

- Sprints 207-215 are complete or have explicit deferred/residual outcomes.
- All sprint retrospectives, artifacts, validation records, and PR evidence
  are merged.
- The Epic 19 review and todo docs are present.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 216.1 | Evidence Reconciliation | Reconcile Sprint 207-215 outcomes, validation records, decisions, and residuals. | 26 |
| 216.2 | Claim Recalibration | Update README, INSTALL, maintainer guide, benchmark docs, API docs, corpus docs, and planning docs so claims match earned evidence. | 34 |
| 216.3 | Project Plan Status | Mark Epic 19 sprint items complete, promoted, deferred, residualized, or superseded with evidence links. | 24 |
| 216.4 | Integrated Validation | Run final focused gates plus full quality gates required by changed surfaces. | 34 |
| 216.5 | Epic Retrospective | Create `EPIC_19_RETROSPECTIVE.md` with outcomes, evidence, non-claims, residuals, and state-of-the-art assessment. | 26 |
| 216.6 | Residual Queue | Publish a prioritized next-epic residual queue with exact closure targets and long-horizon deferrals. | 22 |

### Deliverables

- Final claim-recalibrated documentation.
- Integrated validation record.
- Epic 19 retrospective.
- Prioritized residual queue and state-of-the-art assessment.

### Total Estimate

166 hours
