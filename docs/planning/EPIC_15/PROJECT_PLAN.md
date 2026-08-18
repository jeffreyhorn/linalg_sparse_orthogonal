# Project Plan: Epic 15 - Evidence Publication, ABI Decision & Adoption Hardening

## Overview

Epic 15 focuses on closing the most important residual gaps after Epic 14. The project already has broad solver coverage, substantial tests, static-first install validation, generated reports, and disciplined non-claims. The remaining work should improve credibility by proving selected claims end-to-end, simplifying adoption, and making product decisions explicit.

The primary themes are:

- Promote selected performance and comparison evidence from local or narrow proof to maintained, freshness-checked publication.
- Close the shared-library ABI and package-manager readiness decisions without accidental overclaiming.
- Continue public-header and generated API documentation cleanup.
- Strengthen cross-platform report and install clarity.
- Add targeted allocation-failure evidence for one high-risk subsystem.
- Recalibrate state-of-the-art language to match actual evidence.

## Sprint 167: Epic 15 Baseline, Evidence Ledger & Claim Gate

**Duration:** 14 days, approximately 166 hours

**Goal:** Establish the Epic 15 baseline and define the exact evidence gates for performance, ABI, package, API, comparison, and platform claims.

### Prerequisites from Previous Sprints

- Epic 14 retrospective and PR #184 have landed on master.
- Existing generated reports, install validation, package metadata, and claim docs are available for audit.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 167.1 | Residual Queue Audit | Extract unresolved Epic 13 and Epic 14 residuals and classify them by claim risk, user value, and closure feasibility. | 24 |
| 167.2 | Evidence Ledger | Create a ledger mapping current claims to source files, docs, reports, CI lanes, and unsupported non-claims. | 32 |
| 167.3 | CI and Report Inventory | Inventory hosted/local CI lanes, generated report freshness checks, package checks, and comparison reports. | 28 |
| 167.4 | Gap Selection Gate | Select the exact gaps Epic 15 will close and define acceptance criteria for each. | 30 |
| 167.5 | Sprint Artifact Setup | Create Sprint 167 artifacts, working notes, and a baseline status summary for later retrospectives. | 24 |
| 167.6 | Validation Pass | Run documentation consistency checks and lightweight repository sanity checks for the new planning artifacts. | 28 |

### Deliverables

- Epic 15 evidence ledger.
- Sprint 167 artifacts and working notes.
- Claim gate criteria for the rest of Epic 15.

### Total Estimate

166 hours

## Sprint 168: Hosted Performance Publication Lane

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote one selected performance report into a hosted, freshness-checked CI lane with methodology-bound claims.

### Prerequisites from Previous Sprints

- Sprint 167 evidence ledger identifies the selected performance family and CI target.
- Existing benchmark/report scripts are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 168.1 | Performance Lane Selection | Choose one solver family, benchmark command, matrix scope, and platform lane suitable for hosted publication. | 18 |
| 168.2 | Methodology Metadata | Add compiler, flags, CPU, thread, repeat, warmup, variance, and timestamp metadata to the selected report. | 34 |
| 168.3 | Freshness Check | Add a strict freshness check for the selected hosted performance report. | 28 |
| 168.4 | CI Wiring | Wire the selected performance report into hosted CI with bounded runtime and clear failure output. | 34 |
| 168.5 | Claim-Safe Docs | Update README/report docs to describe only the proven performance scope. | 26 |
| 168.6 | Verification | Run the selected benchmark/report checks and standard docs sanity checks. | 28 |

### Deliverables

- Hosted performance report lane for one selected scope.
- Freshness-checked performance report.
- Updated claim-safe performance documentation.

### Total Estimate

168 hours

## Sprint 169: Performance Methodology Hardening

**Duration:** 14 days, approximately 166 hours

**Goal:** Turn the selected performance lane into a durable methodology-bound publication surface.

### Prerequisites from Previous Sprints

- Sprint 168 hosted performance lane exists.
- Performance report schema includes basic methodology metadata.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 169.1 | Statistical Policy | Define repeat-count, warmup, variance, and threshold policy for selected performance reports. | 28 |
| 169.2 | Report Schema Cleanup | Normalize report fields so generated performance rows are stable and diff-friendly. | 30 |
| 169.3 | Regression Sentinel | Add or tighten a bounded sentinel that catches large regressions without claiming universal speed. | 34 |
| 169.4 | Documentation Indexing | Link the selected performance report from the canonical report index and README evidence table. | 24 |
| 169.5 | Platform Caveats | Document exact platform and backend constraints for the selected performance lane. | 20 |
| 169.6 | Quality Gate | Run report generation, freshness checks, and relevant test commands. | 30 |

### Deliverables

- Methodology-bound performance report policy.
- Stable selected performance report schema.
- Regression sentinel and indexed report documentation.

### Total Estimate

166 hours

## Sprint 170: Shared-Library ABI Product Decision

**Duration:** 14 days, approximately 168 hours

**Goal:** Close the shared-library ABI question with an explicit product decision and enforceable package metadata behavior.

### Prerequisites from Previous Sprints

- Static-first package contract and install validation are present.
- Epic 15 evidence ledger identifies all shared-library and ABI non-claims.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 170.1 | ABI Surface Audit | Audit exported headers, structs, constants, versioning, and lifecycle semantics for ABI readiness. | 34 |
| 170.2 | Build-System Feasibility | Review Make/CMake behavior for static-only, shared-library rejection, symbol visibility, and install metadata. | 28 |
| 170.3 | Product Decision Record | Create a decision record choosing static-first-only continuation or a staged shared-library path. | 30 |
| 170.4 | Guard Updates | Update build/package tests so unsupported shared-library or ABI claims cannot appear accidentally. | 32 |
| 170.5 | Documentation Alignment | Update README, install docs, package docs, and non-claim tables to match the decision. | 24 |
| 170.6 | Validation | Run install/package checks and docs sanity checks. | 20 |

### Deliverables

- Shared-library ABI product decision record.
- Updated guards for package metadata and unsupported ABI claims.
- Documentation aligned with the selected decision.

### Total Estimate

168 hours

## Sprint 171: Package-Manager Readiness First Provider

**Duration:** 14 days, approximately 168 hours

**Goal:** Close one package-manager readiness path or formally document and enforce package-manager deferral.

### Prerequisites from Previous Sprints

- Sprint 170 package and ABI decision is complete.
- Static-first source install behavior is validated.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 171.1 | Provider Selection | Select one package-manager proof path such as vcpkg, Homebrew, or a documented deferral. | 18 |
| 171.2 | Recipe or Deferral Artifact | Add the selected manifest/formula/recipe or a formal unsupported-provider decision artifact. | 34 |
| 171.3 | Local Proof Script | Add a local validation path for install, compile, version query, and cleanup where provider support is selected. | 36 |
| 171.4 | Package Claim Guard | Ensure docs and metadata distinguish source install, CMake/pkg-config install, and package-manager support. | 28 |
| 171.5 | User Documentation | Add concise package-manager guidance or explicit non-claim wording. | 24 |
| 171.6 | Verification | Run install validation and package/provider proof or deferral checks. | 28 |

### Deliverables

- One package-manager readiness decision.
- Provider proof artifact or explicit deferral artifact.
- Updated package documentation and claim guards.

### Total Estimate

168 hours

## Sprint 172: Public Header Coherence Batch 2

**Duration:** 14 days, approximately 168 hours

**Goal:** Normalize one additional high-impact public header family and add lightweight guardrails against documentation drift.

### Prerequisites from Previous Sprints

- Prior header cleanup batches from Epic 14 are available as examples.
- Sprint 167 evidence ledger identifies the next highest-impact header family.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 172.1 | Header Family Selection | Choose the highest-impact public header family based on adoption risk and API complexity. | 16 |
| 172.2 | Contract Cleanup | Normalize ownership, lifecycle, error-code, tolerance, workspace, and threading language. | 42 |
| 172.3 | Declaration Organization | Reorder declarations into coherent sections without changing API behavior. | 30 |
| 172.4 | Example Alignment | Update examples or docs so the cleaned header has a clear recommended workflow. | 28 |
| 172.5 | Mechanical Guard | Add or extend a lightweight check for declaration/docs coverage where practical. | 28 |
| 172.6 | Validation | Run format/lint/test if code headers changed, plus docs checks. | 24 |

### Deliverables

- Cleaned public header family.
- Updated examples or usage docs.
- Mechanical or reviewable guard against header documentation drift.

### Total Estimate

168 hours

## Sprint 173: Generated API HTML Publication Closure

**Duration:** 14 days, approximately 166 hours

**Goal:** Decide and implement the supported generated API HTML publication path.

### Prerequisites from Previous Sprints

- Sprint 172 header cleanup improves generated API inputs.
- Existing local generated API HTML and freshness checks are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 173.1 | Publication Decision | Decide whether generated API HTML is hosted, committed, artifact-only, or local-only. | 22 |
| 173.2 | Generator Audit | Audit generation scripts, inputs, output paths, and freshness behavior. | 28 |
| 173.3 | Publication Implementation | Implement the selected publication or local-only enforcement path. | 40 |
| 173.4 | Freshness Gate | Add CI or local checks proving the selected generated API status remains accurate. | 30 |
| 173.5 | Navigation Update | Update README and docs indexes so users can find the supported API reference. | 22 |
| 173.6 | Verification | Run generator, freshness, docs, and relevant build checks. | 24 |

### Deliverables

- Supported generated API HTML publication decision.
- Implemented publication or local-only enforcement.
- Updated docs navigation and freshness checks.

### Total Estimate

166 hours

## Sprint 174: Additional Bounded External Comparison Family

**Duration:** 14 days, approximately 168 hours

**Goal:** Add one more complete external comparison family with generated report, freshness checks, and bounded claims.

### Prerequisites from Previous Sprints

- Existing comparison harness and report-index architecture are available.
- Sprint 167 evidence ledger identifies the highest-value comparison family.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 174.1 | Family Selection | Select the solver family, fixtures, external comparator, and tolerances. | 22 |
| 174.2 | Fixture Design | Add matrix fixtures that cover the selected family without broad unsupported claims. | 32 |
| 174.3 | Harness Extension | Extend the external comparison runner and outputs for the selected family. | 40 |
| 174.4 | Report Integration | Generate, index, and freshness-check the comparison report. | 28 |
| 174.5 | Claim Documentation | Document exact solver, fixture, comparator, and tolerance scope. | 20 |
| 174.6 | Validation | Run comparison generation, freshness checks, and relevant tests. | 26 |

### Deliverables

- One complete bounded external comparison family.
- Generated and indexed comparison report.
- Claim-safe documentation.

### Total Estimate

168 hours

## Sprint 175: Cross-Platform Report Freshness Promotion

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote one selected report freshness path beyond Linux or formally close the cross-platform deferral with precise blockers.

### Prerequisites from Previous Sprints

- Report index and freshness gates exist for selected families.
- Windows and macOS platform tiers are documented.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 175.1 | Platform Gap Matrix | Classify generated reports by Linux, macOS, Windows, local-only, and hosted status. | 26 |
| 175.2 | Promotion Candidate | Select one report family for macOS or Windows promotion, or select formal deferral. | 20 |
| 175.3 | CI or Deferral Work | Implement the selected platform freshness lane or add enforceable deferral documentation. | 40 |
| 175.4 | Path Normalization | Fix platform path, newline, shell, or executable assumptions in the selected report path. | 30 |
| 175.5 | Tier Documentation | Update README and report indexes to match the precise platform status. | 24 |
| 175.6 | Verification | Run local checks and relevant platform-neutral validation commands. | 28 |

### Deliverables

- Cross-platform report freshness matrix.
- One promoted report freshness lane or formal deferral closure.
- Updated platform-tier documentation.

### Total Estimate

168 hours

## Sprint 176: Allocation-Failure Evidence, Claim Recalibration & Epic Closeout

**Duration:** 14 days, approximately 168 hours

**Goal:** Add one targeted allocation-failure proof, reconcile all Epic 15 claims, and close the epic with evidence-bound documentation.

### Prerequisites from Previous Sprints

- Epic 15 evidence ledger has been updated across Sprints 167-175.
- Performance, package, API, comparison, and platform decisions are complete.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 176.1 | Subsystem Selection | Choose one allocation-heavy solver or shared subsystem for failure-path proof. | 16 |
| 176.2 | Failure Harness | Add deterministic allocation-failure or cleanup-path tests for the selected subsystem. | 38 |
| 176.3 | Cleanup Invariants | Document ownership and cleanup invariants for the selected subsystem. | 22 |
| 176.4 | Claim Recalibration | Update README, docs indexes, non-claim tables, and evidence ledger to match completed work. | 34 |
| 176.5 | Epic Retrospective | Create the Epic 15 retrospective with earned claims, non-claims, residuals, and validation evidence. | 28 |
| 176.6 | Final Validation | Run required quality gates and record final status. | 30 |

### Deliverables

- Allocation-failure proof for one selected subsystem.
- Updated claim and non-claim documentation.
- Epic 15 retrospective and final validation record.

### Total Estimate

168 hours

## Epic 15 Success Criteria

- At least one hosted performance publication path is fresh, indexed, and methodology-bound.
- Shared-library ABI support has a clear product decision and enforceable documentation/build behavior.
- Package-manager readiness is either proven for one selected provider or explicitly deferred with guarded non-claims.
- One additional public-header family has a cleaner, documented API contract.
- Generated API HTML publication status is clear and freshness-checked.
- One additional bounded external comparison family is complete.
- Cross-platform report freshness has either one promoted lane or an explicit deferral closure.
- One allocation-heavy subsystem has deterministic failure-path evidence.
- The README and planning docs make no broad state-of-the-art, performance, package, ABI, or platform claims beyond what the evidence supports.
