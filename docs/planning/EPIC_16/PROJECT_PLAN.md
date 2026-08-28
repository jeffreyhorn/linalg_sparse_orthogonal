# Project Plan: Epic 16 - Product Evidence, Failure-Path Breadth & Distribution Closure

## Overview

Epic 16 starts from the completed Epic 15 closeout. The project now has broad
solver coverage, selected hosted evidence, static-first package validation,
public non-claim discipline, generated-report freshness gates, and one
bounded allocation-failure proof. The remaining highest-value gaps are
productization and evidence gaps rather than missing algorithm count.

Epic 16 focuses on closing fewer gaps completely:

- broaden allocation-failure evidence to one additional subsystem;
- decide generated API HTML publication status with enforcement;
- close one package-manager provider decision;
- centralize selected report target metadata to reduce workflow/report drift;
- promote or formally close Windows report freshness;
- add one bounded external comparison family;
- continue public-header/API coherence cleanup;
- reduce one large source/test review surface;
- recalibrate state-of-the-art and support-tier claims at closeout.

The epic deliberately avoids broad state-of-the-art, portable performance,
external ecosystem parity, dynamic ABI, shared-library, package-manager,
Windows parity, or generated-report parity claims unless the corresponding
recurring evidence is added.

## Sprint 177: Epic 16 Baseline, Evidence Matrix & Closure Gates

**Duration:** 14 days, approximately 166 hours

**Goal:** Freeze the post-Epic-15 baseline, select Epic 16 closure targets,
and publish the acceptance gates for product evidence, failure-path proof,
distribution, generated API status, and report metadata.

### Prerequisites from Previous Sprints

- Epic 15 retrospective and PR #195 are merged on `master`.
- Current README, maintainer guide, install docs, generated reports, package
  guards, and selected hosted workflow lanes are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 177.1 | Residual Queue Audit | Extract Epic 13-15 residuals and classify them by user value, claim risk, feasibility, and support-tier impact. | 24 |
| 177.2 | Evidence Status Matrix | Create a current evidence/status matrix covering package, API docs, reports, comparisons, performance, platform, ABI, and allocation-failure surfaces. | 34 |
| 177.3 | Closure Target Selection | Select the exact Epic 16 gaps to close and record explicit non-goals for broad state-of-the-art, ABI, platform, and ecosystem claims. | 26 |
| 177.4 | Acceptance Gate Templates | Define validation commands, owner files, support tiers, hosted/local boundaries, and claim wording gates for each selected gap. | 30 |
| 177.5 | Quality Surface Map | Map required validation for documentation-only, script, workflow, package, generated-report, header, and C source changes. | 24 |
| 177.6 | Sprint Setup and Handoff | Create Sprint 177 artifacts, working notes, and handoffs for allocation-failure and generated API publication work. | 28 |

### Deliverables

- Epic 16 selected-gap register.
- Evidence/status matrix.
- Acceptance gate templates.
- Quality surface map and sprint handoffs.

### Total Estimate

166 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 177.1 | Complete | `SPRINT_177/artifacts/day2-residual-audit.md`; `SPRINT_177/artifacts/day3-residual-classification.md`; Sprint 186 Day 3 reconciled matrix. |
| 177.2 | Complete | `SPRINT_177/artifacts/day5-matrix-schema.md`; `SPRINT_177/artifacts/day6-populated-matrix.md`; Sprint 186 Day 3 reconciled matrix. |
| 177.3 | Complete | `SPRINT_177/artifacts/day7-target-selection.md`; Sprint 186 Day 3 reconciled matrix. |
| 177.4 | Complete | `SPRINT_177/artifacts/day8-gate-templates.md`; `SPRINT_177/artifacts/day9-gate-completion.md`; later sprint validation records. |
| 177.5 | Complete | `SPRINT_177/artifacts/day10-quality-surface-map.md`; Sprint 186 Day 9 validation-planning inputs. |
| 177.6 | Complete | `SPRINT_177/artifacts/day12-handoff-package.md`; `SPRINT_177/artifacts/day13-reconciliation.md`; `SPRINT_177/artifacts/day14-closeout.md`. |

## Sprint 178: Allocation-Failure Proof Batch 2

**Duration:** 14 days, approximately 168 hours

**Goal:** Add deterministic allocation-failure cleanup evidence for one
additional high-risk subsystem beyond iterative repeated-run handles.

### Prerequisites from Previous Sprints

- Sprint 177 selected the subsystem and acceptance gate.
- Sprint 176 allocation-failure hook semantics and iterative proof are
  available as the reference pattern.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 178.1 | Subsystem Selection Detail | Confirm the selected allocation-heavy subsystem, entry points, ownership graph, and failure sites. | 18 |
| 178.2 | Cleanup Invariant Record | Document expected cleanup behavior, no-publication rules, retry semantics, and unsupported breadth. | 24 |
| 178.3 | Harness Extension | Extend the allocation-failure hook or add a subsystem-local deterministic fail-at-count harness. | 34 |
| 178.4 | Regression Tests | Add tests for failed construction/factorization/conversion, cleanup, no stale state publication, and successful retry. | 40 |
| 178.5 | Focused Gate | Add a Make/CTest target and label for the selected subsystem proof. | 22 |
| 178.6 | Claim Documentation and Validation | Update README/maintainer wording and run required code quality gates. | 30 |

### Deliverables

- Deterministic allocation-failure proof for one additional subsystem.
- Cleanup invariant documentation.
- Focused validation target and scoped public claim wording.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 178.1 | Complete | `SPRINT_178/artifacts/day3-subsystem-selection.md`; Sprint 186 Day 3 reconciled matrix. |
| 178.2 | Complete | `SPRINT_178/artifacts/day4-cleanup-invariant.md`; `SPRINT_178/artifacts/day9-cleanup-error-contracts.md`. |
| 178.3 | Complete | `SPRINT_178/artifacts/day5-harness-design.md`; `SPRINT_178/artifacts/day6-harness-implementation.md`. |
| 178.4 | Complete | `SPRINT_178/artifacts/day7-first-regression.md`; `SPRINT_178/artifacts/day8-coverage-expansion.md`; `SPRINT_178/RETROSPECTIVE.md`. |
| 178.5 | Complete | `SPRINT_178/artifacts/day10-focused-gate.md`; `make matmul-allocation-failure-gate`. |
| 178.6 | Complete | `SPRINT_178/artifacts/day11-scoped-claim-documentation.md`; `SPRINT_178/artifacts/day12-integrated-validation.md`; `SPRINT_178/RETROSPECTIVE.md`. |

## Sprint 179: Generated API HTML Publication Decision

**Duration:** 14 days, approximately 166 hours

**Goal:** Close generated API HTML status with either a hosted publication
path or a stronger enforced local-only product decision.

### Prerequisites from Previous Sprints

- Sprint 177 evidence matrix identifies current generated API status.
- Existing Doxygen configuration and local-only guards are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 179.1 | Doxygen Surface Audit | Audit generated inputs, ignored outputs, warnings, page coverage, source-header authority, and current docs navigation. | 24 |
| 179.2 | Publication Decision | Decide hosted artifact, retained CI artifact, committed output, or continued local-only status. | 22 |
| 179.3 | Implementation | Implement the selected publication or local-only enforcement path. | 40 |
| 179.4 | Freshness and Staging Guard | Add or tighten checks for stale output, missing pages, staged generated files, and publication metadata. | 32 |
| 179.5 | Navigation Update | Update README, API reference, maintainer guide, and support-tier docs to point to the supported API path. | 24 |
| 179.6 | Verification | Run docs generation, docs checks, generated API checks, and `git diff --check`. | 24 |

### Deliverables

- Generated API HTML product decision.
- Implemented publication or enforced local-only behavior.
- Updated docs navigation and freshness evidence.

### Total Estimate

166 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 179.1 | Complete | `SPRINT_179/artifacts/day2-doxygen-surface-audit.md`; `SPRINT_179/artifacts/day3-warning-and-coverage-audit.md`. |
| 179.2 | Narrowed | `SPRINT_179/artifacts/day5-publication-decision-matrix.md`; `SPRINT_179/artifacts/day6-product-decision-record.md`; residual `R186-HOSTED-API`. |
| 179.3 | Complete | `SPRINT_179/artifacts/day7-implementation-design.md`; `SPRINT_179/artifacts/day8-core-implementation.md`; `SPRINT_179/artifacts/day9-enforcement-completion.md`. |
| 179.4 | Complete | `SPRINT_179/artifacts/day10-freshness-and-staging-guard.md`; `make api-docs-freshness`. |
| 179.5 | Complete | `SPRINT_179/artifacts/day11-navigation-and-claim-update.md`; Sprint 186 Day 7 API/header calibration. |
| 179.6 | Complete | `SPRINT_179/artifacts/day12-focused-verification.md`; `SPRINT_179/artifacts/day13-integrated-validation.md`; `SPRINT_179/artifacts/day14-closeout-and-handoff.md`. |

## Sprint 180: Package-Manager Provider Decision

**Duration:** 14 days, approximately 168 hours

**Goal:** Prove one package-manager provider path or close the provider
question with a stronger formal deferral and guard update.

### Prerequisites from Previous Sprints

- Static-first package contract is maintained.
- Sprint 171 package-manager deferral guard exists.
- Sprint 177 acceptance gate selects the preferred provider candidate or
  renewed deferral criteria.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 180.1 | Provider Feasibility Audit | Compare vcpkg, Homebrew, Conan, and pkgsrc for static-first fit, CI feasibility, recipe complexity, and user value. | 24 |
| 180.2 | Product Decision Record | Select one provider proof path or renewed deferral with exact blockers and revisit criteria. | 24 |
| 180.3 | Recipe or Deferral Artifact | Add source-controlled provider prototype material or a stronger provider deferral artifact. | 34 |
| 180.4 | Proof Script | Add local proof for install, downstream compile, version query, cleanup, and claim-safe failure behavior where feasible. | 38 |
| 180.5 | Guard and Docs Update | Update package-manager guards, README, INSTALL, maintainer guide, and package metadata non-claims. | 26 |
| 180.6 | Validation | Run package/install checks, provider proof or deferral checks, and relevant docs checks. | 22 |

### Deliverables

- One package-manager provider product decision.
- Provider proof artifact or stronger formal deferral.
- Updated package docs and guard behavior.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 180.1 | Complete | `SPRINT_180/artifacts/day2-package-surface-audit.md`; `SPRINT_180/artifacts/day3-vcpkg-feasibility.md`; `SPRINT_180/artifacts/day4-homebrew-feasibility.md`; `SPRINT_180/artifacts/day5-conan-feasibility.md`; `SPRINT_180/artifacts/day6-pkgsrc-feasibility.md`; `SPRINT_180/artifacts/day7-provider-decision-matrix.md`. |
| 180.2 | Narrowed | `SPRINT_180/artifacts/day8-product-decision-record.md`; Homebrew selected as a local proof path, not provider support. |
| 180.3 | Complete | `SPRINT_180/artifacts/day9-artifact-design.md`; `SPRINT_180/artifacts/day10-artifact-implementation.md`; Homebrew formula template and package notes. |
| 180.4 | Complete | `SPRINT_180/artifacts/day11-proof-script-design.md`; `SPRINT_180/artifacts/day12-proof-script-implementation.md`; `scripts/homebrew_local_formula_proof.sh`. |
| 180.5 | Complete | `SPRINT_180/artifacts/day13-guard-and-docs-update.md`; package-manager and static-package guards. |
| 180.6 | Residualized | `SPRINT_180/artifacts/day14-integrated-validation-and-closeout.md`; `SPRINT_180/RETROSPECTIVE.md`; residual `R186-PKG-LICENSE` for missing standalone license metadata before full Homebrew proof success. |

## Sprint 181: Selected Report Target Manifest

**Duration:** 14 days, approximately 168 hours

**Goal:** Centralize selected report target metadata so workflows, guards,
docs, and freshness checks stop duplicating target lists by hand.

### Prerequisites from Previous Sprints

- Sprint 177 evidence matrix identifies selected hosted/local report targets.
- Existing oracle, comparison, and performance freshness checks are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 181.1 | Target Inventory | Inventory selected oracle, comparison, performance, artifact, expected-row, and support-tier metadata across workflows and tests. | 26 |
| 181.2 | Manifest Schema | Design a source-controlled selected-target manifest with duplicate detection, required files, expected row counts, and support-tier fields. | 30 |
| 181.3 | Guard Refactor | Update workflow/report guard tests to read the manifest and fail clearly on missing, duplicate, stale, or unsupported rows. | 40 |
| 181.4 | Workflow Scope Checks | Keep YAML guards scoped to exact job and artifact upload blocks while using manifest-owned expectations. | 28 |
| 181.5 | Documentation Alignment | Update maintainer guide and report-index docs to explain the manifest as the selected target authority. | 20 |
| 181.6 | Validation | Run report normalizer tests, selected workflow guards, freshness checks, and Python compile checks. | 24 |

### Deliverables

- Canonical selected report target manifest.
- Manifest-driven workflow/report guards.
- Reduced target-list duplication in docs and tests.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 181.1 | Complete | `SPRINT_181/artifacts/day1-report-target-intake.md`; `SPRINT_181/artifacts/day2-report-surface-inventory.md`; `SPRINT_181/artifacts/day3-workflow-and-guard-duplication.md`. |
| 181.2 | Complete | `SPRINT_181/artifacts/day4-manifest-schema-design.md`; `SPRINT_181/artifacts/day5-manifest-prototype.md`; `SPRINT_181/artifacts/day6-parser-and-schema-checks.md`. |
| 181.3 | Complete | `SPRINT_181/artifacts/day7-normalizer-refactor-design.md`; `SPRINT_181/artifacts/day8-report-guard-refactor-batch-1.md`; `SPRINT_181/artifacts/day9-report-guard-refactor-batch-2.md`. |
| 181.4 | Complete | `SPRINT_181/artifacts/day10-workflow-scope-checks.md`. |
| 181.5 | Complete | `SPRINT_181/artifacts/day11-documentation-alignment.md`; Sprint 186 Day 6 maintainer/report calibration. |
| 181.6 | Complete | `SPRINT_181/artifacts/day12-diagnostics-and-drift-tests.md`; `SPRINT_181/artifacts/day13-integrated-validation.md`; `SPRINT_181/artifacts/day14-closeout-and-handoff.md`. |

## Sprint 182: Windows Report Freshness Decision

**Duration:** 14 days, approximately 166 hours

**Goal:** Promote one Windows-safe generated report freshness path or close
Windows report freshness as an explicit product deferral with guard coverage.

### Prerequisites from Previous Sprints

- Sprint 181 selected target manifest exists.
- Windows CMake-first CI and install/downstream validation are current.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 182.1 | Windows Report Audit | Audit report commands for shell, path, newline, executable, Python, dependency, and runtime assumptions. | 28 |
| 182.2 | Candidate Selection | Select one Windows-safe freshness target or a formal deferral path with exact blockers. | 20 |
| 182.3 | CI or Deferral Implementation | Add the Windows hosted freshness lane or implement the deferral artifact and guard. | 40 |
| 182.4 | Manifest Integration | Register the Windows status in the selected report target manifest and support-tier docs. | 24 |
| 182.5 | Documentation Alignment | Update README, INSTALL, maintainer guide, workflow comments, and report-index language. | 24 |
| 182.6 | Validation | Run feasible local workflow guards, report checks, and CMake/PowerShell syntax review. | 30 |

### Deliverables

- Windows report freshness promotion or formal deferral closure.
- Manifest and documentation aligned with Windows support tier.
- Guard coverage against accidental Windows report freshness claims.

### Total Estimate

166 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 182.1 | Complete | `SPRINT_182/artifacts/day1-windows-freshness-scope-intake.md`; `SPRINT_182/artifacts/day2-windows-workflow-and-toolchain-audit.md`; `SPRINT_182/artifacts/day3-report-command-compatibility-audit.md`; `SPRINT_182/artifacts/day4-artifact-and-data-semantics-audit.md`. |
| 182.2 | Deferred | `SPRINT_182/artifacts/day5-candidate-decision-matrix.md`; `SPRINT_182/artifacts/day6-decision-record-design.md`; `SPRINT_182/windows-report-freshness-deferral-decision.md`; residual `R186-WIN-REPORT-FRESHNESS`. |
| 182.3 | Deferred | `SPRINT_182/artifacts/day7-implementation-batch-1.md`; `SPRINT_182/artifacts/day8-implementation-batch-2.md`; Windows report freshness deferral artifact and guard behavior. |
| 182.4 | Complete | `SPRINT_182/artifacts/day9-manifest-and-support-tier-alignment.md`; selected target manifest and support-tier docs. |
| 182.5 | Complete | `SPRINT_182/artifacts/day10-documentation-alignment.md`; `SPRINT_182/RETROSPECTIVE.md`; Sprint 186 Day 6 maintainer/report calibration. |
| 182.6 | Residualized | `SPRINT_182/artifacts/day11-guard-and-failure-diagnostics-hardening.md`; `SPRINT_182/artifacts/day12-validation-sweep.md`; `SPRINT_182/artifacts/day13-decision-reconciliation.md`; residual `R186-WIN-PWSH` because local `pwsh` is unavailable. |

## Sprint 183: Additional Bounded External Comparison Family

**Duration:** 14 days, approximately 168 hours

**Goal:** Add one fully maintained external comparison family with fixtures,
metrics, report freshness, selected-target metadata, and scoped claims.

### Prerequisites from Previous Sprints

- Sprint 181 selected target manifest is available.
- Existing external comparison runner and selected QR, partial-SVD, and LU
  comparison families are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 183.1 | Family Selection | Select the next comparison family by claim risk, user value, fixture stability, and comparator availability. | 22 |
| 183.2 | Fixture and Metric Contract | Define source-controlled fixtures, expected rows, tolerances, metrics, skip/defer behavior, and non-parity wording. | 30 |
| 183.3 | Harness Extension | Extend comparison runner logic and tests for the selected family. | 38 |
| 183.4 | Report Integration | Generate, index, freshness-check, and manifest-register the selected comparison report. | 30 |
| 183.5 | Documentation Alignment | Update README, solver-selection, maintainer guide, corpus/report docs, and claim boundaries. | 22 |
| 183.6 | Validation | Run comparison generation, selected freshness checks, script tests, and relevant C tests. | 26 |

### Deliverables

- One additional bounded external comparison family.
- Manifest-backed selected comparison report.
- Claim-safe documentation and validation evidence.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 183.1 | Complete | `SPRINT_183/artifacts/day1-comparison-family-intake.md`; `SPRINT_183/artifacts/day2-existing-comparison-surface-audit.md`; `SPRINT_183/artifacts/day3-candidate-family-inventory.md`; `SPRINT_183/artifacts/day4-family-selection.md`. |
| 183.2 | Complete | `SPRINT_183/artifacts/day5-fixture-and-metric-contract.md`; `SPRINT_183/artifacts/day6-helper-and-fixture-implementation.md`. |
| 183.3 | Complete | `SPRINT_183/artifacts/day7-runner-extension-design.md`; `SPRINT_183/artifacts/day8-runner-implementation.md`. |
| 183.4 | Complete | `SPRINT_183/artifacts/day9-report-integration.md`; `SPRINT_183/artifacts/day10-freshness-gate-and-workflow-guard.md`; selected target manifest. |
| 183.5 | Complete | `SPRINT_183/artifacts/day11-documentation-alignment.md`; `SPRINT_183/artifacts/day13-claim-review-and-hardening.md`; Sprint 186 Day 6-7 calibration. |
| 183.6 | Complete | `SPRINT_183/artifacts/day12-integrated-validation.md`; `SPRINT_183/artifacts/day14-closeout-and-handoff.md`; `SPRINT_183/RETROSPECTIVE.md`. |

## Sprint 184: Public Header Coherence Batch 3

**Duration:** 14 days, approximately 168 hours

**Goal:** Normalize one more high-impact public header family without
declaration drift, improving API usability and generated documentation input.

### Prerequisites from Previous Sprints

- Prior declaration-preserving header cleanup batches are available as
  examples.
- Sprint 177 selected the next header family by adoption risk.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 184.1 | Header Family Selection | Confirm the selected family, likely QR/SVD or LDLT, and capture declaration baseline. | 18 |
| 184.2 | Contract Cleanup | Normalize lifecycle, ownership, error-code, tolerance, workspace, option/result, and cancellation wording. | 42 |
| 184.3 | Declaration Organization | Reorder declarations into coherent sections only if declaration-preserving guardrails are in place. | 30 |
| 184.4 | Example and Docs Alignment | Update examples, tutorial, solver-selection, cookbook, or API reference for the cleaned family. | 28 |
| 184.5 | Mechanical Guard | Add or extend declaration checksum, docs coverage, or unsupported-claim guard for the selected family. | 26 |
| 184.6 | Validation | Run `make format && make lint && make test`, docs checks, and declaration guard checks if headers changed. | 24 |

### Deliverables

- Cleaned selected public header family.
- Aligned user docs and generated API inputs.
- Declaration-preserving or documentation coverage guard.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 184.1 | Complete | `SPRINT_184/artifacts/day1-sprint-intake.md`; `SPRINT_184/artifacts/day2-declaration-baseline.md`; `SPRINT_184/artifacts/day3-family-selection-and-contract-map.md`. |
| 184.2 | Complete | `SPRINT_184/artifacts/day4-core-contract-cleanup.md`; `SPRINT_184/artifacts/day5-advanced-contract-cleanup.md`. |
| 184.3 | Complete | `SPRINT_184/artifacts/day6-organization-guardrail-design.md`; `SPRINT_184/artifacts/day7-coherent-header-sections.md`. |
| 184.4 | Complete | `SPRINT_184/artifacts/day8-documentation-alignment-map.md`; `SPRINT_184/artifacts/day9-example-contract-alignment.md`; `SPRINT_184/artifacts/day10-reference-documentation-alignment.md`. |
| 184.5 | Complete | `SPRINT_184/artifacts/day11-mechanical-guard-implementation.md`; `scripts/check_qr_header_docs_guard.sh`. |
| 184.6 | Complete | `SPRINT_184/artifacts/day12-focused-validation-pass.md`; `SPRINT_184/artifacts/day13-full-validation-and-final-cleanup.md`; `SPRINT_184/RETROSPECTIVE.md`; Sprint 186 Day 7 QR guard validation. |

## Sprint 185: Large Test and Solver Review-Surface Reduction

**Duration:** 14 days, approximately 168 hours

**Goal:** Reduce one large review surface by extracting helpers or proof-owner
files while preserving behavior and build registration.

### Prerequisites from Previous Sprints

- Sprint 177 file-size inventory identifies candidate clusters.
- Build source-list drift checks are available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 185.1 | Cluster Selection | Select one large test/source cluster with high review cost and low refactor risk. | 20 |
| 185.2 | Extraction Design | Define helper boundaries, fixture ownership, internal declarations, and no-behavior-change validation. | 28 |
| 185.3 | Mechanical Extraction | Extract helpers, fixtures, or focused proof-owner files and update build registration. | 44 |
| 185.4 | Drift Guard Update | Update Make/CMake metadata checks or add a cluster-specific registration guard if needed. | 24 |
| 185.5 | Maintenance Note | Document invariants and future contribution guidance for the selected cluster. | 20 |
| 185.6 | Validation | Run full C quality gates, affected tests, source-list checks, and `git diff --check`. | 32 |

### Deliverables

- One reduced review surface or extracted helper cluster.
- Updated build/test registration.
- Maintenance note and validation evidence.

### Total Estimate

168 hours

### Closeout Status

| Item # | Final status | Evidence |
| --- | --- | --- |
| 185.1 | Complete | `SPRINT_185/artifacts/day1-review-surface-intake.md`; `SPRINT_185/artifacts/day2-candidate-cluster-baseline.md`; `SPRINT_185/artifacts/day3-selected-cluster-decision.md`. |
| 185.2 | Complete | `SPRINT_185/artifacts/day4-helper-boundary-design.md`; `SPRINT_185/artifacts/day5-registration-guardrail-design.md`. |
| 185.3 | Complete | `SPRINT_185/artifacts/day6-initial-helper-extraction.md`; `SPRINT_185/artifacts/day7-fixture-setup-extraction.md`; `SPRINT_185/artifacts/day8-proof-owner-cleanup.md`; PR #205 review-fix commit `a64c1bc0`. |
| 185.4 | Complete | `SPRINT_185/artifacts/day9-drift-guard-update.md`; `scripts/check_ldlt_csc_helper_guard.sh`. |
| 185.5 | Complete | `SPRINT_185/artifacts/day10-maintenance-invariants.md`; `SPRINT_185/artifacts/day11-contributor-guidance-alignment.md`; `docs/maintainer_guide.md`. |
| 185.6 | Complete | `SPRINT_185/artifacts/day12-focused-cluster-validation.md`; `SPRINT_185/artifacts/day13-full-quality-gate.md`; `SPRINT_185/RETROSPECTIVE.md`; PR #205 review-fix validation. |

## Sprint 186: Epic 16 Final Validation, Claim Calibration & Closeout

**Duration:** 14 days, approximately 168 hours

**Goal:** Reconcile all Epic 16 deliverables, recalibrate public claims, run
final validation, and publish the Epic 16 retrospective and residual queue.

### Prerequisites from Previous Sprints

- Sprints 177-185 deliverables are complete or explicitly residualized.
- All selected product decisions, report metadata, package/provider status,
  generated API status, header cleanup, and failure-path proof records exist.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | ---: |
| 186.1 | Evidence Reconciliation | Reconcile the evidence/status matrix with completed sprint artifacts and validation records. | 26 |
| 186.2 | Claim Recalibration | Update README, INSTALL, maintainer guide, report docs, package docs, and generated API docs to match earned claims and retained non-claims. | 34 |
| 186.3 | Project Plan Status | Mark Epic 16 sprint items complete, narrowed, deferred, or residualized with evidence links. | 24 |
| 186.4 | Integrated Validation | Run required quality gates, package checks, report checks, docs checks, workflow guards, and generated API/package/provider checks. | 34 |
| 186.5 | Epic Retrospective | Create `EPIC_16_RETROSPECTIVE.md` with outcomes, evidence, non-claims, residuals, and state-of-the-art assessment. | 28 |
| 186.6 | Next-Epic Handoff | Publish a prioritized residual queue with exact closure targets and long-horizon deferrals. | 22 |

### Deliverables

- Final claim-recalibrated documentation.
- Integrated validation record.
- Epic 16 retrospective and prioritized residual queue.

### Total Estimate

168 hours

### Closeout Status

| Item # | Current status | Evidence |
| --- | --- | --- |
| 186.1 | Complete | `SPRINT_186/artifacts/day1-final-closeout-intake.md`; `SPRINT_186/artifacts/day2-evidence-matrix-baseline.md`; `SPRINT_186/artifacts/day3-reconciled-evidence-matrix.md`. |
| 186.2 | Complete | `SPRINT_186/artifacts/day4-public-claim-inventory.md`; `SPRINT_186/artifacts/day5-user-facing-claim-calibration.md`; `SPRINT_186/artifacts/day6-maintainer-report-claim-calibration.md`; `SPRINT_186/artifacts/day7-api-header-claim-calibration.md`. |
| 186.3 | Complete | `SPRINT_186/artifacts/day8-project-plan-status-update.md`; project-plan closeout status tables. |
| 186.4 | Complete | `SPRINT_186/artifacts/day9-integrated-validation-plan.md`; `SPRINT_186/artifacts/day10-focused-integrated-validation.md`; `SPRINT_186/artifacts/day11-full-repository-quality-gate.md`. |
| 186.5 | Complete | `SPRINT_186/artifacts/day12-retrospective-draft.md`; `docs/planning/EPIC_16/EPIC_16_RETROSPECTIVE.md`. |
| 186.6 | Complete | `SPRINT_186/artifacts/day13-residual-queue-handoff.md`; `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md`. |

### Closeout Residual Queue

| Residual ID | Source | Closure target |
| --- | --- | --- |
| R186-PKG-LICENSE | Sprint 180 | Add approved standalone license metadata or decide an alternate formula license strategy before claiming full Homebrew proof success. |
| R186-WIN-PWSH | Sprint 182 | Run PowerShell parse/workflow checks in an environment with `pwsh`, or document hosted-only validation ownership. |
| R186-WIN-REPORT-FRESHNESS | Sprint 182 | Promote one Windows-safe selected freshness lane or keep the formal deferral with updated blockers. |
| R186-HOSTED-API | Sprint 179 | Revisit hosted or retained generated API HTML publication only with explicit product value and guards. |
| R186-BROAD-COMPARISON | Sprint 183 | Add future comparison evidence one bounded family at a time. |
| R186-REVIEW-SURFACE-NEXT | Sprint 185 | Select exactly one future large review surface before further extraction. |

The prioritized residual handoff with owner surfaces, expected evidence,
validation commands, and deferral horizons is published in
`docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md`.

## Epic 16 Success Criteria

- One additional subsystem has deterministic allocation-failure proof with
  cleanup invariants and scoped public wording.
- Generated API HTML has a clear enforced product status: hosted, artifact
  retained, committed, or local-only.
- One package-manager provider path is proven or formally deferred with
  stronger blocker evidence and matching guards.
- Selected report target metadata is centralized and used by workflow/report
  guards.
- Windows report freshness is either promoted for one selected family or
  explicitly closed as a guarded deferral.
- One additional external comparison family is fully maintained, indexed, and
  claim-bounded.
- One more public header family is cleaned without declaration drift.
- One large source/test review surface is reduced or documented with improved
  ownership boundaries.
- Public documentation continues to reject unqualified state-of-the-art,
  broad external parity, portable performance, package-manager, shared-library,
  dynamic ABI, runtime-loader, broad Windows, and broad generated-report
  claims unless evidence exists.
