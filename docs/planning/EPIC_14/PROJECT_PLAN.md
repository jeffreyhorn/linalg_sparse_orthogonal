# Project Plan: Evidence Publication, Comparison Breadth & API Reference Closure -- Sprints 157-166 (Epic 14)

## Overview

Epic 14 starts from the Epic 13 residual queue and closes the highest-value
remaining gaps that can end with concrete evidence, explicit product decisions,
or source-controlled documentation. The epic deliberately avoids broad
state-of-the-art, package-manager, shared-library, dynamic ABI, and ecosystem
parity claims unless the corresponding recurring evidence is added.

Primary closure themes:

- generated API reference publication;
- hosted promotion for selected oracle and comparison report freshness;
- one bounded QR comparison family expansion;
- one bounded partial-SVD comparison publication;
- Windows package parity decision for the remaining non-claims;
- methodology-bound benchmark/report publication;
- public-header/API coherence cleanup;
- final claim recalibration and residual publication.

## Sprint 157: Epic 14 Baseline, Evidence Freeze & Claim Targets

**Duration:** 14 days, approximately 166 hours

**Goal:** Freeze the post-Epic-13 baseline and select only the complete-gap
targets Epic 14 will attempt to close.

### Prerequisites from Previous Sprints

- Epic 13 retrospective and residual queue available.
- Current `master` CI baseline available for Linux, macOS, and Windows.
- Existing generated-report, corpus, comparison, package, and API reference
  policy documented.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Baseline Inventory | Capture source, header, test, script, benchmark, example, docs, corpus, generated report, and package inventory. | 22 |
| 2 | Residual Selection | Convert Epic 13 residuals into selected Epic 14 targets, long-horizon deferrals, and explicit non-goals. | 24 |
| 3 | Evidence Contract | Define recurring evidence requirements for API docs, hosted reports, comparison rows, Windows package decisions, and performance reports. | 30 |
| 4 | Claim Target Register | Publish claim targets and non-claims for state-of-art, external parity, performance, package, Windows, ABI, and docs surfaces. | 24 |
| 5 | Quality Surface Map | Map required validation commands for docs, scripts, C/header, build-system, package, CI, and generated artifacts. | 22 |
| 6 | Risk And Handoff | Publish sprint working notes, risk register, and Sprint 158 API-doc handoff. | 22 |
| 7 | Closeout | Reconcile completed artifacts with the plan and update residuals. | 22 |

### Deliverables

- Epic 14 baseline inventory.
- Selected target and non-goal register.
- Evidence contract templates.
- Quality surface map.
- Sprint 158 handoff.

### Total Estimate

166 hours

## Sprint 158: Generated API HTML Publication Closure

**Duration:** 14 days, approximately 168 hours

**Goal:** Close the generated API reference residual with either committed
generated HTML and coverage evidence or an explicit no-commit product decision
with a recurring freshness guard.

### Prerequisites from Previous Sprints

- Sprint 157 evidence contract complete.
- Current Doxygen configuration and public-header list available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Doxygen Baseline | Run `make docs`, capture warnings, generated pages, missing pages, and generated `sparse_version.h` behavior. | 24 |
| 2 | Publication Decision | Decide whether `docs/api/html/` is source-controlled, local-only, or CI-published. | 20 |
| 3 | Coverage Check | Add or document a page-coverage check for the intended public header input set. | 28 |
| 4 | Warning Triage | Fix selected Doxygen warnings or record explicit exclusions with owner and blocker. | 30 |
| 5 | Docs Alignment | Update API reference, maintainer guide, README, tutorial, and header-doc policy wording. | 24 |
| 6 | Validation | Run docs generation, warning/page checks, and documentation whitespace validation. | 18 |
| 7 | Closeout | Publish artifacts, working notes, and Sprint 159 hosted-report handoff. | 24 |

### Deliverables

- Generated API HTML publication decision.
- Doxygen warning and page-coverage evidence.
- Updated API reference and maintainer guidance.
- Sprint 159 handoff.

### Total Estimate

168 hours

## Sprint 159: Hosted Oracle And Comparison Freshness Promotion

**Duration:** 14 days, approximately 168 hours

**Goal:** Promote selected local-only generated oracle and comparison freshness
checks into reviewed hosted evidence without broadening solver claims.

### Prerequisites from Previous Sprints

- Sprint 157 evidence contract complete.
- Sprint 158 docs publication policy available for generated-artifact wording.
- Existing QR, partial-SVD, oracle, and comparison freshness targets available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Family Selection | Select claim-bearing generated families for hosted promotion and leave advisory families explicitly local-only. | 20 |
| 2 | Runtime Budget | Measure local runtime, set hosted timeout expectations, and define artifact retention policy. | 18 |
| 3 | CI Implementation | Add hosted freshness jobs or steps for selected oracle/comparison rows. | 34 |
| 4 | Artifact Publication | Upload or summarize generated indexes, manifests, skips, and comparison reports. | 28 |
| 5 | Normalizer Semantics | Tighten stale/missing/failing row semantics for hosted selected families. | 28 |
| 6 | Docs Alignment | Update maintainer guide, corpus docs, README support-tier language, and report-family rows. | 18 |
| 7 | Validation And Closeout | Run local freshness commands, targeted script tests, and publish Sprint 160 QR handoff. | 22 |

### Deliverables

- Reviewed hosted generated freshness lane for selected rows.
- Artifact publication or deterministic hosted summaries.
- Updated report-index semantics.
- Sprint 160 QR comparison handoff.

### Total Estimate

168 hours

## Sprint 160: QR Comparison Family Closure

**Duration:** 14 days, approximately 168 hours

**Goal:** Add one bounded QR comparison family beyond the current minimum-norm
seed and publish normalized freshness evidence.

### Prerequisites from Previous Sprints

- Sprint 159 hosted generated evidence path available.
- Maintained QR corpus fixtures and external comparison harness available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Target Selection | Select one QR fixture family with stable metrics and clear non-basis-identity semantics. | 18 |
| 2 | Metric Contract | Define residual, rank, nullspace/projector, minimum-norm, tolerance, and skip/defer fields. | 24 |
| 3 | Harness Extension | Extend the external comparison runner and source-controlled contract rows for the selected family. | 34 |
| 4 | Focused Tests | Add targeted script and C proof-owner tests only for touched behavior. | 28 |
| 5 | Report Integration | Normalize generated comparison rows and require freshness for the selected family. | 24 |
| 6 | Docs Alignment | Update QR corpus docs, maintainer guide, solver-selection wording, and public non-claims. | 18 |
| 7 | Validation And Closeout | Run comparison freshness, corpus checks, targeted tests, and publish Sprint 161 handoff. | 22 |

### Deliverables

- One new bounded QR comparison family.
- Normalized QR comparison report rows.
- Updated QR comparison non-claims.
- Sprint 161 partial-SVD comparison handoff.

### Total Estimate

168 hours

## Sprint 161: Partial-SVD Comparison Publication Closure

**Duration:** 14 days, approximately 168 hours

**Goal:** Publish the first bounded partial-SVD comparison family with
subspace-safe metrics and generated freshness checks.

### Prerequisites from Previous Sprints

- Sprint 159 hosted generated evidence path available.
- Sprint 151 partial-SVD corpus families available.
- Sprint 160 comparison lessons available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Target Selection | Select one partial-SVD fixture family with stable subspace-safe behavior. | 18 |
| 2 | Metric Contract | Define singular value, projector, residual, orthogonality, convergence, fail-closed, and recovery metrics. | 28 |
| 3 | Harness Extension | Extend comparison runner, expected rows, dependency status, and skip/defer semantics. | 34 |
| 4 | Focused Tests | Add targeted comparison and normalizer tests, plus C tests if implementation behavior changes. | 28 |
| 5 | Report Integration | Emit normalized partial-SVD comparison rows and selected freshness checks. | 22 |
| 6 | Docs Alignment | Update SVD docs, corpus docs, maintainer guide, README, and non-claim language. | 18 |
| 7 | Validation And Closeout | Run selected oracle/comparison freshness gates and publish Sprint 162 handoff. | 20 |

### Deliverables

- First bounded partial-SVD comparison family.
- Subspace-safe comparison contract.
- Generated freshness proof.
- Sprint 162 Windows package handoff.

### Total Estimate

168 hours

## Sprint 162: Windows Package Parity Decision Closure

**Duration:** 14 days, approximately 166 hours

**Goal:** Decide and close the remaining Windows package parity gap for
`pkg-config` and Makefile support without confusing it with CMake install
validation.

### Prerequisites from Previous Sprints

- Current Windows CMake reviewed subset and install/downstream lane green.
- Existing Linux/macOS Make install and `pkg-config` proof available.
- Static-first package decision preserved.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Windows Package Audit | Compare Windows CMake install proof with Unix Make/`pkg-config` proofs and list remaining parity deltas. | 22 |
| 2 | Product Decision | Decide whether to promote Windows `pkg-config`, Windows Makefile parity, both, or neither. | 22 |
| 3 | Selected Proof | Implement the selected proof or stronger rejection guard with exact unsupported-surface checks. | 38 |
| 4 | CI Alignment | Update Windows workflow lane names, comments, expected counts, and support-tier documentation. | 26 |
| 5 | Downstream Consumer Evidence | Add or update Windows downstream checks for selected package metadata and exact-version behavior. | 24 |
| 6 | Docs Alignment | Update INSTALL, README, maintainer guide, and package metadata comments. | 16 |
| 7 | Validation And Closeout | Run affected install/CMake/package checks and publish Sprint 163 handoff. | 18 |

### Deliverables

- Windows package parity product decision.
- Implemented proof or stronger retained non-claim.
- Updated support-tier docs and CI comments.
- Sprint 163 performance handoff.

### Total Estimate

166 hours

## Sprint 163: Methodology-Bound Performance Publication

**Duration:** 14 days, approximately 168 hours

**Goal:** Publish a methodology-bound performance/report artifact for selected
canonical benchmark and sentinel rows while preserving non-superiority claims.

### Prerequisites from Previous Sprints

- Existing benchmark governance and sentinel semantics available.
- Hosted/report publication lessons from Sprint 159 available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Surface Selection | Select canonical benchmark and sentinel rows for methodology-bound publication. | 20 |
| 2 | Methodology Contract | Define platform, compiler, build, thread, fixture, repeat, variance, threshold, and caveat fields. | 28 |
| 3 | Report Enhancements | Update benchmark/report scripts to emit any missing methodology fields. | 34 |
| 4 | Gate Classification | Separate hard timing gates, threshold-free reports, local-only rows, supplemental rows, and advisory rows. | 24 |
| 5 | Docs Alignment | Update benchmark README, maintainer guide, README summary, and report-index schema wording. | 22 |
| 6 | Validation | Run selected benchmark/report/sentinel commands and targeted script tests. | 20 |
| 7 | Closeout | Publish artifacts, residuals, and Sprint 164 API-header handoff. | 20 |

### Deliverables

- Methodology-bound performance report.
- Updated benchmark/report schemas or docs.
- Clear performance non-superiority boundaries.
- Sprint 164 handoff.

### Total Estimate

168 hours

## Sprint 164: Public Header And API Coherence Batch

**Duration:** 14 days, approximately 166 hours

**Goal:** Complete a declaration-preserving public-header cleanup batch and
keep the API reference, tutorial, cookbook, and solver-selection docs coherent.

### Prerequisites from Previous Sprints

- Sprint 158 generated API decision complete.
- Sprint 157 quality surface map available.
- Current public-header cleanup residuals selected.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Header Selection | Select highest-impact public headers for cleanup based on usability and claim risk. | 18 |
| 2 | Declaration Baseline | Capture normalized public declarations before editing. | 16 |
| 3 | Comment Cleanup | Improve ownership, lifetime, error, output-buffer, option/result, backend, and non-claim wording. | 38 |
| 4 | Cross-Link Cleanup | Align README, tutorial, cookbook, solver-selection, API reference, and headers. | 28 |
| 5 | Declaration Preservation | Re-capture declarations and prove zero signature drift unless explicitly reviewed. | 20 |
| 6 | Generated Reference Check | Apply Sprint 158 API docs policy to the changed header batch. | 20 |
| 7 | Validation And Closeout | Run required quality gates and publish Sprint 165 package/API handoff. | 26 |

### Deliverables

- Cleaned public header batch.
- Declaration-preservation evidence.
- Updated API reference cross-links.
- Sprint 165 handoff.

### Total Estimate

166 hours

## Sprint 165: Static-First Package Boundary Hardening

**Duration:** 14 days, approximately 166 hours

**Goal:** Harden the static-first package boundary so shared-library, dynamic
ABI, runtime-loader, and package-manager non-claims cannot drift.

### Prerequisites from Previous Sprints

- Sprint 162 Windows package decision complete.
- Current static-first deferral guard available.
- Public-header and API cleanup status known.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Package Metadata Audit | Inspect CMake package files, `sparse.pc`, install scripts, and CI checks for unsupported wording or metadata. | 22 |
| 2 | Static Deferral Guard | Strengthen `BUILD_SHARED_LIBS=ON` rejection and unsupported shared metadata checks where needed. | 28 |
| 3 | ABI Non-Claim Audit | Review public structs, version docs, install docs, and package metadata for accidental ABI promises. | 26 |
| 4 | Downstream Proof Refresh | Refresh installed downstream consumer proof for static archive behavior and exact-version handling. | 28 |
| 5 | Package Docs Alignment | Update README, INSTALL, maintainer guide, CMake docs, and package comments. | 22 |
| 6 | Validation | Run package install/export scripts, static deferral guard, and affected quality checks. | 20 |
| 7 | Closeout | Publish residuals for true shared-library/package-manager work and Sprint 166 handoff. | 20 |

### Deliverables

- Hardened static-first package boundary.
- Updated ABI/package non-claim audit.
- Refreshed downstream static package proof.
- Sprint 166 closeout handoff.

### Total Estimate

166 hours

## Sprint 166: Epic 14 Final Validation, Claim Recalibration & Closeout

**Duration:** 14 days, approximately 168 hours

**Goal:** Validate the final Epic 14 state, recalibrate all public claims, and
publish a closeout that makes completed work, retained non-claims, and future
residuals unambiguous.

### Prerequisites from Previous Sprints

- Sprints 157-165 completed or explicitly residualized.
- Latest local validation and hosted CI evidence available.
- Updated generated docs, report, comparison, package, and API artifacts
  available.

### Items

| Item # | Item Name | Item Description | Estimate (hours) |
| --- | --- | --- | --- |
| 1 | Final Evidence Inventory | Inventory API docs, hosted generated reports, QR and partial-SVD comparison rows, Windows package decision, performance report, header cleanup, and package boundary evidence. | 24 |
| 2 | Full Validation Baseline | Run strongest feasible local baseline plus generated docs/report/package/comparison checks required by touched surfaces. | 28 |
| 3 | Hosted CI Reconciliation | Reconcile Linux, macOS, Windows, reviewed, supplemental, hosted, local-only, and advisory evidence. | 22 |
| 4 | Claim Audit | Scan public docs for unsupported state-of-art, external parity, performance, package, Windows, shared-library, ABI, runtime-loader, and generated-report wording. | 24 |
| 5 | Project Plan Reconciliation | Mark every Epic 14 item complete, narrowed, or residualized with evidence links. | 22 |
| 6 | Epic 14 Retrospective | Write the Epic 14 retrospective with earned claims, non-claims, validation evidence, and state-of-art assessment. | 28 |
| 7 | Residual Queue | Publish owner, blocker, prerequisite, and promotion gate for every remaining gap. | 20 |

### Deliverables

- Final evidence inventory.
- Final validation record.
- Public claim/non-claim audit.
- Epic 14 retrospective.
- Final residual queue and next-epic candidates.

### Total Estimate

168 hours

## Epic 14 Success Criteria

- Generated API reference publication is no longer ambiguous.
- Selected generated oracle/comparison evidence has a reviewed hosted path.
- One bounded QR comparison family and one bounded partial-SVD comparison
  family are published with normalized freshness checks.
- Windows package parity is either implemented for selected scope or retained
  as an explicit test-backed non-claim.
- Performance reporting has methodology-bound publication without superiority
  overclaiming.
- Public headers and API docs remain declaration-preserving and coherent.
- Static-first package and ABI non-claims are hardened.
- The final state-of-art assessment maps every positive claim to recurring
  evidence and rejects unsupported broad claims.
