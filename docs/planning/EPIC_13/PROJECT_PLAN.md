# Project Plan: Platform Parity, Corpus Expansion & Competitive Evidence -- Sprints 147-156 (Epic 13)

Epic 13 starts from the completed Epic 12 closeout. The project now has strong
evidence discipline, bounded QR and partial-SVD closures, static-first package
proof, platform support tiers, and a final residual queue. It still should not
claim broad state-of-the-art sparse linear algebra status.

Epic 13 focuses on closing the next highest-value gaps completely:

- Windows staged test portability and install-validation parity decisions;
- broader maintained QR and partial-SVD corpus families;
- generated report freshness for selected claim-bearing families;
- shared-library ABI productization decision;
- first direct external comparison harness;
- tutorial/header adoption completion;
- final claim recalibration.

This plan is based on:

- `reviews/review-codex-2026-08-09.md`
- `reviews/todo-codex-2026-08-09.md`
- `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md`
- `docs/planning/EPIC_12/SPRINT_146/artifacts/day11-published-residual-queue.md`
- prior Epic retrospectives and carry-forward residuals

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 147: Epic 13 Baseline, Claim Targets & Evidence Gates

**Duration:** 14 days (~166 hours)
**Goal:** Freeze the post-Epic-12 baseline, select Epic 13 closure targets,
and define evidence gates for platform, corpus, report, ABI, and comparison
work.

### Prerequisites from Previous Sprints

- Epic 12 closeout merged
- Epic 12 retrospective and Sprint 146 residual queue available
- current `master` CI baseline available for Linux, macOS, and Windows

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Baseline Inventory | Capture current source/test sizes, Windows CTest count, package proofs, corpus rows, report families, and support tiers. | 20 |
| 2 | Residual Selection | Convert Epic 12 residuals R1-R14 into selected Epic 13 gaps, explicit non-goals, and duplicate fences. | 24 |
| 3 | Claim Target Register | Define which narrow claims Epic 13 will attempt to earn and which state-of-the-art claims remain rejected. | 22 |
| 4 | Evidence Gate Templates | Define templates for Windows parity, corpus-family proof, generated freshness, ABI product decision, and external comparison. | 30 |
| 5 | Quality Surface Map | Map required validation for code, headers, scripts, docs, Make/CMake, package, CI, corpus, and generated reports. | 20 |
| 6 | Public Claim Freeze | Re-audit README, INSTALL, benchmark docs, maintainer guide, solver-selection docs, and headers for unsupported widened claims. | 22 |
| 7 | Sprint Closeout | Publish Sprint 147 artifacts, working notes, residual handoff, and Sprint 148 Windows prerequisites. | 28 |

### Deliverables

- Epic 13 selected-gap register
- claim target and non-goal register
- evidence gate templates
- quality surface map
- public claim freeze audit
- Sprint 148 Windows portability handoff

**Total estimate:** ~166 hours

---

## Sprint 148: Windows Staged Test Portability Closure

**Duration:** 14 days (~168 hours)
**Goal:** Promote or replace the staged Windows-excluded test surfaces with
reviewed Windows-compatible coverage.

### Prerequisites from Previous Sprints

- Sprint 147 Windows evidence gate complete
- current Windows CMake reviewed subset and staged blockers inventoried

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Staged Test Audit | Audit `test_threads`, `test_sprint4_integration`, and `test_fuzz` for pthread, POSIX temp-file, and CMake blockers. | 22 |
| 2 | Portability Design | Decide whether to port tests directly, add Windows-native helper shims, or split platform-specific proof owners. | 24 |
| 3 | Thread Test Port | Implement Windows-compatible thread lifecycle coverage or an equivalent reviewed Windows proof path. | 30 |
| 4 | Fuzz/Property Port | Replace POSIX temp-file assumptions or split the fuzz property lane so Windows can run the promoted subset. | 28 |
| 5 | CMake/CI Promotion | Update CMake gating, Windows expected-count policy, workflow comments, and support-tier report rows. | 28 |
| 6 | Documentation Alignment | Update README, INSTALL, maintainer guide, and residual/non-claim wording for the promoted or rejected paths. | 14 |
| 7 | Validation | Run local feasible checks, CMake test enumeration, affected tests, and full quality gates for `.c`/`.h` changes. | 22 |

### Deliverables

- Windows staged test portability closure
- updated CMake and Windows CI registration policy
- updated platform support wording
- validation evidence and residual list
- Sprint 149 install-parity handoff

**Total estimate:** ~168 hours

---

## Sprint 149: Windows Install-Validation Parity Decision

**Duration:** 14 days (~166 hours)
**Goal:** Decide and implement the reviewed Windows install-validation support
tier without confusing it with Unix Makefile or `pkg-config` parity.

### Prerequisites from Previous Sprints

- Sprint 148 Windows CMake subset updated or residualized
- current package tests and Windows supplemental install/downstream lane
  available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Windows Package Audit | Compare Windows supplemental CMake install/downstream proof with Linux/macOS reviewed package proofs. | 22 |
| 2 | Promotion Criteria | Define exact criteria for reviewed Windows install-validation parity or explicit rejection. | 18 |
| 3 | Workflow Implementation | Promote, rename, or split Windows install/downstream jobs according to the product decision. | 28 |
| 4 | Package Metadata Checks | Strengthen Windows CMake package, static `.lib`, header, version, and unsupported shared-artifact checks. | 26 |
| 5 | Downstream Consumer Proof | Add maintained Windows downstream consumer checks for exact-version and mismatch-version behavior. | 28 |
| 6 | Documentation Alignment | Update INSTALL, README, maintainer guide, report-family rows, and non-claims. | 20 |
| 7 | Validation And Closeout | Run CMake install/downstream proof, workflow syntax review, and full quality gates if code changed. | 24 |

### Deliverables

- Windows install-validation product decision
- promoted or explicitly rejected reviewed Windows package lane
- updated support-tier docs and report rows
- downstream consumer proof
- Sprint 150 QR corpus handoff

**Total estimate:** ~166 hours

---

## Sprint 150: QR Maintained Corpus Family Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Close a broader but still bounded QR corpus family beyond the single
Sprint 139 fixture.

### Prerequisites from Previous Sprints

- Sprint 147 corpus evidence gate complete
- Sprint 139 QR fixture-local closure available
- corpus/report schema and oracle commands available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | QR Family Selection | Select two or three QR fixture families for complete closure: rank-deficient rectangular, underdetermined minimum-norm, and reorder/COLAMD paths. | 20 |
| 2 | Fixture Metadata Batch | Add source-controlled corpus fixture rows, generator rows, expected rows, claim scopes, tolerances, and non-claims. | 30 |
| 3 | Oracle Semantics | Define residual, rank, nullspace, minimum-norm, and subspace-safe comparisons without raw-basis identity claims. | 24 |
| 4 | Proof-Owner Tests | Add focused QR corpus proof tests instead of expanding the largest monolithic QR test file. | 34 |
| 5 | Report Integration | Extend local oracle/report generation and normalized report-index handling for the selected QR family. | 22 |
| 6 | Documentation Alignment | Update corpus docs, solver-selection docs, README, tutorial/cookbook references, and maintainer guide boundaries. | 16 |
| 7 | Validation | Run corpus schema, focused QR tests, oracle generation, report checks, and full C quality gates if code/header changed. | 22 |

### Deliverables

- broader maintained QR corpus family
- focused QR proof owners
- QR oracle/report rows
- updated QR claim boundaries
- Sprint 151 partial-SVD handoff

**Total estimate:** ~168 hours

---

## Sprint 151: Partial-SVD Maintained Corpus Family Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Close a broader but still bounded partial-SVD corpus family beyond
the single Sprint 140 fixture.

### Prerequisites from Previous Sprints

- Sprint 150 corpus/report lessons available
- Sprint 140 partial-SVD fixture-local closure available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Partial-SVD Family Selection | Select fixture families for repeated spectra, rank-deficient rectangular matrices, sparse low-rank output, and convergence/fail-closed behavior. | 22 |
| 2 | Comparison Contract | Define singular-value, projector, vector residual, ordering, tolerance, sparse-output, and convergence semantics. | 28 |
| 3 | Fixture Metadata Batch | Add corpus rows, generator rows, expected rows, claim scopes, support tiers, and non-claims. | 28 |
| 4 | Proof-Owner Tests | Add focused partial-SVD corpus tests and helper cleanup without broad raw-vector claims. | 34 |
| 5 | Report Integration | Extend oracle/report generation and normalization for the selected partial-SVD families. | 20 |
| 6 | Documentation Alignment | Update SVD docs, solver-selection docs, cookbook/tutorial references, and maintainer guidance. | 16 |
| 7 | Validation | Run corpus schema, focused SVD tests, oracle generation, report checks, and full quality gates if code/header changed. | 20 |

### Deliverables

- broader maintained partial-SVD corpus family
- subspace-safe comparison contract
- focused proof owners
- oracle/report rows
- updated SVD claim boundaries
- Sprint 152 generated-report handoff

**Total estimate:** ~168 hours

---

## Sprint 152: Generated Report Freshness Publication

**Duration:** 14 days (~166 hours)
**Goal:** Promote selected generated report freshness checks for claim-bearing
families without converting local generated rows into broad release proof.

### Prerequisites from Previous Sprints

- Sprint 150 QR report rows available
- Sprint 151 partial-SVD report rows available
- Sprint 141 report-index architecture available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Generated Family Selection | Select generated families needed for current claims: corpus/oracle, benchmark, sentinel, coverage, dead-code, or guardrail. | 20 |
| 2 | Freshness Policy | Define which families require `--require-generated`, which stay advisory, and which remain deferred. | 22 |
| 3 | Generator Stabilization | Stabilize commands, output paths, manifests, commit/branch fields, platform/compiler fields, and failure messages. | 28 |
| 4 | Freshness Gate Implementation | Add or strengthen report-index checks for selected generated families. | 30 |
| 5 | CI/Artifact Policy | Decide whether any generated freshness gates run in hosted CI and how artifacts are uploaded or ignored. | 24 |
| 6 | Documentation Alignment | Update benchmark docs, corpus docs, maintainer guide, and report schema documentation. | 18 |
| 7 | Validation And Closeout | Regenerate selected reports, run normalization/freshness checks, and record residual generated families. | 24 |

### Deliverables

- selected generated freshness gates
- stabilized report generation commands
- report artifact policy
- updated report docs
- Sprint 153 ABI handoff

**Total estimate:** ~166 hours

---

## Sprint 153: Shared-Library ABI Product Decision

**Duration:** 14 days (~168 hours)
**Goal:** Make a product-level shared-library ABI decision and either
implement the first supported shared surface or publish a stronger deferral
with exact blockers.

### Prerequisites from Previous Sprints

- Sprint 149 package parity decision available
- static-first package proof remains green
- public header/symbol surface inventoried

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | ABI Surface Audit | Inventory public symbols, headers, structs, macros, version metadata, static globals, allocator behavior, and callback contracts. | 24 |
| 2 | Platform Loader Audit | Review Linux `.so`, macOS `.dylib`, Windows `.dll`/import-lib requirements, symbol visibility, and loader tests. | 22 |
| 3 | Product Decision | Decide whether Sprint 153 implements shared-library support or publishes a stronger static-first deferral. | 18 |
| 4 | Selected Implementation | Implement shared build/install/export proof, or strengthen static-first rejection, diagnostics, and tests. | 42 |
| 5 | Downstream Proof | Add shared or deferral-specific downstream consumer tests for CMake, `pkg-config`, loader behavior, and unsupported artifacts. | 28 |
| 6 | Documentation Alignment | Update README, INSTALL, maintainer guide, CMake docs, package metadata comments, and non-claims. | 16 |
| 7 | Validation And Closeout | Run install/package tests, CMake install/export tests, platform-feasible checks, and full quality gates if code/header changed. | 18 |

### Deliverables

- shared-library ABI product decision
- implemented shared support or stronger deferral proof
- downstream package tests
- updated package/ABI docs and residuals
- Sprint 154 comparison handoff

**Total estimate:** ~168 hours

---

## Sprint 154: External Comparison Harness And First Narrow Study

**Duration:** 14 days (~168 hours)
**Goal:** Build the first direct external comparison harness and publish one
narrow evidence-backed comparison study without overclaiming ecosystem parity.

### Prerequisites from Previous Sprints

- Sprint 150 QR or Sprint 151 partial-SVD expanded corpus family available
- Sprint 152 generated report policy available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Comparison Target Selection | Select one narrow comparison target, likely QR or partial-SVD, with fixtures and tolerances already maintained. | 18 |
| 2 | Dependency Pinning | Identify external library/tool versions, installation paths, optional-data rules, and skip/defer semantics. | 24 |
| 3 | Output Schema | Define comparison row fields for library, version, platform, compiler, fixture, metric, tolerance, status, and caveat. | 22 |
| 4 | Harness Implementation | Implement scripts/helpers to run the project and external baseline on selected fixtures. | 36 |
| 5 | Report Integration | Add generated comparison rows to normalized report-index semantics without promoting broad parity. | 22 |
| 6 | Documentation Alignment | Update maintainer guide, benchmark/report docs, solver docs, and public non-claim wording. | 18 |
| 7 | Validation And Study Publication | Run the comparison locally, publish the narrow study artifact, and record residual comparative gaps. | 28 |

### Deliverables

- external comparison harness
- first narrow comparison study
- generated comparison report rows
- updated comparison non-claims
- Sprint 155 adoption/API handoff

**Total estimate:** ~168 hours

---

## Sprint 155: Tutorial, Header Cleanup & API Reference Coherence

**Duration:** 14 days (~166 hours)
**Goal:** Close the remaining adoption/documentation gap from Epic 12 by
aligning tutorial, public headers, and API reference surfaces.

### Prerequisites from Previous Sprints

- Sprint 145 first-use ladder available
- Sprint 146 residuals R8 and R9 available
- current earned claims from Sprints 148-154 known

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Tutorial Audit | Audit `docs/tutorial.md` against README, examples, cookbook, solver-selection, package, and support-tier claims. | 20 |
| 2 | Tutorial Alignment | Rewrite or restructure tutorial flow around build, first solve, data input, solver choice, diagnostics, install, and advanced controls. | 28 |
| 3 | Header Cleanup Selection | Select remaining high-impact public headers for documentation cleanup without signature or ABI drift. | 18 |
| 4 | Header Cleanup Batch | Clean selected headers, ownership notes, error contracts, and claim boundaries. | 32 |
| 5 | API Reference Plan | Add Doxygen/API index guidance or a generated reference publication plan. | 22 |
| 6 | Declaration Preservation | Run declaration-preservation scans and update maintainer guidance for header cleanup. | 18 |
| 7 | Validation And Closeout | Run docs checks, examples/install checks as needed, and full quality gates for header changes. | 28 |

### Deliverables

- aligned tutorial
- cleaned selected public headers
- API reference publication plan
- declaration-preservation evidence
- Sprint 156 closeout handoff

**Total estimate:** ~166 hours

---

## Sprint 156: Epic 13 Final Validation, Claim Recalibration & Closeout

**Duration:** 14 days (~166 hours)
**Goal:** Validate the final Epic 13 product state, recalibrate claims, publish
closed gaps and residuals, and decide whether any narrow state-of-practice
claim is earned.

### Prerequisites from Previous Sprints

- Sprint 147-155 deliverables complete or explicitly deferred
- final package/platform/report/adoption support tiers documented
- quality gates identified for all touched surfaces

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Final Evidence Inventory | Inventory all Epic 13 platform, corpus, report, ABI/package, comparison, adoption, and validation artifacts. | 22 |
| 2 | Full Quality Baseline | Run the strongest feasible local baseline plus package/report/corpus/comparison checks required by touched surfaces. | 28 |
| 3 | Cross-Platform/CI Reconciliation | Reconcile Linux, macOS, Windows, supplemental, staged, reviewed, local-only, and deferred evidence after final CI runs. | 22 |
| 4 | Claim And Non-Claim Audit | Rescan public and support docs for unsupported state-of-the-art, external parity, package, platform, performance, ABI, or report wording. | 24 |
| 5 | Residual Queue Publication | Publish remaining future work with owners, blockers, prerequisites, and promotion gates. | 24 |
| 6 | Epic 13 Retrospective | Write the Epic 13 retrospective with earned claims, non-claims, validation evidence, and competitive assessment. | 24 |
| 7 | Final Project Plan Reconciliation | Reconcile the Epic 13 project plan against completed sprint artifacts and prepare the next-epic handoff. | 22 |

### Deliverables

- final Epic 13 evidence inventory
- final validation package
- public claim/non-claim audit
- residual queue with promotion gates
- Epic 13 retrospective
- next-epic handoff

**Total estimate:** ~166 hours
