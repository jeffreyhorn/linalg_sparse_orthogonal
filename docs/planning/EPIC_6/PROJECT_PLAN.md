# Project Plan: Productization, Performance Architecture & State-of-the-Art Convergence -- Sprints 60-69 (Epic 6)

Epic 6 takes the repository from a structurally strong and fully closed
post-Epic-5 state toward a more state-of-the-art sparse linear algebra
product surface. The main goal is not to add a random new algorithm family.
It is to close the remaining usability, configuration, backend-performance,
platform, maintainability, and assurance gaps identified in
`reviews/review-codex-2026-06-08.md`.

The strongest themes are:

- direct-solver usability convergence
- typed configuration replacing process-global tuning where justified
- bounded backend/performance architecture modernization
- stronger benchmark and platform quality governance
- remaining hotspot and giant-test cleanup
- and final cross-surface product integration

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 60: Epic 6 Baseline, Productization Audit & Architecture Contract

**Duration:** 14 days (~148 hours)  
**Goal:** Freeze the post-Epic-5 baseline, define the Epic 6 state-of-the-art
target, and fix the non-goal and validation fences before implementation work
begins.

### Prerequisites from previous Sprints

- Epic 5 closeout complete
- Epic 5 retrospective and residual queue available

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Baseline Recheck | Reconfirm `make quality-review-full`, reviewed CMake parity, and the live post-Epic-5 public/product baseline. | 20 |
| 2 | Productization Gap Inventory | Inventory the remaining usability, configuration, performance, platform, and assurance gaps against the current codebase. | 26 |
| 3 | State-of-the-Art Target Definition | Decide which Epic 6 goals are real product goals versus explicit non-goals. | 28 |
| 4 | Configuration and Performance Surface Audit | Map the strongest env-var-driven and backend-sensitive surfaces for later implementation sprints. | 24 |
| 5 | Validation and Platform Contract | Freeze the Epic 6 truthfulness/validation/platform contract before code changes begin. | 20 |
| 6 | Sprint Closeout Notes | Capture artifacts, working notes, and the Day 14 handoff for Sprint 60. | 30 |

### Deliverables

- Epic 6 baseline artifact
- explicit state-of-the-art target and non-goal fence
- frozen validation/platform contract for the epic

**Total estimate:** ~148 hours

---

## Sprint 61: Configuration Surface Modernization Phase 1

**Duration:** 14 days (~160 hours)  
**Goal:** Replace the highest-value process-global environment-variable tuning
controls with typed option surfaces and explicit precedence rules.

### Prerequisites from previous Sprints

- Sprint 60 architecture and non-goal contract complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Env-Var Surface Inventory | Re-rank the live `SPARSE_ND_*`, `SPARSE_FM_*`, and related controls by value and risk. | 18 |
| 2 | Typed Options Design | Define public/internal option structs, precedence rules, and backward-compatibility behavior. | 26 |
| 3 | Reordering Option Integration | Land the first typed option surfaces for the highest-value reorder/ND controls. | 34 |
| 4 | Analysis/Postorder Integration | Move the strongest analysis-time env-var controls into typed options where justified. | 24 |
| 5 | Compatibility Layer | Keep bounded env-var compatibility or override behavior where still needed. | 18 |
| 6 | Regression and Doc Updates | Add tests/docs for the new typed configuration story. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 61 closeout. | 20 |

### Deliverables

- first typed advanced-configuration surfaces
- explicit precedence between typed options and legacy env vars
- updated regression/docs support for the new configuration model

**Total estimate:** ~160 hours

---

## Sprint 62: Direct-Solver Usability & Lifecycle Coherence

**Duration:** 14 days (~156 hours)  
**Goal:** Reduce mutable-matrix surprise and make the one-shot and explicit
lifecycle direct-solver stories more coherent.

### Prerequisites from previous Sprints

- Sprint 60 direct-usability target fixed
- Sprint 61 configuration changes not reopening the public lifecycle fence

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Direct Usability Audit | Re-audit one-shot direct APIs for mutation, cancellation, and copy-discipline pain points. | 18 |
| 2 | Lifecycle Coherence Design | Decide the bounded helpers/wrapper/state-hardening changes needed for Epic 6. | 22 |
| 3 | One-Shot Hardening Batch | Land the highest-value usability reductions on the one-shot direct paths without breaking compatibility. | 34 |
| 4 | Lifecycle/Wrapper Convergence | Tighten the relationship between explicit lifecycle and one-shot public entries where safe. | 30 |
| 5 | Example and Doc Adoption | Refresh the highest-signal direct-lifecycle examples/docs to match the refined usability story. | 18 |
| 6 | Regression Expansion | Add direct lifecycle and cancellation/mutation proof for the touched surfaces. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 62 closeout. | 16 |

### Deliverables

- reduced mutable-matrix surprise on direct solves
- clearer relationship between one-shot and explicit lifecycle direct usage
- stronger direct-lifecycle proof and adoption surfaces

**Total estimate:** ~156 hours

---

## Sprint 63: Direct-Lifecycle Uniformity & CSC/LU Follow-Through

**Duration:** 14 days (~164 hours)  
**Goal:** Reduce the remaining internal heterogeneity behind the public
direct-lifecycle model, focusing on LU and the hardest CSC seams.

### Prerequisites from previous Sprints

- Sprint 62 direct-usability boundary complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Internal Path Audit | Re-rank the remaining uneven direct-lifecycle implementation paths. | 18 |
| 2 | LU Lifecycle Follow-Through | Tighten the highest-value LU lifecycle inconsistencies that still leak through the public story. | 34 |
| 3 | CSC Repeated-Run Uniformity | Reduce the strongest remaining CSC lifecycle/dispatch asymmetries behind the public analysis/factors path. | 34 |
| 4 | Solve/Refactor Semantics Alignment | Normalize the strongest remaining result/state semantics across touched direct families. | 22 |
| 5 | Benchmark and Example Proof | Refresh factor-many and CSC proof surfaces where the internal changes matter. | 20 |
| 6 | Regression Expansion | Add focused lifecycle/CSC/LU regression coverage. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 63 closeout. | 18 |

### Deliverables

- more uniform direct repeated-run implementation under the public contract
- stronger LU/CSC lifecycle coherence
- refreshed proof surfaces for touched direct workflows

**Total estimate:** ~164 hours

---

## Sprint 64: Performance Backend Architecture Phase 1

**Duration:** 14 days (~168 hours)  
**Goal:** Introduce a bounded backend/performance abstraction for the most
important dense-kernel and supernodal hotspots while preserving the
self-contained build.

### Prerequisites from previous Sprints

- Sprint 60 performance target fixed
- Sprint 63 direct-lifecycle follow-through complete enough to freeze the hot paths

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Performance Hotspot Audit | Re-rank the dense-kernel, supernodal, and solver-parallelism hotspots by payoff and risk. | 18 |
| 2 | Backend Abstraction Design | Define the bounded backend layer for dense kernels, threading policy, and future acceleration seams. | 28 |
| 3 | Kernel Integration Batch 1 | Land the first backend-aware dense-kernel integration on the highest-value touched path(s). | 38 |
| 4 | Build and Option Surface | Add the needed build/options wiring while preserving the default self-contained path. | 20 |
| 5 | Benchmark Proof Refresh | Add focused benchmarks and output fields proving the new path without turning the sprint into a broad framework rewrite. | 20 |
| 6 | Regression and Safety Checks | Add correctness and fallback coverage for the backend-aware path. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 64 closeout. | 24 |

### Deliverables

- first backend/performance abstraction layer
- backend-aware acceleration on selected hot kernels
- proof that the default build and fallback path remain correct

**Total estimate:** ~168 hours

---

## Sprint 65: Performance Governance, Benchmark Consolidation & Solver Efficiency Follow-Through

**Duration:** 14 days (~152 hours)  
**Goal:** Turn the benchmark surface into a clearer performance-governance
surface and apply the resulting insight to the highest-value solver efficiency
follow-through.

### Prerequisites from previous Sprints

- Sprint 64 backend/performance phase 1 complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Benchmark Role Audit | Re-rank benchmark drivers into regression-sensitive, proof, and exploratory categories. | 18 |
| 2 | Output and Taxonomy Normalization | Normalize machine-readable output and benchmark documentation around the new categories. | 24 |
| 3 | Canonical Performance Surface | Define the smaller set of canonical performance baselines the repo should actually track. | 18 |
| 4 | Solver Efficiency Follow-Through | Apply targeted efficiency improvements where the benchmark/governance pass exposes the strongest payoff. | 34 |
| 5 | Local/CI-Friendly Regression Checks | Add bounded performance-regression checks or reporting where they are genuinely maintainable. | 20 |
| 6 | Docs and Example Alignment | Reconcile benchmark-facing docs/examples with the new performance-governance story. | 16 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 65 closeout. | 22 |

### Deliverables

- clearer benchmark/performance governance model
- smaller canonical performance surface
- targeted solver-efficiency follow-through on the highest-value paths

**Total estimate:** ~152 hours

---

## Sprint 66: Packaging, ABI & Platform Quality Convergence

**Duration:** 14 days (~160 hours)  
**Goal:** Tighten the packaging/release story and reduce the remaining
cross-platform quality asymmetries that most affect product maturity.

### Prerequisites from previous Sprints

- Sprint 60 quality/platform target fixed
- Sprint 65 benchmark/performance governance stable enough not to move the packaging target again

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Packaging and ABI Audit | Audit the current static-only packaging surface, versioning/ABI story, and install claims. | 20 |
| 2 | Platform Residual Recheck | Reassess macOS dead-code, Windows reviewed-wrapper gaps, Windows dead-code, and serialized dead-code topology. | 24 |
| 3 | Packaging/Productization Batch | Land the highest-value packaging/release improvements justified by the audit. | 30 |
| 4 | Dead-Code / Platform Follow-Through | Land the bounded platform-quality improvements that are now justified and affordable. | 30 |
| 5 | CI and Contract Reconciliation | Align workflows, docs, and maintained commands with the new packaging/platform contract. | 22 |
| 6 | Install/Package Regression Coverage | Add focused verification for the touched install/package/platform surfaces. | 16 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 66 closeout. | 18 |

### Deliverables

- stronger packaging/ABI/release story
- narrower platform-quality asymmetries
- updated docs/CI/regression support for the converged contract

**Total estimate:** ~160 hours

---

## Sprint 67: Large-Source Maintainability Phase 3

**Duration:** 14 days (~156 hours)  
**Goal:** Reduce the strongest remaining implementation hotspots after Epic 5,
especially where configuration, graph/reorder, and CSC residuals still compete
with orchestration in the same files.

### Prerequisites from previous Sprints

- Sprint 61 typed configuration phase 1 complete
- Sprint 66 platform/packaging contract stable enough to avoid reopening build surfaces

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Residual Hotspot Audit | Re-rank the remaining production hotspot files by ownership pain and payoff. | 18 |
| 2 | Graph/Reorder Decomposition Batch | Extract the strongest remaining graph/reorder owned seams. | 32 |
| 3 | CSC/Iterative Residual Decomposition | Extract the strongest remaining CSC or iterative owned seams still justified after Epic 5. | 32 |
| 4 | Comment/Chronology Cleanup | Remove stale sprint-history commentary from touched permanent implementation/header files while preserving durable explanations. | 18 |
| 5 | Build/Regression Alignment | Update build surfaces and tests for the new ownership boundaries. | 18 |
| 6 | Validation and Closeout | Full validation, artifacts, and Sprint 67 closeout. | 38 |

### Deliverables

- smaller and more clearly owned remaining hotspot files
- less sprint-local chronology in permanent implementation surfaces
- preserved behavior-level confidence through the decomposition

**Total estimate:** ~156 hours

---

## Sprint 68: Giant-Test Refactor Phase 2 & Numerical Assurance Expansion

**Duration:** 14 days (~164 hours)  
**Goal:** Continue reducing giant-test maintenance cost while adding stronger
second-layer numerical and workflow assurance on the hardest paths.

### Prerequisites from previous Sprints

- Sprint 67 maintainability phase 3 complete enough to freeze the new ownership seams

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Giant-Test Residual Audit | Re-rank the largest remaining test hotspots after Sprint 57. | 18 |
| 2 | Giant-Test Refactor Batch | Extract helpers or split the highest-value remaining test seams. | 30 |
| 3 | Differential/Oracle Coverage | Add stronger external-oracle or cross-method parity checks on the hardest numerical paths. | 28 |
| 4 | Property/Fuzz Expansion | Add bounded property/fuzz coverage where it materially improves assurance. | 24 |
| 5 | Platform-Test Follow-Through | Improve or better document the reduced-platform test coverage where justified. | 18 |
| 6 | Validation and Closeout | Full validation, artifacts, and Sprint 68 closeout. | 46 |

### Deliverables

- more maintainable remaining giant-test surfaces
- stronger second-layer numerical assurance
- clearer platform-specific test-confidence story

**Total estimate:** ~164 hours

---

## Sprint 69: Public Product Surface Finalization, Integration & Epic 6 Closeout

**Duration:** 14 days (~144 hours)  
**Goal:** Finalize the public docs/examples/reference story, run the final
cross-surface integration sweep, and close Epic 6 from a measured baseline.

### Prerequisites from previous Sprints

- Sprint 60-68 work complete enough to freeze the intended final product story

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Public Surface Audit | Audit README, tutorial, examples, benchmark docs, headers, and maintainer guide against the final Epic 6 product story. | 18 |
| 2 | Docs/Examples Productization Batch | Land the highest-value final simplification and adoption edits on the public surfaces. | 28 |
| 3 | Final Cross-Surface Compatibility Sweep | Reconcile API/docs/examples/benchmarks/tests/platform claims from the final Epic 6 state. | 22 |
| 4 | Full Validation Sweep | Run the full maintained quality gates, truthfulness anchors, and targeted follow-ons from the final branch state. | 24 |
| 5 | Epic 6 Summary and Handoff | Write the final sprint closeout, Epic 6 retrospective inputs, and handoff package. | 24 |
| 6 | Project-Level Residual Finalization | Update final residual limits and project-level summary artifacts. | 28 |

### Deliverables

- final integrated public product story
- final Epic 6 measured validation baseline
- Epic 6 closeout and handoff package

**Total estimate:** ~144 hours
