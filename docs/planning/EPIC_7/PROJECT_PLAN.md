# Project Plan: Product-Model Convergence, Capability Expansion & State-of-the-Art Maturity -- Sprints 70-79 (Epic 7)

Epic 7 takes the repository from the strong but still bounded post-Epic-6
state toward a more state-of-the-art sparse linear algebra library. The main
goal is not to add random new features. It is to close the remaining product,
performance, platform, documentation, proof-surface, and capability ceilings
identified in `reviews/review-codex-2026-06-15.md`.

The strongest themes are:

- core matrix/product-model convergence
- capability-surface expansion on the most limiting ceilings
- configuration modernization phase 2
- backend/performance architecture phase 2
- benchmark/performance governance phase 2
- release/platform/install maturity
- remaining source/test/dechronologization cleanup
- and final Epic 7 integration and closeout

Each sprint remains 14 days and stays within the requested cap of 178 hours.

---

## Sprint 70: Epic 7 Baseline, State-of-the-Art Target & Architecture Contract

**Duration:** 14 days (~152 hours)  
**Goal:** Freeze the post-Epic-6 baseline, define the real Epic 7
state-of-the-art target, and fix the product-model, capability, and
validation/platform non-goal fence before implementation starts.

### Prerequisites from previous Sprints

- Epic 6 closeout complete
- Epic 6 retrospective and residual queue available

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Baseline Recheck | Reconfirm `make quality-review-full`, reviewed CMake parity, and the live post-Epic-6 product/package/performance baseline. | 20 |
| 2 | Product-Model Gap Inventory | Inventory the remaining linked-list/product-model, workflow, and conversion-heavy gaps against the live codebase. | 26 |
| 3 | Capability Ceiling Audit | Re-rank the current `idx_t`, scalar-type, and algorithm-surface ceilings by product importance and implementation risk. | 24 |
| 4 | Public-Surface and Proof-Surface Audit | Map the highest-value remaining documentation, header, benchmark, and giant-test contradictions. | 22 |
| 5 | Validation and Platform Contract | Freeze the Epic 7 truthfulness, packaging, and platform contract before code changes begin. | 18 |
| 6 | Sprint Closeout Notes | Capture artifacts, working notes, and the Day 14 handoff for Sprint 70. | 42 |

### Deliverables

- Epic 7 baseline artifact
- explicit state-of-the-art target and non-goal fence
- frozen product-model / capability / validation contract for the epic

**Total estimate:** ~152 hours

---

## Sprint 71: Public Surface De-chronologization & Reference Cleanup

**Duration:** 14 days (~158 hours)  
**Goal:** Remove the strongest remaining sprint-history and policy-overload
from public docs and reference surfaces so the repo reads more like a mature
library and less like a sprint archive.

### Prerequisites from previous Sprints

- Sprint 70 architecture and non-goal contract complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Public-Surface History Audit | Re-rank the worst sprint-history and policy-density surfaces across README, INSTALL, examples, benchmarks, and public headers. | 18 |
| 2 | Front-Door and Install Cleanup | Tighten README and INSTALL around stable product and release claims only. | 24 |
| 3 | Public Header Narrative Cleanup Batch 1 | Reduce the densest non-essential chronology and ABI-history burden in the highest-signal headers. | 32 |
| 4 | Tutorial / Example / Benchmark Cross-Surface Cleanup | Reconcile the teaching/proof split without repeating planning history in user-facing docs. | 24 |
| 5 | Maintainer Guide Re-centering | Keep policy authority in `docs/maintainer_guide.md` while shrinking duplicated policy text elsewhere. | 18 |
| 6 | Regression / Truth-Surface Review | Sanity-check touched public surfaces against current tests, workflows, and benchmark contracts. | 18 |
| 7 | Validation and Closeout | Docs-only sanity checks, artifacts, and Sprint 71 closeout. | 24 |

### Deliverables

- cleaner public docs and reference surfaces
- reduced sprint-history leakage into user-facing surfaces
- clearer audience split across README/tutorial/examples/benchmarks/maintainer guide

**Total estimate:** ~158 hours

---

## Sprint 72: Core Matrix/Product Model Convergence Phase 1

**Duration:** 14 days (~172 hours)  
**Goal:** Reduce the structural cost of the linked-list-first public model by
clarifying ownership boundaries between mutable matrix construction, compressed
working formats, and explicit factor/workspace state.

### Prerequisites from previous Sprints

- Sprint 70 product-model target fixed
- Sprint 71 public-surface cleanup not reopening the architecture fence

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Product-Model Surface Audit | Audit where `SparseMatrix` remains the right owner and where it acts mainly as a compatibility shell around compressed numeric work. | 22 |
| 2 | Ownership Convergence Design | Define the first bounded Epic 7 ownership changes across mutable matrix state, compressed working state, and factor/workspace ownership. | 28 |
| 3 | Direct Workflow Hardening Batch | Land the highest-value direct-workflow reductions in copy/mutation surprise without breaking compatibility. | 34 |
| 4 | Compressed-Path Ownership Batch | Reduce the highest-value round-trip or publication friction in the strongest CSC/CSR-backed paths. | 32 |
| 5 | Public Contract and Example Adoption | Refresh the most affected headers/examples/docs to match the refined product-model story. | 18 |
| 6 | Regression Expansion | Add focused direct-workflow and ownership-boundary regression proof for the touched lanes. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 72 closeout. | 20 |

### Deliverables

- cleaner public relationship between `SparseMatrix` and compressed/factor state
- reduced mutation/copy surprise on the highest-value solver workflows
- stronger proof for the refined product-model boundary

**Total estimate:** ~172 hours

---

## Sprint 73: Configuration Surface Modernization Phase 2

**Duration:** 14 days (~164 hours)  
**Goal:** Finish the highest-value residual configuration modernization work so
advanced solver/reordering behavior is less dependent on process-global env
vars and debug-profile switches.

### Prerequisites from previous Sprints

- Sprint 70 configuration-phase-2 target fixed
- Sprint 61 typed precedence model available as the Phase 1 contract

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Residual Env-Var Inventory | Re-rank the live `SPARSE_FM_*`, debug/profile, SVD-routing, and residual compatibility controls. | 20 |
| 2 | Typed/Internal Policy Design | Decide which remaining controls become public typed options, internal typed policy, compatibility-only overrides, or retirements. | 28 |
| 3 | FM/Graph Policy Integration | Land the highest-value typed or internal policy convergence on the FM/graph lane. | 32 |
| 4 | Debug/Profile Surface Rationalization | Remove or bound the strongest developer-only env-var surfaces from the public story. | 22 |
| 5 | Docs / Maintainer / Header Updates | Align precedence, compatibility, and residual queue wording across maintained surfaces. | 18 |
| 6 | Regression Coverage | Add focused tests proving typed/env/default behavior on the touched controls. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 73 closeout. | 24 |

### Deliverables

- materially smaller residual env-var surface
- clearer typed vs internal-policy vs compatibility-only control split
- updated regression/docs support for the converged configuration story

**Total estimate:** ~164 hours

---

## Sprint 74: Capability Surface Modernization Phase 1

**Duration:** 14 days (~176 hours)  
**Goal:** Start closing the biggest capability ceilings by defining and landing
the first bounded modernization path for index-width and scalar-surface
expansion.

### Prerequisites from previous Sprints

- Sprint 70 capability target fixed
- Sprint 72 product-model work not reopening the same ownership seam

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Capability Audit | Re-rank the current 32-bit, real-only, and algorithm-surface ceilings by user value and implementation risk. | 20 |
| 2 | Index/Scalar Architecture Design | Define the bounded Epic 7 modernization contract for `idx_t`, scalar-type abstractions, and compatibility expectations. | 32 |
| 3 | Index-Width Integration Batch | Land the first end-to-end build/runtime seam for wider-index readiness on the highest-value touched paths. | 38 |
| 4 | Scalar-Surface Preparation Batch | Land the first bounded abstraction or API-local seam needed for later scalar-type broadening. | 28 |
| 5 | Docs / Packaging / Test Alignment | Reconcile public docs, install contract, and proof surfaces with the landed capability seam. | 18 |
| 6 | Regression and Safety Checks | Add correctness, overflow, and compatibility proof for the touched capability surfaces. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 74 closeout. | 20 |

### Deliverables

- first shipped capability-modernization seam
- explicit path beyond the current 32-bit real-only ceiling
- regression/docs/package support for the landed capability work

**Total estimate:** ~176 hours

---

## Sprint 75: Performance Backend Architecture Phase 2

**Duration:** 14 days (~174 hours)  
**Goal:** Deepen the backend/performance architecture beyond the first bounded
Cholesky lane while preserving the self-contained default build.

### Prerequisites from previous Sprints

- Sprint 64 backend/performance phase 1 complete
- Sprint 70 Epic 7 performance target fixed

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Hotspot Re-audit | Re-rank dense-kernel, CSC/QR/SVD, and callback-parity hotspots after Epic 6 closeout. | 18 |
| 2 | Backend/Policy Design | Define the next bounded backend layer responsibilities across dense kernels, callback parity, and optional external math integration. | 30 |
| 3 | Kernel Integration Batch 2 | Land the next backend-aware solver/kernel slice on the highest-value touched path(s). | 38 |
| 4 | Callback and Runtime Policy Follow-Through | Tighten progress/cancel parity and runtime/backend observability where the new path needs it. | 24 |
| 5 | Benchmark Proof Refresh | Extend canonical proof/report fields so the new backend-aware behavior is measurable and reviewable. | 18 |
| 6 | Regression and Fallback Proof | Add correctness and fallback tests for the widened backend-aware surface. | 22 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 75 closeout. | 24 |

### Deliverables

- deeper backend-aware performance architecture
- stronger callback/runtime parity on touched paths
- updated benchmark and regression proof for the widened backend surface

**Total estimate:** ~174 hours

---

## Sprint 76: Benchmark Governance, Profiling & Longitudinal Reporting

**Duration:** 14 days (~160 hours)  
**Goal:** Strengthen the canonical benchmark surface so it supports better
longitudinal comparison, narrower thresholding where justified, and clearer
state-of-the-art performance claims.

### Prerequisites from previous Sprints

- Sprint 65 canonical benchmark surface stable
- Sprint 75 backend/performance phase 2 not reopening benchmark taxonomy

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Governance Re-audit | Re-rank canonical, regression-sensitive, and exploratory benchmark roles after the Epic 7 backend and capability work. | 18 |
| 2 | Longitudinal Report Design | Define the bounded manifest/schema/extensions needed for cross-run and cross-branch comparison. | 24 |
| 3 | Canonical Reporting Batch | Land the next machine-readable reporting and artifact surfaces on the canonical benchmarks. | 28 |
| 4 | Threshold Policy Batch | Add narrowly justified thresholding or comparison policy where it is genuinely stable and maintainable. | 24 |
| 5 | CI / Local Workflow Alignment | Reconcile `bench-fast`, canonical reports, and any new comparison policy across maintained command/workflow surfaces. | 20 |
| 6 | Docs and Maintainer Updates | Update benchmark docs and policy surfaces to match the stronger governance model. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 76 closeout. | 28 |

### Deliverables

- stronger longitudinal canonical benchmark reporting
- narrower but more defensible performance-threshold policy where justified
- aligned CI/local/docs story for the benchmark governance surface

**Total estimate:** ~160 hours

---

## Sprint 77: Packaging, ABI & Cross-Platform Quality Convergence Phase 2

**Duration:** 14 days (~170 hours)  
**Goal:** Reduce the remaining release/install/platform asymmetries that still
limit the repo’s maturity as a broadly deployable sparse library.

### Prerequisites from previous Sprints

- Sprint 66 packaging/platform contract complete
- Sprint 74 capability work not invalidating the release surface unexpectedly

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Release-Surface Re-audit | Reassess the maintained static-first contract, ABI/version claims, and downstream-consumer story after Epic 6 closeout. | 20 |
| 2 | Platform Gap Rerank | Re-rank Windows reviewed exclusions, macOS dead-code staging, install-validation asymmetry, and release-shape pain points. | 24 |
| 3 | Packaging/Productization Batch 2 | Land the highest-value release/install improvements justified by the refreshed audit. | 34 |
| 4 | Windows/macOS Proof Follow-Through | Land the bounded reviewed or supplemental platform-confidence improvements now justified and affordable. | 28 |
| 5 | Workflow and Contract Reconciliation | Align CI job surfaces, maintained commands, and docs with the resulting platform/release contract. | 22 |
| 6 | Install/Package/Platform Regression Coverage | Add focused verification for the touched release/install/platform surfaces. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 77 closeout. | 24 |

### Deliverables

- more mature release/install/package story
- reduced platform-review asymmetry on the highest-value lanes
- updated CI/docs/regression support for the converged contract

**Total estimate:** ~170 hours

---

## Sprint 78: Large-Source Maintainability Phase 4 & Giant-Test Architecture

**Duration:** 14 days (~176 hours)  
**Goal:** Reduce the strongest remaining permanent review hotspots across both
implementation and proof surfaces while removing avoidable sprint-history debt
from touched code.

### Prerequisites from previous Sprints

- Sprint 70 hotspot and proof-surface inventory available
- Sprint 72-77 work not reopening broader subsystem redesign

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Source Hotspot Audit | Re-rank the largest remaining source files by ownership pain after Epic 6 and Epic 7 follow-through. | 16 |
| 2 | Source Decomposition Batch | Land the highest-value bounded extraction or ownership split on the strongest source hotspot(s). | 34 |
| 3 | Giant-Test Audit | Re-rank the largest remaining permanent proof files by review cost, chronology density, and platform value. | 18 |
| 4 | Giant-Test Architecture Batch | Land the highest-value helper/split/taxonomy cleanup on the strongest remaining giant-test surfaces. | 34 |
| 5 | Chronology and Comment Cleanup | Remove stale sprint-history/comment chronology from touched permanent source/test files while preserving durable algorithm explanations. | 20 |
| 6 | Docs / Proof Ownership Alignment | Reconcile touched docs and maintainer policy with the refined source/test ownership boundaries. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 78 closeout. | 36 |

### Deliverables

- smaller or more clearly owned top source/test hotspots
- less sprint-history debt in permanent code and proof surfaces
- clearer proof ownership for the touched behavior families

**Total estimate:** ~176 hours

---

## Sprint 79: Numerical Assurance Expansion, Final Integration & Epic 7 Closeout

**Duration:** 14 days (~168 hours)  
**Goal:** Close Epic 7 from a measured validated baseline after the product,
configuration, performance, capability, and maintainability work has been
reconciled across all maintained surfaces.

### Prerequisites from previous Sprints

- Sprint 71-78 implementation and public-surface work complete
- Epic 7 final residual queue narrowed to bounded closeout items

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Assurance Gap Audit | Re-rank the remaining highest-value oracle/property/platform-confidence gaps after the main Epic 7 implementation work. | 18 |
| 2 | Differential / Oracle Batch | Land the strongest bounded external-oracle, property, or lifecycle-stress proof improvements still justified by the final product story. | 28 |
| 3 | Final Cross-Surface Integration Sweep | Reconcile docs, examples, headers, benchmarks, workflows, package surfaces, and proof owners from the fully integrated Epic 7 tree. | 22 |
| 4 | Full Validation Sweep | Run the full maintained validation set, reviewed parity anchors, and the final targeted follow-ons. | 28 |
| 5 | Epic 7 Summary and Residual Finalization | Write the final Epic 7 summary package, final residual queue, and any needed project-plan closeout notes. | 18 |
| 6 | Epic 7 Retrospective and Handoff | Produce the retrospective, handoff package, and explicit post-Epic-7 carry-forward queue. | 24 |
| 7 | Closeout Buffer | Bounded final reconciliation time for any last measured/documented truthfulness adjustments revealed by the integrated sweep. | 30 |

### Deliverables

- stronger final assurance on the highest-value residual lanes
- one fully reconciled public/product/proof/package story
- validated Epic 7 closeout package and explicit post-epic residual queue

**Total estimate:** ~168 hours
