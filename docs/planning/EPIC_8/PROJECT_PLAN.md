# Project Plan: Architecture, Performance, Capability & Product Convergence -- Sprints 80-89 (Epic 8)

Epic 8 takes the repository from the strong but still bounded post-Epic-7 state
toward a more competitive sparse linear algebra product. The goal is not to
grow the surface randomly. It is to close the most important gaps identified in
`reviews/review-codex-2026-06-18.md`:

- linked-list-first product/storage ceiling
- builtin scalar dense-backend ceiling
- bounded capability surface
- limited external differential proof
- remaining large source / giant-test hotspots
- reviewed runtime concentration
- static-first and asymmetric platform/package maturity
- overloaded front-door usability surfaces

Each sprint remains 14 days and stays within the requested cap of 188 hours.

---

## Sprint 80: Epic 8 Baseline, Competitive Target & External Oracle Contract

**Duration:** 14 days (~166 hours)  
**Goal:** Freeze the post-Epic-7 baseline, define the actual Epic 8 competitive
target, and establish the external-oracle / performance / non-goal contract
before implementation starts.

### Prerequisites from Previous Sprints

- Epic 7 closeout complete
- Epic 7 retrospective and residual queue available

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Baseline Recheck | Reconfirm `make quality-review-full`, reviewed CMake parity, install/export proof, and the live post-Epic-7 product baseline. | 22 |
| 2 | Competitive Gap Inventory | Re-rank the strongest remaining storage, backend, capability, packaging, usability, and assurance gaps against the live tree. | 28 |
| 3 | External Oracle Contract | Decide which external libraries or reference classes Epic 8 will use for differential correctness and performance comparison. | 30 |
| 4 | Performance / Benchmark Contract | Freeze the benchmark-governance and performance-claim model for backend and runtime work. | 22 |
| 5 | Non-goal and Risk Fence | Fix the explicit Epic 8 non-goals so the work does not sprawl into fake genericity or fake platform parity. | 18 |
| 6 | Review Package | Produce the Epic 8 review, todo, and project-plan package from the frozen baseline. | 22 |
| 7 | Sprint Closeout Notes | Capture artifacts, working notes, and the Day 14 handoff for Sprint 80. | 24 |

### Deliverables

- Epic 8 baseline artifact
- explicit competitive target and non-goal fence
- external differential / performance contract for the epic
- review + todo + project-plan package

**Total estimate:** ~166 hours

---

## Sprint 81: Core Product / Storage Modernization Phase 2

**Duration:** 14 days (~182 hours)  
**Goal:** Reduce the largest remaining linked-list-first product costs by
making compressed-first workflows more central and the orthogonal linked-list
shell more clearly bounded.

### Prerequisites from Previous Sprints

- Sprint 80 competitive target and non-goal contract complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Storage / Conversion Audit | Audit the highest-cost `SparseMatrix` conversion, publication, and repeated-run seams against the live direct workflows. | 20 |
| 2 | Compressed-first Architecture Design | Define the bounded Epic 8 contract for linked-list shell containment and compressed-first construction/import paths. | 30 |
| 3 | Construction / Import Batch | Land the highest-value compressed-first construction or import seams without breaking compatibility callers. | 36 |
| 4 | Workflow Convergence Batch | Reduce one-shot versus repeated-run ambiguity on the touched direct-workflow paths. | 32 |
| 5 | Regression / Benchmark Proof | Add focused tests and benchmark follow-through proving the refined product/storage boundary. | 22 |
| 6 | Docs / Examples / Header Alignment | Reconcile the affected public references, examples, and headers with the new workflow reading. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 81 closeout. | 24 |

### Deliverables

- clearer compressed-first workflow story
- smaller linked-list-first tax on the highest-value paths
- stronger proof for the refined product/storage contract

**Total estimate:** ~182 hours

---

## Sprint 82: Dense Backend & Performance Ceiling Phase 3

**Duration:** 14 days (~184 hours)  
**Goal:** Raise the dense-kernel performance ceiling with a bounded optional
backend layer while preserving the builtin self-contained default build.

### Prerequisites from Previous Sprints

- Sprint 80 external-oracle and benchmark contract complete
- Sprint 81 product/storage work not reopening the same ownership seam

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Dense Hotspot Profiling | Re-measure where Cholesky, LDL^T, QR, and SVD spend time in dense helpers on the maintained workloads. | 18 |
| 2 | Backend ABI Design | Define the bounded dense-kernel descriptor / runtime-selection contract for builtin and optional accelerated backends. | 30 |
| 3 | External Dense-Kernel Integration | Land the first optional BLAS/LAPACK-class acceleration slice on the highest-value solver paths. | 38 |
| 4 | Solver Adoption Follow-Through | Wire the widened backend seam through the strongest direct-family consumers without destabilizing fallback behavior. | 34 |
| 5 | Benchmark / Differential Proof | Extend benchmark measurability and add correctness/fallback proof for the widened backend surface. | 20 |
| 6 | Packaging / Runtime Alignment | Reconcile build options, runtime observability, and package wording with the new backend contract. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 82 closeout. | 24 |

### Deliverables

- bounded optional accelerated dense-kernel layer
- stronger backend-aware direct-solver performance ceiling
- benchmark and differential proof for the widened backend path

**Total estimate:** ~184 hours

---

## Sprint 83: Capability Surface Modernization Phase 2

**Duration:** 14 days (~180 hours)  
**Goal:** Move beyond the current real-only and compile-time-bounded capability
surface on the highest-value lanes first.

### Prerequisites from Previous Sprints

- Sprint 80 capability priorities fixed
- Sprint 82 backend work not reopening the same ABI fence

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Capability Re-rank | Re-rank complex scalar support, mixed precision, wider-index maturity, and algorithm-surface expansion by value and risk. | 18 |
| 2 | Scalar / Index Architecture Design | Define the bounded Epic 8 contract for the next capability-breadth step. | 32 |
| 3 | Scalar-Surface Expansion Batch | Land the first real widening of the scalar contract on the highest-value public seams. | 36 |
| 4 | Index / ABI Follow-Through | Finish the strongest remaining 64-bit and package/ABI safety seams on touched paths. | 28 |
| 5 | Algorithm-Surface Widening | Expand one bounded solver-family capability lane where the new scalar/index contract truly needs it. | 22 |
| 6 | Regression / Docs / Package Alignment | Add focused proof and support-surface follow-through for the widened capability surface. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 83 closeout. | 24 |

### Deliverables

- one real capability-breadth improvement beyond Epic 7’s bounded seams
- stronger index-width maturity on touched product paths
- proof and documentation that match the wider capability claim

**Total estimate:** ~180 hours

---

## Sprint 84: Numerical Assurance & Differential Testing Phase 2

**Duration:** 14 days (~172 hours)  
**Goal:** Strengthen assurance with maintained external differential checks,
broader property testing, and better failure-path numerical proof.

### Prerequisites from Previous Sprints

- Sprint 80 external-oracle contract complete
- Sprint 83 touched capability lanes stable enough for proof expansion

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Differential-Proof Audit | Re-rank the best direct, iterative, and eigensolver lanes for maintained external differential checks. | 18 |
| 2 | Oracle / Property Design | Define the bounded architecture for external differential proof, seeded properties, and failure-path invariants. | 26 |
| 3 | Direct-Family Differential Batch | Land the highest-value external comparison harnesses for core direct solves. | 32 |
| 4 | Seeded Property Expansion | Extend deterministic property testing beyond the current bounded lifecycle lanes. | 30 |
| 5 | Failure-Path Numerical Proof | Add cancellation, error-path, and stress-fixture proof where public lifecycle guarantees are most fragile. | 22 |
| 6 | Policy / CI / Support-Surface Alignment | Reconcile proof ownership, CI claims, and support-surface wording with the widened assurance model. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 84 closeout. | 24 |

### Deliverables

- maintained external differential proof on selected core lanes
- broader seeded property coverage
- stronger numerical and lifecycle assurance

**Total estimate:** ~172 hours

---

## Sprint 85: Large-Source Maintainability Phase 5

**Duration:** 14 days (~178 hours)  
**Goal:** Reduce the highest remaining source and giant-test maintainability
costs after Epic 7’s first bounded cleanup.

### Prerequisites from Previous Sprints

- Sprint 80 hotspot map available
- Sprint 84 assurance expansion not reopening unrelated hotspot surfaces

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Hotspot Rerank | Re-rank the largest remaining implementation and proof hotspots after Sprint 78 and Sprint 84 follow-through. | 18 |
| 2 | Decomposition Design | Define the next bounded extraction plan across `src/sparse_iterative.c`, `src/sparse_chol_csc.c`, and the strongest giant tests. | 26 |
| 3 | Iterative Source Cleanup Batch | Land the highest-value decomposition / ownership cleanup inside `src/sparse_iterative.c`. | 34 |
| 4 | Direct-Family Hotspot Batch | Land the next bounded cleanup on one major direct-family source hotspot. | 32 |
| 5 | Giant-Test Architecture Batch | Reduce the densest remaining proof-owner registration / helper concentration in the strongest next giant test. | 24 |
| 6 | Proof / Docs Alignment | Reconcile touched proof-owner and maintainer surfaces with the refined hotspot boundaries. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 85 closeout. | 24 |

### Deliverables

- smaller, clearer maintainability hotspots
- reduced review and reasoning cost on touched implementation and proof surfaces
- preserved proof ownership after extraction

**Total estimate:** ~178 hours

---

## Sprint 86: Reordering Scalability & Reviewed Runtime Convergence

**Duration:** 14 days (~176 hours)  
**Goal:** Reduce the strongest reviewed runtime long pole and improve
reordering/ND scalability without weakening proof quality.

### Prerequisites from Previous Sprints

- Sprint 80 performance contract fixed
- Sprint 85 maintainability work not reopening unrelated reorder policy seams

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Reviewed Runtime Audit | Profile the current reviewed baseline and decompose the `test_reorder_nd` long pole into actionable causes. | 18 |
| 2 | Algorithm / Proof Design | Decide how much of the fix is algorithmic, fixture-organization, or reviewed-surface architecture. | 28 |
| 3 | ND Runtime Reduction Batch | Land the highest-value runtime or scalability improvement on the reorder/ND lane. | 36 |
| 4 | Proof-Surface Rebalancing | Reduce unnecessary reviewed-runtime concentration without losing the strongest correctness proof. | 26 |
| 5 | Benchmark / Comparison Follow-Through | Add bounded measurement artifacts for the touched reorder/runtime seam. | 20 |
| 6 | CI / Reviewed-Path Alignment | Reconcile the reviewed validation path and runtime expectations with the landed change. | 24 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 86 closeout. | 24 |

### Deliverables

- smaller reviewed runtime long pole
- better reordering scalability or proof organization on the ND lane
- stronger evidence for the new runtime shape

**Total estimate:** ~176 hours

---

## Sprint 87: Packaging, ABI & Cross-Platform Quality Convergence Phase 3

**Duration:** 14 days (~170 hours)  
**Goal:** Strengthen the install/export/consumer story and decide whether the
library remains permanently static-first or earns a bounded shared-library
product lane.

### Prerequisites from Previous Sprints

- Sprint 80 packaging/platform target fixed
- Sprint 84 proof-expansion work available for any widened platform claims

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Release / Package Gap Audit | Re-rank the strongest remaining package, ABI, install/export, and workflow asymmetry gaps. | 18 |
| 2 | Product-Matrix Design | Define the explicit static/shared, ABI, and downstream-consumer contract Epic 8 will support. | 30 |
| 3 | Packaging Batch | Land the highest-value shared/static or export-surface modernization required by the chosen contract. | 32 |
| 4 | Consumer-Proof Expansion | Strengthen local install/export proof and the downstream consumer story on the maintained platforms. | 24 |
| 5 | Workflow / Platform Follow-Through | Expand Windows/macOS/Linux proof only where sustained, truthful maintenance is realistic. | 22 |
| 6 | Support-Surface Alignment | Reconcile INSTALL, README, maintainer guide, and workflow wording with the widened package/platform contract. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 87 closeout. | 24 |

### Deliverables

- clearer package / ABI / consumer contract
- stronger maintained platform evidence
- smaller gap between the install story and the reviewed story

**Total estimate:** ~170 hours

---

## Sprint 88: Front-Door Usability & Workflow Simplification

**Duration:** 14 days (~168 hours)  
**Goal:** Make the library easier to adopt without weakening the truthfulness
and proof-ownership discipline built in earlier epics.

### Prerequisites from Previous Sprints

- Sprint 81 product/workflow shape clearer
- Sprint 87 package/platform contract stable enough for public-surface cleanup

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | User-Journey Audit | Re-rank the highest-friction front-door decisions across README, tutorial, examples, install, and benchmark references. | 18 |
| 2 | Workflow-Simplification Design | Define the bounded Epic 8 usability contract for adoption guidance, support-surface layering, and advanced-doc separation. | 26 |
| 3 | README / Tutorial Batch | Tighten the front door around user decisions instead of policy density. | 30 |
| 4 | Examples / Workflow Batch | Simplify examples and example references so workflow adoption is easier without losing correctness guidance. | 24 |
| 5 | Support-Surface Consolidation | Reconcile benchmark, install, and support references so each surface has a clearer audience boundary. | 22 |
| 6 | Header / API Narrative Cleanup | Reduce remaining internal-policy leakage from the highest-signal public headers. | 24 |
| 7 | Validation and Closeout | Docs-focused sanity checks plus any required full validation, artifacts, and Sprint 88 closeout. | 24 |

### Deliverables

- simpler front-door adoption path
- clearer audience split across README/tutorial/examples/benchmarks/maintainer surfaces
- more contract-focused public narrative

**Total estimate:** ~168 hours

---

## Sprint 89: Final Integration, External Comparison & Epic 8 Closeout

**Duration:** 14 days (~180 hours)  
**Goal:** Reassess the project against the Epic 8 opening review, run the full
cross-surface validation and comparison sweep, and close Epic 8 from an
evidence-based end state.

### Prerequisites from Previous Sprints

- Sprints 80-88 complete
- residual queue stable enough for final comparison and calibration

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | End-State Re-audit | Repeat the architecture, performance, capability, usability, and packaging review against the live post-Sprint-88 tree. | 18 |
| 2 | External Comparison Sweep | Compare the library against the chosen reference class on correctness, package shape, and bounded performance signals. | 28 |
| 3 | Final Cross-Surface Fix Batch | Land the highest-value last-mile reconciliations found during the end-state re-audit. | 30 |
| 4 | Full Validation & Reporting Sweep | Run the strongest reviewed baseline, install/export proof, canonical benchmark reporting, and the touched follow-on proofs. | 32 |
| 5 | Residual Queue Finalization | Write the truthful post-Epic-8 carry-forward queue and calibrate non-claims explicitly. | 20 |
| 6 | Retrospective & Handoff | Produce Sprint 89 retrospective material, Epic 8 closeout notes, and the branch handoff package. | 28 |
| 7 | Final Project Summary | Produce the Epic 8 retrospective and final project-summary surfaces from the validated end state. | 24 |

### Deliverables

- final post-Epic-8 review and comparison package
- fully validated Epic 8 close baseline
- explicit residual queue for the next planning cycle

**Total estimate:** ~180 hours
