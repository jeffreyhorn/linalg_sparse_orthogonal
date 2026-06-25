# Project Plan: Product-Model Convergence, Backend Maturity, Capability Breadth & Competitive Calibration -- Sprints 90-99 (Epic 9)

Epic 9 takes the repository from the strong but still bounded post-Epic-8
state toward a more competitive sparse linear algebra product. The goal is not
to widen the surface randomly. It is to close the most important gaps
identified in `reviews/review-codex-2026-06-25.md` and ordered in
`reviews/todo-codex-2026-06-25.md`:

- linked-list-first public/product model
- narrow dense/backend ceiling
- bounded capability surface
- partial concurrency/runtime scalability story
- history-heavy documentation and proof naming
- remaining large source / giant-test maintainability hotspots
- duplicated build/package/workflow topology
- limited external comparison depth
- intentionally asymmetric package/platform maturity

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 90: Epic 9 Baseline, State-of-the-Art Target & Product Contract

**Duration:** 14 days (~166 hours)  
**Goal:** Freeze the post-Epic-8 baseline, define the actual Epic 9
competitive target, and establish the product-model, comparison, and non-goal
contract before structural implementation work begins.

### Prerequisites from Previous Sprints

- Epic 8 closeout complete
- Epic 8 retrospective and residual queue available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Baseline Recheck | Reconfirm `make quality-review-full`, reviewed runtime concentration, install/export proof, and maintained external-comparison status from the live tree. | 20 |
| 2 | Competitive Target Freeze | Define what "state of the art" means for this repo and which claims Epic 9 is actually trying to earn. | 28 |
| 3 | Product-Model Audit | Re-rank the linked-list shell, compressed compute paths, lifecycle ownership, and user-journey contradictions against the live tree. | 30 |
| 4 | Comparison/Measurement Contract | Freeze the external correctness, runtime, and packaging comparison contract for Epic 9. | 24 |
| 5 | Non-goal and Risk Fence | Fix the explicit anti-sprawl fence for package, capability, platform, and benchmark claims. | 18 |
| 6 | Review/Todo/Plan Package | Produce the Epic 9 review, todo, and project-plan package from the frozen baseline. | 22 |
| 7 | Sprint Closeout Notes | Capture artifacts, working notes, and the Day 14 handoff for Sprint 90. | 24 |

### Deliverables

- Epic 9 baseline artifact
- explicit product target and non-goal fence
- product-model and comparison contract for the epic
- review + todo + project-plan package

**Total estimate:** ~166 hours

---

## Sprint 91: Compressed-First Product Convergence Phase 3

**Duration:** 14 days (~168 hours)  
**Goal:** Reduce the largest remaining linked-list-first product costs by
making compressed-first workflows more central and clarifying lifecycle
ownership on the highest-value direct paths.

### Prerequisites from Previous Sprints

- Sprint 90 product target and non-goal contract complete

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Remaining Shell-Cost Audit | Audit the strongest remaining linked-list-first conversion/publication costs on direct and interop workflows. | 18 |
| 2 | Compressed-First Architecture Design | Define the bounded Epic 9 contract for compressed-first construction/import/publication and shell containment. | 28 |
| 3 | Construction/Import Batch | Land the highest-value compressed-first construction or import seams without breaking compatibility callers. | 34 |
| 4 | Publication/Lifecycle Batch | Reduce shell-to-compute ambiguity on one-shot vs repeated-run direct workflows. | 32 |
| 5 | Regression/Proof Follow-Through | Add focused test and benchmark follow-through proving the refined product and lifecycle model. | 18 |
| 6 | Public Surface Alignment | Reconcile touched headers, examples, tutorial text, and maintainer surfaces with the new workflow reading. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 91 closeout. | 20 |

### Deliverables

- clearer compressed-first workflow story
- smaller linked-list-first tax on the highest-value paths
- stronger proof for the refined product/lifecycle contract

**Total estimate:** ~168 hours

---

## Sprint 92: Portable Dense Backend & Kernel Maturity Phase 4

**Duration:** 14 days (~168 hours)  
**Goal:** Raise the dense-kernel performance ceiling with a portable optional
backend lane while preserving the builtin self-contained default build.

### Prerequisites from Previous Sprints

- Sprint 90 comparison and benchmark contract complete
- Sprint 91 product/lifecycle work not reopening the same ownership seam

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Dense Hotspot Profiling | Re-measure the highest-value dense-kernel consumers across direct-family workloads. | 18 |
| 2 | Portable Backend ABI Design | Define the bounded descriptor/runtime-selection contract for builtin and portable accelerated dense backends. | 28 |
| 3 | Portable Backend Integration | Land the first BLAS/LAPACK-class portable acceleration slice on the highest-value direct paths. | 36 |
| 4 | Solver Adoption Follow-Through | Wire the widened backend seam through the strongest direct-family consumers without destabilizing fallback behavior. | 28 |
| 5 | Observability and Bench Proof | Extend benchmark measurability and add correctness/fallback proof for the widened backend surface. | 18 |
| 6 | Build/Package Alignment | Reconcile build options, runtime observability, and package wording with the new backend contract. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 92 closeout. | 22 |

### Deliverables

- bounded portable accelerated dense-kernel lane
- stronger backend-aware direct-solver performance ceiling
- benchmark and differential proof for the widened backend path

**Total estimate:** ~168 hours

---

## Sprint 93: Runtime Scalability, Threading & ND Convergence Phase 2

**Duration:** 14 days (~166 hours)  
**Goal:** Reduce the strongest reviewed runtime long pole and tighten the
product-wide threading/runtime model without weakening proof quality.

### Prerequisites from Previous Sprints

- Sprint 90 runtime/measurement contract fixed
- Sprint 92 backend work stable enough to separate dense from reorder/runtime effects

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Reviewed Runtime Audit | Profile the current reviewed baseline and decompose the remaining `test_reorder_nd` long pole into actionable causes. | 18 |
| 2 | Threading/Runtime Contract Design | Decide how much of the remaining runtime and concurrency debt is algorithmic, proof-topology, or unsupported runtime-control complexity. | 26 |
| 3 | ND Runtime Reduction Batch | Land the highest-value runtime or scalability improvement on the reorder/ND lane. | 34 |
| 4 | Runtime-Control Cleanup | Remove or bound the riskiest internal thread-local/global override seams where practical. | 24 |
| 5 | Proof-Surface Rebalancing | Reduce unnecessary reviewed-runtime concentration without losing correctness trust. | 18 |
| 6 | Runtime Evidence Follow-Through | Extend bounded runtime-comparison artifacts for the touched reorder/runtime seam. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 93 closeout. | 28 |

### Deliverables

- smaller reviewed runtime long pole
- cleaner threading/runtime interpretation
- stronger evidence for the new reorder/runtime shape

**Total estimate:** ~166 hours

---

## Sprint 94: Capability Surface Modernization Phase 3

**Duration:** 14 days (~168 hours)  
**Goal:** Move beyond the current real-only and compile-time-bounded
capability surface on the highest-value lanes first.

### Prerequisites from Previous Sprints

- Sprint 90 capability priorities fixed
- Sprint 91-93 touched ABI/runtime surfaces stable enough for capability widening

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Capability Re-rank | Re-rank complex support, mixed precision, wider index maturity, and solver-family breadth by value and risk. | 18 |
| 2 | Scalar/Index Architecture Design | Define the bounded Epic 9 contract for the next capability-breadth step. | 30 |
| 3 | Scalar-Family Widening Batch | Land the first real widening of the scalar contract on the highest-value public and implementation seams. | 34 |
| 4 | Index/ABI Maturity Batch | Finish the strongest remaining 64-bit and ABI-safety seams on touched paths. | 24 |
| 5 | Solver-Family Breadth Batch | Expand one bounded high-value solver-family capability lane where the new scalar/index contract truly needs it. | 20 |
| 6 | Proof/Docs/Package Alignment | Add focused proof and support-surface follow-through for the widened capability surface. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 94 closeout. | 24 |

### Deliverables

- one real capability-breadth improvement beyond Epic 8's bounded seams
- stronger index-width maturity on touched product paths
- proof and docs that match the wider capability claim

**Total estimate:** ~168 hours

---

## Sprint 95: Public Narrative, Docs & Workflow Coherence

**Duration:** 14 days (~164 hours)  
**Goal:** Remove sprint-era chronology from permanent product surfaces and
make the public workflow narrative smaller, clearer, and more coherent.

### Prerequisites from Previous Sprints

- Sprint 90 public narrative target fixed
- Sprint 91 and Sprint 94 workflow/capability changes stable enough to document cleanly

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Public-Surface Audit | Identify the worst sprint-history and duplicated narrative residue across README, tutorial, examples, install, and public headers. | 18 |
| 2 | Narrative Ownership Design | Define the permanent audience split and naming/style rules for adoption, support, benchmark, and maintainer surfaces. | 22 |
| 3 | README/Tutorial Cleanup Batch | Rewrite the highest-value public narrative surfaces to remove historical clutter while preserving truth. | 28 |
| 4 | Header and Example Narrative Cleanup | Remove sprint-era residue from touched public headers and example surfaces. | 24 |
| 5 | Test/Proof Naming Cleanup | Rename or regroup the highest-value sprint-named integration/proof owners into product-oriented naming. | 24 |
| 6 | Support-Surface Consolidation | Reconcile install, benchmark, and maintainer surfaces with the cleaned ownership model. | 20 |
| 7 | Validation and Closeout | Reviewed/doc validation, artifacts, and Sprint 95 closeout. | 28 |

### Deliverables

- public docs that read more like product docs and less like sprint archives
- clearer proof-owner and example naming
- a smaller, cleaner user-journey narrative

**Total estimate:** ~164 hours

---

## Sprint 96: Large-Source & Giant-Test Maintainability Phase 6

**Duration:** 14 days (~168 hours)  
**Goal:** Reduce the largest remaining mixed-role implementation and proof
hotspots after the Epic 8 cleanup and the Epic 9 narrative/package changes.

### Prerequisites from Previous Sprints

- Sprint 90 hotspot map available
- Sprint 95 naming/narrative cleanup complete enough not to churn the same owners twice

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Hotspot Rerank | Re-rank the largest remaining implementation and proof hotspots after Epic 8 and Epic 9 follow-through. | 18 |
| 2 | Source Extraction Design | Define the next bounded extraction plan across `src/sparse_ldlt_csc.c`, `src/sparse_iterative.c`, and adjacent large owners. | 24 |
| 3 | Direct-Family Source Cleanup Batch | Land the highest-value ownership cleanup in the strongest direct-family source hotspot. | 32 |
| 4 | Solver/Algorithm Source Cleanup Batch | Land the next bounded cleanup in one major iterative, QR, eigensolver, or SVD hotspot. | 28 |
| 5 | Giant-Test Architecture Batch | Reduce the densest remaining proof-owner concentration in one or two giant tests. | 22 |
| 6 | Internal Comment/Rationale Cleanup | Remove stale chronology while preserving durable algorithm rationale on touched files. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 96 closeout. | 26 |

### Deliverables

- smaller, clearer implementation and proof hotspots
- reduced review and reasoning cost on touched large owners
- better long-term maintainability for the next capability/runtime work

**Total estimate:** ~168 hours

---

## Sprint 97: Build, Packaging & Cross-Platform Product Convergence Phase 4

**Duration:** 14 days (~166 hours)  
**Goal:** Reduce build/workflow duplication, sharpen the long-term package
surface, and improve cross-platform product maturity without fake parity.

### Prerequisites from Previous Sprints

- Sprint 90 package/platform claim fence complete
- Sprint 92 backend/package work and Sprint 95 public narrative cleanup in place

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Build-Topology Audit | Identify the highest-cost duplication between Make, CMake, CI, and install/export proof surfaces. | 18 |
| 2 | Convergence Architecture Design | Define the bounded Epic 9 plan for reducing duplication without losing proof strength. | 24 |
| 3 | Source-List/Workflow Reduction Batch | Centralize or generate the highest-value duplicated topology surfaces. | 30 |
| 4 | Package-Surface Decision Batch | Decide whether the repo remains permanently static-first or earns one bounded shared-library lane. | 22 |
| 5 | Consumer/Workflow Follow-Through | Update install/export proof and workflow surfaces to match the refined package/build contract. | 22 |
| 6 | Cross-Platform Truth Calibration | Expand or refine macOS/Windows proof only where the evidence justifies it. | 20 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 97 closeout. | 30 |

### Deliverables

- lower build/workflow duplication
- sharper long-term package/product contract
- more coherent cross-platform product story without fake parity

**Total estimate:** ~166 hours

---

## Sprint 98: Assurance, External Comparison & Coverage Architecture Phase 3

**Duration:** 14 days (~168 hours)  
**Goal:** Broaden the maintained external comparison and proof architecture so
the repo can support stronger competitive claims than the bounded Epic 8 lane.

### Prerequisites from Previous Sprints

- Sprint 90 comparison contract complete
- Sprint 94 capability work and Sprint 97 maintainability extraction stable enough for proof expansion

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Comparison-Surface Rerank | Re-rank the highest-value next external correctness and performance comparison lanes. | 18 |
| 2 | Proof/Comparison Architecture Design | Define the bounded architecture for broader differential proof, runtime comparison, and proof-owner topology cleanup. | 26 |
| 3 | External Correctness Expansion Batch | Land the highest-value next maintained external comparison lane beyond the current bounded SPD path. | 30 |
| 4 | Runtime/Fill Comparison Batch | Add bounded competitive runtime/fill comparison artifacts on the most meaningful touched workloads. | 24 |
| 5 | Coverage-Topology Cleanup | Reduce fragmentation or rename the highest-value proof owners so the widened comparison story is easier to maintain. | 20 |
| 6 | CI/Support-Surface Alignment | Reconcile workflows, docs, and maintainer proof ownership with the widened assurance model. | 18 |
| 7 | Validation and Closeout | Full validation, artifacts, and Sprint 98 closeout. | 32 |

### Deliverables

- broader maintained external correctness evidence
- better comparative runtime/fill evidence
- cleaner proof architecture for future competitive calibration

**Total estimate:** ~168 hours

---

## Sprint 99: Final Integration, Competitive Calibration & Epic 9 Closeout

**Duration:** 14 days (~164 hours)  
**Goal:** Re-audit the repo against the Epic 9 target, land one final bounded
fix batch only if evidence justifies it, and close Epic 9 from a validated
baseline.

### Prerequisites from Previous Sprints

- Sprint 90 target-state and claim fence complete
- Sprints 91-98 landed and validated

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | End-State Re-audit | Re-read the original Epic 9 contradiction classes against the live tree. | 20 |
| 2 | Final Competitive Comparison Sweep | Execute the final correctness, runtime, package, and usability evidence package. | 26 |
| 3 | Final Fix Decision | Decide whether a last bounded implementation/support batch is truly necessary. | 16 |
| 4 | Final Bounded Fix Batch | Land the final implementation/support batch only if the evidence package exposes a real contradiction. | 24 |
| 5 | Residual Queue Finalization | Separate real carry-forward work from deliberate non-claims. | 18 |
| 6 | Full Validation/Reporting Sweep | Rebuild the strongest reviewed, install/export, example, and reporting baseline. | 30 |
| 7 | Sprint/Epic Closeout Package | Write the Sprint 99 retrospective, Epic 9 retrospective, and handoff package from the validated baseline. | 30 |

### Deliverables

- one evidence-backed final Epic 9 close package
- one validated final baseline
- one explicit post-Epic-9 residual queue

**Total estimate:** ~164 hours
