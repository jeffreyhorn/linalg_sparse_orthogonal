# Project Plan: Lifecycle Completion, Solver-Workflow Integration & Final Productization -- Sprints 50-59 (Epic 5)

Epic 5 takes the repository from a structurally strong post-Epic-4 state to a
more uniform product surface. The main goal is not to add a large new solver
family. It is to close the remaining lifecycle, usability, maintainability,
documentation, and quality-platform gaps identified in
`reviews/review-codex-2026-0531.md`.

The strongest themes are:

- explicit direct-solver lifecycle exposure
- deeper analyze/factor/refactor integration
- remaining CSC and factor-many follow-through
- iterative/eigensolver public lifecycle completion or rationalized bounding
- large-file and giant-test maintainability cleanup
- documentation and benchmark simplification
- and final quality/platform convergence

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 50: Direct-Solver Lifecycle Baseline & API Design

**Duration:** 14 days (~132 hours)  
**Goal:** Freeze the post-Epic-4 baseline and define the public direct-solver
lifecycle model that Epic 5 will implement.

### Prerequisites from previous Sprints

- Epic 4 closeout complete
- Epic 4 repeated-run public handle work available as terminology precedent
- Epic 4 maintainer-guide and README ownership boundaries in place

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Baseline Recheck | Reconfirm `make quality-review-full`, reviewed CMake parity, and the live direct-solver / analysis / benchmark / example starting state. | 20 |
| 2 | Lifecycle Surface Inventory | Inventory one-shot direct-solver APIs, `sparse_analysis_t`, `sparse_factors_t`, refactor flows, and current caller/documentation patterns. | 22 |
| 3 | Public API Design | Define the target direct-solver lifecycle model, including prepare/factor/refactor/solve/free semantics and public terminology. | 34 |
| 4 | Non-Goal and Compatibility Fence | Write explicit non-goals so Epic 5 does not collapse into a broad solver API rewrite. | 12 |
| 5 | Validation and Landing Plan | Fix the implementation/validation contract for later public API batches. | 14 |
| 6 | Sprint Closeout Notes | Capture working notes, artifacts, and the Day 14 handoff for Sprint 50. | 30 |

### Deliverables

- Sprint 50 plan and working notes
- baseline artifact for the post-Epic-4 direct-solver lifecycle state
- direct-solver lifecycle API design artifact
- explicit Epic 5 non-goal fence

**Total estimate:** ~132 hours

---

## Sprint 51: Public Direct-Solver Lifecycle API Phase 1

**Duration:** 14 days (~148 hours)  
**Goal:** Land the first public direct-solver lifecycle API batch for the main
factor-and-solve workflows while preserving one-shot compatibility wrappers.

### Prerequisites from previous Sprints

- Sprint 50 direct-solver lifecycle API design complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Public Header Surface | Add the first public lifecycle declarations and contracts to the relevant headers. | 22 |
| 2 | LU Lifecycle Integration | Wire the first direct-solver lifecycle path through LU factor / solve semantics. | 24 |
| 3 | Cholesky Lifecycle Integration | Wire the lifecycle path through the Cholesky side without breaking one-shot callers. | 24 |
| 4 | LDL^T Lifecycle Integration | Add the matching lifecycle bridge on the LDL^T side. | 26 |
| 5 | Wrapper Preservation | Route one-shot public entries through the new lifecycle path where appropriate. | 16 |
| 6 | Focused Regression Expansion | Add direct public lifecycle tests for the touched direct-solver families. | 18 |
| 7 | Validation and Closeout | Run required validation, record artifacts, and close Sprint 51. | 18 |

### Deliverables

- first public direct-solver lifecycle API batch
- compatibility-preserving one-shot wrapper routing for touched paths
- focused direct lifecycle regression coverage

**Total estimate:** ~148 hours

---

## Sprint 52: Analysis/Refactor Integration & Direct-Solver Lifecycle Phase 2

**Duration:** 14 days (~156 hours)  
**Goal:** Deepen the lifecycle model so analyze/factor/refactor is a real
first-class workflow instead of a lightly wrapped alternative path.

### Prerequisites from previous Sprints

- Sprint 51 public direct-solver lifecycle phase 1 complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Analysis Contract Audit | Audit the live `sparse_analysis_t` / `sparse_factors_t` path against the new public lifecycle model. | 18 |
| 2 | Numeric Reuse Integration | Reduce avoidable fallback to underlying one-shot symbolic work in the main analysis-driven paths. | 34 |
| 3 | Refactor Path Tightening | Strengthen refactor semantics and result reuse across the touched direct-solver families. | 28 |
| 4 | Factor-Many Bench Proof | Extend or refresh factor-many benchmarks so the performance story is measurable rather than implied. | 18 |
| 5 | Example / Doc Adoption | Add or update direct-solver repeated-run usage docs/examples as needed for the new lifecycle story. | 20 |
| 6 | Regression Coverage Expansion | Add public lifecycle and factor-many regression cases for the deeper integration. | 20 |
| 7 | Validation and Closeout | Full validation, working notes, artifacts, and closeout. | 18 |

### Deliverables

- stronger analysis/factor/refactor integration
- measurable factor-many benchmark evidence
- updated public lifecycle docs/examples for touched direct-solver workflows

**Total estimate:** ~156 hours

---

## Sprint 53: CSC Direct-Solver Completion & Dispatch Follow-Through

**Duration:** 14 days (~144 hours)  
**Goal:** Close the most important deferred CSC direct-solver items that still
limit the direct-solver lifecycle story.

### Prerequisites from previous Sprints

- Sprint 52 analysis/refactor integration complete
- Sprint 17/19 deferred CSC follow-ons explicitly inventoried

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | LDL^T Analysis-Aware Indefinite Path | Implement `ldlt_csc_from_sparse_with_analysis` or the equivalent full-pattern indefinite preparation path. | 32 |
| 2 | Transparent LDL^T Dispatch | Revisit transparent LDL^T dispatch so CSC/native/supernodal routing is more uniform with Cholesky. | 24 |
| 3 | Indefinite Supernodal Factor-Many Proof | Validate the indefinite batched/factor-many path on the intended workloads and tighten fallbacks where needed. | 24 |
| 4 | Cholesky / LDL^T Dispatch Reconciliation | Reconcile API-local docs and behavior so the direct CSC dispatch story is easier to reason about. | 18 |
| 5 | Benchmark and Regression Updates | Add or refresh targeted CSC factor-many and indefinite-path tests/benchmarks. | 24 |
| 6 | Validation and Closeout | Full validation and sprint closeout package. | 22 |

### Deliverables

- closed or materially reduced CSC deferred direct-solver gaps
- stronger indefinite CSC factor-many story
- clearer CSC dispatch behavior and documentation

**Total estimate:** ~144 hours

---

## Sprint 54: Public Repeated-Run Solver Lifecycle Completion

**Duration:** 14 days (~152 hours)  
**Goal:** Decide and land the steady-state public repeated-run story across the
remaining iterative/eigensolver solver families.

### Prerequisites from previous Sprints

- Epic 4 public iterative/eigensolver handles available
- Sprint 50 Epic 5 lifecycle non-goal fence complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Public Solver Lifecycle Audit | Audit the remaining iterative/eigensolver public repeated-run asymmetries. | 18 |
| 2 | Solver-Surface Decision Batch | Decide whether MINRES, BiCGSTAB, and selected block workflows get public repeated-run exposure or explicit bounded exclusion. | 20 |
| 3 | Iterative Handle Expansion | Implement the chosen public repeated-run extensions for remaining iterative families, if in-scope. | 36 |
| 4 | Eigensolver Lifecycle Tightening | Reconcile any remaining eigensolver lifecycle asymmetry or documentation drift. | 18 |
| 5 | Public Reuse Bench Alignment | Update repeated-run benchmarks to match the final public solver lifecycle support set. | 18 |
| 6 | Regression and Example Adoption | Add direct tests and bounded examples/docs for the final public repeated-run story. | 24 |
| 7 | Validation and Closeout | Full validation and sprint closeout package. | 18 |

### Deliverables

- explicit steady-state public repeated-run support boundary
- public iterative/eigensolver lifecycle surface aligned to that boundary
- matching benchmark/test/docs support

**Total estimate:** ~152 hours

---

## Sprint 55: Large-Source Decomposition Phase 1

**Duration:** 14 days (~160 hours)  
**Goal:** Reduce the largest remaining solver implementation hotspots by
splitting the two most consequential translation units into cleaner ownership
seams.

### Prerequisites from previous Sprints

- Sprint 54 public repeated-run solver lifecycle support set fixed

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | `sparse_eigs.c` Seam Audit | Identify the next bounded extraction seams in the eigensolver implementation. | 16 |
| 2 | Eigensolver Decomposition Batch 1 | Extract the first owned helper/module slice from `src/sparse_eigs.c`. | 34 |
| 3 | Eigensolver Decomposition Batch 2 | Extract the second owned slice and tighten the residual orchestration layer. | 30 |
| 4 | `sparse_iterative.c` Seam Audit | Identify the next bounded extraction seams in the iterative implementation. | 14 |
| 5 | Iterative Decomposition Batch 1 | Extract the first owned helper/module slice from `src/sparse_iterative.c`. | 30 |
| 6 | Historical Comment Reduction | Remove stale sprint-history narrative from touched permanent implementation files while preserving useful algorithm commentary. | 14 |
| 7 | Validation and Closeout | Full validation and sprint closeout package. | 22 |

### Deliverables

- smaller and cleaner eigensolver/iterative implementation ownership
- reduced sprint-history narrative in touched permanent implementation files
- maintained behavior-level validation baseline

**Total estimate:** ~160 hours

---

## Sprint 56: Large-Source Decomposition Phase 2

**Duration:** 14 days (~148 hours)  
**Goal:** Continue hotspot reduction across the remaining large direct-solver
and dense-algorithm implementation files.

### Prerequisites from previous Sprints

- Sprint 55 decomposition phase 1 complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | `sparse_ldlt_csc.c` Residual Audit | Identify extraction seams after the Epic 5 CSC completion batch. | 14 |
| 2 | LDL^T CSC Decomposition | Extract bounded owned slices from `src/sparse_ldlt_csc.c`. | 30 |
| 3 | `sparse_chol_csc.c` Residual Audit | Identify the next bounded extraction seams in the Cholesky CSC path. | 14 |
| 4 | Cholesky CSC Decomposition | Extract bounded owned slices from `src/sparse_chol_csc.c`. | 30 |
| 5 | `sparse_svd.c` Maintainability Batch | Reduce the next highest-value maintainability hotspot in the SVD implementation. | 26 |
| 6 | Touched-Doc and Comment Reconciliation | Normalize touched implementation comments and any coupled header wording. | 14 |
| 7 | Validation and Closeout | Full validation and sprint closeout package. | 20 |

### Deliverables

- reduced direct-solver CSC and SVD implementation hotspot size
- clearer ownership boundaries in remaining large production files

**Total estimate:** ~148 hours

---

## Sprint 57: Giant-Test Refactor & Lifecycle Regression Expansion

**Duration:** 14 days (~144 hours)  
**Goal:** Reduce the largest test-maintainability hotspots while strengthening
the final lifecycle and factor-many regression story.

### Prerequisites from previous Sprints

- Sprint 51-56 public lifecycle and decomposition work complete enough to freeze the intended steady-state behavior

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Large-Test Audit | Re-rank the largest test files by maintainability pain and behavior overlap. | 16 |
| 2 | Direct-Solver Test Refactor Batch | Extract helpers or split the strongest direct-solver test hotspot(s). | 28 |
| 3 | Iterative / Eigensolver Test Refactor Batch | Extract helpers or split the strongest solver-family test hotspot(s). | 24 |
| 4 | Lifecycle Regression Coverage | Add direct coverage for the final public direct-solver and repeated-run lifecycle surfaces. | 26 |
| 5 | Factor-Many / Compatibility Coverage | Add high-signal regression coverage for one-shot compatibility and factor-many workflows. | 22 |
| 6 | Validation and Closeout | Full validation and sprint closeout package. | 28 |

### Deliverables

- materially more maintainable major test binaries
- stronger direct lifecycle and factor-many regression coverage
- preserved behavior-level confidence

**Total estimate:** ~144 hours

---

## Sprint 58: Documentation, Examples & Benchmark Story Simplification

**Duration:** 14 days (~136 hours)  
**Goal:** Reduce public sprint-history narrative, modernize the example and
benchmark workflow story, and make the final product surfaces easier to scan.

### Prerequisites from previous Sprints

- Sprint 52-57 steady-state lifecycle and hotspot behavior mostly fixed

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Public Docs Audit | Audit README, tutorial, public headers, example docs, and benchmark docs for stale sprint chronology and lifecycle drift. | 18 |
| 2 | README and Tutorial Reduction | Tighten the top-level docs around stable workflow guidance rather than sprint narrative. | 26 |
| 3 | Header Narrative Cleanup | Remove stale sprint-history and “future sprint” wording from the strongest public-header offenders. | 26 |
| 4 | Example Modernization | Add or update the smallest high-signal examples for the final lifecycle / factor-many story. | 20 |
| 5 | Benchmark Taxonomy Cleanup | Reorganize benchmark docs around stable workflow categories rather than accumulated sprint-local terminology. | 20 |
| 6 | Sanity Sweep and Closeout | Repo-wide doc sanity checks, artifacts, and closeout package. | 26 |

### Deliverables

- smaller, more product-level public documentation
- examples aligned to the final lifecycle story
- benchmark docs organized by workflow instead of sprint history

**Total estimate:** ~136 hours

---

## Sprint 59: Quality/Platform Follow-Through, Final Integration & Epic 5 Closeout

**Duration:** 14 days (~132 hours)  
**Goal:** Revisit the remaining staged quality-platform limits, run the final
cross-surface integration sweep, and close Epic 5 from a measured baseline.

### Prerequisites from previous Sprints

- Sprint 50-58 implementation and documentation work complete

### Items

| # | Item Name | Item Description | Estimate |
|---|---|---|---:|
| 1 | Quality-Platform Residual Audit | Reassess serialized dead-code execution, macOS staging, Windows reviewed-wrapper parity, Windows dead-code exclusion, and coverage calibration. | 20 |
| 2 | Bounded Quality Follow-Through Batch | Land only the quality/platform improvements that are now clearly justified and bounded. | 24 |
| 3 | Final Cross-Surface Compatibility Sweep | Reconcile API/docs/examples/benchmarks/tests so the final caller story agrees everywhere. | 24 |
| 4 | Full Validation Sweep | Run the full maintained quality gates, truthfulness anchors, and targeted follow-ons from the final Epic 5 state. | 22 |
| 5 | Epic 5 Summary and Handoff | Write the final sprint closeout, Epic 5 retrospective inputs, and handoff summary. | 20 |
| 6 | Project-Plan / Residual Journal Finalization | Update final residual limits and project-level summary artifacts. | 22 |

### Deliverables

- final Epic 5 measured validation baseline
- updated disposition of the remaining staged quality-platform surfaces
- final integrated API/docs/examples/benchmarks/tests story
- Epic 5 closeout and handoff package

**Total estimate:** ~132 hours
