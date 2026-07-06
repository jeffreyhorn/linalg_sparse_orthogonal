# Project Plan: Product-Grade Sparse Linear Algebra Maturity & Competitive Calibration -- Sprints 100-111 (Epic 10)

Epic 10 turns the post-Epic-9 library into a more product-grade sparse linear
algebra system. The goal is not to declare the project state of the art by
ambition. The goal is to earn narrower, evidence-backed claims through
compressed-first workflows, external oracle depth, maintainable implementation
owners, clearer API and documentation surfaces, and stronger package/platform
proof.

This plan is based on:

- `reviews/review-codex-2026-06-30.md`
- `reviews/todo-codex-2026-06-30.md`
- the Epic 7, Epic 8, and Epic 9 retrospectives
- the Epic 9 carry-forward queue and explicit non-claims

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 100: Epic 10 Baseline, State-of-the-Art Target & Evidence Contract

**Duration:** 14 days (~166 hours)
**Goal:** Freeze the post-Epic-9 baseline, define the precise
state-of-the-art comparison target, and create the evidence templates that will
govern implementation work across Epic 10.

### Prerequisites from Previous Sprints

- Epic 9 closeout complete
- Epic 9 retrospective, residual queue, and post-epic handoff available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Baseline Quality Recheck | Reconfirm reviewed Make, CMake, install/export, source-list, benchmark, and supplemental coverage surfaces from the live tree. | 20 |
| 2 | State-of-the-Art Definition | Define the exact libraries, features, evidence standards, and non-goals that bound Epic 10's competitive target. | 28 |
| 3 | Residual Queue Conversion | Convert Epic 9 carry-forward work into an Epic 10 claim map with owners and risk levels. | 24 |
| 4 | Evidence Templates | Create templates for solver comparison, benchmark interpretation, platform tiering, packaging proof, and non-claim tracking. | 26 |
| 5 | Baseline Metrics Artifact | Capture current source/test sizes, CTest counts, install proof, external comparison lanes, and known maintainability hotspots. | 22 |
| 6 | Public Claim Audit | Recheck README, install, benchmark, examples, and maintainer docs for unsupported state-of-the-art wording. | 20 |
| 7 | Sprint Closeout | Produce Sprint 100 artifacts, working notes, and handoff criteria for implementation sprints. | 26 |

### Deliverables

- post-Epic-9 baseline artifact
- Epic 10 state-of-the-art target and non-goal fence
- evidence templates for comparisons, benchmarks, packaging, and platform tiers
- claim map for Sprints 101-109

**Total estimate:** ~166 hours

---

## Sprint 101: Compressed-First Product Model & Storage Front Door

**Duration:** 14 days (~168 hours)
**Goal:** Make compressed CSR/CSC workflows the unmistakable product center
while preserving mutable matrix-shell compatibility as a secondary path.

### Prerequisites from Previous Sprints

- Sprint 100 claim map and evidence templates complete
- compressed-first residuals ranked by user value and compatibility risk

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Storage Surface Audit | Audit public construction, import, mutation, publication, and solver entry paths for remaining linked-list-first product costs. | 20 |
| 2 | Compressed-First API Design | Define the bounded CSR/CSC-first API additions or refinements needed for the highest-value workflows. | 28 |
| 3 | Constructor and Import Batch | Land the first implementation batch for compressed-first construction/import with validation and compatibility behavior. | 34 |
| 4 | Lifecycle and Ownership Batch | Clarify ownership, lifetime, and repeated-run rules for compressed matrices and solver handles. | 28 |
| 5 | Compatibility Path Documentation | Document mutable matrix-shell compatibility as a supported but secondary product path. | 16 |
| 6 | Regression Proof | Add tests and examples proving compressed-first ownership, error handling, and solver entry behavior. | 22 |
| 7 | Validation and Closeout | Run required checks, capture artifacts, and close Sprint 101 with updated product wording. | 20 |

### Deliverables

- clearer compressed-first public workflow
- reduced linked-list shell tax on high-value solver paths
- compatibility documentation for mutable matrix-shell users
- regression proof for compressed-first construction and lifecycle behavior

**Total estimate:** ~168 hours

---

## Sprint 102: Direct Solver Robustness & External Oracle Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Deepen correctness evidence for direct solvers with named external
or dense-reference comparisons and cleaner family-local proof ownership.

### Prerequisites from Previous Sprints

- Sprint 100 evidence templates complete
- Sprint 101 compressed-first ownership rules stable enough for solver tests

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Direct Solver Gap Audit | Re-rank Cholesky, LDLT, LU, QR, SVD, and dispatch paths by oracle depth, fixture diversity, and failure-mode clarity. | 18 |
| 2 | Fixture Taxonomy | Define matrix fixture classes for symmetry, definiteness, rank, scaling, sparsity pattern, ordering, and expected failures. | 22 |
| 3 | Oracle Helper Extraction | Extract reusable dense-reference or external-comparison helpers from giant direct-solver tests. | 28 |
| 4 | LDLT/Cholesky Expansion | Add the highest-value CSC direct-family oracle expansion beyond Epic 9's bounded lane. | 28 |
| 5 | LU/QR/SVD Expansion | Add the highest-value oracle and failure-mode coverage for LU, QR, and SVD. | 34 |
| 6 | Solver Selection Docs | Update direct-solver guidance with capability, failure, and trust-boundary wording. | 18 |
| 7 | Validation and Closeout | Run required checks, regenerate comparison artifacts, and close Sprint 102. | 20 |

### Deliverables

- expanded direct-solver oracle coverage
- reusable direct-solver fixture and comparison helpers
- clearer direct-solver failure-mode tests
- updated direct-solver selection and trust-boundary documentation

**Total estimate:** ~168 hours

---

## Sprint 103: Iterative, Eigensolver & SVD External Comparison

**Duration:** 14 days (~168 hours)
**Goal:** Raise evidence quality for iterative solvers, eigensolvers, and SVD
without claiming broad parity with mature external packages prematurely.

### Prerequisites from Previous Sprints

- Sprint 100 comparison template complete
- Sprint 102 oracle helper patterns available for reuse

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Solver Family Audit | Rank CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and SVD paths by comparison weakness and user impact. | 18 |
| 2 | Convergence Fixture Design | Create matrix families for convergence, stagnation, tolerance, restart, preconditioning, residual, and rank behavior. | 26 |
| 3 | Iterative Oracle Batch | Add focused external or deterministic reference comparisons for the highest-value iterative paths. | 34 |
| 4 | Eigensolver Oracle Batch | Add focused comparison and residual evidence for eigen, thick-restart, and LOBPCG paths. | 30 |
| 5 | SVD Comparison Follow-Through | Extend SVD comparison evidence where it shares fixture or reporting infrastructure with eigensolver work. | 18 |
| 6 | Reporting and Docs | Document convergence-profile interpretation, residual criteria, and non-claims for broad external parity. | 20 |
| 7 | Validation and Closeout | Run required checks, archive artifacts, and close Sprint 103. | 22 |

### Deliverables

- stronger iterative/eigensolver/SVD comparison artifacts
- reusable convergence fixture taxonomy
- clearer residual and convergence reporting
- explicit external-parity non-claims where evidence remains bounded

**Total estimate:** ~168 hours

---

## Sprint 104: Performance Backend & Parallel Runtime Modernization

**Duration:** 14 days (~168 hours)
**Goal:** Establish a durable backend and runtime contract for builtin dense
kernels, optional acceleration, OpenMP behavior, and bounded local regression
evidence.

### Prerequisites from Previous Sprints

- Sprint 100 benchmark and claim templates complete
- Sprint 101-103 solver ownership changes stable enough to measure

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Backend Consumer Audit | Profile dense backend consumers, fallback paths, and optional acceleration points across direct and decomposition families. | 20 |
| 2 | Runtime Contract Design | Define runtime expectations for builtin kernels, optional backends, OpenMP, nested parallelism, and observability. | 30 |
| 3 | Backend Descriptor Batch | Refine or extend the backend descriptor and selection surface without weakening builtin fallback truth. | 32 |
| 4 | OpenMP and Threading Cleanup | Reduce confusing thread-local/global override behavior and document remaining runtime controls. | 24 |
| 5 | Performance Sentinel Batch | Add bounded local regression sentinels for hot paths without claiming portable timing superiority. | 22 |
| 6 | Benchmark Reporting Alignment | Reconcile benchmark docs, scripts, and artifacts with the refined backend/runtime contract. | 18 |
| 7 | Validation and Closeout | Run required checks, compare benchmark artifacts, and close Sprint 104. | 22 |

### Deliverables

- clearer backend descriptor and runtime selection contract
- documented builtin fallback and optional acceleration behavior
- bounded local performance regression sentinels
- benchmark wording aligned with actual evidence

**Total estimate:** ~168 hours

---

## Sprint 105: Reordering, Graph & Large-Matrix Scalability

**Duration:** 14 days (~168 hours)
**Goal:** Improve reorder, graph, and large-matrix evidence with clearer fill,
ordering, runtime, and memory interpretation.

### Prerequisites from Previous Sprints

- Sprint 100 evidence templates complete
- Sprint 104 runtime/benchmark contract stable enough for measurement

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Reorder/Graph Audit | Re-rank AMD, COLAMD, nested dissection, quotient graph, graph partition, and fill-report gaps. | 20 |
| 2 | Fill Metrics Contract | Define canonical fill, runtime, memory, and fixture naming fields for reorder artifacts. | 22 |
| 3 | Named Matrix Expansion | Add or refresh reorder/fill comparisons on named matrices and generated graph families. | 30 |
| 4 | Scalability Guardrails | Add deterministic large-matrix memory/runtime guardrails where suitable for reviewed or supplemental checks. | 26 |
| 5 | Graph Ownership Cleanup | Remove touched history-heavy comments and extract helpers from graph/reorder proof owners where useful. | 26 |
| 6 | Reporting and Docs | Consolidate reorder/fill reporting and user interpretation guidance. | 20 |
| 7 | Validation and Closeout | Run required checks, regenerate artifacts, and close Sprint 105. | 24 |

### Deliverables

- clearer reorder/fill metric contract
- expanded named-matrix and generated-family evidence
- improved large-matrix scalability guardrails
- cleaner graph/reorder ownership and reporting

**Total estimate:** ~168 hours

---

## Sprint 106: Large-Source & Giant-Test Maintainability Phase 7

**Duration:** 14 days (~168 hours)
**Goal:** Continue extracting the largest implementation and proof owners so
future solver, backend, and platform changes are smaller and safer.

### Prerequisites from Previous Sprints

- Sprint 100 hotspot baseline complete
- Sprint 102-105 family work has identified the highest-value extraction seams

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Extraction Target Re-rank | Re-rank large source and giant-test files by churn risk, ownership ambiguity, and failure-localization value. | 18 |
| 2 | LDLT/CSC Source Extraction | Extract one high-value ownership seam from the largest CSC direct-family source path. | 30 |
| 3 | LU/QR/Eigs Source Extraction | Extract one or two focused seams from LU, QR, eigensolver, or iterative source hotspots. | 34 |
| 4 | Giant-Test Fixture Extraction | Split reusable fixtures/helpers from the largest direct, graph, and integration test owners. | 34 |
| 5 | Source-List and CMake Follow-Through | Keep Make/CMake source lists, source-list checker, and reviewed parity exact after extraction. | 18 |
| 6 | Maintainability Documentation | Update maintainer guidance for new file ownership and test layout. | 14 |
| 7 | Validation and Closeout | Run required checks, compare file metrics, and close Sprint 106. | 20 |

### Deliverables

- smaller high-risk source owners
- smaller and more focused giant-test owners
- updated source-list/CMake parity
- maintainability artifact showing before/after ownership metrics

**Total estimate:** ~168 hours

---

## Sprint 107: Residual Maintainability Debt & Proof-Owner Cleanup

**Duration:** 14 days (~168 hours)
**Goal:** Resolve Sprint 106's explicitly deferred maintainability debt with
bounded proof-owner cleanup, fresh source/test boundaries, and no opportunistic
public API or install-header expansion.

### Prerequisites from Previous Sprints

- Sprint 106 extraction artifacts, working notes, and residual-debt queue complete
- source-list and CMake reconciliation from Sprint 106 merged on `master`
- current reviewed test counts and helper-target constraints known

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Residual Boundary Re-rank | Convert Sprint 106 residual debt into a fresh boundary artifact for `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`, `tests/test_svd.c`, `src/sparse_eigs.c`, and `src/sparse_matrix.c` so cleanup order is explicit and non-duplicative. | 18 |
| 2 | LDLT CSC Proof-Owner Follow-Through | Extract one narrow row-adjacency assertion helper or residual/oracle helper from `tests/test_ldlt_csc.c` after the boundary artifact, keeping the direct CSC proof intent visible. | 26 |
| 3 | QR and Iterative Fixture Cleanup | Extract repeated matrix, vector, RHS, or option builders from `tests/test_qr.c` and `tests/test_iterative.c` without hiding solve, convergence, or reconstruction assertions. | 28 |
| 4 | SVD Proof-Owner Cleanup | Perform a dedicated `tests/test_svd.c` cleanup pass with focused validation before moving any rank, oracle, or reconstruction helpers. | 26 |
| 5 | Eigensolver Boundary and First Split | Create a fresh `src/sparse_eigs.c` boundary tied to Sprint 103 comparison surfaces, then split only the safest workspace or dispatch helper if the boundary shows a low-risk cut. | 24 |
| 6 | Central Matrix Shell Deferral Contract | Document why `src/sparse_matrix.c` remains central API/compatibility territory, identify future split preconditions, and prevent opportunistic extraction from bypassing public behavior review. | 20 |
| 7 | Validation and Closeout | Run required checks, verify no public API/install-header or reviewed test-count drift unless intentionally approved, update maintainability metrics, and close Sprint 107. | 26 |

### Deliverables

- residual-debt boundary artifact for remaining large source and proof owners
- one bounded `tests/test_ldlt_csc.c` helper extraction
- focused QR, iterative, and SVD proof-owner cleanup
- `src/sparse_eigs.c` boundary and optional first low-risk split
- documented `src/sparse_matrix.c` non-extraction contract
- updated maintainability metrics and validation closeout

**Total estimate:** ~168 hours

---

## Sprint 108: Residual Proof-Owner & Source Boundary Follow-Through

**Duration:** 14 days (~168 hours)
**Goal:** Convert Sprint 107's residual deferred debt into bounded follow-through
work for remaining proof-owner cleanup and source-owner boundary planning without
duplicating completed Sprint 107 helper extractions or changing public support
surfaces opportunistically.

### Prerequisites from Previous Sprints

- Sprint 107 residual-debt retrospective and boundary artifacts complete
- QR, iterative, SVD, and LDLT CSC helper extractions from Sprint 107 merged
- `src/sparse_eigs.c` and `src/sparse_matrix.c` deferral contracts available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Residual Proof-Owner Boundary Refresh | Re-rank the remaining large proof owners after Sprint 107, explicitly excluding already-completed row-adjacency, QR small-fixture, iterative matrix-free, and SVD diagonal/rank-1 cleanup. | 14 |
| 2 | LDLT CSC Oracle Helper Follow-Through | Extract at most one additional named assertion, residual, or oracle helper from `tests/test_ldlt_csc.c`, keeping direct-solver proof intent and failure localization visible. | 24 |
| 3 | QR Fixture Follow-Through | Extract only safe generated-fixture, tall/economy, diagonal/singleton, or SuiteSparse exact-RHS setup from `tests/test_qr.c` after preserving visible solve, rank, reconstruction, and refinement assertions. | 26 |
| 4 | Iterative Convergence Boundary Cleanup | Perform one bounded `tests/test_iterative.c` cleanup pass that preserves solver options, restart values, preconditioner setup, convergence results, and direct comparisons at call sites. | 24 |
| 5 | SVD Oracle and Reconstruction Boundary Cleanup | Create a dedicated validation lane for remaining `tests/test_svd.c` rank, oracle, reconstruction, pseudoinverse, low-rank, partial-SVD, and condition-number proof logic, then extract only one safe helper family. | 28 |
| 6 | Eigensolver Source Feasibility Plan | Prepare a source-list, Make/CMake, and cross-backend validation plan for a future `src/sparse_eigs.c` extraction, focused on dense Jacobi feasibility or grow-m refinement boundaries without landing the split unless the plan proves low risk. | 22 |
| 7 | Matrix Shell Public-Behavior Review | Review `src/sparse_matrix.c` public behavior, private-header dependencies, allocation/state ownership, and compatibility constraints so any future shell extraction has explicit prerequisites. | 18 |
| 8 | Validation and Closeout | Run required checks, verify no public API/install-header or reviewed test-count drift, update maintainability metrics, and close Sprint 108. | 12 |

### Deliverables

- refreshed residual proof-owner boundary artifact
- one bounded additional LDLT CSC proof helper if the boundary remains low risk
- focused QR, iterative, and SVD residual cleanup artifacts and changes
- eigensolver source extraction feasibility plan
- matrix shell public-behavior and private-header dependency review
- validation and maintainability closeout

**Total estimate:** ~168 hours

---

## Sprint 109: Residual Source Boundary & Proof-Owner Debt Closeout

**Duration:** 14 days (~168 hours)
**Goal:** Convert Sprint 108's residual deferred debt into one bounded
implementation/source-boundary pass without duplicating completed Sprint 108
helpers, broadening public support surfaces, or hiding proof logic in the
remaining giant tests.

### Prerequisites from Previous Sprints

- Sprint 108 retrospective and residual-debt closeout artifacts complete
- Sprint 108 LDLT CSC, QR, iterative, and SVD helper follow-through merged
- Sprint 108 eigensolver feasibility and matrix-shell public-behavior reviews
  available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Residual Debt Intake and Dependency Ordering | Re-read Sprint 108 residual debt, explicitly exclude completed Sprint 108 helpers, and order eigensolver, matrix-shell, and giant-test work so no later item is required by an earlier one. | 12 |
| 2 | Eigensolver Dense Jacobi Source Boundary | Revalidate the `s21_dense_sym_jacobi` split candidate, define the private source owner, Make/CMake/manifest updates, and focused `test_eigs`, `test_eigs_thick_restart`, `test_eigs_lobpcg`, and `test_sprint29_integration` gates before code moves. | 20 |
| 3 | Eigensolver Dense Jacobi Extraction | If Item 2 proves low risk, move only `s21_dense_sym_jacobi` into a private dense spectral helper source with source-list parity and no public header or CTest registration drift; otherwise publish a no-split deferral artifact. | 28 |
| 4 | Eigensolver Behavior-Sensitive Boundary Audit | Audit grow-m refinement, dispatch/defaults, handle/workspace glue, shift-invert, and shared Lanczos kernels, adding behavior evidence and explicit no-go conditions before any future movement. | 24 |
| 5 | Matrix Shell Candidate Boundary Contract | Choose one future `src/sparse_matrix.c` public-behavior owner, document private-header dependencies, source-list requirements, focused public behavior tests, and solver-smoke gates without moving matrix-shell code unless the boundary is independently low risk. | 24 |
| 6 | Giant-Test Proof-Owner Cleanup Pass | Perform one additional bounded helper-family cleanup across `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`, or `tests/test_svd.c`, preserving proof assertions and avoiding new compiled helper targets. | 32 |
| 7 | Validation, Metrics, and Residual Closeout | Run required checks for touched files, verify no public API/install-header/source-list/helper-target/CTest drift, capture maintainability metrics, and publish residuals for the shifted downstream sprints. | 28 |

### Deliverables

- Sprint 108 residual-debt intake and dependency-ordering artifact
- eigensolver dense Jacobi source-boundary artifact and optional private helper
  extraction
- eigensolver grow-m/refinement/dispatch/handle/shared-kernel audit artifact
- matrix-shell public-behavior source-boundary contract
- one bounded giant-test proof-owner cleanup pass if the boundary remains low
  risk
- validation, metrics, and residual closeout

**Total estimate:** ~168 hours

---

## Sprint 110: API Usability, Documentation & Example Coherence

**Duration:** 14 days (~166 hours)
**Goal:** Make the project easier to adopt by giving users a concise
compressed-first path, solver-selection guidance, and examples that match the
actual product contracts.

### Prerequisites from Previous Sprints

- Sprint 101 compressed-first workflow stable
- Sprint 102-105 solver and benchmark evidence available
- Sprint 109 residual source-boundary and proof-owner cleanup complete enough
  that docs do not describe unstable proof-owner layouts

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | User Journey Audit | Audit README, tutorial, examples, install docs, benchmark docs, and public headers from a first-time user perspective. | 18 |
| 2 | Solver Selection Guide | Create a concise guide for matrix format, direct solve, iterative solve, eigen solve, decomposition reuse, and reorder/fill workflows. | 28 |
| 3 | Compressed-First Example Batch | Update or add minimal robust examples that start from CSR/CSC workflows. | 30 |
| 4 | Error and Ownership Docs | Tighten public header and guide wording for ownership, zero/default options, errors, and runtime behavior. | 24 |
| 5 | Benchmark Interpretation Docs | Make local benchmark and comparison artifact interpretation clear without overclaiming portability. | 18 |
| 6 | Maintainer/User Split Cleanup | Move maintainer-only proof language away from adoption surfaces where practical. | 22 |
| 7 | Validation and Closeout | Run doc/example checks, capture artifacts, and close Sprint 110. | 26 |

### Deliverables

- concise solver-selection and matrix-format guide
- compressed-first examples for common workflows
- clearer ownership, error, runtime, and benchmark documentation
- public docs separated more cleanly from maintainer proof language

**Total estimate:** ~166 hours

---

## Sprint 111: Packaging, ABI & Cross-Platform Validation Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Strengthen packaging and platform support truth, including an
explicit decision on static-first versus shared-library/ABI proof.

### Prerequisites from Previous Sprints

- Sprint 100 platform and package evidence template complete
- Sprint 110 user-facing docs ready to describe support tiers accurately

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Package Surface Audit | Audit Make install, CMake install/export, pkg-config, examples, downstream consumers, versioning, and exact-package behavior. | 20 |
| 2 | ABI Support Decision | Decide whether to add shared-library/ABI-version proof in Epic 10 or preserve static-first as the explicit support tier. | 24 |
| 3 | Install/Consumer Proof Batch | Strengthen install/export and downstream consumer validation based on the support-tier decision. | 32 |
| 4 | Platform Tier Contract | Define Linux, macOS, and Windows support tiers, reviewed checks, exclusions, and non-claims. | 24 |
| 5 | Windows/macOS Follow-Through | Move any practical Windows or macOS exclusions into reviewed parity, or document why they remain staged. | 28 |
| 6 | Packaging Docs | Update install, README, CMake, pkg-config, and maintainer docs to match final package/platform truth. | 18 |
| 7 | Validation and Closeout | Run required checks, archive platform/package artifacts, and close Sprint 111. | 22 |

### Deliverables

- explicit static-first or shared-library/ABI support decision
- stronger install/export and downstream consumer proof
- documented platform support tiers
- package and CI docs aligned with actual validation

**Total estimate:** ~168 hours

---

## Sprint 112: Final Integration, Competitive Calibration & Epic 10 Closeout

**Duration:** 14 days (~164 hours)
**Goal:** Integrate Epic 10 outcomes, validate all reviewed surfaces, compare
final evidence against the state-of-the-art target, and close the epic with
truthful earned claims and residuals.

### Prerequisites from Previous Sprints

- Sprints 100-111 complete
- final comparison, benchmark, package, platform, and maintainability artifacts available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | End-State Claim Audit | Compare final implementation and evidence against the Sprint 100 state-of-the-art claim map. | 20 |
| 2 | Full Validation Pass | Run the strongest reviewed quality, CMake parity, install/export, source-list, and supplemental evidence commands required by touched files. | 30 |
| 3 | Final Comparison Package | Regenerate final solver, reorder, benchmark, coverage, and package artifacts for the epic. | 28 |
| 4 | Unsupported Claim Cleanup | Remove, downgrade, or fence any public/support wording not backed by final evidence. | 20 |
| 5 | Residual Queue and Non-Claims | Publish the post-Epic-10 residual queue and explicit non-claims for future work. | 18 |
| 6 | Sprint 112 Retrospective | Create Sprint 112 closeout notes and retrospective from artifacts and working notes. | 20 |
| 7 | Epic 10 Retrospective | Create the Epic 10 retrospective with earned claims, metrics, lessons, and carry-forward work. | 28 |

### Deliverables

- final Epic 10 validation package
- competitive calibration against the state-of-the-art target
- unsupported-claim cleanup
- Sprint 112 retrospective
- Epic 10 retrospective and post-epic handoff queue

**Total estimate:** ~164 hours

---

## Epic 10 Estimate Summary

| sprint | title | estimate |
|---|---|---:|
| 100 | Epic 10 Baseline, State-of-the-Art Target & Evidence Contract | 166 |
| 101 | Compressed-First Product Model & Storage Front Door | 168 |
| 102 | Direct Solver Robustness & External Oracle Expansion | 168 |
| 103 | Iterative, Eigensolver & SVD External Comparison | 168 |
| 104 | Performance Backend & Parallel Runtime Modernization | 168 |
| 105 | Reordering, Graph & Large-Matrix Scalability | 168 |
| 106 | Large-Source & Giant-Test Maintainability Phase 7 | 168 |
| 107 | Residual Maintainability Debt & Proof-Owner Cleanup | 168 |
| 108 | Residual Proof-Owner & Source Boundary Follow-Through | 168 |
| 109 | Residual Source Boundary & Proof-Owner Debt Closeout | 168 |
| 110 | API Usability, Documentation & Example Coherence | 166 |
| 111 | Packaging, ABI & Cross-Platform Validation Expansion | 168 |
| 112 | Final Integration, Competitive Calibration & Epic 10 Closeout | 164 |
| **Total** | | **2,176** |
