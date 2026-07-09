# Project Plan: Post-Epic-10 Product Hardening, Oracle Breadth & Packaging Decisions -- Sprints 118-127 (Epic 11)

Epic 11 starts from the completed Epic 10 closeout. The library is broad,
well-tested, and much more product-disciplined than it was before Epic 10, but
it still should not claim unqualified state-of-the-art status. Epic 11 focuses
on the next level of maturity: reducing remaining source/test ownership risk,
widening numerical oracle architecture, improving local performance governance,
making package/ABI/platform decisions, and simplifying the adoption surface.

This plan is based on:

- `reviews/review-codex-2026-07-09.md`
- `reviews/todo-codex-2026-07-09.md`
- `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md`
- Sprint 117's final residual queue and explicit non-claim register
- prior Epic retrospectives and carry-forward work

Each sprint remains 14 days and stays within the requested cap of 168 hours.

---

## Sprint 118: Epic 11 Baseline, Residual Conversion & Product Truth Freeze

**Duration:** 14 days (~166 hours)
**Goal:** Freeze the post-Epic-10 baseline, convert the final Epic 10 residual
queue into Epic 11 owners, and define the claim/evidence rules for the next
hardening cycle.

### Prerequisites from Previous Sprints

- Epic 10 retrospective complete
- Sprint 117 final validation, comparison, residual, and non-claim artifacts
  available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Baseline Quality Recheck | Reconfirm reviewed Make/CMake parity, source-list, CTest count, install/package, benchmark, coverage, and CI tier surfaces. | 22 |
| 2 | Residual Queue Conversion | Convert Sprint 117 post-Epic residuals into Epic 11 owners, dependencies, proof gates, and duplicate fences. | 24 |
| 3 | Current Product Truth Map | Freeze compressed-first, mutable-shell, solver-family, package/platform, benchmark, and public non-claim truth. | 22 |
| 4 | Source/Test Hotspot Metrics | Capture current file-size, responsibility, and ownership metrics for large source and giant-test owners. | 24 |
| 5 | Evidence Template Refresh | Update templates for source movement, oracle expansion, performance sentinels, package/ABI decisions, and adoption cleanup. | 26 |
| 6 | Public Claim Drift Audit | Recheck public/support docs against final Epic 10 claims and new Epic 11 candidate claims. | 20 |
| 7 | Sprint Closeout | Produce Sprint 118 artifacts, working notes, and handoff requirements for implementation sprints. | 28 |

### Deliverables

- post-Epic-10 baseline package
- Epic 11 residual owner map
- current product truth map
- source/test hotspot metrics
- refreshed evidence templates
- Sprint 119-127 handoff requirements

**Total estimate:** ~166 hours

---

## Sprint 119: Eigensolver Source Boundary & Proof-Owner Follow-Through

**Duration:** 14 days (~168 hours)
**Goal:** Convert the safest Epic 10 eigensolver residual movements into
validated source-boundary improvements without widening public claims.

### Prerequisites from Previous Sprints

- Sprint 118 residual owner map complete
- source-list and CMake parity expectations documented

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Movement Feasibility Audit | Re-rank eigensolver residuals: private owner movement, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup, and `lanczos_iterate_op`. | 18 |
| 2 | Source Boundary Design | Define exact old/new files, internal headers, ownership contracts, source-list updates, CMake updates, and rollback plans. | 26 |
| 3 | First Movement Batch | Move the lowest-risk private eigensolver owner with focused compile and consumer proof. | 30 |
| 4 | Selection/Lifting Batch | Move or explicitly defer `s20_select_indices` and `s20_lift_ritz_vectors` based on grow-m, thick-restart, and LOBPCG proof. | 34 |
| 5 | Shift-Invert Boundary Decision | Split or defer shift-invert setup/conversion after proving LDLT lifecycle, operator selection, error propagation, and cleanup ownership. | 24 |
| 6 | Validation and Parity | Run source-list, Make/CMake, focused eigensolver tests, CTest count checks, and required quality gates. | 20 |
| 7 | Closeout and Non-Claims | Document what moved, what stayed, and why no broad eigensolver parity claim was created. | 16 |

### Deliverables

- eigensolver source-boundary decision package
- validated source movement where safe
- source-list and CMake parity evidence
- focused eigensolver consumer proof
- explicit residuals and non-claims

**Total estimate:** ~168 hours

---

## Sprint 120: Direct/Iterative Oracle Architecture & Giant-Test Split

**Duration:** 14 days (~168 hours)
**Goal:** Create a maintainable direct/iterative oracle architecture and reduce
giant test ownership in the highest-risk direct and iterative proof files.

### Prerequisites from Previous Sprints

- Sprint 118 evidence templates complete
- Sprint 119 source-boundary validation lessons available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Oracle Ownership Audit | Audit QR, CG, GMRES, BiCGSTAB, MINRES, LDLT, LU, and Cholesky generated-RHS and dense-reference proof owners. | 20 |
| 2 | Shared Fixture Design | Design shared direct/iterative fixture builders without hiding solver-specific tolerances or failure modes. | 24 |
| 3 | Direct Test Split Batch | Split selected `test_qr.c`, `test_ldlt.c`, or `test_ldlt_csc.c` proof blocks into focused scenario/helper owners. | 34 |
| 4 | Iterative Test Split Batch | Split selected `test_iterative.c`, `test_bicgstab.c`, or `test_minres.c` oracle/progress blocks into focused scenario/helper owners. | 30 |
| 5 | Cross-Solver Oracle Pilot | Add one bounded shared oracle pilot for generated-RHS or dense-reference comparison across compatible direct/iterative paths. | 28 |
| 6 | Validation | Run focused tests, full reviewed quality if `.c`/`.h` changed, and Make/CMake parity checks. | 20 |
| 7 | Documentation and Closeout | Document oracle interpretation and remaining non-claims. | 12 |

### Deliverables

- direct/iterative oracle architecture artifact
- reduced giant-test ownership in selected files
- bounded cross-solver oracle pilot
- validation and CTest/source-list evidence
- residual direct/iterative oracle queue

**Total estimate:** ~168 hours

---

## Sprint 121: SVD, QR & Rank-Deficient Numerical Oracle Expansion

**Duration:** 14 days (~168 hours)
**Goal:** Strengthen SVD, QR, rank, pseudoinverse, and least-squares evidence
with reusable helpers while keeping LAPACK/SciPy parity as a non-claim.

### Prerequisites from Previous Sprints

- Sprint 120 fixture/oracle patterns available
- SVD and QR proof-owner gaps identified

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | SVD/QR Evidence Audit | Rank current SVD, partial SVD, QR, rank-deficient, pseudoinverse, and low-rank proof gaps. | 18 |
| 2 | Matrix Taxonomy Design | Define deterministic rank, conditioning, rectangularity, sparsity, scaling, and expected-failure fixture classes. | 24 |
| 3 | SVD Helper Extraction | Extract bounded SVD reconstruction, orthogonality, rank, low-rank, and pseudoinverse proof helpers. | 30 |
| 4 | QR/Least-Squares Expansion | Add focused QR/least-squares/rank-deficient evidence using shared fixture taxonomy. | 30 |
| 5 | External/Dense Reference Pilot | Add a deterministic dense-reference or external-process comparison for one high-value SVD/QR lane. | 28 |
| 6 | Validation | Run focused SVD/QR tests and full C quality chain if code changed. | 20 |
| 7 | Docs and Non-Claims | Update solver-selection/maintainer guidance without claiming LAPACK/SciPy parity. | 18 |

### Deliverables

- SVD/QR/rank fixture taxonomy
- reusable SVD/QR proof helpers
- expanded rank-deficient evidence
- bounded dense-reference pilot
- updated trust-boundary docs

**Total estimate:** ~168 hours

---

## Sprint 122: Numerical Corpus, Coverage Architecture & Report Indexes

**Duration:** 14 days (~168 hours)
**Goal:** Turn scattered numerical fixtures, coverage, benchmark, dead-code,
and guardrail outputs into a clearer recurring assurance architecture.

### Prerequisites from Previous Sprints

- Sprint 118 templates complete
- Sprint 120-121 oracle taxonomy decisions available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Corpus Inventory | Inventory Matrix Market fixtures, generated families, known matrices, external-reference scripts, and expected failures. | 20 |
| 2 | Corpus Taxonomy | Define corpus tags for symmetry, definiteness, rank, conditioning, scale, sparsity pattern, ordering, and solver family. | 24 |
| 3 | Report Index Design | Design generated indexes for benchmark, coverage, dead-code, large-matrix, and oracle artifacts. | 24 |
| 4 | Coverage Architecture | Re-rank coverage gaps by risk and decide which remain supplemental versus reviewed. | 24 |
| 5 | Generated Index Batch | Implement or document the first generated report index without changing benchmark semantics. | 30 |
| 6 | Validation | Run report-generation checks, docs hygiene, and affected script/test checks. | 22 |
| 7 | Closeout | Publish corpus/report ownership and residual gaps. | 24 |

### Deliverables

- numerical corpus taxonomy
- report index design
- first generated report/index artifact
- coverage architecture decisions
- residual assurance queue

**Total estimate:** ~168 hours

---

## Sprint 123: Performance Sentinel & Backend Runtime Governance

**Duration:** 14 days (~168 hours)
**Goal:** Strengthen local performance and backend/runtime governance without
turning local measurements into portable performance claims.

### Prerequisites from Previous Sprints

- Sprint 122 report index model available
- current benchmark and backend non-claims preserved

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Hot Path Inventory | Identify hot compressed/direct/iterative/eigensolver/SVD/reorder paths with current sentinel coverage. | 20 |
| 2 | Backend Runtime Contract | Define builtin/optional dense backend observability, fallback, OpenMP, thread-count, and nested-runtime boundaries. | 26 |
| 3 | Sentinel Design | Design bounded local sentinels for high-value paths with explicit non-portable interpretation. | 24 |
| 4 | Sentinel Implementation Batch | Add or refine selected local sentinel/report lanes and generated metadata. | 34 |
| 5 | Benchmark Docs Cleanup | Improve benchmark interpretation and report-index handoff without changing benchmark claims. | 18 |
| 6 | Validation | Run focused benchmarks/report checks plus required C quality if code changed. | 26 |
| 7 | Closeout | Publish local performance claims, residuals, and non-claims. | 20 |

### Deliverables

- backend/runtime contract artifact
- updated local sentinel bundle
- generated benchmark metadata improvements
- validation evidence
- performance non-claim register

**Total estimate:** ~168 hours

---

## Sprint 124: Package, ABI & Shared-Library Product Decision

**Duration:** 14 days (~168 hours)
**Goal:** Decide whether Epic 11 adds shared-library/dynamic ABI support or
explicitly preserves static-first support, then implement the selected product
contract.

### Prerequisites from Previous Sprints

- Epic 10 static-first package truth preserved
- Sprint 118 package/ABI residual owner map complete

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | ABI/Product Decision Audit | Audit public headers, install shape, versioning, symbol exposure, package metadata, and downstream consumer expectations. | 22 |
| 2 | Shared-Library Design Or Deferral | Decide whether to implement shared-library ABI support or explicitly defer it with stronger support wording. | 26 |
| 3 | Build/Install Contract Batch | Implement selected static/shared packaging changes or update static-first enforcement and documentation. | 34 |
| 4 | ABI/Symbol Proof | Add ABI/symbol/version checks if shared support is added, or add deferral checks if static-first remains. | 28 |
| 5 | Downstream Consumer Proof | Strengthen CMake/pkg-config install consumer proof for the selected contract. | 24 |
| 6 | Validation | Run install scripts, CMake package checks, source/build checks, and required quality gates. | 22 |
| 7 | Closeout | Publish package/ABI support truth and residual package-manager work. | 12 |

### Deliverables

- package/ABI product decision
- build/install contract changes or explicit deferral proof
- downstream consumer validation
- updated README/INSTALL/maintainer package truth
- package/ABI residual queue

**Total estimate:** ~168 hours

---

## Sprint 125: Cross-Platform Install, Windows Staged Lanes & CI Tier Follow-Through

**Duration:** 14 days (~168 hours)
**Goal:** Advance platform support where feasible and make Linux/macOS/Windows
install and staged validation tiers even more explicit.

### Prerequisites from Previous Sprints

- Sprint 124 package/ABI decision complete
- current CI tier model preserved

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Platform Gap Audit | Re-audit Linux install CI, macOS install/export parity, Windows install validation, Windows thread/fuzz/property, and Windows Makefile gaps. | 22 |
| 2 | Linux Install CI Decision | Decide whether to promote Linux install proof to reviewed CI and implement or explicitly defer. | 24 |
| 3 | macOS Install/Export Follow-Through | Add, strengthen, or explicitly defer reviewed macOS CMake install/export parity. | 26 |
| 4 | Windows Install Validation Design | Design and implement or defer MSVC install/downstream consumer proof with exact CTest count implications. | 34 |
| 5 | Windows Staged Test Follow-Through | Revisit thread/fuzz/property staged exclusions and CTest membership. | 24 |
| 6 | Validation | Run affected workflow-equivalent local checks and docs hygiene; update CI comments and support docs. | 24 |
| 7 | Closeout | Publish final platform tier and staged-exclusion register. | 14 |

### Deliverables

- platform gap audit
- Linux/macOS/Windows install decisions
- Windows staged-lane decision package
- updated support-tier docs
- validation and non-claim evidence

**Total estimate:** ~168 hours

---

## Sprint 126: Adoption Surface Simplification & Documentation Productization

**Duration:** 14 days (~164 hours)
**Goal:** Simplify the adoption surface after Epic 10 by separating first-use
guides from maintainer history and making compressed-first workflows easier to
find.

### Prerequisites from Previous Sprints

- Sprint 122 report index decisions available
- Sprint 124-125 package/platform truth stable

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Adoption Surface Audit | Audit README, tutorial, solver selection, examples, algorithm docs, benchmark docs, install docs, and maintainer guide for overlap. | 20 |
| 2 | Algorithm Doc Split Design | Design split between concise current algorithm reference and historical measurement appendix. | 22 |
| 3 | Algorithm Doc Split Batch | Implement the split or a bounded first phase with redirects and link checks. | 30 |
| 4 | Compressed-First Cookbook | Add or reorganize examples/docs so compressed-first direct, iterative, Matrix Market, SVD, eigensolver, and benchmark paths are easier to follow. | 30 |
| 5 | Benchmark/Report Index Docs | Surface generated report indexes and local-measurement interpretation in concise adoption language. | 22 |
| 6 | Link and Claim Validation | Run docs hygiene, link/path checks, and claim-boundary scans. | 20 |
| 7 | Closeout | Publish adoption simplification metrics and residual docs work. | 20 |

### Deliverables

- simplified adoption-surface map
- algorithm doc split or first phase
- compressed-first cookbook updates
- benchmark/report index docs
- claim-boundary validation

**Total estimate:** ~164 hours

---

## Sprint 127: Final Integration, Competitive Recalibration & Epic 11 Closeout

**Duration:** 14 days (~164 hours)
**Goal:** Validate Epic 11 outcomes, compare them against the state-of-the-art
target, publish earned claims and non-claims, and close the epic with a
retrospective and post-epic handoff queue.

### Prerequisites from Previous Sprints

- Sprint 118-126 artifacts, validation, residuals, and support-surface
  decisions complete
- package/platform/ABI/adoption truth stable

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Final Evidence Inventory | Inventory source/test ownership changes, oracle evidence, performance sentinels, package/platform proof, docs changes, and residuals. | 22 |
| 2 | Full Validation Design | Define required reviewed and supplemental validation for touched surfaces. | 18 |
| 3 | Full Validation Execution | Run required quality, CMake, source-list, package/install, docs, benchmark, and supplemental lanes as applicable. | 30 |
| 4 | Competitive Claim Recalibration | Compare final evidence against Epic 11 goals and current state-of-the-art non-claims. | 24 |
| 5 | Unsupported-Claim Cleanup | Remove, downgrade, or fence unsupported public/support wording. | 20 |
| 6 | Residual Queue Publication | Publish post-Epic-11 residuals, future-epic candidates, optional work, and explicit non-claims. | 20 |
| 7 | Sprint and Epic Retrospectives | Write Sprint 127 and Epic 11 retrospectives plus final closeout handoff. | 30 |

### Deliverables

- final validation package
- final comparison and claim cleanup package
- post-Epic-11 residual queue
- Sprint 127 retrospective
- Epic 11 retrospective
- final handoff artifact

**Total estimate:** ~164 hours
