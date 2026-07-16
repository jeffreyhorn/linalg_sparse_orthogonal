# Project Plan: Post-Epic-10 Product Hardening, Oracle Breadth & Packaging Decisions -- Sprints 118-134 (Epic 11)

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
- Sprint 122's residual deferred debt and explicit oracle non-claim register
- Sprint 123's residual deferred debt and explicit QR/partial-SVD/helper
  non-claim register
- Sprint 124's residual deferred debt and explicit QR/partial-SVD/helper
  non-claim register
- Sprint 125's residual deferred debt and explicit QR/nullspace/minimum-norm
  non-claim register
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
- Sprint 119-134 handoff requirements

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

## Sprint 122: SVD/QR External Oracle Residual Follow-Through

**Duration:** 14 days (~166 hours)
**Goal:** Convert Sprint 121's residual SVD, QR, partial-SVD, helper-ownership,
and solver-selection claim gates into explicit owners before broader corpus,
reporting, and adoption documentation work proceeds.

### Prerequisites from Previous Sprints

- Sprint 121 SVD/QR/rank fixture taxonomy complete
- Sprint 121 bounded SVD external-reference pilot and non-claim register
  available
- Sprint 120 fixture/oracle patterns available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Residual Oracle Dedupe and Owner Map | Convert Sprint 121 residual deferred debt into explicit SVD, QR, partial-SVD, helper, and documentation owners while confirming that completed audit, taxonomy, helper extraction, fixture expansion, and first SVD external-reference pilot work are not duplicated. | 16 |
| 2 | Additional SVD External Fixture Decision | Decide whether to add more SVD external fixtures beyond `svd_rect_fullrank_6x4`; if accepted, design bounded fixture keys, tolerances, skip behavior, and failure interpretation without claiming LAPACK/SciPy/NumPy parity. | 28 |
| 3 | QR External Dense-Reference Lane Design | Design a QR external dense-reference lane only after fixture size, tolerance, skip behavior, and failure semantics are explicit; implement or defer with an auditable rationale. | 30 |
| 4 | Partial-SVD External Parity Design | Design partial-SVD external comparison separately from full-SVD parity, including vector/subspace, convergence, ordering, tolerance, and non-claim semantics. | 30 |
| 5 | Helper Ownership Boundary Decisions | Revisit minimum-norm helper ownership migration and Bidiagonal/Golub-Kahan helper extraction boundaries, keeping specialized transpose/reconstruction semantics separate from general SVD helpers unless consolidation is proven safe. | 22 |
| 6 | Solver-Selection Claim Gate | Define the evidence threshold required before public solver-selection wording may mention broader external or support-level evidence. | 18 |
| 7 | Validation and Closeout | Run affected docs/script/focused checks, publish decisions, update non-claims, and hand residuals to the corpus/report and adoption sprints. | 22 |

### Deliverables

- Sprint 121 residual oracle owner map
- SVD external fixture expansion decision
- QR external dense-reference lane design or explicit deferral
- partial-SVD external parity design or explicit deferral
- helper ownership boundary decision package
- solver-selection claim gate
- validation and residual handoff package

**Total estimate:** ~166 hours

---

## Sprint 123: Residual SVD/QR Oracle, Helper & Claim Evidence Follow-Through

**Duration:** 14 days (~166 hours)
**Goal:** Promote Sprint 122's residual SVD, QR, partial-SVD, helper, and claim
debt into bounded implementation or explicit deferral packages before corpus,
report-index, performance, package, and adoption sprints consume the oracle
truth.

### Prerequisites from Previous Sprints

- Sprint 121 SVD/QR/rank fixture taxonomy complete
- Sprint 122 bounded SVD, QR, and partial-SVD external oracle lanes available
- Sprint 122 residual deferred debt and non-claim register available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Broader SVD External Fixture Matrix | Decide and implement or explicitly defer the next bounded SVD external fixture batch, including fixture taxonomy, reference trust model, vector/rank/pseudoinverse/low-rank semantics, tolerance policy, skip handling, and failure interpretation without claiming LAPACK, NumPy, or SciPy parity. | 26 |
| 2 | QR External Behavior Evidence Batch | Add or explicitly defer QR external compatible, rank-deficient, underdetermined/minimum-norm, and Q/economy evidence behind behavior-specific fixtures, basis rules, tolerance rules, and preserved QR/minimum-norm ownership. | 28 |
| 3 | Partial-SVD External Semantics Batch | Expand or explicitly defer partial-SVD external semantics beyond top-k values to vector, subspace, convergence-budget, repeated/clustered spectrum, rectangular, and rank-deficient behavior with sign/subspace and failure-interpretation rules. | 32 |
| 4 | Minimum-Norm Helper Migration Decision | Revisit minimum-norm helper migration using behavior-specific helper names while preserving QR, COLAMD, SVD-pseudoinverse, refinement, fallback, and SuiteSparse scenario ownership. | 22 |
| 5 | Bidiagonal/Golub-Kahan Helper Extraction Decision | Extract or explicitly defer Bidiagonal/Golub-Kahan helpers into a dedicated owner that preserves wide-transpose, implicit Householder reconstruction, explicit `U`/`V` reconstruction, and bidiagonal QR iteration semantics. | 24 |
| 6 | Maintainer Evidence Table Refresh | Refresh maintainer evidence tables with Sprint 122 and Sprint 123 oracle lane ownership, trust boundaries, validation commands, and non-claims. | 16 |
| 7 | Solver-Selection Claim Refresh Gate | Refresh public solver-selection wording only if the new evidence supports a user-facing claim; otherwise publish a no-update rationale and residual claim gates. | 18 |

### Deliverables

- broader SVD external fixture decision or bounded implementation
- QR external behavior evidence decision or bounded implementation
- partial-SVD external semantics decision or bounded implementation
- minimum-norm helper migration decision package
- Bidiagonal/Golub-Kahan helper extraction decision package
- maintainer evidence-table refresh
- solver-selection claim refresh or explicit no-update rationale

**Total estimate:** ~166 hours

---

## Sprint 124: Residual QR, Partial-SVD & Helper Oracle Follow-Through

**Duration:** 14 days (~166 hours)
**Goal:** Convert Sprint 123's residual QR, partial-SVD, minimum-norm, and
Bidiagonal/Golub-Kahan deferred debt into bounded oracle decisions or explicit
future-owner packages before corpus, reporting, performance, package, and
adoption work consume the oracle truth.

### Prerequisites from Previous Sprints

- Sprint 121 SVD/QR/rank fixture taxonomy complete
- Sprint 122 and Sprint 123 bounded SVD, QR, and partial-SVD external oracle
  lanes available
- Sprint 123 residual deferred debt, dependency order, and non-claim register
  available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Rank-Deficient QR Oracle Design | Design and implement or explicitly defer rank-deficient QR external evidence only after rank-threshold, nullspace, pseudoinverse, tolerance, skip, and minimum-norm policies are explicit. | 20 |
| 2 | QR Minimum-Norm Oracle Design | Add or explicitly defer QR minimum-norm external evidence under a behavior-specific owner spanning QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, and SuiteSparse scenarios. | 22 |
| 3 | QR Q-Basis and Economy Oracle Design | Add or explicitly defer Q-basis and economy external evidence only after sign, orientation, projection, subspace, and economy-shape semantics are defined. | 20 |
| 4 | Partial-SVD Vector and Subspace Semantics | Design and implement or explicitly defer partial-SVD vector and subspace external evidence with sign-invariant residuals, projection metrics, tolerance rules, and failure interpretation. | 28 |
| 5 | Partial-SVD Residual Semantics Batch | Decide or explicitly defer partial-SVD repeated-spectrum, clustered-spectrum, rank-deficient, convergence-budget, and low-rank optimality evidence without claiming broad partial-SVD parity. | 28 |
| 6 | Helper Ownership Follow-Through | Revisit minimum-norm helper migration and Bidiagonal/Golub-Kahan extraction with behavior-specific helper names and dedicated ownership that preserves scenario-local assertions. | 24 |
| 7 | Validation, Docs, and Claim Gate | Run affected focused checks and required quality gates, update maintainer evidence if ownership changes, and refresh solver-selection wording only if new evidence supports a public claim. | 24 |

### Deliverables

- rank-deficient QR oracle decision or bounded implementation
- QR minimum-norm oracle decision or bounded implementation
- QR Q-basis/economy oracle decision package
- partial-SVD vector/subspace decision or bounded implementation
- partial-SVD repeated/clustered/rank-deficient/convergence/low-rank decision
  package
- minimum-norm and Bidiagonal/Golub-Kahan helper ownership follow-through
- validation, maintainer evidence, and solver-selection claim gate package

**Total estimate:** ~166 hours

---

## Sprint 125: Rank-Deficient QR & Minimum-Norm Residual Evidence

**Duration:** 14 days (~164 hours)
**Goal:** Convert Sprint 124's rank-deficient QR and minimum-norm deferred debt
into behavior-specific evidence or explicit deferrals before broader corpus and
adoption work depend on these claims.

### Prerequisites from Previous Sprints

- Sprint 124 rank-deficient QR policy, minimum-norm behavior contract, and
  helper ownership decisions available
- Sprint 124 non-claim register preserved

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Deferred QR Dedupe Map | Map Sprint 124 deferred rank-deficient and minimum-norm work to existing Sprint 121-124 evidence so completed intake, policy, and fixture work is not duplicated. | 18 |
| 2 | Rank-Deficient Residual Evidence | Add or explicitly defer residual-only rank-deficient QR evidence after proving it increases trust beyond deterministic tests without implying nullspace or minimum-norm behavior. | 24 |
| 3 | Nullspace and Subspace Policy | Define sign, ordering, nullity, projection/subspace metric, and fixture-local tolerance rules before adding rank-deficient QR nullspace/subspace evidence. | 24 |
| 4 | Near-Rank-Deficient Threshold Evidence | Add or explicitly defer near-rank-deficient QR threshold evidence with threshold families, expected ranks, stability policy, and non-global interpretation. | 22 |
| 5 | SuiteSparse Rank-Deficient QR Evidence | Add or explicitly defer SuiteSparse rank-deficient QR evidence only after optional corpus, platform skip behavior, support tier, diagnostics, and claim boundaries are explicit. | 24 |
| 6 | Minimum-Norm Behavior Evidence | Add or explicitly defer QR minimum-norm COLAMD, fallback, rank-deficient, refinement, QR-vs-SVD-pseudoinverse, and SuiteSparse evidence under behavior-specific owners. | 32 |
| 7 | Validation and Claim Gate | Run affected focused checks and required quality gates, update evidence tables, and preserve all broad QR, nullspace, minimum-norm, backend, and corpus non-claims. | 20 |

### Deliverables

- Sprint 124 deferred QR/minimum-norm dedupe map
- rank-deficient residual evidence or explicit deferral
- nullspace/subspace policy package
- near-rank-deficient threshold decision package
- SuiteSparse rank-deficient QR decision package
- minimum-norm behavior evidence package
- validation and QR non-claim register update

**Total estimate:** ~164 hours

---

## Sprint 126: Rank-Deficient QR Residual Corpus & Minimum-Norm Follow-Through

**Duration:** 14 days (~166 hours)
**Goal:** Convert Sprint 125's remaining rank-deficient QR, nullspace,
threshold, SuiteSparse, and minimum-norm residual debt into bounded evidence or
explicit future-owner decisions before Q/economy, corpus-index, and adoption
work consume those truth boundaries.

### Prerequisites from Previous Sprints

- Sprint 125 residual QR, nullspace/subspace, threshold, SuiteSparse, and
  minimum-norm evidence policies complete
- Sprint 125 broad QR, nullspace, minimum-norm, backend, external-library, and
  corpus non-claims preserved

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Sprint 125 Residual Dedupe and Dependency Map | Map Sprint 125 residual deferred debt against completed Sprint 121-125 QR residual, nullspace, threshold, SuiteSparse, minimum-norm, and helper evidence so new work does not duplicate existing fixtures or decisions. | 16 |
| 2 | Compatible and Wide Residual Fixtures | Add or explicitly defer compatible zero-residual, dependent-row, and wide rank-deficient QR residual fixtures only after proving distinct trust value and preserving nullspace/minimum-norm non-claims. | 22 |
| 3 | Nullspace/Subspace Evidence Expansion | Add or explicitly defer multi-dimensional, wide-shape, near-threshold, dependent-row, and SuiteSparse QR nullspace/subspace evidence using projector or two-way projection metrics with pinned rank and nullity metadata. | 28 |
| 4 | Threshold Family Expansion | Add or explicitly defer scaled diagonal, perturbed duplicate-column, dependent-row, wide, and SuiteSparse QR threshold families with fixture-local expected ranks, diagnostics, and non-global rank-policy interpretation. | 24 |
| 5 | SuiteSparse Rank-Deficient QR Corpus Gate | Add or explicitly defer SuiteSparse rank-deficient QR corpus evidence only after expected-rank metadata, support tier, diagnostics, skip behavior, and validation requirements are explicit. | 24 |
| 6 | SuiteSparse and Underdetermined Minimum-Norm Evidence | Add or explicitly defer optional-large SuiteSparse, rank-deficient SuiteSparse, and larger underdetermined minimum-norm evidence with pinned residual, norm, rank, nullity, corpus metadata, and exact-value ownership where justified. | 30 |
| 7 | QR-vs-SVD Minimum-Norm Cross-Check Gate | Add or explicitly defer additional QR-vs-SVD minimum-norm fixtures only as bounded cross-checks with explicit fixture keys, tolerances, and non-oracle wording; preserve broad SVD-pseudoinverse non-claims. | 22 |

### Deliverables

- Sprint 125 residual dedupe and dependency map
- compatible, dependent-row, and wide residual fixture decision package
- nullspace/subspace projector evidence or explicit deferrals
- QR threshold-family evidence or explicit deferrals
- SuiteSparse rank-deficient QR corpus gate
- SuiteSparse and underdetermined minimum-norm evidence decision package
- QR-vs-SVD minimum-norm cross-check gate and non-claim update

**Total estimate:** ~166 hours

---

## Sprint 127: QR Q-Basis, Economy & Helper Ownership Follow-Through

**Duration:** 14 days (~166 hours)
**Goal:** Resolve the remaining Sprint 124 QR Q-basis/economy and helper
ownership debt in dependency order, preserving basis and helper semantics before
the corpus/index architecture consumes them.

### Prerequisites from Previous Sprints

- Sprint 125-126 rank-deficient QR metric, tolerance, corpus, and
  minimum-norm gates complete
- Sprint 124 Q-basis/economy semantic design and helper decision artifacts
  available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Q-Basis Evidence Policy Refresh | Confirm raw Q-column, sign, orientation, projection, economy-shape, skip, and corpus policies from Sprint 124 before extending evidence. | 18 |
| 2 | Raw Q-Column Evidence | Add or explicitly defer raw QR Q-column evidence only where basis orientation and fixture-local tolerance rules make equality meaningful. | 22 |
| 3 | Rank-Deficient Q/Nullspace Evidence | Add or explicitly defer rank-deficient Q/nullspace subspace evidence using the Sprint 125 nullity and projection/subspace metric policy. | 24 |
| 4 | Wide Economy and Sparse-Mode Evidence | Add or explicitly defer wide economy and sparse-mode Q/economy evidence with explicit shape, projection, and sparse-output interpretation. | 26 |
| 5 | SuiteSparse Q/Economy Evidence | Add or explicitly defer SuiteSparse Q/economy evidence after corpus availability, skip behavior, diagnostics, and support-tier wording are bounded. | 22 |
| 6 | Minimum-Norm Helper Movement | Revisit minimum-norm helper movement only with behavior-specific helper names and focused QR solve, COLAMD, SVD, and full quality validation. | 26 |
| 7 | Bidiagonal/Golub-Kahan Helper Extraction | Extract or explicitly defer Bidiagonal/Golub-Kahan helpers into a dedicated owner that preserves transpose, reconstruction, explicit `U`/`V`, wide skip, and QR-iteration semantics. | 28 |

### Deliverables

- Q-basis/economy evidence policy refresh
- raw Q-column decision or bounded implementation
- rank-deficient Q/nullspace decision or bounded implementation
- wide economy and sparse-mode decision package
- SuiteSparse Q/economy decision package
- minimum-norm helper movement package
- Bidiagonal/Golub-Kahan helper extraction package

**Total estimate:** ~166 hours

---

## Sprint 128: Partial-SVD Residual Expansion & Solver-Selection Claim Gate

**Duration:** 14 days (~166 hours)
**Goal:** Expand or explicitly defer Sprint 124 partial-SVD residual evidence
under dedicated metric policies, then refresh public solver-selection wording
only where earned.

### Prerequisites from Previous Sprints

- Sprint 124 partial-SVD vector/subspace semantics and residual scenario matrix
  available
- Sprint 125-127 QR and helper claim gates complete

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Partial-SVD Dedupe and Metric Map | Map completed Sprint 124 `partial_svd_vector_residual_diag6_k2` evidence against deferred rectangular, spectral, subspace, corpus, optimality, and convergence work. | 18 |
| 2 | Rectangular and Nonsymmetric Evidence | Add or explicitly defer rectangular and nonsymmetric rectangular partial-SVD residual evidence with shape-specific metrics and failure interpretation. | 24 |
| 3 | Repeated and Clustered Spectrum Evidence | Add or explicitly defer repeated-spectrum and clustered-spectrum partial-SVD evidence with subspace metrics instead of vector-equality claims. | 24 |
| 4 | Rank-Deficient Subspace Evidence | Add or explicitly defer rank-deficient partial-SVD subspace evidence with explicit rank, nullity, projection, and tolerance policies. | 24 |
| 5 | SuiteSparse and Low-Rank Optimality Evidence | Add or explicitly defer SuiteSparse corpus and low-rank optimality evidence with optional corpus behavior, diagnostics, and bounded claim language. | 28 |
| 6 | Convergence-Budget Evidence | Add or explicitly defer convergence-budget partial-SVD evidence with iteration, tolerance, and partial-result semantics that do not imply broad parity. | 24 |
| 7 | Solver-Selection Wording Gate | Refresh public solver-selection wording only if future evidence supports a user-facing claim beyond current workflow guidance; otherwise publish a no-update rationale. | 24 |

### Deliverables

- partial-SVD deferred-evidence dedupe map
- rectangular and nonsymmetric residual decision package
- repeated/clustered spectrum decision package
- rank-deficient subspace decision package
- SuiteSparse and low-rank optimality decision package
- convergence-budget decision package
- solver-selection wording update or explicit no-update rationale

**Total estimate:** ~166 hours

---

## Sprint 129: Numerical Corpus, Coverage Architecture & Report Indexes

**Duration:** 14 days (~168 hours)
**Goal:** Turn scattered numerical fixtures, coverage, benchmark, dead-code,
and guardrail outputs into a clearer recurring assurance architecture after the
Sprint 124-128 residual QR, partial-SVD, and helper claim gates are resolved.

### Prerequisites from Previous Sprints

- Sprint 118 templates complete
- Sprint 120-128 oracle taxonomy, external-reference decisions, and residual
  claim gates available

### Items

| Item # | Item Name | Item Description | Estimate (in hours) |
|---|---|---|---:|
| 1 | Corpus Inventory | Inventory Matrix Market fixtures, generated families, known matrices, external-reference scripts, expected failures, and Sprint 125-128 optional-corpus decisions. | 20 |
| 2 | Corpus Taxonomy | Define corpus tags for symmetry, definiteness, rank, conditioning, scale, sparsity pattern, ordering, solver family, optional availability, and support tier. | 24 |
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

## Sprint 130: Performance Sentinel & Backend Runtime Governance

**Duration:** 14 days (~168 hours)
**Goal:** Strengthen local performance and backend/runtime governance without
turning local measurements into portable performance claims.

### Prerequisites from Previous Sprints

- Sprint 129 report index model available
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

## Sprint 131: Package, ABI & Shared-Library Product Decision

**Duration:** 14 days (~168 hours)
**Goal:** Decide whether Epic 11 adds shared-library/dynamic ABI support or
explicitly preserves static-first support, then implement the selected product
contract.

### Prerequisites from Previous Sprints

- Epic 10 static-first package truth preserved
- Sprint 118 package/ABI residual owner map complete
- Sprint 129-130 corpus/report and performance-governance truth available

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

## Sprint 132: Cross-Platform Install, Windows Staged Lanes & CI Tier Follow-Through

**Duration:** 14 days (~168 hours)
**Goal:** Advance platform support where feasible and make Linux/macOS/Windows
install and staged validation tiers even more explicit.

### Prerequisites from Previous Sprints

- Sprint 131 package/ABI decision complete
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

## Sprint 133: Adoption Surface Simplification & Documentation Productization

**Duration:** 14 days (~164 hours)
**Goal:** Simplify the adoption surface after Epic 10 by separating first-use
guides from maintainer history and making compressed-first workflows easier to
find.

### Prerequisites from Previous Sprints

- Sprint 129 report index decisions available
- Sprint 131-132 package/platform truth stable

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

## Sprint 134: Final Integration, Competitive Recalibration & Epic 11 Closeout

**Duration:** 14 days (~164 hours)
**Goal:** Validate Epic 11 outcomes, compare them against the state-of-the-art
target, publish earned claims and non-claims, and close the epic with a
retrospective and post-epic handoff queue.

### Prerequisites from Previous Sprints

- Sprint 118-133 artifacts, validation, residuals, and support-surface
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
| 7 | Sprint and Epic Retrospectives | Write Sprint 134 and Epic 11 retrospectives plus final closeout handoff. | 30 |

### Deliverables

- final validation package
- final comparison and claim cleanup package
- post-Epic-11 residual queue
- Sprint 134 retrospective
- Epic 11 retrospective
- final handoff artifact

**Total estimate:** ~164 hours
