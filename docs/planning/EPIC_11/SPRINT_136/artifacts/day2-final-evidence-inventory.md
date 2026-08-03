# Sprint 136 Day 2 - Final Evidence Inventory

## Purpose

Day 2 inventories final Epic 11 evidence before validation design begins. The
goal is to identify owner surfaces and evidence families, not to validate them
or convert them into broader support claims.

This artifact separates evidence from claims. Source changes, tests, oracle
helpers, report indexes, benchmark outputs, package proofs, CI tiers,
documentation navigation, and residual queues are all inputs to later
validation and competitive recalibration, but none of them automatically widen
public support wording.

## Evidence Family Summary

| Evidence family | Primary owner surfaces | Sprint 136 use |
| --- | --- | --- |
| Source and public headers | `src/`, `include/`, `CMakeLists.txt`, `Makefile` | Confirm implementation and public API owner surfaces before validation. |
| Test ownership | `tests/*.c`, `tests/*_helpers.h`, `tests/*_external_dense_reference.py`, `tests/data/` | Map solver, helper, oracle, residual, fixture, and expected-failure evidence. |
| Examples and adoption proof | `examples/*.c`, `examples/README.md`, `examples/cmake_example/` | Identify maintained first-use and downstream-consumer examples. |
| Benchmarks and runtime reports | `benchmarks/*.c`, `benchmarks/README.md`, `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`, `scripts/large_matrix_guardrails.sh` | Inventory local measurement, sentinel, and guardrail evidence with non-claim boundaries. |
| Package/install proof | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in` | Inventory static-first package and downstream-consumer proof surfaces. |
| CI platform tiers | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Inventory reviewed, supplemental, and staged platform evidence. |
| Public docs | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/matrix_market.md`, `docs/algorithm.md`, `docs/algorithm_history.md`, `benchmarks/README.md`, `examples/README.md` | Inventory public claim and adoption surfaces before claim audit. |
| Maintainer docs | `docs/maintainer_guide.md`, Epic 11 planning artifacts | Inventory support-tier, validation, evidence, and non-claim owner surfaces. |
| Generated reports | `build/bench-reports/canonical/`, `build/bench-reports/sentinels/`, `build/bench-reports/large-matrix-guardrails/`, `build/deadcode/`, coverage outputs | Treat as generated evidence with freshness and support-tier context. |
| Residual queues | Sprint 118-135 retrospectives, Sprint 136 project-plan QR queue | Preserve future work, blockers, support tiers, and promotion criteria. |

## Sprint Evidence Inventory

| Sprint | Primary evidence contribution | Claim boundary carried into Sprint 136 |
| --- | --- | --- |
| 118 | Baseline validation inventory, residual conversion, product-truth map, hotspot owners, public claim drift audit, and closeout handoff. | Baseline evidence does not imply wider product truth without owner-specific proof. |
| 119 | Eigensolver source-boundary and proof-owner follow-through, selection lifting, shift-invert deferral, and validation parity package. | Eigensolver movement and selection evidence remain bounded; shift-invert and unsupported solver-selection claims stay deferred. |
| 120 | Direct/iterative oracle architecture, giant-test split, shared fixture architecture, direct/iterative split implementation, and cross-solver oracle pilot. | Oracle ownership is helper- and fixture-specific, not broad external-library parity. |
| 121 | SVD, QR, rank-deficient, pseudoinverse, low-rank, partial-SVD, and external-reference evidence expansion with helper ownership. | No broad LAPACK, SciPy, NumPy, SuiteSparse, vector/subspace parity, performance, package, platform, or state-of-the-art claim. |
| 122 | SVD/QR external oracle residual follow-through, QR external lane requirements, partial-SVD semantics, minnorm helper ownership, and solver-selection claim gate. | External oracle and solver-selection evidence remain scenario-specific. |
| 123 | Residual SVD/QR oracle, helper, partial-SVD evidence package, QR evidence implementation, and solver-selection claim closeout. | Helper movement and residual evidence do not create broad parity or public solver-selection readiness claims. |
| 124 | Residual QR, partial-SVD, helper oracle follow-through, QR rank/minnorm/basis semantics, partial-SVD residual scenario matrix, and validation claim gate. | QR/partial-SVD residuals require distinct trust value, metrics, and support-tier boundaries. |
| 125 | Rank-deficient QR and minimum-norm residual evidence, residual trust gate, nullspace/subspace policy, threshold-family decisions, SuiteSparse gates, and claim gate. | SuiteSparse and near-threshold work remains metadata- and support-tier bounded. |
| 126 | Rank-deficient QR residual corpus, minimum-norm follow-through, residual fixture trust policy, exact minnorm evidence, and validation claim-gate handoff. | Residual-only and minimum-norm lanes are bounded; additional SuiteSparse and QR-vs-SVD lanes remain deferred. |
| 127 | QR deferred evidence semantics, corpus follow-through, compatible/wide residual semantics, nullspace/subspace expansion, threshold-family follow-through, and optional-large gates. | Wide, optional-large, SuiteSparse, threshold, and subspace evidence require pinned semantics before promotion. |
| 128 | QR residual claim-gate closure, corpus semantics, remaining threshold family policy, optional-large minnorm policy, and cross-check helper validation. | Remaining QR residual debt moves to end-of-epic queue instead of another immediate sprint by default. |
| 129 | QR Q-basis, economy, sparse-mode, SuiteSparse Q/economy, minimum-norm helper ownership, and bidiagonal helper movement. | Q/economy evidence is product-specific and does not create raw basis, broad SuiteSparse, helper API, or platform/performance claims. |
| 130 | Partial-SVD residual expansion, rectangular/nonsymmetric residual evidence, clustered spectrum policy, rank-deficient subspace evidence, SuiteSparse corpus gate, and solver-selection claim closeout. | Partial-SVD and solver-selection evidence remains bounded by fixtures, residual metrics, convergence budget, and non-claim wording. |
| 131 | Numerical corpus taxonomy, report-index requirements, first generated index acceptance, coverage architecture, dead-code architecture, freshness policy, and residual assurance queue. | Report rows are traceability/freshness evidence, not broad correctness, coverage-completeness, performance, or release proof. |
| 132 | Performance sentinel and backend runtime governance, hot-path inventory, sentinel gap ranking, backend contract, report-index metadata validation, non-claim register, and runtime residual queue. | Benchmark/sentinel evidence remains local and bounded; backend/runtime claims need explicit proof and support-tier context. |
| 133 | Static-first package/ABI product decision, install/export contract, CMake rejection of shared builds, package metadata docs, package proofs, and package residual queue. | Shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, package-manager support, and static/shared selectors remain deferred. |
| 134 | Cross-platform install, Linux reviewed package-contract CI, macOS supplemental package confidence, Windows supplemental CMake install/downstream confidence, and staged Windows test follow-through. | Linux is reviewed for the package contract; macOS and Windows install/downstream lanes remain supplemental; staged Windows pthread/POSIX tests remain staged. |
| 135 | Adoption surface productization, compressed-first cookbook, algorithm reference/history split, report-index discovery, navigation alignment, and docs validation. | Documentation navigation does not create new solver behavior, package support, report schema, platform parity, or portable performance claims. |

## Source And Test Ownership Summary

| Area | Current owner surfaces | Evidence status for Sprint 136 |
| --- | --- | --- |
| Direct solvers and sparse LU | `src/sparse_lu*.c`, `include/sparse_lu*.h`, `tests/test_sparse_lu.c`, `tests/test_lu_csr.c`, `tests/lu_external_dense_reference.py`, direct solver helpers | Existing direct evidence includes internal tests, external dense-reference helpers, singular expected-failure coverage, and compressed sparse owner tests. |
| Iterative solvers | `src/sparse_iterative*.c`, `include/sparse_iterative.h`, `tests/test_iterative.c`, `tests/test_bicgstab*.c`, `tests/test_minres.c`, iterative helper headers | Existing evidence includes split iterative ownership, convergence and breakdown tests, block solver tests, and preconditioned examples. |
| QR and least squares | `src/sparse_qr*.c`, `include/sparse_qr.h`, `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_colamd.c`, `tests/test_qr_helpers.h`, `tests/qr_external_dense_reference.py` | Existing evidence covers internal QR invariants, rank, residual, least-squares, minimum-norm, Q/economy, sparse-mode, and bounded external fixtures. |
| SVD and partial SVD | `src/sparse_svd*.c`, `include/sparse_svd.h`, `tests/test_svd.c`, `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | Existing evidence covers deterministic SVD/partial-SVD fixtures, low-rank and residual metrics, pseudoinverse cross-checks, and bounded external-reference rows. |
| Eigensolvers | `src/sparse_eigs*.c`, `include/sparse_eigs.h`, `tests/test_eigs*.c`, `examples/example_eigs.c`, eigensolver benchmark sources | Existing evidence covers eigensolver source boundaries, selection lifting, LOBPCG/thick-restart tests, and shift-invert deferral. |
| Factorizations | Cholesky, LDLT, IC, ILU, bidiagonal, dense, CSR, reorder, graph, and matrix/vector source and test owners | Existing evidence spans factorization, reorder, graph, dense, CSR, and utility tests; Sprint 136 should inventory but not reclassify them without validation. |
| Test framework and fixtures | `tests/test_framework.h`, fixture helper headers, `tests/data/*.mtx`, external-reference Python helpers | Existing evidence includes fixture taxonomy and expected-failure/skip treatment from Sprint 131. |

## Oracle And External-Reference Evidence

| Oracle family | Owner surfaces | Boundary |
| --- | --- | --- |
| LU dense reference | `tests/lu_external_dense_reference.py`, direct LU tests | Fixture-specific solve and expected-failure evidence; no broad dense-library parity. |
| Cholesky dense reference | `tests/chol_external_dense_reference.py`, Cholesky/CSC tests | SPD and factorization-specific evidence; no broad backend or ecosystem claim. |
| LDLT dense reference | `tests/ldlt_external_dense_reference.py`, LDLT tests | Factorization-specific oracle evidence; no general indefinite solver parity. |
| QR dense reference | `tests/qr_external_dense_reference.py`, QR/least-squares tests | Scenario-specific QR, residual, rank, and minimum-norm evidence; no broad LAPACK/NumPy/SciPy or raw-basis parity. |
| SVD dense reference | `tests/svd_external_dense_reference.py`, SVD/partial-SVD tests | Singular-value, residual, subspace, and pseudoinverse cross-check evidence remains bounded by fixture semantics. |
| Cross-solver checks | `tests/test_cross_solver_oracle.c`, QR/SVD/minnorm cross-checks | Cross-checks are consistency evidence, not a global oracle or state-of-the-art comparison. |

## Report, Benchmark, And Runtime Evidence

| Evidence | Owner surfaces | Sprint 136 interpretation |
| --- | --- | --- |
| Canonical benchmark report | `make bench-canonical-report`, `scripts/bench_canonical_report.sh`, `benchmarks/README.md`, `build/bench-reports/canonical/` | Threshold-free local snapshot of maintained benchmark surface. |
| Performance sentinels | `make performance-sentinels`, `scripts/performance_sentinels.sh`, `scripts/wall_check.sh`, `build/bench-reports/sentinels/` | Local sentinel bundle; only the existing wall-check lane is thresholded. |
| Large-matrix guardrails | `make large-matrix-guardrails`, `scripts/large_matrix_guardrails.sh`, `build/bench-reports/large-matrix-guardrails/` | Reviewed/supplemental structural/report guardrail rows with explicit pass/fail/skip semantics. |
| Dead-code report | `make deadcode`, `make deadcode-report`, `make deadcode-check`, `scripts/deadcode_report.py`, `build/deadcode/` | Conservative report-completeness evidence, not zero-findings or removal-ready proof. |
| Coverage reports | `make coverage`, `make coverage-lcov`, `make coverage-gcovr`, CI supplemental coverage paths | Supplemental and tree-mutating; coverage percentage is not reviewed behavioral completeness. |
| Backend/runtime governance | Sprint 132 artifacts, benchmark docs, maintainer guide | Local runtime evidence with backend and support-tier context; no portable performance guarantee. |

No generated `build/bench-reports/**/index.tsv`, `sentinels.tsv`, or
`manifest.txt` files were present in this working tree during Day 2 inventory.
That means Day 5-7 validation must either generate fresh reports or record
their absence as a validation/design input.

## Package, Install, ABI, And Platform Evidence

| Evidence | Owner surfaces | Current status |
| --- | --- | --- |
| Static-first product decision | Sprint 133 closeout, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Maintained package shape is static archive install/export. |
| Make install and `pkg-config` proof | `tests/test_install.sh`, `sparse.pc.in`, Make install targets | Proves static install/uninstall and downstream `pkg-config` consumer behavior for the maintained surface. |
| CMake install/export proof | `tests/test_cmake_install.sh`, `examples/cmake_example/`, `cmake/SparseConfig.cmake.in` | Proves installed CMake package target and exact-version behavior for CMake consumers. |
| Static deferral proof | `scripts/static_package_deferral_check.sh`, `CMakeLists.txt` | Guards shared-build rejection and absence of unsupported package/ABI claims. |
| Linux reviewed package CI | `.github/workflows/ci.yml` | Strongest reviewed package-contract CI owner. |
| macOS supplemental package confidence | `.github/workflows/macos-ci.yml` | Supplemental static-first Make install/`pkg-config` and CMake install/export confidence. |
| Windows supplemental install/downstream confidence | `.github/workflows/windows-ci.yml` | Supplemental CMake-first install/downstream confidence, not reviewed install parity. |
| Windows staged tests | `CMakeLists.txt`, Windows CI, `tests/test_threads.c`, `tests/test_sprint4_integration.c`, `tests/test_fuzz.c` | Staged pending portability work and hosted proof. |

## Adoption And Documentation Evidence

| Surface | Owner role after Sprint 135 |
| --- | --- |
| `README.md` | Front-door adoption map, build/test/package/CI summary, generated report-index command context, and repository map. |
| `INSTALL.md` | Static-first install, downstream package use, platform support truth, and package proof locations. |
| `docs/tutorial.md` | First-use tutorial and documentation map. |
| `docs/cookbook.md` | Compressed-first task owner for direct, iterative, Matrix Market, SVD, eigensolver, and benchmark handoff workflows. |
| `docs/solver_selection.md` | Solver-choice guidance and adoption decision support. |
| `docs/matrix_market.md` | Matrix Market and compressed-input workflow guidance. |
| `docs/algorithm.md` | Current algorithm reference. |
| `docs/algorithm_history.md` | Historical measurement and implementation-decision appendix. |
| `docs/maintainer_guide.md` | Maintainer support-tier, validation ownership, evidence, benchmark/report, package/platform, and non-claim policy owner. |
| `examples/README.md` | Maintained example discovery and cookbook handoff. |
| `benchmarks/README.md` | Benchmark/report-index interpretation owner. |

## Initial Residual Grouping

| Residual group | Inputs | Day 12 handling expectation |
| --- | --- | --- |
| Deferred QR residual queue | Sprint 128 closeout, Sprint 129 no-reopen boundary, Sprint 136 project-plan QR queue | Publish with blockers, support tier, and promotion criteria; do not treat as completed implementation. |
| SuiteSparse and optional-large corpus work | Sprints 125-129 residual queues, Sprint 131 corpus taxonomy | Keep metadata-, support-tier-, and runtime-budget requirements explicit. |
| Report/corpus architecture | Sprint 131 generated index, coverage, dead-code, freshness, stale-report, and normalized schema residuals | Preserve row-meaning boundaries and owner surfaces. |
| Performance/runtime governance | Sprint 132 sentinel and runtime residuals | Keep benchmark rows local and support-tier aware. |
| Package/ABI/distribution | Sprint 133 package residual queue | Preserve shared-library, dynamic ABI, runtime-loader, package-manager, and selector deferrals. |
| Platform staging | Sprint 134 staged-exclusion and residual platform queues | Preserve macOS/Windows supplemental status and Windows staged-test blockers. |
| Documentation automation and reference density | Sprint 135 residual queue | Keep docs link automation and algorithm-reference simplification as future work. |
| Competitive and unsupported-claim cleanup | Day 8-11 planned work | Defer final public wording decisions until evidence classification and claim recalibration are complete. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every major Epic 11 evidence family has an owner surface. | Complete | Evidence family summary, source/test ownership summary, report/runtime table, package/platform table, and adoption documentation table. |
| Generated report and validation artifacts are separated from public claims. | Complete | Report/runtime inventory and package/platform inventory describe evidence with freshness/support-tier boundaries. |
| Residuals are visible before validation design begins. | Complete | Initial residual grouping names QR, corpus/report, performance/runtime, package/ABI, platform, documentation, and competitive cleanup groups. |

## Validation Notes

Day 2 changed only Sprint 136 planning artifacts. Required validation remains:

```bash
git diff --check
if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_136; then exit 1; fi
git diff --name-only -- '*.c' '*.h'
```
