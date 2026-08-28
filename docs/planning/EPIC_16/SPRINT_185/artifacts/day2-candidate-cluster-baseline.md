# Sprint 185 Day 2: Candidate Cluster Baseline

## Purpose

Measure candidate review surfaces before selection so Day 3 can choose one
large cluster with high review cost, bounded behavior risk, and a clear
extraction path.

## Scoring Method

| Score | Meaning |
| --- | --- |
| Review cost 1 | Low review burden; size or responsibility split is not a sprint-level concern. |
| Review cost 5 | High review burden; line count, mixed responsibilities, fixture repetition, or function count materially slows review. |
| Refactor risk 1 | Low behavior and registration sensitivity. |
| Refactor risk 5 | High sensitivity from solver behavior, numerical tolerance, allocation-failure proof ownership, registration churn, or cross-family scope. |

Day 3 should prefer a candidate with high review cost, lower refactor risk, a
clear helper seam, and focused validation available before and after movement.

## Candidate Baseline

| Candidate | Lines | Static/function count | Registration owner | Responsibility cluster | Helper/seam evidence | Focused validation | Review cost | Refactor risk | Day 2 disposition |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| `tests/test_ldlt_csc.c` | 3915 | 130 | Make/CMake `test_ldlt_csc` | LDLT CSC allocation, row adjacency, supernode, native pivot, solve, inertia, dense reference, and KKT-style coverage. | Existing row-adjacent implementation split in `src/sparse_ldlt_csc_rowadj.c`; likely test seams around dense/symmetric/KKT builders, row-adjacent assertions, and native-pivot fixtures. | `make build/test_ldlt_csc && ./build/test_ldlt_csc`; full C gate if C/H files change. | 5 | 3 | Strongest Day 3 candidate. |
| `tests/test_svd.c` | 3029 | 90 | Make/CMake `test_svd` | Golub-Kahan, full SVD, partial SVD, rank, pseudoinverse, low-rank approximation, condition estimates, and dense-reference checks. | Existing helper precedent in `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h`, `tests/test_svd_partial_shared_helpers.h`, and partial corpus ownership. | `make build/test_svd && ./build/test_svd`; run partial-SVD focused tests if touched; full C gate if C/H files change. | 4 | 3 | Strong alternate if the LDLT CSC seam proves too coupled. |
| `tests/test_graph.c` | 2764 | 68 | Make/CMake `test_graph` | Graph construction, coarsening, bisection, FM refinement, partition checks, and large-matrix guardrails. | Existing `tests/test_graph_fixtures.h`; possible fixture-builder or environment-helper extraction. | `make build/test_graph && ./build/test_graph`; include large-matrix guardrails if relevant; full C gate if C/H files change. | 4 | 3 | Viable fallback, but graph/FM environment interactions need containment. |
| `tests/test_qr.c` | 3970 | 83 | Make/CMake `test_qr` | QR factorization, rank, nullspace, projectors, reorder, and refinement coverage. | Existing `tests/test_qr_helpers.h` and `tests/test_qr_corpus.c`; prior proof-owner pattern exists. | `make build/test_qr && ./build/test_qr`; QR corpus/solve tests if touched; full C gate if C/H files change. | 5 | 4 | Defer unless the top candidates fail; recent QR work increases review sensitivity. |
| `tests/test_integration.c` | 3279 | 58 | Make/CMake `test_integration` | Cross-solver lifecycle, progress callback, refactor, matrix shell, and end-to-end integration coverage. | Existing `tests/test_integration_fixtures.h`, but the ownership boundary is broad. | `make build/test_integration && ./build/test_integration`; full C gate if C/H files change. | 5 | 5 | Defer because cross-solver scope makes behavior drift harder to bound. |
| `tests/test_ldlt.c` | 3006 | 95 | Make/CMake `test_ldlt` | Public LDLT behavior, reorder, KKT cases, refinement, condition estimates, backend dispatch, and dense helpers. | Possible seams around KKT builders, dense fixtures, and backend helper assertions. | `make build/test_ldlt && ./build/test_ldlt`; backend dispatch if touched; full C gate if C/H files change. | 4 | 4 | Candidate, but less clear than `tests/test_ldlt_csc.c`. |
| `tests/test_etree.c` | 2962 | 111 | Make/CMake `test_etree` | Elimination tree, postorder, column count, symbolic analysis, and writeback coverage. | Likely matrix-fixture and structural-assertion seams. | `make build/test_etree && ./build/test_etree`; full C gate if C/H files change. | 4 | 3 | Viable but lower direct-solver priority. |
| `tests/test_iterative.c` | 2929 | 94 | Make/CMake `test_iterative` | Iterative solvers, allocation-failure proof lane, preconditioners, and solver helper coverage. | Existing `tests/test_iterative_handle_helpers.h` and `tests/test_solver_helpers.h`. | `make build/test_iterative && ./build/test_iterative`; `make iterative-allocation-failure-gate`; full C gate if C/H files change. | 4 | 5 | Defer because allocation-failure proof ownership is sensitive. |
| `tests/test_chol_csc.c` | 2554 | 111 | Make/CMake `test_chol_csc` | Cholesky CSC analysis-backed and publish-back proof surface. | Existing `tests/test_chol_csc_supernodal_helpers.h`; possible fixture/assertion seams. | `make build/test_chol_csc && ./build/test_chol_csc`; full C gate if C/H files change. | 3 | 3 | Viable but lower impact than the top shortlist. |
| `src/sparse_ldlt_csc.c` | 2095 | n/a | Library source lists | LDLT CSC implementation path. | Source extraction would require new implementation ownership and public/internal declaration review. | `make source-list-check`, affected focused tests, and full C gate. | 4 | 5 | Defer; implementation extraction is higher risk than test helper extraction. |
| `scripts/run_external_comparison.py` | 2094 | n/a | Python tooling | Selected external comparison runner and report generation flow. | Possible parser/report helper seams, but tooling is outside the default solver/test target. | `python -m py_compile scripts/run_external_comparison.py`; focused report tests. | 3 | 4 | Defer unless Sprint 185 pivots to tooling. |
| `tests/test_normalize_report_index.py` | 1861 | n/a | Python test tooling | Report index normalization, drift diagnostics, and selected-report freshness checks. | Possible fixture/corpus helper seams. | Focused Python test run and report-index validation commands. | 3 | 3 | Defer because it is not the strongest solver/test surface. |
| `docs/maintainer_guide.md` | 1761 | n/a | Documentation | Maintainer evidence, command, and non-claim guidance. | Documentation split is possible, but not central to Sprint 185. | `git diff --check`. | 2 | 2 | Defer. |

## Registration Map

| Extraction type | Registration impact | Required validation |
| --- | --- | --- |
| Test-only helper header | No new test binary when included by an existing registered test. | Focused test binary plus `make format && make lint && make test` if any `.c` or `.h` file changes. |
| New C proof-owner test binary | Add to `Makefile` `TEST_SRCS` and `CMakeLists.txt` with `add_sparse_test(...)`. | Focused test, `make quality-review-cmake-compile`, and full C gate. |
| New library source file | Add to `Makefile` `LIB_SRCS`, `CMakeLists.txt`, and `build-metadata/library_sources.txt`. | `make source-list-check`, affected focused tests, and full C gate. |
| Python tooling extraction | No C registration unless C/H files also change. | `python -m py_compile ...` plus focused Python/report tests. |

## Day 2 Shortlist

1. `tests/test_ldlt_csc.c`: top candidate because it combines the second
   largest line count, the highest function/static count among current test
   candidates, direct-solver review value, and plausible helper seams.
2. `tests/test_svd.c`: primary alternate because it already has helper-header
   precedent and focused test ownership, while still providing meaningful
   review-surface reduction.
3. `tests/test_graph.c`: viable fallback if Day 3 needs a lower line-count
   cluster with clearer existing fixture ownership.

## Day 3 Handoff

Day 3 should inspect `tests/test_ldlt_csc.c` in detail before selecting the
final cluster. The selection artifact should identify the exact helper or
fixture blocks to move, decide whether extraction stays header-only or creates
a new proof-owner binary, and list no-behavior-change validation before any
mechanical movement begins.

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.
