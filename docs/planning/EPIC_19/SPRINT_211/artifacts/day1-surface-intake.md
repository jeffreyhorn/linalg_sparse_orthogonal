# Sprint 211 Day 1: Surface Intake

## Summary

Day 1 establishes the Sprint 211 baseline for reducing one selected large
review surface. No extraction begins on Day 1. The sprint will rank candidates
on Day 2 and select exactly one cluster only after the behavior-preservation
boundary is concrete.

## Sprint Scope Mapping

| Item | Day 1 owner interpretation |
| --- | --- |
| 211.1 Candidate Ranking | Rank large C tests, C implementations, helper headers, and Python tools by review risk, churn, ownership clarity, and review value. |
| 211.2 Cluster Boundary | Select one cluster and record no-behavior-change invariants before extraction. |
| 211.3 Extraction Design | Design helper/module split, registration impact, include dependencies, and guard strategy after selection. |
| 211.4 Extraction Implementation | Move selected code only after intake, ranking, boundary, design, and baseline validation. |
| 211.5 Guard And Test Coverage | Add or update helper ownership, registration, order, source-list, and focused regression tests for the selected cluster. |
| 211.6 Validation And Closeout | Run focused tests, guards, source-list/CMake parity where relevant, docs checks, and the full C quality gate if `.c` or `.h` files change. |

## Prior Pattern Inputs

| Prior artifact | Reusable Sprint 211 pattern |
| --- | --- |
| Epic 19 review | Large tests, production modules, and Python report tools remain hard to review in one pass. |
| Epic 19 todo track 5 | Select one cluster, define no-behavior-change invariants, extract ownership only where reviewability improves, add guard coverage, and validate. |
| Epic 18 residual queue E18-RQ-004 | Expected evidence includes candidate ranking, selected-cluster rationale, behavior-preservation notes, extraction diff, focused tests, guard coverage, and source-list/CMake parity when registration changes. |
| Sprint 201 closeout | Close exactly one selected helper surface and keep broad review-surface cleanup unclaimed. |
| Sprint 210 closeout | Protect newly added LDLT allocation-failure focused gate and registration guard if `tests/test_ldlt.c` becomes a candidate. |

## Current Large Surface Inventory

| File | Lines | Surface type | Day 1 risk tag |
| --- | ---: | --- | --- |
| `tests/test_ldlt_csc.c` | 3469 | C test | Largest current C test surface; high direct-solver review value. |
| `tests/test_ldlt.c` | 3444 | C test | Large after Sprint 210; selection must preserve focused LDLT allocation-failure proof. |
| `tests/test_etree.c` | 3306 | C test | Large symbolic/etree test surface; protect Sprint 200 proof boundaries. |
| `tests/test_integration.c` | 3279 | C test | Broad cross-solver test owner; high risk of accidental broad refactor. |
| `tests/test_qr.c` | 3040 | C test | Large QR surface; avoid duplicating prior QR helper extraction. |
| `tests/test_iterative.c` | 2929 | C test | Large iterative behavior surface with convergence/callback sensitivity. |
| `tests/test_graph.c` | 2764 | C test | Large graph/reorder surface with heuristic behavior and slower validation. |
| `tests/test_normalize_report_index.py` | 2743 | Python test/tooling | Large schema/CLI validation surface. |
| `tests/test_svd.c` | 2658 | C test | Still large after Sprint 201; remaining SVD clusters must preserve helper guard ownership. |
| `tests/test_chol_csc.c` | 2554 | C test | Large direct-solver test surface with likely helper/fixture candidates. |
| `tests/test_chol_csc_supernodal.c` | 2504 | C test | Large supernodal surface with helper extraction precedent. |
| `scripts/run_external_comparison.py` | 2306 | Python tool | Large comparison implementation with artifact semantics. |
| `tests/test_reorder_nd.c` | 2304 | C test | Large reorder surface with long-running focused tests. |
| `tests/test_eigs.c` | 2155 | C test | Large eigensolver test surface with numerical/vector semantics. |
| `src/sparse_ldlt_csc.c` | 2095 | C implementation | Largest production source; source-list and behavior risk are high. |
| `tests/test_colamd.c` | 2017 | C test | Large ordering surface; candidate if a cohesive cluster is found. |
| `scripts/normalize_report_index.py` | 1893 | Python tool | Large report-index implementation below 2000 lines but review-heavy. |
| `tests/test_svd_partial_helpers.h` | 1519 | Test helper header | Large helper surface; header-only guard and stale-binary concerns apply. |

## Initial Candidate Set

Day 2 should rank these candidate families first:

1. `tests/test_ldlt_csc.c` selected helper or fixture cluster.
2. `tests/test_ldlt.c` selected cluster outside the Sprint 210 allocation proof.
3. `tests/test_etree.c` selected helper cluster outside the Sprint 200 proof.
4. Remaining `tests/test_qr.c` cluster outside prior external-reference work.
5. `tests/test_integration.c` selected fixture cluster only if it avoids broad
   cross-solver refactoring.
6. `tests/test_chol_csc.c` or `tests/test_chol_csc_supernodal.c` helper
   cluster.
7. Graph/reorder cluster from `tests/test_graph.c` or `tests/test_reorder_nd.c`.
8. Python report tooling cluster from `scripts/run_external_comparison.py`,
   `scripts/normalize_report_index.py`, or `tests/test_normalize_report_index.py`.
9. Production extraction from `src/sparse_ldlt_csc.c` only if Day 2 finds the
   behavior and registration risk justified.

## No-Behavior-Change Boundary

- Preserve public APIs, public headers, ABI assumptions, status codes,
  diagnostics, fixture values, random seeds, tolerances, skip behavior, and
  generated artifact formats.
- Preserve test names, registration counts, `RUN_TEST(...)` order, focused gate
  behavior, assertion intent, cleanup order, and process-global state
  restoration.
- Prefer family-local helper extraction when it avoids production source-list
  churn.
- If Make/CMake/source-list registration changes are needed, update guards and
  validation evidence in the same sprint.
- Do not add performance, package, platform, release, public API, ABI, broad
  review-surface, external-library parity, or state-of-the-art claims.

## Source And Registration Owners

| Owner | Sprint 211 use |
| --- | --- |
| `build-metadata/library_sources.txt` | Required owner if production source extraction occurs. |
| `Makefile` | Required owner for new source, test, helper prerequisite, or guard targets. |
| `CMakeLists.txt` | Required owner if compiled source or CTest registration changes. |
| `make source-list-check` | Source registration confidence gate when source lists change. |
| `make quality-review-cmake-compile` | CMake parity gate when registration changes. |
| `make format && make lint && make test` | Required once `.c` or `.h` files change. |

## Day 1 Outcome

Day 1 closes the intake and baseline task. Day 2 should rank the measured
candidate surfaces and produce a preferred shortlist. Day 3 should select one
cluster only after behavior-preservation boundaries are concrete.

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 1 validation command.
