# Sprint 201 Day 1: Large Surface Intake

## Summary

Day 1 establishes the Sprint 201 baseline for reducing one selected large
review surface. The branch is `sprint-201`, and no code extraction begins
until Day 2 ranks candidates and Day 3 selects exactly one cluster with
behavior-preservation boundaries.

## Sprint Scope Mapping

| Item | Day 1 owner interpretation |
| --- | --- |
| 201.1 Candidate Ranking | Rank current >2000-line source/test surfaces by review risk, ownership clarity, and extraction feasibility. |
| 201.2 Cluster Selection | Select one cluster and record behavior-preservation invariants before extraction. |
| 201.3 Helper Or Module Extraction | Extract helpers or source module boundaries only where reviewability improves. |
| 201.4 Ownership Guard | Add or update guard scripts/tests to keep the selected extracted surface registered and reviewable. |
| 201.5 Focused Regression | Run focused behavior-preservation tests and add narrow regressions only for extraction safety. |
| 201.6 Validation And Docs | Run source-list/CMake checks as applicable, full C quality gate for `.c`/`.h` changes, and maintainer docs updates. |

## Prior Pattern Inputs

| Prior artifact | Reusable Sprint 201 pattern |
| --- | --- |
| Sprint 193 retrospective | Reduce exactly one selected review surface, preserve behavior, add guard coverage, and retain broad review-surface residuals. |
| Sprint 193 Day 1 intake | Measure large files, function density, test registration density, helper surfaces, source-list owners, and no-behavior-change rules. |
| Sprint 193 Day 2 ranking | Prefer cohesive test/helper extraction over production source movement unless production movement has a clear selected boundary. |
| Sprint 200 closeout | Protect newly added selected symbolic LU reliability proof if `tests/test_etree.c` becomes a candidate. |
| Existing helper guards | `ldlt-csc-helper-guard` and `qr-external-ref-helper-guard` show how to guard helper/proof-owner boundaries. |

## Current Large Source/Test Inventory

| File | Lines | Function-density signal | RUN_TEST count | Day 1 risk tag |
| --- | ---: | ---: | ---: | --- |
| `tests/test_ldlt_csc.c` | 3469 | 110 | 100 | Largest current C test surface; prior extraction lowers obvious payoff unless a new untouched cluster is isolated. |
| `tests/test_etree.c` | 3306 | 118 | 107 | High helper density and recent Sprint 200 proof additions; protect selected symbolic LU reliability tests. |
| `tests/test_integration.c` | 3279 | 54 | 58 | Broad cross-solver scope; high risk of accidental multi-cluster refactor. |
| `tests/test_qr.c` | 3040 | 71 | 79 | Still large after Sprint 193; avoid duplicating external-reference extraction. |
| `tests/test_svd.c` | 3029 | 85 | 114 | Large numerical surface with likely cohesive helper clusters. |
| `tests/test_ldlt.c` | 3006 | 92 | 89 | Large direct-solver test surface; preserve tolerance/status behavior. |
| `tests/test_iterative.c` | 2929 | 86 | 85 | Convergence and handle-lifetime behavior make extraction sensitive. |
| `tests/test_graph.c` | 2764 | 65 | 61 | Graph and heuristic behavior can be deterministic but subtle. |
| `tests/test_chol_csc.c` | 2554 | 108 | 92 | High helper density and direct-solver relevance. |
| `tests/test_chol_csc_supernodal.c` | 2504 | 72 | 62 | Focused supernodal surface with helper-header precedent. |
| `tests/test_normalize_report_index.py` | 2419 | N/A | N/A | Large Python test surface outside the primary C review-surface examples. |
| `scripts/run_external_comparison.py` | 2306 | N/A | N/A | Large generated-evidence implementation surface; semantic risk is higher. |
| `tests/test_reorder_nd.c` | 2304 | N/A | N/A | Large reorder test surface with deterministic behavior risk. |
| `tests/test_eigs.c` | 2155 | N/A | N/A | Large eigensolver test surface with numerical behavior risk. |
| `src/sparse_ldlt_csc.c` | 2095 | 26 | N/A | Only current >2000-line production C source; high behavior and registration risk. |
| `tests/test_colamd.c` | 2017 | N/A | N/A | Large ordering surface; candidate only if a clean cluster is found. |

## Initial Candidate Set

Day 2 should rank these candidate families first:

1. `tests/test_ldlt_csc.c` selected helper cluster.
2. `tests/test_etree.c` selected fixture/helper cluster outside the new Sprint
   200 symbolic LU allocation-failure owner proof.
3. `tests/test_svd.c` selected helper or dense-reference cluster.
4. Remaining `tests/test_qr.c` cluster outside Sprint 193 external-reference
   extraction.
5. `tests/test_chol_csc.c` helper/fixture cluster.
6. `tests/test_chol_csc_supernodal.c` helper/fixture cluster.
7. `tests/test_ldlt.c` selected direct-solver helper cluster.
8. `tests/test_integration.c` selected fixture cluster only if it does not
   become broad cross-solver refactoring.
9. `src/sparse_ldlt_csc.c` production extraction only if Day 2 finds the
   behavior and registration risk justified.

## No-Behavior-Change Boundary

- Preserve public APIs, public headers, ABI, status codes, diagnostics,
  numerical behavior, fixture values, random seeds, tolerances, and skip
  behavior.
- Preserve test names, `RUN_TEST(...)` order, assertion intent, cleanup order,
  and process-global state restoration.
- Prefer family-local helper extraction where it avoids library source-list
  churn.
- If Make/CMake/source-list registration changes are needed, update guards and
  validation evidence in the same sprint.
- Do not add performance, package, platform, release, public API, ABI, broad
  review-surface, or state-of-the-art claims.

## Source And Registration Owners

| Owner | Sprint 201 use |
| --- | --- |
| `build-metadata/library_sources.txt` | Required owner if production source extraction occurs. |
| `Makefile` | Required owner for new source, test, or guard targets. |
| `CMakeLists.txt` | Required owner if compiled source or test registration changes. |
| `make source-list-check` | Source registration confidence gate. |
| `make quality-review-cmake-compile` | CMake parity gate if registration changes. |
| `make format && make lint && make test` | Required once `.c` or `.h` files change. |

## Day 1 Outcome

Day 1 closes the intake and baseline task. Day 2 should rank the measured
candidate surfaces and produce a preferred shortlist. Day 3 should select one
cluster only after behavior-preservation boundaries are concrete.

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 1 validation command.
