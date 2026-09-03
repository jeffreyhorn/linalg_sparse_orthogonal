# Sprint 193 Day 1: Review-Surface Intake

## Summary

Day 1 established the Sprint 193 baseline for reducing one selected large
review surface. The repository is on branch `sprint-193`, and the only
implementation decision made today is that no code will move until Day 2 ranks
and selects a single cluster from current repository evidence.

## Sprint Scope Mapping

| Item | Day 1 owner interpretation |
| --- | --- |
| 193.1 Candidate Ranking | Use current line counts, helper density, algorithm risk, source-list coupling, tests, and user-facing importance to rank candidates. |
| 193.2 Cluster Selection | Select exactly one cluster and write no-behavior-change invariants before extraction. |
| 193.3 Extraction Design | Define helper/source boundaries, cleanup ownership, process-global restoration, registration behavior, and guard scope. |
| 193.4 Implementation | Move only the selected cluster with behavior-preserving edits. |
| 193.5 Guard And Docs | Add a focused ownership guard and maintainer guidance for the new boundary. |
| 193.6 Validation | Run focused tests, source-list checks, full C quality gate, and CMake parity if registration or source lists change. |

## Prior Pattern Inputs

| Prior artifact | Reusable Sprint 193 pattern |
| --- | --- |
| Sprint 187 gap ranking | Selected review-surface reduction is feasible only if limited to one cluster. |
| Sprint 185 selected-cluster decision | Pick one proof-owner surface, write invariants first, and defer broad alternatives. |
| Sprint 185 retrospective | Header-only family-local helper extraction can reduce review surface without new test binaries or library source-list churn. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Guard proof-owner registration, helper existence, include ownership, and absence from unintended Make/CMake/library registration. |
| `scripts/check_library_sources.py` | Production source changes require manifest, Makefile, and CMake agreement. |

## Large Source/Test Inventory

| File | Lines | Function-density signal | Day 1 risk tag |
| --- | ---: | ---: | --- |
| `tests/test_qr.c` | 3970 | 78 | Largest remaining test surface; high value, high QR behavior sensitivity. |
| `tests/test_ldlt_csc.c` | 3469 | 110 | Still large but recently extracted in Sprint 185; likely lower incremental payoff. |
| `tests/test_integration.c` | 3279 | 54 | Broad cross-solver scope; high risk of accidental multi-cluster refactor. |
| `tests/test_svd.c` | 3029 | 85 | Large, helper-friendly numerical surface with existing helper precedent. |
| `tests/test_ldlt.c` | 3006 | 92 | Large direct-solver surface; tolerance and status behavior must be preserved. |
| `tests/test_etree.c` | 2962 | 102 | High helper density; likely fixture/helper seams but ranking needed. |
| `tests/test_iterative.c` | 2929 | 86 | Allocation, convergence, and handle lifetime risks. |
| `tests/test_graph.c` | 2764 | 65 | Existing fixture precedent but graph/FM behavior can be subtle. |
| `tests/test_chol_csc.c` | 2554 | 108 | High helper density and direct-solver relevance. |
| `tests/test_chol_csc_supernodal.c` | 2504 | 72 | Focused supernodal surface with existing helper-header precedent. |
| `src/sparse_ldlt_csc.c` | 2095 | 26 | Production extraction candidate but high behavior/source-list risk. |
| `src/sparse_lu_csr.c` | 1594 | 9 | Production extraction candidate; defer unless test candidates fail. |
| `src/sparse_ldlt.c` | 1535 | 8 | Production direct-solver surface; defer by default. |
| `src/sparse_iterative.c` | 1503 | 11 | Production iterative surface; convergence risk. |
| `src/sparse_qr.c` | 1448 | 9 | Production QR surface; high public behavior risk. |

## Initial Candidate Set

Day 2 should rank these first:

1. `tests/test_qr.c` selected helper cluster.
2. `tests/test_svd.c` selected helper cluster.
3. `tests/test_chol_csc_supernodal.c` selected helper cluster.
4. `tests/test_chol_csc.c` selected helper cluster.
5. `tests/test_etree.c` selected fixture/helper cluster.
6. `tests/test_integration.c` selected fixture cluster.
7. `tests/test_iterative.c` selected helper cluster.
8. Further `tests/test_ldlt_csc.c` extraction.
9. One production `src/*.c` extraction only if test-only candidates are
   unsuitable.

## No-Behavior-Change Boundary

- Preserve public APIs, public headers, status codes, diagnostics, and solver
  behavior.
- Preserve test names, `RUN_TEST(...)` ordering, fixture values, random seeds,
  numerical tolerances, skip behavior, and assertion intent.
- Preserve cleanup ownership, allocation/free order, temporary-file handling,
  external-helper behavior, and process-global state restoration.
- Prefer family-local header-only helper extraction when it avoids new
  Make/CMake registration churn.
- If a new compiled source or test binary is required, update Make, CMake, and
  source-list/registration guards together.
- Do not add performance, correctness-expansion, platform, package, ABI,
  release, or state-of-the-art claims.

## Source-List Owners

| Owner | Required use |
| --- | --- |
| `build-metadata/library_sources.txt` | Authoritative library-source manifest. |
| `Makefile` `LIB_SRCS` | Must match the library manifest if production sources change. |
| `CMakeLists.txt` `add_library(...)` | Must match the library manifest if production sources change. |
| `Makefile` `TEST_SRCS` | Required for new or changed test proof-owner binaries. |
| `CMakeLists.txt` `add_sparse_test(...)` | Required for new or changed CMake test proof-owner binaries. |
| `make source-list-check` | Required for library source-list validation. |
| `make quality-review-cmake-compile` | Required if test registration or CMake source-list parity changes. |
| `make format && make lint && make test` | Required after any `.c` or `.h` implementation/test extraction. |

## Day 1 Outcome

Day 1 closes the intake and baseline task. The next step is Day 2 candidate
ranking, where exactly one cluster should be selected or explicitly deferred
based on measured review payoff and behavior-preservation risk.

## Validation

Commands run:

```sh
git status --short --branch
sed -n '235,267p' docs/planning/EPIC_17/PROJECT_PLAN.md
sed -n '1,90p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
rg --files src include tests benchmarks scripts -g '*.c' -g '*.h' -g '*.py' -g '*.sh' | xargs wc -l | sort -nr | head -n 45
for f in tests/test_qr.c tests/test_ldlt_csc.c tests/test_integration.c tests/test_svd.c tests/test_ldlt.c tests/test_etree.c tests/test_iterative.c tests/test_graph.c tests/test_chol_csc.c tests/test_chol_csc_supernodal.c src/sparse_ldlt_csc.c src/sparse_lu_csr.c src/sparse_ldlt.c src/sparse_iterative.c src/sparse_qr.c; do printf '%s\t' "$f"; rg -n '^(static[[:space:]]+)?[A-Za-z_][A-Za-z0-9_ *]+[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\([^;]*\)[[:space:]]*\{' "$f" | wc -l; done
find tests -maxdepth 1 -name '*helpers*.h' -o -name '*fixture*.h' -o -name '*oracle*.h' | sort | xargs wc -l | sort -nr
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_187/artifacts/day5-gap-ranking-and-feasibility.md
sed -n '1,220p' docs/planning/EPIC_16/SPRINT_185/artifacts/day3-selected-cluster-decision.md
sed -n '1,220p' scripts/check_ldlt_csc_helper_guard.sh
sed -n '1,220p' scripts/check_library_sources.py
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
