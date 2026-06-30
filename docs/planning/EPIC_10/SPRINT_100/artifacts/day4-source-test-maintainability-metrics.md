# Sprint 100 Day 4 Source, Test & Maintainability Metrics

## Purpose

Day 4 captures measured maintainability evidence before Epic 10 extraction
work begins. This artifact should guide Sprint 106 and inform any earlier
sprint that touches large solver, graph, reorder, eigensolver, iterative, or
test ownership surfaces.

## Repository Size Signals

| metric | observed |
|---|---:|
| files under `src`, `include`, `tests`, `benchmarks`, `examples`, `scripts`, and `docs/planning` | `1,942` |
| lines across `src/*.c`, `include/*.h`, `tests/*.c`, `benchmarks/*.c`, and `examples/*.c` | `110,359` |
| library sources in source-list manifest | `42` |
| CMake reviewed tests from Day 2 | `54` |
| Make/CMake test-count parity from Day 2 | `54` vs `54` |

## Largest Test Owners

| rank | file | lines | maintainability signal |
|---:|---|---:|---|
| 1 | `tests/test_ldlt_csc.c` | `3,878` | largest proof owner; spans allocation, conversion, row adjacency, native kernels, supernodal paths, external references, and solves |
| 2 | `tests/test_integration.c` | `3,421` | broad cross-family lifecycle, callback, and integration owner |
| 3 | `tests/test_qr.c` | `3,234` | large direct-family proof owner with decomposition, solve, rank, sparse/dense, and refinement coverage |
| 4 | `tests/test_ldlt.c` | `2,977` | large LDLT wrapper/backend proof owner |
| 5 | `tests/test_etree.c` | `2,962` | large analysis/etree compatibility and solver lifecycle proof owner |
| 6 | `tests/test_graph.c` | `2,925` | graph/coarsen/bisect/FM/partition proof owner with substantial history |
| 7 | `tests/test_iterative.c` | `2,841` | large CG/GMRES/handle/matrix-free iterative proof owner |
| 8 | `tests/test_svd.c` | `2,766` | large SVD, partial SVD, pseudoinverse, low-rank, and vector proof owner |
| 9 | `tests/test_chol_csc.c` | `2,617` | large Cholesky CSC proof owner |
| 10 | `tests/test_chol_csc_supernodal.c` | `2,482` | large supernodal/dense-backend proof owner |
| 11 | `tests/test_reorder_nd.c` | `2,340` | nested-dissection long-pole proof owner |

## Largest Source Owners

| rank | file | lines | maintainability signal |
|---:|---|---:|---|
| 1 | `src/sparse_ldlt_csc.c` | `2,174` | largest implementation hotspot; direct CSC LDLT, native kernels, pivoting, supernodal support |
| 2 | `src/sparse_lu_csr.c` | `1,665` | large CSR LU implementation owner |
| 3 | `src/sparse_qr.c` | `1,563` | large QR/rank/solve owner |
| 4 | `src/sparse_ldlt.c` | `1,535` | large linked-list/backend dispatch LDLT owner |
| 5 | `src/sparse_eigs.c` | `1,534` | large eigensolver orchestration owner |
| 6 | `src/sparse_iterative.c` | `1,495` | large iterative solver owner |
| 7 | `src/sparse_matrix.c` | `1,355` | central matrix shell and lifecycle owner |
| 8 | `src/sparse_svd.c` | `1,319` | large full SVD owner |
| 9 | `src/sparse_chol_csc.c` | `1,279` | large Cholesky CSC implementation owner |
| 10 | `src/sparse_lu.c` | `1,042` | linked-list LU and one-shot compatibility owner |

## Public Header Size Signals

| file | lines | signal |
|---|---:|---|
| `include/sparse_iterative.h` | `773` | large options/handle/API contract surface |
| `include/sparse_eigs.h` | `651` | large eigensolver API contract surface |
| `include/sparse_matrix.h` | `614` | central matrix shell, compressed import/export, and lifecycle contract |
| `include/sparse_analysis.h` | `499` | explicit analysis/reorder/factor lifecycle contract |

## Comment and History Residue Samples

The Day 4 scan looked for `Sprint`, `temporary`, `fallback`, `compat`,
`legacy`, `TODO`, `FIXME`, `HACK`, `workaround`, `deprecated`, and `future`
across `src`, `include`, and `tests`.

Representative signals:

| area | examples | interpretation |
|---|---|---|
| public matrix shell | `include/sparse_matrix.h` compatibility-shell wording | product truth: compressed paths exist, but matrix-shell compatibility remains central |
| iterative API | `include/sparse_iterative.h` designated-init back-compat wording | compatibility is intentional but contributes to API explanation load |
| LU implementation | `src/sparse_lu.c` sprint-era one-shot temporary-copy comments | historical rationale remains in code paths that own compatibility restoration |
| graph/reorder internals | `src/sparse_graph*.c`, `src/sparse_reorder_nd.c`, `src/sparse_reorder_nd_internal.h` sprint and fallback comments | graph/ND code still carries extensive chronology and env-var compatibility rationale |
| LDLT CSC tests | `tests/test_ldlt_csc.c` many sprint/day section labels | giant proof owner still reads partly as sprint archaeology |
| graph tests | `tests/test_graph.c`, `tests/test_reorder_nd.c` sprint/day labels and long calibration comments | useful calibration history but high review burden |
| direct/eigs tests | `tests/test_ldlt.c`, `tests/test_eigs.c`, `tests/test_direct_csc_regression.c` sprint-era regression labels | proof ownership is strong but public history is dense |

Day 4 does not classify all such comments as wrong. Many preserve important
regression rationale. The maintainability target is to convert touched areas
from sprint chronology to durable product/rationale comments where doing so
reduces review cost.

## Source-List Ownership

The current library source ownership model has three synchronized surfaces:

| owner | file | role |
|---|---|---|
| manifest | `build-metadata/library_sources.txt` | canonical source membership list |
| Makefile | `Makefile` `LIB_SRCS` | Make build membership |
| CMake | `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | CMake build membership |
| checker | `scripts/check_library_sources.py` | compares manifest, Makefile, and CMake membership and ordering |

Focused check run:

```sh
python3 scripts/check_library_sources.py
```

Result:

```text
source-list-check: PASS (42 library sources)
```

## Ranked Maintainability Risks

| priority | risk | evidence | likely Epic 10 owner |
|---:|---|---|---|
| 1 | giant proof owners | 11 tests above 2,300 lines, largest 3,878 lines | Sprint 106 |
| 2 | direct/eigs/iterative implementation hotspots | 8 source files above 1,300 lines | Sprint 106 plus family sprints |
| 3 | chronology-heavy graph/ND surfaces | dense `Sprint` and calibration comments in graph/reorder code/tests | Sprint 105 and Sprint 106 |
| 4 | compatibility-shell explanation load | public matrix and solver surfaces carry compatibility wording | Sprint 101 and Sprint 107 |
| 5 | source membership drift risk after extraction | 42 library source entries mirrored across manifest, Makefile, and CMake | every extraction sprint |

## Day 4 Conclusion

The repository has a strong source-list drift guard and a clear reviewed test
surface, but maintainability pressure remains concentrated in large solver and
proof owners. Epic 10 should prioritize extraction that improves ownership and
failure localization over cosmetic line-count churn.

