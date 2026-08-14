# Day 2 Code And Public Surface Inventory

## Scope

Day 2 captures the current implementation and public API shape for Sprint 157.
This is a baseline artifact only. It does not change source code, select Epic
14 implementation targets, or claim correctness from size/count evidence.

## Inventory Commands

| Purpose | Command |
| --- | --- |
| Branch/worktree state | `git status --short --branch` |
| File counts by owner directory | `find <dir> -type f | wc -l` over `src`, `include`, `tests`, `benchmarks`, `examples`, and `scripts` |
| Language/surface counts | `find src tests benchmarks examples -name '*.c'`, `find include src tests -name '*.h'`, `find scripts tests -name '*.py'`, and `find scripts tests -name '*.sh'` |
| Largest owner files | `find src include tests benchmarks examples scripts -type f ... | xargs wc -l | sort -nr | head -60` |
| Public headers | `find include -maxdepth 1 -type f -name '*.h' | sort` |
| Build source-list ownership | `rg` over `Makefile`, `CMakeLists.txt`, and `scripts/check_library_sources.py` |
| Source-list consistency | `python3 scripts/check_library_sources.py` |

## Directory Baseline

| Directory | File count | Notes |
| --- | ---: | --- |
| `src/` | 69 | Implementation and internal headers. |
| `include/` | 19 | 18 checked-in public `.h` headers plus `sparse_version.h.in` template. |
| `tests/` | 122 | C tests, helper headers/scripts, corpus metadata, schemas, and install proof scripts. |
| `benchmarks/` | 19 | Benchmark C drivers, helper headers, and benchmark documentation. |
| `examples/` | 18 | Maintained examples, helper header, and examples README. |
| `scripts/` | 16 | Quality, generated-report, comparison, package-deferral, benchmark, and guardrail scripts. |

## Language And Surface Counts

| Surface | Count | Interpretation |
| --- | ---: | --- |
| C sources in `src/`, `tests/`, `benchmarks/`, and `examples/` | 140 | Includes 49 library implementation sources, 59 top-level C tests, 16 benchmark drivers, and 14 example programs. |
| Headers in `include/`, `src/`, and `tests/` | 51 | Includes public headers, internal implementation headers, and test helper headers. |
| Python scripts/helpers in `scripts/` and `tests/` | 12 | Includes corpus, report-index, comparison, dense-reference, and source-list tooling. |
| Shell scripts in `scripts/` and `tests/` | 12 | Includes install, package, CI, wall-check, benchmark/report, and deferral proof scripts. |

## Public Header Surface

The checked-in public `.h` headers under `include/` are:

| Header | Surface |
| --- | --- |
| `include/sparse_analysis.h` | Analyze/factor/refactor lifecycle and direct-solver reuse. |
| `include/sparse_bidiag.h` | Bidiagonalization support. |
| `include/sparse_cholesky.h` | Cholesky factorization and solve options/results. |
| `include/sparse_csr.h` | CSR construction and compressed sparse helpers. |
| `include/sparse_dense.h` | Dense helper operations. |
| `include/sparse_eigs.h` | Symmetric eigensolver options, backends, handles, and results. |
| `include/sparse_ic.h` | Incomplete Cholesky preconditioner API. |
| `include/sparse_ilu.h` | ILU preconditioner API. |
| `include/sparse_iterative.h` | Iterative solvers, reusable handles, matrix-free callbacks, and diagnostics. |
| `include/sparse_ldlt.h` | LDL^T factorization, solve, backend, and telemetry contracts. |
| `include/sparse_lu.h` | LU factorization and solve API. |
| `include/sparse_lu_csr.h` | CSR LU support surface. |
| `include/sparse_matrix.h` | Sparse matrix construction, mutation, copy/free, norms, and Matrix Market I/O hooks. |
| `include/sparse_qr.h` | QR, least-squares, rank, nullspace, and minimum-norm contracts. |
| `include/sparse_reorder.h` | Reordering APIs and options. |
| `include/sparse_svd.h` | Full SVD, partial SVD, pseudoinverse, and low-rank contracts. |
| `include/sparse_types.h` | Shared scalar, index, status, callback, timing, and compressed-format types. |
| `include/sparse_vector.h` | Sparse vector helpers. |

Installed packages also include generated `sparse_version.h`, derived from
`VERSION` and `include/sparse_version.h.in`. That generated header is part of
the installed-header surface but is not a checked-in public `.h` declaration
file.

## Build-System Source Ownership

| Surface | Evidence | Day 2 Interpretation |
| --- | --- | --- |
| Library implementation list | `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | CMake owns a static-library source list. |
| Library implementation list | `Makefile` `LIB_SRCS` | Make owns a parallel static-library source list. |
| Manifest source of truth | `build-metadata/library_sources.txt` | Manifest is compared to both Make and CMake by source-list tooling. |
| Drift guard | `scripts/check_library_sources.py` | Parses manifest, Makefile `LIB_SRCS`, and CMake `add_library` block; checks duplicates, missing entries, extras, and ordering. |
| Quality wrapper | `make source-list-check` and `make quality-review-compile` | The reviewed Makefile quality path includes source-list drift detection. |
| Day 2 focused check | `python3 scripts/check_library_sources.py` | Passed: `source-list-check: PASS (49 library sources)`. |

## Largest Owner Files

| Rank | File | Lines | Owner Surface |
| ---: | --- | ---: | --- |
| 1 | `tests/test_qr.c` | 3,970 | QR broad test owner. |
| 2 | `tests/test_ldlt_csc.c` | 3,915 | LDL^T CSC proof owner. |
| 3 | `tests/test_integration.c` | 3,279 | Cross-feature integration owner. |
| 4 | `tests/test_svd.c` | 3,029 | Full SVD proof owner. |
| 5 | `tests/test_ldlt.c` | 3,006 | LDL^T public/backend proof owner. |
| 6 | `tests/test_etree.c` | 2,962 | Elimination-tree proof owner. |
| 7 | `tests/test_iterative.c` | 2,924 | Iterative solver proof owner. |
| 8 | `tests/test_graph.c` | 2,764 | Graph and separator proof owner. |
| 9 | `tests/test_chol_csc.c` | 2,554 | CSC Cholesky proof owner. |
| 10 | `tests/test_chol_csc_supernodal.c` | 2,504 | Supernodal Cholesky proof owner. |
| 11 | `tests/test_reorder_nd.c` | 2,304 | Nested dissection proof owner. |
| 12 | `tests/test_eigs.c` | 2,155 | Eigensolver proof owner. |
| 13 | `src/sparse_ldlt_csc.c` | 2,095 | LDL^T CSC implementation. |
| 14 | `tests/test_colamd.c` | 2,017 | COLAMD proof owner. |
| 15 | `tests/test_ilu.c` | 1,974 | ILU proof owner. |
| 16 | `tests/test_lu_csr.c` | 1,806 | CSR LU proof owner. |
| 17 | `tests/test_minres.c` | 1,649 | MINRES proof owner. |
| 18 | `scripts/run_corpus_oracle.py` | 1,609 | Generated corpus oracle owner. |
| 19 | `src/sparse_lu_csr.c` | 1,594 | CSR LU implementation. |
| 20 | `src/sparse_ldlt.c` | 1,535 | LDL^T public implementation. |

## Maintainability Risks

| Risk | Evidence | Boundary |
| --- | --- | --- |
| Large monolithic proof owners | Multiple test files exceed 2,000 lines, with `tests/test_qr.c` and `tests/test_ldlt_csc.c` near 4,000 lines. | This is maintainability evidence, not a correctness failure. New Sprint 158-166 proof should prefer focused owners where possible. |
| Parallel source-list ownership | Library source membership appears in manifest, Makefile, and CMake. | Existing `source-list-check` mitigates drift; new library sources must update all surfaces. |
| Generated installed header split | `sparse_version.h` is generated at build/install time from `include/sparse_version.h.in`. | Public API docs must distinguish checked-in headers from generated installed headers. |
| Script complexity | `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py`, and `scripts/run_external_comparison.py` are large enough to need focused tests when touched. | Generated-report promotion should avoid broad rewrites. |
| Test expansion pressure | QR and partial-SVD work has dedicated corpus tests, but broad older owners remain large. | Later sprints should add narrowly scoped proof-owner tests rather than expanding already-large files unless behavior ownership requires it. |

## Day 2 Handoff

Day 3 should consume this source/public-surface baseline and capture the test
and CI baseline:

- C test target inventory;
- script/corpus/install/sanitizer/dead-code validation surfaces;
- Linux, macOS, and Windows workflow support tiers;
- Windows CTest expected count;
- reviewed, supplemental, staged, local-only, hosted, advisory, and deferred
  validation boundaries.

## Completion Check

- Current implementation and public surface are captured with concrete paths.
- Maintainability risks are documented without unrelated refactors.
- Source-list drift protection is identified and passed locally.
- Later evidence contracts can reference exact source, public-header,
  generated installed-header, script, example, benchmark, and build-system
  owner surfaces.
