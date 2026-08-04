# Sprint 137 Day 2 - Source, Test & Maintainability Baseline

## Purpose

Day 2 captures the post-Epic-11 source, test, benchmark, example, and
maintainability baseline before Sprint 137 reconciles residuals or selects the
specific gaps Epic 12 will close.

This is a documentation-only metrics artifact. It does not change source,
tests, build rules, package metadata, public documentation, or support claims.

## Reproducible Metrics Commands

The Day 2 metrics were collected with these commands:

```bash
find src include tests benchmarks examples -type f \
  \( -name '*.c' -o -name '*.h' -o -name '*.h.in' \) -print0 |
  xargs -0 wc -l | tail -n 1

printf 'src_c '; find src -name '*.c' | wc -l
printf 'src_h '; find src -name '*.h' | wc -l
printf 'include_h_and_templates '; find include \( -name '*.h' -o -name '*.h.in' \) | wc -l
printf 'test_c '; find tests -name '*.c' | wc -l
printf 'test_h '; find tests -name '*.h' | wc -l
printf 'bench_c '; find benchmarks -name '*.c' | wc -l
printf 'example_c '; find examples -name '*.c' | wc -l

find src -type f \( -name '*.c' -o -name '*.h' \) -print0 |
  xargs -0 wc -l | sort -nr | head -n 25

find tests -type f \( -name '*.c' -o -name '*.h' \) -print0 |
  xargs -0 wc -l | sort -nr | head -n 30

find benchmarks examples include -type f \
  \( -name '*.c' -o -name '*.h' -o -name '*.h.in' \) -print0 |
  xargs -0 wc -l | sort -nr | head -n 30

awk '/^LIB_SRCS =/{flag=1} /^LIB_OBJS =/{flag=0} flag && /\$\(SRCDIR\)\/.*\.c/{count++} END{print count}' Makefile
awk '/add_library\(sparse_lu_ortho STATIC/{flag=1; next} flag && /^\)/{flag=0} flag && /src\/.*\.c/{count++} END{print count}' CMakeLists.txt
awk '/^TEST_SRCS =/{flag=1} /^TEST_BINS =/{flag=0} flag && /\$\(TESTDIR\)\/.*\.c/{count++} END{print count}' Makefile
rg -c '^add_sparse_test\(' CMakeLists.txt
```

## File and Line Baseline

| Surface | Count |
| --- | ---: |
| Total C/header/template files under `src`, `include`, `tests`, `benchmarks`, and `examples` | 191 |
| Total lines across those files | 123,352 |
| implementation `.c` files under `src` | 49 |
| private implementation headers under `src` | 20 |
| public headers/templates under `include` | 19 |
| test `.c` files under `tests` | 58 |
| test helper headers under `tests` | 11 |
| benchmark `.c` files under `benchmarks` | 16 |
| example `.c` files under `examples` | 15 |

## Largest Implementation Owners

| File | Lines | Maintainability signal |
| --- | ---: | --- |
| `src/sparse_ldlt_csc.c` | 2,095 | Largest compressed direct-solver owner; likely mixes numeric factorization, storage, diagnostics, and fallback decisions. |
| `src/sparse_lu_csr.c` | 1,594 | Large CSR LU owner; important compressed-first runtime and package proof surface. |
| `src/sparse_ldlt.c` | 1,535 | Large linked-list LDLT owner; compatibility and direct-solver lifecycle risk. |
| `src/sparse_iterative.c` | 1,495 | Large iterative owner; runtime, convergence, diagnostics, and handle behavior meet here. |
| `src/sparse_qr.c` | 1,448 | Large QR owner; directly tied to Epic 12 QR residual closure. |
| `src/sparse_eigs.c` | 1,336 | Large eigensolver owner; backend/runtime and oracle residual context. |
| `src/sparse_svd.c` | 1,319 | Large SVD owner; tied to partial-SVD residual and comparison semantics. |
| `src/sparse_chol_csc.c` | 1,279 | Large CSC Cholesky owner; direct-solver corpus and package confidence relevance. |
| `src/sparse_matrix.c` | 1,053 | Core matrix-shell owner; user API and compressed-first compatibility risk. |
| `src/sparse_lu.c` | 1,042 | Linked-list LU owner; direct-solver compatibility and fallback risk. |

## Largest Proof Owners

| File | Lines | Maintainability signal |
| --- | ---: | --- |
| `tests/test_qr.c` | 3,970 | Largest proof owner; directly blocks cheap QR residual expansion and failure localization. |
| `tests/test_ldlt_csc.c` | 3,915 | Giant compressed direct-solver proof owner. |
| `tests/test_integration.c` | 3,279 | Broad mixed integration owner; failure triage can cut across solver families. |
| `tests/test_svd.c` | 3,029 | Giant SVD proof owner; tied to partial-SVD residual work. |
| `tests/test_ldlt.c` | 3,006 | Large direct-solver proof owner. |
| `tests/test_etree.c` | 2,962 | Large etree/reorder proof owner. |
| `tests/test_iterative.c` | 2,924 | Giant iterative proof owner; runtime/convergence governance relevance. |
| `tests/test_graph.c` | 2,764 | Large graph proof owner. |
| `tests/test_chol_csc.c` | 2,554 | Large CSC Cholesky proof owner. |
| `tests/test_chol_csc_supernodal.c` | 2,504 | Large supernodal proof owner. |
| `tests/test_reorder_nd.c` | 2,304 | Large nested-dissection proof owner. |
| `tests/test_eigs.c` | 2,155 | Large eigensolver proof owner. |

## Public API, Benchmark, and Example Hotspots

| File | Lines | Signal |
| --- | ---: | --- |
| `benchmarks/bench_eigs.c` | 958 | Largest benchmark driver; backend/runtime and report-index relevance. |
| `benchmarks/bench_main.c` | 841 | General benchmark owner; report normalization and local-measurement wording risk. |
| `include/sparse_iterative.h` | 773 | Largest public header; user-facing runtime, diagnostics, and handle complexity. |
| `include/sparse_eigs.h` | 651 | Large public header; backend selection, refinement, and preconditioner options. |
| `benchmarks/bench_refactor_csc.c` | 648 | Repeated direct lifecycle benchmark owner. |
| `include/sparse_matrix.h` | 617 | Core matrix and storage identity surface. |
| `benchmarks/bench_ldlt_csc.c` | 516 | Compressed direct-solver benchmark owner. |
| `include/sparse_analysis.h` | 499 | Repeated-run direct lifecycle and reorder control surface. |
| `benchmarks/bench_chol_csc.c` | 423 | CSC Cholesky performance evidence owner. |
| `benchmarks/bench_convergence.c` | 421 | Iterative convergence/report relevance. |

## Source-List and CMake Ownership Signals

| Surface | Current signal | Day 2 interpretation |
| --- | ---: | --- |
| `build-metadata/library_sources.txt` | 49 listed library sources | Manifest explicitly says it must stay in the same reviewed order as Makefile and CMake. |
| `Makefile` `LIB_SRCS` | 49 listed library sources | Makefile source ownership matches the manifest and CMake. |
| `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | 49 listed library sources | CMake source ownership matches the manifest and Makefile. |
| `Makefile` `TEST_SRCS` | 57 listed test binaries | The main Make test list excludes one checked-in test source, `tests/smoke_test.c`. |
| `CMakeLists.txt` `add_sparse_test(...)` lines | 54 unconditional-style registrations plus conditional POSIX/fuzz gates | Windows reviewed subset remains intentionally narrower because `test_threads`, `test_sprint4_integration`, and `test_fuzz` are gated away on Windows. |

## Epic 12 Gap Relevance

| Candidate gap | Day 2 maintainability signal |
| --- | --- |
| QR priority residual closure | `tests/test_qr.c` is the largest proof owner and `src/sparse_qr.c` is a large implementation owner; Sprint 139 should add focused proof ownership rather than only appending more cases to the giant file. |
| Partial-SVD residual closure | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, and `src/sparse_svd.c` are large enough that Sprint 140 should define comparison semantics and helper ownership before adding fixtures. |
| Corpus/oracle architecture | Existing tests are numerous but family-local; Day 2 reinforces the need for a maintained corpus lane instead of one-off fixture growth inside giant proof owners. |
| Report normalization | Benchmark drivers are sizeable and varied; report normalization must preserve row meaning instead of flattening benchmark, sentinel, guardrail, coverage, dead-code, and oracle outputs. |
| Runtime/backend governance | `src/sparse_iterative.c`, `src/sparse_eigs.c`, `include/sparse_iterative.h`, `include/sparse_eigs.h`, and benchmark drivers carry runtime/backend complexity that should be governed by explicit precedence rules. |
| Package/platform promotion | Source-list parity is strong, but CMake/Make test ownership and Windows gated tests show platform support remains tiered. |
| Adoption simplification | Large public headers show that first-use docs and examples still need to buffer users from advanced option surfaces. |

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Baseline metrics are reproducible from named commands. | Complete | Commands listed above with file counts, line counts, and ownership counts. |
| High-risk source and proof-owner files are ranked. | Complete | Largest implementation, proof-owner, and public/benchmark/example hotspot tables. |
| Maintainability risks are tied to Epic 12 candidate gaps. | Complete | Gap relevance table maps metrics to QR, partial-SVD, corpus/report, runtime, platform, and adoption work. |

