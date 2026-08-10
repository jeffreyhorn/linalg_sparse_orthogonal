# Day 2 Technical Baseline

## Scope

Day 2 captures the current post-Epic-12 source, test, build, package, and CI
baseline. The goal is reproducible planning evidence for Sprint 147 selection
and gate design, not implementation.

## Reproducible Baseline Commands

The first command is the copy/pasteable counter used for the Day 2 baseline.
If it is rerun after later Sprint 147 artifacts are added, the total and
Markdown counts will include those newer planning files.

```sh
find include src tests benchmarks examples scripts docs .github/workflows cmake CMakeLists.txt -type f | sort | awk '
  {
    total++;
    if ($0 ~ /\.c$/) c++;
    if ($0 ~ /\.h$/) h++;
    if ($0 ~ /\.py$/) py++;
    if ($0 ~ /\.md$/) md++;
    if ($0 ~ /^\.github\/workflows\/.*\.ya?ml$/) workflows++;
    if ($0 == "CMakeLists.txt" || $0 ~ /^cmake\//) cmake_files++;
  }
  END {
    printf("total=%d\nc=%d\nh=%d\npy=%d\nmd=%d\nworkflows=%d\ncmake=%d\n",
           total, c, h, py, md, workflows, cmake_files);
  }'
find src tests include benchmarks examples scripts -type f \( -name '*.c' -o -name '*.h' -o -name '*.py' \) -print0 | xargs -0 wc -l | sort -nr | head -30
rg -n "LIB_SRCS|TEST_SRCS|BENCH_SRCS|EX_SRCS|add_library\(|add_sparse_test|EXPECTED_WINDOWS_CTEST_COUNT|test_threads|test_sprint4_integration|test_fuzz|test_install|test_cmake_install|static_package_deferral_check|BUILD_SHARED_LIBS" Makefile CMakeLists.txt .github/workflows/*.yml tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh
```

## File-Type Baseline

| Surface | Count | Notes |
| --- | ---: | --- |
| Total reviewed files under selected source/docs/build roots | 2,986 | Includes planning docs, implementation, tests, scripts, workflows, and CMake/package files. |
| C source files | 140 | Library, tests, benchmarks, and examples. |
| Header files | 53 | Public headers plus internal/test/benchmark helpers. |
| Python scripts/helpers | 11 | Corpus/report validators and external-reference helpers. |
| Markdown docs | 2,206 | Planning history dominates count; public docs and maintainer docs remain a smaller but still heavy surface. |
| GitHub Actions workflows | 3 | Linux, macOS, and Windows workflows. |
| CMake-related files counted by command | 2 | Root `CMakeLists.txt` plus CMake package template. |

## Largest File Baseline

| Rank | File | Lines | Risk |
| ---: | --- | ---: | --- |
| 1 | `tests/test_qr.c` | 3,970 | Monolithic QR proof owner; hard to audit broad QR changes without accidental claim widening. |
| 2 | `tests/test_ldlt_csc.c` | 3,915 | Large direct-solver proof owner; factorization and fixture concerns are dense. |
| 3 | `tests/test_integration.c` | 3,279 | Broad integration surface; difficult to isolate ownership. |
| 4 | `tests/test_svd.c` | 3,029 | Full SVD coverage is large and intertwined with partial-SVD evidence. |
| 5 | `tests/test_ldlt.c` | 3,006 | Large linked-list LDLT test owner. |
| 6 | `tests/test_etree.c` | 2,962 | Large symbolic analysis proof owner. |
| 7 | `tests/test_iterative.c` | 2,924 | Large iterative-solver coverage with many behavior classes. |
| 8 | `tests/test_graph.c` | 2,764 | Large graph/reorder proof surface. |
| 9 | `tests/test_chol_csc.c` | 2,554 | Large CSC Cholesky proof surface. |
| 10 | `tests/test_chol_csc_supernodal.c` | 2,504 | Large supernodal test owner. |
| 11 | `tests/test_reorder_nd.c` | 2,304 | Large ND/reorder proof owner. |
| 12 | `tests/test_eigs.c` | 2,155 | Large eigensolver proof owner. |
| 13 | `src/sparse_ldlt_csc.c` | 2,095 | Largest implementation file; algorithm, workspace, and diagnostics are tightly packed. |
| 14 | `tests/test_colamd.c` | 2,017 | Large COLAMD/reorder test owner. |
| 15 | `tests/test_ilu.c` | 1,974 | Large preconditioner test owner. |
| 16 | `tests/test_lu_csr.c` | 1,806 | Large CSR LU proof owner. |
| 17 | `tests/test_minres.c` | 1,649 | Large MINRES proof owner. |
| 18 | `src/sparse_lu_csr.c` | 1,594 | Large implementation file for CSR LU. |
| 19 | `src/sparse_ldlt.c` | 1,535 | Large linked-list LDLT implementation. |
| 20 | `tests/test_svd_partial_helpers.h` | 1,519 | Test helper is itself large enough to need ownership discipline. |
| 21 | `src/sparse_iterative.c` | 1,495 | Large iterative implementation with multiple solver paths. |
| 22 | `tests/test_bicgstab.c` | 1,483 | Large BiCGSTAB proof owner. |
| 23 | `src/sparse_qr.c` | 1,448 | Large QR implementation; future QR corpus work should avoid broad churn. |
| 24 | `tests/test_eigs_lobpcg.c` | 1,417 | Large LOBPCG proof owner. |
| 25 | `tests/test_eigs_thick_restart.c` | 1,377 | Large thick-restart proof owner. |
| 26 | `tests/test_stagnation.c` | 1,361 | Large iterative stagnation proof owner. |
| 27 | `src/sparse_eigs.c` | 1,336 | Large eigensolver implementation. |
| 28 | `src/sparse_svd.c` | 1,319 | Large SVD implementation. |
| 29 | `tests/test_sparse_matrix.c` | 1,296 | Large matrix-shell proof owner. |

## Maintainability Baseline

The codebase is test-rich but still has several maintainability pressure
points:

- large test files carry many unrelated fixture classes in one compilation
  unit;
- large solver files mix algorithmic kernels, workspace management, fallback
  behavior, diagnostics, and error handling;
- Make and CMake maintain parallel source/test lists;
- Windows CTest count is hard-coded in workflow policy;
- test helper headers can become large enough to need their own ownership
  boundaries;
- planning and support-tier history is valuable but can create duplication and
  drift in public docs, CI comments, report rows, and maintainer guidance.

## Make And CMake Ownership

| Surface | Current Owner | Drift Risk |
| --- | --- | --- |
| `Makefile::LIB_SRCS` | Primary Make library source list | Must stay aligned with `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`. |
| `Makefile::TEST_SRCS` | Unix-side test source list | Includes pthread/POSIX tests that are staged outside Windows. |
| `Makefile::BENCH_SRCS` | Benchmark compile/runtime owner | Full benchmark runtime remains too heavy for ordinary PR CI. |
| `Makefile::EX_SRCS` | Example build owner | Uses wildcard discovery, reducing manual drift for examples. |
| `CMakeLists.txt::add_library` | CMake library source list | Duplicates Make source ownership. |
| `CMakeLists.txt::add_sparse_test` | CMake test registration owner | Windows gating excludes pthread/POSIX and fuzz surfaces. |
| `build-metadata/library_sources.txt` | Source-list reconciliation support | Should be checked when source ownership changes. |

## Windows CTest And Staged Exclusion Snapshot

| Field | Current Value |
| --- | --- |
| Workflow | `.github/workflows/windows-ci.yml` |
| Reviewed Windows lane | CMake configure, build, `ctest -N`, and full `ctest` on MSVC 2022 |
| Expected CTest registrations | `56` |
| Staged exclusions | `test_threads`, `test_sprint4_integration`, `test_fuzz` |
| Source blockers | pthread APIs for `test_threads` and `test_sprint4_integration`; POSIX temp-file APIs for `test_fuzz` |
| Non-claims preserved | no Windows Makefile parity, no Windows `pkg-config` parity, no separate reviewed Windows install-validation lane |

Sprint 148 should not simply bump the expected count. It should either promote
a test intentionally with hosted proof or preserve the staged exclusion with
updated evidence.

## Package Proof Inventory

| Surface | Command Or File | Current Meaning |
| --- | --- | --- |
| Make install and `pkg-config` proof | `bash tests/test_install.sh` | Local Unix-side static archive install, header install, `sparse.pc`, downstream compile/link/run, maintained example compile/run, and uninstall proof. |
| CMake install/export proof | `bash tests/test_cmake_install.sh` | Local CMake package export, `find_package(Sparse)`, exact-version behavior, mismatch rejection, downstream example, and static metadata proof. |
| Static package deferral guard | `bash scripts/static_package_deferral_check.sh` | Confirms `BUILD_SHARED_LIBS=ON` rejection, static target shape, no shared metadata, no package selector, and deferral wording. |
| Linux reviewed package lane | `.github/workflows/ci.yml` | Runs install scripts plus static-first deferral proof as reviewed Linux package contract. |
| macOS reviewed package lanes | `.github/workflows/macos-ci.yml` | Runs reviewed Make install/`pkg-config`, CMake install/export, and static-first deferral proof. |
| Windows supplemental package lane | `.github/workflows/windows-ci.yml` | Maintained CMake install/downstream confidence, not a separate reviewed Windows install-validation parity claim. |

## Day 3 Handoff

Day 3 should capture the corpus/report baseline with the same discipline:

- source-controlled corpus metadata versus generated local oracle/report rows;
- current QR and partial-SVD promoted fixture rows;
- report-family row meanings and freshness policies;
- validation commands for schema, oracle generation, normalization, and
  freshness checks;
- residual boundaries for broad corpus, external parity, and generated report
  freshness claims.
