# Sprint 137 Day 3 - Build, Package, CI & Report Baseline

## Purpose

Day 3 captures the current build, package, CI, report, benchmark, and
support-tier evidence before Sprint 137 begins residual reconciliation and gap
selection.

This is a documentation-only baseline artifact. It does not run the expensive
quality suite, regenerate reports, change build rules, change CI, or promote
any package/platform/report claim.

## Reviewed Commands and Sources

Day 3 reviewed these surfaces:

```bash
Makefile
CMakeLists.txt
sparse.pc.in
cmake/SparseConfig.cmake.in
tests/test_install.sh
tests/test_cmake_install.sh
scripts/static_package_deferral_check.sh
scripts/bench_canonical_report.sh
scripts/performance_sentinels.sh
scripts/large_matrix_guardrails.sh
scripts/deadcode_report.py
.github/workflows/ci.yml
.github/workflows/macos-ci.yml
.github/workflows/windows-ci.yml
README.md
INSTALL.md
benchmarks/README.md
docs/maintainer_guide.md
```

No generated `build/bench-reports`, `build/deadcode`, or `coverage` outputs
were present in the worktree during this baseline pass, so this artifact
records maintained commands and expected output paths rather than fresh report
contents.

## Build and Package Proof Map

| Surface | Owner path | Current proof role | Support tier |
| --- | --- | --- | --- |
| Make static library build | `Makefile` `all`, `LIB_SRCS`, `build/libsparse_lu_ortho.a` | Builds the maintained static archive from the reviewed library source list. | Reviewed local/build surface; Linux CI also exercises compile-quality. |
| CMake static library build | `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | Builds the same maintained static library and exports target metadata. | Reviewed CMake parity surface on Linux/macOS; reviewed CMake subset on Windows. |
| Shared-library rejection | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh` | Rejects `BUILD_SHARED_LIBS=ON` and guards against unsupported shared/ABI metadata. | Maintained static-first deferral guard. |
| Make install and `pkg-config` | `Makefile`, `sparse.pc.in`, `tests/test_install.sh` | Installs static archive, headers, generated version header, and `sparse.pc`; validates downstream compile/link/run. | Reviewed Linux package-contract lane; supplemental macOS; local Unix proof elsewhere. |
| CMake install/export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `tests/test_cmake_install.sh` | Installs static target, package config/version files, exact-version behavior, and downstream `find_package(Sparse)` consumer. | Reviewed Linux package-contract lane; supplemental macOS and Windows confidence. |
| Version propagation | `VERSION`, `include/sparse_version.h.in`, CMake config, `sparse.pc.in` | Keeps source, installed header, CMake package, and pkg-config version surfaces aligned. | Maintained package metadata surface. |
| Unsupported package modes | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `scripts/static_package_deferral_check.sh` | Keeps shared-library packaging, dynamic ABI, runtime-loader behavior, and package-manager support as explicit non-claims. | Deferred/non-claim until future product decision and proof. |

## CI Lane Summary

| Workflow/job | Commands or proof | Current interpretation |
| --- | --- | --- |
| `.github/workflows/ci.yml` Linux build-and-test | `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast` | Supplemental runtime, sanitizer, and fast benchmark signal. |
| `.github/workflows/ci.yml` Linux CMake | `make quality-review-cmake` | Enforced reviewed CMake parity path. |
| `.github/workflows/ci.yml` Linux package-contract | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Reviewed static-first package contract. |
| `.github/workflows/ci.yml` Linux TSan | `test_threads`, `test_eigs`, `test_eigs_thick_restart` under TSan/OpenMP where configured | Supplemental ThreadSanitizer and OpenMP race signal with documented libomp suppression limits. |
| `.github/workflows/ci.yml` Linux lint | `make quality-review-compile` | Enforced reviewed Makefile compile-quality path. |
| `.github/workflows/ci.yml` Linux dead-code | `make deadcode-report`, `make deadcode-check` | Enforced dead-code report generation and completeness path. |
| `.github/workflows/ci.yml` Linux coverage | `make coverage` plus artifact upload | Supplemental coverage report; not reviewed behavioral completeness. |
| `.github/workflows/macos-ci.yml` Apple Clang | `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` | Reviewed macOS Apple Clang lane plus platform-specific checks. |
| `.github/workflows/macos-ci.yml` Homebrew GCC | direct `make`, `make test`, `make wall-check` | Supplemental second-compiler coverage. |
| `.github/workflows/macos-ci.yml` install/pkg-config | `tests/test_install.sh` | Supplemental static-first Make install/`pkg-config` confidence. |
| `.github/workflows/macos-ci.yml` CMake install/export | `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Supplemental CMake install/export and static-deferral confidence. |
| `.github/workflows/windows-ci.yml` build-and-test | CMake configure/build, `ctest -N`, full `ctest`; expected count 54 | Reviewed Windows CMake consumer subset. |
| `.github/workflows/windows-ci.yml` install/downstream | CMake install, installed example, exact-version and mismatch-version checks | Supplemental Windows CMake-first install/downstream confidence, not a separate reviewed install-validation lane. |

## Platform and Support-Tier Baseline

| Tier | Current owners | Boundary |
| --- | --- | --- |
| Reviewed Linux source of truth | Linux Makefile compile-quality, CMake parity, dead-code, static-first package-contract CI | Strongest reviewed project proof, but still static-first and not shared/ABI/package-manager support. |
| Supplemental Linux signals | direct `make test`, sanitizer jobs, TSan/OpenMP, coverage, `bench-fast` | Useful confidence signals; do not widen reviewed behavioral completeness or portable-performance claims. |
| Reviewed macOS lane | Apple Clang quality-review compile, CMake parity, wall-check, sanitize | Reviewed macOS source/build lane, not reviewed install/export parity. |
| Supplemental macOS lanes | Homebrew GCC direct build/test, Make install/`pkg-config`, CMake install/export | Confidence-building package/compiler lanes. |
| Reviewed Windows lane | MSVC 2022 CMake configure/build/CTest subset with 54 registered tests | CMake-first consumer subset only. |
| Supplemental Windows lane | CMake install/downstream proof | CMake-first package confidence, not reviewed install-validation parity. |
| Staged Windows exclusions | `test_threads`, `test_sprint4_integration`, `test_fuzz` | Blocked by pthread/POSIX assumptions until portability work and hosted proof land. |
| Deferred/unsupported package lanes | shared libraries, dynamic ABI, runtime loader, package-manager recipes | Explicit non-claims until product decision, build metadata, install/export proof, and platform validation exist. |

## Report-Family Inventory

| Report family | Command | Output path | Current interpretation |
| --- | --- | --- | --- |
| Canonical benchmark report | `make bench-canonical-report` | `build/bench-reports/canonical/index.tsv`, `manifest.txt`, benchmark CSVs | Threshold-free local snapshot of the maintained benchmark surface. |
| Performance sentinels | `make performance-sentinels` | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt`, wall-check and Cholesky CSC artifacts | Local sentinel bundle; only the wall-check lane is thresholded. |
| Large-matrix guardrails | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt`, structural/test/report artifacts | Reviewed structural and bounded CSV-shape lanes plus explicit supplemental skip/report rows. |
| Dead-code report | `make deadcode-report`, `make deadcode-check` | `build/deadcode/report.md`, `report.tsv`, raw tool outputs, coverage notes | Enforced report generation/completeness evidence; not removal-ready proof. |
| Coverage report | `make coverage`, `make coverage-lcov`, `make coverage-gcovr` | `coverage/coverage-src.info`, `coverage/html/` or gcovr HTML outputs | Supplemental tree-mutating line-coverage signal with threshold; not reviewed behavioral completeness. |
| Package proof logs | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | temporary install/build logs and command output | Package confidence/proof command output; not a generated report index or ABI proof. |

## Package and Report Non-Claims

- Make and CMake install/export proof covers the maintained static archive
  package surface only.
- `pkg-config` and `find_package(Sparse)` describe the static library package
  shape, not shared-library ABI compatibility.
- `BUILD_SHARED_LIBS=ON` is intentionally rejected.
- No shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, package-manager support, or static/shared selector is currently
  claimed.
- macOS install/export and Windows install/downstream confidence are
  supplemental, not reviewed parity.
- Generated report indexes provide freshness, row interpretation, command,
  platform/compiler/configuration, and artifact navigation context.
- Generated report indexes do not prove broad correctness, release readiness,
  coverage completeness, portable performance, backend parity, or platform
  parity.
- Coverage percentage remains supplemental and tree-mutating.
- Dead-code reports remain triage/report-completeness evidence and require API
  owner review before any public-surface interpretation.

## Epic 12 Gap Relevance

| Candidate gap | Day 3 signal |
| --- | --- |
| Report normalization and freshness | Existing benchmark/sentinel/guardrail reports already emit useful metadata, but row meanings differ and cross-report normalization remains deferred. |
| Runtime/backend governance | Sentinel rows carry backend and local runtime context, but they remain local evidence and do not create portable performance claims. |
| Package/ABI decision | Static-first support is enforceable and documented; shared-library ABI work remains a product decision, not an incidental build toggle. |
| Platform promotion | Linux package-contract proof is reviewed; macOS and Windows install/downstream confidence remains supplemental; Windows staged tests remain source-portability blocked. |
| Adoption simplification | README, INSTALL, benchmark docs, and maintainer guide are aligned but dense; later adoption work should summarize without widening support tiers. |

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every current proof lane has an owner and support tier. | Complete | Build/package proof map, CI lane summary, platform baseline, and report-family inventory. |
| Package and platform asymmetries are explicit. | Complete | Static-first, Linux-reviewed, macOS-supplemental, Windows-reviewed-subset, Windows-staged, and ABI/package-manager non-claims are listed. |
| Generated reports are separated from public correctness or performance claims. | Complete | Report-family inventory and package/report non-claim register. |

