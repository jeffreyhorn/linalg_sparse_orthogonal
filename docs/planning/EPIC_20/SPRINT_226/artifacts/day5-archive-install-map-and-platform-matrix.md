# Sprint 226 Day 5: Exploratory Archive-to-Install Map and Platform Matrix

## Decision and scope

The [Day 3 exploratory archive](day3-exploratory-source-archive.md) was built and installed through CMake into a fresh, isolated prefix on macOS. The installed static package passed `check_static_install_contract.py`. All 25 installed files have a source/build mapping and SHA-256 in the [file-by-file install manifest](day5-exploratory-installed-manifest.tsv). The prior [Day 4](day4-independent-archive-comparison.md) archive comparison covered the same base commit.

**This is an exploratory mapping, not release-candidate installed proof.** The actual candidate SHA remains unselected because the changelog, clean committed sprint inputs, and release/rollback owner decisions are missing. The installed-prefix results and hashes below cannot be reused as exact-candidate evidence after those inputs change.

## Clean-prefix receipt

| Field | Observed value |
| --- | --- |
| Source archive | `build/sprint226/day3/sparse-lu-ortho-2.2.0-src-EXPLORATORY.tar.gz`; SHA-256 `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f` |
| Source commit | `c42d6715fad7e895bae4941a006df3563f1afc38` (`VERSION=2.2.0`) |
| Extracted source | `build/sprint226/day3/sparse-lu-ortho-2.2.0/` |
| Independent build tree | `build/sprint226/day5/cmake/`; default empty `CMAKE_BUILD_TYPE` and `CMAKE_C_FLAGS`, `/usr/bin/cc` (AppleClang 17), `SPARSE_IDX_BITS=32`, `SPARSE_MUTEX=OFF`, `SPARSE_OPENMP=OFF` |
| Fresh install prefix | `/Users/jeff/experiments/linalg_sparse_orthogonal/build/sprint226/day5/prefix` |
| Configure/build/install | All exited 0. `configure.log` and `install.log` have no compiler/error diagnostics; `build.log` has 17 nonfatal compiler warnings and no `error:` diagnostics. |
| Installed static contract | `python3 scripts/check_static_install_contract.py build/sprint226/day5/prefix --mode cmake --platform unix --version 2.2.0` exited 0: `static-install-contract: PASS`. |
| Installed manifest | 25 files; [TSV](day5-exploratory-installed-manifest.tsv) SHA-256 `6b4be68af29e129c1959f4556d52ac661eb6aa03093f382c17171733c8664cf5` |

The recorded commands were:

```sh
cmake -S build/sprint226/day3/sparse-lu-ortho-2.2.0 \
  -B build/sprint226/day5/cmake \
  -DCMAKE_INSTALL_PREFIX=/Users/jeff/experiments/linalg_sparse_orthogonal/build/sprint226/day5/prefix \
  -DSPARSE_IDX_BITS=32 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/sprint226/day5/cmake --target sparse_lu_ortho --parallel 2
cmake --install build/sprint226/day5/cmake
python3 scripts/check_static_install_contract.py build/sprint226/day5/prefix \
  --mode cmake --platform unix --version 2.2.0
```

Each command's output is retained in the corresponding ignored `build/sprint226/day5/{configure,build,install,contract}.log`. The prefix did not exist before installation. No downstream consumer, alternate index width, Make install, full test suite, or hosted job was run on Day 5.

## Source and installed-file mapping

| Installed group | Count and mapping | Verification |
| --- | --- | --- |
| Public headers | 18 files at `include/sparse/*.h` map one-to-one to the same names under the extracted archive's `include/`. | Every installed byte stream equals its archive source file; each installed SHA-256 is in the TSV. |
| Generated version header | One `include/sparse/sparse_version.h` generated from `include/sparse_version.h.in`, `VERSION`, and `CMakeLists.txt`. | Installed bytes equal the build-tree generated header; it defines `SPARSE_BUILT_IDX_BITS 32`. |
| Static library | One `lib/libsparse_lu_ortho.a` built from the 49 library C sources, headers, CMake recipe, compiler, and selected width. | Installed SHA-256 `f9ee20a7fdc169ad4559009234284780a2bc5e690a613736cd4bcbb7b8b9db40`. All 50 extracted `ar` members match the pre-install library by name and SHA-256. |
| CMake package | Four files: `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, and `SparseTargets-noconfig.cmake`, generated from `cmake/SparseConfig.cmake.in`, `CMakeLists.txt`, and `VERSION` as applicable. | Config and version files equal their build-tree outputs; export files are produced by CMake install and validated by the static contract checker. Their hashes are in the TSV. |
| pkg-config metadata | One `lib/pkgconfig/sparse.pc` configured from `sparse.pc.in`, `VERSION`, and `CMakeLists.txt`. | Installed bytes equal build-tree output; static contract checker accepted prefix and width metadata. |

The installed archive differs from the pre-install build-tree `.a` at exactly byte offset 33 (one byte in the ar symbol-index timestamp). The generated `cmake_install.cmake` explicitly invokes `/usr/bin/ranlib` on the installed static library. Both archives have the same size and identical 50 extracted member contents, so the installed hash is recorded independently instead of asserting raw byte identity. This is a toolchain/install transformation, not an unmapped source file. No binary-level reproducibility claim follows from this check.

The [Day 2 source inventory](day2-static-candidate-definition.md) and [Day 4 blob check](day4-independent-archive-comparison.md) trace all 4,557 archive files to the base Git tree. `LICENSE` is present with SHA-256 `ddc5e5f268b30436950b0bd1b16a510eabafde89b94808f5ee762b7977e90315`; generated/local-only content was absent. The missing `CHANGELOG.md` is still a **release blocker**. The package's license and source scope still require owner review; this map is technical provenance, not legal signoff.

## Proposed exact-candidate platform proof matrix

The release owner has **not** adopted a platform claim. These rows define the proposed smallest static routes and the exact evidence to collect after candidate freeze. For each hosted receipt, record the candidate SHA, run URL and attempt, job conclusion, runner image/architecture, compiler version, width, logs, artifact IDs, and downloaded SHA-256. A workflow name or historical green run does not satisfy a row.

| Proposed platform/tier | Width and installed consumer | Required hosted selectors and checks | Current gate |
| --- | --- | --- | --- |
| Linux Ubuntu / reviewed GCC | Default 32-bit Make/`pkg-config` static consumer and CMake `find_package(Sparse)` consumer; selected 64-bit functional/installed-width proof separately. | `.github/workflows/ci.yml`: `lint`, `cmake-build-and-test`, `package-contract`, `selected-index-width-64`; retain reviewed compile, CMake/CTest, install, static contract and width results. | **Blocked:** no candidate SHA or exact-SHA run/artifact. |
| macOS / reviewed Apple Clang | Default 32-bit Make/`pkg-config` and CMake installed consumers; selected 64-bit lane separately. Supplemental Homebrew GCC remains a different compiler tier. | `.github/workflows/macos-ci.yml`: Apple Clang `build-and-test`, `install-and-pkgconfig`, `cmake-install-export`, `selected-index-width-64`; retain package and quality logs. | **Blocked:** no candidate SHA or exact-SHA run/artifact. |
| Windows / MSVC 2022 CMake subset | Default 32-bit CMake install/downstream only; no Make or `pkg-config` execution parity. | `.github/workflows/windows-ci.yml`: `build-and-test`, `install-and-downstream`, applicable `powershell-validation`; retain CTest count, install contract, downstream execution, and artifact provenance. | **Blocked:** no candidate SHA or exact-SHA run/artifact. |
| Canonical macOS timing, separate decision | `s225-v1` serial 32-bit `bench_refactor_csc`/`nos4`; never a broad speed claim. | `.github/workflows/sprint225-baseline.yml` must establish an accepted compatible baseline; `.github/workflows/sprint225-timing.yml` must then measure and decide at exact candidate SHA, or product/release owners must explicitly retain the blocked scope. | **Blocked:** no accepted baseline or candidate measurement. |

Local candidate prerequisites include `make format && make lint && make test` if C/header files change; reviewed Make/CMake quality; static install, width, and negative package tests; first-use examples; docs/source/claim guards; and source archive/manifest comparison. The [Sprint 214 release checklist](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md) remains the governing gate list. The final candidate's claimed platforms must be explicitly chosen, and any source change requires archive, install, and hosted evidence to be rerun for the new SHA.

## Day 6 handoff

The exploratory mapping is complete for the 25 installed files, but the **candidate-specific** map and platform matrix cannot pass until a clean reviewed candidate and owner decisions exist. Day 6 can exercise local clean-prefix consumers against this exploratory source to validate the procedure; such results must remain labeled as exploratory and cannot be promoted to release proof.
