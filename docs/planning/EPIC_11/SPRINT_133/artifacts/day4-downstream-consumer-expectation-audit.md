# Sprint 133 Day 4 - Downstream Consumer Expectation Audit

## Purpose

Day 4 audits downstream consumer workflows that depend on the package
contract. It separates local build-tree examples from installed-package
consumers, records what CMake and `pkg-config` currently prove for
static-first support, and defines the additional proof required before
shared-library support could become reviewed.

This is a documentation-only audit. It does not change examples, tests,
build-system files, package metadata, install behavior, or public support
wording.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `tests/test_install.sh` | Make install/uninstall, `pkg-config` metadata, and installed `pkg-config` consumer proof. |
| `tests/test_cmake_install.sh` | CMake install/export, `find_package(Sparse)`, exact-version package proof, and installed CMake consumer proof. |
| `examples/cmake_example/CMakeLists.txt` | Maintained installed CMake consumer project. |
| `examples/cmake_example/main.c` | Installed CMake consumer source and header include set. |
| `examples/README.md` | Separates installed consumer example from local build-tree teaching examples. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | User and maintainer package support expectations. |
| Day 2 public header audit | Header and ABI-sensitive declaration context. |
| Day 3 install-shape audit | Static archive, CMake export, `pkg-config`, version, and metadata baseline. |

## Downstream Consumer Workflow Inventory

| Workflow | Consumer route | Current proof | Static-first interpretation |
| --- | --- | --- | --- |
| `tests/test_install.sh` generated consumer | Installed `pkg-config` | Installs to a temporary prefix, resolves `pkg-config --cflags --libs sparse`, compiles a small matrix program, links, runs, and checks `OK`. | Primary Unix `pkg-config` installed-consumer proof. |
| `tests/test_install.sh` maintained example source | Installed `pkg-config` | Compiles and runs `examples/cmake_example/main.c` using `pkg-config` flags. | Cross-checks the maintained CMake example source through the `pkg-config` route. |
| `tests/test_cmake_install.sh` CMake example | Installed CMake package | Installs to a temporary prefix, configures `examples/cmake_example` with `CMAKE_PREFIX_PATH`, builds, runs, and checks `OK`. | Primary installed `find_package(Sparse)` consumer proof. |
| `tests/test_cmake_install.sh` exact-version consumer | Installed CMake package | Generates a temporary consumer with `find_package(Sparse <version> EXACT REQUIRED)` and links `Sparse::sparse_lu_ortho`. | Exact package-version proof, not ABI compatibility proof. |
| `tests/test_cmake_install.sh` mismatch-version consumer | Installed CMake package | Generates a lower-version consumer and expects configure rejection when a lower version exists. | Confirms exact-version stance. |
| `examples/cmake_example/` | Installed CMake package | Standalone consumer project using `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho`. | Maintained installed CMake consumer fixture. |
| `examples/*.c` teaching examples | Local build tree | Built by repo targets and include public headers with local `-Iinclude` and local static library. | Public API examples, but not installed-package proof. |
| Benchmarks and tests | Local build tree | Link directly against in-tree `sparse_lu_ortho` target or Make-built archive. | Product and regression coverage, not downstream install proof. |
| `make quality-review-cmake` | CMake build-tree parity | Configures, builds, lists tests, checks Make/CMake test count parity, and runs CTest. | Reviewed CMake build-tree parity, not installed CMake package proof. |
| macOS supplemental install job | Installed Make/pkg-config | Runs `bash tests/test_install.sh` in a supplemental macOS workflow job. | Supplemental static-first confidence, not full reviewed macOS install/export parity. |
| Windows workflow | CMake build-tree subset | Reviewed MSVC CMake-first consumer subset. | Windows CMake-first support, not separate install-validation or Makefile parity. |

## CMake Consumer Proof Map

| Proof point | Current evidence | Gap or limitation |
| --- | --- | --- |
| Installed package discovery | `find_package(Sparse REQUIRED)` in `examples/cmake_example/CMakeLists.txt`. | No component selection for static/shared variants. |
| Imported target name | `Sparse::sparse_lu_ortho`. | Static archive name is embedded in the consumer-facing target identity. |
| Installed include layout | Example includes `<sparse/sparse_types.h>`, `<sparse/sparse_matrix.h>`, `<sparse/sparse_lu.h>`, and `<sparse/sparse_lu_csr.h>`. | Does not include every installed public header. |
| Link interface | Example links `Sparse::sparse_lu_ortho` and runs successfully. | Proves current static target; does not prove shared runtime loading. |
| Version behavior | Exact-version consumer succeeds; mismatched lower-version consumer is rejected. | Version proof is package metadata, not ABI compatibility. |
| No shared artifacts | CMake install test fails if shared artifacts appear. | Useful for static-first enforcement; incompatible with claiming shared support without redesign. |
| Header count | CMake install test requires at least 14 headers. | Weaker than current 19-header install contract and maintainer snapshot. |
| Build-tree leakage | Example is configured with `CMAKE_PREFIX_PATH` against a temporary install prefix. | Good installed-prefix proof, but Day 11 should consider explicit checks that target include/link paths point at the prefix. |

## pkg-config Consumer Proof Map

| Proof point | Current evidence | Gap or limitation |
| --- | --- | --- |
| Package discovery | `PKG_CONFIG_PATH` points to staged prefix and resolves `pkg-config sparse`. | Unix-oriented proof; no Windows package-manager story. |
| Include flags | `pkg-config --cflags sparse` must include an include path. | Does not assert exact `-I${prefix}/include` value. |
| Link flags | `pkg-config --libs sparse` must include `-lsparse_lu_ortho`. | Static archive identity is consumer-visible. |
| Version metadata | `pkg-config --modversion sparse` must equal repo `VERSION`. | Package metadata only, not ABI compatibility. |
| Basic compiled consumer | Generated source includes `<sparse/sparse_types.h>` and `<sparse/sparse_matrix.h>`, compiles, links, and runs. | Minimal matrix API proof only. |
| Maintained example source | `examples/cmake_example/main.c` compiles, links, and runs through `pkg-config`. | Reuses CMake example source but not CMake package metadata. |
| Optional build flags | `sparse.pc` can append `-pthread`, OpenMP flags, or Darwin libomp flags. | Build-mode identity is not explicit; `Libs.private` is absent. |
| No shared artifacts | Install test fails if shared artifacts appear. | Static-first proof, not shared runtime proof. |

## Static Consumer Contract Notes

A downstream consumer can rely on the current static-first contract only when
these expectations hold:

- the installed library artifact is the static archive
  `libsparse_lu_ortho.a`;
- headers are available under `${prefix}/include/sparse`;
- generated `sparse_version.h` matches the repo `VERSION` used by package
  metadata;
- CMake consumers use `find_package(Sparse REQUIRED)` and link
  `Sparse::sparse_lu_ortho`;
- `pkg-config` consumers use `pkg-config --cflags --libs sparse` and link the
  static archive via `-lsparse_lu_ortho`;
- package version behavior is exact-version for CMake and exact repo version
  for `pkg-config`;
- optional OpenMP or mutex builds may add public compile/link metadata, but do
  not create a separate runtime governance or ABI claim;
- local build-tree examples and benchmark/test binaries are not substitutes
  for installed downstream consumer proof.

## Shared-Library Consumer Proof Requirements

Shared-library support should not become reviewed unless a later design and
implementation can prove all of these separately from the static-first proof:

| Requirement | Required proof before claim |
| --- | --- |
| Shared artifact creation | CMake and/or Make install intentionally emits platform-appropriate `.so`, `.so.*`, `.dylib`, or `.dll`/import-library artifacts. |
| Static/shared selection | CMake and `pkg-config` consumers can intentionally select or identify static versus shared linkage. |
| Symbol export policy | Public symbols are exported intentionally through a visibility/export policy, with private symbols hidden or explicitly accepted. |
| ABI/version policy | ABI epoch, soname/install-name, package version, and source version semantics are documented and validated. |
| Loader/runtime proof | Installed shared consumers run from the staged install prefix without build-tree library leakage. |
| Transitive dependency policy | Math, threads, OpenMP, libomp, and dlopen dependencies have public/private metadata rules for shared and static consumers. |
| Header/layout policy | Public struct layout, enum values, callback payloads, `idx_t`, and `sparse_scalar_t` have compatibility policy. |
| Platform proof | Linux, macOS, and Windows runtime-loader behavior is either reviewed, supplemental, or explicitly deferred. |
| Negative static-first updates | Current no-shared-artifact checks are redesigned so they do not conflict with the selected shared contract. |

## Build-Tree Assumption Register

| Surface | Build-tree dependency | Current risk |
| --- | --- | --- |
| Local examples under `examples/*.c` | Compile with local `-Iinclude` and local `-Lbuild -lsparse_lu_ortho`. | Useful public API teaching references, not installed package proof. |
| CMake test and benchmark targets | Link in-tree `sparse_lu_ortho`. | Exercise product code but not install metadata. |
| `make quality-review-cmake` | Uses `build/quality-review-cmake` and CTest build tree. | Reviewed CMake parity, not installed package export proof. |
| Installed CMake example | Configures with `CMAKE_PREFIX_PATH` to temporary prefix. | Stronger installed proof; Day 11 can add path-origin checks if needed. |
| pkg-config consumers | Use `PKG_CONFIG_PATH` to temporary prefix. | Stronger installed proof; Day 12 can tighten exact include/link flag checks. |

## Consumer Gap Queue

| Gap | Impact | Candidate owner |
| --- | --- | --- |
| CMake installed consumer does not include every installed public header. | Header install regressions outside the example include set may not be caught by compile proof. | Day 11 CMake consumer proof. |
| `pkg-config --cflags` only checks for any include path. | Could pass with a wrong include directory shape. | Day 12 pkg-config proof. |
| `pkg-config --libs` checks for `-lsparse_lu_ortho` but not exact library directory or dependency split. | Static link metadata can drift while still matching the broad library flag. | Day 12 pkg-config proof. |
| CMake install test header-count check is weaker than current 19-header contract. | Installed-header drift could pass CMake proof. | Day 11 or Day 13 validation. |
| No build-tree leakage check on installed CMake target paths. | A misconfigured consumer could accidentally use source/build include paths. | Day 11 CMake consumer proof. |
| No static/shared selection contract exists. | Blocks shared-library support and dual package metadata. | Day 5-6 product decision/design. |
| No loader/runtime proof exists. | Blocks reviewed shared-library support. | Days 9-10 proof if shared support is selected. |
| No package-manager consumer proof exists. | Package-manager support remains a residual non-claim. | Day 14 residual queue or future epic. |

## Day 5 Handoff

Day 5 can make the product decision with these consumer facts:

- current downstream installed consumer proof is static-first and functional
  through both CMake and `pkg-config`;
- the maintained CMake consumer target name and pkg-config library flag expose
  the static archive identity;
- current validation intentionally fails if shared-library artifacts appear;
- package version proof is exact package metadata, not ABI compatibility;
- shared-library support would require new artifact, selection, symbol,
  loader, dependency, ABI-layout, and platform proof;
- static-first continuation can still be strengthened by tighter CMake
  header-count checks, exact pkg-config path checks, installed target path
  checks, and clearer deferral/enforcement policy.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Downstream expectations are evidence-backed, not inferred from package names. | Complete | Workflow inventory, CMake proof map, and pkg-config proof map cite the actual install scripts and maintained example consumer. |
| Static consumer proof and shared consumer proof are separated. | Complete | Static consumer contract notes and shared-library consumer proof requirements list different evidence gates. |
| Day 5 can make a product decision with known consumer impact. | Complete | Day 5 handoff records current static-first consumer proof, shared-support blockers, and static-first strengthening candidates. |
