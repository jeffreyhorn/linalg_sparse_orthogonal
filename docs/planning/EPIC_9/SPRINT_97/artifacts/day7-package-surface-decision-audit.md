# Sprint 97 Day 7: Package-Surface Decision Audit

## Purpose

Day 7 audits the current package, install, export, and consumer claims before
Sprint 97 makes a package-surface decision. The audit frames the decision as an
evidence question: preserve the maintained static-first contract, earn one
bounded shared-library lane with new proof, or explicitly defer shared-library
work.

## Claim Inventory

Current public and operational package claims are static-first:

| Surface | Current claim |
| --- | --- |
| `README.md` installation summary | installed consumers use `pkg-config` or `find_package(Sparse)` against the maintained static package surface |
| `INSTALL.md` support split | Unix-side `make install` plus `pkg-config`; installed CMake consumer path through `cmake --install` and `find_package(Sparse)` |
| `INSTALL.md` install contract | static archive on Unix, static `.lib` on Windows/MSVC, exact-version CMake package metadata, no broad shared-library or dynamic-ABI promise |
| `CMakeLists.txt` configure path | `BUILD_SHARED_LIBS=ON` emits an explicit static-first status message and still declares `add_library(sparse_lu_ortho STATIC ...)` |
| `CMakeLists.txt` install/export rules | exports `Sparse::sparse_lu_ortho` and installs archive, headers, CMake package files, and `sparse.pc` |
| `Makefile` install recipe | installs `build/libsparse_lu_ortho.a`, public headers, generated version header, and `sparse.pc` |
| `sparse.pc.in` | exposes `-lsparse_lu_ortho` plus math and optional threading/OpenMP flags |
| `cmake/SparseConfig.cmake.in` | imports installed `SparseTargets.cmake` and checks package components |
| `tests/test_install.sh` | validates Make install/uninstall plus `pkg-config` downstream consumers |
| `tests/test_cmake_install.sh` | validates CMake install/export, `find_package(Sparse)`, exact version acceptance, and mismatched version rejection |
| `.github/workflows/macos-ci.yml` | provides supplemental static-first Make install/`pkg-config` confidence, not reviewed install/export parity |
| `.github/workflows/windows-ci.yml` | remains reviewed CMake-first consumer proof only, not a separate reviewed install-validation lane |

No audited surface currently promises a maintained shared-library package,
dynamic ABI, package-manager integration, or full platform install/export
parity.

## Live Evidence

Focused Day 7 probes:

```sh
make -n install PREFIX=/tmp/sparse-sprint97-day7
cmake -S . -B build/sprint97-day7-shared-probe \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/sparse-sprint97-day7
```

Observed evidence:

- Make install dry run installs `build/libsparse_lu_ortho.a`, public headers,
  generated `sparse_version.h`, and `sparse.pc`.
- CMake configure with `BUILD_SHARED_LIBS=ON` succeeds but prints the explicit
  static-first message:

```text
BUILD_SHARED_LIBS=ON requested, but sparse_lu_ortho remains a maintained static archive package surface; continuing with STATIC library output.
```

This matches the documented package contract.

## Capability Audit

### Make

Current Make capability supports:

- building a static archive with `ar rcs`
- installing the static archive
- installing public headers and generated version metadata
- generating and installing `sparse.pc`
- validating a downstream `pkg-config` consumer through
  `tests/test_install.sh`
- uninstalling the static package files

Current Make capability does not support:

- shared-library build output
- shared-library install naming
- runtime loader path handling
- symbol visibility policy
- platform-specific dynamic library conventions

### CMake

Current CMake capability supports:

- building `sparse_lu_ortho` as `STATIC`
- installing archive output, public headers, generated version metadata,
  CMake package files, and `sparse.pc`
- exporting `Sparse::sparse_lu_ortho`
- exact-version package compatibility
- validating an installed `find_package(Sparse)` consumer through
  `tests/test_cmake_install.sh`

Current CMake capability does not support:

- a maintained `SHARED` target contract
- dynamic ABI/version compatibility beyond exact package versioning
- Windows DLL/import-library packaging semantics
- reviewed shared-library consumer tests
- shared/static dual-install conflict rules

## Decision Options

### Option A: Preserve Static-First Contract

Cost:

- low implementation cost
- mostly documentation and proof-surface cleanup
- no new platform-specific runtime-loader obligations

Proof burden:

- keep `tests/test_install.sh` and `tests/test_cmake_install.sh` as local
  install/export proof
- keep CI wording narrow about reviewed platform confidence
- keep CMake's `BUILD_SHARED_LIBS=ON` static-first message explicit
- optionally refresh stale workflow-wrapper wording found during the audit

Risk:

- static-only users remain well served
- users needing dynamic libraries must read the limitation as deliberate
- no new unsupported package promise is created

### Option B: Earn One Bounded Shared-Library Lane

Cost:

- medium to high implementation cost
- requires new build behavior, package metadata, tests, and CI interpretation
- likely needs platform-specific handling for Unix shared objects, macOS
  dylibs, and Windows DLL/import libraries

Proof burden:

- define whether shared builds are CMake-only or also Make-supported
- add an explicit shared target or shared build option
- decide install/export names and coexistence with static artifacts
- update CMake package exports and `sparse.pc` semantics
- add installed shared-library consumer tests
- verify runtime loader behavior on at least one reviewed platform
- decide version compatibility and ABI wording
- keep Windows claims narrow unless DLL proof is actually added

Risk:

- easy to overstate package maturity
- shared-library behavior can pass locally but fail for downstream consumers
  due to loader paths, symbol visibility, or import-library differences
- adds maintenance burden to every future package and platform change

### Option C: Document Shared-Library Work As Deferred

Cost:

- low implementation cost
- requires explicit non-claim language where users are most likely to infer
  shared-library support

Proof burden:

- no shared-library proof added this sprint
- ensure `BUILD_SHARED_LIBS=ON`, README, INSTALL, workflows, and install tests
  tell the same story
- carry a residual task for a future sprint if shared-library support becomes
  product-relevant

Risk:

- similar user limitation as Option A
- clearer than leaving shared-library work implicit

## Audit Finding

The current evidence favors preserving the static-first package contract for
Sprint 97 unless Day 8 explicitly chooses to fund a bounded shared-library
lane. The existing Make, CMake, docs, package metadata, install scripts, and CI
wording already align around static-first support. A shared-library lane would
be real product work, not a documentation edit.

Recommended Day 8 decision:

1. Preserve static-first as the maintained package contract.
2. Treat shared-library support as deferred unless new proof is implemented.
3. Update only surfaces that clarify the current contract or remove stale
   workflow-wrapper wording.
4. Do not imply reviewed macOS or Windows install/export parity beyond the
   current narrow lanes.

## Residual Queue

Carry forward to Day 8-10:

- decide whether to add one short shared-library non-claim to README or
  INSTALL where users first see installation guidance
- update stale `quality-review-compile` wording in `INSTALL.md` so it includes
  `source-list-check` after the Day 5 build-topology change
- consider whether `tests/test_cmake_install.sh` should assert the installed
  target resolves to a static archive path
- keep Windows as CMake-first consumer proof only unless a reviewed install
  validation lane is added
- keep macOS Make install/`pkg-config` proof described as supplemental
  static-first confidence, not reviewed install/export parity

## Validation

Day 7 changed planning documentation only.

Focused audit commands:

```sh
make -n install PREFIX=/tmp/sparse-sprint97-day7
cmake -S . -B build/sprint97-day7-shared-probe \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/sparse-sprint97-day7
```

Required hygiene after writing this artifact:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_97
```

No `.c` or `.h` files were modified, so the full
`make format && make lint && make test` chain is not required for this
docs-only package audit.
