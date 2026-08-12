# Sprint 153 Day 1 ABI Intake Baseline

## Purpose

Day 1 establishes the Sprint 153 shared-library ABI product-decision baseline.
The current project state is intentionally static-first; shared-library
packaging and dynamic ABI support are deferred until a product decision,
implementation, downstream proof, and platform-loader validation exist.

## Scope Inputs

Sprint 153 implements the project-plan section:

- `docs/planning/EPIC_13/PROJECT_PLAN.md`
- `Sprint 153: Shared-Library ABI Product Decision`

Sprint 152 hands off the ABI/package boundary in:

- `docs/planning/EPIC_13/SPRINT_152/artifacts/sprint153-abi-package-handoff.md`

That handoff explicitly says selected local oracle freshness is not package,
ABI, loader, shared-library, hosted CI, platform, performance, or release
evidence.

## Current Static-First Package Surface

### Make Install

`Makefile` install behavior:

- installs `$(LIB)` to `$(DESTDIR)$(PREFIX)/lib`;
- installs public headers to `$(DESTDIR)$(PREFIX)/include/sparse`;
- installs generated `sparse_version.h`;
- writes `$(DESTDIR)$(PREFIX)/lib/pkgconfig/sparse.pc` from `sparse.pc.in`;
- has no shared-library install target or static/shared selector.

### CMake Install

`CMakeLists.txt` package behavior:

- rejects `BUILD_SHARED_LIBS=ON` at configure time;
- declares `add_library(sparse_lu_ortho STATIC ...)`;
- installs the library through `ARCHIVE DESTINATION`;
- installs public headers and generated `sparse_version.h`;
- exports `SparseTargets.cmake` with namespace `Sparse::`;
- writes exact-version `SparseConfigVersion.cmake`;
- writes and installs `sparse.pc`.

### Package Metadata

`sparse.pc.in` describes:

- `Name: sparse`;
- `Description: Static archive package metadata for sparse linear algebra`;
- `Libs: -L${libdir} -lsparse_lu_ortho -lm ...`;
- no `Libs.private`;
- no shared-library, SONAME, package-manager, or dynamic ABI wording.

`cmake/SparseConfig.cmake.in` imports `SparseTargets.cmake` and uses
`check_required_components(Sparse)`.

## Public Header Baseline

Installed source headers under `include/`:

- `sparse_analysis.h`
- `sparse_bidiag.h`
- `sparse_cholesky.h`
- `sparse_csr.h`
- `sparse_dense.h`
- `sparse_eigs.h`
- `sparse_ic.h`
- `sparse_ilu.h`
- `sparse_iterative.h`
- `sparse_ldlt.h`
- `sparse_lu.h`
- `sparse_lu_csr.h`
- `sparse_matrix.h`
- `sparse_qr.h`
- `sparse_reorder.h`
- `sparse_svd.h`
- `sparse_types.h`
- `sparse_vector.h`

Generated installed header:

- `sparse_version.h`, generated from `include/sparse_version.h.in`

Current install tests expect `19` installed headers.

## Current Validation Owners

| Surface | Current owner | Current claim |
| --- | --- | --- |
| Make install and `pkg-config` | `tests/test_install.sh` | Unix-side static archive install and downstream `pkg-config` proof. |
| CMake install/export | `tests/test_cmake_install.sh` | Static CMake package export and downstream `find_package(Sparse)` proof. |
| CMake shared deferral | `CMakeLists.txt`; `scripts/static_package_deferral_check.sh`; CI lanes | `BUILD_SHARED_LIBS=ON` is rejected, not silently accepted. |
| Linux package CI | `.github/workflows/ci.yml` | Reviewed Linux static-first package contract. |
| macOS package CI | `.github/workflows/macos-ci.yml` | Reviewed macOS Make install/`pkg-config` and CMake install/export proof. |
| Windows package CI | `.github/workflows/windows-ci.yml` | Reviewed Windows CMake static install/downstream proof; no Windows Makefile or `pkg-config` parity claim. |

## Shared-Library Deferral Snapshot

Current deferral is explicit:

- CMake rejects `BUILD_SHARED_LIBS=ON`.
- The library target is static.
- Make install installs only a static archive.
- CMake install checks assert no `.so`, `.dylib`, or `.dll` artifacts.
- Windows install checks assert no `.dll` artifacts.
- Installed CMake target metadata is expected to be `STATIC IMPORTED`.
- Documentation states shared-library packaging, dynamic ABI compatibility,
  runtime-loader behavior, package-manager distribution, static/shared
  selectors, Windows Makefile parity, and Windows `pkg-config` parity are out
  of scope.

## Stop Conditions

- Do not describe static archive install proof as shared-library support.
- Do not allow `BUILD_SHARED_LIBS=ON` without a product decision and proof.
- Do not claim `.so`, `.dylib`, `.dll`, import-library, SONAME, install-name,
  or loader behavior without platform-specific validation.
- Do not imply package-manager availability from CMake or `pkg-config`
  metadata.
- Do not imply Windows Makefile or Windows `pkg-config` parity from the Windows
  CMake lane.
- Do not cite Sprint 152 local oracle freshness as package, ABI, loader,
  hosted CI, platform, or release evidence.
- Do not publish a vague static-first deferral after Day 5; exact blockers and
  tests must back the selected decision.

## Day 2 Handoff

Day 2 should inventory ABI-relevant public surface:

- installed headers and generated version header;
- public functions, structs, macros, typedefs, and constants;
- allocator/lifetime/error-code/callback contracts;
- static globals and mutable process-wide state;
- internal symbols that could leak under shared builds;
- symbol visibility controls needed before any shared-library support claim.
