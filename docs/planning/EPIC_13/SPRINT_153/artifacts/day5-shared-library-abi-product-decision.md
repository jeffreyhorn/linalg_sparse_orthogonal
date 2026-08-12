# Sprint 153 Day 5 Shared-Library ABI Product Decision

## Decision

Sprint 153 will not implement shared-library support. The selected product path
is stronger static-first deferral with exact, test-backed blockers.

The maintained package contract remains:

- static archive install/export only;
- `BUILD_SHARED_LIBS=ON` rejected at CMake configure time;
- CMake package metadata describes `Sparse::sparse_lu_ortho` as a static
  imported target;
- `sparse.pc` describes static archive package metadata;
- install tests reject `.so`, `.so.*`, `.dylib`, and `.dll` artifacts;
- Linux, macOS, and Windows CI evidence remains static package proof, not
  dynamic-loader or dynamic ABI proof.

## Rationale

The Day 2 ABI audit and Day 3 loader audit show that a credible shared-library
claim would require more than changing the target type. The project currently
lacks the prerequisites needed to avoid accidentally shipping implementation
details as ABI.

Primary reasons:

- The installed public surface contains many concrete public structs, callback
  typedefs, enum values, version macros, and lifetime conventions that would
  need an explicit compatibility policy.
- A naive shared build would risk exporting non-static internal helper symbols
  from compiled objects.
- No public `SPARSE_API` export/import macro exists.
- No Linux SONAME, macOS install-name/RPATH, or Windows DLL/import-library
  policy exists.
- No downstream installed shared consumer proof exists.
- No runtime loader inspection or execution proof exists.
- Windows shared support would require source/header-level
  `__declspec(dllexport/dllimport)` decisions and allocator/C runtime boundary
  review.

Implementing a partial shared surface in this sprint would create a high risk
of overclaiming binary compatibility. Strengthening the static-first deferral
is the product path that can be completed cleanly while preserving user trust.

## Selected Implementation Scope

Days 6-14 should implement and validate this selected scope:

1. Preserve the static target and static install/export behavior.
2. Keep `BUILD_SHARED_LIBS=ON` as a hard configure-time rejection.
3. Strengthen the rejection diagnostics so the exact blockers are visible:
   visibility/export policy, dynamic ABI policy, SONAME/install-name/import
   library policy, downstream shared consumer proof, and runtime loader proof.
4. Strengthen static package proof where needed so shared artifacts, shared
   imported CMake metadata, and shared/static package selectors remain blocked.
5. Align README, INSTALL, maintainer guide, CMake comments, package metadata,
   and sprint artifacts around the selected decision.
6. Leave a Sprint 154 handoff that external comparison work can cite without
   rediscovering the shared-library blockers.

## Out-Of-Scope Claims

The following remain explicit non-claims after the Day 5 decision:

- shared-library packaging;
- dynamic ABI compatibility;
- Linux `.so` support;
- macOS `.dylib` support;
- Windows `.dll` or import-library support;
- SONAME, install-name, RPATH, or runtime loader behavior;
- static/shared package selectors;
- package-manager distribution;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- ABI stability for public concrete structs, callbacks, enum values,
  allocator boundaries, or error-state behavior.

## Proof Owner Map

| Proof Surface | Owner File Or Command | Selected Claim |
| --- | --- | --- |
| CMake shared request rejection | `scripts/static_package_deferral_check.sh` | `BUILD_SHARED_LIBS=ON` is rejected and names static-first deferral. |
| Static CMake target | `CMakeLists.txt` and static deferral guard | `sparse_lu_ortho` remains explicitly static. |
| Make install package proof | `tests/test_install.sh` | Make install provides static archive, headers, `sparse.pc`, downstream `pkg-config` compile/link/run proof, and no shared artifacts. |
| CMake install/export proof | `tests/test_cmake_install.sh` | CMake install provides static archive, headers, static imported target, version metadata, downstream `find_package` proof, and no shared metadata. |
| Package metadata guard | `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install tests | Package metadata does not claim shared-library or dynamic ABI support. |
| Documentation alignment | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | User and maintainer docs describe static-first package support and exact shared deferral boundaries. |
| Platform CI evidence | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | CI lanes prove static package behavior only, with platform-specific non-claims preserved. |

## Exact Shared-Library Blockers

Shared-library support is deferred until all blockers below are closed for the
selected support platform set:

1. Public export/import macro policy exists and is applied to intended public
   symbols.
2. Internal symbols are hidden through visibility attributes, export lists,
   linker scripts, `.def` files, or static/private refactoring.
3. Public ABI compatibility policy exists for structs, enum values, callbacks,
   allocator/lifetime boundaries, error state, and version metadata.
4. Linux SONAME policy exists for any `.so` claim.
5. macOS install-name and RPATH policy exists for any `.dylib` claim.
6. Windows DLL/import-library naming, installation, import/export, runtime
   lookup, and C runtime allocator policy exists for any `.dll` claim.
7. Installed downstream CMake consumer proof links and runs against the
   installed shared artifact.
8. Installed Unix `pkg-config` consumer proof defines shared/static semantics
   and verifies `--libs` and `--static` behavior.
9. Runtime loader inspection proof validates the selected platform metadata and
   dependencies.
10. Documentation and CI wording agree on the exact supported and unsupported
    platform scope.

## Residual Register

| Residual | Owner | Disposition |
| --- | --- | --- |
| Dynamic ABI policy for public concrete structs | Future ABI sprint | Deferred; no binary compatibility claim. |
| Symbol visibility/export allowlist | Future shared-library implementation | Deferred; required before shared artifacts. |
| Linux SONAME and loader proof | Future shared-library implementation | Deferred; required before `.so` support. |
| macOS install-name/RPATH proof | Future shared-library implementation | Deferred; required before `.dylib` support. |
| Windows DLL/import-library proof | Future shared-library implementation | Deferred; required before `.dll` support. |
| Package static/shared selectors | Future package design | Deferred; current metadata remains static archive scoped. |
| Package-manager distribution | Future release/productization epic | Deferred; not part of Sprint 153. |

## Day 6 Implementation Checklist

Day 6 should design the build/install changes for the selected static-first
deferral path:

1. Inspect `CMakeLists.txt`, `Makefile`, `sparse.pc.in`,
   `cmake/SparseConfig.cmake.in`, `tests/test_install.sh`,
   `tests/test_cmake_install.sh`, and
   `scripts/static_package_deferral_check.sh`.
2. Decide the exact stronger `BUILD_SHARED_LIBS=ON` diagnostic wording.
3. Decide whether `scripts/static_package_deferral_check.sh` needs additional
   grep checks for the exact blocker wording.
4. Decide whether install tests need stronger package metadata or shared
   artifact absence checks.
5. Decide whether docs need Day 7 edits or can be saved for Day 11 alignment.
6. Preserve all current static package behavior unless a focused proof requires
   a stronger assertion.
