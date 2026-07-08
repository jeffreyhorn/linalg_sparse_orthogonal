# Day 11: Shared-Library and Dynamic ABI Product Contract

## Purpose

Day 11 decides whether Sprint 115 should add shared-library or dynamic ABI
support, or keep that support as an explicit future product contract. The
decision must preserve the maintained static-first package story without
turning install/export metadata into an unsupported ABI promise.

## Current Static-First Evidence

| Surface | Current behavior |
|---|---|
| `CMakeLists.txt` | Emits a status message when `BUILD_SHARED_LIBS=ON` is requested and still builds `sparse_lu_ortho` as `STATIC`. |
| CMake install/export | Exports `Sparse::sparse_lu_ortho` for the static target. |
| `SparseConfigVersion.cmake` | Uses exact-version compatibility to avoid broad ABI compatibility claims. |
| `Makefile` install | Installs the static archive, public headers, generated `sparse_version.h`, and `sparse.pc`. |
| `sparse.pc.in` | Describes the static library link surface with optional build-mode flags. |
| `README.md` | Describes downstream use through `pkg-config` or `find_package(Sparse)` against the maintained static package surface. |
| `INSTALL.md` | States that the install/export story is real but not a shared-library or dynamic-ABI promise. |
| `docs/maintainer_guide.md` | Defines shared-library and wider ABI support as a separate product contract. |

## Decision

Sprint 115 does not add shared-library packaging or dynamic ABI support.

The maintained package surface remains static-first. Shared-library support,
dynamic ABI compatibility, runtime-loader behavior, and platform-specific
shared artifact handling remain future product work.

## Rationale

Adding shared-library support is not a metadata toggle. It would require
coordinated build rules, package metadata, loader behavior, symbol policy,
versioning policy, and platform validation. Enabling `BUILD_SHARED_LIBS` or
changing the CMake target type without those pieces would create a support
claim the project does not review.

The current exact-version CMake package behavior is also intentional. It lets
installed CMake consumers find the matching static package while avoiding a
claim that different versions are dynamically ABI-compatible.

## Future Acceptance Criteria

A future shared-library/dynamic ABI sprint should land all of the following
before changing public support wording:

1. Build rules:
   - Makefile shared-library build target;
   - CMake shared target or option with explicit static/shared selection;
   - no accidental loss of the existing static package surface.
2. Install/export metadata:
   - installed shared artifacts under the right platform directories;
   - CMake exported targets for shared linkage;
   - `pkg-config` metadata that distinguishes static and shared requirements.
3. Platform runtime-loader proof:
   - Linux `.so` loader path and downstream run proof;
   - macOS `.dylib` install-name/rpath proof;
   - Windows DLL/import-library proof if Windows support is claimed.
4. Symbol policy:
   - exported public symbol list;
   - hidden/internal symbol policy;
   - visibility macro or export-definition mechanism where needed.
5. Versioning policy:
   - SONAME/SOVERSION or platform-equivalent policy;
   - documented compatibility rules for major/minor/patch changes;
   - exact rules for public struct layout changes.
6. ABI validation:
   - compatibility or rejection tests for old/new installed consumers;
   - explicit handling for headers that already document ABI-breaking struct
     changes;
   - CI evidence for any platform where dynamic ABI support is claimed.
7. Documentation:
   - clear split between source API compatibility, static package support, and
     dynamic ABI compatibility;
   - updated README, INSTALL, maintainer guide, and package metadata notes.

## Support Wording Assessment

No wording changes are needed for Day 11.

The existing docs already state:

- the maintained package surface is static-first;
- shared-library packaging is deferred;
- dynamic ABI compatibility is not promised;
- exact-version CMake package compatibility is deliberate;
- runtime-loader behavior is not reviewed.

## Non-Claims Preserved

Day 11 does not claim:

- shared-library package support;
- dynamic ABI compatibility;
- SONAME/SOVERSION policy;
- DLL/import-library support;
- macOS dylib install-name or rpath behavior;
- Linux `.so` runtime-loader behavior;
- symbol-level compatibility;
- package-manager support.

## Validation

Day 11 is documentation-only. No CMake, Makefile, source, header, workflow, or
package metadata changes were made.
