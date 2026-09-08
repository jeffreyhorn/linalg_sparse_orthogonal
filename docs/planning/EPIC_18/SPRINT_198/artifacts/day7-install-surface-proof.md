# Sprint 198 Day 7: Install Surface Proof

## Purpose

Review the installed static package surface expected from the selected local
Homebrew formula proof and record why install execution remains blocked until
approved license metadata allows formula rendering.

## Day 7 Disposition

The local formula install path is not executed on Day 7 because the proof
still exits `2` before archive/render/install/test work while standalone root
license metadata is absent. This preserves the Day 2 license decision and
prevents an install result from being interpreted as package-manager support
without approved provider metadata.

Day 7 therefore validates the install-surface contract by reviewing the
formula template, proof-script install checks, static package guard, and
shared/ABI rejection rules.

## Installed Artifact Checklist

When approved metadata exists and formula rendering succeeds, the proof must
verify these installed artifacts:

| Artifact | Required location | Evidence owner |
| --- | --- | --- |
| Homebrew prefix | `brew --prefix sparse-lu-ortho-local` | `check_installed_static_surface()` |
| Library directory | `<prefix>/lib` | `check_installed_static_surface()` |
| Static archive | `<prefix>/lib/libsparse_lu_ortho.a` | Formula `install` block and proof script |
| Public headers | `<prefix>/include/sparse` | Proof script |
| CMake config | `<prefix>/lib/cmake/Sparse/SparseConfig.cmake` | Proof script |
| CMake version file | `<prefix>/lib/cmake/Sparse/SparseConfigVersion.cmake` | Proof script |
| CMake targets file | `<prefix>/lib/cmake/Sparse/SparseTargets.cmake` | Proof script |
| CMake target location file | `<prefix>/lib/cmake/Sparse/SparseTargets-noconfig.cmake` | Proof script |
| pkg-config metadata | `<prefix>/lib/pkgconfig/sparse.pc` | Proof script |

## Static Package Assertions

The proof requires installed CMake metadata to prove the maintained static
package surface:

- `SparseTargets.cmake` must contain `Sparse::sparse_lu_ortho STATIC
  IMPORTED`;
- `SparseTargets-noconfig.cmake` must point at
  `${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a`;
- `sparse.pc` must remain static-link metadata and must not introduce provider
  or shared-library wording.

These checks align with `scripts/static_package_deferral_check.sh`, which also
guards `BUILD_SHARED_LIBS=ON` rejection, Makefile static archive installation,
CMake static archive installation, package metadata neutrality, and public
shared-library/dynamic ABI non-claims.

## Unsupported Surface Rejection

The formula and proof script reject unsupported installed surfaces:

- shared artifacts under `lib` or `bin`, including `.dylib`, `.so`, `.so.*`,
  and `.dll`;
- `Libs.private`;
- Homebrew/vcpkg/Conan/pkgsrc/distro provider wording in installed metadata;
- package-manager support wording in installed metadata;
- `SOVERSION`, `SONAME`, DLL, dylib, dynamic ABI, static/shared selector, or
  shared target metadata.

These rejection checks keep local formula proof scoped to the static source
package surface and prevent shared-library or ABI evidence from being inferred.

## Cleanup and Retry Expectations

The proof sets `UNINSTALL_ON_EXIT=1` before local formula installation and
uses an exit trap to uninstall `sparse-lu-ortho-local` when a later proof step
fails. Temporary archives, rendered formulas, tap paths, install logs, and
test logs live under a temporary root and are removed unless `--keep-temp` is
used for diagnostics.

Day 7 found no generated Homebrew proof outputs under `packaging/homebrew`.

## Claim Boundary

No installed package proof is earned on Day 7 because install execution is
blocked before rendering. Homebrew/core readiness, bottles, Linuxbrew, public
tap support, binary packages, package-manager distribution, shared-library
support, dynamic ABI support, runtime-loader behavior, and broad
package-manager support remain unclaimed.

## Day 8 Handoff

Day 8 should review the downstream `brew test` consumer path. If approved
license metadata is still absent, Day 8 should keep the test proof as a
contract review and record that no downstream formula test execution is
possible without a rendered and installed formula.

## Validation

Day 7 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
