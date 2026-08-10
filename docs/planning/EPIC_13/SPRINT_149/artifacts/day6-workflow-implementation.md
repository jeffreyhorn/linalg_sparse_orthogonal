# Sprint 149 Day 6: Workflow Implementation

## Purpose

Implement the selected Windows install/downstream workflow changes without
widening unsupported package claims.

## Files Changed

| File | Change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Promoted install/downstream lane wording to reviewed Windows CMake install/downstream validation and added CMake package metadata assertions. |
| `docs/planning/EPIC_13/SPRINT_149/WORKING_NOTES.md` | Recorded Day 6 implementation details and handoff. |
| `docs/planning/EPIC_13/SPRINT_149/artifacts/day6-workflow-implementation.md` | Published this implementation artifact. |

No `.c` or `.h` files were changed.

## Workflow Label Changes

| Surface | Before | After |
| --- | --- | --- |
| Windows install/downstream job name | `Windows supplemental CMake install/downstream confidence path` | `Windows reviewed CMake install/downstream validation path` |
| Windows install/downstream step name | `Run maintained supplemental CMake install/downstream proof` | `Run reviewed CMake install/downstream validation proof` |
| CTest support-scope message | no separate reviewed install-validation lane | reviewed install validation is CMake install/downstream scoped, with no reviewed Makefile or `pkg-config` execution parity |

## Added WIV Checks

| WIV Row | Implementation |
| --- | --- |
| WIV-09 | Reads installed `SparseTargets.cmake` and requires `add_library(Sparse::sparse_lu_ortho STATIC IMPORTED)`. |
| WIV-10 | Reads installed `SparseTargets.cmake` and requires `INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"`. |
| WIV-11 | Reads installed `SparseTargets-release.cmake` and requires `IMPORTED_LOCATION_RELEASE` plus `${_IMPORT_PREFIX}/lib/sparse_lu_ortho.lib`. |
| WIV-12 | Reads all installed CMake package files and rejects native or slash-normalized source/build path leaks. |

The workflow now also requires `SparseTargets-release.cmake` to be installed
alongside `SparseTargets.cmake`.

## Preserved Checks

Day 6 preserved existing proof for:

- Visual Studio CMake configure/build/install;
- installed `lib/sparse_lu_ortho.lib`;
- absence of installed DLLs;
- 19 installed headers;
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`,
  and `lib/pkgconfig/sparse.pc` presence;
- absence of shared/module imported metadata and imported `.so`, `.dylib`, or
  `.dll` locations;
- `sparse.pc` static archive description and unsupported wording rejection;
- maintained installed CMake example configure/build/run;
- exact-version generated installed CMake consumer configure/build/run;
- lower same-major mismatch-version configure rejection.

## Support-Tier Boundaries Preserved

The workflow wording now supports the narrow promoted claim:

> reviewed Windows CMake install/downstream validation for the maintained
> static-first package surface

The workflow still does not claim:

- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library support;
- dynamic ABI support;
- runtime-loader behavior;
- broad Windows parity.

`scripts/static_package_deferral_check.sh` remains Linux/macOS reviewed
package-contract ownership, as decided on Day 4.

## Hosted Evidence Requirement

The implementation is designed for hosted Windows. Local macOS review can
check YAML structure and text, but the reviewed lane is not earned until the
pull request passes:

`Windows reviewed CMake install/downstream validation path`

If hosted Windows reports different CMake target file naming, imported-location
formatting, or path text, treat that as a Sprint 149 workflow fix before
closeout.

## Local Review Plan

Day 6 local review should include:

1. `git diff --check`
2. trailing-whitespace search over `.github/workflows/windows-ci.yml` and
   Sprint 149 artifacts
3. unsupported-claim search for stale supplemental Windows install wording
4. YAML parsing if available locally
5. static review of PowerShell literal strings and path handling

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Workflow names and comments match the Day 4 decision. | Complete | Job/step names and comments now use reviewed Windows CMake install/downstream validation wording. |
| No unsupported Windows Makefile, package-manager, or shared-library claim is introduced. | Complete | Support-tier boundaries remain explicit in comments and this artifact. |
| Hosted validation requirements are ready for Day 13 closeout. | Complete | Hosted evidence requirement names the exact Windows job to verify. |
