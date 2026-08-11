# Sprint 149 Day 5: Workflow Implementation Design

## Purpose

Design the workflow edits required by the Day 4 conditional-promotion decision
before changing `.github/workflows/windows-ci.yml`.

The design promotes only the Windows CMake install/downstream validation lane
after criteria-backed assertions are added. It does not split the workflow and
does not widen Windows support claims beyond the hosted MSVC CMake install and
installed-consumer path.

## Current Workflow Ownership

| Field | Current Value |
| --- | --- |
| Workflow file | `.github/workflows/windows-ci.yml` |
| Triggers | `push` and `pull_request` to `main` and `master` |
| Reviewed CTest job id | `build-and-test` |
| Reviewed CTest job name | `Windows enforced reviewed CMake consumer subset (MSVC)` |
| Install/downstream job id | `install-and-downstream` |
| Current install/downstream job name | `Windows supplemental CMake install/downstream confidence path` |
| Runner | `windows-2022` |
| Shell | `pwsh` |
| Generator | `Visual Studio 17 2022` |
| Architecture | `x64` |
| Build config | `Release` |

The reviewed CTest job is not part of the Sprint 149 implementation target.
It must stay intact while the package lane changes.

## Selected Workflow Shape

Day 5 selects a **single-lane promotion design**:

- keep job id `install-and-downstream`;
- rename the job after checks land to
  `Windows reviewed CMake install/downstream validation path`;
- rename the main step to
  `Run reviewed CMake install/downstream validation proof`;
- update top-level workflow comments to state that Windows now has reviewed
  CMake install/downstream validation for the maintained static-first package
  surface;
- preserve explicit non-claims for Windows Makefile parity, Windows
  `pkg-config` execution parity, shared-library support, dynamic ABI support,
  runtime-loader behavior, package-manager support, and broad Windows parity.

No separate supplemental Windows package job is needed in Sprint 149 because
Day 4 rejected the split-lane option. `sparse.pc` checks remain in the
reviewed Windows lane as metadata checks only, not as `pkg-config` execution
proof.

## Planned Job And Step Names

| Surface | Current Name | Planned Name |
| --- | --- | --- |
| `install-and-downstream` job | `Windows supplemental CMake install/downstream confidence path` | `Windows reviewed CMake install/downstream validation path` |
| main PowerShell step | `Run maintained supplemental CMake install/downstream proof` | `Run reviewed CMake install/downstream validation proof` |

The existing `build-and-test` job name remains unchanged:

`Windows enforced reviewed CMake consumer subset (MSVC)`

## Comment Wording Design

Top-level workflow comments should change from supplemental wording to:

- Windows enforces the reviewed CMake subset through configure/build,
  `ctest -N`, and full `ctest`;
- Windows also carries reviewed CMake install/downstream validation for the
  maintained static-first package surface;
- the install/downstream lane checks installed static `.lib`, headers, CMake
  package metadata, `sparse.pc` metadata, installed downstream consumers,
  exact-version behavior, mismatch-version rejection, and absence of DLL/shared
  imported metadata;
- Windows still does not claim Makefile parity, `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, or broad Windows parity;
- `scripts/static_package_deferral_check.sh` remains part of Linux/macOS
  reviewed package-contract ownership, not a Windows claim.

The CTest inspection step should update the last `Write-Host` line to avoid
claiming "no separate reviewed install-validation lane" after Day 6 promotes
the package lane. It should instead say:

`Windows reviewed install validation remains CMake install/downstream scoped: no reviewed Makefile parity and no pkg-config execution parity.`

## PowerShell Assertion Insertion Plan

The current PowerShell script already installs package files and builds
downstream consumers. Day 6 should add WIV-09 through WIV-12 immediately after
the package-file existence loop and before the current combined
`$cmakePackageText` shared-metadata rejection.

Planned variables:

```powershell
$cmakePackageDir = Join-Path $prefix "lib/cmake/Sparse"
$targetsFile = Join-Path $cmakePackageDir "SparseTargets.cmake"
$targetsReleaseFile = Join-Path $cmakePackageDir "SparseTargets-release.cmake"
$cmakePackageText = (Get-ChildItem -Path $cmakePackageDir -File |
  ForEach-Object { Get-Content -Path $_.FullName -Raw }) -join "`n"
$normalizedPrefix = ($prefix -replace "\\", "/")
```

Planned assertions:

| WIV Row | Assertion Design | Failure Message Intent |
| --- | --- | --- |
| WIV-09 | `SparseTargets.cmake` must contain `add_library(Sparse::sparse_lu_ortho STATIC IMPORTED)`. | Installed CMake package target is not explicitly static imported. |
| WIV-10 | `SparseTargets.cmake` must contain `INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"`. | Installed CMake package target does not use install-prefix include metadata. |
| WIV-11 | `SparseTargets-release.cmake` must contain `IMPORTED_LOCATION_RELEASE` and an installed static `.lib` path using `${_IMPORT_PREFIX}/lib/sparse_lu_ortho.lib`. | Installed CMake package target does not point at the installed static `.lib`. |
| WIV-12 | combined installed CMake package text must not contain the repository path or the build directory path. | Installed CMake package metadata leaked source/build paths. |

The existing WIV-13 check should remain after these positive checks:

- reject `SHARED IMPORTED`;
- reject `MODULE IMPORTED`;
- reject imported `.so`, `.dylib`, or `.dll` locations.

## Source And Build Path Leak Design

The workflow can compute local paths after checkout and build variable setup:

```powershell
$sourceRoot = (Get-Location).Path
$buildPath = (Resolve-Path $build).Path
```

After CMake package files are installed and read, Day 6 should reject both:

- native Windows path text;
- slash-normalized path text.

This guards against generated package files embedding either
`D:\a\...\linalg_sparse_orthogonal` or `D:/a/.../linalg_sparse_orthogonal`.

## Reviewed/Supplemental Split Decision

No split is planned.

| Candidate Split | Day 5 Decision |
| --- | --- |
| Reviewed CMake install/export plus supplemental `sparse.pc` metadata | Rejected because `sparse.pc` is installed package metadata and can be checked without claiming `pkg-config` execution. |
| Reviewed installed consumer proof plus supplemental metadata proof | Rejected because package metadata is core to CMake install/downstream validation. |
| Reviewed Windows CMake install/downstream lane plus Linux/macOS-only static deferral guard | Accepted; WIV-19 remains Linux/macOS reviewed package-contract ownership. |

## Hosted Evidence Expectations

The following evidence cannot be fully proven locally on the macOS development
machine:

- hosted `windows-2022` Visual Studio generator availability;
- hosted Windows CMake configure/build/install behavior;
- hosted Windows installed package metadata content;
- hosted Windows downstream example and exact-version executable behavior;
- hosted mismatch-version configure failure behavior.

Day 13 closeout must treat promotion as pending until the PR has a passing
`Windows reviewed CMake install/downstream validation path` job.

## Local Review Checklist Before Implementation

Day 6 should run local documentation/workflow checks after editing:

1. `git diff --check`
2. trailing-whitespace search over `.github/workflows/windows-ci.yml` and
   Sprint 149 artifacts
3. unsupported-claim search for stale supplemental wording and overbroad
   Windows package claims
4. YAML parse if a local parser is available
5. static inspection of PowerShell variable names and regex escaping

No `.c` or `.h` files are expected to change on Day 6, so the full C quality
gate is not required unless implementation scope expands into source code.

## Day 6 Handoff

Day 6 should edit `.github/workflows/windows-ci.yml` only:

1. update top-level comments;
2. keep the CTest job intact except for the final support-scope message;
3. rename the install/downstream job and step;
4. add WIV-09 through WIV-12 assertions;
5. preserve existing static `.lib`, no-DLL, header, package-file,
   `sparse.pc`, consumer, exact-version, and mismatch-version checks;
6. run local workflow/documentation hygiene checks.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Workflow edits are planned before modifying CI files. | Complete | This artifact defines job names, comments, assertion insertion points, and local checks before Day 6 edits. |
| Job names and comments express the support tier precisely. | Complete | Planned names and comment wording use reviewed Windows CMake install/downstream validation, not broad package parity. |
| Hosted-only residual risk is recorded before implementation. | Complete | Hosted evidence expectations identify what remains pending until PR CI runs. |
