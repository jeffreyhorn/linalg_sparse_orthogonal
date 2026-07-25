# Sprint 134 Day 9 - Windows Install Validation Implementation

## Purpose

Day 9 implements the Day 8 decision to add Windows CMake-first install and
downstream-consumer evidence without promoting Windows to a separate reviewed
install-validation lane.

## Implemented Workflow Change

Added a standalone supplemental job to `.github/workflows/windows-ci.yml`:

| Field | Value |
| --- | --- |
| Job id | `install-and-downstream` |
| Job name | `Windows supplemental CMake install/downstream confidence path` |
| Runner | `windows-2022` |
| Generator | `Visual Studio 17 2022`, `x64` |
| Tier | Supplemental CMake-first install/downstream confidence |

The existing reviewed Windows job remains unchanged:

- CMake configure
- CMake build
- `ctest -N`
- expected CTest count check
- full `ctest`

## Windows Proof Coverage

The supplemental PowerShell proof now checks:

1. Configure a fresh install build with `CMAKE_INSTALL_PREFIX` under
   `RUNNER_TEMP`.
2. Build `Release` with MSVC.
3. Install with `cmake --install`.
4. Confirm `lib/sparse_lu_ortho.lib` exists.
5. Confirm no `.dll` artifacts were installed.
6. Confirm 19 public headers were installed.
7. Confirm `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
   `SparseTargets.cmake` were installed.
8. Configure `examples/cmake_example` with `CMAKE_PREFIX_PATH` pointing at the
   install prefix.
9. Build and run the installed downstream example, requiring `OK` output.
10. Confirm exact installed package version resolution works.
11. Confirm an older same-major mismatched version is rejected.

The script uses explicit checked native-command execution so failed CMake
configure, build, install, or downstream configure/build commands fail the job
at the point of failure.

## Support-Tier Wording

Updated the Windows workflow comment, README CI summary, INSTALL platform table
and validation interpretation, and maintainer guide to classify the new lane as
supplemental CMake install/downstream confidence.

The wording preserves these boundaries:

- Windows reviewed support remains the MSVC CMake configure/build/CTest subset.
- The new Windows install job is supplemental confidence, not a reviewed
  install-validation lane.
- Windows Makefile parity remains a non-claim.
- Windows `pkg-config`, package-manager support, shared-library packaging, and
  dynamic ABI compatibility remain non-claims or deferred product decisions.
- Windows staged CTest exclusions remain owned by Days 10-11.

## CTest Count Impact

| Surface | Impact |
| --- | --- |
| `CMakeLists.txt` test registration | Unchanged |
| `.github/workflows/windows-ci.yml` `EXPECTED_WINDOWS_CTEST_COUNT` | Unchanged at `54` |
| staged Windows exclusions | Unchanged: `test_threads`, `test_sprint4_integration`, `test_fuzz` |

The selected Day 9 implementation adds a separate workflow job, not a CTest
test, so no CTest count update is required.

## Local Evidence

| Check | Result |
| --- | --- |
| YAML parse for `.github/workflows/windows-ci.yml` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `bash tests/test_cmake_install.sh` | Passed 21 checks, 0 failures, 0 skips |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header/CMake registration diff scan | No `.c`, `.h`, or `CMakeLists.txt` changes |

This host does not have `pwsh`, the Visual Studio 2022 generator, or MSVC, so
the new Windows PowerShell job cannot be executed locally. The local CMake
install/export proof validates the package semantics, while the hosted
`windows-2022` lane is the Windows-specific source of truth.

## Rollback Notes

If the hosted Windows supplemental job fails, remove only the
`install-and-downstream` job and revert the corresponding supplemental wording.
The reviewed Windows CMake configure/build/CTest job does not depend on the new
install proof.

If the failure is caused by an installed-artifact path difference, fix the
PowerShell checks only after confirming the actual CMake/MSVC install output.
Do not widen Windows install-validation claims until hosted-runner evidence is
stable.

## Residual Windows Queue

| Residual | Status |
| --- | --- |
| Reviewed Windows install-validation lane | Deferred pending hosted-runner history from the supplemental lane. |
| Windows Makefile parity | Deferred non-claim. |
| Windows `pkg-config` consumer support | Deferred non-claim. |
| Windows package-manager support | Deferred non-claim. |
| Windows shared-library/dynamic ABI support | Deferred by Sprint 133 product decision. |
| Windows staged CTest exclusions | Days 10-11 owner. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected Windows install decision is reflected in changed surfaces. | Complete | Added the supplemental `install-and-downstream` Windows workflow job and aligned support wording. |
| Expected test membership impact is documented. | Complete | CTest registration and `EXPECTED_WINDOWS_CTEST_COUNT=54` remain unchanged. |
| Windows-only validation gap is explicit rather than hidden. | Complete | Artifact records the local inability to execute the MSVC/PowerShell job and assigns hosted `windows-2022` as source of truth. |
