# Sprint 134 Day 8 - Windows Install Validation Design

## Purpose

Day 8 audits the Windows install/downstream-consumer gap and decides the Day 9
implementation path. The goal is to add useful Windows CMake-first install
evidence without conflating it with Windows Makefile parity, package-manager
support, shared-library support, or staged CTest promotions.

## Current Windows Baseline

| Surface | Current state |
| --- | --- |
| Workflow | `.github/workflows/windows-ci.yml` |
| Reviewed job | `Windows enforced reviewed CMake consumer subset (MSVC)` |
| Runner | `windows-2022` |
| Generator | `Visual Studio 17 2022`, `x64` |
| Reviewed commands | CMake configure, CMake build, `ctest -N`, full `ctest` |
| Expected CTest count | `54` |
| Staged CTest exclusions | `test_threads`, `test_sprint4_integration`, `test_fuzz` |
| Current install-validation claim | No separate reviewed install-validation lane |
| Current Makefile claim | No Windows Makefile parity; Windows uses CMake exclusively |

The Windows workflow already proves the build-tree CMake consumer subset under
MSVC. It does not run `cmake --install`, does not configure an installed
downstream consumer through `find_package(Sparse)`, and does not inspect the
installed static package artifacts.

## Existing Local Proof Reuse Assessment

| Existing proof | Windows fit | Decision |
| --- | --- | --- |
| `tests/test_install.sh` | Unix Make install/`pkg-config` script; not a Windows CMake-first proof. | Do not reuse for Windows. |
| `tests/test_cmake_install.sh` | Validates the right package semantics, but is Bash/Unix-path oriented and expects Unix archive naming. | Use as the semantic template, not as the direct Windows job command. |
| `scripts/static_package_deferral_check.sh` | Bash text guard for static-first non-claims. | Keep as Unix/Linux/macOS package-contract guard unless Day 9 adds a native Windows equivalent. |
| `CMakeLists.txt` install/export rules | Already define static target export, config files, version file, and installed headers. | Reuse directly through a Windows `cmake --install` proof. |

## Selected Day 9 Path

Add a separate **supplemental Windows CMake install/downstream confidence job**
on Day 9.

The selected job should remain separate from the reviewed Windows CTest job. It
should run on `windows-2022` with the Visual Studio 2022 generator and prove:

1. Configure a fresh build tree with `-DCMAKE_INSTALL_PREFIX` set to a temporary
   prefix.
2. Build the project in `Release`.
3. Install with `cmake --install ... --config Release`.
4. Confirm the installed static library exists as `sparse_lu_ortho.lib`.
5. Confirm no `.dll` artifacts were installed.
6. Confirm installed headers count remains 19.
7. Confirm `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
   `SparseTargets.cmake` exist.
8. Configure `examples/cmake_example` with `CMAKE_PREFIX_PATH` pointing at the
   install prefix.
9. Build and run the installed downstream example executable.
10. Confirm `find_package(Sparse 2.2.0 EXACT REQUIRED)` works.
11. Confirm an intentionally mismatched older same-major version is rejected.

Classify this first Windows install job as supplemental confidence, not a
reviewed install-validation lane. Promotion to reviewed Windows install
validation should require hosted-runner history, runtime/flake evidence, and a
later support-tier decision.

## CTest Count Impact

| Candidate change | CTest count impact |
| --- | --- |
| Add a separate Windows install/downstream workflow job | No CTest membership change; `EXPECTED_WINDOWS_CTEST_COUNT` remains `54`. |
| Add installed downstream proof as a CTest test | Would require count update and generated install-prefix orchestration; not selected for Day 9. |
| Promote `test_threads`, `test_sprint4_integration`, or `test_fuzz` | Out of scope for Day 8; handled by Days 10-11. |

The selected Day 9 path should not change `CMakeLists.txt` test registration
and should not change the Windows CTest count.

## Windows Makefile Parity Separation

Windows install validation should stay CMake-first:

- do not add or claim Windows Makefile `install` support;
- do not add `pkg-config` as a Windows support requirement;
- do not require Windows package-manager setup;
- do not infer shared-library or dynamic ABI support from a static CMake
  install proof.

## Proposed Day 9 Workflow Shape

The PowerShell proof should use a temporary prefix and a separate build tree.
The exact script can be embedded in `.github/workflows/windows-ci.yml` unless a
future review asks for a reusable script.

Expected command shape:

```powershell
$prefix = Join-Path $env:RUNNER_TEMP "sparse-install"
cmake -S . -B build-install -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-install --config Release
cmake --install build-install --config Release
```

The downstream proof should configure `examples/cmake_example` with:

```powershell
cmake -S examples/cmake_example -B build-installed-example `
  -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$prefix"
cmake --build build-installed-example --config Release
```

Then run `build-installed-example\Release\example.exe` and require `OK` in the
output.

## Local Evidence

| Check | Result |
| --- | --- |
| CMake configure for local CTest registration audit | Passed |
| Local `ctest -N` registration count | `57` on this non-Windows host |
| Windows count reconciliation | `57 - 3 staged Windows exclusions = 54`, matching workflow expectation |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header diff scan | No `.c` or `.h` changes |

This local host cannot execute the MSVC/Windows install proof. Day 9 should
validate the workflow YAML and keep the hosted Windows job as the platform
source of truth.

## Residual Windows Queue

| Residual | Status |
| --- | --- |
| Reviewed Windows install-validation lane | Deferred pending supplemental hosted-runner evidence. |
| Windows Makefile parity | Deferred non-claim. |
| Windows `pkg-config` consumer support | Deferred non-claim. |
| Windows package-manager support | Deferred non-claim. |
| Windows shared-library/dynamic ABI support | Deferred by Sprint 133 product decision. |
| Windows staged CTest exclusions | Days 10-11 owner. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Windows install validation has an implementation or deferral decision. | Complete | Selected a supplemental Windows CMake install/downstream confidence job for Day 9. |
| CTest count implications are explicit before workflow edits. | Complete | Selected path leaves `EXPECTED_WINDOWS_CTEST_COUNT=54` unchanged. |
| Windows Makefile parity is not conflated with CMake-first support. | Complete | Design explicitly excludes Makefile install, `pkg-config`, package-manager, and shared-library claims. |
