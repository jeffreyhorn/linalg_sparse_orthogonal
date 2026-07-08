# Day 6 Windows Install-Validation Lane Design

## Purpose

Day 6 designs the proof required before claiming Windows installed-package
support. The current Windows lane is reviewed CMake-first consumer proof, not
a separate install-validation lane. Day 6 names the exact evidence a future
reviewed install-validation lane would need and the criteria Day 7 should use
to add or defer it.

No workflow, CMake, documentation, or package claim changes are made on Day 6.

## Current Reviewed Windows Surface

`.github/workflows/windows-ci.yml` currently provides:

| Surface | Evidence |
|---|---|
| runner | `windows-2022`, pinned for the VS 2022 generator |
| configure | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` |
| build | `cmake --build build --config Release` |
| registration guard | `ctest --test-dir build -C Release -N` with `EXPECTED_WINDOWS_CTEST_COUNT=51` |
| execution | `ctest --test-dir build -C Release --output-on-failure` |
| claim boundary | workflow comments and output state CMake-first consumer proof only, no Makefile parity, and no separate install-validation lane |

## Current Staged Exclusions

| Exclusion | Current reason |
|---|---|
| `test_threads` | Source uses pthread APIs directly; not registered on Windows. |
| `test_sprint4_integration` | Coupled to pthread-dependent test path; not registered on Windows. |
| `test_fuzz` | Gated off for Windows/MSVC; fuzz/property evidence is not reviewed on Windows. |
| Makefile reviewed wrappers | Unix-oriented; Windows maintained path is CMake. |
| dead-code flow | Linux-side tooling and workflow ownership. |
| separate install validation | No reviewed `cmake --install` plus downstream installed consumer proof exists on Windows. |
| shared-library/DLL behavior | Static-first support tier; no DLL/import-library/runtime-loader proof exists. |
| package-manager support | No vcpkg, Chocolatey, winget, or other package-manager proof exists. |

## Required Windows Install-Validation Sequence

A reviewed Windows install-validation lane should prove:

1. Configure:
   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
     -DCMAKE_INSTALL_PREFIX="$installPrefix"
   ```
2. Build:
   ```powershell
   cmake --build build --config Release
   ```
3. Install:
   ```powershell
   cmake --install build --config Release
   ```
4. Verify installed package shape:
   - static library/import artifact expected for the static-first target;
   - public headers under the install prefix;
   - `SparseConfig.cmake`;
   - `SparseConfigVersion.cmake`;
   - `SparseTargets.cmake` and config-specific target files.
5. Configure downstream installed consumer:
   ```powershell
   cmake -S examples/cmake_example -B consumer-build `
     -DCMAKE_PREFIX_PATH="$installPrefix"
   ```
6. Build downstream consumer:
   ```powershell
   cmake --build consumer-build --config Release
   ```
7. Run downstream consumer and require `OK` output.
8. Preserve static-first non-claims:
   - no DLL/shared-library claim;
   - no dynamic ABI claim;
   - no package-manager claim.

## Reviewed Count and Staged-Exclusion Impact

If the install-validation lane is added as a separate job that only configures,
installs, builds, and runs `examples/cmake_example`, it should not change the
main Windows CTest registration count. The existing `EXPECTED_WINDOWS_CTEST_COUNT`
guard should remain unchanged unless test registration is intentionally
modified.

Any change that registers additional Windows tests must include:

- `ctest -N` evidence;
- expected count update;
- staged-exclusion wording update;
- explanation of whether `test_threads`, `test_sprint4_integration`, or
  `test_fuzz` moved into reviewed scope.

## MSVC and Environment Risks

| Risk | Impact | Mitigation if promoted |
|---|---|---|
| generator availability | reviewed lane depends on VS 2022 generator | keep `windows-2022` pin |
| config-specific artifacts | Release builds may install config-specific target files | check CMake package files generically rather than hard-coding every generated file name |
| path quoting | Windows temp prefixes and PowerShell quoting can fail downstream configure | use explicit variables and quoted paths |
| runtime path confusion | static-first target should not depend on DLL lookup | preserve no-DLL expectation and run only the static consumer |
| reviewed-scope drift | install lane could be mistaken for Makefile, package-manager, or full Windows parity | workflow/docs must say CMake install-validation only |
| duplicated CMake build cost | lane repeats configure/build separate from reviewed CTest lane | only add if Windows installed-package support needs reviewed proof |

## Deferral Criteria

Day 7 should defer reviewed Windows install validation if any of these hold:

1. Current public wording remains accurate with Windows as reviewed CMake-first
   consumer subset only.
2. Adding install validation would broaden Windows support beyond current
   Sprint 115 needs.
3. PowerShell/install-prefix/downstream-consumer scaffolding requires more
   implementation than a bounded package decision should introduce.
4. The lane risks implying Windows Makefile parity, package-manager support,
   shared-library/DLL support, or dynamic ABI support.
5. Reviewed CTest count and staged exclusions would become less clear.

## Support Claims to Fence Until Proof Lands

- Windows installed-package support.
- Windows Makefile parity.
- Windows `pkg-config` support.
- Windows package-manager support.
- Windows shared-library, DLL, import-library, runtime-loader, or dynamic ABI
  support.
- Windows `test_threads`, `test_sprint4_integration`, or `test_fuzz` reviewed
  parity.

## Day 7 Decision Checklist

Before changing Windows CI on Day 7, answer:

1. Does reviewed Windows install validation materially improve support truth
   beyond the current CMake-first reviewed subset?
2. Can a lane prove `cmake --install` plus installed
   `find_package(Sparse)` consumer without changing CTest registration?
3. Are install prefix, Release config, path quoting, and generated package
   files handled robustly?
4. Are workflow output and docs ready to describe the promoted or deferred
   status?
5. Is validation clear for every touched file?

## Non-Claims

- No Windows workflow was changed on Day 6.
- No Windows install-validation lane was added on Day 6.
- No Windows installed-package support claim was added on Day 6.
- No Windows Makefile, package-manager, shared-library/DLL, dynamic ABI, or
  runtime-loader support claim changed on Day 6.
- No CTest membership changed on Day 6.

## Day 6 Validation

Day 6 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 6 because no `.c` or `.h` files
changed.
