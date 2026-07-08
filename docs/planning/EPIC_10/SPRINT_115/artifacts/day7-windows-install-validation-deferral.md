# Day 7 Windows Install-Validation Deferral

## Purpose

Day 7 applies the Day 6 Windows install-validation criteria and decides
whether to add a reviewed Windows `cmake --install` plus downstream installed
consumer lane, or explicitly defer that proof.

## Decision

Defer a separate reviewed Windows install-validation lane in Sprint 115.

No Windows workflow change is made. Windows remains a reviewed MSVC
CMake-first consumer subset:

- configure with `Visual Studio 17 2022` on `windows-2022`;
- build Release;
- inspect `ctest -N` against `EXPECTED_WINDOWS_CTEST_COUNT=51`;
- run full `ctest` for the registered reviewed subset;
- keep `test_threads`, `test_sprint4_integration`, and `test_fuzz` as staged
  exclusions.

## Evidence Reviewed

| Evidence | Result |
|---|---|
| `.github/workflows/windows-ci.yml` | Already states Windows is reviewed CMake-first consumer proof only, with no Makefile parity and no separate install-validation lane. |
| workflow CTest guard | `EXPECTED_WINDOWS_CTEST_COUNT=51` protects the reviewed registered subset. |
| workflow output | Prints staged exclusions and explicitly says no separate reviewed install-validation lane. |
| `INSTALL.md` | Says Windows is the reviewed CMake subset and CMake-first consumer path, not separate install validation. |
| `README.md` | Summarizes Windows as reviewed CMake subset and CMake-first consumer story. |
| `docs/maintainer_guide.md` | Records no Windows install-validation parity and preserves staged exclusions. |
| `CMakeLists.txt` | Keeps pthread-based and fuzz/property tests gated out of Windows/MSVC reviewed registration. |

## Criteria Application

| Criterion | Assessment |
|---|---|
| Material support-truth improvement | Limited for Sprint 115. Current Windows wording already avoids installed-package support claims. |
| Can lane prove `cmake --install` without CTest changes? | Likely possible as a separate job, but would require new PowerShell scaffolding and generated package-file checks. |
| Path/config robustness | Needs careful temp-prefix quoting and Release config-specific installed target handling. |
| Reviewed-count clarity | Deferral keeps `EXPECTED_WINDOWS_CTEST_COUNT=51` and staged exclusions unchanged. |
| Overclaim risk | A new install lane could be misread as Windows package-manager, Makefile, shared-library/DLL, or dynamic ABI support unless heavily fenced. |

## Deferral Contract

Until a future sprint explicitly adds reviewed Windows install validation:

- Windows reviewed support remains the CMake-first consumer subset.
- Windows does not claim `cmake --install` reviewed proof.
- Windows does not claim installed package support beyond the current CMake
  build/test subset.
- Windows does not claim Makefile or `pkg-config` parity.
- Windows does not claim package-manager support.
- Windows does not claim shared-library, DLL/import-library, dynamic ABI, or
  runtime-loader behavior.
- The Windows CTest registration count remains `51` unless future work
  intentionally changes reviewed test membership.

## Missing Proof for Future Promotion

A future reviewed Windows install-validation lane should add:

1. `cmake --install build --config Release` into a temp prefix.
2. Verification of installed static library, public headers, and CMake package
   files under that prefix.
3. A downstream installed consumer configured with `CMAKE_PREFIX_PATH`.
4. `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho` link proof.
5. Downstream build and run proof requiring `OK` output.
6. Explicit no-DLL/no-dynamic-ABI/no-package-manager wording.
7. Confirmation that the main Windows `ctest -N` count remains unchanged, or
   an intentional reviewed-count update with staged-exclusion explanation.

## Support Wording Assessment

No wording changes are required on Day 7:

- `.github/workflows/windows-ci.yml` already says no separate reviewed
  install-validation lane.
- `INSTALL.md` already says Windows is reviewed CMake subset only and does not
  claim separate install validation.
- `README.md` already says Windows enforces the reviewed CMake subset and
  CMake-first consumer story.
- `docs/maintainer_guide.md` already records Windows install-validation as a
  non-claim.

## Non-Claims

- No Windows install-validation lane was added.
- No Windows installed-package support claim was added.
- No Windows Makefile parity claim was added.
- No Windows `pkg-config` or package-manager support claim was added.
- No Windows shared-library, DLL/import-library, dynamic ABI, or runtime-loader
  support claim was added.
- No Windows CTest membership changed.

## Day 7 Validation

Day 7 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 7 because no `.c` or `.h` files
changed.
