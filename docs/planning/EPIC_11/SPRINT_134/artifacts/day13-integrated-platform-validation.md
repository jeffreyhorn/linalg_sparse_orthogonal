# Sprint 134 Day 13 - Integrated Platform Validation

## Purpose

Day 13 runs integrated validation for the Sprint 134 platform and package
surfaces touched so far. The goal is to confirm that Linux reviewed package
contract proof, macOS supplemental package confidence, Windows supplemental
install/downstream confidence, and Windows staged-test boundaries all have
matching validation evidence before closeout.

## Diff Surface

| Surface | Status |
| --- | --- |
| Workflows | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` changed |
| Support docs | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` changed |
| Sprint artifacts | `docs/planning/EPIC_11/SPRINT_134` added/updated |
| C sources/headers | No `.c` or `.h` changes |
| CMake registration | No `CMakeLists.txt` changes |
| Package proof scripts | Post-PR CI follow-up changed `tests/test_install.sh`; `tests/test_cmake_install.sh` and `scripts/static_package_deferral_check.sh` unchanged |

Because no C source or public header changed, the sprint rule does not require
`make format && make lint && make test`.

## Integrated Validation Results

| Check | Result |
| --- | --- |
| YAML parse for Linux, macOS, and Windows workflows | Passed |
| Shell syntax for `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh` | Passed |
| `bash tests/test_install.sh` | Passed 22 checks, 0 failures |
| `bash tests/test_cmake_install.sh` | Passed 21 checks, 0 failures, 0 skips |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| Local CMake configure for CTest registration audit | Passed |
| Local `ctest -N` registration count | `57` on this non-Windows host |
| Windows count reconciliation | `57 - 3 staged Windows exclusions = 54` |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header/CMake registration diff scan | No `.c`, `.h`, or `CMakeLists.txt` changes |
| package proof script diff scan | `tests/test_install.sh` changed to parse `pkg-config --cflags` tokens |
| temporary audit build cleanup | `build-sprint134-day13` removed |

Post-PR CI follow-up reran `bash -n tests/test_install.sh`,
`bash tests/test_install.sh`, and `bash scripts/static_package_deferral_check.sh`;
all passed. The install proof passed 22 checks with 0 failures.

## Workflow-Equivalent Evidence

| Workflow surface | Local evidence | Limit |
| --- | --- | --- |
| Linux reviewed static-first package-contract lane | `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh` passed locally. | Hosted Linux CI remains the final runner-specific proof. |
| macOS supplemental Make install/`pkg-config` and CMake install/export confidence | Same package proof scripts passed locally. | This host is not the hosted macOS runner. |
| Windows reviewed CMake subset and staged count | Local CMake configure plus `ctest -N` registered 57 non-Windows tests and reconciled to Windows 54 after staged exclusions. | This host cannot run the MSVC/Visual Studio generator. |
| Windows supplemental CMake install/downstream confidence | Local CMake install/export proof passed; Windows workflow YAML parsed. | This host has neither `pwsh` nor MSVC, so the PowerShell job is hosted-runner proof only. |

## Claim Drift Scan

The claim-drift scan found only:

- explicit non-claim wording, such as “not a reviewed macOS install/export
  parity” and “not a reviewed Windows install-validation lane”;
- historical Sprint 134 notes that mention the older Sprint 112 Windows count
  only to explain why it was corrected earlier in this sprint.

No positive overclaim was found for:

- shared-library packaging;
- dynamic ABI compatibility;
- package-manager support;
- runtime-loader behavior;
- full reviewed macOS install/export parity;
- separate reviewed Windows install-validation parity;
- Windows Makefile parity;
- Windows reviewed thread/fuzz/property coverage.

## Residual Validation Limits

| Residual | Status |
| --- | --- |
| Hosted macOS runner behavior for supplemental CMake install/export | Pending CI evidence. |
| Hosted Windows runner behavior for supplemental PowerShell install/downstream job | Pending CI evidence. |
| Windows staged-test promotion proof | Deferred; current tests remain pthread/POSIX-bound. |
| Full C quality gate | Not required because no `.c` or `.h` files changed. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every touched platform surface has matching validation evidence. | Complete | Workflow YAML, package proofs, CTest registration, static deferral, and docs hygiene all passed. |
| Required quality gates pass or blockers are explicit. | Complete | Required gates passed; full C gate is not required by diff surface. |
| Validation evidence is ready for closeout and PR review. | Complete | This artifact consolidates Day 13 results and residual hosted-runner limits. |
