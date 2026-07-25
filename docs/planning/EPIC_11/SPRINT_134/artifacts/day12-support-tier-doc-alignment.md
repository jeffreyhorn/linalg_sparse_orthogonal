# Sprint 134 Day 12 - Support-Tier Documentation Alignment

## Purpose

Day 12 aligns public docs, maintainer docs, workflow comments, and Sprint 134
working notes with the implemented platform decisions from Days 4, 7, 9, and
11. Historical Day 1 and Day 2 audit artifacts are intentionally left as
pre-decision snapshots; this artifact records the current Sprint 134 platform
truth.

## Final Platform Truth Table

| Platform | Reviewed evidence | Supplemental evidence | Staged/deferred/non-claims |
| --- | --- | --- | --- |
| Linux | Reviewed Makefile compile-quality path, reviewed CMake parity path, reviewed dead-code path, and reviewed static-first package-contract lane. | Direct runtime, sanitizer, `bench-fast`, TSan, and coverage signals. | Package-manager support, shared-library packaging, dynamic ABI compatibility, and runtime-loader behavior remain non-claims. |
| macOS | Reviewed Apple Clang compile-quality, CMake parity, wall-check, and sanitizer path. | Homebrew GCC second-compiler path, static-first Make install/`pkg-config` confidence, and CMake install/export confidence. | Full reviewed macOS install/export parity, package-manager support, shared-library packaging, and dynamic ABI compatibility remain non-claims. |
| Windows | Reviewed MSVC CMake configure/build/`ctest -N`/full `ctest` subset with `EXPECTED_WINDOWS_CTEST_COUNT=54`. | CMake install/downstream confidence job for the CMake-first consumer story. | Separate reviewed install-validation parity, Windows Makefile parity, Windows `pkg-config`, package-manager support, shared-library packaging, dynamic ABI compatibility, and pthread/POSIX-backed staged tests remain non-claims or staged. |

## Aligned Surfaces

| Surface | Day 12 status |
| --- | --- |
| `.github/workflows/ci.yml` | Linux reviewed static-first package-contract lane and static-first non-claims are explicit. |
| `.github/workflows/macos-ci.yml` | macOS package jobs are explicitly supplemental and do not claim reviewed install/export parity. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake subset, supplemental install/downstream job, staged blockers, and non-claims are explicit. |
| `README.md` | CI summary distinguishes Linux reviewed package contract, macOS supplemental package confidence, Windows supplemental install/downstream confidence, and staged Windows tests. |
| `INSTALL.md` | Platform table and verification section preserve reviewed/supplemental/local/non-claim boundaries. |
| `docs/maintainer_guide.md` | Maintainer support interpretation names the final package tiers, Windows count, staged blockers, and promotion gates. |
| `docs/planning/EPIC_11/SPRINT_134/WORKING_NOTES.md` | Live Sprint 134 baseline updated to include Windows supplemental CMake install/downstream confidence and final claim fences. |

## Day 12 Updates

Updated `WORKING_NOTES.md` to:

- include the Windows supplemental CMake install/downstream confidence job in
  the live Sprint 134 baseline;
- update the Windows workflow role in the input and candidate-surface tables;
- replace pre-decision Linux/macOS/Windows inference fences with final
  post-decision claim fences.

No workflow, package script, source, header, or CMake registration behavior was
changed on Day 12.

## Claim Drift Scan Notes

The drift scan intentionally found older Day 1 and Day 2 Sprint artifacts that
describe the pre-decision baseline, including:

- Linux install CI as not yet promoted;
- macOS CMake install/export confidence as not yet selected;
- Windows install/downstream confidence as not yet added.

Those artifacts remain historically accurate for their day. The current truth
is captured in this Day 12 artifact and the live `WORKING_NOTES.md` baseline.

## Validation Evidence

| Check | Result |
| --- | --- |
| YAML parse for Linux, macOS, and Windows workflows | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header/CMake/package-script diff scan | No `.c`, `.h`, `CMakeLists.txt`, package proof script, or static deferral script changes |

## Preserved Non-Claims

| Area | Day 12 status |
| --- | --- |
| Shared-library packaging | Still deferred and unsupported. |
| Dynamic ABI compatibility | Still deferred and unsupported. |
| Runtime-loader behavior | Still not claimed. |
| Package-manager support | Still not claimed. |
| Full reviewed macOS install/export parity | Still not claimed. |
| Separate reviewed Windows install-validation parity | Still not claimed. |
| Windows Makefile parity | Still not claimed. |
| Windows `pkg-config` support | Still not claimed. |
| Windows reviewed thread/fuzz/property coverage | Still staged. |

## Residual Wording Queue

| Residual | Status |
| --- | --- |
| Historical Day 1/Day 2 baseline artifacts | Leave unchanged as snapshots; use Day 12 and later artifacts for final truth. |
| Day 13 integrated validation record | Pending Day 13. |
| Day 14 final closeout/register | Pending Day 14. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public and maintainer docs agree with implemented platform decisions. | Complete | README, INSTALL, maintainer guide, and workflow comments name reviewed/supplemental/staged boundaries consistently. |
| No platform install parity or staged-lane claim is overstated. | Complete | macOS and Windows install confidence remains supplemental; staged Windows tests remain outside reviewed membership. |
| Sprint 133 static-first package contract remains intact. | Complete | Static-first non-claims remain explicit across support docs and workflow comments. |
