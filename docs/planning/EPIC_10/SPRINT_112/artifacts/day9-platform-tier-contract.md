# Day 9 Platform-Tier Contract

## Purpose

Day 9 defines the Sprint 112 platform support contract after the static-first
package decision and the refreshed install/export consumer proof from Days 6-8.
The contract separates reviewed CI evidence from supplemental, local-only,
staged, and unsupported lanes so later Windows, macOS, and documentation work
does not infer support from unrelated package or header stability evidence.

## Evidence Inputs

| Evidence source | Platform | Interpretation |
|---|---|---|
| `.github/workflows/ci.yml` | Linux | Strongest reviewed source of truth: reviewed Makefile compile-quality, reviewed CMake parity, and dead-code completeness. |
| `.github/workflows/ci.yml::build-and-test` | Linux | Supplemental direct runtime, sanitizer, and bounded `bench-fast` signal. |
| `.github/workflows/ci.yml::tsan` | Linux | Supplemental ThreadSanitizer and OpenMP race-detection signal with documented runtime suppressions and staged LOBPCG omission. |
| `.github/workflows/ci.yml::coverage` | Linux | Supplemental coverage report and threshold signal. |
| `.github/workflows/macos-ci.yml` Apple Clang leg | macOS | Reviewed macOS lane: Makefile compile-quality, CMake parity, `wall-check`, and sanitizer checks. |
| `.github/workflows/macos-ci.yml` Homebrew GCC leg | macOS | Supplemental second-compiler direct build/test and `wall-check` coverage. |
| `.github/workflows/macos-ci.yml::install-and-pkgconfig` | macOS | Supplemental static-first Make install and `pkg-config` confidence; not reviewed install/export parity. |
| `.github/workflows/windows-ci.yml` | Windows | Reviewed MSVC CMake-first consumer subset: configure, build, `ctest -N`, and full `ctest`. |
| `bash tests/test_install.sh` | Local Unix-side | Local direct Make install/uninstall and `pkg-config` proof; not itself a cross-platform reviewed CI lane. |
| `bash tests/test_cmake_install.sh` | Local Unix-side | Local direct CMake install/export and `find_package(Sparse)` proof; not itself Windows install validation. |

## Platform Support Tiers

| Platform | Tier | Reviewed evidence | Supplemental evidence | Explicit boundary |
|---|---|---|---|---|
| Linux | Strongest reviewed source of truth | `make quality-review-compile`, `make quality-review-cmake`, dead-code report/check completeness | direct `make test`, UBSan, ASan, `bench-fast`, TSan/OpenMP, coverage | Focused install scripts are local Unix-side proof surfaces, not a separate reviewed Linux install lane unless CI promotes them. |
| macOS | Reviewed Apple Clang platform lane plus supplemental package confidence | Apple Clang `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` | Homebrew GCC direct build/test/wall-check; Make install/`pkg-config` proof | Supplemental install/pkg-config proof does not become full reviewed install/export parity. |
| Windows | Reviewed CMake-first consumer subset | MSVC 2022 CMake configure, build, `ctest -N`, full `ctest`; expected registered CTest count is 51 | none currently promoted beyond the same reviewed CMake subset | No Makefile parity, no dead-code path, no separate install-validation lane, and staged exclusions remain explicit. |

## Lane Classification

| Lane | Classification | Owner / command | Notes |
|---|---|---|---|
| Linux Makefile compile-quality | CI-enforced reviewed | `make quality-review-compile` | Reviewed format/source-list/lint wrapper. |
| Linux CMake parity | CI-enforced reviewed | `make quality-review-cmake` | Reviewed configure, rebuild, `ctest -N`, and `ctest` path. |
| Linux dead-code completeness | CI-enforced reviewed | `make deadcode-report`, `make deadcode-check` | Reviewed completeness guard with uploaded diagnostics. |
| Linux direct runtime and fast benchmarks | CI-enforced supplemental | `make test`, `make sanitize`, `make asan`, `make bench-fast` | Runtime confidence, not the strongest reviewed baseline label. |
| Linux TSan/OpenMP | CI-enforced supplemental | dedicated TSan job | Suppressed runtime internals and documented LOBPCG omission keep claim bounded. |
| Linux coverage | CI-enforced supplemental | `make coverage` | Coverage signal and artifact, not platform parity proof. |
| macOS Apple Clang reviewed path | CI-enforced reviewed | `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` | Narrower than Linux because dead-code, coverage, and broader runtime lanes are not promoted there. |
| macOS Homebrew GCC | CI-enforced supplemental | `make CC=gcc-15`, `make CC=gcc-15 test`, `make CC=gcc-15 wall-check` | Second-compiler confidence. |
| macOS Make install/pkg-config | CI-enforced supplemental | `bash tests/test_install.sh` | Static-first package confidence only. |
| Windows MSVC CMake subset | CI-enforced reviewed | CMake configure/build, `ctest -N`, `ctest` | CMake-first consumer proof only. |
| Windows Makefile parity | Staged / unsupported | none | Unix Makefile targets are not claimed on Windows. |
| Windows install validation | Staged / unsupported | none | `cmake --install` docs exist for manual workflow, but no separate reviewed Windows install lane is claimed. |
| Shared-library package behavior | Unsupported | none | Sprint 112 selected the static-first package tier. |
| Dynamic ABI compatibility | Unsupported | none | Exact-version package metadata only; no binary compatibility promise. |

## Staged Exclusions and Non-Claims

- Linux remains the strongest reviewed source of truth, but local
  `tests/test_install.sh` and `tests/test_cmake_install.sh` are developer-side
  Unix install/export proof surfaces unless they are explicitly promoted to CI
  reviewed lanes.
- macOS reviewed scope does not include dead-code completeness, full coverage
  parity, full install/export parity, or shared-library/runtime-loader proof.
- macOS Homebrew GCC is supplemental compiler confidence, not a second
  reviewed platform baseline.
- macOS Make install/`pkg-config` proof strengthens static-first package
  confidence only; it does not imply reviewed CMake install/export parity.
- Windows reviewed scope remains the MSVC CMake subset. It does not imply
  Makefile parity, Unix install-script parity, dead-code coverage, package
  manager support, shared-library support, or a separate reviewed
  install-validation lane.
- Windows `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain
  staged exclusions; the bounded lifecycle property/fuzz lane is not reviewed
  Windows evidence.
- CTest registration count on Windows is currently expected to be 51 and is a
  guard for the reviewed subset, not evidence that excluded tests became
  reviewed.
- No platform claims shared-library package behavior, dynamic ABI
  compatibility, SONAME/SOVERSION stability, DLL/import-library behavior, or
  runtime-loader compatibility.
- No platform claim should be widened from Sprint 110 no-public-header-drift
  evidence; that evidence supports source/package stability only.

## Documentation and CI Comment Update Queue

| Surface | Day 9 assessment | Follow-through |
|---|---|---|
| `README.md` cross-platform CI summary | Already matches the reviewed/supplemental split at a compact level. | Day 12 should keep wording concise and avoid adding maintainer proof detail. |
| `INSTALL.md` supported-platform and verification sections | Already distinguishes local install scripts from reviewed platform confidence. | Day 12 should preserve the static-first package and non-claim wording. |
| `docs/maintainer_guide.md` package/platform ownership | Already owns detailed interpretation and package proof boundaries. | Day 12 can add or refine only if Days 10-11 change evidence. |
| `.github/workflows/ci.yml` comments | Already describe Linux reviewed baseline and supplemental lanes. | No Day 9 change required. |
| `.github/workflows/macos-ci.yml` comments | Already classify install/pkg-config as supplemental. | Day 11 should revisit after macOS follow-through. |
| `.github/workflows/windows-ci.yml` comments | Already classify Windows as reviewed CMake subset and list exclusions. | Day 10 should revisit after Windows follow-through. |

## Day 9 Contract Statement

> Sprint 112 platform support is evidence-tiered. Linux is the strongest
> reviewed source of truth. macOS has a reviewed Apple Clang lane plus
> supplemental package and second-compiler confidence. Windows has a reviewed
> MSVC CMake-first consumer subset only. The selected package tier remains
> static-first; shared-library support, dynamic ABI compatibility, broad
> install/export parity, and unreviewed platform lanes are explicit non-claims.

## Completion Criteria

- Linux, macOS, and Windows support tiers are defined from concrete workflow
  evidence.
- CI-enforced reviewed, supplemental, staged, local-only, and unsupported lanes
  are separated.
- Windows and macOS coverage is not inferred from unrelated header stability or
  Unix-side install proof.
- Documentation and CI follow-through needs are ready for Days 10-12.
