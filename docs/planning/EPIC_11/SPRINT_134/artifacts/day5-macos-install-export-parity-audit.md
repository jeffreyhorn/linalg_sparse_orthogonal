# Sprint 134 Day 5 - macOS Install/Export Parity Audit

## Purpose

Day 5 audits macOS package/install coverage after the Day 4 Linux package
contract promotion. The audit separates macOS reviewed Apple Clang evidence,
supplemental Make install/`pkg-config` confidence, supplemental Homebrew GCC
coverage, and the still-missing reviewed CMake install/export parity decision.

## Audited Inputs

| Input | Audit role |
| --- | --- |
| `.github/workflows/macos-ci.yml` | macOS reviewed and supplemental workflow topology. |
| `tests/test_install.sh` | Existing supplemental macOS Make install/`pkg-config` proof command. |
| `tests/test_cmake_install.sh` | Candidate macOS CMake install/export parity proof command. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Current macOS support-tier wording. |
| Day 2 platform gap audit | Initial macOS install/export parity gap classification. |
| Day 4 Linux CI implementation | New Linux reviewed package-contract baseline that macOS should not inherit by implication. |

## Current macOS CI Topology

| Lane | Job or command | Current tier | Notes |
| --- | --- | --- | --- |
| Apple Clang compile-quality | `make quality-review-compile` | Reviewed macOS lane. | Runs in the `apple-clang` matrix leg after installing Homebrew LLVM and cppcheck. |
| Apple Clang CMake parity | `make quality-review-cmake` | Reviewed macOS lane. | Proves configure/build/CTest parity, not install/export parity. |
| Apple Clang wall-check | `make CC=cc wall-check` | Reviewed macOS signal. | Runs for the Apple Clang matrix leg. |
| Apple Clang sanitizer | `make CC=cc sanitize` | Reviewed macOS signal. | Runs only for the Apple Clang matrix leg. |
| Homebrew GCC direct build/test | `make CC=gcc-15` and `make CC=gcc-15 test` | Supplemental second-compiler coverage. | Does not run reviewed Makefile/CMake wrapper paths. |
| Homebrew GCC wall-check | `make CC=gcc-15 wall-check` | Supplemental second-compiler signal. | Keeps second-compiler warning coverage. |
| Make install/`pkg-config` | `bash tests/test_install.sh` | Supplemental static-first package confidence. | Separate job; explicitly not reviewed macOS install/export parity. |

## Make install Versus CMake Install/Export

| Surface | Existing macOS evidence | Missing parity evidence |
| --- | --- | --- |
| Make install | Supplemental macOS CI runs `tests/test_install.sh`. | Already covered as supplemental confidence, not reviewed parity. |
| `pkg-config` consumer | Supplemental macOS CI compiles and runs downstream consumers through `pkg-config`. | Already covered as supplemental confidence. |
| CMake configure/build/CTest | Reviewed Apple Clang CI runs `make quality-review-cmake`. | This is build-tree CMake parity, not installed package proof. |
| CMake install/export | Local `tests/test_cmake_install.sh` exists and Linux reviewed package lane now runs it. | No macOS CI lane currently runs CMake install/export, installed `find_package(Sparse)`, static imported-target metadata, or source/build path leak checks. |
| Static deferral guard | Linux reviewed package lane runs it; local proof exists. | macOS workflow does not run the guard today; Day 6 should decide whether a macOS package lane needs it. |

## CMake Install/Export Gap List

| Gap | Current evidence | Decision needed |
| --- | --- | --- |
| Reviewed macOS CMake install/export parity | None. `tests/test_cmake_install.sh` is local and Linux reviewed after Day 4. | Day 6 should decide whether to add a macOS reviewed or supplemental CMake install/export job. |
| Installed `find_package(Sparse)` on macOS | Not run in macOS CI. | Day 6 should decide whether this becomes reviewed, supplemental, or deferred. |
| Installed target metadata on macOS | Not run in macOS CI. | Day 6 should evaluate whether the Day 11 CMake package metadata checks are appropriate on macOS hosted runners. |
| Source/build path leak scans on macOS | Not run in macOS CI. | Day 6 should decide whether this proof is worth the runtime in macOS CI. |
| Static deferral guard on macOS | Not run in macOS CI. | Day 6 should decide whether macOS package proof needs the static deferral guard, or whether Linux reviewed package proof is enough. |

## Toolchain and Runtime Risk Notes

| Risk | Audit note |
| --- | --- |
| CMake availability | GitHub macOS runners normally provide CMake, and the reviewed `make quality-review-cmake` path already relies on CMake. |
| Compiler selection | The reviewed macOS lane is Apple Clang; Homebrew GCC is supplemental. A CMake install/export parity lane should state whether it is Apple Clang reviewed, standalone supplemental, or both. |
| `pkg-config` behavior | Day 12/PR #148 showed `pkg-config` path formatting can differ by environment. `tests/test_install.sh` now validates emitted paths semantically and already passes in macOS supplemental CI. |
| Runtime cost | `tests/test_cmake_install.sh` performs a full CMake configure/build/install and temporary downstream consumer build. It is heavier than `tests/test_install.sh` and should probably live in a separate job if added. |
| Failure classification | Combining CMake install/export with the existing Apple Clang matrix could make failures harder to attribute; a separate package job would keep failures clearer. |
| Support widening | Adding macOS CMake install/export proof would still not imply shared-library packaging, dynamic ABI support, package-manager support, or Windows parity. |

## Support Wording Review

| Surface | Current wording status |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Correctly states the Make install/`pkg-config` job is supplemental and not reviewed macOS install/export parity. |
| `README.md` | Correctly says macOS enforces the Apple Clang reviewed path with supplemental Homebrew GCC and static-first Make install/`pkg-config` verification. |
| `INSTALL.md` | Correctly lists macOS Apple Clang reviewed lane, Homebrew GCC supplemental lane, and says macOS does not claim full reviewed install/export parity. |
| `docs/maintainer_guide.md` | Correctly keeps macOS narrower than Linux and says local CMake install/export proof does not become reviewed macOS install/export parity. |

No Day 5 wording drift requires immediate docs changes. Day 7 should update
wording only if Day 6 selects a new macOS CMake install/export tier.

## macOS Parity Options

| Option | Description | Strengths | Risks |
| --- | --- | --- | --- |
| Reviewed Apple Clang CMake install/export job | Add a macOS CI job or step that runs `bash tests/test_cmake_install.sh` under Apple Clang and call it reviewed macOS CMake install/export parity. | Highest macOS parity improvement; mirrors Linux package-contract proof for CMake install/export. | Adds runtime; must be explicit that this is still static-first package proof only. |
| Supplemental CMake install/export job | Add a separate macOS supplemental job that runs `bash tests/test_cmake_install.sh`. | Builds confidence without widening reviewed macOS tier as much. | Leaves reviewed parity gap open; still adds runtime. |
| Explicit deferral | Keep only current Make install/`pkg-config` supplemental job and document CMake install/export parity as deferred. | No runtime cost; preserves current tier clarity. | Does not advance macOS CMake install/export evidence. |

## Day 6 Handoff

Day 6 should decide between reviewed promotion, supplemental addition, or
explicit deferral. If implementing a lane, prefer a separate macOS package job
so CMake install/export failures are distinct from Apple Clang compile-quality
and sanitizer failures.

Recommended Day 6 decision questions:

1. Should macOS CMake install/export parity become reviewed or supplemental?
2. Should the selected lane run only `tests/test_cmake_install.sh`, or also
   `scripts/static_package_deferral_check.sh`?
3. Should the lane use Apple Clang only, or include Homebrew GCC?
4. What public/support wording must change if the lane is added?

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| macOS Make install and CMake install/export support are separated. | Complete | Audit distinguishes supplemental `tests/test_install.sh` from missing `tests/test_cmake_install.sh` CI proof. |
| Parity blockers and feasible proof paths are visible. | Complete | Options table records reviewed, supplemental, and explicit deferral paths plus runtime/toolchain risks. |
| macOS support tier wording remains evidence-bounded. | Complete | Current workflow, README, INSTALL, and maintainer guide wording already preserve the current non-claim. |
