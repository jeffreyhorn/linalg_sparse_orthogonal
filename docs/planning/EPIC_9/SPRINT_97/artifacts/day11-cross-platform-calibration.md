# Sprint 97 Day 11: Cross-Platform Truth Calibration

## Purpose

Day 11 recalibrates macOS, Windows, and generic POSIX proof claims after the
Day 5-10 build, package, consumer-proof, and workflow updates. The goal is to
keep platform claims evidence-backed without turning local validation or
supplemental CI lanes into broad parity promises.

## Evidence Map

| Surface | Evidence currently owned | Limits that remain explicit |
| --- | --- | --- |
| Linux CI | reviewed Makefile compile-quality, reviewed CMake parity, dead-code report/check, supplemental runtime, `bench-fast`, coverage, TSan | install/export scripts remain developer-side proof rather than a separate reviewed Linux install lane |
| macOS Apple Clang CI | reviewed Makefile compile-quality, reviewed CMake parity, `wall-check`, sanitizer path | narrower than Linux; no dead-code lane; install/export parity is not claimed |
| macOS Homebrew GCC CI | supplemental direct build/test and `wall-check` | second-compiler confidence only, not reviewed CMake or package parity |
| macOS install/`pkg-config` job | supplemental static-first Make install and `pkg-config` confidence through `tests/test_install.sh` | not reviewed macOS install/export parity; does not replace local Unix-side install scripts |
| Windows CI | reviewed CMake configure/build, `ctest -N`, full `ctest`, expected CTest count `51` | no reviewed Makefile parity; no separate reviewed install-validation lane; staged exclusions remain visible |
| Local Make install proof | `tests/test_install.sh` validates static archive, no shared artifacts, headers, `sparse.pc`, downstream `pkg-config` consumers, uninstall | Unix-side local proof, not a broad platform CI parity claim |
| Local CMake install proof | `tests/test_cmake_install.sh` validates static archive, no shared artifacts, CMake package exports, `find_package`, exact versioning, mismatched version rejection | Unix-oriented local proof, not reviewed Windows install validation |

## Claim Updates

Day 11 updates `docs/maintainer_guide.md` only.

Changes:

- Replace older Windows wording that described the platform as an
  `install-consumer lane`.
- State the current Windows claim as the reviewed CMake-first consumer subset.
- Remove the stale Sprint 68 timestamp from the active platform-confidence
  interpretation while preserving the same fuzz/property-lane boundary.

No README, INSTALL, or workflow YAML update is needed:

- README already says Linux is strongest, macOS is narrower with supplemental
  install confidence, and Windows is the reviewed CMake subset plus
  CMake-first consumer story.
- INSTALL already distinguishes local install/export proof from narrower
  reviewed platform confidence.
- macOS workflow comments already state the install/`pkg-config` job is
  supplemental static-first confidence, not reviewed install/export parity.
- Windows workflow comments and output already state the CMake-first consumer
  scope, expected CTest count, and staged exclusions.

## Staged Exclusions

Staged or intentionally deferred lanes after Day 11:

- Windows Makefile reviewed wrappers
- Windows dead-code flow
- Windows `test_threads`
- Windows `test_sprint4_integration`
- Windows `test_fuzz`
- Windows separate install-validation lane
- macOS dead-code lane
- macOS full reviewed install/export parity
- shared-library package lane on every platform

These are deliberate non-claims until future evidence changes their ownership.

## Verification

Focused Day 11 checks:

```sh
python3 scripts/check_library_sources.py
git diff --check
rg -n "[ \t]+$" docs/maintainer_guide.md docs/planning/EPIC_9/SPRINT_97
```

Observed results:

- source-list checker passed: `source-list-check: PASS (42 library sources)`
- `git diff --check` passed
- trailing-whitespace scan passed with no matches
- maintainer guide, README, INSTALL, and workflow comments all describe the
  same platform proof boundaries

No `.c` or `.h` files are modified by this documentation calibration, so the
full `make format && make lint && make test` chain is not required.

## Day 11 Result

The platform story is calibrated around current evidence: Linux remains the
strongest reviewed source of truth, macOS remains narrower with supplemental
static-first install confidence, and Windows remains the reviewed CMake-first
consumer subset with explicit staged exclusions.
