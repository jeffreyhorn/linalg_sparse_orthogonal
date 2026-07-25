# Sprint 134 Day 6 - macOS Install/Export Decision

## Purpose

Day 6 decides how Sprint 134 should handle macOS CMake install/export proof
after the Day 5 audit. The decision must improve evidence where feasible
without silently widening macOS into full reviewed install/export parity.

## Decision

Add a separate **supplemental macOS CMake install/export confidence job** on
Day 7.

The selected job should run:

1. `bash tests/test_cmake_install.sh`
2. `bash scripts/static_package_deferral_check.sh`

This is supplemental macOS evidence for the maintained static-first package
contract. It is not a reviewed macOS install/export parity lane, does not
replace the reviewed Apple Clang compile/CMake/sanitize path, and does not
imply shared-library packaging, dynamic ABI compatibility, package-manager
support, runtime-loader behavior, Windows install validation, or Windows
Makefile parity.

## Rationale

| Criterion | Assessment |
| --- | --- |
| Evidence value | High. macOS already runs supplemental Make install/`pkg-config`; adding CMake install/export closes the companion installed CMake consumer evidence gap. |
| Reviewed-tier confidence | Not enough to call reviewed parity yet. The lane should first run as supplemental so runtime, flakes, and hosted-runner behavior can be observed. |
| Runtime cost | Moderate. `tests/test_cmake_install.sh` performs configure/build/install plus temporary downstream CMake consumers. A separate job keeps the cost and failures isolated. |
| Tool availability | Acceptable. The existing macOS reviewed CMake path already relies on CMake being present on `macos-latest`. |
| Compiler tiering | Use default Apple Clang on `macos-latest`; do not add Homebrew GCC to this install/export lane. |
| Support clarity | Good if workflow comments and docs keep the lane supplemental and static-first. |

## Rejected Alternatives

| Alternative | Reason rejected for Sprint 134 |
| --- | --- |
| Reviewed macOS CMake install/export parity | Too strong before the lane has hosted-runner runtime/flakiness evidence. Current macOS reviewed tier remains Apple Clang compile-quality/CMake parity/sanitize/wall-check. |
| Explicit deferral with no CI change | Leaves an actionable, low-risk evidence gap open even though a local proof script already exists. |
| Add CMake install/export to the existing matrix job | Would mix package install/export failures with compile-quality, CMake parity, sanitizer, and second-compiler failures. |
| Run the lane under Homebrew GCC too | Adds install/export matrix cost without a current support claim that requires second-compiler package parity. |

## Day 7 Implementation Plan

Update `.github/workflows/macos-ci.yml` with a new standalone supplemental job.

Recommended job:

- job id: `cmake-install-export`
- job name: `macOS supplemental CMake install/export confidence path`
- runner: `macos-latest`
- steps:
  - checkout
  - run `bash tests/test_cmake_install.sh`
  - run `bash scripts/static_package_deferral_check.sh`

Keep the existing `install-and-pkgconfig` supplemental job unchanged.

## Documentation and Workflow Comment Plan

| Surface | Day 7 update |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Update top comment and add job comments that the new CMake install/export job is supplemental. |
| `README.md` | Update CI summary only if needed to mention supplemental macOS static-first CMake install/export confidence. |
| `INSTALL.md` | Update macOS/platform interpretation to separate supplemental Make install/`pkg-config` and supplemental CMake install/export confidence. |
| `docs/maintainer_guide.md` | Update package/platform truth to record the supplemental macOS CMake install/export lane while preserving the no-reviewed-parity claim. |

## Validation Plan

Day 7 should run locally:

1. `bash -n scripts/static_package_deferral_check.sh`
2. YAML parse for `.github/workflows/macos-ci.yml`
3. `bash tests/test_cmake_install.sh`
4. `bash scripts/static_package_deferral_check.sh`
5. `git diff --check`
6. focused whitespace scan over touched workflow, support docs, and Sprint 134
   paths

If Day 7 edits only workflow/docs/planning files and no `.c` or `.h` files,
`make format && make lint && make test` is not required by the sprint rule.

## Rollback and Triage

| Scenario | Response |
| --- | --- |
| Hosted macOS runner lacks expected CMake/package tooling | Add explicit tool setup if bounded and stable; otherwise revert the supplemental lane and record deferral. |
| Runtime is too high | Keep the lane separate so it can be disabled without affecting reviewed Apple Clang or supplemental Make install jobs. |
| Static deferral guard fails on macOS only | Treat as support wording or tool behavior drift; do not widen package claims until the cause is understood. |
| `tests/test_cmake_install.sh` exposes macOS-specific install/export behavior | Fix the package proof if the contract is wrong, or explicitly classify the macOS behavior before claiming parity. |

## Residual macOS Package Queue

| Residual | Status |
| --- | --- |
| Reviewed macOS CMake install/export parity | Deferred pending hosted-runner evidence from the supplemental lane. |
| Homebrew GCC package install/export matrix | Deferred; no current support claim requires it. |
| macOS shared-library/dynamic ABI support | Deferred by Sprint 133 product decision. |
| macOS package-manager support | Deferred non-claim. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| macOS install/export parity has an explicit decision. | Complete | Decision selects a supplemental CMake install/export confidence job, not reviewed parity. |
| Selected support tier is reflected in planned validation. | Complete | Day 7 plan runs CMake install/export proof and static deferral guard as supplemental macOS CI. |
| No macOS install parity is implied without matching proof. | Complete | Reviewed parity remains deferred until the supplemental lane provides hosted-runner evidence and a future decision promotes it. |
