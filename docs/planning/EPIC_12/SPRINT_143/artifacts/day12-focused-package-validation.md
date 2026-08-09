# Sprint 143 Day 12 Focused Package Validation

## Purpose

Run the focused selected-path install, export, downstream consumer, static
deferral, package report, and generated-output hygiene checks before the full
quality and claim closure pass.

## Validation Results

| Check | Result |
| --- | --- |
| Shell syntax: `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| `bash tests/test_install.sh` | Passed: 23 passed, 0 failed |
| `bash tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped |

## Install And Downstream Proof Summary

The Make install and `pkg-config` proof validated:

- static archive installation;
- no shared-library artifacts;
- 19 installed headers;
- installed `sparse.pc`;
- exact `pkg-config` version resolution;
- installed prefix/libdir/includedir variables;
- installed `pkg-config --cflags` and `--libs`;
- no `Libs.private` stanza;
- static archive `.pc` description;
- no unsupported package/ABI wording;
- basic `pkg-config` consumer compile/link/run;
- maintained example compile/link/run via `pkg-config`;
- uninstall cleanup.

The CMake install/export proof validated:

- CMake configure/build/install;
- static archive installation;
- no shared-library artifacts;
- 19 installed headers;
- installed CMake package files and `sparse.pc`;
- static imported target metadata;
- no shared imported metadata;
- install-prefix include/archive paths;
- no source/build-tree path leaks;
- static archive `.pc` description;
- no unsupported package/ABI wording;
- installed `find_package(Sparse)` example configure/build/run;
- exact-version consumer configure/build/run;
- mismatched-version rejection;
- `pkg-config` version.

## Generated-Output Hygiene

`git ls-files -o --exclude-standard` showed only the intentional
`docs/planning/EPIC_12/SPRINT_143/` planning files as untracked.

`git status --ignored --short build` showed `build/` remains ignored, so local
generated build artifacts did not become source-controlled outputs.

## Scoped Repairs

No scoped repair was needed on Day 12. All focused selected-path package
checks passed.

## Remaining Risks And Owners

| Risk | Owner |
| --- | --- |
| Shared-library build/install/export remains deferred. | Future package/ABI owner |
| Dynamic ABI compatibility remains deferred. | Future package/ABI owner |
| Runtime-loader behavior remains deferred. | Future package/platform owner |
| Package-manager distribution remains deferred. | Future package/adoption owner |
| macOS and Windows reviewed install/export parity remain Sprint 144 platform-promotion questions. | Sprint 144 platform owner |

## Day 13 Input

Day 13 should run the required full quality and claim-closure pass for the
touched surfaces, confirm whether C/header quality gates are required, and
publish earned static-first package claims with residual non-claims.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Focused package checks pass or scoped failures are repaired. | Complete | All focused package, install, CMake, script, and report checks passed. |
| Generated outputs remain ignored unless intentionally source-controlled. | Complete | Only Sprint 143 planning files are untracked; `build/` remains ignored. |
| Remaining risks have owners and stop conditions. | Complete | Shared ABI, loader, package-manager, and platform-promotion residuals are routed forward. |
