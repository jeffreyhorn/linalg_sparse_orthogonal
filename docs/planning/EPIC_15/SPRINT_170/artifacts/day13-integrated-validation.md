# Sprint 170 Day 13: Integrated Validation

## Purpose

Day 13 validates the Sprint 170 package/ABI decision work as an integrated
change set. The validation covers the installed static package proofs, the
static-first deferral guard, shell syntax, documentation claim scan, diff
hygiene, C quality-gate decision, and generated-output staging check.

## Validation Summary

| Check | Command | Result |
| --- | --- | --- |
| Shell syntax | `bash -n scripts/static_package_deferral_check.sh` | Passed |
| Static-first package guard | `bash scripts/static_package_deferral_check.sh` | Passed |
| Make install and `pkg-config` proof | `bash tests/test_install.sh` | Passed: 23 passed, 0 failed |
| CMake install/export proof | `bash tests/test_cmake_install.sh` | Passed: 27 passed, 0 failed, 0 skipped |
| Documentation claim scan | `rg -n "Shared-library packaging|BUILD_SHARED_LIBS|dynamic ABI|static-first|Sprint 170|pkg-config execution parity|package-manager|state-of-the-art" README.md INSTALL.md docs/maintainer_guide.md` | Passed: expected static-first claims and non-claim wording only |
| Diff hygiene | `git diff --check` | Passed |
| Generated-output staging | `git status --short --branch` | Passed: no generated build, report, or cache artifacts listed |

## Guard Validation Details

`scripts/static_package_deferral_check.sh` passed all current contract checks:

- Sprint 170 product decision record exists and retains selected decision
  tokens;
- `BUILD_SHARED_LIBS=ON` remains rejected with shared-library, dynamic ABI,
  symbol visibility, loader, and consumer-proof blocker wording;
- CMake target declaration remains explicit static;
- Makefile static archive contract remains source-visible;
- `sparse.pc.in` retains exact static archive metadata;
- public headers remain free of export/import or static/shared ABI macros;
- CMake and `pkg-config` package metadata retain no static/shared selector;
- README, INSTALL, and the maintainer guide retain deferred support wording
  and decision-record citations;
- Windows workflow retains metadata-only `sparse.pc` inspection and no
  unselected Makefile or `pkg-config` execution.

## Install/Package Validation Details

`tests/test_install.sh` passed the Unix Make install/uninstall and
`pkg-config` proof:

- installed the static archive;
- installed the expected 19 headers;
- installed `sparse.pc`;
- rejected shared-library artifacts;
- validated `pkg-config` prefix, libdir, includedir, `--cflags`, `--libs`,
  `--static --libs`, and exact version;
- compiled, linked, and ran both the basic consumer and maintained example;
- removed library, headers, and `sparse.pc` during uninstall.

`tests/test_cmake_install.sh` passed the Unix CMake install/export proof:

- configured, built, and installed successfully;
- installed the static archive, headers, CMake package files, and `sparse.pc`;
- exported a static imported target with install-prefix-relative include and
  archive locations;
- rejected shared imported metadata, unsupported loader/shared-selector
  metadata, and source/build-tree path leaks;
- built and ran installed `find_package(Sparse)` consumers;
- accepted exact package version and rejected a lower same-major mismatch.

## C Quality-Gate Decision

No `.c` or `.h` files were modified in the Sprint 170 Day 13 change set.
The full C quality gate:

```sh
make format && make lint && make test
```

was therefore not required for Day 13.

## Day 13 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Install/package validation log | Complete | Make and CMake install proofs passed locally. |
| Guard validation log | Complete | Static package deferral guard passed with Day 11/Day 12 checks. |
| Docs claim-scan log | Complete | Targeted scan showed expected static-first and non-claim wording. |
| C quality-gate decision | Complete | No `.c` or `.h` changes, so full C gate was not required. |
| Day 13 integrated-validation artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| All required checks pass. | Complete | Focused syntax, guard, install, docs, and diff checks passed. |
| Generated build/report/cache artifacts are not staged unintentionally. | Complete | `git status --short --branch` showed only source/docs/planning changes. |
| Failures stop the sprint for user input rather than weakening claims. | Complete | No failures occurred; no claims were weakened to pass validation. |
