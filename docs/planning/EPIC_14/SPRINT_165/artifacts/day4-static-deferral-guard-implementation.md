# Sprint 165 Day 4 Static Deferral Guard Implementation

## Purpose

Day 4 implements the narrow static-first guard hardening identified by the
Day 3 design and validates that the static archive install/export behavior
still passes. This day does not add shared-library support.

## Implementation Summary

One guard gap was closed in `scripts/static_package_deferral_check.sh`.

Before Day 4, `check_no_export_or_abi_metadata` rejected these public header
macro names:

- `SPARSE_API`
- `SPARSE_EXPORT`
- `SPARSE_IMPORT`

Day 3 defined a wider public-header absence rule for export/import and
static/shared ABI scaffolding. Day 4 updated the guard to also reject:

- `SPARSE_SHARED`
- `SPARSE_STATIC`
- `SPARSE_ABI`

The failure message now reports:

```text
public export/import or static/shared ABI macro appeared without a shared ABI decision
```

## Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `scripts/static_package_deferral_check.sh` | Expanded the public header forbidden macro regex from `SPARSE_API|SPARSE_EXPORT|SPARSE_IMPORT` to `SPARSE_API|SPARSE_EXPORT|SPARSE_IMPORT|SPARSE_SHARED|SPARSE_STATIC|SPARSE_ABI`. | Blocks static/shared ABI macro scaffolding before a shared-library product decision. |

## Files Reviewed But Not Changed

| File | Reason No Change Was Needed |
| --- | --- |
| `CMakeLists.txt` | Existing `BUILD_SHARED_LIBS=ON` rejection, explicit `STATIC` library target, archive-only install destination, exact-version package metadata, and `sparse.pc` generation already match the Day 3 design. |
| `sparse.pc.in` | Existing description is static archive scoped and has no `Libs.private`, shared/static selector, loader, package-manager, or ABI wording. |
| `cmake/SparseConfig.cmake.in` | Template remains intentionally minimal; generated package output is validated by install tests. |
| `tests/test_install.sh` | Already validates installed static archive shape, no shared artifacts, `sparse.pc` fields, no `Libs.private`, unsupported package/ABI wording absence, downstream consumers, exact version, and uninstall cleanup. |
| `tests/test_cmake_install.sh` | Already validates static imported CMake target metadata, no shared imported metadata, no unsupported loader/static-shared selector metadata, no source/build path leaks, exact/mismatched version behavior, and installed CMake consumers. |
| `.github/workflows/windows-ci.yml` | Already keeps Windows package validation CMake-first and `sparse.pc` metadata-only, with explicit Makefile and `pkg-config` execution non-claims. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Existing package-boundary wording already preserves static-first support and shared-library, dynamic ABI, runtime-loader, package-manager, and Windows parity non-claims. |

## Guard Coverage After Day 4

The central static deferral guard now checks:

- `BUILD_SHARED_LIBS=ON` fails at CMake configure time;
- the rejection message preserves static package and shared/dynamic ABI
  blocker wording;
- `sparse_lu_ortho` remains an explicit `STATIC` CMake target;
- CMake install metadata remains archive-only;
- `sparse.pc.in` keeps the static archive package description;
- public headers do not contain export/import or static/shared ABI macros;
- CMake source does not contain shared-library ABI metadata;
- CMake package config and `sparse.pc` templates do not contain static/shared
  selectors;
- README, INSTALL, maintainer guide, and Windows workflow keep package
  non-claim wording;
- Windows workflow does not execute unselected Make install/uninstall or
  `pkg-config` commands.

## Validation Results

Focused validation commands run:

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
```

Results:

| Command | Result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed; all static deferral guard checks completed successfully. |
| `bash tests/test_install.sh` | Passed; 23 passed, 0 failed. |
| `bash tests/test_cmake_install.sh` | Passed; 27 passed, 0 failed, 0 skipped. |

The install validations confirmed:

- static archive install behavior remains intact;
- no shared-library artifacts were installed;
- installed header count remained 19;
- `sparse.pc` still describes static archive package metadata;
- installed CMake package metadata still imports a static target;
- downstream pkg-config and CMake consumers still build and run;
- exact-version package behavior remains covered;
- mismatched CMake package version remains rejected.

## Quality Gate Notes

Day 4 changed a shell validation script and planning documentation only. No
`.c` or `.h` files were changed, so `make format`, `make lint`, and
`make test` were not required for Day 4.

## Completion Check

- `BUILD_SHARED_LIBS=ON` deferral remains fail-closed.
- Installed package metadata cannot imply shared-library support under the
  checked CMake and pkg-config validation surfaces.
- Static archive install behavior remains unchanged and validated.
