# Sprint 153 Day 9 Downstream Proof Implementation

## Purpose

Day 9 implements the Day 8 downstream proof design for the selected
static-first product decision. The implementation strengthens unsupported
loader metadata proof while preserving installed static package consumers.

## Files Changed

| File | Change | Claim Impact |
| --- | --- | --- |
| `tests/test_cmake_install.sh` | Added an installed CMake package scan for unsupported loader and shared-selector metadata. | CMake install proof now fails if loader metadata appears before runtime-loader support is selected. |

## Preserved Downstream Proof

The implementation preserves existing proof for:

- installed static archive presence;
- absence of `.so`, `.so.*`, `.dylib`, and `.dll` artifacts;
- exact installed header count;
- installed CMake package files;
- static imported target metadata;
- installed-prefix include/archive paths;
- no source/build tree leaks;
- static archive `sparse.pc` metadata;
- installed maintained CMake example configure/build/run;
- exact-version CMake consumer configure/build/run;
- mismatched-version rejection;
- Unix `pkg-config` consumers in `tests/test_install.sh`.

## New Unsupported-Loader Metadata Check

`tests/test_cmake_install.sh` now scans installed CMake package files for:

- `SOVERSION`;
- `IMPORTED_SONAME`;
- `INSTALL_NAME`;
- `MACOSX_RPATH`;
- `IMPORTED_IMPLIB`;
- standalone `RUNTIME`;
- package component selectors mentioning `static` or `shared`;
- `Sparse::*shared` target naming.

If any token appears, the test fails with wording tied to the static-first
product decision and the absence of selected runtime-loader support.

## Non-Claims Preserved

The Day 9 implementation does not claim:

- shared-library packaging;
- dynamic ABI compatibility;
- Linux `.so` support;
- macOS `.dylib` support;
- Windows `.dll` or import-library support;
- SONAME, install-name, RPATH, or runtime-loader behavior;
- static/shared package selectors;
- Windows Makefile parity;
- Windows `pkg-config` execution parity.

## Validation Result

Focused validation for Day 9 passed:

- `bash scripts/static_package_deferral_check.sh` passed.
- `bash tests/test_install.sh` passed with `23` checks and `0` failures.
- `bash tests/test_cmake_install.sh` passed with `27` checks, `0` failures,
  and `0` skips.

No C or public header files changed, so the full
`make format && make lint && make test` gate is not required for Day 9.
