# Sprint 153 Day 7 Build And Install Implementation

## Purpose

Day 7 implements the selected static-first product decision from Day 5 using
the Day 6 build/install design. The implementation strengthens unsupported
shared-library diagnostics without adding shared artifacts or widening package
claims.

## Files Changed

| File | Change | Claim Impact |
| --- | --- | --- |
| `CMakeLists.txt` | Strengthened the `BUILD_SHARED_LIBS=ON` fatal error with exact blocker wording. | Shared-library requests fail with clearer static-first deferral diagnostics. |
| `scripts/static_package_deferral_check.sh` | Added assertions that the CMake failure text names the exact blocker tokens. | The deferral wording is now test-backed. |

## Preserved Behavior

The implementation intentionally preserves:

- `add_library(sparse_lu_ortho STATIC ...)`;
- CMake archive-only install behavior;
- Make static archive install behavior;
- static imported target metadata for `Sparse::sparse_lu_ortho`;
- static archive `sparse.pc` metadata;
- absence of shared-library artifacts in install proofs;
- absence of shared/static package selectors;
- absence of public `SPARSE_API`, `SPARSE_EXPORT`, or `SPARSE_IMPORT` macros.

## New Diagnostic Blockers

`BUILD_SHARED_LIBS=ON` rejection now names these blockers:

- export/import policy;
- symbol visibility policy;
- dynamic ABI policy;
- Linux SONAME metadata;
- macOS install-name/RPATH metadata;
- Windows DLL/import-library behavior;
- installed shared consumer proof;
- runtime-loader validation.

## Focused Proof Updates

`scripts/static_package_deferral_check.sh` now fails if the configure-failure
text loses any of those blocker tokens. This makes the Day 5 product decision
actionable in the build guard rather than leaving it as planning prose only.

## Validation Result

Focused validation for Day 7 passed:

- `bash scripts/static_package_deferral_check.sh` passed.
- `bash tests/test_install.sh` passed with `23` checks and `0` failures.
- `bash tests/test_cmake_install.sh` passed with `26` checks, `0` failures,
  and `0` skips.

No C or public header files changed, so the full
`make format && make lint && make test` gate is not required for Day 7.
