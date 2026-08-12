# Sprint 153 Day 11 CI And Documentation Implementation

## Purpose

Day 11 implements the Day 10 CI follow-through policy and aligns active
package/ABI documentation with the selected static-first product decision.

## Files Changed

| File | Change | Claim Impact |
| --- | --- | --- |
| `.github/workflows/windows-ci.yml` | Mirrored the Day 9 unsupported-loader/static-shared selector metadata scan in the Windows inline CMake install/downstream proof. | Windows hosted proof now rejects unsupported loader metadata without claiming DLL/runtime-loader support. |
| `INSTALL.md` | Replaced generic shared deferral blockers with the exact export/import, visibility, ABI, platform-loader, shared-consumer, and runtime-loader blockers. | User install docs match the CMake rejection and static-first decision. |
| `README.md` | Added concise wording that `BUILD_SHARED_LIBS=ON` rejection names the missing shared-library policies and proofs. | Top-level package docs point users at exact shared blockers. |
| `docs/maintainer_guide.md` | Updated proof-owner descriptions for unsupported-loader metadata checks and exact shared deferral blocker wording. | Maintainer docs match the Day 9 proof and Day 11 Windows CI mirror. |

## Windows CI Follow-Through

The Windows inline CMake install/downstream validation now scans installed
CMake package metadata for:

- `SOVERSION`;
- `IMPORTED_SONAME`;
- `INSTALL_NAME`;
- `MACOSX_RPATH`;
- `IMPORTED_IMPLIB`;
- standalone `RUNTIME`;
- package component selectors mentioning `static` or `shared`;
- `Sparse::*shared` target naming.

This mirrors the Unix `tests/test_cmake_install.sh` unsupported-loader proof
without adding Windows Makefile parity, Windows `pkg-config` execution parity,
DLL support, dynamic ABI support, or runtime-loader support.

## Documentation Alignment

Active docs now consistently describe shared-library support as deferred until
these blockers are closed:

- export/import policy;
- symbol visibility policy;
- dynamic ABI policy;
- Linux SONAME metadata;
- macOS install-name/RPATH metadata;
- Windows DLL/import-library behavior;
- installed shared consumer proof;
- runtime-loader validation.

## Stale Wording Review

Reviewed active package/ABI wording in:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `.github/workflows/ci.yml`;
- `.github/workflows/macos-ci.yml`;
- `.github/workflows/windows-ci.yml`;
- `sparse.pc.in`;
- `cmake/SparseConfig.cmake.in`;
- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `scripts/static_package_deferral_check.sh`;
- `CMakeLists.txt`.

The reviewed wording remains static-first and does not claim shared-library
packaging, dynamic ABI compatibility, runtime-loader support, package-manager
distribution, static/shared selectors, Windows Makefile parity, or Windows
`pkg-config` execution parity.

## Validation Result

Focused validation for Day 11 passed:

- `bash scripts/static_package_deferral_check.sh` passed.
- `bash tests/test_install.sh` passed with `23` checks and `0` failures.
- `bash tests/test_cmake_install.sh` passed with `27` checks, `0` failures,
  and `0` skips.

No C or public header files changed, so the full
`make format && make lint && make test` gate is not required for Day 11.
