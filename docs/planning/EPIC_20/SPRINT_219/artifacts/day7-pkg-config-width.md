# Day 7: pkg-config Width Propagation

**Date:** 2026-10-24

**Scope:** Sprint 219 item 219.3, second 10 of 26 planned hours
**Host:** macOS x86_64, Apple Clang 17.0.0

## Package Contract

Both Make and CMake now substitute their validated `SPARSE_IDX_BITS` into
the same `sparse.pc.in` template. `pkg-config --cflags sparse` returns only
the installed include path and `-DSPARSE_PACKAGE_IDX_BITS=32` or `64`; the
static `-L... -lsparse_lu_ortho -lm` link contract is unchanged. The
package flag is distinct from `SPARSE_IDX_BITS`, so it cannot mask a
consumer's explicit conflicting width. The installed header remains the
source for the default public `idx_t` width and its compile-time guard.

`check_static_install_contract.py` now requires exactly one supported width
in the pkg-config flags and generated version header. For a CMake install,
it also checks the exported target's package-width definition. After static
metadata checks pass, a native probe links the installed archive and calls
`sparse_idx_bits()`; all reported widths must agree. The Unix probe uses the
host C compiler. The Windows path uses a temporary CMake executable linked
to the installed `.lib`. An unqueryable archive fails validation rather
than being accepted on metadata alone.

## Verification

| Check | Result |
| --- | --- |
| `python3 tests/test_static_install_contract.py` | 26 tests passed; missing, duplicated, unsupported, misplaced, and mismatched width mutations rejected. |
| `bash tests/test_pkgconfig_idx_width_install.sh` | 16/16 passed across Make/CMake at 32/64 bits. |
| `bash tests/test_install.sh` | 26/26 passed with the new two-flag pkg-config contract. |
| `bash tests/test_cmake_install.sh` | 38/38 passed, including validator and downstream CMake consumer. |
| `make format-check`; `bash -n tests/test_pkgconfig_idx_width_install.sh` | Both passed. |
| `git diff --check` | Passed. |

The new clean-prefix test uses `PKG_CONFIG_LIBDIR` to isolate each package.
For each of the four routes it validates the real archive query, checks
that the pkg-config flags contain no build-tree path or extra definition,
then compiles and runs the maintained LU example using the emitted flags
without a manual width selection. Each consumer reports the selected width
and completes the solve.

## Boundary And Handoff

These local runs establish installed pkg-config propagation at both widths
on Unix-like hosts and verify the validator's mutation behavior. The native
Windows archive-probe path remains to be exercised by hosted Windows CI.
Day 8 owns additional pre-use and swapped-archive consumer negatives; the
broader isolated matrix remains scheduled for Days 9-10. Consumers linking
manually without package flags or a pre-use archive query are still outside
the maintained installed-package contract.
