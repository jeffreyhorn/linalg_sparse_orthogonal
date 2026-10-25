# Day 8: Executable Width Check And First Consumers

**Date:** 2026-10-24

**Scope:** Sprint 219 item 219.3, final 6 of 26 hours; item 219.4, first 4 hours
**Host:** macOS x86_64, Apple Clang 17.0.0

## Pre-Use Check

`examples/cmake_example/installed_width_check.h` is a fixture-local helper
shared by the CMake example and the basic pkg-config install consumer. At
the beginning of `main`, before any matrix construction or other
index-bearing library call, it queries `sparse_idx_bits()` once and compares
the archive result and the consumer's `sizeof(idx_t) * CHAR_BIT` with the
installed `SPARSE_BUILT_IDX_BITS`. A mismatch prints all three widths and
returns failure. The subsequent solver path is reached only after the check
passes. This is an executable pre-use check, not a new public ABI or an
automatic guard for arbitrary manually linked callers.

## Clean-Prefix Evidence

`bash tests/test_installed_width_check.sh` passed **12/12**. Its positive
fixtures configured separate CMake libraries at 32 and 64 bits, installed
them into clean prefixes, passed the static package validator, then built
and ran both `find_package(Sparse)` and pkg-config consumers. Both consumers
reported the selected width and completed the LU solve with `OK`. The
pkg-config fixture used the emitted `-DSPARSE_PACKAGE_IDX_BITS=N` flag and
no manual `SPARSE_IDX_BITS` definition.

The successful run recorded these exact temporary package files (the test
removes the temporary directory on exit):

| Width | Installed header | Installed archive |
| --- | --- | --- |
| 32 | `/private/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T/sparse-width-check.LltkAm/install-32/include/sparse/sparse_version.h` | `/private/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T/sparse-width-check.LltkAm/install-32/lib/libsparse_lu_ortho.a` |
| 64 | `/private/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T/sparse-width-check.LltkAm/install-64/include/sparse/sparse_version.h` | `/private/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T/sparse-width-check.LltkAm/install-64/lib/libsparse_lu_ortho.a` |

For each width, the test then replaced only that temporary prefix's archive
with the opposite-width archive. The static validator rejected the package
with `installed index widths disagree`. To exercise the consumer guard even
when validation is deliberately bypassed, the test configured, compiled,
and linked new CMake and pkg-config executables against each swapped prefix.
The 32-bit consumers exited with `consumer=32 header=32 archive=64`; the
64-bit consumers exited with `consumer=64 header=64 archive=32`. Neither
negative run printed the version, solution, or `OK` line that precedes or
follows the solver path. The source returns immediately on a failed check,
so no index-bearing API receives mismatched data in these fixtures.

## Quality And Handoff

| Check | Result |
| --- | --- |
| `bash -n tests/test_installed_width_check.sh`; direct `clang-format --dry-run --Werror` on the example and helper | Passed. |
| `make format-check` | Passed. |
| `make -s BUILDDIR=build/s219-day8-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck. |
| `make -s BUILDDIR=build/s219-day8-quality -j4 test` | Full Make suite passed. |
| `bash tests/test_install.sh` | 26/26 passed, including the updated basic pkg-config consumer. |
| `bash tests/test_cmake_install.sh` | 38/38 passed, including the maintained example. |

The generated header, CMake usage requirement, pkg-config flag, validator,
and unconditional fixture check now form the planned installed-width path;
item 219.3's 26-hour propagation allocation is closed. Day 9 owns the
broader CMake consumer matrix, including wrong headers, explicit width,
archive paths, and stale caches. A consumer that manually links an archive
and skips this helper remains outside the maintained contract.
