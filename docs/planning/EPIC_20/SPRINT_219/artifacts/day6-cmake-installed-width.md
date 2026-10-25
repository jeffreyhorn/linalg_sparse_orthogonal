# Day 6: CMake Installed Target Propagation

**Date:** 2026-10-24
**Scope:** Sprint 219 item 219.3, first 10 of 26 planned hours
**Host:** macOS x86_64, Apple Clang 17.0.0

## Installed Contract

`Sparse::sparse_lu_ortho` remains a static imported target. Its install
interface now publishes `SPARSE_PACKAGE_IDX_BITS=32` or `64` from the same
validated CMake selector that builds the archive and generates
`sparse_version.h`. The build interface keeps its existing
`SPARSE_IDX_BITS=N` definition. The installed target does **not** define
`SPARSE_IDX_BITS` for the consumer, so an explicit consumer selection is
not overwritten by a same-name target definition.

`sparse_types.h` now defaults an unspecified `SPARSE_IDX_BITS` to the
generated `SPARSE_BUILT_IDX_BITS`. It rejects unsupported generated values,
an explicit effective `SPARSE_IDX_BITS` that disagrees with the installed
header, and a supplied `SPARSE_PACKAGE_IDX_BITS` that disagrees with it.
This brings the core compile-time header check forward from the Day 8
consumer matrix because a 64-bit imported target must give a downstream
consumer the correct public typedef without a manual flag, while an explicit
conflict must fail before use. The existing typedef-size assertion remains.

The maintained CMake example queries `sparse_idx_bits()` once and compares
the archive result with its generated header and `idx_t` before constructing
a matrix or calling an index-bearing solver. This covers a swapped archive
in that fixture, not arbitrary consumers that skip the pre-use check.

## Clean-Prefix Evidence

`bash tests/test_cmake_idx_width_install.sh` passed **8/8**. For each width
it configured and built an independent library tree, installed into a clean
prefix, moved that prefix, and then configured a downstream
`find_package(Sparse)` consumer against the relocated package. The script
checked the installed generated header, exported target definition, absence
of source/build paths in the target file, consumer compile command, archive
query, and a successful LU solve. The consumer did not set `SPARSE_IDX_BITS`.

| Installed width | Imported definition | Consumer result | Opposite explicit width |
| --- | --- | --- | --- |
| 32 | `SPARSE_PACKAGE_IDX_BITS=32` | `Sparse index width 32`, then LU solve and `OK`. | `-DSPARSE_IDX_BITS=64` rejected at compile with `SPARSE_IDX_BITS must match SPARSE_BUILT_IDX_BITS`. |
| 64 | `SPARSE_PACKAGE_IDX_BITS=64` | `Sparse index width 64`, then LU solve and `OK`. | `-DSPARSE_IDX_BITS=32` rejected at compile with the same diagnostic. |

The relocated consumer's `Sparse_DIR` points into the moved prefix. The
archive query proves the linked archive reports the same width as the
installed header in each positive fixture. `sparse.pc` is not part of this
relocation claim; its stored prefix remains absolute and Day 7 owns its
width flag and validator checks.

The existing `CMAKE_BUILD_PARALLEL_LEVEL=4 bash tests/test_cmake_install.sh`
passed **38/38**, including the default installed example, exact-version
consumer, static package validator, and missing-archive negatives.
`bash tests/test_make_idx_width.sh` passed **26/26** after the header change,
covering both Make widths and the safe mixed-archive negative. The new
public-header checks therefore did not break the selected Make build-tree
lanes.

The new header check requires cppcheck to read the generated version header.
`make lint` now passes `$(BUILDDIR)/include` to cppcheck, matching the strict
compiler and clang-tidy include paths. The reusable identity test checks
typedef/header agreement with `_Static_assert` and keeps only the archive
query as a runtime condition; this avoids cppcheck's correct redundant-test
diagnostic after the header guard. The final
`make -s BUILDDIR=build/s219-day6-quality -j4 lint` and
`make -s BUILDDIR=build/s219-day6-quality -j4 test` both passed. The full
Make suite was rerun after the assertion change. `make format-check`, a
direct clang-format check of `examples/cmake_example/main.c`, and
`make BUILDDIR=build/s219-day6-quality source-list-check` also passed
(49 library sources). The default unstamped `build/` tree was not modified.

## Boundary And Handoff

The imported target automatically supplies the package identity, and both
tested explicit width conflicts fail at compilation before the first
index-bearing call. That closes the Day 6 CMake target and consumer-smoke
acceptance criteria. It does not yet establish full installed-package
integrity: the static validator does not compare header, target, pkg-config,
and archive widths, and `sparse.pc` still emits no width definition. Day 7
owns those fields and mutation checks; Days 8-10 own the broader isolated
positive/negative consumer matrix. Raw manual links that bypass metadata and
omit the pre-use query remain outside the maintained contract.
