# Day 5: Build-System Parity And Width Identity

**Date:** 2026-10-24
**Scope:** Sprint 219 item 219.2, final 10 of 30 planned hours
**Host:** macOS x86_64, Apple Clang 17.0.0, GNU Make 3.81

## Build-Tree Parity

| Contract point | CMake | Make | Result |
| --- | --- | --- | --- |
| Omitted selector | `SPARSE_IDX_BITS` cache defaults to `32`; fresh default archive/test reports 32. | `SPARSE_IDX_BITS ?= 32`; fresh default archive/test reports 32. | Same default. |
| Explicit selector | `-DSPARSE_IDX_BITS=32` or `64`. | `SPARSE_IDX_BITS=32` or `64`. | Same allowed values, route-specific syntax. |
| Invalid value | Configure rejects with `SPARSE_IDX_BITS must be 32 or 64`. | Makefile parse rejects with the same allowed-value diagnostic. | Both fail before build. |
| Ambient conflicting definition | `CMAKE_C_FLAGS*` width definitions rejected at configure. | `CFLAGS` and `CC` width definitions rejected at parse. | Both reject checked override routes; compiler and build-system behavior outside these routes remains unclaimed. |
| Build identity | `configure_file` writes `SPARSE_BUILT_IDX_BITS` into generated `sparse_version.h`. | Template substitution writes the same macro; `.sparse-idx-bits` stamps the tree. | Generated headers are byte-for-byte identical at 32 and at 64 (`cmp`). |
| Same-width link | `test_idx_width` compares `sizeof(idx_t) * CHAR_BIT`, generated header, and `sparse_idx_bits()`. | The same test is in `TEST_SRCS` and the focused Make script. | 32 and 64 passed on both routes. |
| Opposite-width tree reuse | Separate clean trees used; reconfigure behavior remains an N8 consumer-matrix case. | Opposite-width stamped-tree request is rejected before archive mutation. | No stale Make archive silently reused. |
| Installed width metadata | Exported target does not yet publish a width usage requirement. | `sparse.pc` does not yet publish a width compile flag. | **Not yet an installed-consumer contract.** |

The focused test uses only the width-neutral `sparse_idx_bits()` API. Linking
its 64-bit compilation against the 32-bit archive produced
`typedef=64 header=64 archive=32` and a nonzero exit before any index-bearing
call. This is a reusable pre-use check for later consumer fixtures; merely
adding it to the library's own test lists does not make arbitrary consumers
run it automatically.

The source header `include/sparse_types.h` is unchanged. Both build systems
derive the generated numeric width from their validated selector, install the
generated header next to source headers, and keep build-tree paths out of its
content. The one generated macro is the stable input for Days 6-8. Day 6
must export a distinct package usage definition from the CMake target; Day 7
must emit it in both pkg-config generators and compare metadata with the
archive; Day 8 must make the public header reject an effective mismatch and
put the pre-use check before index-bearing consumer calls. A copied or
manually swapped archive remains unsafe unless a maintained consumer runs
the query check; no general automatic link-time guard is claimed.

## Verification And Warning Audit

| Check | Observed result | Boundary |
| --- | --- | --- |
| Fresh CMake default, 32, and 64 trees: build `test_idx_width`; focused `ctest` | Default reports 32; explicit 32 and 64 each pass `test_idx_width` and `test_sparse_matrix` (2/2 per explicit tree). | Focused build-tree tests only. |
| Fresh Make default, 32, and 64 trees via `bash tests/test_make_idx_width.sh` | 26 passed, 0 failed; includes the mixed-archive negative check, selector negatives, and stale-tree rejection. | Local macOS only; full hosted matrix is later work. |
| Generated-header comparison | Make and CMake output match byte-for-byte at both widths. | Does not validate installed metadata or archive integrity. |
| `make format-check` | Passed. | New C test is formatted. |
| `make BUILDDIR=build/s219-day5-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck. | Default 32-bit lint lane; 64-bit lint flag propagation was dry-run checked on Day 4. |
| `make -s BUILDDIR=build/s219-day5-quality -j4 test` | All Make test binaries passed, including `test_idx_width`. | Default 32-bit full suite; selected 64-bit functional lane remains Day 11 work. |
| `make BUILDDIR=build/s219-day5-quality source-list-check` | Passed: 49 library sources. | A first run without `BUILDDIR` was correctly refused because the pre-existing default `build/` tree is unstamped; it was not cleaned. |

The CMake smoke builds emitted Apple Clang `-Wdouble-promotion` warnings for
existing `NAN`/`INFINITY` uses. No new 64-bit format/cast diagnostic appeared
in the focused build. The selected 64-bit functional solver/IO lane and a
broader warning audit are Day 11 work; this smoke cannot establish algorithm
correctness at large indices. Item 219.2's planned 30-hour build-selection
allocation is closed with the Day 3, Day 4, and Day 5 evidence, while items
219.3-219.5 and installed-package safety remain open.
