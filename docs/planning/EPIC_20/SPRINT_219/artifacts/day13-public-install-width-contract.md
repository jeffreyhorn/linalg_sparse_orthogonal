# Day 13: Public And Installed Width Contract

Date: 2026-10-24

## Consumer Path

`include/sparse_types.h` and `INSTALL.md` now describe the default 32-bit
width and separate Make/CMake build selectors for 32 and 64 bits. A consumer
uses an isolated prefix, the exported CMake target or pkg-config flags, and
checks the linked archive with `sparse_idx_bits()` before its first
index-bearing call. The installed header's `SPARSE_BUILT_IDX_BITS` and
`sizeof(idx_t) * CHAR_BIT` provide the other two values. A conflicting
consumer definition fails at compile time; a foreign archive can link but
the maintained consumer exits before solver use. The README links to the
operational guide rather than implying broad 64-bit algorithm coverage.

The generated package fields used by the examples are:

| Install route | Generated width field | Consumer use |
| --- | --- | --- |
| Make or CMake | `sparse_version.h`: `SPARSE_BUILT_IDX_BITS` | Header/type width comparison. |
| CMake | `SparseTargets.cmake`: `SPARSE_PACKAGE_IDX_BITS=<width>` | Propagated by `Sparse::sparse_lu_ortho`. |
| Make or CMake | `sparse.pc`: `Cflags: -I${includedir} -DSPARSE_PACKAGE_IDX_BITS=<width>` | Selected via an isolated `PKG_CONFIG_LIBDIR`. |
| Both | Static archive: `sparse_idx_bits()` | Runtime pre-use comparison; static validator queries the real archive. |

## Verification

| Check | Outcome | Scope |
| --- | --- | --- |
| `bash tests/test_cmake_idx_width_install.sh` | PASS, 18/18. | Local macOS 32/64 installed CMake consumers and negative conflicts. |
| `bash tests/test_pkgconfig_idx_width_install.sh` | PASS, 40/40. | Local macOS Make/CMake installs at both widths, exact package flags, solves, and mixed-prefix negatives. |
| `make format-check` | PASS. | Public header formatting. |
| `make -s BUILDDIR=build/s219-day13-quality -j4 lint` | PASS. | Default-width strict compile, clang-tidy, cppcheck. |
| `make -s BUILDDIR=build/s219-day13-quality -j4 test` | PASS, all Make test binaries. | Default-width runtime suite; two duplicate `-lm` linker warnings were non-fatal. |

This closes Day 13's documentation work, not the complete Sprint 219 width
gate. The package remains static-only; no mixed-width or cross-version ABI,
shared-library behavior, or universal 64-bit algorithm claim is made. Windows
pkg-config execution and 64-bit selected testing remain outside the reviewed
Windows route. The [Day 12 hosted result gate](day12-hosted-64-bit-registration.md)
is still open because this branch has no hosted run yet. Day 14 owns final
full-quality and contract closeout checks.
