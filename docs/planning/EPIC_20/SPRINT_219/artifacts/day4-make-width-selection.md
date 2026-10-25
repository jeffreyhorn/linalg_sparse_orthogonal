# Day 4: Make Width Selection

**Date:** 2026-10-24
**Scope:** Sprint 219 item 219.2, second 10 of 30 planned hours
**Host:** macOS x86_64, Apple Clang 17.0.0, GNU Make 3.81

## Implementation

The Makefile now accepts `SPARSE_IDX_BITS=32` or `64`, defaulting to `32`.
Other values fail while parsing the Makefile with
`SPARSE_IDX_BITS must be 32 or 64`. Definitions of the same macro hidden in
`CFLAGS` or `CC` are rejected before build. `SPARSE_CFLAGS` appends the
validated width to compiler invocations for library objects, test binaries,
examples, benchmarks, and lint's strict compile and clang-tidy paths. The
separate thread-sanitizer compiler flags also carry the selector. User
`CFLAGS` can still set other flags without suppressing the selected width.

Each `BUILDDIR` gets a `.sparse-idx-bits` stamp before its generated version
header or any library object is built. Stamp creation uses an atomic hard
link so two fresh Make processes requesting opposite widths cannot both
claim the same tree. A later opposite-width request fails at Make parse time,
before an existing archive can satisfy a target. A legacy tree containing
an unstamped generated header, object, archive, test, benchmark, or example
also fails and must use a new directory or be cleaned. Reusing a stamped
tree at the same width is allowed.

The generated version header now has one recipe, with the stamp as an
order-only prerequisite. Its variable is defined before object rules, so
their header prerequisite is effective under parallel Make. This avoids
relying on the build-directory recipe to generate the header concurrently
with object compilation. `lint` depends on the selected `BUILDDIR` header
instead of hard-coded `build/include/sparse_version.h`.

## Verification

Run from the repository root:

```sh
bash -n tests/test_make_idx_width.sh
bash tests/test_make_idx_width.sh
make format-check
```

The isolated script uses temporary, independent 32- and 64-bit build
directories. It builds the static library, `test_sparse_matrix`,
`bench_matmul`, and `example_basic_solve` at each width. It then runs
`test_sparse_matrix`, whose `test_idx_width_contract` compares
`sizeof(idx_t) * CHAR_BIT`, `SPARSE_IDX_BITS`, and the archive's
`sparse_idx_bits()` result. Both widths built and passed. The benchmark and
example were built, not executed.

| Case | Expected result | Observed result |
| --- | --- | --- |
| Omitted selector | Compiler commands use only `-DSPARSE_IDX_BITS=32`. | Dry run passed. |
| Explicit 32 and 64 | Library/test/benchmark/example build; width stamp matches; matrix test passes. | Both paths passed. |
| Same-width reuse | Existing stamped tree remains usable. | Both paths passed. |
| `0`, `16`, `128`, `text`, empty | Make parse error with allowed-values diagnostic. | All five rejected before build. |
| Conflicting macro in `CFLAGS`, including split `-D`; conflicting macro in `CC` | Make parse error naming the conflicting variable. | All three rejected before build. |
| 32-bit tree requested as 64-bit | Error names cached and requested widths; archive checksum unchanged. | Rejected before archive change. |
| Unstamped object or generated header | Error requires a new/clean build directory. | Both rejected before build. |
| 64-bit lint | Strict compile and clang-tidy commands carry `-DSPARSE_IDX_BITS=64`. | Dry-run commands passed inspection. |

The script summary was **23 passed, 0 failed**. It does not clean or modify
the repository's default `build/` tree. `make format-check` also passed.
`make clean BUILDDIR=<tree>` resets that build directory and the shared
`coverage/` directory. Separate `BUILDDIR` values are preferred for
side-by-side widths. Duplicate
command-line assignments to the same Make variable cannot expose their
earlier values to the Makefile; the effective final selector is still
validated and stamped.

## Remaining Boundary

This proves two width-consistent Make *build trees*, not two safe installed
packages. The generated header still records only the library version, and
`sparse.pc` still lacks a width flag. Day 5 owns Make/CMake parity and
generated identity inputs. Days 6-10 own installed metadata, validator
mutations, and consumer mismatch rejection. The focused matrix test does not
establish broad 64-bit algorithm correctness.
