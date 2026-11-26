# Sprint 228 Day 13: Focused Cross-Build and Sanitizer Validation

## Decision

The changed low-rank reconstruction surface passes the focused Make, CMake,
index-width, sanitizer, leak, formatting, documentation, and dependency-surface
gates required by Day 13. The validation used fresh, width-specific build
directories so ordinary, 64-bit, and instrumented objects could not be mixed.

This result validates correctness, ownership, allocation-failure cleanup, and
the narrow reconstruction-memory contract. It does not override the Day 11/12
timing decision: square and wide timing remain outside the frozen promotion
limits, so no speed claim is accepted.

## Focused Make matrix

The following targets were built independently in
`build/s228-day13-make32` and `build/s228-day13-make64`:

- `test_svd`;
- `test_svd_partial_corpus`;
- `test_idx_width_functional`; and
- `bench_svd_lowrank_memory`.

| Width | SVD suite | Partial-SVD corpus | Width lane | Result |
| --- | ---: | ---: | ---: | --- |
| 32-bit | 136 tests, 19,369 assertions | 10 tests, 247 assertions | 5 tests, 111 assertions | Pass; typedef, generated header, and archive all report 32 bits |
| 64-bit | 136 tests, 19,369 assertions | 10 tests, 247 assertions | 5 tests, 115 assertions | Pass; typedef, generated header, and archive all report 64 bits |

Each width therefore passed 151 focused tests. The SVD suite includes the
square/tall/wide differential corpus, threshold and cancellation cases,
selector validation, exhaustive allocation sweeps, numeric failures, mixed-call
lifecycle, and component-memory accounting. The separate partial-SVD corpus
retains its projector, residual, fail-closed, recovery, and sparse-output cases.

One direct benchmark smoke pair was also run at each width. All four processes
returned status zero, `correct=true`, `nnz=16`, and checksum
`628ad8ec84c5f843`. At each width the default route reported zero
reconstruction-scratch bytes and allocations, while exact `off` reported one
2,097,152-byte accumulator. The 64-bit shell/map and slab byte counts are
larger because `idx_t` is wider; this is expected and is not a reconstruction
scratch regression. These smoke runs are contract checks, not timing evidence.

## CMake and CTest matrix

Fresh CMake trees were configured at both widths with exported compile
commands. Each tree built the same three focused tests and the memory probe,
and each selected CTest run passed 3/3 tests. The maintained reviewed parity
wrapper then performed a clean complete build in
`build/s228-day13-parity` and reported:

- 63 registered CTests;
- 62 Make test binaries;
- one intentional CTest-only allocation-failure selector; and
- exact registration parity.

Thus the new benchmark is present in both build systems and the affected SVD
tests remain registered under CMake. Day 14 still owns full execution of all 63
CTest entries.

## Sanitizer and leak matrix

| Configuration | Selected execution | Result |
| --- | --- | --- |
| Apple Clang 17, 32-bit, ASan+UBSan | SVD, partial-SVD corpus, width lane | 3/3 CTests pass; no sanitizer finding |
| Apple Clang 17, 64-bit, ASan+UBSan | SVD, partial-SVD corpus, width lane | 3/3 CTests pass; no sanitizer finding |
| Homebrew Clang 22.1.5, 32-bit, ASan+UBSan with leak detection enabled | SVD and partial-SVD corpus | 2/2 CTests pass; no ASan, LeakSanitizer, or UBSan report |

Apple's local sanitizer runtime was run with leak detection disabled; the
separate Homebrew-Clang execution supplies the maintained leak check. The
instrumented SVD run includes every public-call allocation failure index,
partial construction cleanup, one-beyond success, and post-failure retry.

## Warning and generated-surface audit

`make lint` passes on the final C/header tree, including strict `-Werror`
compilation, clang-tidy, and cppcheck. `make format-check` also passes. The
non-`-Werror` CMake build prints existing Apple-Clang diagnostics in two
understood classes:

1. `-Wdouble-promotion` where the SDK defines `NAN` and `INFINITY` as
   float-valued macros used in double status paths and explicit non-finite test
   fixtures; and
2. an unchanged test-only unused-variable diagnostic outside the SVD surface.

The changed production files `src/sparse_bidiag.c`, `src/sparse_matrix.c`,
`src/sparse_matrix_build_internal.c`, and `src/sparse_svd.c` emit no new
warning. The new SVD test diagnostics are the deliberate non-finite fixtures
that verify `SPARSE_ERR_BADARG` and `SPARSE_ERR_NUMERIC`. No warning indicates
an implicit narrowing, unchecked allocation, or ownership fault in the changed
route.

Generated and documentation surfaces also pass:

- `make source-list-check`: 49 library sources, with Make/CMake parity;
- `make docs-check`: all 18 checked-in public headers have generated reference
  and source pages;
- `python3 -m py_compile scripts/run_sprint228_memory_packet.py`;
- `make format-check`; and
- `git diff --check` after recording this packet.

## Day 14 handoff

Day 13 found no focused correctness, width, registration, sanitizer, leak, or
dependency-surface blocker. Day 14 must run the complete final-tree quality
chain, all CTests, and evidence-integrity checks, then reconcile the sprint
deliverables. The timing promotion remains blocked and must not be converted
into a universal speed, lower-peak-memory, portable performance, or release
claim.
