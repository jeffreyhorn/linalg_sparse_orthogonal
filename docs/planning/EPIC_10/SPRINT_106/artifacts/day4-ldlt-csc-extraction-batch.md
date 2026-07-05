# Sprint 106 Day 4 LDLT/CSC Extraction Batch

## Purpose

Day 4 implements the Day 3 LDLT CSC boundary decision by extracting
row-adjacency ownership from the largest direct CSC source owner. The change is
a private implementation split only: it reduces mixed responsibility in
`src/sparse_ldlt_csc.c` without changing solver semantics, public APIs, test
registration, or product claims.

## Implemented Change

### New Private Owner

Added:

```text
src/sparse_ldlt_csc_rowadj.c
```

The new file owns:

- `ldlt_csc_row_adj_append(...)`
- `ldlt_csc_row_adj_swap_slots(...)`
- `ldlt_csc_populate_row_adj(...)`

These helpers are private to the LDLT CSC backend and are declared through
`src/sparse_ldlt_csc_internal.h`.

### Original Source Owner After Extraction

`src/sparse_ldlt_csc.c` still owns:

- `LdltCsc` lifecycle entry points;
- sparse-to-CSC conversion;
- analysis-aware conversion;
- public LDLT writeback;
- validation;
- wrapper/native kernel selection;
- workspace lifecycle;
- native Bunch-Kaufman cmod and pivot logic;
- supernodal orchestration.

It now delegates row-adjacency slot swapping and row-adjacency population to
the new private owner.

## Files Changed

| file | change |
|---|---|
| `src/sparse_ldlt_csc_rowadj.c` | new private row-adjacency implementation owner |
| `src/sparse_ldlt_csc.c` | removed row-adj append/populate bodies and replaced inline row-adj swap block with helper call |
| `src/sparse_ldlt_csc_internal.h` | added private declarations for row-adj swap and population helpers |
| `build-metadata/library_sources.txt` | added new library source |
| `Makefile` | added new source to `LIB_SRCS` |
| `CMakeLists.txt` | added new source to CMake library target |

No public header under `include/` changed. No test registration changed.

## Source-List and Build Membership

The synchronized LDLT CSC source order is now:

```text
src/sparse_ldlt_dense.c
src/sparse_ldlt_csc.c
src/sparse_ldlt_csc_rowadj.c
src/sparse_ldlt_csc_supernodal.c
```

This order is reflected in:

- `build-metadata/library_sources.txt`
- `Makefile`
- `CMakeLists.txt`

Source-list validation result:

```text
source-list-check: PASS (43 library sources)
```

## Before/After Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 lines | 2,092 lines | -82 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 lines | 82 lines | +82 |
| `src/sparse_ldlt_csc_internal.h` | 929 lines | 947 lines | +18 |

Interpretation:

- The monolithic LDLT CSC owner is smaller and has one less helper family.
- The new row-adj owner is narrow enough to review independently.
- Header growth is limited to private helper declarations and documentation.

## Behavior Preservation

The extraction preserved:

- row-adj append error codes and geometric growth behavior;
- row-adj slot swap semantics during symmetric pivot swaps;
- row-adj population after scalar and supernodal writeback;
- native/wrapper dispatch policy;
- conversion and analysis-aware conversion semantics;
- public writeback semantics;
- public API surface;
- reviewed CTest registration count.

## Focused Validation

Focused direct solver validation passed:

```sh
python3 scripts/check_library_sources.py
make build/test_ldlt_csc build/test_direct_csc_regression build/test_ldlt build/test_ldlt_backend_dispatch
./build/test_ldlt_csc
./build/test_direct_csc_regression
./build/test_ldlt
./build/test_ldlt_backend_dispatch
```

Observed focused test results:

| test | result |
|---|---|
| `test_ldlt_csc` | 99 tests, 0 failures, 0 skips, 2,318 assertions |
| `test_direct_csc_regression` | 8 tests, 0 failures, 0 skips, 42 assertions |
| `test_ldlt` | 89 tests, 0 failures, 0 skips, 912 assertions |
| `test_ldlt_backend_dispatch` | 20 tests, 0 failures, 0 skips, 128 assertions |

## Required Full Gate

Because Day 4 modified `.c` and `.h` files, the required full quality gate was
run:

```sh
make format && make lint && make test
```

Result:

```text
All tests passed.
```

## CMake Parity Validation

Reviewed CMake compile/parity validation passed:

```sh
make quality-review-cmake-compile
```

Result:

```text
quality-review-cmake-compile: CMake tests: 54, Makefile tests: 54
quality-review-cmake-compile: PASS: test counts match
quality-review-cmake-compile: passed (configure + clean rebuild + ctest -N + test-count parity)
```

## Final Hygiene

Final hygiene checks passed:

```sh
git diff --check
rg -n "[ \t]+$" src/sparse_ldlt_csc.c src/sparse_ldlt_csc_internal.h \
  src/sparse_ldlt_csc_rowadj.c build-metadata/library_sources.txt \
  Makefile CMakeLists.txt docs/planning/EPIC_10/SPRINT_106
python3 scripts/check_library_sources.py
```

The trailing-whitespace scan produced no matches, and the final source-list
recheck still reported 43 library sources.

## Non-Changes

Day 4 did not:

- change the `LdltCsc` struct layout;
- change public headers;
- change public APIs;
- change test registration;
- change native versus wrapper dispatch;
- change conversion, symbolic-analysis-aware conversion, or writeback
  semantics;
- change supernodal detection or dense LDLT behavior.

## Day 4 Result

Day 4 completed the first Sprint 106 source extraction. LDLT CSC
row-adjacency ownership now lives in a focused private source file, and both
Make and CMake maintained build surfaces validate the new source membership.
