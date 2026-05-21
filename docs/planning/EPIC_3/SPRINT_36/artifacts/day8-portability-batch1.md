# Sprint 36 Day 8: Script & Target Portability Fixes — Batch I

## Scope

Implement the first portability cleanup batch in the maintained reviewed
Makefile path without changing the actual Sprint 34 reviewed-contract behavior.

Files changed:

- `Makefile`

## Main Result

The maintained reviewed Makefile path now depends less on external Unix shell
tooling:

- no `find` in the maintained format/lint source enumeration
- no hardcoded `/bin/mkdir` or `/bin/rm` paths in the touched reviewed-path
  surfaces

This is a real portability improvement, but it stays intentionally narrow:

- it improves the Linux/macOS maintainer path
- it does not overclaim Windows Makefile parity
- it does not try to port dead-code or maintenance-only helper workflows

## Changes

### 1. Reviewed source enumeration now uses repo-native file lists

Updated `Makefile` source discovery:

- `ALL_SRC = $(LIB_SRCS) $(wildcard $(SRCDIR)/*.h)`
- `ALL_TEST_SRC = $(wildcard $(TESTDIR)/*.c) $(wildcard $(TESTDIR)/*.h)`
- `ALL_BENCH_SRC = $(BENCH_SRCS)`
- `ALL_EX_SRC = $(EX_SRCS)`
- `ALL_HEADERS = $(wildcard include/*.h)`

Updated `lint` to use:

- `$(LIB_SRCS)` for strict syntax-only compile
- `$(LIB_SRCS)` for `clang-tidy`

Why this matters:

- the repo already maintains explicit build lists
- reusing those lists is more portable and less fragile than re-discovering the
  files through external `find`
- the `tests/*.c` glob keeps `tests/smoke_test.c` and future standalone test C
  files under the formatting/checking contract even though they are not part of
  `$(TEST_SRCS)`
- the maintained reviewed path is now closer to the actual build truth source

### 2. Removed hardcoded absolute Unix tool paths

Updated:

- `mkdir -p` instead of `/bin/mkdir -p`
- `rm -rf` / `rm -f` instead of `/bin/rm ...`

Why this matters:

- absolute tool paths add unnecessary environment assumptions
- removing them is low risk and improves portability hygiene on the maintained
  shell-based paths

## Validation

Ran the touched reviewed commands directly:

- `make quality-review-compile`
- `make quality-review-cmake-compile`

Results:

- `quality-review-compile`: passed
- `quality-review-cmake-compile`: passed
- `ctest -N`: `53`
- Makefile/CMake test-count parity: `53` vs `53`

## What Did Not Change

This batch intentionally did **not** attempt to port or reclassify:

- `deadcode*`
- `wall-check`
- `warning-workflow`
- coverage helpers

Reason:

- those surfaces still have real POSIX/toolchain assumptions
- Sprint 36 should keep them explicit and staged rather than falsely claiming
  all-platform parity

## Conclusion

Day 8 closed the most avoidable portability debt in the maintained reviewed
Makefile path without changing the validated Linux/macOS contract and without
pretending the Unix-maintainer helper workflows are already Windows-ready.
