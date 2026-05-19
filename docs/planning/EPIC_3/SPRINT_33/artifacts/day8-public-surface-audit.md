# Sprint 33 Day 8 Public-Surface Audit

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Review the four Day 7 `public-surface-review` findings against the actual exported/documented contract so Sprint 33 does not mix public API removal into the first internal dead-code cleanup pass.

## Audit Set

From the Day 7 classified report:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

## Decision Summary

All four items are `keep`.

None of them should remain in Sprint 33’s cleanup queue after Day 8.

## Per-Symbol Decisions

### `givens_apply_right`

Decision:

- `keep`

Reason:

- declared in installed public header `include/sparse_dense.h`
- documented as part of the dense utility API
- paired conceptually with `givens_apply_left`, which is actively used internally
- absence from current in-repo call sites does not override the exported API contract

## `sparse_print_dense`

Decision:

- `keep`

Reason:

- declared in installed public header `include/sparse_matrix.h`
- documented as public display/debug surface
- exercised by `tests/smoke_test.c`, which backs the maintained `make smoke` operator path

## `sparse_print_entries`

Decision:

- `keep`

Reason:

- declared in installed public header `include/sparse_matrix.h`
- documented as the non-zero-entry print helper
- part of the long-standing matrix-display API surface in earlier project planning/history

## `sparse_print_info`

Decision:

- `keep`

Reason:

- declared in installed public header `include/sparse_matrix.h`
- documented as the matrix-summary print helper
- exercised by `tests/smoke_test.c` on the maintained `make smoke` path

## Install-Surface Confirmation

These are real public API exports, not accidental local declarations:

- CMake installs the `include/` tree under `${CMAKE_INSTALL_INCLUDEDIR}/sparse`
- therefore removal would be a public API change

## Refined Cleanup Queue

After Day 8, the first-pass Sprint 33 cleanup queue is:

- `chol_csc_dump_supernodes`

Public-surface findings no longer in scope for Sprint 33 cleanup:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

## Day 8 Conclusion

The Day 7 public-review bucket did its job: it prevented exported utility symbols from being misread as easy dead-code wins.

Sprint 33 can now move into Day 9 cleanup design with a tighter and safer queue:

- one internal candidate for deletion planning
- zero approved public API removals
- no unresolved public-surface questions from the current `xunused` set
