# Sprint 41 Day 4 Artifact: Core Utility Implementation

## Purpose

Record the first implemented landing of Sprint 41's shared internal
allocation/overflow helper layer, including the actual build wiring, the
narrow proof integrations, and the validation results that turned the Day 3
design into a real maintained module.

## Implemented Shared Helper Layer

Day 4 added the planned private helper pair:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

### Header-inline arithmetic / bounds helpers

The new private header now provides:

- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`

These cover the strongest shared arithmetic seam Day 2 measured:

- repeated `size_mul_overflow(...)`
- count-to-bytes derivation
- representability checks for later migration work

### Source-backed allocation helpers

The new private source now provides:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`

Current contract:

- zero-size requests return success with `*out = NULL`
- overflow maps to `SPARSE_ERR_ALLOC`
- allocation failure maps to `SPARSE_ERR_ALLOC`
- no public API exposure is introduced

## Build-Surface Wiring

Day 4 added `src/sparse_alloc_internal.c` to both maintained library build
lists:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` library source list

This makes the helper layer part of the normal maintained build rather than a
one-off local experiment.

## First Low-Risk Proof Integrations

### `src/sparse_dense.c`

The first source-backed allocation proof point is in `dense_create()`:

- dense element-count validation now uses:
  - `sparse_size_mul_overflow(...)`
- dense storage allocation now uses:
  - `sparse_calloc_array(...)`

This replaces one local hand-written overflow/allocation path with the shared
helper layer while preserving the caller-visible `NULL` failure contract.

The file also now uses `sparse_size_mul_overflow(...)` in the
`tridiag_qr_eigenpairs(...)` workspace-size path.

### `src/sparse_qr.c`

The first arithmetic-only proof batch in a broader reusable module is:

- removal of the file-local `size_mul_overflow(...)`
- replacement of its overflow-check call sites with:
  - `sparse_size_mul_overflow(...)`

This proves the new inline helper tier is practical in a non-Day-2-hotspot
file that already carried the same duplication pattern.

## Validation / Cleanup Notes

The first validation pass surfaced two helper-source issues:

1. clang-analyzer rejected a zero-byte `calloc(...)` path in
   `sparse_calloc_array(...)`
2. cppcheck flagged a redundant overflow branch after the new zero-size
   early-return

Both were fixed inside `src/sparse_alloc_internal.c`:

- zero-size requests now return before any allocator call
- `sparse_calloc_array(...)` now uses one validated-bytes path with
  `calloc(1, bytes)`

These fixes are part of the Day 4 result, not postscript cleanup.

## Day 4 Validation Result

Because `*.c` / `*.h` files changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed after the two helper-source cleanups above.

## Highest-Value Day 4 Conclusion

Sprint 41 now has a real shared internal safety-helper layer, not just a
design note. The implementation is intentionally narrow:

- shared helper module exists
- both build systems include it
- one source-backed allocation proof point landed
- one broader arithmetic-helper proof point landed
- the full required gate passed

That is the right foundation for Day 5 and Day 6, where the first planned
hotspot migration batches can begin removing larger amounts of local helper
duplication.
