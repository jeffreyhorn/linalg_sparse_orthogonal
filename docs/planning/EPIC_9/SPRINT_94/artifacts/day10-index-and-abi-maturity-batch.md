# Sprint 94 Day 10 - Index and ABI Maturity Batch

## Scope
- Required center:
  - `src/sparse_matrix.c`
- Directly forced follow-through:
  - `tests/test_sparse_io.c`
  - `tests/test_sparse_matrix.c`
- Explicitly not widened in this batch:
  - broader solver-family owners
  - public width typedefs or ABI-policy claims
  - broader README / INSTALL / maintainer wording

## Landed Changes
- Added checked stream-print helpers on the touched matrix-shell owner so:
  - `sparse_save_mm(...)`
  - `sparse_print_dense(...)`
  - `sparse_print_entries(...)`
  - `sparse_print_info(...)`
  now fail closed with `SPARSE_ERR_IO` when the write path fails
- Tightened Matrix Market parsing on the touched matrix-shell load path so it
  now rejects:
  - negative dimensions or entry counts
  - rectangular `symmetric` headers
  - zero 1-based coordinates
  - out-of-range coordinates
- Kept the batch bounded to the matrix-shell save/load and diagnostic seam

## Focused Proof Follow-Through
- `tests/test_sparse_io.c`
  - malformed negative-dimension file rejected
  - malformed rectangular symmetric header rejected
  - malformed out-of-range coordinate rejected
  - malformed zero coordinate rejected
- `tests/test_sparse_matrix.c`
  - width-aware diagnostic output smoke test over:
    - `sparse_print_info(...)`
    - `sparse_print_entries(...)`
    - `sparse_print_dense(...)`

## Preserved Invariants
- `SPARSE_IDX_BITS`, `idx_t`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` remain the
  authoritative public width contract
- no typedef rewrite or broad ABI-policy change was introduced
- no solver-family breadth claim widened
- no new shared-library or broad binary-compatibility claim was implied

## Validation
- `make format`
- `make lint`
- `make test`

## Result
- The matrix-shell consumer and diagnostic seam now reads as more trustworthy
  under both `SPARSE_IDX_BITS=32` and `SPARSE_IDX_BITS=64`.
- The touched owner now rejects several malformed Matrix Market cases
  explicitly instead of tolerating them silently, and it treats write failures
  as real I/O errors.
