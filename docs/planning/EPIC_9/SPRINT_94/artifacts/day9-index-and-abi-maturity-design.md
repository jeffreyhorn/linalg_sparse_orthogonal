# Sprint 94 Day 9 - Index and ABI Maturity Design

## Scope
- Required Day 10 center:
  - `src/sparse_matrix.c`
- Directly forced follow-through only if the implementation truly needs it:
  - `include/sparse_matrix.h`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`
- Explicitly later:
  - broader solver-family owners
  - broader package or maintainer wording
  - broader binary or shared-library claim widening

## Main Result

Sprint 94 now has one exact touched index/ABI implementation contract.

The next batch should stay inside the matrix-shell owner and strengthen the
width-aware reading of:

- Matrix Market save/load paths
- debug and inspection printing paths
- touched consumer interpretation of the configured `idx_t` width

This is not a typedef or product-claim rewrite. The public width contract is
already real through:

- `SPARSE_IDX_BITS`
- `idx_t`
- `SPARSE_PRIDX`
- `SPARSE_SCNIDX`
- `sparse_idx_bits()`

The remaining job is touched-path maturity on the owner that actually prints,
parses, and exposes matrix dimensions and nonzero counts to users and
downstream diagnostics.

## Exact Day 10 Target

- strengthen the matrix-shell save/load and debug/inspection seam so it reads
  as width-aware and ABI-safe under both `SPARSE_IDX_BITS=32` and
  `SPARSE_IDX_BITS=64`
- keep the batch bounded to formatting, parsing, and touched-path consumer
  interpretation
- avoid widening into solver-family, package, or fake dynamic-ABI claims

## Preserved Invariants

- `SPARSE_IDX_BITS`, `idx_t`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` stay the
  authoritative width contract
- the reviewed default build remains unchanged unless the touched batch
  explicitly says otherwise
- no broad binary-compatibility, shared-library, or platform-symmetry claim is
  introduced
- proof remains focused on matrix-shell consumer and diagnostic surfaces

## Directly Forced Proof/Support Owners

- `include/sparse_matrix.h`
- `tests/test_sparse_matrix.c`
- `tests/test_sparse_io.c`

## Deferred From This Batch

- solver-family breadth follow-through
- broader README / INSTALL / maintainer wording
- benchmark or reporting interpretation
- generic 64-bit modernization claims outside the touched matrix-shell seam

## Interpretation

The right Day 10 move is one bounded matrix-shell index-width maturity batch.
It should make the touched consumer and diagnostic surfaces stronger without
pretending the whole repo has already completed broad ABI modernization.
