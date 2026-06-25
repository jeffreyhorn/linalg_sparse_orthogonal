# Sprint 91 Day 6: Construction / Import Batch

## Purpose

Land the first bounded compressed-first implementation seam by promoting CSR
and CSC inputs to first-class public construction entry paths without
reopening broader lifecycle or publication ownership.

## Main Result

Sprint 91 now has a real compressed-first public construction lane:

- `include/sparse_csr.h` now exposes:
  - `sparse_create_from_csr(const SparseCsr *csr)`
  - `sparse_create_from_csc(const SparseCsc *csc)`
- the legacy out-parameter imports remain:
  - `sparse_from_csr(const SparseCsr *csr, SparseMatrix **mat)`
  - `sparse_from_csc(const SparseCsc *csc, SparseMatrix **mat)`
- the old imports now behave as compatibility wrappers around the same shared
  validated build seam

That means callers who already own compressed sparse inputs no longer need to
conceptually begin from `sparse_create()` plus linked-list insertion just to
enter the public matrix-shell workflow.

## Landed Implementation Shape

The landed code stayed inside the Day 5 fence:

- public header:
  - `include/sparse_csr.h`
- matching implementation seam:
  - `src/sparse_csr.c`
- directly forced proof-owner follow-through:
  - `tests/test_csr.c`

The implementation split is now:

- shared validators:
  - CSR and CSC structural validation are factored into local helper seams
- shared builders:
  - CSR and CSC import both route through shared validated build helpers
- first-class constructor entry points:
  - the new `sparse_create_from_*` APIs return a new `SparseMatrix *`
  - they preserve current physical-index-space truth and shell compatibility
- retained compatibility wrappers:
  - `sparse_from_*` still returns `sparse_err_t`
  - both legacy entry points now reuse the same shared builder path

## Proof Follow-Through

The proof follow-through stayed tightly bounded:

- `tests/test_csr.c` now proves:
  - direct constructor-style CSR entry
  - direct constructor-style CSC entry
  - null rejection on the new compressed-first entry points
- the existing round-trip and SuiteSparse conversion tests continue to prove
  compatibility on the retained `sparse_from_*` path

No additional support-only follow-through was required in:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`

## Validation

Because `*.h` and `*.c` files changed, the full implementation-day queue was
run:

- `make format`
- `make lint`
- `make test`

All passed cleanly after one local `-Wshadow` fix inside the new shared helper
seams.

Representative retained proof:

- `test_csr` now includes explicit compressed-first constructor coverage for
  both CSR and CSC entry
- the repo-wide test suite stayed green under `make test`

## Exit State

- Sprint 91 now has a real first-class compressed-input construction lane.
- The linked-list shell remains the mutable compatibility owner, but it is no
  longer the only public conceptual entry path for CSR/CSC-backed callers.
- Day 7 can rerank the remaining shell-first costs from a landed validated
  implementation rather than from a pure design contract.
