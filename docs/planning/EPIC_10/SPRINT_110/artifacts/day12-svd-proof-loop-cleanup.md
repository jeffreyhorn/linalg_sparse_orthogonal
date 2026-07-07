# Sprint 110 Day 12: SVD Proof-Loop Cleanup

## Purpose

Day 12 implemented the Day 11-selected SVD setup-helper family in
`tests/test_svd.c`. The cleanup is limited to repeated 5x4 rank-deficient
matrix construction and does not move rank, QR, reconstruction,
orthogonality, pseudoinverse, low-rank, partial-SVD, or condition-number proof
logic.

## Implemented Helper

Added one local static fixture builder:

```c
static SparseMatrix *make_svd_rank_deficient_colpair_5x4(void);
```

The helper builds the repeated 5x4 fixture where:

- column 1 duplicates column 0;
- column 3 duplicates column 2;
- the expected numerical rank is 2.

The helper owns only matrix construction and insertion failure cleanup.

## Updated Call Sites

The helper replaced duplicated matrix setup in:

- `test_svd_rank_vs_qr`;
- `test_svd_rank_deficient`.

## Proof Visibility Preserved

The following proof values and actions remain visible at call sites:

- `sparse_svd_compute`;
- `sparse_svd_rank`;
- `tol_svd = 1e-8 * svd.sigma[0]`;
- the manual SVD rank-count loop;
- QR factorization and `qr.rank` comparison;
- expected rank value `2`;
- printed rank labels;
- cleanup ownership.

## Explicit Non-Changes

- No public SVD header or option default changed.
- No `tests/test_svd_partial_helpers.h` change.
- No helper target or shared test helper header was added.
- No reconstruction, orthogonality, pseudoinverse, low-rank, partial-SVD, or
  condition-number proof loop moved.
- No `RUN_TEST` registration or CTest count changed.

## Validation Plan

Because Day 12 modifies a `.c` test file, required validation is:

- `make build/test_svd`;
- `build/test_svd`;
- `make format && make lint && make test`;
- `git diff --check`;
- trailing-whitespace scan over touched Sprint 110 docs and `tests/test_svd.c`.

Validation results are recorded in `WORKING_NOTES.md`.

## Residual Deferrals

Remaining SVD proof-owner cleanup candidates stay deferred:

- reconstruction helper movement;
- U/Vt orthogonality helper movement;
- Moore-Penrose helper extraction;
- dense and sparse low-rank proof-loop cleanup;
- partial-SVD vector/residual cleanup;
- condition-number proof cleanup.
