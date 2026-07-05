# Day 10 SVD Proof-Owner Cleanup

## Purpose

Day 10 implements the bounded SVD fixture cleanup selected by the Day 9
boundary. The cleanup reduces repeated matrix setup in `tests/test_svd.c`
without moving rank interpretation, singular-value oracle checks,
reconstruction math, pseudoinverse interpretation, low-rank error bounds,
partial-SVD comparisons, or condition-number expectations out of the tests.

## Implemented Cleanup

### Diagonal Matrix Builder

Added one local static fixture builder near the existing SVD helpers:

```c
static SparseMatrix *make_svd_diag_matrix(idx_t rows, idx_t cols, const double *diag,
                                          idx_t diag_len);
```

The builder only creates a sparse matrix and inserts nonzero diagonal values up
to the matrix diagonal and `diag_len`. It does not sort expected singular
values, compute rank, compute condition numbers, compute low-rank errors, or
assert any SVD behavior.

### Rank-1 Row-Progression Builder

Added one local static fixture builder for the repeated 4x3 rank-1 pattern:

```c
static SparseMatrix *make_svd_rank1_row_progression(idx_t rows, idx_t cols);
```

The builder only inserts `A[i,j] = (double)(i + 1)`. It does not compute the
expected leading singular value, rank, reconstruction residual, or low-rank
error.

## Updated Call Sites

### Diagonal Fixture Call Sites

The diagonal builder is used in:

- `test_svd_diagonal_5x5`
- `test_lowrank_diagonal`
- `test_lowrank_sparse_diagonal`
- `test_cond_diagonal`
- `test_cond_ill_conditioned`
- `test_cond_rectangular`

Each test still owns its expected singular values, low-rank entries,
Frobenius-error calculation, sparse low-rank expectations, or condition-number
claim inline.

### Rank-1 Fixture Call Sites

The rank-1 row-progression builder is used in:

- `test_svd_rank1`
- `test_svd_rank1_uv`

The expected leading singular value and explicit U sigma Vt reconstruction
loop remain inline at the test sites.

## Proof Preservation

The cleanup keeps the following proof logic inline:

- singular-value ordering and expected singular values;
- rank thresholds and tolerance interpretation;
- reconstruction loops and residual thresholds;
- U/Vt orthogonality checks;
- pseudoinverse and Moore-Penrose products;
- dense and sparse low-rank error calculations;
- sparse-vs-dense low-rank comparisons;
- condition-number expectations;
- SuiteSparse corpus labels and tolerances.

No partial-SVD helpers, external oracle helpers, SuiteSparse corpus tests,
full-mode U/Vt fixtures, or `RUN_TEST` entries were changed.

## Build-System and Registration Impact

- No public header changes.
- No internal header changes.
- No Makefile or CMake changes.
- No new compiled helper target.
- No `RUN_TEST` additions, removals, or renames.
- No reviewed CTest registration count change is expected.

## Validation Plan

Because Day 10 edits a `.c` file, validation is:

```sh
make build/test_svd && ./build/test_svd
make format && make lint && make test
git diff --check
```

Also run a trailing-whitespace scan over Sprint 107 planning docs, the Epic 10
project plan, and touched C files.

## Deferred SVD Cleanup

The following cleanup remains deferred:

- duplicated-column rank-deficient builders;
- square and wide rank-1 builders;
- shared reconstruction assertion helpers;
- pseudoinverse and Moore-Penrose helper extraction;
- partial-SVD oracle helper changes;
- SuiteSparse corpus fixture helpers;
- full-mode U/Vt fixture unification;
- condition-number assertion helpers.
