# Day 9 SVD Proof-Owner Boundary

## Purpose

Day 9 defines the safe cleanup boundary for `tests/test_svd.c` before any SVD
test code is edited. The Day 10 cleanup should reduce repeated fixture setup
without moving rank interpretation, singular-value oracle checks,
reconstruction math, pseudoinverse interpretation, low-rank error bounds, or
condition-number expectations away from the tests that own those claims.

## File Snapshot

- File: `tests/test_svd.c`
- Current size: 2,879 lines
- Existing local proof helpers:
  - `gk_reconstruction_error`
  - `orthogonality_error`
  - `validate_gk`
- Existing included helper surface:
  - `test_svd_partial_helpers.h` for partial-SVD vector and oracle helpers

The file already has proof helpers for reconstruction and partial-SVD checks.
Day 10 should not add a shared helper header, compiled helper target, public
header, Makefile entry, CMake entry, or test registration change.

## Proof-Owner Inventory

### Rank and Singular-Value Proofs

These tests intentionally keep expected rank or singular-value interpretation
near the fixture:

- `test_svd_basic_sigma`
- `test_svd_diagonal_5x5`
- `test_svd_descending`
- `test_svd_rank1`
- `test_svd_rank1_uv`
- `test_svd_rank2`
- `test_svd_rank5_in_10x10`
- `test_svd_rank1_square`
- `test_svd_rank1_wide`
- `test_svd_near_singular`
- `test_svd_multi_zero_diag`
- `test_svd_rank2_dense`
- `test_svd_suitesparse_rank_deficient`
- `test_svd_rank_full`
- `test_svd_rank_deficient`
- `test_svd_rank_nearly_singular`
- `test_s103_svd_diag6_rank_threshold_claim`

Day 10 may reuse matrix builders for selected fixtures, but it must not move
the expected rank, tolerance, singular-value ordering, or threshold claims.

### Reconstruction and Orthogonality Proofs

These tests own reconstruction or orthogonality interpretation:

- Golub-Kahan extraction tests using `gk_reconstruction_error`
- `validate_gk` callers
- `test_svd_with_uv`
- `test_svd_rank1_uv`
- `test_svd_rank2_dense`
- `test_svd_wide_5x10_uv`
- full-mode U/Vt tests around lines 2246-2460
- `test_s103_svd_diag6_rank_threshold_claim`

Day 10 must not introduce new reconstruction assertion helpers. Any fixture
builder call should describe only the input matrix.

### Partial-SVD and External Oracle Proofs

Partial-SVD vector, residual, corpus, and full-SVD comparison logic is already
covered by `test_svd_partial_helpers.h` and test-local assertions. Day 10 must
not move:

- partial-SVD versus full-SVD comparisons;
- vector orthogonality checks;
- `A*v ~= sigma*u` residual checks;
- SuiteSparse corpus tolerances;
- timing or bounded-corpus notes.

### Pseudoinverse, Low-Rank, and Condition-Number Proofs

These sections use many diagonal and structured fixtures, but their proof
intent is in the expected pseudoinverse entries, Moore-Penrose products,
low-rank residual/error bounds, sparse-vs-dense comparisons, and condition
number values. Day 10 may use a literal diagonal fixture builder in selected
call sites, while all proof interpretation must remain inline.

## Safe Fixture Candidates

### Diagonal Matrix Builder

A local diagonal builder is safe when its name and arguments make the fixture
obvious:

```c
static SparseMatrix *make_svd_diag_matrix(idx_t rows, idx_t cols, const double *diag,
                                          idx_t diag_len);
```

Rules:

- Insert only the first `diag_len` diagonal values.
- Skip zero diagonal values to preserve sparse fixture shape.
- Return `SparseMatrix *` or `NULL`.
- Do not sort expected singular values.
- Do not compute rank, condition number, residuals, or low-rank errors.
- Tests retain `ASSERT_NOT_NULL(A); if (!A) return;`.

### Rank-1 Row-Progression Builder

The repeated rank-1 fixture `A[i,j] = i + 1` is safe if the builder is named
literally and the expected singular value or reconstruction assertion stays at
the call site:

```c
static SparseMatrix *make_svd_rank1_row_progression(idx_t rows, idx_t cols);
```

Rules:

- Insert every entry as `(double)(i + 1)`.
- Do not compute expected singular values.
- Do not assert rank, reconstruction, or low-rank error.
- Use only where this exact fixture is already present.

## Selected Day 10 Batch

Limit Day 10 to local fixture builders and approved call sites.

### Approved Diagonal Call Sites

Use `make_svd_diag_matrix` only in:

| call site | fixture values | proof preserved inline |
|---|---|---|
| `test_svd_diagonal_5x5` | `{7, -3, 5, 1, -9}` | descending singular values |
| `test_lowrank_diagonal` | `{10, 5, 2, 1}` | dense low-rank entries and Frobenius error |
| `test_lowrank_sparse_diagonal` | `{10, 5, 2, 1}` | sparse rank-2 entries |
| `test_cond_diagonal` | `{100, 10, 1}` | condition number 100 |
| `test_cond_ill_conditioned` | `{1e6, 1, 1e-6}` | condition number 1e12 |
| `test_cond_rectangular` | `{3, 1}` into a 4x2 matrix | rectangular condition number 3 |

### Approved Rank-1 Call Sites

Use `make_svd_rank1_row_progression` only in:

| call site | fixture shape | proof preserved inline |
|---|---|---|
| `test_svd_rank1` | 4x3 | expected leading singular value and near-zero tail |
| `test_svd_rank1_uv` | 4x3 | explicit U sigma Vt reconstruction loop |

Do not broaden Day 10 to square/wide rank-1 tests, low-rank sparse rank-1
tests, rank-deficient duplicated-column tests, full-mode fixtures, or
SuiteSparse corpus tests.

## No-Move Rules

Day 10 must keep these statements visible at the call sites:

- expected singular values and ordering;
- rank thresholds and tolerance interpretation;
- reconstruction loops and residual thresholds;
- U/Vt orthogonality checks;
- pseudoinverse and Moore-Penrose products;
- dense low-rank and sparse low-rank error bounds;
- sparse-vs-dense low-rank comparisons;
- condition-number expectations;
- SuiteSparse corpus labels and tolerances;
- `RUN_TEST` registration.

## Placement and Build Impact

- File: `tests/test_svd.c`
- Placement: local helpers near existing SVD test helpers, before the first
  test section
- Linkage: `static`
- Build-system impact: none

No public header, internal header, Makefile, CMake, or `RUN_TEST` update is
needed.

## Validation Plan

Because Day 10 will edit a `.c` file, required validation is:

```sh
make build/test_svd && ./build/test_svd
make format && make lint && make test
git diff --check
```

No CTest registration count check is expected because Day 10 should not add,
remove, or rename `RUN_TEST` entries.

## Deferred SVD Cleanup

The following cleanup remains intentionally deferred:

- rank-deficient duplicated-column builders;
- square and wide rank-1 builders beyond the two approved 4x3 call sites;
- shared reconstruction assertion helpers;
- pseudoinverse and Moore-Penrose helper movement;
- partial-SVD oracle helper changes;
- SuiteSparse corpus fixture helpers;
- full-mode U/Vt fixture unification;
- condition-number assertion helpers.

These are candidates for later cleanup only after focused validation preserves
the Day 10 local-builder approach.
