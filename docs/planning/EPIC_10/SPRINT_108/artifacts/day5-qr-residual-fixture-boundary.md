# Day 5 QR Residual Fixture Boundary

## Purpose

Day 5 defines the QR fixture-cleanup boundary for Sprint 108. The goal is to
identify setup that can move on Day 6 without hiding rank, solve,
reconstruction, refinement, residual, or dense/sparse parity proof intent in
`tests/test_qr.c`.

## Live Inventory

Current `tests/test_qr.c` state:

| Area | Current State | Day 5 Disposition |
|---|---|---|
| Sprint 107 4x3 fixture builders | `make_qr_small_banded_4x3`, `make_qr_duplicate_column_4x3`, `make_qr_near_duplicate_4x3` already exist. | Exclude from Sprint 108 follow-through. |
| Reconstruction helpers | `qr_reconstruction_error` and `assert_qr_reconstruction_below` already isolate reconstruction mechanics. | Do not move further. |
| Residual helpers | `compute_rel_residual` and `assert_qr_true_residual_below` already isolate true-residual mechanics. | Do not move further. |
| SuiteSparse exact-RHS setup | `nos4`, `bcsstk04`, and `west0067` tests allocate exact vectors, build `b = A*x`, factor, solve, and compare locally. | Defer. |
| Tall/economy generated fixtures | Several tests build diagonal-dominant tall fixtures inline. | Select one bounded builder. |
| Diagonal/singleton setup | Small diagonal, single-row, single-column, and 1x1 tests remain clear and local. | Defer. |
| Sparse-mode parity fixtures | Some fixtures are unique to dense/sparse parity tests; one tall diagonal-dominant case repeats economy setup. | Select only repeated tall fixture construction. |
| Refinement fixtures | Refinement tests mix setup with convergence evidence. One overdetermined tall fixture matches the selected construction pattern. | Include only matrix construction. |

## Selected Day 6 Candidate

Add one local fixture builder near the existing QR fixture helpers:

```c
static SparseMatrix *make_qr_tall_diagonal_dominant(idx_t m, idx_t n_cols,
                                                    double diag_value,
                                                    double offdiag_value,
                                                    int include_lower_neighbor);
```

Expected construction:

- create an `m x n_cols` sparse matrix;
- insert `diag_value` at `(i, i)` for `i < min(m, n_cols)`;
- insert `offdiag_value` at first-neighbor positions within the leading
  `n_cols x n_cols` band;
- use `qr_insert_or_free` so insert failures free the partially built matrix;
- return `NULL` on allocation or insertion failure.

The `include_lower_neighbor` flag exists because current call sites differ
slightly:

- `test_economy_solve_tall` and `test_sparse_mode_tall` use upper and lower
  first-neighbor entries with diagonal value `10.0`;
- `test_qr_refine_overdetermined` uses first-neighbor entries in a smaller
  leading band with diagonal value `5.0`.

## Approved Day 6 Call Sites

Only these call sites are approved for Day 6 updates:

- `test_economy_solve_tall`
- `test_sparse_mode_tall`
- `test_qr_refine_overdetermined`

The helper may replace only the matrix construction loop. The following must
remain visible at call sites:

- matrix dimensions;
- RHS allocation and initialization;
- QR option values;
- full/economy/sparse-mode factorization calls;
- solve or refinement calls;
- rank/economy assertions;
- residual comparisons;
- dense/sparse parity assertions.

## Explicit Non-Candidates

### SuiteSparse exact-RHS setup

Do not move the exact-RHS setup in:

- `test_qr_solve_nos4`
- `test_qr_bcsstk04`
- `test_qr_west0067`
- `test_qr_vs_lu`
- `test_qr_refine_nos4`

Rationale: these tests intentionally keep Matrix Market loading, allocation,
exact-solution construction, `sparse_matvec`, factorization, solve, and residual
proof close together. Moving this setup now would hide skip/failure locality and
mix corpus-specific proof behavior with generic fixture construction.

### Existing residual and reconstruction helpers

Do not move or rename:

- `qr_reconstruction_error`
- `assert_qr_reconstruction_below`
- `compute_rel_residual`
- `assert_qr_true_residual_below`

Rationale: these helpers already express mechanics while call sites retain the
actual proof thresholds and labels.

### Small diagonal and singleton setup

Do not move small one-off setup in:

- `test_qr_diagonal`
- `test_qr_single_row`
- `test_qr_single_col`
- `test_economy_1x1`
- `test_sparse_mode_1x1`
- `test_sparse_mode_diagonal`
- `test_sparse_mode_single_col`
- `test_sparse_mode_single_row`

Rationale: each is short enough that a helper would reduce local readability or
hide the exact edge case.

## Call-Site Readability Rules

Day 6 must preserve visible proof intent:

- rank assertions stay inline;
- reconstruction thresholds stay inline;
- residual thresholds stay inline;
- refinement monotonicity assertions stay inline;
- full/economy/sparse-mode options stay inline;
- dense/sparse solution comparison stays inline.

The helper may hide only the repeated diagonal-dominant sparse insertion
mechanics.

## Placement and Target Rules

- Place the helper in `tests/test_qr.c` near the existing QR fixture builders.
- Do not add a helper header.
- Do not create a new compiled test target.
- Do not change Makefile or CMake membership.
- Do not change CTest registration or reviewed test counts.
- Do not touch public headers or implementation sources.

## Focused Validation Plan

If Day 6 changes `tests/test_qr.c`, run:

```sh
make build/test_qr && ./build/test_qr
make format && make lint && make test
git diff --check
```

Because Day 6 would modify a `.c` test file, the full quality gate is required.

## Day 5 Decision

Proceed to Day 6 with exactly one bounded QR fixture candidate:
`make_qr_tall_diagonal_dominant`. All broader SuiteSparse exact-RHS,
diagonal/singleton, sparse-mode parity, residual, reconstruction, and
refinement proof movement remains deferred.

## Completion Criteria Status

- Remaining QR generated fixture and proof areas were inventoried.
- Safe construction-only movement was separated from inline proof logic.
- One bounded Day 6 candidate was selected.
- Validation commands are known before edits begin.
