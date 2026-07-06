# Day 6 QR Fixture Follow-Through

## Purpose

Day 6 implements the Day 5 QR residual fixture-boundary decision. The change
extracts only repeated tall diagonal-dominant matrix construction from
`tests/test_qr.c` while keeping QR proof assertions visible at the call sites.

## Implemented Helper

Added near the existing local QR fixture builders:

```c
static SparseMatrix *make_qr_tall_diagonal_dominant(idx_t m, idx_t n_cols,
                                                    double diag_value,
                                                    double offdiag_value,
                                                    int include_lower_neighbor);
```

The helper:

- creates an `m x n_cols` sparse matrix;
- inserts diagonal entries in the leading band;
- inserts first-neighbor off-diagonal entries in the leading band;
- uses `qr_insert_or_free` so insertion failures free partial state;
- returns `NULL` on allocation or insertion failure.

## Updated Call Sites

Only the Day 5 approved call sites were updated:

- `test_economy_solve_tall`
- `test_sparse_mode_tall`
- `test_qr_refine_overdetermined`

Each call site still shows:

- dimensions;
- RHS allocation and initialization;
- QR options;
- full/economy/sparse-mode factorization calls;
- solve or refinement calls;
- economy checks;
- residual comparisons;
- dense/sparse parity assertions.

## Explicit Non-Changes

The Day 6 change did not touch:

- SuiteSparse exact-RHS setup;
- `qr_reconstruction_error`;
- `assert_qr_reconstruction_below`;
- `compute_rel_residual`;
- `assert_qr_true_residual_below`;
- diagonal, singleton, single-row, or single-column one-off setup;
- public headers;
- implementation sources;
- Makefile or CMake membership;
- CTest registration.

## Before/After Metrics

| Metric | Before Day 6 | After Day 6 |
|---|---:|---:|
| `tests/test_qr.c` lines | 3,213 | 3,210 |
| Approved call sites with inline tall diagonal-dominant construction loops | 3 | 0 |
| Approved call sites using the local builder | 0 | 3 |
| New compiled helper target | 0 | 0 |
| Public or install-header changes | 0 | 0 |

## Remaining QR Debt

Remaining QR cleanup is deferred:

- SuiteSparse exact-RHS setup needs separate corpus-specific boundary work;
- diagonal and singleton fixtures remain clearer inline;
- sparse-mode parity helpers should not absorb proof assertions;
- refinement behavior should remain visible unless a future boundary approves
  a convergence-specific helper.

## Validation Plan

Because Day 6 changes a `.c` test file, the required validation is:

```sh
make build/test_qr && ./build/test_qr
make format && make lint && make test
git diff --check
```

## Completion Criteria Status

- The Day 5 selected builder was implemented.
- Only approved call sites were updated.
- QR proof assertions remain visible at call sites.
- No helper target, reviewed test-count, public API, or install-header surface
  changed.
