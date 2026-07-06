# Day 10 SVD Oracle and Reconstruction Cleanup

## Purpose

Day 10 applies the Day 9 SVD validation lane by extracting one setup-only
fixture family from `tests/test_svd.c`. The change must not hide SVD oracle,
rank, reconstruction, orthogonality, pseudoinverse, low-rank, partial-SVD, or
condition-number proof logic.

## Implemented Helper

Added one local helper near the existing SVD fixture builders:

```c
static SparseMatrix *make_svd_full_uv_fixture_16x8(void);
```

The helper owns only the repeated deterministic dense fixture:

- creates a `16 x 8` sparse matrix;
- fills every `(i, j)` with the existing deterministic expression;
- checks every insert through `svd_insert_or_free`;
- frees partial matrix state and returns `NULL` on insert failure.

## Updated Call Sites

Only the approved Day 10 call sites were changed:

- `test_svd_full_u_v_orthonormality`
- `test_svd_full_u_v_economy_mode_unchanged`
- `test_svd_full_u_v_reconstruction`

All proof logic remains visible at the call sites:

- `m = 16` and `n_cols = 8`;
- full/economy `sparse_svd_opts_t` literals;
- `sparse_svd_compute` calls;
- `svd.m`, `svd.n`, `svd.k`, `svd.U`, and `svd.Vt` assertions;
- U orthogonality loop;
- Vt orthogonality loop;
- economy/full singular-triplet parity loops;
- full-mode reconstruction loop;
- residual thresholds and diagnostic logging.

## Explicitly Preserved Surfaces

No changes were made to:

- rank-threshold assertions;
- singular-value oracle checks;
- QR rank comparisons;
- pseudoinverse Moore-Penrose checks;
- dense or sparse low-rank comparisons;
- drop-tolerance assertions;
- partial-SVD vector, corpus, timing, or full-SVD comparison tests;
- condition-number behavior;
- public headers;
- implementation sources;
- Makefile or CMake membership;
- CTest registration or reviewed test counts.

## Metrics

| Metric | Before Day 10 | After Day 10 |
|---|---:|---:|
| `tests/test_svd.c` lines | 2,897 | 2,896 |
| Local deterministic full-SVD `16x8` fixture builder | no | yes |
| Approved call sites using shared fixture builder | 0 | 3 |
| New compiled helper target | 0 | 0 |
| Public headers touched | 0 | 0 |

## Residual SVD Proof-Owner Debt

Remaining SVD cleanup should stay boundary-first:

- rank and singular-value oracle checks should remain visible unless a future
  boundary selects a narrow assertion helper.
- reconstruction and orthogonality loops should stay inline because they prove
  storage layout and leading-dimension behavior.
- pseudoinverse and low-rank lanes need separate boundaries because they prove
  different API behavior than full SVD.
- partial-SVD vector and corpus checks should not be mixed with full-SVD
  fixture cleanup.
- condition-number and rank-threshold behavior should remain colocated with
  the specific matrices and tolerances being tested.

## Completion Criteria Status

- The approved setup-only helper was implemented locally.
- Only approved call sites were updated.
- SVD proof assertions remain visible.
- No helper target, public API, build membership, or CTest surface changed.
- Focused and full quality validation are required because `tests/test_svd.c`
  changed.
