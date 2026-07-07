# Sprint 110 Day 11: SVD Proof-Loop Boundary Review

## Purpose

Day 11 reviewed `tests/test_svd.c` for one safe Day 12 setup-helper family.
The review explicitly avoids repeating earlier SVD cleanup work from Sprint
107, Sprint 108, Sprint 103, and Sprint 109. No SVD code moved on Day 11.

## Existing Completed SVD Cleanup Excluded

The following work is already complete and is not Sprint 110 Day 12 scope:

- `make_svd_diag_matrix`, which already owns repeated diagonal fixture setup;
- `make_svd_rank1_row_progression`, which already owns repeated rank-1 row
  progression fixture setup;
- `make_svd_full_uv_fixture_16x8`, which already owns the Sprint 108 full-UV
  fixture;
- Sprint 103's claim-owned diagonal rank/full-UV evidence fixture;
- existing reconstruction and orthogonality helpers:
  - `gk_reconstruction_error`;
  - `orthogonality_error`;
  - partial-SVD helper routines in `tests/test_svd_partial_helpers.h`.

## Proof-Loop Map

| Area | Representative Tests | Proof Values That Must Stay Visible |
|---|---|---|
| Golub-Kahan extraction | `test_gk_extract_*`, `validate_gk` users | reconstruction residuals, U/V orthogonality thresholds, transposed-wide skip rationale |
| Full SVD singular values | diagonal, trace, rank-1/rank-2/rank-5, near-singular, repeated, SuiteSparse smoke tests | expected singular values, descending checks, tolerance thresholds, rank-sensitive near-zero claims |
| Full/economy UV behavior | `test_svd_with_uv`, `test_svd_wide_5x10_uv`, full-UV Sprint 29/Sprint 103 tests | `svd.m`, `svd.n`, `svd.k`, `svd.U`, `svd.Vt`, U/Vt layout loops, reconstruction residuals |
| Rank comparison | `test_svd_rank_vs_qr`, `test_svd_rank_full`, `test_svd_rank_deficient`, `test_svd_rank_nearly_singular` | rank thresholds, QR comparison, explicit tolerance interpretation |
| Pseudoinverse | `test_pinv_*` | expected inverse entries, Moore-Penrose products, dense intermediate dimensions |
| Dense low-rank | `test_lowrank_*` | retained singular-value error bound, Frobenius residual, rank-k values |
| Sparse low-rank | `test_lowrank_sparse_*`, outer-product corpus tests | dense-vs-sparse residual, drop tolerance, env-on/off comparison, corpus fixture names |
| Partial SVD | `test_partial_svd_*` and vector checks in `tests/test_svd_partial_helpers.h` | full-vs-partial singular values, vector orthogonality, `A*v ~= sigma*u`, corpus residuals |
| Condition number | `test_cond_*` | expected finite/infinite condition values and rectangular interpretation |

## Unsafe Helper Families

These families should not move in Sprint 110 Day 12 because they would hide
the proof logic that makes failures interpretable:

- reconstruction loops for full/economy SVD;
- U/Vt dot-product orthogonality loops;
- Moore-Penrose product loops;
- dense low-rank Frobenius error computations;
- sparse low-rank dense-vs-sparse residual comparisons;
- partial-SVD vector residual and orthogonality checks;
- condition-number expected-value assertions.

## Selected Day 12 Candidate

Select exactly one safe setup-helper family:

```c
static SparseMatrix *make_svd_rank_deficient_colpair_5x4(void);
```

The helper may build only the repeated 5x4 rank-deficient fixture where
column 1 duplicates column 0 and column 3 duplicates column 2:

- column 0: `i + 1`;
- column 1: `i + 1`;
- column 2: `2*i + 1`;
- column 3: `2*i + 1`.

Allowed call sites:

- `test_svd_rank_vs_qr`;
- `test_svd_rank_deficient`.

This is safe because the helper hides only duplicate matrix construction. It
does not compute rank, choose tolerances, run QR, compare rank values, or
perform any reconstruction/orthogonality/math proof.

## Proof Visibility Requirements For Day 12

Day 12 must keep the following visible at call sites:

- `sparse_svd_compute` and `sparse_svd_rank` calls;
- `tol_svd = 1e-8 * svd.sigma[0]`;
- the manual singular-value rank-count loop;
- the QR factorization and `qr.rank` comparison;
- expected rank value `2`;
- printed rank labels;
- all cleanup ownership.

## Explicit Non-Changes

Day 12 must not:

- change public SVD headers or option defaults;
- change `tests/test_svd_partial_helpers.h`;
- add a helper target or shared test helper header;
- move reconstruction, orthogonality, pseudoinverse, low-rank, partial-SVD, or
  condition-number proof loops;
- alter `RUN_TEST` registration or reviewed CTest counts.

## Focused Validation Checklist

If Day 12 modifies `tests/test_svd.c`, required validation is:

```sh
make build/test_svd
build/test_svd
make format && make lint && make test
git diff --check
```

Also run a trailing-whitespace scan over touched Sprint 110 docs and
`tests/test_svd.c`.

## Residual Deferrals

The following remain future work unless a later boundary artifact narrows them
further:

- reconstruction helper movement;
- U/Vt orthogonality helper movement;
- Moore-Penrose helper extraction;
- dense and sparse low-rank proof-loop cleanup;
- partial-SVD vector/residual cleanup;
- condition-number proof cleanup.
