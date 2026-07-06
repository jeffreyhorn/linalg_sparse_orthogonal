# Day 9 SVD Validation Lane Boundary

## Purpose

Day 9 creates the required validation lane for any remaining
`tests/test_svd.c` proof-owner movement. SVD tests combine fixture setup,
rank/oracle behavior, reconstruction proofs, low-rank guarantees, and
condition-sensitive checks, so any helper movement must be limited to setup
that does not hide the assertion surface.

## Live Inventory

| Area | Current State | Day 9 Disposition |
|---|---|---|
| Diagonal fixtures | `make_svd_diag_matrix` already covers repeated diagonal matrix setup. | Exclude Sprint 107 completed work. |
| Rank-1 row-progression fixtures | `make_svd_rank1_row_progression` already covers repeated rank-1 dense setup. | Exclude Sprint 107 completed work. |
| Rank and singular-value checks | Exact sigma values, rank thresholds, descending order, and QR rank comparisons are asserted inline. | Keep inline. |
| Reconstruction proofs | Wide/economy/full SVD reconstruction loops directly encode storage-layout and residual expectations. | Keep proof loops inline. |
| Orthogonality proofs | U and Vt dot-product loops validate output layout and full/economy semantics. | Keep proof loops inline. |
| Pseudoinverse proofs | Moore-Penrose `A * A^+ * A ~= A` loops encode correctness directly. | Defer. |
| Low-rank proofs | Dense/sparse low-rank Frobenius comparisons and drop-tolerance checks are behavior assertions. | Defer. |
| Partial-SVD proofs | Vector orthogonality, Av checks, full-SVD comparisons, and corpus tests form a distinct validation surface. | Defer. |
| Condition-number behavior | Condition estimates and rank-sensitive thresholds are proof logic. | Defer. |
| Full-SVD deterministic `16x8` fixture | Three full-mode tests repeat the same deterministic dense fixture before distinct assertions. | Select setup-only helper. |

## Selected Day 10 Candidate

Add one local helper near the existing SVD fixture builders:

```c
static SparseMatrix *make_svd_full_uv_fixture_16x8(void);
```

Expected construction:

- create a `16 x 8` sparse matrix;
- fill every `(i, j)` with the existing deterministic expression used by the
  full-SVD tests;
- check each insert through the existing `svd_insert_or_free` helper;
- return `NULL` on allocation or insert failure.

This helper may hide only deterministic matrix construction.

## Approved Day 10 Call Sites

Only these call sites are approved for Day 10 updates:

- `test_svd_full_u_v_orthonormality`
- `test_svd_full_u_v_economy_mode_unchanged`
- `test_svd_full_u_v_reconstruction`

The following must remain visible at the call sites:

- `m = 16` and `n_cols = 8` dimensions;
- `sparse_svd_opts_t` values for economy and full modes;
- `sparse_svd_compute` calls;
- `svd.m`, `svd.n`, `svd.k`, `svd.U`, and `svd.Vt` assertions;
- singular-value and singular-vector parity loops;
- U orthogonality loop;
- Vt orthogonality loop;
- reconstruction loop;
- residual thresholds;
- diagnostic logging.

## Explicit Non-Candidates

### Rank and sigma proof logic

Do not move singular-value assertions, descending-order checks, QR rank
comparisons, or rank-threshold calls behind helpers. These are the main SVD
behavioral evidence.

### Reconstruction and orthogonality assertions

Do not move reconstruction loops or dot-product orthogonality loops into a
shared assertion helper during Day 10. These loops make the storage layout,
leading dimension, and economy/full semantics reviewable.

### Pseudoinverse and low-rank lanes

Do not move Moore-Penrose, dense low-rank, sparse low-rank, drop-tolerance, or
Frobenius-error logic. Those lanes need their own dedicated boundary because
the assertions encode separate API behavior.

### Partial-SVD lanes

Do not move partial-SVD vector, corpus, timing, or full-SVD comparison logic.
Partial SVD has a different validation profile and should not be mixed with
full-mode fixture cleanup.

## Validation Lane

If Day 10 changes `tests/test_svd.c`, run:

```sh
make build/test_svd && ./build/test_svd
make format && make lint && make test
git diff --check
```

Because Day 10 would modify a `.c` test file, the full quality gate is
required.

## Metrics

| Metric | Current Value |
|---|---:|
| `tests/test_svd.c` lines | 2,897 |
| Static functions | 81 |
| Registered tests | 98 |
| Assertion/proof macro references | 461 |
| Sparse creates | 55 |
| Sparse inserts | 160 |
| Approved Day 10 full-SVD fixture call sites | 3 |
| New helper target approved | 0 |

## Day 9 Decision

Proceed to Day 10 with exactly one setup-only SVD fixture candidate:
`make_svd_full_uv_fixture_16x8`. All rank, oracle, reconstruction,
pseudoinverse, low-rank, partial-SVD, and condition-number proof logic remains
visible at call sites or deferred to a future dedicated boundary.

## Completion Criteria Status

- Remaining SVD proof surfaces were inventoried.
- SVD proof movement now has a focused validation lane.
- One setup-only helper family was selected.
- Unsafe helper candidates were explicitly deferred.
