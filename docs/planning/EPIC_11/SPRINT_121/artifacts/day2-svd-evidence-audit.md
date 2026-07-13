# Sprint 121 Day 2: SVD Evidence Audit

## Purpose

Inventory the current SVD, partial-SVD, rank, pseudoinverse, low-rank, and
condition-number proof surface before helper extraction or fixture expansion.
This audit identifies proof owners, tolerance semantics, dense-reference
assumptions, repeated helper logic, and Day 4 matrix-taxonomy inputs.

## Scope

Inspected surfaces:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `src/sparse_svd_partial.c`
- `src/sparse_bidiag.c`
- `tests/test_svd.c`
- `tests/test_svd_partial_helpers.h`
- `examples/example_svd_lowrank.c`
- `benchmarks/bench_svd.c`

No `.c` or `.h` files were modified by this audit.

## Current Proof-Owner Table

| Capability | Current proof owner | Fixture classes | Current assertions | Gap / Day owner |
|---|---|---|---|---|
| Golub-Kahan extraction | `test_gk_extract_3x3`, `test_gk_extract_tall`, `test_gk_extract_wide`, `test_gk_square_5x5`, `test_gk_tall_ortho`, `test_gk_wide_ortho`, `test_gk_rank_deficient`, `test_gk_nos4`, `test_gk_west0067`, `test_gk_1x1` | small explicit, rectangular, rank deficient, SuiteSparse | reconstruction for non-transposed path, U/V orthogonality, basic shape behavior | Extract reconstruction and orthogonality helpers without hiding tall/wide interpretation on Days 5-6. |
| Bidiagonal SVD iteration | `test_bidiag_svd_diagonal`, `test_bidiag_svd_2x2`, `test_bidiag_svd_3x3_uv`, `test_bidiag_svd_k1`, `test_bidiag_svd_zero_super` | diagonal, 2x2/3x3, zero superdiagonal | known singular values and vector availability | Keep as low-level owners; do not merge into high-level SVD helpers. |
| Full SVD singular values | `test_svd_basic_sigma`, `test_svd_diagonal_5x5`, `test_svd_trace_invariant`, `test_svd_rank1`, `test_svd_rank2`, `test_svd_rank5_in_10x10`, `test_svd_rank1_square`, `test_svd_rank1_wide`, `test_svd_near_singular`, `test_svd_multi_zero_diag`, `test_svd_rank2_dense`, `test_svd_descending`, `test_svd_all_zero`, `test_svd_repeated`, `test_svd_diag_20x20`, `test_svd_tall_10x5`, `test_svd_wide_5x10`, `test_svd_nos4`, `test_svd_west0067` | diagonal, rank-1/2/5, near singular, zero, repeated, rectangular, SuiteSparse | singular-value values, descending order, non-negativity, trace/Frobenius invariants | Needs taxonomy labels and reusable singular-value expectation helpers on Days 4-6. |
| Full SVD vectors and reconstruction | `test_svd_with_uv`, `test_svd_rank1_uv`, `test_svd_rank2_dense`, `test_svd_wide_5x10_uv`, `test_svd_sigma_only_vs_uv`, `test_svd_full_u_v_orthonormality`, `test_svd_full_u_v_reconstruction`, `test_svd_full_u_v_economy_mode_unchanged`, `test_s103_svd_diag6_rank_threshold_claim` | small dense, rank deficient, wide rectangular, full vs economy, rank-threshold diagonal | reconstruction residual, U/Vt orthogonality, sigma consistency, economy/full shared-triplet stability | Helper extraction should preserve leading-dimension semantics for economy vs full mode on Days 5-6. |
| Partial SVD singular values | `test_partial_svd_diag_10x10`, `test_partial_svd_full_k`, `test_partial_svd_dense_8x8`, `test_partial_svd_tall`, `test_partial_svd_wide`, `test_partial_svd_nos4`, `test_partial_svd_k1`, `test_partial_svd_rank_deficient`, `test_partial_svd_west0067`, `test_partial_svd_descending`, `test_partial_svd_timing`, `test_partial_svd_nonsymmetric` | diagonal, Hilbert-like dense, rectangular, rank deficient, SuiteSparse, nonsymmetric | compare top-k sigma to full SVD, order, shape, basic timing smoke | Current oracle is mostly library full SVD, not external dense truth; Day 10 should add bounded partial-SVD evidence with explicit non-claim framing. |
| Partial SVD vectors | `test_partial_svd_vectors_ortho`, `test_partial_svd_vectors_Av`, `test_partial_svd_vectors_vs_full`, `test_partial_svd_vectors_nos4`, `test_partial_svd_vectors_west0067`, `test_partial_svd_vectors_recon`, `test_partial_svd_vectors_k1`, `test_partial_svd_vectors_wide`, `test_partial_svd_no_vectors` | small synthetic, SuiteSparse, wide | orthogonality, `A*v ~= sigma*u`, comparison to full SVD, reconstruction, no-vector mode | Candidate helper family for partial-vector residuals; preserve looser tolerances than full SVD. |
| Rank estimation | `test_svd_rank_full`, `test_svd_rank_deficient`, `test_svd_rank_nearly_singular`, `test_svd_suitesparse_rank_deficient`, `test_svd_rank_vs_qr`, `test_s103_svd_diag6_rank_threshold_claim` | full rank, duplicate columns, near singular, SuiteSparse truncated columns, QR comparison | exact rank under default/explicit tolerances, SVD-vs-QR agreement | Day 4 taxonomy should split exact rank, numerical rank threshold, and cross-solver agreement classes. |
| Pseudoinverse | `test_pinv_diagonal`, `test_pinv_moore_penrose`, `test_pinv_rectangular`, `test_pinv_null` | diagonal, tall rectangular | diagonal entries and first Moore-Penrose condition `A A+ A ~= A` | Missing other Moore-Penrose conditions, wide/rank-deficient fixtures, and tolerance-threshold pinning; Days 8-9 should own expansion. |
| Dense low-rank | `test_lowrank_diagonal`, `test_lowrank_error_bound`, `test_lowrank_errors` | diagonal, tridiagonal | exact diagonal truncation, Frobenius error equals tail singular values, bad args | Needs rectangular and rank-deficient low-rank fixtures with helper-owned residual interpretation; Day 10 owner. |
| Sparse low-rank | `test_lowrank_sparse_diagonal`, `test_lowrank_sparse_sparsity`, `test_lowrank_sparse_vs_dense`, `test_lowrank_sparse_rank1`, `test_lowrank_sparse_rectangular`, `test_lowrank_sparse_errors`, `test_sparse_svd_lowrank_outer_product_matches_dense`, `test_sparse_svd_lowrank_outer_product_corpus_safety` | diagonal, tridiagonal, rank-1 dense, rectangular, SuiteSparse corpus | sparse dimensions, drop tolerance behavior, dense-vs-sparse residual, env-on/off equivalence | Good accumulator proof exists; Day 10 should add clearer fixture taxonomy and avoid claiming broad large-matrix performance. |
| Condition number | `test_cond_identity`, `test_cond_diagonal`, `test_cond_singular`, `test_cond_1x1`, `test_cond_1x1_zero`, `test_cond_rectangular`, `test_cond_ill_conditioned`, `test_cond_null` | identity, diagonal, singular, rectangular, ill-conditioned | finite/infinite condition behavior and error handling | Keep as SVD application owner; do not fold into rank helpers unless tolerance semantics remain explicit. |

## Current Fixture Classes

- Exact diagonal spectra: full-rank, repeated singular values, thresholded
  rank, zero singular values, condition-number diagonals.
- Low-rank synthetic matrices: row-progression rank-1, rank-2 dense,
  rank-5-in-10, duplicate-column rank deficient, rank-1 square/wide.
- Rectangular shape coverage: tall full SVD, wide full SVD, tall/wide partial
  SVD, rectangular low-rank sparse output, rectangular pseudoinverse.
- Bidiagonal and extraction fixtures: 1x1, 2x2, 3x3, tall, wide, zero
  superdiagonal, rank deficient.
- SuiteSparse fixtures: `nos4`, `west0067`, `bcsstk04` in sparse low-rank
  corpus safety.
- Dense Hilbert-like and tridiagonal fixtures: partial-SVD top-k comparison,
  low-rank error-bound checks, sparse-vs-dense low-rank comparison.

## Tolerance and Failure-Mode Map

| Area | Current tolerance pattern | Failure behavior | Notes |
|---|---|---|---|
| Full SVD exact/diagonal | `1e-10` common, `1e-12` for small near-threshold value | direct assertion failures | Good for deterministic exact spectra; helper should accept per-fixture tolerance. |
| Full reconstruction | relative Frobenius or max-abs residual usually below `1e-10` | direct assertion failures | Leading dimension differs between economy and full Vt; helper must make stride explicit. |
| Full orthogonality | Frobenius orthogonality usually below `1e-10` | direct assertion failures | U and Vt checks have different storage shape semantics. |
| Partial SVD sigma | exact diagonals `1e-10`; dense/SuiteSparse relative windows of `5%` or `10%` | direct assertion failures | Full SVD is the internal oracle; not an external dense library oracle. |
| Partial SVD vectors | approximate orthogonality and residuals around `1e-6` or fixture-specific residual windows | direct assertion failures or skip on load/setenv limits | Must not reuse full-SVD `1e-10` helper thresholds. |
| Rank | default tolerance `eps * max(m,n) * sigma_max`; explicit tolerances such as `1e-12`, `1e-10`, `1e-8` | direct assertion failures | Needs taxonomy distinction between exact algebraic rank and numerical rank. |
| Pseudoinverse | diagonal entries and `A A+ A` residual below `1e-10` | direct assertion failures | Other Moore-Penrose identities are not yet covered. |
| Dense low-rank | exact diagonal entries `1e-10`; Frobenius tail error `1e-8` to `1e-10` | bad-arg tests expect `SPARSE_ERR_BADARG` / `SPARSE_ERR_NULL` | Error-bound helper can be extracted if rank_k and tail interpretation stay visible. |
| Sparse low-rank | drop-tolerance residual relative to dense output; env-on/off residual below `1e-10` | setenv/load failures are skipped or reported; API errors asserted | Env-var path proof is bounded to selected fixtures, not a performance guarantee. |
| Reuse/factored matrices | SVD and partial SVD reject non-identity permutation/factored matrices | `SPARSE_ERR_BADARG` | Keep as API contract owner, separate from numerical proof helpers. |

## Low-Rank and Pseudoinverse Gap Inventory

- Pseudoinverse currently proves diagonal inversion and the first
  Moore-Penrose condition on tall/full-column-rank style fixtures. It does not
  yet own `A+ A A+ ~= A+`, symmetry of `A A+`, symmetry of `A+ A`, wide
  matrices, or rank-deficient pseudoinverse tolerance behavior.
- Low-rank dense output proves exact diagonal truncation and a tridiagonal
  Frobenius tail bound. It does not yet provide a shared rectangular
  rank-deficient dense low-rank proof owner.
- Sparse low-rank has stronger env-on/off equivalence evidence than dense
  low-rank has taxonomy naming. Day 10 should connect sparse low-rank fixtures
  to matrix taxonomy keys instead of only historical sprint comments.
- Partial SVD top-k uses full SVD as the primary reference. This is useful for
  regression, but it is not independent dense-library parity.
- Examples and benchmarks demonstrate SVD/low-rank workflows and timings but
  are not proof owners for reconstruction, pseudoinverse identities, or
  partial-SVD vector residuals.

## Helper Extraction Candidates

- `svd_reconstruction_max_error` and `svd_reconstruction_rel_frobenius` can
  become reusable full-SVD reconstruction helpers if they keep explicit
  economy/full leading-dimension inputs.
- `orthogonality_error` and `svd_vt_row_orthogonality_error` can become
  helper-owned U/Vt checks, but the helper API should name matrix orientation
  and storage stride.
- `svd_pinv_first_moore_penrose_error` should remain pseudoinverse-specific
  and expand toward additional Moore-Penrose identities instead of becoming a
  generic dense-multiply helper first.
- `svd_dense_lowrank_frobenius_error`,
  `svd_sparse_dense_frobenius_diff`, and
  `svd_sparse_sparse_rel_frobenius_diff` are good low-rank helper candidates,
  but dense-vs-sparse and sparse-vs-sparse residual semantics should remain
  separate.
- Partial-SVD vector residual logic in `tests/test_svd_partial_helpers.h`
  should become partial-specific helper code with looser tolerance defaults and
  top-k triplet terminology.
- Fixture builders such as `make_svd_rank1_row_progression`,
  `make_svd_rank_deficient_colpair_5x4`, diagonal builders, and full-UV
  fixtures are candidates for taxonomy-owned fixture helpers on Days 4-6.

## Day 4 Matrix Taxonomy Inputs

- `svd-diag-exact`: exact separated diagonal spectra with known sigma order.
- `svd-diag-threshold`: diagonal spectra with values near rank thresholds.
- `svd-rank-def-duplicate-columns`: duplicate-column rank-deficient matrices.
- `svd-lowrank-outer-product`: synthetic rank-1/rank-k outer-product fixtures.
- `svd-rectangular-tall` and `svd-rectangular-wide`: shape-specific SVD,
  partial-SVD, pseudoinverse, and low-rank fixtures.
- `svd-suite-sparse-smoke`: `nos4`, `west0067`, and bounded low-rank corpus
  fixtures.
- `svd-partial-internal-reference`: partial-SVD comparisons against this
  library's full SVD, explicitly not external parity.
- `svd-pinv-moore-penrose`: pseudoinverse identity fixtures, currently only
  first identity.
- `svd-lowrank-drop-tolerance`: dense/sparse low-rank drop and residual
  fixtures.
- `svd-condition-number`: finite/infinite condition-number behavior fixtures.

## Validation Notes

This was a documentation-only audit. Required validation is limited to
`git diff --check` and a focused trailing-whitespace scan over
`docs/planning/EPIC_11/SPRINT_121`.

## Completion Criteria Status

- Item 1 SVD audit inputs are complete for the current SVD, partial-SVD,
  rank, pseudoinverse, low-rank, condition-number, example, and benchmark
  surfaces.
- Every identified SVD proof gap has a candidate owner or explicit deferral
  reason in the table and gap inventory.
- Helper opportunities preserve solver-specific tolerance interpretation by
  keeping full SVD, partial SVD, pseudoinverse, dense low-rank, sparse
  low-rank, rank, and condition-number semantics separate.
