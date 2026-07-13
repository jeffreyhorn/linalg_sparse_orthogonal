# Sprint 121 Day 3: QR and Rank-Deficient Evidence Audit

## Purpose

Inventory the current QR, least-squares, rank-deficient, rectangular,
minimum-norm, nullspace, reconstruction, and refinement proof surface before
Sprint 121 helper extraction or fixture taxonomy work.

## Scope

Inspected surfaces:

- `include/sparse_qr.h`
- `src/sparse_qr.c`
- `tests/test_qr.c`
- `tests/test_qr_solve.c`
- `tests/test_colamd.c` minimum-norm QR sections
- `examples/example_minnorm.c`
- `benchmarks/bench_colamd.c`

There is no separate `src/sparse_qr_solve.c`; QR factorization, solve, rank,
nullspace, minimum-norm, and refinement code live in `src/sparse_qr.c`.

No `.c` or `.h` files were modified by this audit.

## Current Proof-Owner Table

| Capability | Current proof owner | Fixture classes | Current assertions | Gap / Day owner |
|---|---|---|---|---|
| Basic QR factorization | `test_householder_3vec`, `test_householder_identity`, `test_qr_zero_matrix`, `test_qr_1x1`, `test_qr_upper_triangular`, `test_qr_perm_valid` | small dense/sparse, identity, zero, 1x1, triangular | rank, R diagonal, upper-triangular shape, valid permutation | Keep low-level owners separate from solve and least-squares helpers. |
| Reconstruction | `test_qr_reconstruction`, `test_qr_wide`, `test_qr_reconstruction_large`, `test_qr_rank_1`, `test_qr_nearly_singular`, `test_qr_diagonal`, `test_qr_reorder_none`, `test_sparse_mode_reconstruction`, `test_qr_bcsstk04`, `test_qr_tall_synthetic` | square, wide, tall, rank-1, near-singular, diagonal, SuiteSparse, sparse-mode | max reconstruction error for `A*P = Q*R` | Candidate helper already exists in two test files; Day 7 extraction should preserve permutation and Q-shape semantics. |
| Q orthogonality / application | `test_q_roundtrip`, `test_q_orthogonality_tall`, `test_q_transpose_b`, `test_q_apply_multiple`, `test_q_apply_inplace`, `test_q_orthogonality_wide`, `test_economy_q_orthogonality`, `test_sparse_mode_q_ortho` | tall, wide, economy, sparse-mode | round-trip Q/QT, explicit Q orthogonality, in-place behavior | Helper must distinguish full Q, economy Q, and sparse-mode Q formation. |
| QR rank and nullspace | `test_qr_rank_deficient`, `test_qr_rank_1`, `test_qr_nearly_singular`, `test_rank_full`, `test_rank_1_nullspace`, `test_known_nullspace`, `test_rank_rect_deficient`, `test_rank_explicit_tol`, `test_economy_rank_deficient`, `test_sparse_mode_rank_deficient` | duplicate columns, rank-1 outer product, near duplicate, rectangular deficient, explicit tolerance | rank, null dimension, `A*v ~= 0`, monotonic tolerance behavior | Day 4 should separate exact structural rank from numerical threshold rank and nullspace proof classes. |
| Square QR solve | `test_qr_solve_square`, `test_qr_vs_lu`, `test_qr_public_scalar_alias` | square synthetic, SuiteSparse `nos4`, scalar alias | QR-vs-LU solution agreement, residual below `1e-10`/`1e-8`, public scalar alias | Cross-solver agreement is bounded to LU and should not imply broad direct-solver parity. |
| Overdetermined least squares | `test_qr_solve_overdetermined`, `test_qr_solve_analytical`, `test_qr_tall_synthetic`, `test_qr_refine_overdetermined`, `test_minnorm_fallback_overdetermined` | 2x1 analytical, 5x3, 50x20, 20x5, 4x3 fallback | expected analytical solution, residual reporting, relative residual, fallback behavior | Needs helper ownership for normal-equation-free least-squares residual interpretation; Day 9 owner. |
| Rank-deficient solve | `test_qr_solve_rank_deficient`, `test_minnorm_rank_deficient`, `test_minnorm_zero_row` | duplicate column, dependent rows, zero row | effective rank, residual bounded, consistent underdetermined solution | Missing inconsistent rank-deficient least-squares proof with explicit expected residual owner; Day 8-9 candidate. |
| Underdetermined / minimum-norm solve | `test_minnorm_2x4_known`, `test_minnorm_is_minimal`, `test_minnorm_3x6`, `test_minnorm_with_colamd`, `test_minnorm_5x10`, `test_minnorm_square`, `test_minnorm_1xn`, `test_minnorm_vs_pinv`, `test_minnorm_ss_submatrix`, `example_minnorm.c` | 2x4, 3x6, 5x10, 1xn, square fallback, COLAMD, SuiteSparse submatrix | `A*x ~= b`, known minimum norm, smaller norm than alternate solution, QR-vs-SVD-pinv agreement | Tests live in `test_colamd.c`; Day 7-9 should consider moving/owning minimum-norm helpers under QR evidence without disrupting existing test membership. |
| Iterative refinement | `test_qr_refine_well_conditioned`, `test_qr_refine_ill_conditioned`, `test_qr_refine_zero_iter`, `test_qr_refine_nos4`, `test_qr_refine_overdetermined`, `test_refine_minnorm`, `test_refine_minnorm_null` | well-conditioned, ill-conditioned, zero-iter, SuiteSparse, overdetermined, minimum-norm | residual non-increase, exact residual recomputation, null handling | Helper should keep absolute residual vs relative residual semantics visible. |
| Reordering / fill | `test_qr_reorder_amd_solve`, `test_qr_reorder_nos4_fillin`, `test_qr_reorder_none`, `bench_colamd.c`, `example_colamd.c` | AMD, COLAMD, none, SuiteSparse | solve residual, fill comparison, reconstruction | Reordering evidence is QR-adjacent but not the main Day 8-9 numerical oracle target. |
| Economy and sparse-mode QR | `test_economy_solve_tall`, `test_economy_q_orthogonality`, `test_economy_square`, `test_economy_r_shape`, `test_economy_rank_deficient`, `test_economy_wide`, `test_economy_1x1`, `test_economy_nos4`, `test_sparse_mode_basic`, `test_sparse_mode_nos4`, `test_sparse_mode_tall`, `test_sparse_mode_wide`, `test_sparse_mode_west0067`, `test_sparse_mode_bcsstk04`, `test_sparse_mode_diagonal`, `test_sparse_mode_single_col`, `test_sparse_mode_single_row`, `test_sparse_mode_timing` | tall, square, wide, 1x1, SuiteSparse, sparse/dense backend comparison | solve agreement, rank agreement, Q orthogonality, shape, reconstruction, timing smoke | Day 7 helper extraction must not hide backend/mode-specific storage and shape expectations. |

## Current Fixture Classes

- Exact square systems: small synthetic 3x3, diagonal, identity, 1x1, and
  `nos4` QR-vs-LU comparisons.
- Overdetermined least-squares systems: analytical 2x1, small 5x3, tall
  diagonal-dominant, 50x20 synthetic, and SuiteSparse square/tall-like smoke.
- Underdetermined minimum-norm systems: 2x4 known, 3x6 example, 5x10,
  1x5, COLAMD-backed 2x5, west0067 30x67 submatrix.
- Rank-deficient systems: duplicate columns, rank-1 outer product,
  near-duplicate columns, dependent rows, zero-row consistent system,
  rectangular 3x5 rank-deficient nullspace.
- Reconstruction fixtures: square, wide, tall, large synthetic, diagonal,
  rank-1, near singular, SuiteSparse `bcsstk04`, sparse-mode `nos4`.
- Q-application fixtures: tall, wide, economy, sparse-mode, in-place and
  multi-vector application.
- Reordering fixtures: AMD, COLAMD, none, SuiteSparse fill comparison.

## Tolerance and Failure-Mode Map

| Area | Current tolerance pattern | Failure behavior | Notes |
|---|---|---|---|
| QR reconstruction | mostly `1e-10`; `1e-8` for near singular; `1e-6` for `bcsstk04` | direct assertion failures | Reconstruction helper must apply `A*P = Q*R`, not unpermuted `A = Q*R`. |
| Q orthogonality / roundtrip | usually `1e-10` to `1e-12` | direct assertion failures | Economy Q has m-by-k shape; full Q has m-by-m shape. |
| Square solve residual | `1e-10` synthetic, `1e-8` `nos4`, `1e-4` solution diff on QR-vs-LU | direct assertion failures | Cross-solver proof is bounded to LU and selected fixtures. |
| Overdetermined residual | analytical exact value, broad relative residual bound such as `< 1.0`, or generated-RHS exact residual `1e-8` | direct assertion failures | Needs clearer distinction between exact generated-RHS and true least-squares residual floors. |
| Rank / nullspace | `sparse_qr_rank(..., 0.0)` default threshold, explicit tolerance monotonic checks, nullspace `A*v < 1e-10` | direct assertion failures | `qr->rank` and post-factor `sparse_qr_rank()` may use different thresholds by API design. |
| Minimum-norm | known solution `1e-12`, residual `1e-10`, rank-deficient `1e-8`, norm comparison against alternate solution | direct assertion failures or skip for SuiteSparse solve failure | Tests are housed under `test_colamd.c`, which makes ownership less obvious. |
| Refinement | residual non-increase with `1e-12` to `1e-15` slack; `nos4` absolute residual `< 1e-10` | direct assertion failures | Must preserve absolute residual vs relative residual distinction. |
| Sparse-mode / economy backend comparison | solution max-diff and rank equality, mode-specific orthogonality/reconstruction | direct assertion failures | Helper extraction should keep dense-mode and sparse-mode labels in assertion messages. |
| API errors / invalid reuse | `SPARSE_ERR_NULL` and `SPARSE_ERR_BADARG` for nulls and factored-matrix reuse | direct assertion failures | Keep contract tests separate from numerical proof helpers. |

## Rank-Deficient and Rectangular Gap Inventory

- Underdetermined minimum-norm proof is strong but lives in `tests/test_colamd.c`;
  Sprint 121 should either document that ownership explicitly or extract shared
  QR/minimum-norm helpers without changing reviewed test membership casually.
- Rank-deficient least-squares currently proves bounded residual on duplicate
  columns and consistent minimum-norm cases, but lacks a clearly named
  inconsistent rank-deficient fixture with an expected least-squares residual.
- Overdetermined least-squares mixes exact generated-RHS cases with true
  least-squares cases. Day 4 taxonomy should distinguish "compatible tall"
  from "incompatible tall" fixtures.
- QR-vs-LU and QR-vs-SVD-pseudoinverse comparisons are useful bounded
  cross-solver checks, but they are not external dense-library parity.
- Near-rank-deficient behavior has rank and reconstruction coverage, but does
  not yet have a single fixture family that pins rank threshold, residual, and
  nullspace behavior together.
- Examples and benchmarks demonstrate adoption and fill/timing behavior; they
  are not proof owners for numerical residual or minimum-norm claims.

## Helper Extraction Candidates

- `qr_reconstruction_error` and `qr_solve_reconstruction_error` should be
  consolidated or wrapped behind one QR reconstruction helper that names
  permutation semantics and Q shape explicitly.
- `compute_rel_residual` and `qr_solve_rel_residual` should become shared
  residual helpers only if callers still choose absolute vs relative residual
  interpretation at the test boundary.
- `make_qr_exact_rhs` and `make_qr_solve_exact_rhs` are duplicated generated-RHS
  helpers and should be candidates for Day 7 extraction.
- Duplicate-column fixture builders in `tests/test_qr.c` and
  `tests/test_qr_solve.c` should be unified or taxonomy-owned to prevent drift
  in rank-deficient expectations.
- Minimum-norm residual/norm helpers in `tests/test_colamd.c` can become
  QR-owned helper code if Day 9 expands underdetermined evidence.
- Economy and sparse-mode backend comparisons should keep separate helper
  labels for mode, shape, rank, and solution-diff expectations.

## Day 4 Matrix Taxonomy Inputs

- `qr-square-exact`: square full-rank direct solve and reconstruction fixtures.
- `qr-overdetermined-compatible`: tall generated-RHS fixtures with near-zero
  residual expected.
- `qr-overdetermined-incompatible`: analytical and noisy least-squares fixtures
  with nonzero expected residual.
- `qr-underdetermined-minnorm`: m < n minimum-norm fixtures with norm ownership.
- `qr-rank-def-duplicate-column`: duplicate-column structural deficiency.
- `qr-rank-def-dependent-row`: dependent-row / zero-row consistent systems.
- `qr-near-rank-def-threshold`: near-duplicate and explicit tolerance fixtures.
- `qr-nullspace-owned`: nullspace basis and `A*v ~= 0` fixtures.
- `qr-economy-mode`: thin-Q shape, solve, rank, and orthogonality fixtures.
- `qr-sparse-mode`: dense-vs-sparse backend agreement fixtures.
- `qr-reordered`: AMD/COLAMD/none reorder fixtures.
- `qr-suite-sparse-smoke`: `nos4`, `west0067`, `bcsstk04`, and submatrix
  bounded evidence.

## Validation Notes

This was a documentation-only audit. Required validation is limited to
`git diff --check` and a focused trailing-whitespace scan over
`docs/planning/EPIC_11/SPRINT_121`.

## Completion Criteria Status

- Item 1 QR audit inputs are complete for QR factorization, solve, rank,
  nullspace, minimum-norm, refinement, example, and benchmark surfaces.
- Every identified QR/rank-deficient proof gap has a candidate owner or
  explicit deferral reason in the proof-owner table and gap inventory.
- Candidate helpers preserve rank, residual, reconstruction, Q-shape,
  least-squares, minimum-norm, and backend/mode semantics at visible test
  boundaries.
