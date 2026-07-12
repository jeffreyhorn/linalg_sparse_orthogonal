# Sprint 120 Day 2 Direct Oracle Ownership Audit

## Purpose

Inventory the direct-solver generated-RHS, dense-reference, residual,
lifecycle, and cross-backend proof owners before any direct proof-owner split
is selected. This artifact covers `tests/test_qr.c`, `tests/test_ldlt.c`,
`tests/test_ldlt_csc.c`, and existing shared helper headers used by direct
solver proof lanes.

## File Size And Hotspot Baseline

| File | Lines | Day 2 Classification |
|---|---:|---|
| `tests/test_qr.c` | 3234 | Giant direct test with QR factor, solve, rank, reorder, economy, sparse-mode, and refinement proof owners. |
| `tests/test_ldlt.c` | 3006 | Giant direct test with linked-list LDLT, KKT, reorder, refinement, condition, backend-dispatch, and dense-backend proof owners. |
| `tests/test_ldlt_csc.c` | 3915 | Highest-risk giant direct test with CSC allocation, analysis-aware factor, external dense-reference, native/wrapper, solve, inertia, and singular-detection owners. |
| `tests/test_direct_solver_helpers.h` | 93 | Narrow direct helper layer for sparse-matrix equality, LU CSR factorization checks, and infinity-norm residuals. |
| `tests/test_solver_helpers.h` | 200 | Shared residual and external-reference helper layer used across solver/integration tests. |

## Direct Oracle Owner Table

| Area | Current Owner | Proof Role | Split Risk |
|---|---|---|---|
| QR exact RHS generation | `make_qr_exact_rhs` in `tests/test_qr.c` | Builds `x_exact` and `b = A*x_exact` for QR solve and SuiteSparse fixtures. | Medium; useful fixture-builder candidate, but QR-specific dimensions and allocation assertions must remain clear. |
| QR relative residual | `compute_rel_residual` and `assert_qr_true_residual_below` in `tests/test_qr.c` | Compares reported QR solve residual with true residual for square, tall, rank-deficient, and SuiteSparse cases. | Medium; overlaps with `tf_relative_residual_l2`, but QR also checks reported residual semantics. |
| QR reconstruction | `qr_reconstruction_error` and `assert_qr_reconstruction_below` in `tests/test_qr.c` | Validates `Q*R*P^T` reconstruction across dense, rank, SuiteSparse, and sparse-mode paths. | High for Sprint 120 direct/iterative oracle scope; it is direct-only and not generated-RHS focused. |
| QR least-squares solve block | `test_qr_solve_*` in `tests/test_qr.c` | Covers square, overdetermined, analytical, rank-deficient, nos4, null-residual, bcsstk04, west0067, and QR-vs-LU proof. | Medium-high; candidate for a focused QR solve scenario owner after fixture design. |
| QR refinement block | `test_qr_refine_*` in `tests/test_qr.c` | Covers well-conditioned, ill-conditioned, zero-iter, nos4, overdetermined, and null refine behavior. | Medium; could move with QR solve helpers, but refine semantics should stay QR-local. |
| LDLT Matrix Market validation | `ldlt_validate_mm` in `tests/test_ldlt.c` | Loads Matrix Market fixtures, builds `b = A*ones`, solves, and checks L2 relative residual. | Medium; similar generated-RHS pattern to QR and iterative solvers. |
| LDLT KKT fixture family | `make_kkt` and `test_ldlt_kkt_*` in `tests/test_ldlt.c` | Builds KKT matrices, checks inertia, solves known/generated RHS, and validates residuals. | Medium-high; strong candidate for direct fixture extraction, but inertia expectations are LDLT-specific. |
| LDLT cross-solver comparison | `test_ldlt_vs_lu`, `test_ldlt_kkt_vs_lu`, `test_ldlt_bcsstk04_vs_cholesky` | Compares LDLT solutions with LU/Cholesky on compatible direct paths. | Medium; useful input for bounded cross-solver pilot, but not broad parity. |
| LDLT backend dispatch | `day4_build_indefinite_4x4`, `test_ldlt_backend_*` | Validates linked-list/CSC backend selection, telemetry, invalid backend, and dense backend env contracts. | High; dispatch telemetry and env behavior should stay local unless split as a focused backend owner. |
| LDLT cross-backend agreement | `day5_cross_backend_solves_agree` and Day 5 tests | Compares linked-list and CSC backend solutions on SPD, indefinite, KKT, and threshold fixtures. | Medium; strong direct oracle candidate if split with explicit backend-local tolerance ownership. |
| LDLT CSC KKT fixtures | `build_kkt_5x5`, `build_kkt_10x10`, `build_kkt_scaled_10x10` in `tests/test_ldlt_csc.c` | Feed analysis-aware CSC, residual, and external dense-reference lanes. | Medium; good fixture extraction candidate if lifetime and ownership remain explicit. |
| LDLT CSC two-pass factor workflow | `s20_two_pass_indefinite_factor` in `tests/test_ldlt_csc.c` | Owns scalar pre-pass, permutation, analysis, with-analysis CSC construction, pivot-size seeding, and supernodal factor. | High; dense lifecycle with many cleanup paths. Split only after exact context owner design. |
| LDLT CSC external dense reference | `assert_ldlt_external_dense_reference` and state helpers in `tests/test_ldlt_csc.c` | Runs local CSC solve and compares with Python dense reference for KKT fixtures. | Medium-high; useful oracle owner candidate, but platform skip behavior and permutation lifecycle must stay visible. |
| LDLT CSC residual helpers | `rel_residual`, `s20_solve_residual`, `assert_s20_solve_residual_below` | Check solve correctness for analysis-aware and solve blocks. | Medium; overlaps with shared residual helpers but uses infinity-norm semantics and CSC-specific failure handling. |
| LDLT CSC solve block | Day 9 `test_solve_*` in `tests/test_ldlt_csc.c` | Covers null args, identity, diagonal indefinite, forced 2x2, tridiag indefinite, linked-list comparison, in-place solve, AMD, inertia, and singular detection. | Medium-high; candidate for focused solve scenario owner after fixture design. |

## Existing Helper And Fixture Reuse Map

| Helper | Current Scope | Reuse Notes |
|---|---|---|
| `tf_relative_residual_l2` | Shared L2 relative residual in `tests/test_solver_helpers.h`. | Useful for generated-RHS direct/iterative comparisons, but QR reported residual and LDLT CSC infinity-norm residuals must not be silently changed. |
| `tf_block_relative_residual_l2` | Shared block residual in `tests/test_solver_helpers.h`. | More relevant for Day 3 iterative/block audit than Day 2 direct split. |
| `tf_read_external_reference_vector` | External reference vector reader in `tests/test_solver_helpers.h` under `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`. | Already supports LDLT CSC external dense-reference lane; any split must preserve Windows skip and nonzero-exit failure semantics. |
| `tf_sparse_residual_norminf` | Direct helper infinity-norm residual in `tests/test_direct_solver_helpers.h`. | Potentially reusable for LDLT CSC solve residuals, but current LDLT CSC helper returns relative infinity norm and NaN on allocation failure. |
| `tf_assert_sparse_matrices_equal` | Direct helper matrix equality in `tests/test_direct_solver_helpers.h`. | Useful for direct fixture or reconstruction helpers, but not enough for QR `Q*R*P^T` reconstruction semantics. |
| `make_qr_exact_rhs` | QR-local exact RHS helper. | Generated-RHS pattern is reusable, but current helper asserts QR-specific row/column dimensions and allocation overflow checks. |
| `make_kkt` / `build_kkt_*` | LDLT and LDLT CSC local KKT builders. | Candidate for shared direct fixture builders only if expected inertia, permutation, and external-reference fixture keys remain explicit. |
| `day4_build_indefinite_4x4` | LDLT backend-dispatch fixture. | Candidate for direct backend scenario fixture, but tied to telemetry and backend routing assertions. |

## Giant-Test Hotspot Inventory

| Hotspot | Lines / Region | Why It Is A Hotspot | Possible Split Direction |
|---|---|---|---|
| QR solve and SuiteSparse block | `tests/test_qr.c` around `test_qr_solve_*`, `test_qr_bcsstk04`, `test_qr_west0067`, `test_qr_vs_lu` | Repeats generated RHS, factor/solve/cleanup, residual checks, and direct cross-solver comparison in one giant file. | Focused QR solve scenario owner or shared exact-RHS helper after Day 4 design. |
| QR refinement block | `tests/test_qr.c` around `test_qr_refine_*` | Shares solve setup with QR solve block but has QR-specific residual-improvement semantics. | Keep QR-local or split into QR refinement scenario owner, not generic direct/iterative helper. |
| LDLT Matrix Market and KKT block | `tests/test_ldlt.c` around `ldlt_validate_mm`, `make_kkt`, `test_ldlt_kkt_*` | Repeats generated RHS, residual, inertia, and fixture construction; useful direct oracle material. | Shared direct fixture builder plus LDLT-local inertia assertions. |
| LDLT cross-solver/cross-backend block | `tests/test_ldlt.c` around `test_ldlt_vs_lu`, `test_ldlt_kkt_vs_lu`, `day5_cross_backend_solves_agree` | Good bounded direct oracle material, but includes backend telemetry and path selection. | Candidate for direct split batch if Day 5 ranking accepts backend-local semantics. |
| LDLT CSC external dense-reference block | `tests/test_ldlt_csc.c` around `build_kkt_*`, `s20_two_pass_indefinite_factor`, `assert_ldlt_external_dense_reference` | Dense lifecycle and external reference ownership concentrated in one file. | Candidate for helper owner only with explicit state/free contract and platform skip behavior. |
| LDLT CSC solve block | `tests/test_ldlt_csc.c` Day 9 solve tests | Many solve scenarios, residual semantics, linked-list comparison, in-place behavior, inertia, and singular detection live together. | Candidate for focused solve scenario owner after direct fixture design. |

## Tolerance And Failure-Mode Notes

| Owner | Tolerance / Failure Behavior That Must Stay Visible |
|---|---|
| QR solve | Square QR residual uses `1e-10`; overdetermined and rank-deficient cases intentionally allow larger least-squares residuals; SuiteSparse bcsstk04 uses looser `1e-4`; reported residual is compared against true residual. |
| QR exact RHS | Allocation overflow and dimension mismatches assert locally; generated RHS is `A * [1, 2, ...]`, not a generic ones vector. |
| QR refinement | Well-conditioned and overdetermined refinement must not worsen residual; nos4 may introduce tiny rounding noise but must remain under `1e-10`. |
| LDLT Matrix Market | Generated RHS is `A * ones`; residual is L2 relative residual with fixture-specific tolerance. |
| LDLT KKT | Inertia expectations are part of the oracle, not a generic solve helper. |
| LDLT cross-backend | Solution vectors must agree and recover `ones`; CSC layout may differ from linked-list layout. |
| LDLT dense backend env | Environment variables and platform backend fallback behavior are part of the proof. |
| LDLT CSC external reference | Windows skips external dense-reference helper; Python command failure is a test failure unless it returns an explicit skip; permutation/unpermutation lifecycle is part of the assertion. |
| LDLT CSC analysis-aware residuals | Numeric solve residual is the primary correctness signal when factor storage layouts legitimately differ. |
| LDLT CSC solve | In-place solve, linked-list agreement, AMD permutation, inertia, near-zero pivot, and singular 2x2 block failure behavior must remain explicit. |

## Candidate Split Recommendations For Day 5 Ranking

| Candidate | Initial Recommendation | Reason |
|---|---|---|
| QR exact-RHS/residual helper extraction | Consider | Repeated generated-RHS and residual setup is visible and moderately bounded. Must preserve QR reported-residual semantics. |
| QR solve scenario split | Consider | High line-count reduction potential; behavior is focused around solve residuals and direct QR-vs-LU comparison. |
| LDLT KKT fixture helper extraction | Consider | Reused KKT builders could support direct oracle architecture, but inertia assertions must stay solver-local. |
| LDLT cross-backend scenario split | Consider | Good bounded direct oracle proof; needs explicit backend-local claim boundary. |
| LDLT CSC external dense-reference split | Defer unless Day 4 designs a state owner | Lifecycle, permutation, external process, platform skip, and cleanup are too coupled for casual helper movement. |
| LDLT CSC solve scenario split | Consider later | Valuable, but larger than QR/LDLT linked-list split and depends on shared residual/fixture design. |
| QR reconstruction or sparse-mode split | Defer for Sprint 120 direct/iterative oracle scope | Important direct coverage but less aligned with generated-RHS/dense-reference oracle architecture. |

## Day 4 Shared-Fixture Design Inputs

- Shared generated-RHS builders should support at least:
  - `A * ones`;
  - `A * [1, 2, ...]`;
  - caller-owned expected solution vectors.
- Shared residual helpers must name norm semantics:
  - L2 relative residual;
  - infinity relative residual;
  - raw solver-reported residual where applicable.
- Shared fixture builders must not own solver-specific assertions:
  - QR reported residual interpretation;
  - LDLT inertia expectations;
  - LDLT CSC permutation/unpermutation lifecycle;
  - backend telemetry and environment fallback behavior;
  - external-reference skip/error policy.
- Any new helper owner should be narrow, test-only, and paired with focused
  tests plus full quality if `.c` or `.h` files change.

## Completion Criteria

| Criterion | Status |
|---|---|
| Item 1 direct audit inputs are complete | Complete |
| Every direct candidate has named proof owners and behavior boundaries | Complete |
| Existing helper and fixture reuse is mapped | Complete |
| Tolerance and failure-mode ownership is visible before split selection | Complete |
| No split candidate is selected for implementation before Day 5 ranking | Complete |
