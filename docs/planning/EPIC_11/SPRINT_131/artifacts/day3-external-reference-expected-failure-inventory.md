# Sprint 131 Day 3 - External-Reference And Expected-Failure Inventory

## Purpose

Day 3 inventories external-reference helpers, helper fixture keys, expected
failures, skip paths, optional-data gates, and claim-boundary risks. It
connects the Day 2 fixture inventory to oracle and failure semantics without
promoting product-observed corpus output into independent evidence.

## External-Reference Helper Inventory

| Helper | Invocation model | Fixture keys or paths | Output class | Oracle source | Current test owner | Skip/error interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| `tests/chol_external_dense_reference.py` | `python3 tests/chol_external_dense_reference.py <matrix_path>` | Matrix Market paths, currently `tests/data/suitesparse/nos4.mtx` and `tests/data/suitesparse/bcsstk04.mtx` | Dense Cholesky solve vector for `x_true = [1..n]` RHS | Helper-local dense Matrix Market loader and dense Cholesky factor/solve | `tests/test_chol_csc.c` | Missing file prints `SKIP`; non-SPD/helper protocol errors print `ERROR`; Windows harness skips helper use. |
| `tests/ldlt_external_dense_reference.py` | `python3 tests/ldlt_external_dense_reference.py <fixture_key>` | `kkt5`, `kkt10`, `ldlt_kkt_scaled_10` | Dense solve vector | Helper-local dense KKT construction plus pivoted dense solve | `tests/test_ldlt_csc.c` | Unknown key or singular dense reference is helper `ERROR`; Windows harness skips helper use. |
| `tests/lu_external_dense_reference.py` | `python3 tests/lu_external_dense_reference.py <fixture_key>` | `lu_nonsym_square_5`, `lu_singular_square_4` | Dense solve vector or expected dense singular failure | Helper-local dense construction plus pivoted dense solve | `tests/test_sparse_lu.c` | `lu_singular_square_4` is an expected-failure oracle for singularity, not a solve-vector oracle. |
| `tests/qr_external_dense_reference.py` | `python3 tests/qr_external_dense_reference.py <fixture_key>` | QR least-squares, rank, threshold-rank, minnorm, economy-projector, and nullspace-projector keys listed below | Numeric reference vector whose fields depend on fixture key | Helper-local dense exact arithmetic, normal equations, projector construction, or threshold policy | `tests/test_qr.c`, `tests/test_qr_solve.c` | Unknown key or unsupported protocol is helper `ERROR`; Windows harness skips helper use. |
| `tests/svd_external_dense_reference.py` | `python3 tests/svd_external_dense_reference.py <fixture_key>` | Full SVD and partial-SVD singular-value keys listed below | Singular values only | Helper-local dense construction plus Jacobi eigenvalues of `A^T A` | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Helper output is value-only; vector residual tests reuse product vectors and external singular values. |

## Helper Fixture Map

| Fixture key | Solver family | Output class | Current owner | Claim boundary |
| --- | --- | --- | --- | --- |
| `nos4.mtx` path | Cholesky CSC | dense solve vector | `tests/test_chol_csc.c` | Bounded external dense solve agreement for Cholesky CSC on one checked-in SPD-like corpus fixture. |
| `bcsstk04.mtx` path | Cholesky CSC | dense solve vector | `tests/test_chol_csc.c` | Bounded external dense solve agreement for Cholesky CSC with AMD reorder on one checked-in SPD-like corpus fixture. |
| `kkt5` | LDLT CSC | dense solve vector | `tests/test_ldlt_csc.c` | Local analytic indefinite KKT solve evidence only. |
| `kkt10` | LDLT CSC | dense solve vector | `tests/test_ldlt_csc.c` | Local analytic indefinite KKT solve evidence only. |
| `ldlt_kkt_scaled_10` | LDLT CSC | dense solve vector | `tests/test_ldlt_csc.c` | Scaled local analytic KKT solve evidence; no broad indefinite-corpus parity. |
| `lu_nonsym_square_5` | Linked-list LU | dense solve vector | `tests/test_sparse_lu.c` | Bounded nonsymmetric square solve evidence. |
| `lu_singular_square_4` | Linked-list LU | expected dense singular failure | `tests/test_sparse_lu.c` | Singular failure semantics; not a successful solve oracle. |
| `qr_overdetermined_incompatible_4x2` | QR solve | least-squares vector plus residual norm | `tests/test_qr_solve.c` | Bounded incompatible least-squares evidence. |
| `qr_overdetermined_compatible_5x3` | QR solve | least-squares vector plus residual norm | `tests/test_qr_solve.c` | Bounded compatible least-squares evidence. |
| `qr_rankdef_duplicate_5x4_rank_only` | QR rank | rank scalar | `tests/test_qr_solve.c` | Rank-only evidence; no residual or nullspace claim by itself. |
| `qr_rankdef_duplicate_5x4_residual_only` | QR solve | column-space residual norm | `tests/test_qr_solve.c` | Residual-only evidence; no full solution uniqueness claim. |
| `qr_rankdef_dependent_row_4x3_residual_only` | QR solve | column-space residual norm | `tests/test_qr_solve.c` | Residual-only dependent-row evidence. |
| `qr_underdetermined_minnorm_2x4` | QR solve | minimum-norm solution vector | `tests/test_qr_solve.c` | Exact bounded underdetermined minimum-norm evidence. |
| `qr_rank_threshold_diag4_family` | QR rank | threshold/rank triples | `tests/test_qr.c` | Fixture-local threshold behavior; no global rank policy. |
| `qr_rank_threshold_diag4_scaled_family` | QR rank | scaled threshold/rank triples | `tests/test_qr.c` | Scaled fixture-local threshold behavior. |
| `qr_rank_threshold_duplicate_5x4_perturbed_family` | QR rank | perturbation/threshold/rank triples | `tests/test_qr.c` | Perturbed duplicate-column threshold evidence only. |
| `qr_rank_threshold_dependent_row_4x3_perturbed_family` | QR rank | perturbation/threshold/rank triples | `tests/test_qr.c` | Perturbed dependent-row threshold evidence only. |
| `qr_economy_projector_5x3` | QR basis/economy | projector values | `tests/test_qr.c` | Economy projector evidence; no raw Q sign/orientation claim. |
| `qr_rank1_4x3_nullspace_projector` | QR nullspace | projector values | `tests/test_qr.c` | Nullspace projector evidence; no raw basis claim. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | QR nullspace | projector values | `tests/test_qr.c` | Dependent-row nullspace projector evidence. |
| `qr_rankdef_wide_3x5_nullspace_subspace` | QR nullspace | projector/subspace values | `tests/test_qr.c` | Wide nullspace subspace evidence; no broad wide QR claim. |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | QR nullspace | projector values | `tests/test_qr.c` | Duplicate-column nullspace projector evidence. |
| `svd_rect_fullrank_6x4` | SVD | singular values | `tests/test_svd.c` | Bounded rectangular singular-value evidence only. |
| `svd_rankdef_duplicate_5x4` | SVD | singular values | `tests/test_svd.c` | Bounded rank-deficient singular-value evidence only. |
| `svd_wide_fullrank_4x6` | SVD | singular values | `tests/test_svd.c` | Bounded wide singular-value evidence only. |
| `partial_svd_diag6_k2` | Partial SVD | top-2 singular values | `tests/test_svd_partial_helpers.h` | Value-only unless paired with a separate vector-residual lane. |
| `partial_svd_tall_diag_8x5_k3` | Partial SVD | top-3 singular values | `tests/test_svd_partial_helpers.h` | Tall value-only evidence unless paired with residual checks. |
| `partial_svd_nonsym_rect10x8_k3` | Partial SVD | top-3 singular values | `tests/test_svd_partial_helpers.h` | Nonsymmetric rectangular singular-value evidence; not subspace or corpus evidence. |

## Expected-Failure And Skip Inventory

| Surface | Current examples | Failure or skip interpretation | Corpus/taxonomy implication |
| --- | --- | --- | --- |
| Parser failures | `bad_header.mtx` and temporary malformed Matrix Market files in `tests/test_sparse_io.c` | Expected `SPARSE_ERR_PARSE`; protects parser behavior. | Parser-negative tag, not numerical evidence. |
| Singular direct-solver cases | `lu_singular_square_4`, hand-built singular LU/CSR fixtures, tiny diagonal Cholesky solve case | Expected `SPARSE_ERR_SINGULAR` or dense-reference singular failure. | Expected-failure tag with fixture-local singularity reason. |
| Non-SPD Cholesky/CSC cases | negative/zero diagonal and non-SPD fixtures | Expected `SPARSE_ERR_NOT_SPD`. | Definiteness-negative tag; not failed Cholesky evidence. |
| Shape/API rejection cases | rectangular direct-solver inputs, NULL args, invalid options | Expected `SPARSE_ERR_SHAPE`, `SPARSE_ERR_NULL`, or `SPARSE_ERR_BADARG`. | API contract tag; usually outside numerical corpus. |
| Iterative non-convergence budgets | BiCGSTAB and MINRES small-budget or hard-scale cases | Expected `SPARSE_ERR_NOT_CONVERGED` or documented graceful numeric failure. | Convergence-budget tag, not solve-quality evidence. |
| External helper disabled on Windows | Cholesky, LDLT, LU, QR, SVD helper tests | `SKIP_TEST` with platform helper-disabled reason. | External-reference support tier must record platform skip behavior. |
| Missing checked-in corpus file | SuiteSparse loads that print `[SKIP]` or `skipped` on missing file | Optional smoke skip unless the owner declares the file required. | Availability tag must distinguish required checked-in fixture from optional smoke. |
| Environment-variable setup failure | graph/reorder/backend tests using `tf_setenv` | Test skips because plumbing cannot be exercised. | Platform/environment skip tag, not evidence failure. |
| Slow/experimental wrappers | `RUN_TEST_SLOW`, `RUN_TEST_EXPERIMENTAL`, and `SPARSE_TEST_LARGE` gates | Skipped unless opt-in env var is enabled. | Slow/experimental support tier; not default reviewed evidence. |
| Product helper fallback skips | e.g. `sparse_pinv failed`, `sparse_cond failed`, factor/preconditioner failed on corpus smoke | Local prerequisite failed, so the test does not prove the downstream claim. | Needs blocker/failure class before promotion. |

## Optional-Corpus Decision Map

| Prior sprint source | Reusable decision for Sprint 131 |
| --- | --- |
| Sprint 125 SuiteSparse rank-deficient QR policy | Missing optional corpus data and platform skips are not failures unless the lane declares the data required; broad SuiteSparse/backend parity remains fenced. |
| Sprint 125-128 SuiteSparse QR/minimum-norm gates | SuiteSparse rank-deficient, nullspace/subspace, threshold, and minimum-norm lanes require independent expected-rank/nullity, projector/residual metrics, support tier, diagnostics, skip behavior, runtime, and validation before promotion. |
| Sprint 129 SuiteSparse Q/economy gate | Checked-in SuiteSparse evidence can be accepted only under a fixture-specific metric and support-tier gate; raw Q-column, broad economy, and large SuiteSparse lanes remain deferred without metadata. |
| Sprint 130 SuiteSparse corpus gate | Product-observed SuiteSparse SVD values, vector residuals, and env-off/env-on low-rank comparisons are smoke unless independent metadata and oracle provenance exist. |
| Sprint 130 convergence/solver-selection closeout | Public solver-selection wording stays unchanged unless evidence directly earns bounded wording. |

## Fixture-Name Versus Claim-Boundary Notes

| Name pattern | Risk | Required wording |
| --- | --- | --- |
| `external_dense_reference` | Can sound like broad dense-library parity. | Name the fixture key, output class, helper protocol, and tolerance; avoid LAPACK/NumPy/SciPy parity unless explicitly sourced. |
| `suitesparse` or corpus file names | Can imply SuiteSparse collection coverage. | Say checked-in SuiteSparse-derived fixture or smoke unless independent corpus metadata exists. |
| `rankdef`, `nullspace`, `subspace`, `projector` | Can imply raw basis stability. | State projector/subspace metric and preserve no raw Q/U/V sign, orientation, or basis claim. |
| `threshold_family` | Can imply global rank-threshold policy. | State fixture-local thresholds, perturbations, and expected ranks. |
| `minnorm` | Can imply all underdetermined solves use minimum norm. | State exact fixture, solver path, and whether SVD-pseudoinverse is an oracle or cross-check. |
| `vector_residual` | Can imply vector parity. | State residual equations and singular-value oracle; preserve no raw singular-vector orientation claim. |
| `benchmark`, `sentinel`, `guardrail` | Can imply performance or scalability proof. | Keep report, timing, threshold, and structural guardrail semantics separate. |

## Support-Tier Gaps

| Gap | Affected sources | Required Day 4+ action |
| --- | --- | --- |
| Helper output schema is implicit in code. | All five external-reference helpers. | Add taxonomy fields for output class and parser protocol before report indexing. |
| Platform skip behavior is not centralized. | External helper tests and env-var tests. | Tag platform skip, helper skip, optional-data skip, and setup skip separately. |
| Expected failures are mixed with positive evidence in test files. | Parser, singular, non-SPD, shape/API, non-convergence tests. | Assign expected-failure tags so report indexes do not count them as positive corpus evidence. |
| Optional checked-in corpus loads are owner-local. | SuiteSparse tests across Cholesky, LDLT, QR, SVD, graph, reorder, iterative, and benchmarks. | Record per-owner required/optional status and runtime tier. |
| Fixture names sometimes overstate scope. | `external`, `suitesparse`, `rankdef`, `minnorm`, `vector_residual` lanes. | Require claim-boundary text in future taxonomy and report-index artifacts. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every external-reference helper fixture has a declared output class. | Complete | Helper fixture map lists solve vector, singular values, rank scalar, threshold triples, residual norm, minnorm vector, projector values, and expected dense singular failure. |
| Expected failures and skips have failure interpretation notes. | Complete | Expected-failure and skip inventory separates parse, singular, non-SPD, shape/API, non-convergence, helper, optional-data, env-var, slow/experimental, and product-prerequisite skips. |
| Optional corpus decisions are traceable to Sprint 125-130 evidence gates. | Complete | Optional-corpus decision map records reusable gates from Sprints 125-130. |

## Day 4 Handoff

Day 4 should turn the Day 2 and Day 3 inventories into taxonomy tags for
structure, numerical properties, solver ownership, optional availability,
support tier, oracle output class, skip/failure class, reviewed status, and
claim-boundary wording.

