# Sprint 128 Day 4 Wide and Near-Threshold Subspace Policy

## Purpose

Day 4 defines the policy for Sprint 128's remaining wide-shape,
near-threshold, and SuiteSparse QR nullspace/subspace candidates. The policy
keeps subspace evidence separate from rank-only, residual-only,
threshold-family, minimum-norm, pseudoinverse, Q-basis, economy, sparse-mode,
SuiteSparse corpus, backend, platform, and performance claims.

This artifact is policy-only. It does not add tests, external-reference
helper output, Matrix Market data, build membership, maintainer wording,
public solver wording, or claims.

## Inputs Reviewed

| Input | Policy Use |
| --- | --- |
| Sprint 128 Plan Day 4 | Requires a nullspace/subspace candidate matrix, metadata requirements, projector or two-way projection metrics, sparse/economy semantics, support tier, skip behavior, tolerances, and diagnostics. |
| Sprint 128 Day 1 artifact | Provides duplicate fences and day-level owners for Sprint 127 carry-forward debt. |
| Sprint 128 Day 2-3 artifacts | Separate compatible/wide residual-only evidence from nullspace/subspace, minimum-norm, and Q/economy claims. |
| Sprint 127 Day 4-5 artifacts | Provide the latest nullspace/subspace policy and completed dependent-row projector evidence. |
| Sprint 125-126 Day 4-5 artifacts | Provide earlier projector policies and completed duplicate-column and rank-1/nullity-2 projector evidence. |
| Sprint 125-127 threshold artifacts | Provide rank-threshold rules needed before near-threshold nullspace/subspace evidence. |
| Sprint 126-127 SuiteSparse artifacts | Provide corpus support-tier, skip-behavior, runtime, and expected-rank metadata gates. |
| `tests/test_qr.c` | Current owner for QR nullspace, projector, threshold, wide, Q/economy, sparse-mode, and SuiteSparse-adjacent deterministic evidence. |
| `tests/qr_external_dense_reference.py` | Current owner for standard-library external QR projector references. |
| `docs/maintainer_guide.md` QR evidence row | Records maintained QR evidence and public non-claim boundaries. |

## Completed Baseline

| Evidence | Status | Sprint 128 Interpretation |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Complete from Sprint 125 | Baseline nullity-1 external projector evidence. Do not repeat as wide, near-threshold, dependent-row, or SuiteSparse evidence. |
| `qr_rank1_4x3_nullspace_projector` | Complete from Sprint 126 | Baseline multi-dimensional nullity-2 external projector evidence. Do not repeat as wide, near-threshold, sparse/economy, or SuiteSparse evidence. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | Complete from Sprint 127 | Baseline dependent-row external projector evidence. Do not repeat as wide-shape, near-threshold, SuiteSparse, sparse/economy, or raw-basis evidence. |
| `test_rank_rect_deficient` | Existing deterministic product test | Internal wide 3 x 5 rank/nullity/vector-residual evidence. Not external wide subspace evidence. |
| `test_qr_wide`, `test_q_orthogonality_wide`, `test_economy_wide`, and `test_sparse_mode_wide` | Existing deterministic product tests | Wide, Q/economy, and sparse-mode coverage. Not nullspace/subspace equivalence proof. |
| `qr_rank_threshold_diag4_family`, `qr_rank_threshold_diag4_scaled_family`, and `qr_rank_threshold_duplicate_5x4_perturbed_family` | Complete rank-threshold evidence | Rank-threshold evidence only. Near-threshold nullspace/subspace evidence needs threshold-specific nullity metadata. |
| Checked-in SuiteSparse QR controls | Existing product/corpus controls | Full-rank or control evidence. Not rank-deficient subspace evidence without independent expected-rank/nullity metadata. |

## Candidate Matrix

| Candidate | Shape and Metadata | Metric | Trust Value | Risk | Day 4 Disposition |
| --- | --- | --- | --- | --- | --- |
| `qr_rankdef_wide_3x5_nullspace_subspace` | Existing or small wide 3 x 5 shape; expected rank and nullity pinned before use | Two-way projection residual preferred; full 5 x 5 projector acceptable only if tiny and readable | Moderate. Would add first wide-shape external subspace evidence. | High minimum-norm, underdetermined solution-selection, Q/economy, and sparse-mode confusion risk. | Candidate for Day 5 only if rank/nullity, projection metric, and non-claim wording are pinned. |
| Wide exact nullity-2 or higher fixture | New small wide shape with exact construction and closed-form subspace | Two-way projection residual | Moderate if non-duplicate. | New fixture family may duplicate existing wide/minimum-norm evidence or expand scope. | Defer unless Day 5 proves non-duplicate trust value and exact metadata. |
| Near-threshold diagonal nullspace family | Existing diagonal threshold family with threshold-specific expected ranks/nullities | Projector or two-way projection residual per threshold | Moderate later. Connects rank-threshold evidence to nullity behavior. | Easy to overclaim global threshold/default-threshold behavior. | Defer to Days 6-7 threshold-family owners unless Day 5 only records gates. |
| Near-threshold perturbed duplicate-column subspace | Existing perturbed duplicate-column threshold fixture with threshold-specific rank/nullity | Two-way projection residual per threshold | Moderate later. Connects accepted perturbation threshold evidence to subspace behavior. | Mixes perturbation, threshold, and subspace interpretation; raw basis can be unstable. | Defer until threshold metadata and expected nullity are complete. |
| Wide near-threshold nullspace | Wide fixture with threshold-specific expected ranks/nullities | Two-way projection residual | Low for Day 5. | Combines wide output semantics with threshold policy and minimum-norm confusion risk. | Defer until wide semantics and threshold metadata are both complete. |
| SuiteSparse nullspace/subspace fixture | Checked-in or optional corpus matrix/submatrix with independent expected rank/nullity metadata | Two-way projection residual; avoid full projectors for large dimensions | Potentially high later. Adds corpus-scale subspace evidence. | Requires support tier, optional-data skips, runtime budget, platform expectations, rank/nullity metadata, diagnostics, and validation. | Defer to Days 8-9 corpus gate unless all metadata is available. |
| Sparse-mode nullspace/subspace parity | Sparse-mode QR output compared with dense QR for rank-deficient/wide shape | Projection metric between product subspaces | Moderate later. | Belongs to Sprint 129 Q/economy/sparse-mode output semantics. | Defer to Sprint 129. |

## Required Metadata

Every accepted nullspace/subspace candidate must define all of the following
before code edits:

| Metadata | Requirement |
| --- | --- |
| Fixture key | Stable key naming the exact matrix and behavior. |
| Matrix shape | Explicit `m`, `n`, and construction source. |
| Expected rank | Pinned rank for the fixture and threshold. |
| Expected nullity | Pinned as `n - expected_rank`; do not infer from residual-only or minimum-norm evidence. |
| Rank threshold | Fixture-local threshold, commonly `0.0` for exact structural rank deficiency. |
| Reference source | Python standard-library helper or documented exact derivation; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependency. |
| Metric | Full projector for tiny fixtures or two-way projection residual for wide, near-threshold, multi-dimensional, or corpus fixtures. |
| Tolerance | Fixture-local tolerance for projector/projection metric, null residual, orthonormality, and metadata checks. |
| Sparse/economy semantics | Explicit statement whether sparse-mode or economy output is involved; default is no sparse/economy claim. |
| Support tier | Required for SuiteSparse or optional-data candidates. |
| Skip behavior | Platform/helper/corpus skip behavior before implementation. |
| Diagnostics | Required printed values and failure classes. |
| Proof boundary | Explicit non-claims for raw basis equality, Q/economy, minimum-norm, pseudoinverse, backend, corpus, platform, and performance parity. |

## Metric Policy

| Metric | Default Use | Acceptance Rule |
| --- | --- | --- |
| Full projector `Z Z^T` | Tiny fixtures where `n * n` values are manageable and readable. | Compare product and reference projectors with max absolute or Frobenius error below fixture tolerance. |
| Two-way projection residual | Wide, multi-dimensional, near-threshold, or larger fixtures where storing a full projector is noisy. | Assert both `||(I - P_ref)Z_product||` and `||(I - P_product)Z_ref||` are below fixture tolerance. |
| Null residual `||A*z_i||_2` | Required diagnostic and secondary correctness check for product basis vectors. | May be asserted, but cannot by itself prove external subspace equivalence. |
| Orthonormality `||Z^T Z - I||` | Required when product basis is used in projector/projection metrics. | Product basis must be normalized or orthonormalized before metric comparison. |
| Principal angles | Strong mathematical metric but implementation-heavy. | Deferred unless a future artifact justifies helper protocol complexity. |
| Raw vector equality | Sign, ordering, and rotation sensitive. | Disallowed by default; no Day 5 candidate may use raw basis equality. |

## Sparse/Economy Output Semantics

| Scenario | Sprint 128 Rule |
| --- | --- |
| Dense QR product nullspace | Allowed when fixture metadata, projector/projection metric, and tolerances are pinned. |
| Economy QR output | No economy claim unless the artifact states the economy shape, output columns, and projection boundary. Otherwise defer to Sprint 129. |
| Sparse-mode QR output | No sparse-mode claim unless sparse-mode output representation and dense/sparse comparison semantics are explicit. Otherwise defer to Sprint 129. |
| Wide shape | Must state that nullspace/subspace evidence is about the nullspace of `A`, not which underdetermined solution `sparse_qr_solve` or `sparse_qr_solve_minnorm` returns. |
| Minimum-norm overlap | Any fixture that needs norm minimization, pseudoinverse comparison, or free-variable selection belongs to the minimum-norm owner, not Day 5 subspace policy. |

## Fixture Tolerances

| Fixture Class | Rank Threshold | Nullity | Subspace Tolerance | Null Residual | Notes |
| --- | ---:| ---:| ---:| ---:| --- |
| Wide exact fixture | fixture-pinned, often `0.0` | `n - rank`, often `> 1` | two-way projection residual `<= 1e-8` | `<= 1e-10` | Must explicitly exclude minimum-norm and underdetermined solve claims. |
| Near-threshold diagonal or perturbed fixture | threshold-specific | threshold-specific | threshold-specific | threshold-specific | Defer until threshold family owns expected rank/nullity. |
| SuiteSparse fixture | corpus-specific | corpus-specific | corpus-specific, likely relative/two-way | corpus-specific | Requires support-tier, optional-data, runtime, and expected-rank policy first. |
| Tiny exact structural fixture | `0.0` | fixture-specific | projector max diff `<= 1e-8` | `<= 1e-10` | Only acceptable when it is not a duplicate of the three completed projector baselines. |

Tolerances may be tightened only if the implementation artifact records the
reference values, product diagnostics, and reason for the tighter bound. They
may be loosened only with a fixture-local numerical explanation and an explicit
non-claim update.

## Diagnostics Policy

Accepted evidence must print or record:

- fixture key;
- matrix shape;
- expected rank, product rank, expected nullity, and product nullity;
- rank threshold;
- reference helper status and output count;
- product null residual maximum;
- product basis orthonormality error;
- reference basis orthonormality error if a basis is emitted;
- selected subspace metric and tolerance;
- product/reference projector or two-way projection residual maximum;
- sparse/economy output-semantics status;
- support tier and optional gate state for corpus candidates;
- skip reason for unsupported platform, helper absence, or optional corpus
  absence;
- failure class: metadata, rank, nullity, helper protocol, product basis,
  reference basis, subspace metric, tolerance, output semantics, corpus
  support, runtime budget, or unsupported optional data.

## Candidate Ordering for Day 5

Day 5 should evaluate candidates in this order:

1. `qr_rankdef_wide_3x5_nullspace_subspace`, because Sprint 128 Item 3 names
   wide-shape evidence and the existing deterministic wide tests provide a
   local starting point.
2. Near-threshold candidates, only if Day 5 can pin threshold-specific rank,
   nullity, projection metric, tolerance, and non-global interpretation before
   code changes.
3. SuiteSparse candidates, only if expected-rank/nullity metadata, support
   tier, optional-data behavior, runtime budget, and diagnostics are all
   available before code registration.
4. Sparse-mode or economy subspace candidates should be deferred to Sprint 129
   unless Day 5 can prove they are pure dense-product subspace evidence.

## Day 5 Acceptance Gate

Day 5 may implement a nullspace/subspace fixture only if all of the following
are true:

1. Expected rank, nullity, threshold, matrix shape, and metric are pinned.
2. The external reference is generated by Python standard-library code or an
   exact derivation documented in the artifact.
3. The comparison uses full projector or two-way projection metrics, not raw
   basis-vector equality.
4. The test asserts only metadata, rank/nullity, null residual,
   orthonormality, and subspace metric behavior.
5. The artifact explains sign, ordering, and rotation ambiguity.
6. The fixture cannot be read as minimum-norm, pseudoinverse, Q-basis,
   economy, sparse-mode, backend, SuiteSparse corpus, optional-data, platform,
   or performance evidence.
7. Focused helper/test commands and the required quality gate are known before
   editing code.

If any of these gates fail, Day 5 should explicitly defer the candidate and
name the future owner and promotion gate.

## Deferred Promotion Gates

| Deferred Lane | Future Owner | Promotion Gate |
| --- | --- | --- |
| Wide-shape nullspace/subspace | Day 5 QR subspace owner or future minimum-norm owner | Pin expected rank/nullity and prove projection evidence cannot imply underdetermined minimum-norm solution selection. |
| Near-threshold nullspace/subspace | Threshold/subspace owner | Complete threshold-family expected-rank metadata first, then define threshold-specific nullity and projector/projection expectations. |
| SuiteSparse nullspace/subspace | SuiteSparse corpus owner | Define support tier, optional-data skip behavior, expected-rank/nullity metadata, diagnostics, platform/runtime expectation, and validation. |
| Sparse-mode nullspace/subspace | Sprint 129 Q/economy/sparse-mode owner | Define sparse-mode output semantics, Q/economy boundaries, projection metric, and failure interpretation. |
| Principal-angle metric | Future subspace metric owner | Justify helper complexity and prove it adds value over projector or two-way projection residuals. |
| Raw basis equality | Future deterministic-basis owner | Prove deterministic ordering, normalization, sign convention, and a reason projection metrics are insufficient. |

## Non-Claim Register

Day 4 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  subspace, Q-basis, economy, sparse-mode, reorder, backend, corpus,
  optional-data, platform, or performance parity;
- new nullspace/subspace evidence beyond completed Sprint 125-127 projector
  fixtures;
- raw nullspace basis equality, basis ordering, unique orientation, sign,
  principal-angle parity, or raw vector-column parity;
- global QR rank-threshold, default-threshold, or numerical-rank policy;
- minimum-norm optimality, solution uniqueness, solution-selection policy,
  pseudoinverse behavior, QR-vs-SVD oracle behavior, COLAMD behavior,
  fallback behavior, or refinement behavior;
- SuiteSparse corpus correctness, optional-data behavior, runtime behavior,
  platform support, or performance behavior;
- generic QR/SVD helper API or helper consolidation behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 4 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed, so no code quality gate is required for Day 4.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No subspace candidate proceeds without pinned rank and nullity. | Complete | See required metadata, fixture tolerances, candidate matrix, and Day 5 acceptance gate. |
| Raw basis equality, basis ordering, and unique-basis claims remain fenced. | Complete | See metric policy, deferred promotion gates, and non-claim register. |
| SuiteSparse candidates have explicit support-tier and skip requirements. | Complete | See candidate matrix, diagnostics policy, and deferred promotion gates. |
