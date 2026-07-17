# Sprint 126 Day 4 Nullspace/Subspace Expansion Policy

## Purpose

Day 4 extends Sprint 125's nullspace/subspace policy to the Sprint 126
candidate set: multi-dimensional, wide-shape, dependent-row, near-threshold,
and SuiteSparse QR nullspace/subspace evidence. The policy keeps rank,
nullity, threshold, residual, subspace metric, minimum-norm, pseudoinverse,
Q-basis, economy, sparse-mode, backend, corpus, and performance claims
separate.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, maintainer, or public wording files are
changed by Day 4.

## Inputs Reviewed

| Input | Policy Use |
| --- | --- |
| Sprint 126 Plan Day 4 | Requires expanded nullspace/subspace candidate matrix, pinned metadata requirements, projector or two-way projection metrics, tolerances, diagnostics, skip behavior, and raw-basis non-claims. |
| Sprint 126 Day 1 artifact | Provides duplicate fences and day ownership for Sprint 126 residual debt. |
| Sprint 126 Day 2-3 artifacts | Separate residual-only evidence from nullspace/subspace, minimum-norm, and wide-shape solution-selection claims. |
| Sprint 125 Day 4 policy | Defines sign, ordering, nullity, rank-threshold, projection metric, tolerance, diagnostic, and non-claim rules. |
| Sprint 125 Day 5 decision | Provides completed `qr_rankdef_duplicate_5x4_nullspace_projector` baseline evidence and deferred candidate list. |
| Sprint 125 Day 6-7 artifacts | Provide threshold-family rules required before near-threshold nullspace evidence. |
| Sprint 125 Day 8-9 artifacts | Provide SuiteSparse corpus support-tier and skip-policy prerequisites. |
| Sprint 125 Day 14 handoff | Names future multi-dimensional, wide, near-threshold, SuiteSparse, and Q/economy owners. |
| `tests/test_qr.c` | Current owner for QR nullspace, projector, threshold, wide, Q/economy, sparse-mode, and SuiteSparse-adjacent deterministic evidence. |
| `tests/test_qr_helpers.h` | Current owner for reusable duplicate-column and dependent-row QR fixture builders. |
| `tests/qr_external_dense_reference.py` | Current owner for standard-library external QR projector references. |

## Completed Baseline

| Evidence | Status | Sprint 126 Interpretation |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | Complete from Sprint 125 | Baseline nullity-1 projector evidence. Do not repeat as a new Sprint 126 candidate. |
| `test_rank_1_nullspace` | Existing deterministic test | Internal vector-residual evidence for nullity 2. Not external subspace equivalence. |
| `test_rank_rect_deficient` | Existing deterministic test | Internal wide 3x5 rank/nullity/vector-residual evidence. Not external wide subspace evidence. |
| `test_qr_rank_dependent_row_fixture` | Existing deterministic test | Internal dependent-row rank 2/nullity 1/null residual evidence. Candidate only if an external projector adds trust. |
| `qr_rank_threshold_diag4_family` | Complete from Sprint 125 | Rank-threshold evidence only. Near-threshold nullspace evidence needs a separate threshold/nullity policy. |
| `test_sparse_mode_rank_deficient` and wide sparse-mode tests | Existing deterministic tests | Sparse-mode parity evidence, not nullspace/subspace external evidence. |

## Candidate Matrix

| Candidate | Shape and Metadata | Metric | Trust Value | Risk | Day 4 Disposition |
| --- | --- | --- | --- | --- | --- |
| `qr_rank1_4x3_nullspace_projector` | Existing rank-1 4x3 style fixture; expected rank 1, nullity 2, threshold 0.0 | Full 3x3 projector or two-way projection residual | High. Adds first multi-dimensional nullspace external evidence and exercises rotation-invariant comparison. | Requires stable standard-library orthonormal reference and diagnostics for a nullity greater than 1. | Preferred Day 5 candidate if helper can emit projector or two-way projection data. |
| `qr_rankdef_wide_3x5_nullspace_projector` | Existing wide 3x5 shape; expected rank 2, nullity 3, threshold 0.0 or fixture-pinned value | Two-way projection residual preferred; full 5x5 projector acceptable for tiny fixture | Moderate. Adds wide-shape subspace evidence. | High minimum-norm and underdetermined solution-selection confusion risk. | Accept only after multi-dimensional policy path is proven or explicitly document as deferred to minimum-norm/nullspace owner. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | Existing dependent-row 4x3 helper; expected rank 2, nullity 1, threshold 0.0 | 3x3 projector | Moderate. Complements Day 3 dependent-row residual evidence without reusing residual assertions. | Duplicates deterministic dependent-row null residual unless the external projector adds clear value. | Secondary Day 5 candidate if multi-dimensional evidence is too large or helper-stability fails. |
| Near-threshold diagonal nullspace family | Diagonal threshold fixture with threshold-specific expected ranks and nullities | Separate projector/nullity result per threshold | Moderate later. Connects rank-threshold evidence to nullity behavior. | Requires threshold-family owner; easy to overclaim global threshold behavior. | Defer to Days 6-7 unless Day 5 only documents gates. |
| Dependent-row or wide near-threshold nullspace | Perturbed dependent-row or wide fixture with explicit threshold ladder | Two-way projection residual per accepted threshold | Low for Day 5. Could add future stability evidence. | Mixes rank, residual, threshold, and subspace interpretation. | Defer to threshold/subspace owner after Days 6-7. |
| SuiteSparse nullspace/subspace fixture | Optional or checked-in corpus matrix/submatrix with expected rank/nullity metadata | Two-way projection residual; no full projector for large dimensions | Potentially high later. Adds corpus-scale evidence. | Requires support tier, optional-data skips, platform expectations, rank metadata, and diagnostics. | Defer to Days 8-9 corpus gate. |
| Sparse-mode nullspace/subspace parity | Sparse-mode QR output compared with dense QR for rank-deficient/wide shape | Projection metric between product subspaces | Low for Day 5. | Belongs to Sprint 127 Q/economy/sparse-mode owner. | Defer to Sprint 127. |

## Required Metadata

Every accepted nullspace/subspace candidate must define all of the following
before code edits:

| Metadata | Requirement |
| --- | --- |
| Fixture key | Stable key naming the exact matrix and behavior. |
| Matrix shape | Explicit `m`, `n`, and construction source. |
| Expected rank | Pinned rank for the fixture and threshold. |
| Expected nullity | Pinned as `n - expected_rank`; do not infer from residual-only evidence. |
| Rank threshold | Fixture-local threshold, commonly `0.0` for exact structural rank deficiency. |
| Reference source | Python standard-library helper or documented exact derivation; no NumPy, SciPy, LAPACK, BLAS, or SuiteSparse dependency. |
| Metric | Full projector for small fixtures or two-way projection residual for larger/multi-dimensional fixtures. |
| Tolerance | Fixture-local tolerance for projector/projection metric, null residual, orthonormality, and metadata checks. |
| Diagnostics | Required printed values and failure classes. |
| Skip behavior | Platform/helper/corpus skip behavior before implementation. |
| Proof boundary | Explicit non-claims for raw basis equality, Q/economy, minimum-norm, pseudoinverse, backend, and corpus parity. |

## Metric Policy

| Metric | Default Use | Acceptance Rule |
| --- | --- | --- |
| Full projector `Z Z^T` | Small fixtures where `n * n` values are manageable. | Compare product and reference projectors with max absolute or Frobenius error below fixture tolerance. |
| Two-way projection residual | Multi-dimensional, wide, or larger fixtures where storing a full projector is noisy. | Assert both `||(I - P_ref)Z||` and `||(I - P)Z_ref||` are below fixture tolerance. |
| Null residual `||A*z_i||_2` | Required diagnostic and secondary correctness check for product basis vectors. | May be asserted, but cannot by itself prove external subspace equivalence. |
| Orthonormality `||Z^T Z - I||` | Required when product basis is used in projector/projection metrics. | Product basis must be normalized or orthonormalized before metric comparison. |
| Principal angles | Strong mathematical metric, but implementation-heavy. | Deferred unless a future artifact justifies helper protocol complexity. |
| Raw vector equality | Sign, ordering, and rotation sensitive. | Disallowed by default; no Sprint 126 Day 5 candidate may use raw basis equality. |

## Fixture Tolerances

| Fixture Class | Rank Threshold | Nullity | Subspace Tolerance | Null Residual | Notes |
| --- | ---:| ---:| ---:| ---:| --- |
| Multi-dimensional exact fixture | `0.0` | `> 1` | projector or two-way projection residual `<= 1e-8` | `<= 1e-10` | Preferred Day 5 expansion if helper output is stable. |
| Dependent-row exact fixture | `0.0` | `1` | projector max diff `<= 1e-8` | `<= 1e-10` | Must show value beyond deterministic dependent-row test. |
| Wide exact fixture | fixture-pinned | `n - rank`, typically `> 1` | two-way projection residual `<= 1e-8` | `<= 1e-10` | Must explicitly exclude minimum-norm and underdetermined solve claims. |
| Near-threshold fixture | threshold-specific | threshold-specific | threshold-specific | threshold-specific | Defer until threshold family owns expected rank/nullity. |
| SuiteSparse fixture | corpus-specific | corpus-specific | corpus-specific, likely relative/two-way | corpus-specific | Requires support-tier and optional-data policy first. |

Tolerances may be tightened only if the Day 5 implementation artifact records
the reference values, product diagnostics, and reason for the tighter bound.
They may be loosened only with a fixture-local numerical explanation and an
explicit non-claim update.

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
- product/reference projector or projection residual maximum;
- skip reason for unsupported platform, helper absence, or optional corpus
  absence;
- failure class: metadata, rank, nullity, helper protocol, product basis,
  reference basis, subspace metric, tolerance, corpus support, or unsupported
  optional data.

## Candidate Ordering for Day 5

Day 5 should evaluate candidates in this order:

1. `qr_rank1_4x3_nullspace_projector` or equivalent multi-dimensional exact
   nullspace fixture because it adds the most distinct trust beyond Sprint
   125's nullity-1 projector evidence.
2. `qr_rankdef_dependent_row_4x3_nullspace_projector` if the
   multi-dimensional helper path is unstable or too broad for Day 5.
3. `qr_rankdef_wide_3x5_nullspace_projector` only if the artifact can
   explicitly fence minimum-norm and underdetermined solution-selection
   interpretations.
4. Near-threshold and SuiteSparse candidates should be deferred unless Day 5
   remains documentation-only and records promotion gates.

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
   economy, sparse-mode, backend, or SuiteSparse corpus evidence.
7. Focused helper/test commands and the required quality gate are known before
   editing code.

If any of these gates fail, Day 5 should explicitly defer the candidate and
name the future owner and promotion gate.

## Deferred Promotion Gates

| Deferred Lane | Future Owner | Promotion Gate |
| --- | --- | --- |
| Wide-shape nullspace/subspace | QR nullspace or minimum-norm owner | Pin expected rank/nullity and prove projection evidence cannot imply underdetermined minimum-norm solution selection. |
| Near-threshold nullspace/subspace | Threshold/subspace owner | Complete threshold-family expected-rank metadata first, then define threshold-specific nullity and projector/projection expectations. |
| SuiteSparse nullspace/subspace | SuiteSparse corpus owner | Define support tier, optional-data skip behavior, expected-rank/nullity metadata, diagnostics, platform expectation, and validation. |
| Sparse-mode nullspace/subspace | Sprint 127 Q/economy/sparse-mode owner | Define sparse-mode output semantics, Q/economy boundaries, projection metric, and failure interpretation. |
| Principal-angle metric | Future subspace metric owner | Justify helper complexity and prove it adds value over projector or two-way projection residuals. |
| Raw basis equality | Future deterministic-basis owner | Prove deterministic ordering, normalization, sign convention, and a reason projection metrics are insufficient. |

## Non-Claim Register

Day 4 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  subspace, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity;
- raw nullspace basis equality, basis ordering, unique orientation, sign,
  principal-angle parity, or raw vector-column parity;
- global QR rank-threshold policy;
- minimum-norm optimality, solution uniqueness, pseudoinverse behavior,
  QR-vs-SVD oracle behavior, COLAMD behavior, fallback behavior, or refinement
  behavior;
- SuiteSparse corpus correctness, optional-data behavior, platform support, or
  performance behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 4 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_126`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed for Day 4, so no new code quality gate is required. The current
branch already passed `make format && make lint && make test` after Day 3's
code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No nullspace/subspace candidate proceeds without pinned rank and nullity. | Complete | See required metadata, fixture tolerances, candidate matrix, and Day 5 acceptance gate. |
| Raw vector equality, basis ordering, and unique-basis claims remain fenced. | Complete | See metric policy, deferred promotion gates, and non-claim register. |
| SuiteSparse candidates have explicit support-tier and skip requirements. | Complete | See candidate matrix, diagnostics policy, and deferred promotion gates. |
