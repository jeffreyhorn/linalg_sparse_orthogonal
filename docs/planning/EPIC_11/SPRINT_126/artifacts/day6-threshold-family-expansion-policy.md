# Sprint 126 Day 6 Threshold Family Expansion Policy

## Purpose

Day 6 extends Sprint 125's QR threshold-family policy to Sprint 126's scaled
diagonal, perturbed duplicate-column, dependent-row, wide, and SuiteSparse
threshold candidates. The policy keeps fixture-local rank-threshold evidence
separate from residual, nullspace, minimum-norm, pseudoinverse, Q-basis,
economy, sparse-mode, reorder, backend, corpus, platform, performance, and
global rank-policy claims.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, maintainer, or public wording files are
changed by Day 6.

## Inputs Reviewed

| Input | Policy Use |
| --- | --- |
| Sprint 126 Plan Day 6 | Requires candidate threshold families, fixture-local expected rank metadata, diagnostics, tolerance policy, SuiteSparse support notes, and non-global rank-policy non-claims. |
| Sprint 126 Day 1 artifact | Provides duplicate fences for completed Sprint 125 threshold evidence and future threshold families. |
| Sprint 126 Day 4-5 artifacts | Require threshold-specific rank/nullity metadata before near-threshold nullspace or subspace evidence can proceed. |
| Sprint 125 Day 6 policy | Defines relative-threshold semantics, candidate families, scale policy, diagnostics, and non-global interpretation rules. |
| Sprint 125 Day 7 evidence | Provides completed `qr_rank_threshold_diag4_family` threshold-rank evidence. |
| Sprint 125 Day 8-9 artifacts | Define SuiteSparse corpus support-tier, optional-data, skip, and diagnostics prerequisites. |
| `tests/test_qr.c` | Current QR rank-threshold product owner. |
| `tests/qr_external_dense_reference.py` | Current standard-library external threshold reference owner. |
| `tests/test_svd.c` | Related SVD threshold evidence owner; not the QR threshold-family owner. |

## Completed Baseline

| Evidence | Status | Sprint 126 Interpretation |
| --- | --- | --- |
| `qr_rank_threshold_diag4_family` | Complete from Sprint 125 | Baseline diagonal threshold ladder with ranks 3, 2, and 1 at `1e-14`, `1e-10`, and `1e-6`. Do not repeat as new Sprint 126 evidence. |
| `test_qr_rank_diagonal_threshold_fixture` | Existing deterministic test | Product-side deterministic coverage for the same diagonal ladder. |
| `test_qr_external_dense_reference_rank_threshold_diag4_family` | Existing external helper-backed test | Bounded helper-backed threshold evidence for the unscaled ladder. |
| `test_svd_rank_diagonal_threshold_fixture` | Existing SVD test | SVD threshold evidence, not QR threshold-family expansion. |

## Threshold Semantics

Sprint 126 threshold evidence must use the product QR semantics already
documented by Sprint 125:

- explicit `tol > 0` maps to absolute threshold `tol * abs(R(0,0))`;
- `tol <= 0` uses the product default threshold and is diagnostic only unless
  a fixture explicitly accepts default-threshold behavior;
- rank is the count of leading `R` diagonal values whose absolute value is
  strictly greater than the absolute threshold;
- expected ranks are fixture-local and must be tied to exact fixture values,
  scale values, perturbation sizes, and named thresholds;
- threshold evidence cannot be inferred from solve residuals, null residuals,
  minimum-norm behavior, or corpus smoke tests.

## Candidate Matrix

| Candidate Family | Candidate Key | Fixture Metadata | Expected Ranks | Trust Value | Day 6 Disposition |
| --- | --- | --- | --- | --- | --- |
| Scaled diagonal ladder | `qr_rank_threshold_diag4_scaled_family` | Diagonal `[s, s*1e-8, s*1e-12, 0]` for named scales such as `1e-6`, `1`, and `1e6`; thresholds `1e-14`, `1e-10`, `1e-6` | For every scale: `3`, `2`, `1` | High. Adds relative-threshold scale invariance beyond the completed unscaled ladder. | Preferred Day 7 candidate. |
| Perturbed duplicate-column ladder | `qr_rank_threshold_duplicate_5x4_perturbed_family` | Duplicate-column 5x4 baseline with perturbations separated from thresholds by at least two orders of magnitude | Threshold-specific, perturbation-specific ranks | Moderate. Connects structural duplicate-column evidence to numerical rank behavior. | Secondary candidate only if perturbation math and diagnostics are stable. |
| Dependent-row threshold ladder | `qr_rank_threshold_dependent_row_4x3_family` | Dependent-row 4x3 baseline with a controlled row or column perturbation | Threshold-specific ranks, likely `2` or `3` depending on perturbation | Moderate later. Relates Day 3 dependent-row residual and deterministic rank evidence to threshold behavior. | Defer unless Day 7 remains documentation-only or proves stable perturbation metadata. |
| Wide threshold ladder | `qr_rank_threshold_wide_3x5_family` | Wide matrix with tiny independent column or controlled dependency | Threshold-specific rank and nullity | Moderate later. Useful for nullity/subspace owners. | Defer to threshold/subspace owner because nullity changes can imply nullspace or minimum-norm behavior. |
| SuiteSparse threshold candidate | corpus-specific key | Checked-in or optional matrix/submatrix with pinned support tier, expected rank, thresholds, and skip behavior | Corpus-specific threshold/rank pairs | Potentially high later. Adds corpus-scale rank-threshold evidence. | Defer to Days 8-9 SuiteSparse corpus gate. |
| Default-threshold diagnostic | no new key yet | Existing diagonal or scaled diagonal fixture with `tol <= 0` | Product default rank only | Low for Day 7. | Diagnostic only; do not promote without explicit default-threshold claim. |

## Required Metadata

Every accepted threshold-family fixture must define:

| Metadata | Requirement |
| --- | --- |
| Fixture key | Stable key naming family, shape, and threshold behavior. |
| Matrix shape | Explicit `m`, `n`, and construction source. |
| Family parameters | Scale values, perturbation values, or corpus matrix/submatrix identifier. |
| Threshold list | Exact relative thresholds in comparison order. |
| Expected ranks | Expected rank for every family parameter and threshold pair. |
| Strict comparison rule | Expected ranks must follow `abs(R(i,i)) > abs_threshold`, not `>=`. |
| Absolute-threshold diagnostics | Product test must print `tol * abs(R(0,0))` for each check. |
| R diagonal diagnostics | Product test must print or record relevant `R` diagonal magnitudes. |
| Reference source | Python standard-library helper or exact derivation; no NumPy, SciPy, LAPACK, BLAS, or SuiteSparse dependency for tiny fixtures. |
| Tolerance/stability rule | Perturbations must be separated from adjacent thresholds enough to avoid roundoff ambiguity. |
| Skip behavior | Platform/helper/corpus skip behavior must be explicit before implementation. |
| Proof boundary | Non-global rank-policy and non-parity wording must be recorded. |

## Diagnostics Policy

Accepted evidence must report:

- fixture key;
- matrix family and shape;
- scale or perturbation value when applicable;
- relative threshold;
- computed absolute threshold;
- expected rank;
- product rank from `sparse_qr_rank()`;
- product rank from `sparse_qr_rank_info()` when available;
- `R` diagonal magnitudes in factorization order;
- `r_max`, `r_min`, `condest`, and `near_deficient` when rank-info is used;
- helper status and output count;
- skip reason for unsupported platform, helper absence, optional corpus
  absence, or unsupported support tier;
- failure class: metadata, helper protocol, threshold calculation, product
  rank, rank-info rank, R diagonal diagnostic, perturbation instability,
  corpus support, or unsupported optional data.

## Tolerance and Stability Policy

| Fixture Class | Stability Rule | Day 7 Requirement |
| --- | --- | --- |
| Scaled diagonal ladder | Scale must not change expected ranks at the same relative thresholds. | Include scale in helper output or product-side metadata and print absolute thresholds. |
| Perturbed duplicate-column | Perturbation magnitude must be at least two orders of magnitude away from adjacent accepted thresholds after scaling by `abs(R(0,0))`. | Do not accept if QR pivoting or roundoff makes expected ranks ambiguous. |
| Dependent-row threshold | Perturbation must identify whether the claim is rank, residual, or nullspace threshold behavior. | Defer unless the primary claim and expected ranks are unambiguous. |
| Wide threshold | Expected rank and nullity must be pinned for each threshold. | Defer unless nullity/subspace and minimum-norm non-claims are explicit. |
| SuiteSparse threshold | Expected ranks must come from independent metadata or a documented support-tier process. | Defer until corpus policy defines optional-data and platform behavior. |

## SuiteSparse Support-Tier Notes

SuiteSparse threshold evidence must not be accepted only because a checked-in
matrix happens to pass a current QR run. It requires:

- matrix path and support tier;
- whether data is checked-in or optional;
- expected rank or threshold/rank pairs;
- relative thresholds and absolute-threshold diagnostics;
- platform expectations;
- skip behavior for absent optional data;
- failure behavior once data is present;
- output that distinguishes rank mismatch from corpus/support problems.

Until those are present, SuiteSparse threshold candidates remain Day 8-9
corpus-gate work.

## Day 7 Candidate Order

Day 7 should evaluate candidates in this order:

1. Scaled diagonal threshold family, because it is the highest-value
   non-duplicative extension of the completed diagonal ladder.
2. Perturbed duplicate-column family, only if the perturbation and expected
   ranks can be made exact enough for stable CI.
3. Dependent-row and wide threshold families, only as explicit deferrals or
   if their primary claim is separated from residual, nullspace, and
   minimum-norm behavior.
4. SuiteSparse threshold candidates should be deferred to Days 8-9 unless
   Day 7 only records promotion gates.

## Day 7 Acceptance Gate

Day 7 may implement a threshold-family fixture only if all of the following
are true:

1. Fixture key, matrix family, parameters, thresholds, and expected ranks are
   pinned.
2. Expected ranks follow the strict product threshold rule.
3. Helper output or exact derivation is available before C test assertions are
   added.
4. Diagnostics include relative threshold, absolute threshold, expected rank,
   product rank, rank-info rank, and R diagonal magnitudes.
5. The artifact states that evidence is fixture-local and not a global QR
   threshold policy.
6. The fixture cannot be read as residual, nullspace, minimum-norm,
   pseudoinverse, Q/economy, sparse-mode, backend, SuiteSparse, or ecosystem
   parity evidence.
7. Focused helper/test commands and required quality gates are known before
   code edits.

If any gate fails, Day 7 should explicitly defer the candidate and record
future owner, blocker, and promotion gate.

## Deferred Promotion Gates

| Deferred Family | Future Owner | Promotion Gate |
| --- | --- | --- |
| Perturbed duplicate-column threshold | Numerical-rank owner | Define perturbation values, expected ranks, strict comparison behavior, and roundoff margin. |
| Dependent-row threshold | Threshold/residual/nullspace owner | Define primary claim and prove it does not mix residual or nullspace evidence into threshold evidence. |
| Wide threshold | Threshold/subspace/minimum-norm owner | Pin rank/nullity per threshold and preserve underdetermined minimum-norm non-claims. |
| SuiteSparse threshold | SuiteSparse corpus owner | Define support tier, expected rank metadata, optional-data behavior, diagnostics, skip/fail policy, and validation. |
| Default-threshold evidence | QR rank-policy owner | Define default-threshold claim separately from explicit-threshold family evidence. |

## Non-Claim Register

Day 6 does not claim:

- a global QR rank-threshold policy;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package threshold
  parity;
- broad QR factorization, QR solve, rank-deficient solve, numerical-rank,
  residual, nullspace, subspace, Q-basis, economy, sparse-mode, reorder,
  backend, corpus, or performance parity;
- minimum-norm optimality, solution uniqueness, pseudoinverse behavior,
  QR-vs-SVD oracle behavior, COLAMD behavior, fallback behavior, or refinement
  behavior;
- SuiteSparse corpus correctness, optional-data behavior, platform support, or
  performance behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 6 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_126`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed for Day 6, so no new code quality gate is required. The current
branch already passed `make format && make lint && make test` after Day 5's
code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Each candidate has fixture-local expected rank metadata. | Complete | See candidate matrix, required metadata, tolerance/stability policy, and Day 7 acceptance gate. |
| Threshold evidence cannot be mistaken for a global rank policy. | Complete | See threshold semantics, non-claim register, and Day 7 acceptance gate. |
| Day 7 implementation inputs are explicit. | Complete | See candidate order, acceptance gate, diagnostics policy, and deferred promotion gates. |
