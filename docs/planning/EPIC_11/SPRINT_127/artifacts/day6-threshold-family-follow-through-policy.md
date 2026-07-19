# Sprint 127 Day 6 Threshold Family Follow-Through Policy

## Purpose

Day 6 defines the Sprint 127 follow-through policy for the remaining QR
threshold-family debt: perturbed duplicate-column, dependent-row, wide,
default-threshold, and SuiteSparse threshold candidates. The policy starts
from the completed Sprint 125 unscaled diagonal ladder and Sprint 126 scaled
diagonal ladder, then decides what may proceed on Day 7 and what must remain
behind corpus, subspace, or rank-policy gates.

This is a policy artifact only. It does not change C source, headers, Python
helpers, build files, CMake, CTest, workflow files, public APIs, maintainer
wording, or public solver-selection wording.

## Inputs Reviewed

| Input | Day 6 Use |
| --- | --- |
| Sprint 127 project-plan Item 4 | Requires add-or-defer handling for perturbed duplicate-column, dependent-row, wide, default-threshold, and SuiteSparse QR threshold families with expected ranks, diagnostics, support tier, and non-global interpretation. |
| Sprint 127 Day 1 artifact | Provides duplicate fences for completed Sprint 121-126 QR threshold, residual, nullspace, SuiteSparse, and minimum-norm evidence. |
| Sprint 127 Day 2-3 artifacts | Keep threshold evidence separate from compatible and wide residual-only behavior. |
| Sprint 127 Day 4-5 artifacts | Keep threshold evidence separate from nullspace/subspace evidence and require threshold-specific nullity metadata before near-threshold subspace promotion. |
| Sprint 125 Day 6-7 artifacts | Provide the original QR threshold-family policy and completed `qr_rank_threshold_diag4_family` evidence. |
| Sprint 126 Day 6-7 artifacts | Provide the expanded threshold-family policy and completed `qr_rank_threshold_diag4_scaled_family` evidence. |
| Sprint 126 Day 8-9 artifacts | Define SuiteSparse corpus support-tier and expected-rank metadata gates. |
| `tests/test_qr.c` | Current QR threshold-rank product owner for `sparse_qr_rank()` and `sparse_qr_rank_info()` evidence. |
| `tests/qr_external_dense_reference.py` | Current standard-library helper owner for bounded QR threshold references. |

## Completed Baseline

| Evidence | Status | Sprint 127 Interpretation |
| --- | --- | --- |
| `qr_rank_threshold_diag4_family` | Complete from Sprint 125 | Baseline explicit-threshold diagonal ladder with expected ranks `3`, `2`, and `1` at `1e-14`, `1e-10`, and `1e-6`. Do not repeat as Sprint 127 evidence. |
| `qr_rank_threshold_diag4_scaled_family` | Complete from Sprint 126 | Scale-invariance follow-through for the same diagonal ladder at scales `1e-6`, `1`, and `1e6`. Do not repeat as Sprint 127 evidence. |
| `test_qr_rank_diagonal_threshold_fixture` | Existing deterministic product test | Product-side deterministic threshold smoke; not new Sprint 127 evidence. |
| `sparse_qr_rank_info()` diagnostics | Existing product API | Useful diagnostics for accepted threshold fixtures; not itself a proof of global threshold policy. |
| Day 5 `qr_rankdef_dependent_row_4x3_nullspace_projector` | Complete from Sprint 127 | Dependent-row subspace evidence at threshold `0.0`; not threshold-family evidence. |

## Threshold Semantics

Sprint 127 threshold evidence must preserve the product semantics used by the
completed Sprint 125-126 evidence:

- explicit `tol > 0` maps to absolute threshold `tol * abs(R(0,0))`;
- `tol <= 0` invokes the product default threshold and cannot be treated as
  explicit-threshold evidence unless a separate default-threshold claim is
  accepted;
- rank is the count of leading `R` diagonal magnitudes strictly greater than
  the absolute threshold;
- expected ranks are fixture-local and must be tied to exact fixture values,
  perturbation values, scale values, corpus matrix identifiers, and named
  thresholds;
- solve residuals, null residuals, subspace projector checks, and
  minimum-norm behavior cannot be used as substitute threshold proof.

## Candidate Matrix

| Candidate Family | Candidate Key | Required Metadata | Expected Ranks | Trust Value | Day 6 Disposition |
| --- | --- | --- | --- | --- | --- |
| Perturbed duplicate-column ladder | `qr_rank_threshold_duplicate_5x4_perturbed_family` | Existing duplicate-column 5 x 4 fixture plus one controlled perturbation magnitude per record, explicit thresholds, strict comparison rule, and R-diagonal diagnostics | Threshold- and perturbation-specific, likely rank `3` or `4` depending on perturbation separation | Moderate. Connects completed duplicate-column rank/residual/nullspace evidence to numerical-rank behavior. | Preferred Day 7 implementation candidate only if perturbation values can be separated from thresholds by at least two orders of magnitude after factoring. |
| Dependent-row threshold ladder | `qr_rank_threshold_dependent_row_4x3_family` | Existing dependent-row 4 x 3 fixture plus controlled perturbation source, explicit primary claim, threshold list, expected ranks, and residual/subspace non-claims | Threshold-specific, likely rank `2` for exact dependency and `3` when perturbation is above threshold | Moderate but mixed. Relates dependent-row residual and subspace fixtures to threshold behavior. | Secondary Day 7 candidate only if it is framed as rank-threshold evidence and does not reuse residual or projector metrics as proof. |
| Wide threshold ladder | `qr_rank_threshold_wide_3x5_family` | Wide matrix shape, rank/nullity per threshold, perturbation source, thresholds, and underdetermined/minimum-norm non-claims | Threshold-specific rank and nullity | Useful later for subspace owners. | Defer by default because rank changes imply nullity changes and can be misread as underdetermined solution-selection or minimum-norm evidence. |
| Default-threshold diagnostic | `qr_rank_threshold_diag4_default_diagnostic` or no key until accepted | Exact fixture, `tol <= 0` semantics, product default formula, expected default rank, and platform/compiler stability note | Product-default rank only | Low. Useful to document current implementation default, but high risk of global policy overclaim. | Defer unless Day 7 only records a diagnostic with explicit no-policy wording. |
| SuiteSparse threshold candidate | Corpus-specific key | Matrix path, support tier, checked-in or optional-data status, expected rank metadata, threshold list, runtime budget, skip/fail behavior, and platform diagnostics | Corpus-specific threshold/rank pairs | Potentially high later. Adds corpus-scale threshold evidence. | Defer to Days 8-9 corpus gate. |
| Near-threshold nullspace/subspace family | No Day 7 key | Threshold-specific rank and nullity plus projector or two-way projection residual metric | Rank and nullity per threshold | Belongs to subspace follow-through. | Defer until threshold evidence and subspace metrics are both accepted. |

## Perturbation and Separation Rules

Accepted perturbation-family evidence must satisfy all of these rules before
code edits:

1. The baseline matrix, perturbation location, perturbation sign, and
   perturbation magnitude are named in the artifact and helper/test code.
2. Each perturbation magnitude is at least two orders of magnitude away from
   the adjacent accepted relative thresholds after scaling by `abs(R(0,0))`.
3. Expected ranks follow strict `abs(R(i,i)) > tol * abs(R(0,0))`
   comparison, not `>=`.
4. The product test prints R diagonal magnitudes so a failure can be classified
   as metadata drift, perturbation instability, or product rank behavior.
5. The fixture does not depend on pivot-order-sensitive raw values unless the
   expected R diagonal ordering is part of the fixture contract.

## Required Metadata

Every accepted Day 7 threshold fixture must define:

| Metadata | Requirement |
| --- | --- |
| Fixture key | Stable key naming the matrix family and threshold behavior. |
| Matrix shape | Explicit `m`, `n`, and construction source. |
| Family parameters | Perturbation values, scale values, default-threshold flag, or corpus identifier. |
| Threshold list | Exact relative thresholds in comparison order, or explicit `tol <= 0` default behavior when accepted. |
| Expected ranks | Expected rank for every parameter and threshold pair. |
| Strict comparison rule | Evidence follows `abs(R(i,i)) > abs_threshold`. |
| Diagnostics | Relative threshold, absolute threshold, expected rank, product rank, rank-info rank, R diagonal magnitudes, and helper output count. |
| Reference source | Python standard-library helper or exact derivation for tiny fixtures; no NumPy, SciPy, LAPACK, BLAS, or SuiteSparse dependency. |
| Support tier | Required before any SuiteSparse or optional-data candidate proceeds. |
| Skip behavior | Required for helper absence, optional data absence, unsupported support tier, or platform-specific corpus conditions. |
| Proof boundary | Explicit non-claims for global rank policy, default-threshold policy unless accepted, residual, nullspace, minimum-norm, Q/economy, sparse-mode, backend, corpus, platform, performance, and external-library parity. |

## Diagnostics Policy

Accepted evidence must report enough data to identify the failure class:

- fixture key;
- matrix family and shape;
- perturbation, scale, default-threshold, or corpus parameter;
- relative threshold and computed absolute threshold;
- expected rank;
- product rank from `sparse_qr_rank()`;
- rank-info rank from `sparse_qr_rank_info()`;
- `R` diagonal magnitudes in factorization order;
- `r_max`, `r_min`, `condest`, and `near_deficient` when rank-info is used;
- helper status and output count;
- support-tier and optional-data status for corpus candidates;
- skip reason, if any;
- failure class: metadata, helper protocol, perturbation separation, product
  rank, rank-info rank, default-threshold interpretation, support tier,
  optional data, or unsupported platform.

## Day 7 Candidate Order

Day 7 should evaluate candidates in this order:

1. `qr_rank_threshold_duplicate_5x4_perturbed_family`, only if exact
   perturbation metadata and stable expected ranks can be pinned before code
   edits.
2. `qr_rank_threshold_dependent_row_4x3_family`, only if the primary claim is
   rank-threshold behavior and residual/projector evidence remains fenced.
3. Default-threshold diagnostics, only if they are recorded as product-local
   diagnostics rather than a global default-threshold policy.
4. Wide threshold and near-threshold subspace lanes should be explicit
   deferrals unless rank/nullity and minimum-norm boundaries are fully pinned.
5. SuiteSparse threshold candidates should be deferred to Days 8-9 unless Day
   7 only records support-tier promotion gates.

## Day 7 Acceptance Gate

Day 7 may implement a threshold-family fixture only if all of the following
are true:

1. Fixture key, matrix family, parameters, thresholds, and expected ranks are
   pinned.
2. Perturbation and threshold values are separated enough to avoid roundoff
   ambiguity.
3. Helper output or exact derivation exists before C assertions are added.
4. Product diagnostics include relative threshold, absolute threshold,
   expected rank, product rank, rank-info rank, and R diagonal magnitudes.
5. Evidence is described as fixture-local threshold behavior only.
6. The artifact states that no global QR rank-threshold, default-threshold, or
   numerical-rank policy is introduced.
7. The fixture cannot be read as residual, nullspace, subspace, minimum-norm,
   pseudoinverse, Q/economy, sparse-mode, backend, SuiteSparse corpus,
   platform, performance, or external-library parity evidence.
8. Focused helper/test commands and required quality gates are known before
   editing source.

If any gate fails, Day 7 should explicitly defer the candidate and record the
future owner, blocker, and promotion gate.

## Deferred Promotion Gates

| Deferred Family | Future Owner | Promotion Gate |
| --- | --- | --- |
| Wide threshold family | Threshold/subspace/minimum-norm owner | Pin rank and nullity for each threshold, define output semantics, and preserve underdetermined minimum-norm non-claims. |
| SuiteSparse threshold family | SuiteSparse corpus owner | Define matrix support tier, expected-rank metadata, optional-data behavior, runtime budget, platform diagnostics, skip/fail behavior, and validation. |
| Default-threshold evidence | QR rank-policy owner | Define the exact product-local default-threshold claim and prove it will not be documented as a global numerical-rank policy. |
| Near-threshold nullspace/subspace | Threshold/subspace owner | Complete threshold-family expected-rank metadata first, then define threshold-specific nullity and projection metrics. |
| Raw R-diagonal ordering claim | Future deterministic factorization owner | Prove deterministic pivot/order semantics before relying on raw ordering beyond diagnostics. |

## Non-Claim Register

Day 6 does not claim:

- a global QR rank-threshold, default-threshold, or numerical-rank policy;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package threshold
  parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, residual, nullspace, subspace, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, platform, or performance parity;
- minimum-norm optimality, solution uniqueness, solution-selection policy,
  pseudoinverse behavior, QR-vs-SVD oracle behavior, COLAMD behavior,
  fallback behavior, or refinement behavior;
- SuiteSparse corpus correctness, optional-data behavior, optional-large
  support, platform support, or runtime behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 6 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_127`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed for Day 6, so no new code quality gate is required.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Perturbation sizes and thresholds are separated enough for stable claims. | Complete as policy | See perturbation and separation rules plus the Day 7 acceptance gate. |
| Each candidate has fixture-local expected-rank semantics. | Complete as policy | See candidate matrix, required metadata, diagnostics policy, and Day 7 candidate order. |
| No global default-threshold or numerical-rank policy claim is introduced. | Complete | See threshold semantics, non-claim register, and deferred promotion gates. |
