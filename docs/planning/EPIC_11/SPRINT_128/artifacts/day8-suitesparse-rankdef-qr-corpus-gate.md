# Sprint 128 Day 8 SuiteSparse Rank-Deficient QR Corpus Gate

## Decision

Day 8 keeps SuiteSparse rank-deficient QR corpus evidence behind an explicit
metadata, support-tier, skip, diagnostic, and runtime gate. No new
SuiteSparse QR corpus fixture is accepted today.

The current checked-in SuiteSparse matrices that already exercise QR remain
controls, not rank-deficient corpus evidence. The remaining default,
optional-large, and report-only matrices still lack independent expected-rank
or threshold/rank metadata for a QR rank-deficient claim.

This artifact is policy-only. It does not change C source, headers, Python
helpers, Matrix Market data, test registration, optional-data gates, public
solver wording, or maintainer claims.

## Inputs Reviewed

| Input | Day 8 Use |
| --- | --- |
| Sprint 128 Plan Day 8 | Defines SuiteSparse rank-deficient QR corpus gate deliverables. |
| Sprint 128 Day 1 artifact | Provides duplicate fences for existing SuiteSparse controls, deferred corpus evidence, and optional-large lanes. |
| Sprint 128 Days 6-7 artifacts | Keep SuiteSparse threshold candidates behind corpus support-tier and expected-rank gates. |
| Sprint 126 Day 8 and Sprint 127 Day 8 artifacts | Provide the prior metadata gates, support-tier policies, and explicit no-acceptance decisions. |
| Sprint 127 Day 9 artifact | Records the prior explicit deferral and preserves `west0067`, `nos4`, and `bcsstk04` as controls. |
| `tests/data/suitesparse/*.mtx` | Current checked-in Matrix Market inventory. |
| `tests/test_qr.c` | Current QR rank, threshold, nullspace, economy, sparse-mode, refine, reorder, and SuiteSparse-adjacent control owner. |
| `tests/test_qr_solve.c` | Current QR solve and SuiteSparse QR solve control owner. |
| `tests/test_suitesparse.c` | Current SuiteSparse default and `SPARSE_TEST_LARGE=1` optional-large convention owner. |

## Corpus Inventory

| Matrix | Size and nnz | Current QR or SuiteSparse use | Support tier | Day 9 disposition |
| --- | ---: | --- | --- | --- |
| `west0067.mtx` | 67 x 67, 294 nnz | QR solve control and sparse-mode QR comparison; prior diagnostics reported rank `67`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `nos4.mtx` | 100 x 100, 347 nnz | QR solve, QR-vs-LU, refine, economy, reorder/fillin, and sparse-mode control; prior diagnostics reported rank `100`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `bcsstk04.mtx` | 132 x 132, 1890 nnz | QR reconstruction, QR solve, and sparse-mode comparison; product tests assert full-rank behavior. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `steam1.mtx` | 240 x 240, 3762 nnz | SuiteSparse LU, condition, COLAMD, and iterative corpus; no current QR rank-deficient owner. | Default checked-in | Deferred until independent expected-rank metadata and focused QR diagnostics exist. |
| `fs_541_1.mtx` | 541 x 541, 4285 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Deferred until expected-rank metadata, QR runtime budget, and skip proof exist. |
| `orsirr_1.mtx` | 1030 x 1030, 6858 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Deferred until expected-rank metadata, QR runtime budget, and skip proof exist. |
| `bcsstk14.mtx` | 1806 x 1806, 32630 nnz | Reorder, eigensolver, and direct-solver corpus paths. | Report-only for QR rank-deficient evidence | Not eligible for default evidence without support-tier promotion and independent rank metadata. |
| `s3rmt3m3.mtx` | 5357 x 5357, 106526 nnz | Reorder/corpus paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `Kuu.mtx` | 7102 x 7102, 173651 nnz | Reorder/corpus paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `bloweybq.mtx` | 10001 x 10001, 39996 nnz | No current QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `Pres_Poisson.mtx` | 14822 x 14822, 365313 nnz | Reorder/corpus paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `tuma1.mtx` | 22967 x 22967, 50560 nnz | No current QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |

## Expected-Rank Metadata Policy

Accepted SuiteSparse rank-deficient QR evidence must define all of the
following before code edits:

1. Matrix path, shape, nnz, and support tier.
2. Claim type: rank-only, threshold/rank transition, residual,
   reconstruction, nullspace/subspace, or minimum-norm-adjacent control.
3. Expected rank and nullity, or an explicit list of threshold/rank pairs.
4. Independent source for expected rank metadata. Product QR observations may
   be diagnostics, but they are not enough to create expected values.
5. Threshold semantics, including relative threshold, computed absolute
   threshold, and R-diagonal scale interpretation.
6. Factorization status expectations and failure interpretation.
7. Diagnostics: matrix key, support tier, optional gate state, load status,
   factorization status, `qr.rank`, `sparse_qr_rank()` or
   `sparse_qr_rank_info()` results, absolute threshold, representative R
   diagonal magnitudes or rank-transition summary, and residual or
   reconstruction metrics when claimed.
8. Validation commands for focused and full quality gates.

If any field is missing, Day 9 must defer rather than converting product
observations into an ambiguous corpus claim.

## Threshold Policy

SuiteSparse QR corpus threshold evidence must follow the product threshold
semantics used by current QR tests:

- explicit `tol > 0` maps to absolute threshold `tol * abs(R(0,0))`;
- `tol <= 0` is product-default behavior and cannot become a global
  numerical-rank policy;
- expected rank must be pinned independently for every named threshold;
- threshold/rank evidence is not residual, subspace, minimum-norm, Q/economy,
  sparse-mode, reorder, backend, platform, or performance evidence.

## Support-Tier, Skip, Diagnostic, And Runtime Policy

| Tier | Rule |
| --- | --- |
| Default checked-in | Missing data is a failure. Once evidence is accepted, numerical disagreement is a failure. Added default runtime must fit normal `make test`. |
| Optional large | Must remain behind `SPARSE_TEST_LARGE=1` or a narrower explicit QR gate. Missing data may skip only with a message naming the matrix, gate, and owner. |
| Report-only | May be inventoried, but may not enter default CI as rank-deficient QR evidence without support-tier promotion and runtime budget. |
| Platform | Pure C SuiteSparse QR tests should not inherit external-helper skips. Any platform skip must name a concrete blocker. |

Optional-data skip behavior must be proven separately from numerical failure
behavior. If a matrix is present and has accepted expected-rank metadata, a
rank mismatch must fail rather than skip.

## Candidate Decision

No bounded SuiteSparse rank-deficient QR implementation candidate is accepted
on Day 8.

Day 9 has two valid paths:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Promote one bounded candidate | Provide independent expected-rank or threshold/rank metadata and satisfy support-tier, diagnostics, runtime, skip, and validation policies before code edits. | Add one narrowly named fixture-local test. |
| Explicitly defer | Show no candidate satisfies the Day 8 metadata protocol. | Preserve current SuiteSparse matrices as controls and record a future-owner promotion gate. |

Day 9 should investigate in this order:

1. Reconfirm `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` remain controls.
2. Consider `steam1.mtx` only if independent expected-rank metadata exists.
3. Keep `fs_541_1.mtx` and `orsirr_1.mtx` optional-large until runtime,
   expected-rank, and skip behavior are proven.
4. Keep report-only matrices out of default QR rank-deficient evidence.

## Day 9 Checklist

Before implementation, Day 9 must answer:

1. What exact matrix path and support tier is being claimed?
2. What independent expected-rank or threshold/rank metadata exists?
3. What explicit threshold semantics and absolute-threshold diagnostics will
   be printed?
4. What runtime budget and optional-data behavior apply?
5. What focused validation command proves the exact executable path?
6. What non-claims prevent corpus-wide, platform, performance, threshold,
   residual, nullspace/subspace, minimum-norm, Q/economy, sparse-mode,
   backend, and external-library parity interpretations?

If any answer is missing, Day 9 should explicitly defer.

## Evidence Boundaries

Day 8 proves only that Sprint 128 has a current gate for SuiteSparse
rank-deficient QR corpus evidence. It does not prove:

- a SuiteSparse rank-deficient QR fixture;
- broad SuiteSparse corpus correctness;
- broad QR rank-deficient behavior;
- global QR rank-threshold, default-threshold, or numerical-rank behavior;
- LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity;
- residual, compatible-solve, minimum-norm, pseudoinverse, nullspace,
  subspace, Q-basis, economy, sparse-mode, reorder, backend, platform,
  performance, scalability, memory, package, ABI, public API, CI, CMake, or
  Makefile behavior.

## Validation Notes

Day 8 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, Matrix Market,
optional-data, or public wording files changed for Day 8, so no new code
quality gate is required. The current branch already passed
`make format && make lint && make test` after Day 7's code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 5 has an explicit implementation or deferral path. | Complete | Candidate decision allows only metadata-backed promotion or explicit deferral on Day 9. |
| No SuiteSparse fixture is accepted without independent expected-rank metadata. | Complete | Expected-rank metadata policy rejects product QR diagnostics as sole expected-value source. |
| Missing optional data and runtime limits have deterministic behavior. | Complete | Support-tier, skip, diagnostic, and runtime policy separates default, optional-large, report-only, platform, and numerical-failure paths. |
