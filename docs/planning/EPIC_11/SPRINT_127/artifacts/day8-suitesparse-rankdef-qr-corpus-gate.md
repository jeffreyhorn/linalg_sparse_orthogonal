# Sprint 127 Day 8 SuiteSparse Rank-Deficient QR Corpus Gate

## Decision

Day 8 keeps SuiteSparse rank-deficient QR corpus evidence behind an explicit
metadata and support-tier gate. No new SuiteSparse QR corpus fixture is
accepted today.

The checked-in SuiteSparse matrices that already exercise QR remain full-rank
or non-rank-deficient controls. The remaining checked-in and report-only
matrices still lack independent expected-rank metadata, threshold/rank pairs,
rank-deficient claim type, runtime budget, optional-data behavior, and focused
QR diagnostics needed before Day 9 can register evidence.

This artifact is policy-only. It does not change C source, headers, Python
helpers, Matrix Market data, test registration, optional-data gates, public
solver wording, or maintainer claims.

## Inputs Reviewed

| Input | Day 8 Use |
| --- | --- |
| Sprint 127 Plan Day 8 | Requires SuiteSparse candidate inventory, expected-rank policy, support-tier decisions, skip/runtime behavior, diagnostics, and a Day 9 checklist. |
| Sprint 127 Day 1 artifact | Provides duplicate fences for Sprint 125-126 SuiteSparse QR deferrals and full-rank controls. |
| Sprint 127 Day 6-7 artifacts | Keep SuiteSparse threshold-family evidence behind corpus support-tier and expected-rank gates. |
| Sprint 125 Day 8-9 artifacts | Provide the original SuiteSparse rank-deficient QR corpus policy and explicit deferral. |
| Sprint 126 Day 8-9 artifacts | Provide the refreshed metadata gate and explicit deferral. |
| Sprint 126 Day 14 artifact | Preserves corpus support-tier non-claims and handoff requirements. |
| `tests/data/suitesparse/*.mtx` | Current checked-in Matrix Market corpus inventory. |
| `tests/test_qr.c` | Current QR rank, threshold, nullspace, economy, sparse-mode, refine, reorder, and SuiteSparse-adjacent control owner. |
| `tests/test_qr_solve.c` | Current QR solve and SuiteSparse QR solve control owner. |
| `tests/test_suitesparse.c` | Current SuiteSparse default and `SPARSE_TEST_LARGE=1` optional-large convention owner. |

## Corpus Inventory

| Matrix | Size and nnz | Current QR or SuiteSparse Use | Support Tier | Day 9 Disposition |
| --- | ---: | --- | --- | --- |
| `west0067.mtx` | 67 x 67, 294 nnz | QR solve control and sparse-mode QR comparison; prior focused diagnostics reported rank `67`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `nos4.mtx` | 100 x 100, 347 nnz | QR solve, QR-vs-LU, refine, economy, reorder/fillin, sparse-mode control; prior diagnostics reported rank `100`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `bcsstk04.mtx` | 132 x 132, 1890 nnz | QR reconstruction, QR solve, and sparse-mode comparison; product tests assert full-rank behavior. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `steam1.mtx` | 240 x 240, 3762 nnz | SuiteSparse LU/condition/COLAMD corpus; no current QR rank-deficient owner. | Default checked-in | Deferred until independent expected-rank metadata and QR diagnostics exist. |
| `fs_541_1.mtx` | 541 x 541, 4285 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Deferred until expected-rank metadata, QR runtime budget, and skip proof exist. |
| `orsirr_1.mtx` | 1030 x 1030, 6858 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Deferred until expected-rank metadata, QR runtime budget, and skip proof exist. |
| `bcsstk14.mtx` | 1806 x 1806, 32630 nnz | Reorder, eigen, and direct-solver corpus paths. | Report-only for QR rank-deficient evidence | Not eligible for Day 9 default evidence without support-tier promotion. |
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
   reconstruction, or nullspace/subspace.
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

If any field is missing, Day 9 must defer instead of converting product
observations into an ambiguous corpus claim.

## Support-Tier And Skip Policy

| Tier | Rule |
| --- | --- |
| Default checked-in | Missing data is a failure. Once evidence is accepted, numerical disagreement is a failure. |
| Optional large | Must be gated, preferably by the existing `SPARSE_TEST_LARGE=1` convention unless a narrower QR-specific gate is justified. Missing data may skip only with a message naming the matrix, gate, and owner. |
| Report-only | May not enter default CI as rank-deficient QR evidence without support-tier promotion and runtime budget. |
| Platform | Pure C SuiteSparse QR tests should not inherit external-helper skips. Any platform skip must name a concrete blocker. |

Optional-data skip behavior must be proven separately from numerical failure
behavior. If a matrix is present and has accepted expected-rank metadata, a
rank mismatch must fail rather than skip.

## Runtime Budget

| Candidate Tier | Day 9 Runtime Rule |
| --- | --- |
| Default checked-in, small controls | Focused QR diagnostics may run in default local validation if already part of existing executables. |
| Default checked-in, new rank-deficient candidate | Must record focused runtime expectation and keep the added default path small enough for normal `make test`. |
| Optional large | Must stay behind `SPARSE_TEST_LARGE=1` or an equally explicit QR gate, with missing-data skip proof and runtime note. |
| Report-only | May be inventoried, but not registered as Day 9 default evidence. |

## Candidate Decision

No bounded SuiteSparse rank-deficient QR implementation candidate is accepted
on Day 8. Day 9 has two valid paths:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Promote one bounded candidate | Provide independent expected-rank metadata and satisfy the support-tier, diagnostics, runtime, skip, and validation policies before code changes. | Add one narrowly named fixture-local test. |
| Explicitly defer | Show no candidate satisfies the Day 8 metadata protocol. | Preserve current SuiteSparse matrices as controls and record a future-owner promotion gate. |

Day 9 should investigate in this order:

1. Reconfirm `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` remain full-rank
   controls.
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
5. What focused validation command proves the exact path?
6. What non-claims prevent corpus-wide, platform, performance, threshold,
   minimum-norm, nullspace, Q/economy, sparse-mode, and external-library
   parity interpretations?

If any answer is missing, Day 9 should explicitly defer.

## Evidence Boundaries

Day 8 proves only that Sprint 127 has a current gate for SuiteSparse
rank-deficient QR corpus evidence. It does not prove:

- a SuiteSparse rank-deficient QR fixture;
- broad SuiteSparse corpus correctness;
- broad QR rank-deficient behavior;
- global QR rank-threshold, default-threshold, or numerical-rank behavior;
- LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity;
- minimum-norm, pseudoinverse, nullspace, subspace, Q-basis, economy,
  sparse-mode, reorder, backend, platform, performance, scalability, memory,
  package, ABI, public API, CI, CMake, or Makefile behavior.

## Validation Notes

Day 8 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_127`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed for Day 8, so no new code quality gate is required. The current
branch already passed `make format && make lint && make test` after Day 7's
code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No SuiteSparse QR corpus candidate proceeds without expected-rank metadata. | Complete | Candidate decision requires independent expected-rank or threshold/rank metadata before Day 9 code edits. |
| Runtime and missing-data behavior are explicit. | Complete | Support-tier, skip, and runtime policies separate default, optional-large, and report-only paths. |
| Corpus support wording remains bounded. | Complete | Evidence boundaries keep current matrices as controls and reject corpus-wide claims. |
