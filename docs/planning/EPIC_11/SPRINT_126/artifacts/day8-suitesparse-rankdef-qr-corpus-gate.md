# Sprint 126 Day 8 SuiteSparse Rank-Deficient QR Corpus Gate

## Decision

Day 8 keeps SuiteSparse rank-deficient QR corpus evidence behind an explicit
metadata gate and does not accept a new corpus fixture today.

The checked-in SuiteSparse matrices that already exercise QR are still
full-rank controls under the current product tests. The remaining checked-in
SuiteSparse matrices do not have pinned rank-deficient QR metadata, threshold
semantics, support-tier behavior, or focused QR diagnostics. Day 9 may proceed
only if it can satisfy the metadata protocol below before registering evidence;
otherwise it must explicitly defer Project Plan Item 5.

This artifact is policy-only. It does not change test registration, Matrix
Market fixtures, optional-data behavior, public solver wording, or maintainer
claims.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` Day 8 | Defines the SuiteSparse QR corpus gate deliverables. |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 126 Item 5 | Requires SuiteSparse rank-deficient QR corpus evidence to be added or explicitly deferred only after metadata, support tier, diagnostics, skip behavior, and validation are explicit. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day8-suitesparse-rankdef-qr-policy.md` | Provides the prior SuiteSparse QR corpus policy. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day9-suitesparse-rankdef-qr-decision.md` | Records the prior deferral and current full-rank QR controls. |
| `tests/data/suitesparse/*.mtx` | Checked-in Matrix Market corpus inventory. |
| `tests/test_qr.c` | Current QR rank, threshold, nullspace, economy, sparse-mode, refine, and SuiteSparse control owners. |
| `tests/test_qr_solve.c` | Current QR solve and SuiteSparse QR solve control owners. |
| `tests/test_suitesparse.c` | Existing SuiteSparse support-tier and `SPARSE_TEST_LARGE=1` conventions. |

## Corpus Inventory

| Matrix | Size and nnz | Current QR or SuiteSparse use | Support tier | Day 9 disposition |
| --- | ---: | --- | --- | --- |
| `west0067.mtx` | 67 x 67, 294 nnz | QR solve control and sparse-mode QR comparison; prior focused run reported rank `67`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `nos4.mtx` | 100 x 100, 347 nnz | QR solve, QR-vs-LU, refine, economy, reorder/fillin, sparse-mode control; prior focused run reported rank `100`. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `bcsstk04.mtx` | 132 x 132, 1890 nnz | QR reconstruction, solve, and sparse-mode comparison; product test asserts full rank. | Default checked-in | Control only. Do not relabel as rank-deficient. |
| `steam1.mtx` | 240 x 240, 3762 nnz | SuiteSparse LU/condition and COLAMD corpus; no current QR rank-deficient owner. | Default checked-in | Candidate only after expected-rank metadata and focused QR diagnostics are pinned. |
| `fs_541_1.mtx` | 541 x 541, 4285 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Optional candidate only after expected-rank metadata, runtime budget, skip proof, and diagnostics are pinned. |
| `orsirr_1.mtx` | 1030 x 1030, 6858 nnz | Existing large SuiteSparse lane. | Optional large, `SPARSE_TEST_LARGE=1` | Optional candidate only after expected-rank metadata, runtime budget, skip proof, and diagnostics are pinned. |
| `bcsstk14.mtx` | 1806 x 1806, 32630 nnz | Reorder, eigen, and direct-solver style corpus paths. | Report-only for QR rank-deficient evidence | Too large for default QR rank-deficient evidence without a separate support-tier decision. |
| `s3rmt3m3.mtx` | 5357 x 5357, 106526 nnz | Reorder/corpus style paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `Kuu.mtx` | 7102 x 7102, 173651 nnz | Reorder/corpus style paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `bloweybq.mtx` | 10001 x 10001, 39996 nnz | No current QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `Pres_Poisson.mtx` | 14822 x 14822, 365313 nnz | Reorder/corpus style paths; no QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |
| `tuma1.mtx` | 22967 x 22967, 50560 nnz | No current QR rank-deficient owner. | Report-only | Not eligible for Day 9 default evidence. |

## Expected-Rank Metadata Protocol

Accepted SuiteSparse rank-deficient QR evidence must have all of the following
before code registration:

1. Matrix path, shape, nnz, and support tier.
2. Claim type: rank-only, threshold/rank transition, residual, reconstruction,
   or nullspace/subspace. Day 9 should prefer rank-only unless a stronger
   metric has independent metadata.
3. Expected rank and nullity, or an explicit list of threshold/rank pairs.
4. Threshold semantics, including relative threshold, computed absolute
   threshold, and how the largest `|R_ii|` scale is interpreted.
5. Independent source for the expected rank metadata. The product QR result
   alone may be a diagnostic, but it is not enough to create the expected
   value for a new corpus claim.
6. Factorization status expectations and failure interpretation.
7. Diagnostics: matrix key, support tier, optional gate state, load status,
   factorization status, `qr.rank`, `sparse_qr_rank()` or
   `sparse_qr_rank_info()` values, absolute threshold, representative `R`
   diagonal magnitudes or rank-transition summary, and residual or
   reconstruction metrics if claimed.
8. Validation commands for focused and full quality gates.

If any field is missing, the evidence must be deferred rather than converted
into a weaker or ambiguous claim.

## Support-Tier And Skip Policy

| Tier | Rule |
| --- | --- |
| Default checked-in | Missing data is a failure. Numerical disagreement after acceptance is a failure. |
| Optional large | Must be gated, preferably with the existing `SPARSE_TEST_LARGE=1` convention unless Day 9 defines a narrower QR-specific gate. Missing data may skip only with a message naming the matrix, gate, and owner. |
| Report-only | May not enter default CI as rank-deficient QR evidence without a support-tier promotion decision. |
| Platform | Pure C SuiteSparse QR tests should not inherit external-helper Windows skips. Any platform skip must name the concrete blocker. |

Accepted optional tests must prove missing-data skip behavior separately from
numeric failure behavior. A matrix that loads and factors but disagrees with
pinned rank metadata must fail, not skip.

## Candidate Decision

No bounded SuiteSparse rank-deficient QR evidence batch is accepted on Day 8.

Day 9 has two valid paths:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Promote a bounded candidate | Provide an independent expected-rank source and satisfy the metadata, support-tier, diagnostics, skip, and validation protocol above before code changes. | Add one narrowly named fixture-local test and record focused plus full validation. |
| Explicitly defer | Show no candidate satisfies the Day 8 metadata protocol. | Preserve existing SuiteSparse matrices as full-rank or non-QR controls and record a future-owner promotion gate. |

The most plausible Day 9 investigation order is:

1. Reconfirm `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` remain controls,
   not rank-deficient candidates.
2. Consider `steam1.mtx` only as a default checked-in candidate if expected
   rank metadata exists independently of product QR.
3. Keep `fs_541_1.mtx` and `orsirr_1.mtx` optional-large until runtime,
   expected-rank, and skip behavior are proven.
4. Keep report-only matrices out of Day 9 default evidence.

## Evidence Boundaries

Day 8 proves only that Sprint 126 has an explicit gate for SuiteSparse
rank-deficient QR corpus evidence. It does not prove:

- a SuiteSparse rank-deficient QR fixture;
- broad SuiteSparse corpus correctness;
- broad QR rank-deficient behavior;
- global QR rank-threshold policy;
- LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, vendor-backend,
  dense-library, ecosystem, or external package parity;
- minimum-norm, pseudoinverse, nullspace, Q-basis, economy, sparse-mode,
  reorder, backend, platform, performance, scalability, memory, package, ABI,
  public API, CI, CMake, or Makefile behavior.

## Validation

Day 8 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 5 has explicit acceptance or deferral criteria. | Complete | Metadata protocol, support-tier policy, and Day 9 candidate decision define the gate. |
| Optional corpus behavior is deterministic and documented. | Complete | Default, optional-large, report-only, missing-data, platform, and numerical-failure rules are separated. |
| No broad SuiteSparse corpus or platform support claim is introduced. | Complete | Existing matrices remain controls unless Day 9 satisfies the fixture-local metadata protocol. |
