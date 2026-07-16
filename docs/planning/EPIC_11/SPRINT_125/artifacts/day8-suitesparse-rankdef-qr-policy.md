# Sprint 125 Day 8 SuiteSparse Rank-Deficient QR Corpus Policy

## Purpose

Define bounded SuiteSparse corpus rules before accepting or deferring
rank-deficient QR evidence on Day 9.

This artifact is policy-only. It does not add SuiteSparse QR evidence, change
test registration, change platform behavior, or update public solver claims.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `tests/data/suitesparse/*.mtx` | Available checked-in Matrix Market corpus. |
| `tests/test_suitesparse.c` | Existing SuiteSparse support tiers, large-test gate, and skip convention. |
| `tests/test_qr.c` | Current QR rank, sparse-mode, refine, reorder, and external-reference owners. |
| `tests/test_qr_solve.c` | Current SuiteSparse QR solve and reconstruction owners. |
| `docs/maintainer_guide.md` | Maintainer-facing evidence and non-claim wording. |
| Sprint 125 Day 7 artifact | Threshold evidence boundaries and SuiteSparse deferral gate. |

## Corpus Inventory

| Matrix | Size and nnz | Current QR use | Support tier | Day 9 disposition |
| --- | ---: | --- | --- | --- |
| `west0067.mtx` | 67 x 67, 294 nnz | QR solve residual; sparse-mode QR comparison | Default checked-in | Eligible only for named residual/rank diagnostics if expected rank is pinned. No rank-deficient claim today. |
| `nos4.mtx` | 100 x 100, 347 nnz | QR solve, QR-vs-LU, refine, economy, reorder/fillin | Default checked-in | Control matrix only unless future diagnostics prove a rank-deficient or near-threshold expectation. |
| `bcsstk04.mtx` | 132 x 132, 1890 nnz | QR reconstruction, solve residual, sparse-mode comparison; current test asserts full rank | Default checked-in | Full-rank control only. Do not reuse as rank-deficient evidence without changing the claim and expectation. |
| `steam1.mtx` | 240 x 240, 3762 nnz | SuiteSparse LU/condition corpus; no current QR rank-deficient owner | Default checked-in | Possible future smoke candidate, but not a first rank-deficient QR proof without expected-rank metadata. |
| `fs_541_1.mtx` | 541 x 541, 4285 nnz | Existing SuiteSparse large lane | Optional large, `SPARSE_TEST_LARGE=1` | Optional/report-only unless Day 9 proves runtime and expected-rank stability. |
| `orsirr_1.mtx` | 1030 x 1030, 6858 nnz | Existing SuiteSparse large lane | Optional large, `SPARSE_TEST_LARGE=1` | Optional/report-only unless Day 9 proves runtime and expected-rank stability. |
| `bcsstk14.mtx` | 1806 x 1806, 32630 nnz | Used by other reorder/eigen/Cholesky style paths | Optional/report-only | Too large for default rank-deficient QR evidence without a separate support-tier decision. |
| `Kuu.mtx` | 7102 x 7102, 173651 nnz | No current QR rank-deficient owner | Report-only | Not eligible for default Day 9 evidence. |
| `s3rmt3m3.mtx` | 5357 x 5357, 106526 nnz | No current QR rank-deficient owner | Report-only | Not eligible for default Day 9 evidence. |
| `Pres_Poisson.mtx` | 14822 x 14822, 365313 nnz | No current QR rank-deficient owner | Report-only | Not eligible for default Day 9 evidence. |
| `bloweybq.mtx` | 10001 x 10001, 39996 nnz | No current QR rank-deficient owner | Report-only | Not eligible for default Day 9 evidence. |
| `tuma1.mtx` | 22967 x 22967, 50560 nnz | No current QR rank-deficient owner | Report-only | Not eligible for default Day 9 evidence. |

## Candidate Policy

The checked-in corpus does not currently contain a documented, small,
rank-deficient SuiteSparse QR fixture with pinned expected rank, threshold,
nullity, and residual semantics.

Day 9 may accept SuiteSparse rank-deficient QR evidence only if it can name all
of the following before test registration:

- matrix path and support tier
- expected rank or expected threshold/rank pairs
- threshold semantics and absolute threshold diagnostics
- factorization status expectations
- residual or reconstruction metric, if the evidence is not rank-only
- skip behavior for optional data or platform limitations
- focused validation command and full quality gate requirements

If no candidate satisfies those requirements, Day 9 should explicitly defer
SuiteSparse rank-deficient QR evidence rather than weakening the claim boundary
or relabeling full-rank corpus controls.

## Optional Corpus And Skip Policy

- Default checked-in matrices may be required by normal CI when the test is
  already in the default suite. Missing default data is a test failure, not a
  skip.
- Optional large or report-only matrices must be gated. The existing
  `SPARSE_TEST_LARGE=1` convention remains the preferred gate for large
  SuiteSparse paths unless a narrower QR-specific gate is introduced with its
  own diagnostics.
- Missing optional data is a skip only for explicitly optional tests. The skip
  message must name the matrix path, gate, and owner.
- Platform skips must be narrow. Existing external-reference helper tests skip
  on Windows because the helper path is not enabled there; pure C SuiteSparse
  QR tests should not inherit that skip unless the new code path proves a
  platform-specific blocker.
- Numerical failures are failures, not skips, once the matrix, threshold,
  tolerance, and support tier are accepted.

## Required Diagnostics

Accepted SuiteSparse QR rank-deficient or near-threshold evidence must print or
otherwise expose enough context to reproduce failures:

- matrix key, path, shape, and nnz
- support tier and optional gate state
- load status and factorization status
- `qr.rank` and any `sparse_qr_rank()` or `sparse_qr_rank_info()` result
- relative tolerance and computed absolute threshold for threshold evidence
- relevant `R` diagonal magnitudes or rank-transition summary
- residual or reconstruction metric when the claim includes solve or
  reconstruction behavior
- skip reason for optional-data or platform skips

## Evidence Boundaries

SuiteSparse corpus evidence may prove only fixture-local behavior for the named
matrix and named threshold or tolerance. It can show that the product QR code
loads, factors, and reports a bounded rank/residual/reconstruction result for
that fixture.

It must not be described as evidence for:

- broad SuiteSparse corpus correctness
- LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, or backend parity
- portable performance, scalability, or memory behavior
- global QR rank-threshold policy
- broad rank-deficient QR behavior
- raw nullspace or Q-basis orientation
- minimum-norm or pseudoinverse behavior
- package, ABI, public API, or platform support beyond the exercised tier

## Day 9 Checklist

1. Re-check the available corpus and current QR owners.
2. Select a single bounded candidate only if expected rank or threshold
   behavior is pinned before implementation.
3. Keep default, optional-large, and report-only matrices in separate support
   tiers.
4. Add diagnostics before assertions so CI failures identify matrix, threshold,
   rank, and residual context.
5. If code or headers change, run `make format && make lint && make test`.
6. If Day 9 is documentation-only deferral, run `git diff --check` and the
   focused markdown whitespace scan.

## Validation

Day 8 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 5 has bounded corpus rules. | Complete | See corpus inventory, candidate policy, and support tiers. |
| Missing optional data and platform skips are not treated as failures. | Complete | Optional skip policy separates default data failures from gated optional skips. |
| Broad SuiteSparse and backend parity claims remain fenced. | Complete | See evidence boundaries and non-claims. |
