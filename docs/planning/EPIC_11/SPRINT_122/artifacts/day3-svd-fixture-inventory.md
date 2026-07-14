# Sprint 122 Day 3 SVD External Fixture Decision Inventory

## Purpose

Day 3 inventories candidate SVD external fixtures before Day 4 decides whether
to add one bounded fixture or explicitly defer additional SVD external evidence.
The inventory keeps the Sprint 121 `svd_rect_fullrank_6x4` pilot as the baseline
and filters candidates against existing deterministic SVD, rank, low-rank,
partial-SVD, pseudoinverse, and condition-number coverage.

## Existing External Pilot Summary

| Field | Current Pilot |
| --- | --- |
| Fixture key | `svd_rect_fullrank_6x4` |
| Matrix shape | 6x4 dense rectangular, full column rank, mixed signs, non-diagonal |
| Reference path | `tests/svd_external_dense_reference.py` computes `A^T A`, runs bounded Jacobi eigenvalue iteration, and emits singular values. |
| Product path | `tests/test_svd.c` runs full SVD and compares four singular values. |
| Tolerance | Max absolute singular-value difference below `1e-8`. |
| Dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, or BLAS dependency. |
| Skip behavior | Existing external-reference helper skips missing `python3`; Windows skip remains explicit. |
| Test registration impact | None; the pilot lives inside existing `test_svd`. |
| Trust boundary | Independent dense arithmetic for one fixed small singular-value fixture only. |

The pilot does not compare singular vectors, subspaces, partial SVD, low-rank
approximations, pseudoinverse identities, QR behavior, performance, platform
support, package support, or broad external dense-library parity.

## Existing Internal Coverage Filter

| Coverage Area | Sprint 121 / Current Owner | Day 3 Filter |
| --- | --- | --- |
| Exact diagonal spectra | `test_svd_basic_sigma`, diagonal SVD/rank/condition tests | Do not add a diagonal-only external fixture unless it exercises a new external-reference failure mode. |
| Rectangular full-rank SVD | `svd_rect_fullrank_6x4` external pilot plus tall/wide internal tests | Do not add another full-rank rectangular external fixture with similar conditioning. |
| Exact rank-deficient SVD/rank | duplicate-column and rank-threshold internal tests | Candidate only if the external fixture independently checks singular values for a non-diagonal exact-rank-deficient matrix. |
| Near-singular or threshold rank | internal rank and condition-number tests | Defer unless threshold policy, expected small singular values, and reference roundoff are explicitly pinned. |
| Partial-SVD top-k and vectors | `tests/test_svd_partial_helpers.h` internal full-SVD and vector residual checks | Defer to Days 7-8; do not mix partial-SVD parity into full-SVD fixture expansion. |
| Low-rank dense/sparse output | low-rank deterministic and dense/sparse consistency tests | Reject for Day 3; low-rank optimality is not a singular-value external fixture. |
| Pseudoinverse and minimum-norm | pseudoinverse and QR/minimum-norm deterministic fixtures | Reject for Day 3; pseudoinverse identities and minimum-norm behavior have separate owners. |
| SuiteSparse smoke fixtures | bounded load/smoke tests | Reject for Day 3; optional corpus fixtures would broaden scope and platform variability. |
| Singular-vector/subspace evidence | internal reconstruction and orthogonality checks | Reject for Day 3; vector/subspace parity needs separate sign and basis semantics. |

## Candidate SVD External Fixture Table

| Candidate Fixture | Adds New Evidence? | Risk | Day 3 Disposition |
| --- | --- | --- | --- |
| `svd_rankdef_duplicate_5x4_external_sigma` | Yes. Adds independent singular-value evidence for a small non-diagonal exact-rank-deficient rectangular matrix with one zero singular value. | Moderate: zero singular value requires clear tolerance and failure interpretation. | Strongest Day 4 candidate. |
| `svd_wide_fullrank_4x6_external_sigma` | Partial. Adds wide shape, but existing Python `A^T A` path would emit extra zero singular values unless the fixture contract is redesigned around `min(m,n)`. | High: output length and zero-padding semantics could hide shape assumptions. | Defer unless a future wide-shape external owner designs output semantics. |
| `svd_near_dependent_5x4_external_sigma` | Partial. Could add threshold-sensitive singular-value evidence. | High: near-zero sigma interpretation overlaps rank policy and condition-number tolerance. | Defer until rank-threshold external policy is explicit. |
| `svd_diag_repeated_5x5_external_sigma` | Low. Analytical diagonal repeated-spectrum behavior is already deterministic and does not need an external process. | Low implementation risk but high duplication. | Reject as duplicate. |
| `svd_lowrank_outer_product_external_sigma` | Low to moderate. External singular values would duplicate known rank-k construction while not proving low-rank output optimality. | Moderate: can blur low-rank claims. | Reject for Day 3; keep low-rank under low-rank owners. |
| `svd_suite_sparse_external_sigma` | Potentially broad. | High: optional files, platform variability, runtime, and corpus interpretation. | Reject for Sprint 122 SVD fixture expansion. |
| `svd_vector_subspace_external_check` | Not a singular-value fixture. | High: sign, basis, repeated-spectrum, and subspace semantics need separate design. | Reject for Day 3; keep vector/subspace out of scope. |

## Preferred Day 4 Candidate

Day 3 recommends that Day 4 either implement or explicitly defer one bounded
candidate:

`svd_rankdef_duplicate_5x4_external_sigma`

Candidate fixture intent:

- Matrix shape: 5x4 dense rectangular.
- Rank model: exact rank deficient, expected rank 3.
- Structure: non-diagonal rows with one duplicate or linearly dependent column.
- Expected singular-value shape: three positive singular values and one zero or
  near-zero singular value from the pure-Python reference.
- Comparison: full SVD singular values only.
- Tolerance: positive singular values within `1e-8` max absolute difference;
  smallest singular value accepted below a separate zero tolerance such as
  `1e-8` so rank-threshold policy is not silently asserted.
- Build impact: keep inside existing `test_svd`; no Makefile, CMake, or CTest
  membership change.
- Dependency policy: Python standard library only.

This candidate adds evidence because the current pilot covers full column rank
only. It still remains bounded and does not become a broad rank-deficient SVD
parity claim.

## Duplicate and Non-Claim Filter

| Filter | Required Day 4 Handling |
| --- | --- |
| Full-rank rectangular duplication | Do not add another fixture with the same full-rank mixed dense shape as `svd_rect_fullrank_6x4`. |
| Analytical diagonal duplication | Do not use external process work for diagonal spectra already proven by direct expected values. |
| Internal partial-SVD duplication | Do not call internal full-SVD partial comparisons external parity. |
| Low-rank optimality drift | Do not use singular-value fixture evidence to claim low-rank global optimality. |
| Pseudoinverse drift | Do not use SVD singular-value fixture evidence to claim additional Moore-Penrose or minimum-norm behavior. |
| External package drift | Do not introduce NumPy, SciPy, LAPACK, BLAS, or platform package assumptions. |
| Public wording drift | Do not update solver-selection or README wording from this fixture alone. |

## SVD Fixture Decision Criteria

| Criterion | Required Answer Before Day 4 Code Change |
| --- | --- |
| Fixture uniqueness | Does the candidate prove a singular-value behavior not already covered by Sprint 121 deterministic tests or `svd_rect_fullrank_6x4`? |
| Fixture size | Is the matrix small enough for deterministic C and pure-Python reference execution in the existing unit-test lane? |
| Reference trust boundary | Is the reference independent enough for a bounded check without depending on NumPy, SciPy, LAPACK, BLAS, or external data? |
| Output shape | Does the helper emit exactly the singular values compared by the C test, including zero-sigma handling? |
| Tolerance ownership | Are positive and near-zero singular-value tolerances explicit and separate from rank policy claims? |
| Skip behavior | Are missing `python3`, Windows skip, and helper `ERROR` output paths clear and consistent with existing external-reference tests? |
| Failure interpretation | Will a failure identify fixture key, reference-read status, product SVD status, singular-value index, and max difference? |
| Test-surface accounting | Does the change avoid Makefile, CMake, and CTest registration updates unless Day 4 explicitly accepts those impacts? |
| Non-claims | Does the artifact preserve no LAPACK, SciPy, NumPy, broad dense-library, singular-vector, subspace, partial-SVD, low-rank, pseudoinverse, performance, package, ABI, platform, public API, and state-of-the-art claims? |

## Day 4 Decision Checklist

Day 4 should choose exactly one of these outcomes:

| Outcome | Required Day 4 Evidence |
| --- | --- |
| Implement `svd_rankdef_duplicate_5x4_external_sigma` | Add the Python fixture and one `test_svd` case; run `make format && make build/test_svd && ./build/test_svd && make lint && make test`; record no build-membership change. |
| Defer additional SVD external fixture work | Record why the strongest candidate does not clear uniqueness, tolerance, skip, failure, or scope gates; keep residual handoff explicit. |
| Reject additional SVD external fixture work for Epic 11 | Record why current external SVD evidence is sufficient for this epic and why remaining candidates belong to future oracle/corpus work. |

## Validation Notes

Day 3 changed documentation only. Required validation is `git diff --check` and
a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| SVD candidates are evaluated against existing Sprint 121 coverage. | Complete | See existing internal coverage filter and candidate table. |
| Any proposed external fixture has explicit trust boundaries. | Complete | See preferred Day 4 candidate and decision criteria. |
| No broad SVD external-library parity claim is introduced. | Complete | See duplicate and non-claim filter. |
