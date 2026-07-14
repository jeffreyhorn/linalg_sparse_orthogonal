# Sprint 123 Day 3 SVD External Fixture Batch Decision

## Purpose

Day 3 decides whether Sprint 123 should add another bounded full-SVD external
fixture batch after the Day 2 SVD taxonomy and trust model. The decision must
be specific enough for Day 4 to either implement the accepted work or publish a
complete deferral package without widening the external-reference claim.

This is a decision artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
3.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 123 Plan Day 3 | Requires an accepted or deferred SVD fixture batch decision, fixture protocol, future-owner handoff, non-claims, and validation checklist. |
| Sprint 123 Day 2 SVD trust model | Identifies wide full-rank singular-value output as the strongest next bounded SVD external fixture candidate. |
| Sprint 122 Day 4 SVD decision | Provides the completed rank-deficient fixture contract and duplicate fences for `svd_rect_fullrank_6x4` and `svd_rankdef_duplicate_5x4`. |
| `tests/svd_external_dense_reference.py` | Current Python standard-library reference helper and fixture dispatch surface. |
| `tests/test_svd.c` | Current full-SVD external-reference test owner and fixture-key allow-list. |
| `tests/test_svd_partial_helpers.h` | Current partial-SVD external fixture owner, used only to keep full-SVD and partial-SVD evidence separate. |

## Decision Summary

Accepted one bounded full-SVD external fixture for Day 4 implementation:

`svd_wide_fullrank_4x6`

No other Day 2 SVD candidate is accepted for Day 4. The accepted fixture adds
one missing external-reference shape class: full-SVD singular values for a
wide, full-row-rank dense matrix where the product must compare exactly
`min(m,n)` singular values.

## Accepted Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `svd_wide_fullrank_4x6` |
| Matrix shape | 4x6 dense rectangular, full row rank |
| Matrix class | Wide, mixed-sign, full-rank matrix with no intentionally repeated or tiny singular values |
| Reference path | Extend `tests/svd_external_dense_reference.py` with a deterministic matrix builder and fixture-key dispatch. |
| Product path | Extend `tests/test_svd.c` with a matching sparse matrix builder and a full-SVD singular-value comparison test. |
| Compared quantity | Singular values only. |
| Expected output count | Exactly `4`, equal to `min(4, 6)`. No column-count padding and no top-k truncation. |
| Positive singular-value tolerance | Max absolute difference below `1e-8`. |
| Tail policy | No zero-tail assertion is part of this fixture because the fixture is full row rank. |
| Dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, package, or external-data dependency. |
| Skip behavior | Preserve the existing external-reference helper behavior: missing `python3` may skip; Windows remains explicitly skipped. |
| Failure semantics | Reference helper `ERROR` output fails; output-count mismatch fails as a fixture protocol error; SVD compute failure fails; value mismatch fails as fixture-local disagreement only. |
| Build membership impact | None expected. The test remains inside existing `test_svd`; no new CTest target, Makefile test target, or CMake test member should be added. |

## Proposed Matrix

Day 4 should use this matrix unless implementation discovers a concrete
conditioning or reference-convergence problem:

```text
[ 2.0, -1.0,  0.5,  3.0, -2.0,  1.0 ]
[ 0.0,  4.0, -1.5,  2.0,  1.0, -0.5 ]
[ 3.0,  0.0,  2.5, -1.0,  0.0,  4.0 ]
[-2.0,  1.0,  3.0,  0.5,  2.0, -1.0 ]
```

The matrix is intentionally small and mixed-sign. The accepted evidence is the
shape/output-count protocol plus singular-value agreement, not any vector,
subspace, pseudoinverse, low-rank, or performance behavior.

## Affected Surface Matrix

| Surface | Day 4 Action |
| --- | --- |
| `tests/svd_external_dense_reference.py` | Add `build_svd_wide_fullrank_4x6`, route the fixture key, and confirm helper output starts with `OK 4`. |
| `tests/test_svd.c` | Add `build_svd_external_ref_wide_fullrank_4x6`, allow the fixture key, add `test_svd_external_dense_reference_wide_fullrank_4x6`, and register it in existing `test_svd`. |
| `tests/test_svd_partial_helpers.h` | No change. This is full-SVD singular-value evidence, not partial-SVD evidence. |
| Makefile | No expected change. |
| CMake / CTest | No expected change. |
| Public docs / API | No expected change. |
| Maintainer evidence tables | No Day 4 change unless a later Sprint 123 maintainer-evidence day refreshes them. |

## Deferred SVD Candidates

| Candidate | Disposition | Future Owner | Promotion Gate |
| --- | --- | --- | --- |
| Near-dependent threshold singular values | Deferred | Future SVD rank/threshold owner | Define positive, tiny, and tail buckets plus rank-threshold interpretation before implementation. |
| Repeated non-diagonal spectrum | Deferred | Future SVD value/subspace owner | Decide whether singular-value-only repetition evidence is worth adding without vector or subspace parity claims. |
| Rectangular low-rank tail-energy fixture | Deferred | Future low-rank approximation owner | Define tail-energy metric and separate it from low-rank output optimality claims. |
| Pseudoinverse singular-value threshold fixture | Deferred | Future pseudoinverse/minimum-norm owner | Define Moore-Penrose, threshold, and minimum-norm metrics before any external fixture is accepted. |
| SuiteSparse SVD external fixture | Rejected for Sprint 123 Day 4 | Future corpus/platform owner | Requires optional-corpus, runtime, platform, and broad-corpus interpretation policy. |
| Singular-vector or subspace external check | Deferred | Future vector/subspace owner | Define sign, orientation, repeated-spectrum, and subspace-angle policy. |

## Non-Claim Register

The accepted Day 3 batch does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad ecosystem parity;
- broad external dense-library SVD correctness;
- singular-vector, subspace, repeated-spectrum basis, or orientation parity;
- partial-SVD external vector, subspace, convergence, or residual parity;
- pseudoinverse, Moore-Penrose, or minimum-norm correctness;
- low-rank global optimality or approximation quality;
- rank-threshold policy correctness;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- portable performance, scalability, memory behavior, or state-of-the-art
  behavior.

## Day 4 Validation Checklist

If Day 4 implements the accepted fixture, run:

1. `python3 tests/svd_external_dense_reference.py svd_wide_fullrank_4x6`
2. `make format`
3. `make build/test_svd && ./build/test_svd`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 123 docs and touched files

The helper check must emit `OK 4`. A different output count is a fixture
protocol failure even if the emitted values are otherwise plausible.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 1 is complete or explicitly deferred. | Decision complete for Day 3; implementation remains Day 4. | Accepted `svd_wide_fullrank_4x6`; all other Day 2 candidates are deferred or rejected with gates. |
| Accepted SVD work is bounded and testable. | Complete | Fixture key, matrix, output count, tolerance, skip behavior, failure semantics, and affected surfaces are explicit. |
| Deferred SVD work has clear promotion gates. | Complete | See deferred SVD candidates table. |
