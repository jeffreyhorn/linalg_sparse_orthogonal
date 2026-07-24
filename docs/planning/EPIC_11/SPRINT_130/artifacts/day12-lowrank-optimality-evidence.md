# Sprint 130 Day 12: Low-Rank Optimality Evidence

## Purpose

Day 12 applies the Day 11 SuiteSparse corpus gate and implements the accepted
Tier 0 analytic low-rank evidence lane:
`partial_svd_lowrank_diag6x4_k2_frobenius_optimality`.

The lane validates dense reconstruction error for one exact rectangular
diagonal fixture against an independent analytic discarded-tail Frobenius
target. It does not promote SuiteSparse corpus parity, sparse-output
drop-tolerance behavior, or a broad best-rank approximation claim.

## Day 11 Gate Application

| Candidate | Day 12 disposition | Reason |
|---|---|---|
| SuiteSparse singular-value or vector evidence | Deferred | Checked-in corpus tests use product full-SVD values or product residuals, not independent expected values. |
| SuiteSparse large-matrix SVD evidence | Deferred | Runtime and platform policy are not ready for default Sprint 130 evidence. |
| Sparse low-rank output optimality | Deferred | Dense reconstruction optimality does not prove sparse-output or drop-tolerance behavior. |
| Local analytic dense low-rank fixture | Accepted | Singular values and discarded-tail Frobenius target are known without external or product-observed metadata. |

## Accepted Lane

| Field | Value |
|---|---|
| Fixture key | `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` |
| Matrix | 6x4 diagonal with entries `9`, `6`, `3`, `1` and two structural zero rows |
| Requested `k` | `2` |
| Retained singular values | `9`, `6` |
| Discarded singular values | `3`, `1` |
| Dense reconstruction metric | `||A - U_k Sigma_k V_k^T||_F` |
| Independent expected value | `sqrt(3^2 + 1^2) = sqrt(10)` |
| Vector diagnostics | `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, U/V orthogonality |
| Tolerance | `1e-8` for the exact analytic fixture only |
| Support tier | Tier 0 local analytic |

## Implementation Summary

| File | Change |
|---|---|
| `tests/test_svd_partial_helpers.h` | Added `test_partial_svd_lowrank_diag6x4_k2_frobenius_optimality`, computing dense partial reconstruction error from product U/sigma/Vt and comparing it to the analytic discarded-tail Frobenius target. |
| `tests/test_svd.c` | Registered the new low-rank fixture next to existing partial-SVD vector/reconstruction tests. |
| `docs/maintainer_guide.md` | Added the bounded dense low-rank Frobenius fixture while preserving sparse-output/drop-tolerance, broad parity, and platform non-claims. |

## Evidence Diagnostics

Observed focused diagnostics:

| Diagnostic | Observed | Bound |
|---|---:|---:|
| Max retained singular-value error | `1.776e-15` | `< 1e-8` |
| Dense reconstruction Frobenius error | `3.162277660168` | `sqrt(10) +/- 1e-8` |
| Expected discarded-tail Frobenius error | `3.162277660168` | exact analytic target |
| Max `A v - sigma u` residual | `3.928e-15` | `< 1e-8` |
| Max `A^T u - sigma v` residual | `7.128e-15` | `< 1e-8` |
| U orthogonality error | `8.882e-16` | `< 1e-8` |
| V orthogonality error | `8.921e-16` | `< 1e-8` |

## Deferrals

| Deferred lane | Blocker | Future owner |
|---|---|---|
| SuiteSparse corpus residual parity | No independent singular-value/vector/projector metadata is checked in. | Future corpus evidence owner must add oracle metadata and skip/runtime policy. |
| `nos4`/`west0067` promotion beyond smoke | Current checks compare against product full-SVD output or product residuals. | Future corpus owner. |
| `bcsstk04` sparse low-rank corpus optimality | Current corpus-safety check compares env-off/env-on product paths. | Future sparse-output owner. |
| `bcsstk14` and larger SVD corpus lanes | Default runtime and platform budgets are not declared. | Future benchmark/slow-test owner. |
| Sparse-output/drop-tolerance optimality | Dense reconstruction target does not describe thresholded sparse output. | Future low-rank sparse-output owner. |
| Broad best-rank approximation wording | One analytic fixture does not prove global optimality across matrix classes. | Day 14 solver-selection/claim gate. |
| Convergence-budget behavior | The accepted fixture uses default unbounded options. | Day 13 convergence-budget owner. |

## Non-Claims

Day 12 does not claim:

- SuiteSparse singular-value, vector, projector, or corpus parity;
- broad platform or optional-data support;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  MATLAB parity;
- sparse-output or drop-tolerance optimality;
- global low-rank or best-rank approximation optimality;
- convergence-budget semantics;
- public solver-selection guidance.

## Validation

1. `make format && make build/test_svd && ./build/test_svd`
2. `make format && make lint && make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Corpus and optimality evidence are bounded and validated, or explicitly deferred. | Complete | SuiteSparse corpus lanes are deferred; analytic low-rank lane passed focused SVD and full quality validation. |
| No broad SuiteSparse, platform, or best-rank approximation claim is added. | Complete | Non-claims preserve SuiteSparse, platform, sparse-output, and global optimality boundaries. |
| Optional-data behavior is visible to maintainers. | Complete | Day 11 support-tier and Day 12 deferral tables keep optional corpus behavior explicit. |
