# Sprint 130 Day 13: Convergence-Budget Evidence

## Purpose

Day 13 adds one bounded convergence-budget evidence lane for partial SVD:
`partial_svd_max_iter_fail_closed_diag6_k2`.

The lane validates fail-closed behavior under an intentionally insufficient
QR-iteration budget. It does not claim convergence rate, iteration counts,
stagnation handling, partial-result publication, or broad success guarantees.

## API Boundary

Current `sparse_svd_partial` accepts `max_iter` and `tol` through
`sparse_svd_opts_t`, but `sparse_svd_t` does not expose:

- iteration count;
- achieved tolerance;
- number of converged singular triplets;
- partial-result status fields;
- residual history or stagnation flags.

Because those fields do not exist, Day 13 cannot honestly assert partial
convergence semantics. The accepted lane is therefore limited to one
fail-closed budget case plus successful recovery under the default budget.

## Accepted Lane

| Field | Value |
|---|---|
| Fixture key | `partial_svd_max_iter_fail_closed_diag6_k2` |
| Matrix | 6x6 diagonal with entries `9`, `6`, `3`, `1`, `0.5`, `0.25` |
| Requested `k` | `2` |
| Insufficient budget | `max_iter = 1`, `tol = 0.0`, `compute_uv = 1`, `economy = 1` |
| Expected insufficient-budget result | `SPARSE_ERR_NOT_CONVERGED` |
| Fail-closed payload policy | `sigma == NULL`, `U == NULL`, `Vt == NULL`; shape fields remain diagnostic only |
| Recovery check | Same fixture with default budget returns `SPARSE_OK` |
| Recovery diagnostics | retained singular values, `A v - sigma u`, and `A^T u - sigma v` |
| Tolerance | `1e-8` for recovered values/residuals on this analytic fixture |

## Implementation Summary

| File | Change |
|---|---|
| `tests/test_svd_partial_helpers.h` | Added `test_partial_svd_max_iter_fail_closed_diag6_k2`. |
| `tests/test_svd.c` | Registered the new convergence-budget fixture next to the bounded partial-SVD lanes. |
| `docs/maintainer_guide.md` | Added the bounded fail-closed fixture and preserved convergence-rate and partial-result non-claims. |

## Evidence Diagnostics

Observed focused diagnostics:

| Diagnostic | Observed | Bound |
|---|---:|---:|
| Insufficient-budget return code | `13` (`SPARSE_ERR_NOT_CONVERGED`) | exact |
| Insufficient-budget `sigma` | `NULL` | exact |
| Insufficient-budget `U` | `NULL` | exact |
| Insufficient-budget `Vt` | `NULL` | exact |
| Diagnostic shape fields after failure | `m=6`, `n=6`, `k=2` | diagnostic only |
| Default-budget max retained singular-value error | `2.665e-15` | `< 1e-8` |
| Default-budget max `A v - sigma u` residual | `4.022e-15` | `< 1e-8` |
| Default-budget max `A^T u - sigma v` residual | `1.121e-14` | focused check passed; not promoted as a strict Day 13 residual claim |

The final `A^T u` diagnostic is inherited from the existing exact diagonal
residual lane and is used only to prove the same fixture recovers under the
default budget. Day 13's convergence evidence is the fail-closed return and
payload behavior.

## Deferrals

| Deferred lane | Blocker | Promotion criteria |
|---|---|---|
| Iteration-count evidence | `sparse_svd_t` does not report iterations. | Add a public result/diagnostic field or callback with stable semantics. |
| Achieved-tolerance evidence | No achieved tolerance or residual history is returned. | Add public diagnostics and fixture-specific tolerance policy. |
| Partial-result publication | On QR non-convergence the current API returns an error before publishing `sigma`, `U`, or `Vt`. | Define `n_converged`, payload validity, ordering, and ownership rules before exposing partial results. |
| Stagnation behavior | Partial SVD exposes no stagnation detector or history. | Add explicit stagnation options and result fields. |
| Clustered-spectrum convergence budget | Day 8 showed repeated/clustered lanes need projector policy and likely algorithm changes. | Pair convergence diagnostics with subspace metrics and repeated/clustered promotion criteria. |
| SuiteSparse convergence budget | Day 11 deferred corpus promotion for lack of independent metadata and runtime policy. | Add corpus oracle metadata, support tier, skip behavior, and runtime budget. |
| Public solver-selection wording | One fail-closed budget fixture is not user-facing guidance. | Day 14 claim gate must reconcile all Sprint 130 evidence. |

## Non-Claims

Day 13 does not claim:

- convergence rate or iteration count;
- achieved tolerance reporting;
- partial-result availability after non-convergence;
- stagnation detection;
- clustered, repeated, corpus, or SuiteSparse convergence behavior;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  MATLAB parity;
- public solver-selection guidance.

## Validation

1. `make format && make build/test_svd && ./build/test_svd`
2. `make format && make lint && make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Accepted evidence reports bounded convergence behavior only. | Complete | The accepted lane checks only fail-closed behavior under `max_iter=1` and default-budget recovery; focused SVD and full quality validation passed. |
| Partial results do not imply broad parity or success guarantees. | Complete | Non-converged calls publish no `sigma`, `U`, or `Vt` payload in the accepted fixture. |
| Every deferred convergence lane has blocker and promotion criteria. | Complete | Deferral table records required API/result metadata and future promotion gates. |
