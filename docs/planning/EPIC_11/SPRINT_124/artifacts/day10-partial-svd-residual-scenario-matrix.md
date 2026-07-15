# Day 10 Partial-SVD Residual Scenario Matrix

## Purpose

Decide the Sprint 124 scope for partial-SVD repeated-spectrum,
clustered-spectrum, rank-deficient, convergence-budget, corpus, and low-rank
optimality residual scenarios after Day 9 landed the bounded
`partial_svd_vector_residual_diag6_k2` lane.

This artifact keeps each residual class independent so Day 11 can either land
one narrowed follow-up or publish an explicit deferral package without losing
the carry-forward work.

## Current Accepted Baseline

| Evidence | Owner | What it proves | Boundary |
| --- | --- | --- | --- |
| `partial_svd_diag6_k2` | `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | Bounded external top-2 singular-value agreement for a square diagonal fixture | Value-only; no vector/subspace or convergence claim. |
| `partial_svd_tall_diag_8x5_k3` | `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | Bounded external top-3 singular-value agreement for a tall diagonal fixture | Value-only rectangular evidence. |
| `partial_svd_vector_residual_diag6_k2` | `tests/test_svd_partial_helpers.h` | Bounded sign-invariant vector-residual evidence for one exact square diagonal fixture | No repeated, clustered, rank-deficient, corpus, low-rank, or convergence-budget claim. |
| Internal partial-SVD vector and reconstruction tests | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Regression coverage for vectors, reconstruction, rectangular shapes, and SuiteSparse smoke | Internal consistency evidence, not independent dense-reference parity. |

## Scenario Matrix

| Scenario | Candidate fixture | Required diagnostics | Trust boundary | Day 10 decision | Day 11 handoff |
| --- | --- | --- | --- | --- | --- |
| Repeated leading singular values | `partial_svd_repeated_diag6_k3` | unordered leading value multiset, left/right projector distance, basis-dimension check | Individual vectors may rotate or swap inside the repeated subspace | Defer | Future subspace owner must add projector or principal-angle helper output before implementation. |
| Clustered leading singular values | `partial_svd_clustered_diag6_k3` | declared spectral gap, ordered/set value policy, projector distance, iteration budget, residual tolerance | Strict ordering can be algorithm-dependent when values are close | Defer | Future convergence/subspace owner must define near-tie interpretation and budgeted failure meaning. |
| Rank-deficient top-k crossing rank | `partial_svd_rankdef_diag_6x4_k3` | positive-rank threshold, zero singular-value tolerance, left/right range projector, optional null-space projector | Zero-space basis is ambiguous and rank threshold is fixture-specific | Defer | Future rank owner must define threshold and whether zero-space evidence is range-only or null-space evidence. |
| Rectangular vector residual | `partial_svd_vector_residual_tall8x5_k3` or wide analogue | `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, `U`/`V` orthogonality, output dimensions | Day 9 square fixture proved protocol; rectangular lanes add shape risk | Defer to Day 11 only if a single tall or wide lane is narrowed without changing helper protocol | Future vector owner may promote one rectangular residual fixture with exact diagonal values and `1e-8` residual targets. |
| SuiteSparse vector-residual corpus | `nos4`, `west0067`, or bounded corpus subset | corpus-specific residual window, availability skip, matrix conditioning note, no external dense helper dependency | Optional data and conditioning make exact tolerances unsafe | Defer | Future corpus owner must state skip behavior and fixture-specific residual windows. |
| Low-rank optimality | `partial_svd_lowrank_rect_5x4_k2` | dense reconstruction error, Frobenius or spectral norm target, sparse-output drop semantics if sparse output is used | Low-rank reconstruction is not the same evidence class as top-k values or vector residuals | Defer | Future low-rank owner must decide dense-only versus sparse-output evidence and name the optimality metric. |
| Convergence budget | clustered or difficult corpus fixture | options, iteration cap, tolerance, deterministic initialization policy, residual failure class | Timing smoke is not convergence proof | Defer | Future convergence owner must add budget controls and distinguish non-convergence from reference mismatch. |
| Nonsymmetric rectangular value residual | existing `test_partial_svd_nonsymmetric` class | value agreement plus residual diagnostic | Internal full-SVD comparison only today | Defer | Future external value owner may add a dense-reference rectangular non-diagonal fixture only if it avoids vector/subspace claims. |

## Accepted Scenarios for Day 11

No new Day 10 residual scenario is accepted for immediate implementation.

The reason is deliberately narrow: Day 9 already added the safe bounded vector
residual proof. The remaining scenarios require new semantics that are not
local to a single exact diagonal vector-residual fixture:

- repeated and clustered spectra require projector or principal-angle helpers;
- rank-deficient scenarios require numerical-rank and zero-space thresholds;
- corpus residuals require fixture availability and conditioning policy;
- low-rank optimality requires a separate reconstruction-optimality metric;
- convergence budgets require option-surface and iteration-budget ownership.

Day 11 should therefore produce a deferral package unless it narrows the
rectangular vector-residual scenario to a single exact diagonal fixture without
adding new helper protocol or public claims.

## Tolerance and Skip Policy

| Scenario class | Tolerance policy | Skip policy |
| --- | --- | --- |
| Existing exact diagonal value/vector lanes | `1e-8` for singular values, product residuals, and orthogonality | Existing external-helper skip for missing `python3`; explicit Windows skip. |
| Repeated subspace | Projector/principal-angle tolerance must be fixture-specific; do not inherit vector residual tolerance | No skip beyond helper availability once a helper exists. |
| Clustered subspace/convergence | Tolerance must include spectral gap and iteration budget | Budget exhaustion is failure only when the budget is part of the fixture contract. |
| Rank-deficient threshold | Positive-rank and zero singular-value tolerances must be declared together | No silent skip for threshold ambiguity. |
| SuiteSparse corpus residual | Fixture-specific residual windows; exact diagonal tolerance is invalid | Missing optional corpus file may skip only when the test owner says so. |
| Low-rank optimality | Frobenius or spectral norm threshold must be named; sparse drop tolerance must be explicit | No skip for metric ambiguity. |

## Failure Interpretation

| Failure class | Meaning |
| --- | --- |
| Helper protocol error | Reference generator or parser failed; this is a test infrastructure failure. |
| Singular-value mismatch | Bounded value regression for the named fixture only. |
| Vector residual mismatch | Bounded triplet-quality regression; not a raw sign/orientation failure. |
| Orthogonality mismatch | Vector publication quality regression for the named fixture. |
| Projector/subspace mismatch | Subspace quality regression only when projector metrics are explicitly implemented. |
| Rank-threshold mismatch | Fixture-specific numerical-rank policy failure. |
| Convergence-budget miss | Algorithm did not satisfy the declared budget; not a broad convergence guarantee. |
| Low-rank reconstruction mismatch | Fixture-specific reconstruction/optimality regression, not global low-rank optimality failure unless the metric says so. |

## Future-Owner Handoff

| Future owner | Required promotion gates |
| --- | --- |
| Day 11 deferral owner | Publish the above matrix as the residual package and avoid new claims, unless one exact rectangular residual lane is narrowed enough to implement safely. |
| Subspace owner | Add projector or principal-angle reference protocol and shape checks before repeated or clustered fixtures land. |
| Rank-deficient owner | Define rank threshold, zero singular-value tolerance, and range/null-space evidence class. |
| Corpus owner | Define optional-data skip rules, residual windows, and conditioning notes per matrix. |
| Low-rank owner | Separate dense low-rank optimality from sparse low-rank output/drop-tolerance semantics. |
| Convergence owner | Define options, iteration cap, deterministic start policy, residual tolerance, and budget-failure meaning. |

## Non-Claim Register

Day 10 preserves the following non-claims:

- no LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no broad partial-SVD external parity claim;
- no repeated-spectrum or clustered-spectrum correctness claim;
- no rank-deficient subspace or null-space parity claim;
- no SuiteSparse corpus vector-residual parity claim;
- no low-rank global optimality claim;
- no convergence-budget guarantee;
- no package, ABI, platform, performance, scalability, public API, or
  state-of-the-art claim.

## Validation

Day 10 changes documentation only. Validation is limited to `git diff --check`
and a focused trailing-whitespace scan of Sprint 124 documentation files.
