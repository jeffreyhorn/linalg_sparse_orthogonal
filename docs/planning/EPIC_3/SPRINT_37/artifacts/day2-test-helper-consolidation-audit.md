# Sprint 37 Day 2 Test-Helper Consolidation Audit

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Audit the maintained test tree for duplicated helper logic, separate true
consolidation targets from intentionally local helpers, and define a bounded
first-pass cleanup queue that preserves the Sprint 32 truthfulness/opt-in test
contract.

## Executive Summary

Sprint 37 does have a real test-helper consolidation queue, but it is not a
repo-wide “create one test helper framework” problem.

The duplication is concentrated in a few helper families:

- synthetic SPD/tridiagonal builders
- KKT builders
- residual calculators
- a smaller Jacobi/precondition helper family

The current shared test-support layer remains intentionally small:

- `tests/test_framework.h`

That means the first cleanup batch should stay cluster-based and low-risk:

- small shared headers or tightly scoped shared helpers for related tests
- not a broad new test-support `.c` library as the first move

## Current Shared Test-Support Floor

The test tree already centralizes exactly one major shared concern:

- test execution and policy behavior in `tests/test_framework.h`

That file owns:

- `RUN_TEST`
- `RUN_TEST_SLOW`
- `RUN_TEST_EXPERIMENTAL`
- `SKIP_TEST`
- assertion/reporting macros
- portable env-var and temp-path support

There is no existing broad helper layer for:

- matrix construction
- solver residual checks
- reusable test fixtures
- common numerical utility helpers

Interpretation:

- helper consolidation in Sprint 37 should stay additive and targeted
- the audit does not justify turning the test tree into a large frameworked
  support layer

## Truthfulness / Opt-In Baseline

Re-check against the Sprint 32 contract:

- commented-out `RUN_TEST(...)` registrations in `tests/*.c`: `0`
- live opt-in/skip behavior remains owned by `test_framework.h`

Implication:

- no consolidation pass should attempt to centralize test registration logic
  beyond the existing framework layer
- historical evidence must remain in docs/artifacts, not shared helper stubs

## Density Hotspots

Highest helper-heavy maintained tests:

| File | Lines | Static functions |
|---|---:|---:|
| `tests/test_chol_csc.c` | `4,643` | `154` |
| `tests/test_ldlt_csc.c` | `3,637` | `115` |
| `tests/test_etree.c` | `2,890` | `101` |
| `tests/test_svd.c` | `3,712` | `99` |
| `tests/test_ldlt.c` | `2,774` | `85` |
| `tests/test_iterative.c` | `2,819` | `82` |
| `tests/test_colamd.c` | `1,957` | `78` |
| `tests/test_qr.c` | `3,259` | `74` |

Interpretation:

- large files matter, but helper count alone is not enough to justify sharing
- some large files are assertion-heavy around local internal structures and are
  poor first extraction targets

## Real Consolidation Candidate Families

### 1. Residual helper family

Strongest semantic duplication signal in the audit.

Observed variants:

- `compute_relative_residual`
- `relative_residual`
- `compute_rel_residual`
- `rel_residual`

Observed across:

- `tests/test_iterative.c`
- `tests/test_bicgstab.c`
- `tests/test_ilu.c`
- `tests/test_ic.c`
- `tests/test_minres.c`
- `tests/test_qr.c`
- `tests/test_chol_csc.c`
- several `tests/test_sprint*_integration.c`

Assessment:

- **Candidate strength:** high
- **Why:** repeated logic, repeated naming drift, repeated signatures that are
  conceptually the same, and broad use across solver and integration tests
- **Risk:** low to medium if the helper contract stays numerically simple and
  explicit

Recommended landing shape:

- small cluster helper header for solver/integration residual checks
- or a pair of tightly scoped helper headers if vector and block-residual paths
  need to stay separate

### 2. SPD / tridiagonal matrix builder family

Observed variants:

- `build_spd_tridiag`
- `make_spd_tridiag`
- `make_tridiag`

Observed across:

- `tests/test_iterative.c`
- `tests/test_bicgstab.c`
- `tests/test_ilu.c`
- `tests/test_ic.c`
- `tests/test_minres.c`
- `tests/test_omp.c`
- `tests/test_sprint11_integration.c`
- `tests/test_sprint13_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`
- `tests/test_sprint29_integration.c`
- `tests/test_stagnation.c`

Assessment:

- **Candidate strength:** high
- **Why:** repeated synthetic-matrix setup across closely related solver and
  integration tests
- **Risk:** medium because some variants carry slightly different parameter
  surfaces or scaling assumptions

Recommended landing shape:

- shared helper header for simple synthetic SPD builders
- keep specialized scaled/banded/day-specific builders local

### 3. KKT builder family

Observed variants:

- `build_kkt`
- `make_kkt`

Observed across:

- `tests/test_minres.c`
- `tests/test_ldlt.c`
- `tests/test_eigs.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint13_integration.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint20_integration.c`

Assessment:

- **Candidate strength:** medium to high
- **Why:** same conceptual test fixture repeated across LDLT/MINRES/integration
  surfaces
- **Risk:** medium because some tests intentionally vary shape or scaling

Recommended landing shape:

- cluster-based helper around KKT synthetic fixtures
- keep highly specialized named fixtures local

### 4. Identity / diagonal builder family

Observed variants:

- `build_identity`
- `make_identity`

Observed across:

- `tests/test_iterative.c`
- `tests/test_bicgstab.c`
- `tests/test_sparse_lu.c`

Assessment:

- **Candidate strength:** low to medium
- **Why:** real duplication exists, but the queue is small and payoff is lower
- **Risk:** low

Recommended treatment:

- opportunistic cleanup only if touched by a stronger nearby batch

### 5. Jacobi / precondition callback family

Observed variants and neighbors:

- `make_jacobi`
- `jacobi_precond`
- `cholesky_precond_apply`
- `identity_precond`

Observed across:

- `tests/test_minres.c`
- `tests/test_sprint13_integration.c`
- `tests/test_sprint5_integration.c`
- `tests/test_eigs_lobpcg.c`
- `tests/test_ilu.c`

Assessment:

- **Candidate strength:** medium
- **Why:** conceptually related, but callback signatures and ownership are less
  uniform than the residual/matrix-builder families
- **Risk:** medium to high

Recommended treatment:

- defer until after clearer residual/builder consolidation wins

## Keep Local: Not Good First Extraction Targets

These areas are large but should stay mostly local in the first pass:

### `tests/test_chol_csc.c`

Reasons to keep local:

- many helpers assert local CSC/supernodal invariants
- helper logic is tightly coupled to local factorization structures
- extracting early would likely reduce readability more than it helps

### `tests/test_ldlt_csc.c`

Reasons to keep local:

- structure-specific local assertions and fixture logic
- repeated names exist, but semantics are not broadly generic

### `tests/test_svd.c`

Reasons to keep local:

- helper logic is tied to bidiagonal/SVD-specific invariants and validation
  paths

### `tests/test_qr.c`

Reasons to keep local:

- local factorization and reconstruction helpers are tightly coupled to QR test
  intent

## Build-System Constraint

Current test build model:

- Makefile builds each test binary from its own single `tests/test_*.c`
- CMake adds tests one-by-one via `add_sparse_test(...)`

Implication:

- the safest helper-extraction forms are:
  - small shared headers
  - `static inline` utility helpers
  - tightly scoped cluster headers for related tests
- the riskiest first step would be a broad shared helper `.c` library that
  requires immediate Makefile and CMake relinking changes across the full suite

## Ranked First-Pass Queue

### Priority A

Residual helper consolidation across iterative and integration tests.

Why first:

- strongest duplication signal
- clearest shared semantics
- lowest risk of obscuring local test structure

### Priority B

SPD/tridiagonal synthetic matrix builder consolidation across iterative,
preconditioner, and selected integration tests.

Why second:

- broad duplication
- good maintainability payoff
- manageable risk if specialized variants remain local

### Priority C

KKT synthetic fixture consolidation across LDLT/MINRES/integration tests.

Why third:

- real duplication
- useful but slightly more specialized than the first two families

### Deferred / opportunistic

- identity/diagonal builders
- Jacobi/precondition callback family
- local assertion-heavy helpers in very large factorization tests

## Day 2 Conclusion

Sprint 37 has a real, bounded test-helper consolidation queue:

- high-value first targets are repeated residual helpers and simple synthetic
  matrix builders
- the right landing shape is narrow cluster-based shared support, not a broad
  new framework layer
- the Sprint 32 truthfulness floor remains intact and should stay untouched

That gives Day 5 a concrete and low-risk starting point:

- consolidate residual helpers first
- then consolidate simple SPD/tridiagonal builders
- leave large factorization-local assertion helpers in place for now
