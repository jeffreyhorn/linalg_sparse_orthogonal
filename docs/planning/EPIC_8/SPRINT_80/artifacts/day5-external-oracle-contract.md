# Sprint 80 Day 5 - External Oracle Contract

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Freeze the bounded Epic 8 external-oracle contract so later sprints know which
comparison lanes are maintained, which are advisory, and which are explicitly
outside the first contract.

## Main Result
The first maintained Epic 8 external-oracle contract is now fixed explicitly:

- maintained correctness reference lane:
  - bounded SuiteSparse-family direct-solver comparison centered first on the
    CHOLMOD-class SPD Cholesky lane
- maintained performance-reference support lane:
  - BLAS/LAPACK-class dense-kernel calibration for backend-aware performance
    work

These two lanes are intentionally **not** the same kind of proof:

- CHOLMOD-class comparison is the strongest first maintained correctness
  candidate
- BLAS/LAPACK-class comparison is bounded performance-reference support, not
  broad product correctness proof

## Deferred or Advisory Lanes

### Strongest second-tier but not first-contract maintained candidate
- a narrower unsymmetric/direct external comparison may become valid later
- it is not part of the first maintained Sprint 80 contract

### Strongest advisory comparison context
- METIS-class graph/reordering comparison remains useful advisory context
- it is not part of the first maintained external-oracle contract

### Explicitly non-contract candidates
- broad external sparse solver families such as:
  - SuperLU
  - MUMPS
  - PARDISO
  - wider comparison-layer ecosystems

These remain exploratory only.

## Preserved Non-goal Fence
- no mandatory heavyweight external stack for normal builds
- no fake cross-platform proof parity from locally convenient dependencies
- no “compare to everything” benchmark theater
- no broad correctness claim inflation from performance-reference lanes
- no dependency matrix that outruns the maintained CI/package surface

## Interpretation
The useful Day 5 clarification is now explicit:

- Epic 8 should begin with one bounded maintained direct-solver correctness
  comparison lane
- it should also carry one bounded dense-kernel calibration lane for backend
  work
- broader ecosystem comparison belongs later, and only if the earlier
  structural ceilings move successfully

## Exit State
- Epic 8 now has one explicit external-oracle contract instead of one generic
  ecosystem-comparison aspiration.
- Maintained, advisory, and non-contract external lanes are fixed in writing.
- Later Sprint 80 work can build against this stable comparison reading.
