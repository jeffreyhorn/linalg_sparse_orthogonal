# Sprint 80 Day 4 - External Oracle Candidate Audit

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Identify the strongest realistic external correctness and performance reference
classes for Epic 8 and reduce the broad “compare against the ecosystem”
pressure to one ranked maintained-vs-advisory candidate map.

## Main Result
The live tree already has strong corpus realism but not strong maintained
external solver-oracle proof:

- SuiteSparse matrices are already used widely across tests, examples, and
  benchmarks
- the tree does **not** currently maintain external solver-library linkage or
  CI-backed comparison proof

That means the missing piece is not more fixture realism. It is bounded
external numerical-oracle realism.

## Ranked Candidate Set

### Strongest maintained correctness candidate
- SuiteSparse-family direct references
  - especially CHOLMOD-class SPD Cholesky comparison
  - plus a narrower unsymmetric/direct counterpart only if packaging burden
    stays tolerable

Why it ranks first:
- highest correctness value
- strongest alignment to the library's best current CSC direct lanes
- realistic first maintained comparison candidate if kept bounded

### Strongest maintained performance-reference candidate
- BLAS/LAPACK-class dense-kernel references

Why it ranks second:
- highest backend-calibration value
- strongest alignment to the Epic 8 dense/backend ceiling
- best read as bounded performance-reference support, not as broad
  correctness proof

### Strongest advisory but not first maintained candidate
- METIS-class nested-dissection / graph-quality comparison

Why it stays advisory first:
- interesting algorithmic comparison value
- weaker first-sprint payoff than direct-solver and dense-kernel references
- better later support context than first maintained contract

### Strongest exploratory-only candidates
- broad external sparse solver families such as:
  - SuperLU
  - MUMPS
  - PARDISO
  - wider Eigen-style comparison layers

Why they stay exploratory:
- high dependency burden
- high platform/CI burden
- too broad for the first maintained Epic 8 contract

## Candidate Suitability Reading

### CHOLMOD-class SPD differential proof
- correctness value: high
- packaging/tooling burden: moderate
- portability risk: moderate
- maintenance realism: good if bounded

### Narrower unsymmetric/direct external comparison
- correctness value: meaningful
- packaging/tooling burden: higher
- portability risk: higher
- maintenance realism: plausible later, weaker as the first lane

### BLAS/LAPACK dense-kernel calibration
- performance value: high
- correctness value: narrower
- packaging/tooling burden: moderate
- maintenance realism: strong as bounded backend-calibration support

### METIS-style ND comparison
- correctness/product value: indirect
- packaging/tooling burden: moderate
- maintenance realism: better as advisory or later support context

## Interpretation
The useful Day 4 clarification is now explicit:

- Epic 8 should **not** try to compare against every major sparse package.
- It should begin with:
  - one bounded maintained direct-solver correctness lane
  - one bounded dense-kernel performance-reference lane
- broader ecosystem comparison belongs in advisory or later-stage context, not
  in the first maintained contract.

## Exit State
- The external-oracle conversation is now reduced to one ranked candidate map.
- The first realistic maintained candidate reads as bounded SuiteSparse-family
  direct-solver correctness comparison.
- BLAS/LAPACK-class dense-kernel calibration is fixed as the strongest
  performance-reference support lane for the Day 5 contract.
