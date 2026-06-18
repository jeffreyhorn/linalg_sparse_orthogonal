# Sprint 80 Day 6 - Performance and Benchmark Contract

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Freeze the benchmark and performance-governance interpretation Epic 8 backend
and runtime work must preserve so later comparison or acceleration work does
not widen into fake timing gates or broader product claims.

## Main Result
The benchmark/performance contract Epic 8 must preserve is now fixed
explicitly:

- canonical maintained benchmark face:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- threshold-free canonical reporting surface:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
- bounded runtime lane:
  - `bench-fast`
- narrow thresholded regression gate:
  - `wall-check`
- exploratory/context-only benchmark lanes:
  - broader bench surfaces outside the compact maintained face

## Interpretation
The strongest current benchmark-governance reading is now explicit:

- canonical report bundles are artifact-friendly longitudinal snapshots
- they are intentionally **not** pass/fail timing gates
- benchmark binaries own row semantics and path measurability
- tests remain the owners of regression/oracle/property truth

## Composition with the Day 5 External-Oracle Contract

- maintained CHOLMOD-class correctness comparison belongs in test or
  differential-proof surfaces first, not in canonical benchmark pass/fail
  interpretation
- BLAS/LAPACK-class dense-kernel calibration belongs as backend-aware
  performance-reference support and may widen benchmark fields or advisory
  comparison output
- that backend-aware widening does **not** change the threshold-free reading of
  the canonical report bundle

## Preserved Non-goal Fence
- no portable timing verdicts from single-run benchmark numbers
- no historical-diff pass/fail gate hidden inside canonical reporting
- no broad benchmark-surface widening before the structural ceilings move
- no confusion between benchmark proof and test-owned correctness proof

## Exit State
- Epic 8 now has one explicit benchmark/performance contract.
- Backend-aware comparison work can move later without reopening benchmark
  governance every time.
- The canonical reporting surface, runtime lane, and narrow threshold gate are
  fixed in writing before later Sprint 80 contract work continues.
