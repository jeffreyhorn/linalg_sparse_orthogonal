# Sprint 65 Day 7: Solver-Efficiency Target Selection and Landing Design

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Use the Day 3-6 benchmark-role audit, normalization contract, and canonical
surface selection to freeze one exact first solver-efficiency target and one
bounded implementation fence before code lands.

## Selected First Efficiency Target

The first Sprint 65 efficiency target should remain:

- direct repeated-run CSC/Cholesky follow-through

Not the first target:

- iterative public-handle reuse follow-through
- eigensolver public-handle reuse follow-through
- LDL^T CSC symmetry work

## Why This Target Wins

The maintained benchmark evidence is strongest here:

- `bench_refactor_csc`
  - already compares the public repeated-run direct path against a more direct
    CSC path with stable CSV output
- `bench_chol_csc`
  - already reports linked-list versus CSC scalar versus CSC supernodal path
    identity with stable CSV output

That evidence is stronger than the next candidates because:

- both maintained surfaces point at the same Cholesky CSC implementation family
- the likely touched solver seam is smaller than the iterative/eigensolver
  workspace stories
- the proof burden is already concentrated in:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`

## First Implementation Fence

### Required first efficiency surface

- `src/sparse_chol_csc_supernodal.c`

### Likely support surfaces only if the landed change truly needs them

- `src/sparse_chol_csc.c`
- `src/sparse_dense.c`

### Likely proof surfaces

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

### Explicit non-targets for the first efficiency batch

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_iterative.c`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_workspace_internal.c`
- public headers
- `CMakeLists.txt`
- `Makefile`

## Benchmark Normalization Versus Efficiency Work

The benchmark normalization batch and the solver-efficiency batch are related
but not identical.

Normalization-first surfaces remain:

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`
- `README.md`
- `docs/maintainer_guide.md`

Interpretation:

- normalize the maintained benchmark surface first
- then land one bounded direct repeated-run CSC/Cholesky efficiency change
- then tighten the maintained interpretation after the code path is stable

## Proof Plan

The first efficiency proof burden should stay bounded:

- family-local correctness and regression proof:
  - `tests/test_chol_csc.c`
- bounded public non-regression proof:
  - `tests/test_integration.c`
- maintained runtime evidence:
  - `bench_refactor_csc`
  - `bench_chol_csc`

The intent is to prove a real efficiency follow-through on the maintained
direct repeated-run CSC/Cholesky lane without widening into unrelated solver
families or benchmark policy work.

## Explicit Non-Goals

- no public API or header widening
- no build-option or build-system batch unless implementation is truly blocked
- no LDL^T symmetry batch in the same landing
- no iterative/eigensolver efficiency landing in the same batch
- no broad benchmark catalog rewrite
- no CI runtime-lane expansion

## Day 7 Exit State

Sprint 65 now has:

- one exact first solver-efficiency target chosen from maintained benchmark
  evidence
- one narrow implementation fence centered on the Cholesky CSC supernodal lane
- one clear split between benchmark normalization work and later efficiency work
- one bounded proof and non-goal set for the first code landing
