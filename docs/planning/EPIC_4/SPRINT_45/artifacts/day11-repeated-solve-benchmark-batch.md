# Sprint 45 Day 11 Artifact: Repeated-Solve Benchmark Batch

## Purpose

Land the narrow repeated-solve benchmark slice defined on Day 10 so Sprint 45
has direct evidence for the reusable iterative workspace seam on the migrated
scalar solver paths, without widening into benchmark CLI redesign or a larger
benchmark framework.

## Main Day 11 Conclusion

Sprint 45 now has a dedicated repeated-solve benchmark that compares:

- repeated one-shot public scalar iterative solves
- repeated reusable-workspace-backed internal scalar iterative solves

for:

- scalar CG
- scalar GMRES

The measured result is honest and narrow:

- convergence behavior matched exactly on the benchmarked cases
- the local repeated-solve timing win was modest
- the batch still proves the reusable-workspace seam is directly usable for
  repeated-call benchmarking

## Landed Scope

### 1. A dedicated repeated-solve benchmark now exists

Day 11 added:

- `benchmarks/bench_iterative_reuse.c`

The benchmark uses the `bench_refactor.c` A/B comparison style:

- one-shot public solve loop
- reusable-workspace-backed internal solve loop
- stable matrix shape
- stable options/tolerances
- concise wall-time plus convergence reporting

It intentionally stays limited to:

- scalar CG on a generated SPD tridiagonal system
- scalar GMRES on a generated nonsymmetric tridiagonal system

Interpretation:

- Sprint 45 now has direct evidence for the landed workspace seam
- the benchmark avoids broader harness or CLI churn

### 2. A bounded private internal iterative header now supports direct benchmark reuse

Day 11 added:

- `src/sparse_iterative_internal.h`

This private header exposes only:

- `sparse_solve_cg_with_workspace_internal(...)`
- `sparse_solve_gmres_with_workspace_internal(...)`

Interpretation:

- the benchmark uses a real internal seam instead of ad hoc implementation
  reach-through
- public API shape stayed unchanged

### 3. Scalar one-shot iterative entries now compose around the reusable workspace seam

After Day 11:

- `sparse_solve_cg(...)`
  - allocates/frees one-shot local workspace
  - delegates the actual solve to the reusable internal helper
- `sparse_solve_gmres(...)`
  - allocates/frees one-shot local workspace
  - delegates the actual solve to the reusable internal helper
- `sparse_solve_gmres_mf(...)`
  - now follows the same internal reusable-workspace pattern

Interpretation:

- one-shot public behavior is preserved
- the reusable-workspace path now represents the direct internal truth for the
  touched scalar benchmarked cases

## Measured Outputs

Direct benchmark output on this local run was:

### CG repeated-solve case

- case:
  - `cg-tridiag-300`
- repeats:
  - `400`
- one-shot:
  - `24.7220 ms`
- reuse:
  - `24.7000 ms`
- speedup:
  - `1.00x`
- last solve outcome, both paths:
  - `17` iterations
  - relative residual `5.192e-11`
  - converged

### GMRES repeated-solve case

- case:
  - `gmres-unsym-220`
- repeats:
  - `300`
- one-shot:
  - `17.4030 ms`
- reuse:
  - `17.1030 ms`
- speedup:
  - `1.02x`
- last solve outcome, both paths:
  - `12` iterations
  - relative residual `7.364e-11`
  - converged

Interpretation:

- the benchmarked reusable-workspace path preserved solver behavior exactly on
  the touched scalar cases
- the local timing gain is small rather than dramatic
- that is still valid Sprint 45 evidence because the batch was about direct
  repeated-solve measurement, not about forcing a large speedup claim

## Preserved Boundaries

Day 11 deliberately did **not** widen into:

- block GMRES / MINRES / BiCGSTAB benchmark work
- benchmark CLI redesign
- `bench_main` mode expansion
- new public iterative workspace APIs
- new solver math or convergence-policy changes

Interpretation:

- the batch stayed aligned with the actual Sprint 45 migration surface
- the benchmark remains a bounded evidence artifact, not a new benchmark
  subsystem

## Validation

Because `*.c` and `*.h` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

Direct Day 11 follow-ons also passed:

- `./build/test_iterative`
- `./build/bench_iterative_reuse`

Representative direct follow-on outcomes:

- `test_iterative`
  - all `76` tests passed
- `bench_iterative_reuse`
  - both benchmark cases completed successfully
  - one-shot and reuse paths matched on convergence and iteration counts

## Sprint 45 Position After Day 11

After this batch, Sprint 45 has now landed:

- reusable internal iterative workspace owner
- scalar CG / matrix-free CG migration
- scalar GMRES / matrix-free GMRES migration
- block-CG migration
- wrapper compatibility cleanup for block GMRES / MINRES / BiCGSTAB
- direct repeated-solve benchmark evidence for the migrated scalar paths

Interpretation:

- the main structural workspace-reuse objective is now materially complete
- the sprint can move into validation/closeout work without another broad
  iterative redesign batch

## Bottom Line

Day 11 delivered:

- a dedicated repeated-solve benchmark
- a bounded private internal benchmark seam
- compatibility-preserving scalar wrapper normalization
- measured local repeated-solve evidence for CG and GMRES
- a fully green validation baseline

The measured effect is modest on this machine, but the benchmark is now real,
bounded, and truthful, which is the right Sprint 45 Day 11 outcome.
