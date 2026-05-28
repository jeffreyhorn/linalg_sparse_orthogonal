# Sprint 46 Day 11 Artifact: Repeated-Run Benchmark Batch

## Purpose

Land the narrow repeated-run eigensolver benchmark slice defined on Day 10 so
Sprint 46 has direct evidence for the migrated reusable workspace/state seam on
the Lanczos-family backends, without broadening into `bench_eigs.c` CLI churn,
public API changes, or broader benchmark-framework work.

## Main Day 11 Conclusion

Sprint 46 now has a dedicated repeated-run eigensolver benchmark that compares:

- repeated one-shot public eigensolver calls
- repeated reusable-workspace-backed internal eigensolver calls

for:

- grow-m Lanczos
- thick-restart Lanczos

The measured result is honest and narrow:

- convergence behavior matched exactly on the benchmarked cases
- the local repeated-run timing win was modest
- the batch still proves the reusable-workspace seam is directly usable for
  repeated-run benchmarking without changing solver behavior

## Landed Scope

### 1. A dedicated repeated-run benchmark now exists

Day 11 added:

- `benchmarks/bench_eigs_reuse.c`

The benchmark uses the Sprint 45 A/B comparison style:

- one-shot public solve loop
- reusable-workspace-backed internal solve loop
- stable matrix shape
- stable options/tolerances
- concise wall-time plus parity reporting

It intentionally stays limited to:

- grow-m Lanczos on `nos4`
- thick-restart Lanczos on `bcsstk14`

Interpretation:

- Sprint 46 now has direct repeated-run evidence for the landed reusable
  eigensolver seam
- the benchmark avoids broader harness or CLI churn

### 2. A bounded private internal eigensolver entry now supports direct benchmark reuse

Day 11 added:

- `sparse_eigs_sym_with_workspace_internal(...)`

This private helper mirrors the public `sparse_eigs_sym(...)` path’s:

- defaults
- validation
- shift-invert setup
- backend dispatch
- result-field contract
- refinement handoff

while accepting a caller-owned `sparse_eigs_workspace_t`.

Interpretation:

- the benchmark uses a real internal seam instead of ad hoc implementation
  reach-through
- public API shape stayed unchanged

### 3. Public one-shot eigensolver execution now composes around the same shared implementation

After Day 11:

- `sparse_eigs_sym(...)`
  - remains the compatibility-facing one-shot public entry
  - delegates to a shared implementation with `workspace == NULL`
- `sparse_eigs_sym_with_workspace_internal(...)`
  - uses the same implementation
  - supplies a caller-owned reusable workspace

The reusable workspace path is currently active for:

- `SPARSE_EIGS_BACKEND_LANCZOS`
- `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`

LOBPCG intentionally kept its existing local allocation model for this batch.

Interpretation:

- one-shot public behavior is preserved
- the reusable-workspace benchmark path now represents a direct internal truth
  for the migrated repeated-run cases

## Measured Outputs

Direct benchmark output on this local run was:

### Grow-m Lanczos repeated-run case

- case:
  - `growm-nos4-k5`
- fixture:
  - `nos4`
- backend:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS`
- repeats:
  - `40`
- one-shot median:
  - `1.3680 ms`
- reuse median:
  - `1.3610 ms`
- speedup:
  - `1.01x`
- last-run parity, both paths:
  - `115` iterations
  - converged
  - residual `4.326e-14`
  - peak basis size `100`
  - `|lambda|max diff = 0.000e+00`

### Thick-restart Lanczos repeated-run case

- case:
  - `thick-bcsstk14-k5`
- fixture:
  - `bcsstk14`
- backend:
  - explicit `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
- repeats:
  - `8`
- one-shot median:
  - `49.7370 ms`
- reuse median:
  - `47.7710 ms`
- speedup:
  - `1.04x`
- last-run parity, both paths:
  - `105` iterations
  - converged
  - residual `7.864e-14`
  - peak basis size `40`
  - `|lambda|max diff = 0.000e+00`

Interpretation:

- the reusable-workspace path preserved eigensolver behavior exactly on the
  benchmarked cases
- the local timing gain is modest rather than dramatic
- that is still valid Sprint 46 evidence because the batch was about direct
  repeated-run measurement, not about forcing a large speedup claim

## Preserved Boundaries

Day 11 deliberately did **not** widen into:

- broad `bench_eigs.c` CLI redesign
- broad corpus/backend repeated-run sweeps
- public explicit eigensolver workspace APIs
- mandatory LOBPCG repeated-run benchmark work
- tutorial/example refresh

Interpretation:

- the batch stayed aligned with the actual Sprint 46 migration surface
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

The targeted Day 11 eigensolver follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`
- `./build/bench_eigs_reuse`

Representative direct follow-on outcomes:

- `test_eigs`
  - all `25` tests passed
- `test_eigs_thick_restart`
  - all `20` tests passed
- `test_eigs_lobpcg`
  - all `26` tests passed
- `example_eigs`
  - completed successfully across the nos4 / KKT / LOBPCG example slices
- `bench_eigs_reuse`
  - both repeated-run benchmark cases completed successfully
  - one-shot and reuse paths matched on convergence, iteration counts,
    residuals, and eigenvalues

## Sprint 46 Position After Day 11

After this batch, Sprint 46 has now landed:

- shared internal eigensolver workspace owner
- grow-m Lanczos migration
- thick-restart Lanczos migration
- LOBPCG migration
- compatibility-preserving public wrapper cleanup
- direct repeated-run benchmark evidence for the migrated Lanczos-family paths

Interpretation:

- the main structural workspace-reuse objective is now materially complete
- the sprint can move into documentation/residual audit and validation closeout
  work without another broad eigensolver migration batch

## Bottom Line

Day 11 delivered:

- a dedicated repeated-run eigensolver benchmark
- a bounded private internal benchmark seam
- compatibility-preserving public/internal implementation sharing
- measured local repeated-run evidence for grow-m and thick-restart Lanczos
- a fully green validation baseline

The measured effect is modest on this machine, but the benchmark is now real,
bounded, and truthful, which is the right Sprint 46 Day 11 outcome.
