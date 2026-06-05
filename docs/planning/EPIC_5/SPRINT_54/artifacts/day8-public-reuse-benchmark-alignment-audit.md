# Sprint 54 Day 8 - public reuse benchmark alignment audit

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Audit the repeated-run benchmark surfaces against the final Sprint 54 public
support boundary before editing benchmark drivers, so the next landing batch
can stay small, explicit, and free of benchmark-framework churn.

## What Day 8 audited

The audit focused on the benchmark surfaces that are supposed to prove the
public repeated-run handle story directly:

- `bench_iterative_reuse.c`
- `bench_eigs_reuse.c`
- `benchmarks/README.md`

Cross-check references:

- `README.md`
- `examples/README.md`
- Sprint 54 Day 4 / Day 6 / Day 7 decisions and outcomes

## Main findings

### 1. No reuse benchmark is still proving only an internal seam

The most important positive result is that the benchmark drift is not a
truthfulness failure at the public-contract layer:

- `bench_iterative_reuse.c`
  - already uses `sparse_iter_handle_prepare_*`
  - already uses `*_with_handle(...)`
  - already compares one-shot vs public handle paths
- `bench_eigs_reuse.c`
  - already uses `sparse_eigs_handle_prepare(...)`
  - already uses `sparse_eigs_sym_with_handle(...)`
  - already enforces parity between one-shot and repeated-run outcomes

Interpretation:

- Day 8 did not find a benchmark still centered on private/internal reuse
  machinery
- the remaining problem is support-set completeness rather than public-surface
  dishonesty

### 2. The iterative reuse benchmark now lags the final supported set by exactly one family: MINRES

After Day 6, Sprint 54’s supported iterative repeated-run public handle set is:

- `CG`
- `GMRES`
- `MINRES`

But `bench_iterative_reuse.c` still only covers:

- `CG`
- `GMRES`

Interpretation:

- `MINRES` is now the highest-value iterative benchmark alignment target
- this is a real support-set drift, not an optional nice-to-have
- the likely Day 9 fix is one bounded `MINRES` repeated-run case in the
  existing reuse driver

### 3. The eigensolver reuse benchmark now lags the final supported set by one backend path: explicit LOBPCG

After Day 7, the supported public repeated-run eigensolver handle surface now
reads explicitly as covering:

- grow-m Lanczos
- thick-restart Lanczos
- explicit LOBPCG

But `bench_eigs_reuse.c` still only covers:

- grow-m Lanczos on `nos4`
- thick-restart Lanczos on `bcsstk14`

Interpretation:

- the strongest eigensolver benchmark drift is now explicit LOBPCG undercoverage
- the smallest likely Day 9 fix is one bounded LOBPCG repeated-run case
- this should stay narrow and fixed-shape rather than turning into a second
  general backend sweep

### 4. `benchmarks/README.md` currently under-documents the public repeated-run benchmark proof surfaces

The benchmark README names many permanent benchmark binaries, but it currently
omits the two dedicated repeated-run public-handle drivers:

- `bench_iterative_reuse`
- `bench_eigs_reuse`

Interpretation:

- benchmark-local docs currently understate the benchmark proof surface Sprint
  54 is relying on
- the docs gap is real but small
- it belongs in the Day 9 benchmark-alignment batch, not as a separate
  framework or README rewrite

## Explicit non-goals

Day 8 also fixed the benchmark non-goal boundary more sharply:

- do not add `BiCGSTAB` repeated-run-handle benchmarking
  - `BiCGSTAB` remains outside the Sprint 54 public handle boundary
- do not add block iterative repeated-run-handle benchmarking
  - block workflows remain compatibility surfaces rather than first public
    handle targets
- do not turn `bench_eigs_reuse` into a broad backend/preconditioner sweep
  - `bench_eigs` already owns that space
- do not redesign the benchmark framework or CLI just to close the Sprint 54
  support-set gap

## Ranked target list for the next batch

1. `bench_iterative_reuse.c`
   - add one bounded repeated-run `MINRES` case
2. `bench_eigs_reuse.c`
   - add one bounded explicit LOBPCG repeated-run case
3. `benchmarks/README.md`
   - document the two reuse benchmarks and their intentionally narrow
     public-handle proof role

## Conclusion

Day 8 reduces the benchmark queue to a small, concrete, support-boundary
alignment set:

- no benchmark is still proving only an internal reuse seam
- the remaining drift is support-set completeness
- the next batch can stay tightly bounded to:
  - `MINRES`
  - explicit `LOBPCG`
  - small benchmark README synchronization
