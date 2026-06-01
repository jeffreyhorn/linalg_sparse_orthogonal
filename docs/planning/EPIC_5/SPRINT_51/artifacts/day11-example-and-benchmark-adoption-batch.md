# Sprint 51 Day 11: Example and Benchmark Adoption Batch

## Objective

Align the strongest shipped repeated-run direct example and benchmark docs with
the now-live public lifecycle path, without broadening back into source or
tutorial churn.

## Files Changed

- `examples/README.md`
- `benchmarks/README.md`

## What Landed

### 1. Added the missing `example_analysis` entry

`examples/README.md` now includes the strongest shipped repeated-run direct
example explicitly.

The new entry calls out:

- zero-init `sparse_analysis_t` / `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many

Why this mattered:

- `example_analysis.c` was already the best direct repeated-run teaching
  surface in the repo
- the README omission made that workflow harder to discover than the one-shot
  examples

### 2. Corrected the `bench_refactor` benchmark description

`benchmarks/README.md` no longer describes `bench_refactor` as an LDL^T
re-factor benchmark.

It now states the live behavior:

- `bench_refactor`
  - Cholesky analyze-once / refactor-many path
- `bench_refactor_csc`
  - the same repeated-run caller story, plus CSC/supernodal comparison

Why this mattered:

- the benchmark table had drifted behind the real driver ownership
- Sprint 51’s benchmark-side caller story is now aligned with the public
  direct lifecycle path landed earlier in the sprint

## What Did Not Change

The batch intentionally did not:

- touch `examples/example_analysis.c`
- touch `benchmarks/bench_refactor.c`
- broaden into tutorial or top-level README churn
- change any public API or implementation behavior

The source surfaces were checked and already matched the intended Sprint 51
story closely enough to leave alone.

## Validation

This was a documentation-only batch, so the `make format && make lint && make
test` C/C-header gate was not required.

Targeted touched-surface sanity checks still ran and passed:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative direct outcomes:

- `example_analysis` kept repeated-run residuals at `4.44e-16`
- `bench_refactor` still completed the one-shot vs analyze-once comparison
- `bench_refactor_csc` still completed the repeated-run linked-list vs CSC
  comparison on `nos4`

## Bottom Line

Sprint 51’s strongest repeated-run direct adoption surfaces are now aligned at
the docs level:

- the example index includes `example_analysis`
- the benchmark index accurately labels `bench_refactor`
- the remaining work can move on to compatibility audit / closeout instead of
  basic caller-surface discoverability repair
