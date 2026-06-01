# Sprint 51 Day 12: Post-Landing Compatibility Audit

## Objective

Re-audit the landed Sprint 51 public lifecycle surface against the Sprint 50
contract and compatibility fence, with emphasis on public caller truth rather
than only compiler/test success.

## Surfaces Rechecked

Primary public contract surfaces:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `README.md`

Adoption / caller-story surfaces:

- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/README.md`

Supporting context surfaces:

- `docs/tutorial.md`
- `docs/maintainer_guide.md`

## Findings

### 1. One-shot entries still hold their intended compatibility position

The landed branch still presents LU / Cholesky / LDL^T one-shot entry points
as:

- fully supported
- first-class peer APIs
- the simple/default path for one-off direct solves

This remains true in both the family headers and the top-level README.

### 2. The repeated-run direct story still uses the intended public vocabulary

The shared direct repeated-run path remains centered on:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many
- explicit free

That vocabulary still anchors the live caller story across the shared header,
the README, the strongest example, and the strongest repeated-run benchmarks.

### 3. Reuse semantics remain honest and bounded

The landed surfaces continue to describe refactorization as reusing:

- symbolic/permutation/setup state

while not preserving:

- old numeric factor contents

The audit did not find any touched public surface that now overstates numeric
state reuse.

### 4. The main adoption-side drifts are resolved

The previously explicit caller-surface drifts are now gone:

- `examples/README.md` includes `example_analysis`
- `benchmarks/README.md` matches the live Cholesky repeated-run benchmark
  ownership for `bench_refactor` and `bench_refactor_csc`

So the strongest repeated-run direct example/benchmark docs now align with the
live landed branch state.

### 5. No Day 12 corrective code patch is needed

The audit did not surface any residual contradiction that requires more Sprint
51 implementation work before validation.

The remaining queue is normal closeout work:

- full validation sweep
- closeout / handoff synthesis

## Residual Drift List

No blocker-level residual drift remains.

The remaining items are normal final-sprint tasks rather than compatibility
fixes:

- Day 13 full validation
- Day 14 closeout/handoff write-up

## Pre-Validation Checklist

Day 13 should run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- reviewed CMake `ctest -N` parity recheck (`53`)
- targeted Sprint 51 follow-ons:
  - `./build/example_analysis`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

## Bottom Line

Sprint 51’s live branch state now matches the Sprint 50 contract in practice:

- one-shot entries remain first-class
- repeated direct runs remain analysis/factors-centric
- reuse semantics remain correctly bounded
- the strongest adoption surfaces are aligned
- no hidden scope creep remains before the validation sweep
