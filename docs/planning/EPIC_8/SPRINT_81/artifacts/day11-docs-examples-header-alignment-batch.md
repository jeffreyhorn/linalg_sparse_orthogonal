# Sprint 81 Day 11 - Docs / Examples / Header Alignment Batch

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Land the bounded follow-through from Day 10 so the public repeated-run direct
contract reads truthfully after the Day 9 workflow-convergence batch without
spreading into a generic README, maintainer, benchmark, or example cleanup.

## Main Result

The bounded Day 11 follow-through batch landed only in:

- `include/sparse_analysis.h`

The required contract correction is now explicit:

- the `sparse_factor_numeric(...)` public header block no longer describes the
  shared Cholesky repeated-run CSC-aware path as larger-problem-only
- it now says directly that the shared Cholesky repeated-run path stays on the
  analysis-backed CSC-aware route for all problem sizes
- it also makes the residual split clearer:
  - LDL^T remains analysis-backed CSC-aware with its documented
    pivot-prepass-conditioned fallback
  - LU remains the direct family that still delegates through the one-shot
    routine

## No-Op Support Surfaces

No support-only follow-through was actually needed:

- `README.md` already stayed broadly truthful
- `docs/maintainer_guide.md` already stayed broadly truthful
- `benchmarks/README.md` already stayed aligned with the landed proof and
  benchmark ownership split
- `examples/README.md` already stayed aligned with the landed repeated-run
  adoption split

## Preserved Fence

- no new proof-code expansion
- no benchmark logic changes
- no generic docs/examples sweep
- no reopening of implementation surfaces

## Validation

- `make format` passed
- `make lint` passed
- `make test` passed

## Exit State

- Sprint 81's public repeated-run header contract now matches the landed Day 9
  workflow behavior.
- The docs/examples/header follow-through batch stayed bounded to one required
  surface.
- Day 12 can now focus on final proof alignment instead of support-surface
  drift.
