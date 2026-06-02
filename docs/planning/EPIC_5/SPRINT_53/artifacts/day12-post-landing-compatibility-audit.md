# Sprint 53 Day 12: Post-Landing Compatibility Audit

## Purpose

Day 12 checks the landed Sprint 53 branch against the compatibility and scope
fence preserved from Sprints 50-52. The goal is to confirm that the CSC
follow-through work strengthened the intended direct-solver story without
quietly expanding it into a new public abstraction, raw storage exposure, or
overclaimed repeated-run contract.

## Main Day 12 Result

The landed Sprint 53 branch still matches the preserved Epic 5 fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- repeated direct runs remain centered on:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - analyze once
  - factor / solve
  - refactor / solve many
- reuse semantics remain honestly bounded:
  - symbolic / permutation setup is reused
  - stale numeric factor contents are not
- the CSC work did not broaden into:
  - raw CSC/native storage exposure
  - a generic public direct handle
  - a broad direct-solver API redesign

No blocker-level compatibility drift surfaced before Day 13.

## Audit Scope

Day 12 checked the landed branch across:

- `README.md`
- `benchmarks/README.md`
- `include/sparse_analysis.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `docs/maintainer_guide.md`
- Sprint 53 Day 1 baseline notes
- Sprint 53 Day 10-11 artifacts

## Compatibility Findings

### 1. One-shot direct APIs still read as first-class, not demoted compatibility leftovers

The current public surfaces still present:

- one-shot Cholesky as the normal in-place SPD path
- one-shot LDL^T as the normal owned-factor indefinite path
- one-shot LU / Cholesky / LDL^T as first-class peer entry points in the
  top-level README repeated-run section

Interpretation:

- Sprint 53 did not quietly reframe explicit analysis/factors lifecycle work as
  a replacement-only path
- the compatibility promise from Sprint 50 still holds

### 2. The repeated-run direct story is still analysis/factors-centric

The shared repeated-run contract is still centered on:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

Interpretation:

- Sprint 53 deepened the existing lifecycle rather than inventing a new public
  abstraction
- the branch still matches the Sprint 50-52 design choice to stay
  analysis-centric first

### 3. Reuse/refactor semantics remain honestly bounded

The landed wording and tests still agree on the actual contract:

- reuse preserves symbolic / permutation setup
- old numeric factor contents are rebuilt, not incrementally preserved
- `sparse_refactor_numeric(...)` remains the same-pattern numeric refresh path
- gross-structure rejection is still cheap and bounded, not a full structural
  verifier

Day 11 strengthened that truthfulness by proving the cheap `nnz`-drift guard
and old-factor preservation on the high-value indefinite KKT repeated-run path.

Interpretation:

- Sprint 53 improved proof strength without inflating the public promise

### 4. The CSC follow-through stayed internal and bounded

The landed Sprint 53 work clarified and deepened CSC behavior, but the public
surface still does not expose:

- raw `LdltCsc` / Cholesky CSC storage
- direct CSC-native factor containers as public API
- a generic "CSC direct handle" abstraction

Interpretation:

- the sprint stayed within the explicit non-goal fence from Sprint 50
- the repo still presents CSC work as implementation and dispatch depth behind
  stable public direct APIs

### 5. README and benchmark claims now match the measured/proved scope

The top-level README now says:

- Cholesky CSC dispatch is the simpler family-local case
- LDL^T CSC dispatch is the layered CSC-pipeline case
- `bench_refactor_csc --indefinite-kkt` is the bounded indefinite repeated-run
  proof surface

The benchmark-local README still carries the detailed benchmark contract, while
the top-level README stays compact.

Interpretation:

- the landed branch no longer has a major top-level truthfulness mismatch
- Sprint 53's public docs now read from the measured / proved state rather than
  from the older pre-follow-through mental model

## No-Drift Conclusions

Day 12 did **not** find evidence of:

- one-shot direct API demotion
- raw CSC/native storage exposure
- generic direct-handle redesign
- benchmark or README wording that outruns measured indefinite proof
- hidden compatibility break in the repeated-run direct lifecycle

## Day 13 Validation Checklist

Run from the landed Day 12 state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- targeted Sprint 53 follow-ons:
  - `./build/bench_refactor_csc`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_integration`
  - `./build/example_analysis`

## Future-Facing Queue

Still future-facing rather than a Sprint 53 blocker:

- broader benchmark modernization beyond the bounded `bench_refactor*` work
- any future public factor-container redesign discussion
- any broader tutorial/example rewrite around CSC internals
- deeper structural-pattern verification beyond the current cheap guard

## Operational Result

Sprint 53 now enters Day 13 from a clean audited state:

1. the preserved scope fence still holds
2. the README / benchmark / header language does not outrun the branch
3. the remaining queue is future-facing rather than a hidden closeout defect
