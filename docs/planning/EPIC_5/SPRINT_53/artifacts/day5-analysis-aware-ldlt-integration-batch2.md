# Sprint 53 Day 5: Analysis-Aware LDL^T Integration Batch II

## Purpose

Day 5 removes the next highest-value indefinite CSC seam by unifying the
shared scalar-prepass / resolved-analysis preparation front half that was still
duplicated between:

- the one-shot LDL^T CSC dispatch path
- the repeated-run analysis/factors LDL^T CSC path

It also adds direct public repeated-run proof on an indefinite same-pattern
refactor workload.

## Main Day 5 Result

Day 5 extracted a new shared internal helper:

- `ldlt_csc_prepare_resolved_analysis(...)`

That helper now owns the indefinite CSC preparation front half:

1. scalar BK pre-pass via `ldlt_csc_from_sparse(...)`
2. `ldlt_csc_eliminate_native(...)`
3. decision between direct caller-analysis reuse vs pre-permute + derived
   analysis fallback
4. construction of the final matrix / analysis pair for the later
   `ldlt_csc_factor_with_resolved_analysis(...)` helper

This complements the Day 4 helper instead of overlapping it:

- Day 4 unified CSC completion after the resolved analysis state already
  existed
- Day 5 unified the front half that decides what that resolved analysis state
  actually is

## Code Changes

### 1. Shared indefinite CSC preparation helper

`src/sparse_ldlt_csc_internal.h` now declares:

- `ldlt_csc_prepare_resolved_analysis(...)`

`src/sparse_ldlt.c` now implements it.

The helper accepts:

- the input matrix
- an optional caller analysis hint

It returns:

- the scalar pre-pass factor
- any owned pre-permuted matrix
- any derived analysis built on that pre-permuted matrix
- the final matrix / analysis pair that the CSC completion helper should use

### 2. The repeated-run shared LDL^T CSC path now uses the helper

`factor_ldlt_with_analysis_csc(...)` in `src/sparse_analysis.c` now delegates
the indefinite preparation decision through:

- `ldlt_csc_prepare_resolved_analysis(...)`

That removes the inlined duplicated logic for:

- scalar pre-pass
- BK-permutation comparison with caller analysis
- pre-permute + derived-analysis fallback

### 3. The one-shot CSC path now uses the same preparation helper

`ldlt_factor_csc_path(...)` in `src/sparse_ldlt.c` now also delegates its
preparation half through the same helper.

The one-shot path still naturally differs from the repeated-run path:

- it has no caller analysis hint
- it therefore always resolves through the pre-permuted matrix +
  `SPARSE_REORDER_NONE` derived analysis path

But that difference is now represented as one shared helper contract instead
of two similar hand-written orchestration blocks.

## Preserved Contract

Day 5 intentionally preserved the bounded Sprint 50-52 semantics:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- the scalar BK pre-pass still owns final symmetric permutation resolution
- reuse still preserves symbolic/permutation setup only when that resolved BK
  structure stays compatible with the caller analysis
- stale numeric factor values and old pivot choices are still not promised as
  reusable state

This is a cleanup and proof-strengthening batch, not a public contract
expansion.

## Regression Proof Added

Day 5 added focused integration proof in `tests/test_integration.c`:

- `test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt`

The test uses an above-threshold indefinite KKT matrix and checks:

1. explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)`
2. correct solve on the original matrix
3. same-pattern value perturbation on a fresh KKT matrix
4. `sparse_refactor_numeric(...)` success on the perturbed matrix
5. correct solve after refactor via the same public analysis/factors objects

That gives Sprint 53 more direct proof than before that the public LDL^T
repeated-run path works on an intended indefinite CSC same-pattern refresh
workflow.

## Validation

Because `*.c` / `*.h` changed, Day 5 ran the full required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Day 5 also ran the touched follow-ons justified by the batch:

- `./build/test_integration`
- `./build/test_ldlt`
- `./build/test_ldlt_csc`
- `./build/test_sprint20_integration`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/example_analysis`

Representative direct results:

- `test_integration` = `35 / 35`
- `test_ldlt` = `83 / 83`
- `test_ldlt_csc` = `95 / 95`
- `test_sprint20_integration` = `20 / 20`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.70x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`
- `example_analysis` residual = `4.44e-16`

## What Day 5 Solved

- reduced duplicated indefinite CSC preparation logic between shared and
  one-shot paths
- made the front-half decision boundary easier to reason about
- strengthened public repeated-run LDL^T indefinite refactor proof

## What Day 5 Did Not Solve

- public backend/telemetry dispatch wording is still later work
- LDL^T-specific factor-many benchmark proof is still a separate batch
- the scalar BK pre-pass still remains the authoritative pivot/permutation
  resolution step

## Operational Result

Sprint 53 now has a cleaner LDL^T CSC internal structure before the dispatch
days:

- shared preparation helper
- shared completion helper
- stronger same-pattern indefinite repeated-run proof

That leaves the next queue more clearly about dispatch reasoning and measured
factor-many evidence, not about duplicated indefinite CSC setup plumbing.
