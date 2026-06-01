# Sprint 51 Day 1 Artifact: Scope and Direct Lifecycle Implementation Baseline

## Purpose

Turn the Sprint 51 project-plan section plus the Sprint 50 closeout package
into a bounded implementation baseline for the first public direct-solver
lifecycle API batch.

## Main Day 1 Conclusion

Sprint 51 starts from a preserved implementation contract, not from design
ambiguity.

The direct repeated-run public story is already fixed around:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve many
- free explicitly

Sprint 51’s job is therefore to implement that contract through the main
direct-solver families while preserving the one-shot compatibility path.

## Preserved Starting Contract

Sprint 50 already fixed the key rules Sprint 51 must inherit:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- one-shot direct usage remains the simple/default path for one-off or
  low-context solves
- mutable-`SparseMatrix` one-shot behavior for LU / Cholesky remains an
  accepted compatibility tradeoff
- the public repeated-run direct path is analysis/factors-centric, not a new
  generic direct-handle redesign

Interpretation:

- Sprint 51 should not reopen the architecture search
- it should implement the already-bounded phase-1 lifecycle contract

## Preserved Validation Baseline

The maintained reviewed close state remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 51 should preserve the exact baseline language
- substantial public direct lifecycle batches should keep the reviewed CMake
  parity contract explicit

## Direct Public-Surface Starting Point

The main public repeated-run precedent is already real:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `examples/example_analysis.c`

The direct one-shot family-local surfaces remain the dominant simpler caller
path:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Interpretation:

- Sprint 51 does not need to create a repeated-run direct story from nothing
- it needs to make the existing analysis/factor/refactor story more concretely
  present in the headers and routed implementation path

## Highest-Risk Implementation Seams

### 1. Mutable matrix compatibility on LU / Cholesky

The one-shot direct story still depends on:

- copied-matrix preservation discipline
- in-place factorization behavior
- family-local one-shot semantics staying visible and supported

Risk:

- Sprint 51 could accidentally make the repeated-run path sound like a
  replacement rather than an additive peer path

### 2. Shared-vs-family-local wording drift

The shared repeated-run contract belongs in:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze/factor/refactor/solve/free wording

But family-local semantics still belong in:

- LU one-shot copy/pivot behavior
- Cholesky one-shot copy/backend behavior
- LDL^T factor-object and identity-permutation behavior

Risk:

- header edits could over-flatten real family differences

### 3. Reuse semantics becoming overstated

Sprint 50 already fixed the truthful sentence:

- reuse preserves symbolic/permutation setup, not old numeric factor state

Risk:

- Sprint 51 implementation and wrapper work could accidentally imply stronger
  persistence or backend-specific storage guarantees than the design allows

### 4. One-shot-vs-lifecycle behavior drift

The wrapper-preservation batch must ensure:

- one-shot entries remain supported
- repeated-run entries are real
- the two paths agree where they should

Risk:

- wrapper integration could introduce behavior mismatches or confusing support
  status

## Hotspot Map

The live file sizes already identify the strongest Sprint 51 landing surfaces:

- public headers:
  - `include/sparse_analysis.h` = `334`
  - `include/sparse_lu.h` = `327`
  - `include/sparse_cholesky.h` = `191`
  - `include/sparse_ldlt.h` = `310`
- implementation seams:
  - `src/sparse_analysis.c` = `614`
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
- repeated-run example / benchmark surfaces:
  - `examples/example_analysis.c` = `191`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
- strongest direct regression concentrations:
  - `tests/test_cholesky.c` = `535`
  - `tests/test_ldlt.c` = `2774`
  - `tests/test_etree.c` = `2962`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`

Interpretation:

- Sprint 51 should stay centered on a small public API batch
- the main implementation and regression hotspots are already obvious before
  Day 3’s header map and Day 4’s code landing

## Primary Repeated-Run Teaching Surface

`examples/example_analysis.c` already demonstrates the clearest intended public
direct repeated-run lifecycle:

- zeroed `sparse_analysis_t`
- zeroed `sparse_factors_t`
- analyze once
- factor
- solve
- refactor
- solve again
- explicit free

Interpretation:

- Sprint 51 should treat this as a primary adoption surface
- broad conversion of small one-shot examples remains lower-value than keeping
  the strongest repeated-run direct example aligned with the final phase-1 API

## Sprint 51 Workstreams

The implementation workstreams are now explicit:

1. public header surface
2. LU lifecycle integration
3. Cholesky lifecycle integration
4. LDL^T lifecycle integration
5. wrapper preservation
6. focused regression expansion
7. validation and closeout

## Highest-Value Day 1 Conclusions

### 1. Sprint 51 is the first implementation sprint for a design that is already fixed

The main direct repeated-run contract and the compatibility fence are already
done.

### 2. The analysis/factors path is the public repeated-run center of gravity

Sprint 51 should extend and route through that path, not compete with it.

### 3. The main risks are compatibility wording and behavior drift, not missing lifecycle vocabulary

The hardest part is preserving one-shot truth while making the repeated-run
path more concretely public.
