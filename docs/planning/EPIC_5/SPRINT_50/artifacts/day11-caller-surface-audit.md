# Sprint 50 Day 11 Artifact: Caller-Surface Audit

## Purpose

Audit the current caller-facing docs and example/benchmark surfaces against the
finished Sprint 50 direct repeated-run lifecycle contract, then identify the
smallest high-signal adoption set for later implementation sprints.

## Audit Scope

This pass checked:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

The question was not “which files mention direct solvers?” but:

- which surfaces should stay one-shot-first
- which surfaces should explain the explicit repeated direct lifecycle
- which surfaces should only cross-reference it
- where the current wording already drifts from the live repo state

## Surfaces Already Aligned Enough

### 1. `include/sparse_analysis.h`

Status:

- already the strongest direct repeated-run contract surface

Why:

- it already teaches analyze once / factor / refactor / solve / free
- it already uses zeroed `analysis` and `factors` in the public workflow
- it already frames same-pattern refactorization as the core repeated-run
  value

Needed later work:

- alignment and sharpening against the Day 8 wording
- not a conceptual rewrite

### 2. `include/sparse_lu.h`

Status:

- aligned enough as a one-shot-first surface

Why:

- it is explicit about in-place factorization
- it teaches `sparse_copy()` before factorization
- it does not pretend to be the repeated-run direct lifecycle surface

Needed later work:

- likely only a small relationship note or cross-reference boundary if Sprint
  51 header work touches it

### 3. `include/sparse_cholesky.h`

Status:

- aligned enough as a one-shot-first surface

Why:

- it is explicit about in-place factorization and copy-first discipline
- it already teaches the matrix-mutation contract honestly

Needed later work:

- likely only small wording alignment if the explicit repeated-run story is
  cross-referenced from the family headers

### 4. `include/sparse_ldlt.h`

Status:

- aligned enough as a factor-object one-shot surface

Why:

- it already documents the separate output-factor object model clearly
- it already requires original/identity-permutation matrix input

Needed later work:

- probably only bounded wording alignment, not a structural rewrite

## Surfaces That Should Explain The Advanced Lifecycle Path

### 1. `README.md`

Status:

- partially aligned, but under-centered for direct repeated-run guidance

What it already does well:

- includes `sparse_analysis.h` in the API overview
- names the analyze-once / factor-many workflow in features and key functions
- documents repeated-run lifecycle handles for iterative and eigensolver code

What it still lacks:

- a direct migration-path section equivalent to the Sprint 49 iterative/eigs
  repeated-run explanation
- explicit caller guidance for:
  - stay on one-shot direct APIs for one-off solves
  - use analysis/factor/refactor for stable-pattern repeated direct solves
  - reuse preserves symbolic/permutation setup, not old numeric factor state

Later role:

- should explain the advanced direct repeated-run path at top level

### 2. `examples/example_analysis.c`

Status:

- already the strongest shipped direct repeated-run teaching surface

Why:

- it demonstrates analyze once / factor / solve / refactor / solve directly
- it already matches the repeated-run public lifecycle contract in spirit

Later role:

- should become the main example-surface anchor for the advanced lifecycle path

## Surfaces That Should Stay One-Shot-First

### 1. Most small direct examples

These should stay one-shot-first:

- simple LU solve examples
- simple Cholesky solve examples
- examples whose main job is basic factor-and-solve teaching

Reason:

- the Sprint 50 contract explicitly preserves one-shot direct APIs as the
  simple/default path for one-off or low-context solves

### 2. `docs/tutorial.md`

Status:

- should remain mostly one-shot-first in its introductory direct-solver flow

Why:

- the current tutorial is a practical getting-started guide
- it already teaches copy-before-factorization and identity-permutation
  discipline in several sections

What it lacks:

- a bounded later direct repeated-run section or cross-reference

Later role:

- keep the main direct-solver walkthrough simple
- add only a bounded explicit repeated-run note or cross-reference, not a broad
  rewrite

### 3. `examples/README.md`

Status:

- should stay mostly one-shot-first by design

Why:

- that file is intentionally small and example-local
- it already states that shipped examples lean on one-shot APIs

What it still needs later:

- add `example_analysis` explicitly, since it is the clearest shipped direct
  repeated-run example
- keep the file scope-correct rather than turning it into a second migration
  guide

## Surfaces That Should Mostly Cross-Reference

### 1. Family-specific direct headers

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

These should not all be rewritten into full repeated-run guides. They should
stay family-local and, if needed, point to:

- `include/sparse_analysis.h`
- top-level `README.md`

### 2. Backend/perf-heavy benchmark docs

- `benchmarks/README.md`
- `benchmarks/bench_refactor_csc.c` surrounding docs

These should reflect the explicit repeated-run story where directly relevant,
but they should not become the main public migration narrative.

## Concrete Documentation Drift Found

### 1. `benchmarks/README.md` mislabels `bench_refactor`

Current table entry:

- `bench_refactor` -> “LDL^T re-factor with cached symbolic”

Live repo reality:

- `bench_refactor.c` is a Cholesky analyze-once / factor-many benchmark

Why this matters:

- this is a real correctness/documentation drift, not only a future
  enhancement idea

Priority:

- high-signal later fix

### 2. `examples/README.md` omits `example_analysis`

Current state:

- the file explains why examples mostly stay one-shot-first
- but it does not list or explain `example_analysis`

Live repo reality:

- `example_analysis.c` is the clearest shipped direct repeated-run example

Why this matters:

- the best repeated-run direct example is currently absent from the examples
  index

Priority:

- high-signal later fix

### 3. `README.md` still lacks a direct repeated-run migration section parallel to Sprint 49’s iterative/eigs guidance

Current state:

- direct repeated-run capability is present in features and API-overview form
- but not framed as a caller decision path

Why this matters:

- top-level users can still miss the intended direct repeated-run story even
  though the contract now exists

Priority:

- later high-signal adoption target

## Terminology Alignment Notes

The main terminology alignment issue is not contradictory vocabulary. It is
missing emphasis.

### Terms that already align well

- analysis
- refactor
- same sparsity pattern
- identity permutations
- copy before in-place factorization

### Terms that need more explicit adoption later

- analyze once / factor-refactor many
- explicit repeated-run path
- simple/default path
- first-class peer entry points
- reuse preserves symbolic/permutation setup, not old numeric factor state

## Later Adoption Target List

### Highest-signal later updates

1. `README.md`
2. `examples/example_analysis.c` supporting docs around it
3. `examples/README.md`
4. `benchmarks/README.md`

### Lower-priority or bounded later updates

1. `docs/tutorial.md` repeated-run cross-reference only
2. family-specific direct headers, if touched during implementation
3. broader example/benchmark wording after the first direct lifecycle landing

## Highest-Value Day 11 Conclusions

### 1. The strongest repeated-run direct caller surface already exists, but it is not yet centered in the docs

That surface is:

- `include/sparse_analysis.h`
- `examples/example_analysis.c`

### 2. Most direct docs should not be converted into full repeated-run guides

The right shape is selective:

- top-level README explains the advanced repeated-run path
- `sparse_analysis.h` remains the main header contract
- small examples and tutorial flows stay mostly one-shot-first

### 3. Two real contradictions are now explicitly recorded

- `benchmarks/README.md` mislabels `bench_refactor`
- `examples/README.md` omits `example_analysis`

### 4. Later caller-surface work is now bounded

Sprint 51+ does not need to update every direct-facing doc. It needs to fix the
small set of high-signal surfaces that actually govern the repeated-run public
story.
