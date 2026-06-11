# Sprint 65 Day 4: Benchmark Role Rerank and Canonical-Surface Candidates

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Take the Day 3 benchmark-role map and turn it into a smaller canonical
candidate set, a first normalization target set, and an explicit deferred
benchmark queue before output/taxonomy design begins.

## Day 4 Rerank

### 1. Canonical maintained performance-surface candidates

The strongest current candidates for Sprint 65’s canonical maintained
performance surface are:

- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

Why these four lead:

- each corresponds directly to a shipped repeated-run workflow or bounded
  backend lane
- each already has structured or naturally normalizable output
- each is narrower and more maintainable than the broader sweep harnesses
- together they cover the most important bounded Epic 6 performance stories:
  - repeated-run direct throughput and CSC follow-through
  - backend/path identity on the first backend-aware lane
  - iterative repeated-run efficiency
  - eigensolver repeated-run efficiency

### 2. Proof surfaces that stay important but should not define the first normalization batch

These should remain benchmark-side proof surfaces, but not Day 5’s first
canonical normalization batch:

- `bench_refactor`
- `bench_ldlt_csc`

Why they stay secondary for now:

- `bench_refactor`
  - high-signal workflow proof
  - still human-readable rather than stable-CSV first
  - overlaps materially with the more structured `bench_refactor_csc`
- `bench_ldlt_csc`
  - high-value backend-comparison surface
  - still mixes multiple interpretation modes
  - current docs do not yet interpret its maintained role as tightly as
    `bench_chol_csc`

### 3. Regression-sensitive runtime lane remains distinct

The regression-sensitive runtime lane should remain:

- `bench_scaling`
- `bench_fillin`
- `bench_colamd`
- `bench_reorder --skip-factor`
- possibly `bench_amd_qg`

This lane exists for bounded local/CI runtime signal, not for the same
product-facing interpretation burden as the canonical maintained surface.

### 4. Exploratory or later benchmark queue

Sprint 65 should explicitly keep these out of the first canonical batch:

- `bench_main`
- `bench_convergence`
- `bench_svd`
- `bench_bicgstab`
- `bench_eigs`
- broader `bench_reorder` sweep modes
- `bench_amd_qg` as a long-term canonical signal unless later justified

## First Normalization Target Set

The first normalization target set should be:

- benchmark binary output:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- benchmark/docs/policy explanation:
  - `benchmarks/README.md`
  - `README.md`
  - `docs/maintainer_guide.md`

This set is small enough to support:

- stable category vocabulary
- stable machine-readable output expectations
- explicit path/backend identifiers where relevant
- a believable maintained canonical performance story

## Deferred Queue

Sprint 65 should not absorb the following into the first normalization pass:

- broad exploratory comparison harnesses
- mixed-role benchmark behavior that still needs interpretation cleanup
- proof surfaces whose current output shape or narrative is weaker than the
  top canonical candidates

Concretely:

- `bench_main`
- `bench_convergence`
- `bench_svd`
- `bench_bicgstab`
- `bench_eigs`
- broader `bench_reorder`
- `bench_amd_qg`
- first-batch canonical treatment for `bench_refactor`
- first-batch canonical treatment for `bench_ldlt_csc`

## Day 4 Exit State

Sprint 65 now has a smaller and sharper target set than the original broad
epic review:

- one four-surface canonical candidate set
- one bounded first normalization batch
- one distinct regression-sensitive runtime lane
- one explicitly deferred exploratory queue

That gives Day 5 an exact design surface instead of another generic benchmark
cleanup prompt.
