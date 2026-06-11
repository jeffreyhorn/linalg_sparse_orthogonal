# Sprint 65 Day 3: Benchmark Role Audit

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Reduce the broad Sprint 65 “performance governance” problem to one explicit
benchmark-role map by classifying the live benchmark binaries into
regression-sensitive, proof, and exploratory lanes from the current repo
state.

## Authoritative Audit Inputs

- `benchmarks/README.md`
- `Makefile`
- `README.md`
- `docs/maintainer_guide.md`
- benchmark file headers under `benchmarks/*.c`
- current benchmark inventory from the live tree

## Live Benchmark Inventory

The repo currently ships `16` benchmark binaries:

- `bench_amd_qg`
- `bench_bicgstab`
- `bench_chol_csc`
- `bench_colamd`
- `bench_convergence`
- `bench_eigs`
- `bench_eigs_reuse`
- `bench_fillin`
- `bench_iterative_reuse`
- `bench_ldlt_csc`
- `bench_main`
- `bench_refactor`
- `bench_refactor_csc`
- `bench_reorder`
- `bench_scaling`
- `bench_svd`

## Day 3 Role Map

### 1. Regression-sensitive runtime lane

These are the binaries the repo is already using as bounded, CI-friendly
runtime signals rather than as broad product-facing proof:

- `bench_scaling`
- `bench_fillin`
- `bench_colamd`
- `bench_amd_qg`
- `bench_reorder --skip-factor`

Why they fit here:

- `make bench-fast` already selects this subset for PR runtime signal
- they are narrower and faster than the heavier benchmark catalog
- they are useful as drift sentinels without needing to become the main
  product-facing benchmark story

### 2. Proof-oriented benchmark lane

These are the strongest current benchmark-side proof surfaces:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`
- likely `bench_ldlt_csc`

Why they fit here:

- they correspond directly to shipped public workflows or bounded backend lanes
- docs already interpret their outputs as specific workflow/backend proof
- they sit next to strong non-benchmark proof homes in `tests/` and examples
- they are narrower and more maintainable than the broader sweep harnesses

### 3. Exploratory or broad comparison lane

These are the broad sweep or comparison tools that remain valuable, but do not
read like first candidates for normalized canonical governance:

- `bench_main`
- `bench_convergence`
- `bench_svd`
- `bench_bicgstab`
- `bench_eigs`
- broader `bench_reorder` modes outside `--skip-factor`

Why they fit here:

- they cover wider mode combinations or corpus sweeps
- their outputs are less obviously stable as authoritative regression signals
- they are better aligned with developer investigation than with the bounded
  maintained proof surface

## Strongest Current Category Mismatches

### 1. `bench_reorder` mixes two roles

It currently acts as both:

- a CI-friendly runtime sentinel through `--skip-factor`
- a broader exploratory cross-ordering and threshold sweep harness

Sprint 65 should separate those meanings explicitly instead of letting one
binary imply one stable role.

### 2. `bench_amd_qg` is in the runtime subset but reads as a historical A/B harness

Its header is explicitly about comparing the production quotient-graph AMD
against a reconstituted deleted bitset implementation. That still makes it
useful, but it is weaker as a long-term canonical benchmark story than the
other runtime sentinels.

### 3. `bench_main` is too broad to be the canonical benchmark face of the repo

It covers:

- LU / Cholesky one-shot
- SpMV
- iterative mode
- multiple input modes
- reorder selection

That makes it useful as a broad compatibility harness, but too wide and
multi-purpose to serve as the first normalized canonical performance surface.

### 4. `bench_ldlt_csc` is close to the proof lane, but under-interpreted

It is a strong backend-comparison surface and appears in the Sprint 65 rerun
set, but the current docs do not yet interpret it as explicitly as
`bench_chol_csc` or the repeated-run proof surfaces.

## First Canonical-Surface Candidate Set

The strongest current candidate set for the smaller maintained Sprint 65
surface is:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`
- likely `bench_ldlt_csc`

This is materially smaller and sharper than treating the full benchmark
catalog as equally authoritative.

## Day 3 Exit State

Sprint 65 now has a concrete benchmark-role map:

- one regression-sensitive runtime lane
- one proof-oriented candidate canonical lane
- one exploratory comparison lane
- one explicit mismatch queue where current binaries still mix roles

That gives Day 4 a real current-state taxonomy to rerank rather than another
generic “benchmarks need cleanup” prompt.
