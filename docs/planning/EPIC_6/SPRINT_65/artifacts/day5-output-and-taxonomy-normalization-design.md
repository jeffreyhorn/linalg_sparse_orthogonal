# Sprint 65 Day 5: Output and Taxonomy Normalization Design

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Define the exact benchmark taxonomy vocabulary, normalized output contract, and
ownership split for the first Sprint 65 normalization batch before any
benchmark binaries or docs move.

## Maintained Taxonomy

Sprint 65 should use three explicit maintained categories:

### 1. `regression-sensitive`

Meaning:

- bounded local/CI runtime sentinel
- strong enough for repeatable drift detection
- not automatically a product-claim benchmark

### 2. `proof`

Meaning:

- benchmark-side evidence for a bounded shipped workflow or backend lane
- narrower and more interpretable than broad comparison or sweep harnesses
- may be machine-readable or human-readable

### 3. `exploratory`

Meaning:

- broader developer comparison, corpus sweep, or historical A/B surface
- useful, but outside the first authoritative regression/canonical lane

## First-Batch Output Families

The first normalization target set already splits into two output families:

### 1. Structured CSV proof surfaces

- `bench_refactor_csc`
- `bench_chol_csc`

These already have explicit machine-readable schemas and should become the
strongest first normalization anchors.

### 2. Human-readable repeated-run proof summaries

- `bench_iterative_reuse`
- `bench_eigs_reuse`

These already have stable conceptual output:

- one-shot timing
- reuse timing
- speedup
- last-run solver summary

But they do not yet expose a stable machine-readable schema.

## Normalized Output Contract

The first-batch normalized contract should require:

- stable benchmark identity field:
  - benchmark or case label
- stable category field:
  - `proof`
  - later `regression-sensitive`
  - later `exploratory` only when intentionally surfaced
- stable workflow/scenario field where applicable
- stable timing fields with `_ms` suffix for machine-readable timing output
- explicit path/backend identity fields where relevant
- speedup fields only where the comparison semantics are honest and stable
- residual or agreement fields where correctness signal is part of the
  benchmark’s maintained story

This is intentionally a compact shared contract with family-local extensions,
not a forced universal giant schema.

## Ownership Split

### Benchmark binary output owns

- stable emitted fields
- stable field names
- family-local scenario labels

### `benchmarks/README.md` owns

- category and usage explanation
- per-benchmark schema description
- interpretation notes for path/speedup/residual fields

### `README.md` owns

- compact top-level performance-governance story
- where the maintained proof surfaces live

### `docs/maintainer_guide.md` owns

- authoritative category policy
- which surfaces are canonical candidates versus proof-only versus runtime
  sentinels

### CI/reporting owns

- only bounded runtime-sentinel use
- no broad claim-bearing benchmark governance rewrite unless the local proof
  surface stays maintainable

## First Implementation Batch Contract

The first implementation batch should target:

- benchmark binaries:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- explanation layers:
  - `benchmarks/README.md`
  - `README.md`
  - `docs/maintainer_guide.md`

It should preserve:

- no misleading benchmark claims
- no unstable pseudo-regression gates
- no output churn without governance clarity as the reason
- no fake claim that all benchmark binaries belong to one canonical set

## Day 5 Exit State

Sprint 65 now has an explicit normalization design:

- one three-class benchmark taxonomy
- one shared normalized output contract with family-local extensions
- one clean ownership split across binaries/docs/policy/CI
- one bounded first implementation contract for the selected target set
