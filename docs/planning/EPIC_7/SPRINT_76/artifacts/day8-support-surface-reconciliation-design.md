# Sprint 76 Day 8 Artifact: Support-Surface Reconciliation Design

Date: 2026-06-17
Branch: sprint-76

## Purpose

Define the bounded documentation and policy follow-through contract for the
landed Day 6 canonical report bundle before any support-surface edits begin.

## Main Result

Sprint 76 now has one exact support-surface reconciliation batch:

- required Day 9 batch:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- support only if wording truly forces it:
  - `README.md`

## Why This Is the Right Next Batch

The strongest remaining contradiction is now support-surface drift, not
workflow drift:

- the workflow and bundle already landed in:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- the benchmark-local and maintainer-policy surfaces still describe the older
  manifest-only bundle too narrowly

That makes the best next move a bounded wording reconciliation batch, not:

- another workflow/code batch
- threshold-policy widening
- benchmark-driver churn

## Support-Surface Contract

### `benchmarks/README.md`

This should become the clearer benchmark-local explanation of the stronger
canonical report bundle:

- one CSV per canonical maintained benchmark still remains true
- `manifest.txt` still remains true
- the file should now also name:
  - `index.tsv`
  - bounded report-label support
  - bounded git metadata support
- it should still keep the same narrow reading:
  - threshold-free reporting
  - artifact-friendly comparison
  - no pass/fail portability claim

### `docs/maintainer_guide.md`

This should move with the benchmark-local README because it owns the
authoritative policy reading:

- current threshold-free reporting surface
- canonical/runtime/exploratory category split
- ownership split between benchmark binaries, benchmark docs, README, and the
  maintainer guide

It should now reflect the landed Day 6 bundle shape without widening the
policy claim.

### `README.md`

This remains support-only because its current compact statement is still
broadly truthful:

- `make bench-canonical-report` still writes one bounded snapshot of the
  maintained canonical surface
- the top-level README does not need to become the detailed bundle-schema
  owner unless the Day 9 wording proves the compact summary became inaccurate

## Preserved Guarantees

The Day 9 batch must preserve:

- threshold-free interpretation
- the same four canonical maintained benchmark emitters
- benchmark binaries as owners of CSV row semantics and proof fields
- runtime and exploratory lanes staying outside the canonical report bundle
- `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg` remaining
  separate from the canonical report surface

## Explicit Non-Touch Set

The support-surface batch explicitly does not include:

- `scripts/bench_canonical_report.sh`
- `Makefile`
- canonical benchmark driver edits
- threshold-policy work
- reviewed proof-owner tests
- examples
- broad benchmark/docs cleanup outside the targeted canonical report wording

## Day 9 Implication

The Day 9 batch should therefore start from:

- exact required center:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- support only if truly forced:
  - `README.md`
- explicitly not next:
  - workflow/code reopen
  - threshold-policy widening
  - benchmark-driver edits
