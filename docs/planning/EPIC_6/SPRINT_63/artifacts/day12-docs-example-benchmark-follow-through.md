# Sprint 63 Day 12: Docs, Example, and Benchmark Follow-Through

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Align the highest-signal public and maintainer surfaces with the landed Sprint
63 lifecycle-uniformity story:

- tell the repeated-run direct failure-preserve rule directly in the README
  and tutorial
- keep example and benchmark ownership boundaries clear
- refresh the maintainer-side direct-family interpretation to the Sprint 63
  state

## Landed Surfaces

Public docs:

- `README.md`
- `docs/tutorial.md`

Local adoption/perf docs:

- `examples/README.md`
- `benchmarks/README.md`

Maintainer policy:

- `docs/maintainer_guide.md`

## Main Result

Sprint 63 Day 12 closes the remaining wording gap without reopening code work.

The batch stayed inside the Day 12 fence:

- no public-header widening
- no implementation changes
- no test changes
- no broad docs-density cleanup

## What Changed

### README and tutorial

The repeated-run direct workflow story now states the shipped failure contract
more explicitly:

- failed `sparse_refactor_numeric(...)` calls preserve the previous usable
  factor state
- the large-`n` CSC-backed Cholesky lane follows that same rule on same-pattern
  non-SPD failure and obvious nnz drift rejection

The tutorial now teaches that rule at the real adoption point: the handoff
from one-shot Cholesky to `example_analysis.c` and the explicit
analyze/factor/refactor lifecycle.

### Examples and benchmarks

The example and benchmark docs now separate their roles more cleanly:

- `example_analysis` stays the strongest adoption example for the repeated-run
  direct lifecycle
- `bench_refactor_csc` stays the main throughput/proof surface for the
  large-`n` CSC-backed repeated-run direct lane
- failure-preserve semantics stay intentionally owned by
  `tests/test_integration.c`, not by the adoption or benchmark docs

### Maintainer guide

The direct-family interpretation is now updated from Sprint 62 to Sprint 63.

It explicitly records:

- invalid LU pivot/reorder enums and invalid Cholesky reorder/backend enums
  reject before reorder or factor mutation begins
- the public repeated-run direct lifecycle preserves previous usable factors
  on refactor failure
- the large-`n` CSC-backed Cholesky lane follows that same old-factor-
  preservation rule on same-pattern non-SPD failure and obvious nnz drift

## Sanity Checks

Ran:

- `git diff -- README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
- `rg -n "old-factor|nnz drift|example_analysis|bench_refactor_csc|repeated-run direct" README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
- `wc -l README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
- `git status --short --branch`

Measured touched-surface result:

- `README.md`: `982 -> 988`
- `docs/tutorial.md`: `464 -> 469`
- `examples/README.md`: `142 -> 147`
- `benchmarks/README.md`: `246 -> 249`
- `docs/maintainer_guide.md`: `391 -> 398`

## Exit State

Sprint 63 Day 12 now hands off a cleaner final validation target:

- callers see the shipped repeated-run direct failure semantics directly
- example and benchmark docs point to the correct proof homes
- the maintainer guide owns the Sprint 63 direct-family interpretation
  explicitly
