# Sprint 71 Day 10: Tutorial / Example / Benchmark Cross-Surface Design

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Define the bounded support-surface reconciliation still justified after the
front-door and header cleanups.

## Main Result

The exact Day 11 support batch should be:

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

`docs/maintainer_guide.md` remains support-only and should move only if the
Day 11 wording truly forces follow-through.

## Support Rerank

### Strongest remaining support cleanup center

- `docs/tutorial.md`

The tutorial still carries the densest repeated-run teaching-flow and
ownership-handoff explanation after the Day 6 and Day 9 landings.

### Support surfaces that should move with it

- `examples/README.md`
- `benchmarks/README.md`

These should move with the tutorial because they carry the example-side and
benchmark-side halves of the same support split:

- examples = adoption and workflow teaching
- benchmarks = retained workflow/performance proof
- tests = regression/oracle/property guarantees

### Support-only policy authority

- `docs/maintainer_guide.md`

This remains the correct policy surface and is not the primary cleanup center
for the Day 11 batch.

## Preserved Support Split

The Day 11 batch must preserve:

- the tutorial as the step-by-step teaching flow
- examples as adoption and workflow-teaching surfaces
- benchmarks as retained workflow/performance proof surfaces
- tests as regression/oracle/property owners
- `make bench-canonical-report` as threshold-free artifact reporting

## Day 11 Non-Touch Set

The support batch should not touch:

- `README.md`
- `INSTALL.md`
- `include/sparse_cholesky.h`
- other public headers
- implementation `src/` files
- permanent proof-owner test files
- platform/install workflow files

## Exit State

Sprint 71 Day 10 closes with one exact support-surface design:

1. `docs/tutorial.md` is the strongest remaining support cleanup center
2. `examples/README.md` and `benchmarks/README.md` should move with it
3. `docs/maintainer_guide.md` remains policy authority and support-only
4. the Day 11 support batch is explicitly bounded
