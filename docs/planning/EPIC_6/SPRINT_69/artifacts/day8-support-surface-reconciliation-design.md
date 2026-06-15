# Sprint 69 Day 8: Support-Surface Reconciliation Design

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Turn the Day 7 rerank into one exact support-surface reconciliation contract
so the next Sprint 69 batch closes the adoption/proof-owner drift without
widening into headers, implementation, or project-level closeout surfaces.

## Chosen Support Owners

The next support batch should be owned by:

- `examples/README.md`
- `benchmarks/README.md`

Why this is the right owner pair:

- `examples/README.md` is the direct adoption-side mirror of the landed Day 6
  README/tutorial simplification
- `benchmarks/README.md` is the direct benchmark-side mirror of that same
  ownership split
- together they can close the strongest remaining user-facing drift without
  needing a broad policy or header rewrite

The current policy-side support context remains:

- `docs/maintainer_guide.md`

But that file is support only if the final reconciliation shape truly requires
it.

## Chosen Reconciliation Shape

The strongest additive support batch is a bounded role-alignment batch:

1. keep `example_analysis` as the strongest repeated-run adoption example
2. keep example surfaces explicitly outside regression/oracle/property
   ownership
3. keep `bench_refactor_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse`
   as maintained benchmark-side proof surfaces after adoption
4. keep the canonical report surface bounded to:
   - `make bench-canonical-report`
5. keep tests as the owners of regression/oracle/property guarantees

Why this shape is stronger than the current split state:

- README/tutorial already tell the compact product story
- what is missing is one continuous support-surface story that mirrors that
  landed ownership split without reintroducing longer alternate phrasings

## Non-Widening and Policy Contract

The Day 9 batch is not:

- not another README/tutorial rewrite
- not a public-header cleanup batch
- not an implementation or behavior batch
- not a project-level residual-finalization batch
- not a broad maintainer-policy rewrite

If the support batch forces a policy follow-through edit, keep it bounded to
`docs/maintainer_guide.md` only.

## Exact Day 9 File Fence

Required likely implementation surfaces:

- `examples/README.md`
- `benchmarks/README.md`

Support only if the final reconciliation shape truly needs it:

- `docs/maintainer_guide.md`

Explicit non-touch set:

- `README.md`
- `docs/tutorial.md`
- public headers
- implementation `src/` files
- permanent proof-owner test files
- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- install/package or platform workflow surfaces

## Exit State

Sprint 69 Day 8 closes with one exact support-surface reconciliation
contract:

1. owner pair:
   - `examples/README.md`
   - `benchmarks/README.md`
2. likely support only if needed:
   - `docs/maintainer_guide.md`
3. proof/ownership shape:
   - examples = adoption entry point
   - benchmarks = workflow/performance proof
   - tests = regression/oracle/property guarantees
   - `make bench-canonical-report` = bounded threshold-free reporting surface
4. explicit non-touch set:
   - README/tutorial
   - public headers
   - implementation files
   - project-level residual/finalization surfaces

That gives Day 9 one exact job:

- land one bounded support-surface reconciliation batch on examples and
  benchmarks, with maintainer follow-through only if the landed wording truly
  requires it
