# Sprint 69 Day 5: Docs/Examples Productization Design

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Turn the Day 4 first-landing boundary into one explicit productization
contract so the first Sprint 69 implementation batch stays bounded to the
highest-value README/tutorial simplification seam.

## First-Landing Productization Contract

The first landing remains fixed to:

- `README.md`
- `docs/tutorial.md`

But Day 5 now makes the intended durable ownership more explicit.

`README.md` should converge toward:

- one compact top-level product narrative
- one compact workflow-choice front door
- one compact ownership summary for examples, benchmarks, and tests
- one compact platform/install truth summary

`docs/tutorial.md` should converge toward:

- user-facing teaching flow
- step-by-step public-API usage guidance
- workflow handoff from one-shot entry points to repeated-run or handle paths
- explicit links out when policy, benchmark, or regression-owner detail matters

So the first landing is not a broad docs rewrite. It is a bounded ownership
simplification across the two densest public-facing narrative surfaces.

## Keep One Compact Product Front Door, Not Multiple Parallel Product Stories

The strongest current public-surface problem is duplicated framing across
README, tutorial, and support docs.

The safe first-batch contract is:

- keep `README.md` as the compact front door
- keep `docs/tutorial.md` as the teaching flow
- do not let either one fully restate the other’s role

Design consequence:

- remove overlap where README and tutorial both try to be the full workflow
  owner
- preserve links and handoffs instead of parallel long-form explanations

## Keep Examples and Benchmarks as Support Surfaces, Not First-Batch Centers

The first batch may need support follow-through in:

- `examples/README.md`
- `benchmarks/README.md`

Their durable ownership remains:

- examples:
  - adoption entry points
  - example-local behavior
  - no regression/oracle/property ownership expansion
- benchmarks:
  - workflow/performance proof and schema explanation
  - no test-owned guarantee expansion

Design consequence:

- support wording may move only if the simplified README/tutorial story
  requires it
- examples and benchmarks should not become alternate top-level product-story
  homes

## Keep the Maintainer Guide as Policy Authority, Not Another First-Batch Explainer

`docs/maintainer_guide.md` already owns:

- documentation ownership interpretation
- benchmark-governance interpretation
- quality/platform/packaging policy interpretation

The first landing should not rewrite that policy layer broadly.

The safe first-batch contract is:

- keep the maintainer guide as support-only unless the simplified README /
  tutorial story leaves an obvious policy contradiction behind
- avoid turning Day 6 into a repo-wide policy rewrite instead of a bounded
  productization batch

## Exact Day 6-7 Touched-File Fence

Required first-batch implementation surfaces:

- `README.md`
- `docs/tutorial.md`

Support only if the landed simplification truly needs them:

- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

Explicitly not in the first batch:

- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- implementation `src/` files
- permanent proof-owner test files
- install/package or platform workflow surfaces unless the first docs design
  truly proves they must move

## Explicit Non-Widening Rules

The first productization landing should not widen into:

- public header cleanup
- implementation or behavior changes
- benchmark-governance redesign
- platform/install contract redesign
- project-level residual-finalization edits
- broad maintainer-policy churn

That non-widening fence matters because Sprint 69 still has real later lanes
after the first productization landing:

- support-surface follow-through
- cross-surface compatibility sweep
- final validation
- Epic 6 summary and residual finalization

## Exit State

Sprint 69 Day 5 closes with one exact first implementation contract:

1. required first batch:
   - `README.md`
   - `docs/tutorial.md`
2. support only if needed:
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
3. durable ownership target:
   - README = compact product-story front door
   - tutorial = user-facing teaching flow
   - examples = adoption/local entry points
   - benchmarks = workflow/performance proof and schema explanation
   - maintainer guide = policy authority
4. explicit non-touch set:
   - public headers
   - implementation files
   - proof-owner tests
   - project-level residual surfaces

That gives Day 6 one exact job:

- land one bounded README/tutorial productization batch without widening into
  broad cross-surface or header cleanup
