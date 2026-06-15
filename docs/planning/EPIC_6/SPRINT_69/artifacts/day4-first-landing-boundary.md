# Sprint 69 Day 4: First-Landing Boundary

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Convert the Day 3 public-surface ranking into one exact first implementation
fence so Sprint 69 starts from a bounded product-story simplification batch
instead of a generic multi-surface cleanup set.

## Exact First Landing

The exact first landing is now fixed to:

- `README.md`
- `docs/tutorial.md`

Why this is the right first batch:

- together they carry the densest user-facing repeated-run workflow and
  product-story overlap
- they are the strongest pair for simplifying:
  - workflow choice
  - top-level adoption guidance
  - examples vs benchmarks vs tests interpretation
  - canonical benchmark/report wording at the compact public-story layer
- simplifying these two first reduces explanation pressure on the surrounding
  maintained surfaces

So the first Sprint 69 landing should reduce the densest mixed-owner public
narrative pair, not spread evenly across every doc and header surface.

## Support Context, Not First-Batch Center

The first landing may rely on the strongest current support surfaces:

- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

Why they stay support-only unless the design proves otherwise:

- examples and benchmarks already read as relatively local surfaces
- the maintainer guide already owns the policy layer well
- the first Sprint 69 goal is to simplify the top-level public story first
- widening into all three immediately would blur whether Sprint 69 is still
  doing bounded productization or broad cross-surface cleanup

## Strongest Header-Side Support Candidate, Explicitly Deferred

The strongest header-side support candidate remains:

- `include/sparse_cholesky.h`

Why it is not first:

- its strongest remaining pressure is public-path caveat/reference density
- the immediate contradiction is duplicated top-level public-story framing in
  README/tutorial
- touching headers too early risks widening the sprint into reference cleanup
  before the public narrative is simplified

## Explicit Non-Touch Set

The following stay outside the first landing fence:

- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- implementation `src/` files
- permanent proof-owner test files
- install/package or platform workflow surfaces unless the first docs design
  truly proves they must move

## Ranked Order After Day 4

Sprint 69 now has one explicit implementation order:

1. exact first landing:
   - `README.md`
   - `docs/tutorial.md`
2. support only if needed:
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
3. strongest header-side support candidate, explicitly deferred:
   - `include/sparse_cholesky.h`
4. later/deferred support:
   - `include/sparse_analysis.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - project-level residual/finalization surfaces

## Exit State

Sprint 69 Day 4 closes with one exact first landing boundary:

- `README.md` and `docs/tutorial.md` first
- examples/benchmark/maintainer surfaces support only if needed
- `include/sparse_cholesky.h` explicitly deferred behind the first docs batch

That gives Day 5 one exact job:

- define the bounded productization contract centered on README/tutorial and
  only pull support surfaces in where the simplified public story truly
  requires it
