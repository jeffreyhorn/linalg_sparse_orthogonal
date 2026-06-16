# Sprint 71 Day 4: Public-Surface History Audit II & First Landing Boundary

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Refine the Day 3 contradiction ranking into one exact first cleanup fence for
Sprint 71 before any public-surface edits land.

## Re-ranked Cleanup Map

### Required first landing

- `README.md`
- `INSTALL.md`

These are the strongest first-batch surfaces because they carry the densest
remaining mix of:

- product-story framing
- workflow and ownership guidance
- install/package interpretation
- reviewed-platform wording

This is the highest-leverage public cleanup pair now visible in the repo.

### Support only if needed

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

These surfaces still matter, but they do not need to move in the first landing
unless the `README.md` / `INSTALL.md` cleanup forces a follow-through change.

### Strongest deferred reference center

- `include/sparse_cholesky.h`

This remains the strongest header/reference candidate because it still carries
dense narrative around one-shot versus repeated-run ownership, transparent CSC
dispatch, and contract semantics. But it should stay deferred behind the first
public-docs batch so Sprint 71 does not widen too early.

## Main Day 4 Clarification

The strongest first Sprint 71 cleanup fence is:

- front door
- install story

It is not:

- tutorial cleanup first
- public-header cleanup first
- support-surface reconciliation first

That is the useful Day 4 separation. The first landing should improve the two
highest-value user-facing surfaces before it touches adjacent support or
reference docs.

## First-Batch Non-Touch Set

The explicit non-touch set for the first landing is:

- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- implementation `src/` files
- permanent proof-owner test files
- platform workflow files
- benchmark-governance or install/package claim widening

## Exit State

Sprint 71 Day 4 closes with one explicit first landing boundary:

1. required first landing:
   - `README.md`
   - `INSTALL.md`
2. support only if needed:
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
3. strongest deferred reference center:
   - `include/sparse_cholesky.h`

Day 5 can now design the bounded front-door and install cleanup batch from a
real fence rather than from the full public-surface backlog.
