# Sprint 71 Day 3: Public-Surface History Audit I

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Reduce Sprint 71's broad public/reference cleanup concern to a ranked live
contradiction map across the current user-facing docs before the sprint fixes
its first landing boundary.

## Input Surfaces

Primary public and support surfaces re-read on Day 3:

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

Measured raw Day 1 `wc -l` newline counts for the strongest public/support
surfaces:

- `README.md` = `1034`
- `INSTALL.md` = `237`
- `docs/tutorial.md` = `479`
- `examples/README.md` = `166`
- `benchmarks/README.md` = `370`
- `docs/maintainer_guide.md` = `578`

## Ranking

### 1. `README.md`

`README.md` is the strongest first contradiction center because it still
carries the densest mix of:

- top-level product story
- workflow-choice guidance
- examples versus benchmarks versus tests ownership
- threshold-free benchmark-report interpretation
- platform-confidence summary
- install/package summary
- capability and limitation framing

This is not a missing-content problem. It is a density problem: too much of
the public product, workflow, and proof story still lands in one front-door
surface.

### 2. `INSTALL.md`

`INSTALL.md` is the strongest second target because it still layers together:

- operator install steps
- static-first package-shape explanation
- reviewed versus supplemental platform-lane interpretation
- local install/package proof ownership
- downstream consumer guidance

That makes it valuable, but also makes it read partly like a runbook and
partly like a policy explainer.

### 3. `docs/tutorial.md`

The tutorial is the strongest third target. Its remaining burden is repeated
teaching-flow framing rather than front-door contradiction:

- repeated-run direct lifecycle explanation
- handoff to `example_analysis`
- handoff to retained benchmark surfaces
- reminders that tests own regression, oracle, and property guarantees

This should be cleaned later, but behind the front-door and install surfaces.

### 4. `benchmarks/README.md`

This is the strongest support-surface contradiction center because it still
holds a dense mix of:

- benchmark-governance interpretation
- canonical-report framing
- benchmark versus test ownership explanation
- retained benchmark-lane history

It matters, but should remain behind the first public landing.

### 5. `examples/README.md`

`examples/README.md` is lower-risk support context. It already reads more
narrowly as:

- workflow/adoption guidance
- explicit non-ownership of regression/oracle/property guarantees
- benchmark handoff after adoption

It is still a real support surface, but not a first cleanup center.

### 6. `docs/maintainer_guide.md`

The maintainer guide remains support-first. Its policy density is largely the
right kind of density:

- deeper rationale
- deferred queue
- platform/package/proof interpretation

Sprint 71 should simplify public surfaces by recentering them around this
authority, not by turning it into the first cleanup target.

## Main Day 3 Clarification

The broad Sprint 71 cleanup problem is now concretely ranked:

1. `README.md`
2. `INSTALL.md`
3. `docs/tutorial.md`
4. `benchmarks/README.md`
5. `examples/README.md`
6. `docs/maintainer_guide.md` as support-first policy authority

That ranking is the most useful Day 3 result:

- the strongest first cleanup is front-door and install density
- the strongest support contradiction is benchmark-governance explanation
- the maintainer guide should stay policy-first, not become the first rewrite
  center

## Exit State

Sprint 71 Day 3 closes with one explicit public-surface contradiction map:

- `README.md` is the strongest first cleanup center
- `INSTALL.md` is the strongest second cleanup center
- `docs/tutorial.md` is a real later teaching-flow target
- `benchmarks/README.md` is the strongest support-surface contradiction center
- `examples/README.md` and `docs/maintainer_guide.md` stay support-first

Day 4 can now freeze the first Sprint 71 cleanup boundary from a real current
ranking instead of from a generic public-docs backlog.
