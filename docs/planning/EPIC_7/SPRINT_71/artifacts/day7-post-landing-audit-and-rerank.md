# Sprint 71 Day 7: Post-Landing Audit & Header/Support Rerank

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Audit the live post-Day-6 state and fix the exact next Sprint 71 cleanup
center from what still actually reads poorly rather than from the pre-landing
backlog.

## Main Result

The Day 6 `README.md` / `INSTALL.md` landing closed the strongest public
contradiction. The next highest-value cleanup center is now:

- `include/sparse_cholesky.h`

That is the useful Day 7 rerank. Sprint 71 no longer needs a second primary
front-door/install batch, and it should not force one.

## Post-Landing Rerank

### Required next batch

- `include/sparse_cholesky.h`

This header still carries the densest remaining mix of:

- one-shot versus repeated-run ownership explanation
- transparent CSC-dispatch narrative
- ABI-history spill
- benchmark and contract-reference spill
- cancellation and backend-contract interpretation

### Support only if later cleanup forces it

- `docs/tutorial.md`
- `benchmarks/README.md`
- `examples/README.md`
- `docs/maintainer_guide.md`

These surfaces still matter, but they are not the next strongest contradiction
centers after the Day 6 landing.

### Still explicitly deferred

- other public headers
- implementation files
- permanent proof-owner tests
- platform/install workflow files

## Surface-by-Surface Reading

### `docs/tutorial.md`

This remains the strongest support-only follow-through surface because it still
carries repeated direct-lifecycle and ownership handoff text. But it now reads
more like follow-through than like the next primary cleanup center.

### `benchmarks/README.md`

This still carries dense benchmark-governance explanation, but its ownership
split remains coherent. It is a real support surface, not the next main
center.

### `examples/README.md`

This still carries the example-side adoption handoff, but that handoff remains
coherent after Day 6 and does not justify moving ahead of the Cholesky header.

### `docs/maintainer_guide.md`

This remains the right policy authority. The Day 6 landing did not create a
contradiction that forces a maintainer-guide recentering pass.

## Exit State

Sprint 71 Day 7 closes with one exact next-batch target:

1. `include/sparse_cholesky.h` is now the strongest remaining cleanup center
2. `docs/tutorial.md` is the strongest support-only follow-through surface
3. `benchmarks/README.md`, `examples/README.md`, and
   `docs/maintainer_guide.md` remain support-only
4. no fake second front-door/install batch is needed
