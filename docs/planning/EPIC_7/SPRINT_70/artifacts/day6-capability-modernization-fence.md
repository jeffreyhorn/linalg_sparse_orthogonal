# Sprint 70 Day 6: Capability Ceiling Audit II & Modernization Fence

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Convert the Day 5 capability ranking into one exact first Epic 7 capability
modernization fence so later work starts from a bounded first lane instead of
from a broad "64-bit + scalar genericity + more algorithms" promise.

## Refined Ranking

Re-ranking the Day 5 ceilings against:

- user value
- ecosystem impact
- implementation risk
- proof burden

leaves the following order:

1. first bounded modernization candidate:
   - index-width path
2. medium-term capability target:
   - scalar-surface broadening
3. later capability target:
   - unsymmetric sparse eigensolver expansion

## Why Index Width Stays First

The index-width ceiling remains the strongest first capability lane because:

- it is the broadest current product ceiling
- it affects the entire sparse product surface rather than one subsystem
- it is easier to isolate into one typedef/overflow/build contract than
  scalar-type generalization
- it creates a real modernization path without requiring a full product-line
  rewrite

So the first Epic 7 capability question is not:

- "can the repo become type-generic all at once?"

It is:

- "can the repo make the 32-bit ceiling non-permanent through one real bounded
  width-modernization seam?"

## Why Scalar Breadth Is Second, Not First

Real-only double-precision numerics remain the strongest second capability
ceiling because they exclude:

- complex-valued sparse workloads
- precision-product variants such as single precision or mixed precision
- broader product-line flexibility expected from more state-of-the-art sparse
  libraries

But scalar breadth is not the right first landing because it cuts across:

- public headers
- factor/result structs
- callbacks
- examples/tests/docs
- packaging and downstream expectations

That makes it the strongest second lane, not the first one.

## Why Unsymmetric Eigensolver Expansion Is Explicitly Later

The unsymmetric sparse eigensolver gap is real, but it is not the right first
capability landing because:

- it is narrower than width and scalar breadth
- the current symmetric eigensolver story is already comparatively strong
- the proof and documentation burden is high relative to the first Epic 7
  payoff

So Day 6 makes that queue position explicit:

- important later target
- not first-lane capability work

## Exact First Modernization Fence

The exact first Epic 7 capability modernization fence should center on the
index-width contract:

- required first modernization center:
  - `include/sparse_types.h`
  - `README.md`
  - `docs/maintainer_guide.md`
- likely support only if needed:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `INSTALL.md`
- proof and safety support only if later implementation demands it:
  - width-sensitive overflow and bounds tests
  - install/package sanity where the public typedef or build reading changes

Why this is the right first fence:

- `idx_t` is the public width center
- README and maintainer policy already own the documented width limit
- the first lane can define one real modernization path without pretending to
  widen every numeric family at once

## Explicit Non-Goals

The first Epic 7 capability lane explicitly does not include:

- full type-generic conversion in one sprint
- fake complex-readiness claims without end-to-end proof
- broad unsymmetric eigensolver expansion in the first batch
- package/ABI claims wider than the actual landed width seam
- one combined batch that tries to solve width, scalar breadth, and
  algorithm-family expansion simultaneously

## Exit State

Sprint 70 Day 6 closes with one exact capability modernization order:

1. first:
   - index-width modernization path
2. second:
   - scalar-surface preparation and broadening
3. later:
   - unsymmetric sparse eigensolver expansion

That gives later Sprint 70 and Sprint 74 planning one exact job:

- land the first real width-modernization seam first, then widen only where
  that bounded path proves broader scalar or algorithm-family work is
  justified
