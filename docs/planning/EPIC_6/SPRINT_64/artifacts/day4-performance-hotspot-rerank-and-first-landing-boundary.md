# Sprint 64 Day 4: Performance Hotspot Rerank and First Landing Boundary

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Take the Day 3 hotspot map, compare it against the explicit Epic 6
state-of-the-art target, and reduce Sprint 64 to one exact first landing
boundary instead of a broad performance shortlist.

## Reranked Sprint 64 Surface

### 1. The Cholesky CSC supernodal lane remains the strongest must-touch Phase 1 seam

Against the Epic 6 target definition, the Cholesky CSC supernodal path is
still the best first landing because it combines:

- bounded touched surface
- real runtime relevance
- existing family-local proof
- explicit fallback neighbors
- low risk of widening the public product story

Why it stays first:

- it supports a real backend/performance architecture claim without pretending
  the whole repository is already backend-pluggable
- it matches the Epic 6 requirement for a bounded backend seam on selected hot
  paths
- it can prove value through the existing benchmark and CSC proof surfaces

### 2. LDL^T supernodal follow-through stays important but second

The Day 3 ranking still holds after the target-definition rerank:

- LDL^T supernodal work is valuable
- but it remains more correctness-sensitive and pivot-state-heavy
- it belongs in the next backend follow-through lane, not the
  abstraction-defining first landing

Why it stays second:

- it should benefit from the first bounded kernel abstraction rather than
  define it
- forcing Bunch-Kaufman-specific complexity into the first landing would widen
  Sprint 64 too early

### 3. `src/sparse_dense.c` belongs inside the first landing only as an internal seam

The rerank tightens the role of the generic dense helper layer:

- it is now part of the likely first landing boundary
- but only as an internal dependency seam serving the selected Cholesky CSC
  path
- it should not become a repo-wide universal dense backend rewrite in Sprint 64

Why it ranks this way:

- it matters for the abstraction seam
- but QR/SVD-wide dense unification would immediately over-broaden the sprint

### 4. Build/options work is required, but only in support of the selected kernel path

The rerank confirms that build and option wiring is real work, but not the
thing that should define the first landing:

- it should follow the selected kernel abstraction
- it must preserve the default self-contained build
- it should avoid public API widening unless the first landing truly needs it

### 5. Benchmark-governance and broad packaging work remain out of the first landing

The target-definition rerank sharpens the out-of-scope line:

- benchmark proof refresh is in scope
- broad benchmark-governance redesign is not
- platform/packaging maturity remains an Epic 6 band, but not part of the
  first Sprint 64 landing

## First Selected Sprint 64 Landing Surface

The exact first selected Sprint 64 landing surface is now:

- required first kernel lane:
  - Cholesky CSC supernodal dense-kernel path
- required nearby internal seam:
  - bounded dense-helper abstraction support
- required proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
- likely supporting truth surfaces later:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - build wiring only if the selected abstraction actually needs it

## Explicit Deferred / Later Queue

The Day 4 rerank fixes the later queue explicitly:

- second backend target:
  - LDL^T supernodal follow-through
- later dense-kernel/backend candidates:
  - QR
  - SVD
- later support bands:
  - broader benchmark-governance work
  - packaging/platform maturity work
  - broader threading-policy generalization

## Exit State

Sprint 64 now has one exact first landing boundary instead of a generic
backend shortlist:

- the Cholesky CSC supernodal dense-kernel lane is fixed as the first landing
- LDL^T supernodal follow-through remains the strongest second target
- `src/sparse_dense.c` is part of the first landing only as a bounded internal
  seam
- build/options work is confirmed as support work, not the first design center
- packaging/platform and broad benchmark-governance work remain explicitly out
  of the first landing
