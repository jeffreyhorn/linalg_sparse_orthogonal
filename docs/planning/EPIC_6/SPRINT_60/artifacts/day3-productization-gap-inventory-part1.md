# Sprint 60 Day 3: Productization Gap Inventory, Part I

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Reduce the broad Epic 6 review into a live-repo productization inventory
focused on the strongest user-facing gap classes:

- public API ergonomics
- examples and onboarding flows
- configuration discoverability
- benchmark/story clarity
- packaging/install shape

## Highest-Value Findings

### 1. Split direct-solver usability model

The largest remaining user-facing gap is that the direct-solver story is still
split between:

- first-class one-shot APIs
- an explicit repeated-run analysis/factors lifecycle

That is already much better than a hidden repeated-run story, but it still
leaves callers juggling:

- mutation-oriented one-shot behavior
- explicit matrix-copy or matrix-rebuild discipline
- separate conceptual models for “simple” versus “advanced” direct solves

This is the strongest user-facing Epic 6 product gap.

### 2. Env-var-driven advanced configuration

The strongest architecture/product-control gap remains the current advanced
configuration surface:

- `SPARSE_ND_*`
- `SPARSE_FM_*`
- `SPARSE_SUPERNODAL_POSTORDER`
- additional advanced or profiling env vars around reorder/graph/SVD behavior

Those controls are real and valuable, but they still live too much in:

- internal `getenv(...)` logic
- implementation comments
- long-form README detail

That makes the advanced-tuning story too process-global and too weakly typed
for a state-of-the-art product surface.

### 3. Dense onboarding/reference layering

The repo now has a coherent workflow-first README/tutorial/examples story, but
the adoption path is still denser than it should be:

- the README still mixes product overview, workflow map, algorithm detail,
  benchmark notes, quality contract, and repo layout
- the tutorial is practical but still broad
- examples stay intentionally one-shot-first, which is coherent, but leaves the
  advanced repeated-run flows less discoverable than they should be

This is a real productization gap, though lower-priority than direct usability
and typed configuration.

### 4. Rich but fragmented benchmark/performance story

The benchmark surface is strong, but still not yet a compact performance
governance system:

- workflow-specific proof drivers exist
- benchmark-local docs are much better than before
- but the benchmark story still does not cleanly separate:
  - regression-sensitive baselines
  - exploratory characterization
  - stable product-performance claims

This is a state-of-the-art maturity gap more than a basic functionality gap.

### 5. Credible but static-first packaging/distribution story

The repo already supports:

- `pkg-config`
- `find_package(Sparse)`
- install/export flows

But the current product/distribution shape still reads as:

- developer-install friendly
- static-library first
- bounded ABI/release story

That is real Epic 6 work, but not the first product-facing seam to fix.

## Ranked Day 3 Gap Classes

1. **Usability friction**
   - split direct-solver workflow model
2. **Configuration opacity**
   - env-var-first advanced control surfaces
3. **Documentation overload / ambiguity**
   - dense README/tutorial/example layering
4. **Advanced-user control gaps**
   - typed/public option surface lag behind real internal tuning power
5. **Packaging/platform asymmetry**
   - credible install surface, but not yet a full product-distribution story

## Candidate Product Goals Suggested by Day 3

- make the direct-solver story easier to teach and use safely
- replace the highest-value process-global tuning knobs with typed controls
- make the benchmark/performance story easier to consume as a product claim
- improve the install/packaging story enough to feel less developer-only
- reduce adoption-surface density without losing technical truthfulness

## Non-Goals Not Justified by Day 3

- large new algorithm-family expansion
- distributed/HPC scope
- immediate vendor-backend parity
- pretending the current strong engineering state already equals state-of-the-
  art product maturity

## Day 3 Exit State

Sprint 60 now has a live-repo productization inventory that is concrete enough
to rank further on Day 4 and precise enough to drive a real Epic 6 target
definition on Day 5.
