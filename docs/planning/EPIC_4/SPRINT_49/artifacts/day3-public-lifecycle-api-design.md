# Sprint 49 Day 3 Artifact: Public Lifecycle/API Design

## Purpose

Define the bounded public lifecycle/workspace model Sprint 49 can now expose
after the lifecycle scaffolding, internal reusable-workspace seams, and
documentation-policy homes are already in place.

## Design Goal

Sprint 49 should expose a final public lifecycle layer for iterative and
eigensolver repeated-run work without:

- mirroring internal helper APIs directly
- removing the existing one-shot public solver/eigensolver entry points
- redesigning the broader solver/eigensolver option and result surfaces

The exposed model should feel consistent with the public analyze/factor/reuse
precedent already present in `include/sparse_analysis.h`.

## Core Public Design Decision

### Public lifecycle/workspace exposure should be lifecycle-centric, not storage-layout-centric

The internal groundwork is storage-layout-centric:

- reusable contiguous workspace owners
- typed internal views
- internal benchmark entry points

The public API should not expose those details directly.

Instead, the public contract should be lifecycle-centric:

1. initialize a public handle
2. prepare it for a stable-dimension repeated-run use case
3. run one or more solves / eigensolver calls through it
4. reuse or reset it across repeated calls
5. free it explicitly

This keeps the public API aligned with how callers think about repeated runs,
while still letting the implementation route through the internal workspace
owners and typed view helpers already landed in Sprints 45 and 46.

## Intended Public Surface Shape

### 1. Keep the existing one-shot public calls

The existing public entries remain first-class:

- iterative one-shot entries
  - `sparse_solve_cg(...)`
  - `sparse_solve_gmres(...)`
  - matrix-free and block convenience variants
- eigensolver one-shot entry
  - `sparse_eigs_sym(...)`

These stay supported and documented.

Sprint 49 should make them more explicit as:

- compatibility-oriented
- convenience-oriented
- layered over the new lifecycle-enabled path where appropriate

but not deprecated or treated as second-class leftovers.

### 2. Add a bounded public reusable-handle layer

The new public layer should expose:

- a public iterative repeated-run handle family
- a public eigensolver repeated-run handle family
- prepare / run / free style entry points

But it should avoid exposing:

- raw internal workspace owners
- typed internal view structs
- internal benchmark entry points
- algorithm-private scratch layout

### 3. Preserve the existing option/result model

Sprint 49 should not redesign the current caller-facing config/result approach.

The existing surfaces should remain the main caller contract:

- option structs
- result structs
- caller-owned eigenvalue/eigenvector buffers where already required
- designated-initializer usage patterns

The new lifecycle layer should compose around those surfaces instead of
replacing them with a large new option/handle ownership model.

## Public Contract Rules

### Initialization / prepare

The public lifecycle contract should follow the same broad pattern already
taught by `sparse_analysis.h`:

- zero-init or explicit init is valid
- prepare binds the stable-dimension repeated-run context
- invalid dimensions or incompatible inputs fail before solver work starts

### Solve / run

The run step should:

- accept the existing options and result surfaces
- use the prepared handle as the repeated-run ownership seam
- preserve the same numerical semantics as the equivalent one-shot entry

### Reuse / reset

Reuse semantics should be explicit:

- reuse preserves allocation capacity and prepared ownership
- reuse does not preserve old Krylov / Ritz / search state
- each call is still a fresh numerical run

Reset semantics should stay narrow:

- reset should clear numerical-run leftovers
- reset should not promise stable public access to internal packed storage

### Teardown / free

The handle must have an explicit free path, safe on a zeroed or empty state,
matching the public lifecycle expectations already established by:

- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

## Compatibility Rules

### One-shot wrappers remain first-class

Sprint 49 should preserve the old one-shot usage style as a valid path for:

- simple callers
- single solve/eigensolver runs
- examples that are intentionally teaching the basic API

### New explicit lifecycle path is preferable when repeated runs are real

The new explicit lifecycle/workspace path should be the preferred route when:

- the same dimension/problem shape repeats
- allocation churn matters
- benchmarked repeated-run scenarios justify handle reuse
- the caller benefits from explicit ownership over setup/reuse/free boundaries

### Public precedent alignment

The public wording should align with the already-exposed reusable-lifecycle
story in `sparse_analysis.h`:

- prepare once
- run repeatedly
- free explicitly

That consistency is more important than perfectly mirroring internal helper
naming.

## Option / Result and Style Rules

### Option structs

Keep:

- current option structs
- current defaults model
- current designated-initializer expectations

Do not:

- replace them with large handle-only config objects
- create parallel “v2” option structs unless implementation is blocked without
  that change

### Result structs

Keep:

- current result structs
- current caller-owned eigensolver output buffer model
- current progress/cancellation callback positioning

Do not:

- redesign result ownership just because handles become public

## Explicit Non-Goals

Sprint 49 should not claim to do any of the following:

- broad solver API redesign
- removal of the one-shot public entry points
- public exposure of every internal helper or workspace view
- public API guarantees around internal storage layout
- broad example/README/tutorial rewrite before the public landing is stable
- post-Epic-4 feature expansion disguised as lifecycle cleanup

## Day 3 Design Conclusions

### 1. The public API target is now concrete

Sprint 49 should expose:

- bounded public repeated-run lifecycle handles for iterative and eigensolver
  work
- prepare / run / free style entry points
- compatibility-preserving one-shot wrappers that remain supported

### 2. The public API should follow lifecycle language already proven in the library

The best public anchor remains:

- `sparse_analysis.h`

That gives Sprint 49 a consistent public lifecycle story instead of a new
special-case vocabulary.

### 3. The design is bounded enough for implementation

Day 5/6 implementation should now:

- target public headers and wrapper routing
- avoid publicizing raw internal workspace machinery
- preserve current option/result surfaces
- leave examples, README, and benchmark reconciliation for later sweep work
