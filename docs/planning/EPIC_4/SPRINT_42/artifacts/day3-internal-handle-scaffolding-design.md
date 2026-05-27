# Sprint 42 Day 3 Artifact: Internal Handle Scaffolding Design

## Purpose

Define the first concrete internal handle scaffolding for Sprint 42 so the
implementation-heavy lifecycle refactor work can land through explicit internal
ownership seams without forcing premature public API changes.

## Design Inputs

This design is derived from:

- `docs/planning/EPIC_4/SPRINT_42/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day2-lifecycle-seam-refresh-inventory.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day9-handle-model-design-1.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day10-handle-model-design-2-and-migration-strategy.md`

## Design Goals

The first Sprint 42 handle layer should:

1. reduce hidden `SparseMatrix` lifecycle overloading internally
2. preserve all current public entry-point behavior
3. target the highest-value ownership seams first:
   - LU
   - Cholesky
   - `sparse_factors_t`
4. keep the first object set intentionally small
5. create a clean staging point for later public explicit-handle enrichment

## Proposed First-Phase Internal Handle Families

Sprint 42 only needs three first-phase handle/payload families:

1. LU numeric payload handle
2. Cholesky numeric payload handle
3. bridge payload normalization around `sparse_factors_t`

### 1. LU numeric payload handle

#### Purpose

Provide an internal owner for LU factor-state data so the matrix stops being
the only effective numeric factor container.

#### Intended internal responsibilities

- numeric LU payload representation
- factor-local solve-readiness state
- factor-local permutation / telemetry ownership where separation is practical
- cleanup and cancellation-sensitive partial-state ownership

#### Public compatibility rule

The existing LU one-shot APIs remain the public surface in Sprint 42. The new
payload seam exists behind them.

### 2. Cholesky numeric payload handle

#### Purpose

Provide an internal owner for Cholesky factor-state data so the matrix stops
being the sole owner of both SPD coefficient semantics and lower-factor solve
state.

#### Intended internal responsibilities

- numeric Cholesky payload representation
- factor-local solve-readiness state
- factor-local permutation / telemetry ownership where separation is practical
- ownership of cancellation-sensitive post-mutation state

#### Public compatibility rule

The existing Cholesky one-shot APIs remain the public surface in Sprint 42.
The new payload seam exists behind them.

### 3. `sparse_factors_t` bridge payload normalization

#### Purpose

Keep `sparse_factors_t` as the public analyze-once factor wrapper while
reducing its dependence on matrix-centric internal payload ownership.

#### Intended internal responsibilities

- preserve `sparse_factors_t` as the public bridge object
- normalize factor payload ownership behind the bridge
- reduce the direct “wrapper around factored matrix” coupling where Sprint 42
  can do so safely

#### Public compatibility rule

`sparse_factors_t` remains the public factor handle for the analyze-once
workflow. Sprint 42 changes what it owns internally, not how callers obtain or
free it.

## Relationship To Current `SparseMatrix`-Centric Code

### Stable role for `SparseMatrix` during Sprint 42

`SparseMatrix` should continue to own:

- coefficient/value storage
- structural editing behavior
- matrix query and arithmetic behavior
- compatibility-facing wrapper behavior where public entry points still accept
  matrix objects directly

### State that should start moving behind internal handle seams

Sprint 42 should begin moving the following concerns behind internal payload
owners:

- LU numeric factor payload ownership
- Cholesky numeric factor payload ownership
- factor-local solve-readiness state
- factor-local permutation/telemetry ownership where internal seams permit it
- bridge-owned numeric payload inside `sparse_factors_t`

### State that remains intentionally public/visible for now

Sprint 42 is not yet trying to eliminate or redesign:

- the current public matrix entry points
- lifecycle-sensitive installed-header contracts
- the current public solve and free entry points
- broader field-level public escape hatches in one step

## Proposed File / Ownership Layout Direction

The first Sprint 42 landing should follow the same internal-first pattern used
in Sprint 41:

- private/internal handle or payload definitions stay in `src/`
- public API signatures remain in existing installed headers
- compatibility wrappers stay in the current implementation files for the
  factorization families they already serve

### Ownership rule

The internal handle/payload layer should become the true owner of newly
separated numeric factor state. The matrix object should remain the caller's
coefficient/value object and compatibility-facing wrapper surface, not the only
long-term owner of factor semantics.

## Keep-In-Matrix vs Move-Behind-Handle Split

### Keep on `SparseMatrix` in Sprint 42

- coefficient/value semantics
- structural mutation semantics
- matrix query/arithmetic behavior
- compatibility-facing public wrapper role
- any purely value-derived utility/cache behavior not directly tied to factor
  ownership redesign

### Move or begin moving behind internal handles / bridge payloads

- LU numeric factor payload ownership
- Cholesky numeric factor payload ownership
- factor-local solve-readiness state
- factor-local cancellation cleanup ownership
- bridge-owned numeric payload inside `sparse_factors_t`

## Compatibility Contract

Sprint 42's first handle layer is explicitly compatibility-preserving:

- current LU one-shot entry points stay public
- current Cholesky one-shot entry points stay public
- current analyze-once workflow stays public
- current docs/header examples do not need broad public-handle rewrite yet

The handle layer therefore changes architecture first, not user-facing API
shape first.

## Implementation Consequences For Days 5-10

### Day 5 target

Land the first LU/Cholesky internal payload seam.

### Day 6 target

Land the shared matrix-state guard helper layer separately from the handle seam.

### Day 7-9 target

Normalize factor-entry paths and begin bounded `sparse_factors_t` bridge
cleanup using the new ownership model.

### Day 10 target

Use the new ownership boundary to reason more clearly about cancellation and
mutation contracts in the touched families.

## Day 3 Conclusions

1. Sprint 42 only needs a small first-phase internal handle set:
   - LU payload
   - Cholesky payload
   - `sparse_factors_t` bridge normalization
2. The first handle layer is about internal ownership change, not public API
   change.
3. `SparseMatrix` remains the coefficient/value object and compatibility-facing
   wrapper surface during Sprint 42.
4. `sparse_factors_t` should be preserved as the public bridge while its
   internals get less matrix-centric.
5. Day 5 now has a concrete ownership-seam target that matches the Day 2 live
   seam inventory and the Sprint 40 migration strategy.
