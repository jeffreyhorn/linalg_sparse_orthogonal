# Sprint 40 Day 10: Future Handle-Model Design II & Migration Strategy

## Objective

Complete the Sprint 40 handle-model design by defining the staged migration
contract from the current hidden-state architecture to the future explicit
handle model. Day 9 defined the target object families and responsibility
split; Day 10 defines how that design can land safely without destabilizing the
validated Epic 3 quality baseline.

## Design Inputs

This migration strategy is derived from:

- `day5-lifecycle-inventory-lu-cholesky-ldlt.md`
- `day6-lifecycle-inventory-qr-svd-analysis-iterative-eigs.md`
- `day7-state-model-taxonomy.md`
- `day8-lifecycle-contract-map.md`
- `day9-handle-model-design-1.md`
- the Epic 4 remediation plan

## Migration Goals

The migration path should:

1. preserve current caller-visible semantics while reducing hidden lifecycle
   state internally
2. start with internal ownership changes before public API churn
3. add explicit handles first where the architecture benefit is highest
4. avoid reopening the validated quality, dead-code, or cross-platform
   contracts inherited from Epic 3
5. keep compatibility scaffolding narrow, temporary, and behavior-oriented

## Migration Phases

Epic 4 should land in four broad phases.

### Phase 1: Internal payload separation

#### Purpose

Introduce explicit internal factor payloads and context boundaries without
changing the public API shape.

#### Main targets

- LU internal factor payload
- Cholesky internal factor payload
- internal bridge cleanup around `sparse_factors_t`
- workspace-boundary definition for iterative/eigensolver internals

#### Expected public effect

None by contract. Public one-shot entry points should continue to behave the
same while the internal owners of factor state change.

### Phase 2: Bridge-object normalization

#### Purpose

Make the mixed-boundary objects intentional rather than accidental.

#### Main targets

- evolve `sparse_factors_t` away from “wrapper around factored matrix”
- normalize analysis-to-factor boundaries
- normalize factor/preconditioner composition seams
- make cancellation cleanup belong to factor/context objects rather than to
  callers reasoning about matrix mutation

#### Expected public effect

Possibly clearer documentation and internal helper accessors, but still not a
required broad public migration.

### Phase 3: Public explicit-handle enrichment

#### Purpose

Expose clearer public explicit-handle entry points where the internal model is
already proven and materially simpler for callers.

#### Main targets

- LU explicit factor-handle family
- Cholesky explicit factor-handle family
- possible clearer workspace/context entry points for repeated iterative/eigs
  runs

#### Expected public effect

New opt-in APIs or richer public entry points may appear here, but the
one-shot convenience routines should remain as wrappers during the transition.

### Phase 4: Compatibility narrowing

#### Purpose

Reduce caller dependence on field-level lifecycle details once explicit-handle
paths are stable.

#### Main targets

- de-emphasize matrix-centric factor-state reasoning in docs
- narrow reliance on direct `SparseMatrix` lifecycle escape hatches
- reevaluate which old lifecycle details remain necessary for compatibility

#### Expected public effect

This is the earliest point where deprecation or stronger migration guidance
would be appropriate, and only if earlier phases have already proven the new
model.

## Earliest Safe Insertion Points

The safest first landing points are the places where internal ownership can
change behind stable public behavior.

### 1. LU and Cholesky internal builders

These are the strongest initial insertion points because they currently carry
the largest matrix-as-factor-handle burden.

Safe initial landing shape:

- keep public one-shot APIs stable
- construct internal factor payloads immediately
- treat matrix mutation as wrapper-compatible behavior rather than the primary
  architectural owner of factor state

### 2. `sparse_factors_t` payload normalization

This is the strongest bridge insertion point because the analyze-once surface
already matches the target architecture conceptually.

Safe initial landing shape:

- preserve `sparse_analysis_t` and `sparse_factors_t` public roles
- change what `sparse_factors_t` owns internally before changing how callers
  think about it

### 3. Iterative/eigensolver internal workspaces

These are safe because they are already operator consumers rather than direct
factor-state carriers.

Safe initial landing shape:

- add internal reusable work buffers first
- keep current one-shot entry points as wrappers
- delay public workspace API decisions until repeated-run value is proven

### 4. Preconditioner/factor composition seams

ILU/ILUT/IC already sit closer to explicit handle families than LU/Cholesky
do, so they are useful composition boundaries but not the first compatibility
problem to solve.

Safe initial landing shape:

- normalize internal ownership and cleanup boundaries
- preserve current public builder/cleanup semantics

## Compatibility Layers Likely Required

The Day 10 conclusion is that compatibility help will be needed, but the
required set is bounded.

### Wrapper preservation

Keep the following families as stable one-shot wrappers during migration:

- LU solve/factor paths
- Cholesky solve/factor paths
- iterative solver one-shot entry points
- eigensolver one-shot entry points

### Bridge adapters

Temporary adapters will likely be needed for:

- matrix-to-factor transitions inside LU/Cholesky internals
- analysis-to-factor payload transitions inside `sparse_factors_t`
- preconditioner/factor composition where old code still expects matrix-shaped
  ownership

### Documentation shims

Docs should temporarily explain:

- stable semantic requirements
- when original matrix state still matters
- when explicit cleanup remains required

They should not teach transitional internal field ownership unless absolutely
necessary.

### Deprecation layer

Day 10 does not justify an immediate deprecation campaign. If needed, that
should come only after:

- explicit handle internals are proven
- wrappers remain stable
- the real migration burden is measured rather than assumed

## High-Risk Migration Edges

The highest-risk edges are now explicit and ranked.

### 1. Factorization entry points

Highest-risk families:

- LU
- Cholesky

Why they are risky:

- they overload `SparseMatrix` as coefficient owner, mutated factor carrier,
  solve-state indicator, and cleanup target
- cancellation can leave non-original state early

Migration implication:

- these should get the strongest internal-first treatment
- public wrapper preservation is load-bearing here

### 2. Analysis/reorder surfaces

Why they are risky:

- they are already close to the future architecture, so careless refactors
  could damage one of the cleanest existing seams
- `sparse_factors_t` currently bridges good public shape with matrix-centric
  internal payload

Migration implication:

- preserve conceptual surface shape
- change payload ownership behind it first

### 3. Cancellation semantics

Why they are risky:

- cancellation is not just an error path; it is part of the lifecycle
  contract
- the current audit shows that some builders can mutate state before the first
  callback observation

Migration implication:

- later refactors must define cancellation cleanup at the factor/context owner
  level
- “preserves input or not” stays a semantic contract

### 4. Copy-before-reuse pitfalls

Why they are risky:

- some APIs are difficult today not because they lack handles, but because
  callers must infer when “original matrix required” or “copy before reuse”
  still applies

Migration implication:

- later work should reduce inference burden either by:
  - explicit entry-point naming
  - protected internal working views
  - or clearer split between coefficient objects and factor objects

### 5. Field-level lifecycle escape hatches

Why they are risky:

- direct or implied dependence on `factored`, permutation arrays, and
  `factor_norm` is exactly the internal-state leakage Epic 4 is trying to
  narrow

Migration implication:

- compatibility should preserve behavior first
- but later public guidance should shift callers away from field-level state
  reasoning where possible

## What Should Not Change Early

The safest migration strategy is defined partly by what it should avoid doing
in the early phases.

Do not start by:

- breaking one-shot public APIs
- rewriting iterative/eigensolver APIs into factor-style abstractions
- collapsing cache mutation and factor lifecycle mutation into one problem
- changing the public analysis/factor pipeline shape before its internals are
  normalized
- introducing a large generic deprecation program before the real migration
  burden is measured

## Day 10 Design Decisions

1. The first implementation-heavy lifecycle work should be internal-first.
2. LU and Cholesky are the earliest and highest-value handle landing points.
3. `sparse_factors_t` should evolve inwardly before its public conceptual role
   is reconsidered.
4. Iterative/eigensolver workspace work should follow the same wrapper-backed
   model, but through composition rather than factor-handle ownership.
5. Compatibility scaffolding should stay narrow:
   - wrapper preservation
   - bridge adapters
   - documentation shims
   - delayed deprecation
6. Cancellation and copy-before-reuse remain first-class migration risks, not
   secondary cleanup details.

## Day 10 Output for Later Sprints

Sprint 40 now leaves later Epic 4 sprints with:

- a target object model from Day 9
- a staged migration contract from Day 10
- earliest safe insertion points
- a bounded compatibility-scaffolding model
- an explicit high-risk edge inventory

That is enough to let Sprint 42 and later lifecycle refactors start from a
written architecture contract instead of rediscovering the migration shape
during code churn.
