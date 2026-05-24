# Sprint 40 Day 9: Future Handle-Model Design I

## Objective

Define the first concrete explicit handle model for Epic 4 using the Sprint 40
taxonomy and contract map. This is not yet a migration plan; it is the target
object model and responsibility split that later implementation work should
move toward.

## Design Inputs

This design is derived from:

- `day7-state-model-taxonomy.md`
- `day8-lifecycle-contract-map.md`
- the Epic 4 review / remediation plan

## Design Goals

The target model should:

1. preserve the strongest caller-visible semantics that already matter:
   - whether an API mutates input or not
   - whether original matrix state is required
   - whether cancellation preserves input
2. reduce caller dependence on hidden `SparseMatrix` lifecycle state
3. eliminate matrix-as-factor-handle overloading where practical
4. preserve iterative/eigensolver “operator consumer” semantics rather than
   forcing them into direct-factor object shapes
5. support an internal-first migration path before any broad public API churn

## Proposed High-Level Object Model

Epic 4 should converge on four main lifecycle object families:

1. coefficient matrix objects
2. symbolic analysis objects
3. numeric factor / decomposition objects
4. solve / operation context objects

### 1. Coefficient Matrix Object

#### Purpose

`SparseMatrix` should primarily represent:

- sparse coefficient storage
- matrix shape
- value-level editing
- query/matvec utility behavior

#### Long-term role

It should remain the public “matrix value object,” but it should stop carrying
the majority of long-lived factor lifecycle semantics.

#### Responsibilities to keep on `SparseMatrix`

- dimensions
- structural/value storage
- logical element access
- matrix arithmetic / transpose / SpMV
- cached read-mostly utility state that is purely value-derived
  - example: norm cache, if still worthwhile

#### Responsibilities to move away from `SparseMatrix`

- “this matrix is now a factor handle”
- permutation ownership as the primary public solve-state carrier
- factor-specific solve acceptance state (`factored`)
- factor-specific singularity scaling state (`factor_norm`) as a public-facing
  lifecycle dependency
- cancellation-sensitive partial factor state

### 2. Symbolic Analysis Object

#### Purpose

A symbolic analysis handle should own:

- fill-reducing permutation choice/result
- etree / postorder / symbolic structures
- matrix-eligibility validation results, if retained

#### Existing bridge

`sparse_analysis_t` already occupies this role well enough to serve as the
prototype for the future model.

#### Design direction

The analysis object should remain distinct from both:

- the coefficient matrix
- the numeric factor object

That separation already matches caller mental models and should be preserved.

### 3. Numeric Factor / Decomposition Objects

#### Purpose

Factor/decomposition objects should own:

- numeric factor data
- factor-specific permutations
- factor-specific norm/tolerance telemetry
- factor-specific cancellation cleanup
- solve readiness

#### Target family

The future internal family should conceptually look like:

- LU factor handle
- Cholesky factor handle
- LDLT factor handle
- QR factor handle
- SVD result/decomposition handle
- ILU / IC preconditioner handle

Some of these already exist publicly. LU and Cholesky are the major gaps.

#### Key design rule

No factor object should require callers to reason about mutating the original
matrix into a solve handle. The factor object, not the matrix, should own
“factorized state.”

### 4. Solve / Operation Context Objects

#### Purpose

Repeated-run mutable state that is not part of the matrix or factor itself
should live in context/workspace objects:

- iterative solver workspaces
- eigensolver workspaces
- optional reusable scratch for repeated factorizations/solves

#### Scope

Sprint 40 is only defining the object family boundary here, not its final API.
The main point is to separate:

- persistent mathematical result
from
- reusable temporary workspace

## Role Split By Current Taxonomy Class

### Class 1: Matrix-mutating factor builders

#### Current state

- LU / Cholesky overload `SparseMatrix`

#### Target state

Move toward:

- `SparseMatrix` as coefficient input
- explicit LU/Cholesky factor handles as solve state owners

#### Compatibility principle

Public convenience routines may still exist in one-shot form, but they should
be wrappers over explicit handle-backed internals rather than direct matrix
state mutation as the core architecture.

### Class 2: Original-matrix consumers with separate result handles

#### Current state

- already close to the desired model

#### Target state

Preserve separate-handle shape, but reduce caller burden around:

- identity-permutation eligibility
- “must still be original” reasoning

#### Design implication

Where possible, matrix eligibility validation should become:

- explicit in API naming / handle creation boundaries
or
- internalized by constructing protected working views internally

instead of relying on callers to infer state from prior operations.

### Class 3: Analysis / factor handle pipelines

#### Current state

- public handle split is already good
- `sparse_factors_t` still wraps matrix-centric payload internally

#### Target state

Preserve:

- analysis handle
- numeric factor handle
- solve against factor handle

But evolve `sparse_factors_t` toward a true factor payload, not a wrapper
around a factored `SparseMatrix`.

### Class 4: Read-only operator consumers

#### Current state

- iterative/eigensolver entry points already have the right basic semantics

#### Target state

Keep them as operator consumers. Do not redesign them around direct-factor
handle ownership unless they explicitly compose with such handles.

#### Design implication

Their improvement path is:

- clearer composition boundaries
- optional reusable workspaces
- cleaner preconditioner/context contracts

not “become factor APIs.”

### Class 5: Bridge / mixed-boundary surfaces

#### Current state

- `sparse_factors_t`
- solver+preconditioner workflows
- eigensolver + shift-invert / refinement composition

#### Target state

These should become the intentional seam objects between families:

- matrix object
- factor handle
- operation context

Rather than accidental seams that still expose internal matrix lifecycle
details.

## Keep / Move Responsibility Split

### Keep on `SparseMatrix`

- coefficient/value semantics
- structural editing semantics
- matrix arithmetic/query semantics
- utility read-only operations
- possibly purely value-derived caches

### Move behind explicit factor/decomposition handles

- factor-specific permutation ownership
- solve eligibility state
- factor-specific norm/tolerance state
- factor-specific cancellation cleanup
- structural transformations that produce solve-ready state

### Move behind analysis handles

- reorder choice/result ownership
- symbolic factor structure ownership
- analysis-specific norm/reference data where it exists only to support
  factorization

### Move behind reusable operation contexts

- iterative solver workspaces
- eigensolver workspaces
- repeated-run temporary buffers

## Proposed Internal-First Migration Shape

Day 9 is not the full migration plan, but the safest staged shape is already
clear:

### Stage 1: Internal handle introduction

- introduce internal LU/Cholesky factor payloads
- stop treating raw `SparseMatrix` as the only internal factored-state carrier
- keep public entry points stable

### Stage 2: Bridge-object normalization

- evolve `sparse_factors_t` to wrap explicit factor payloads instead of a
  factored matrix-centric payload
- normalize solver/preconditioner composition boundaries

### Stage 3: Eligibility/cancellation simplification

- reduce direct caller dependence on hidden matrix lifecycle state
- make cancellation semantics derive from handle boundaries rather than
  matrix-field knowledge

## Minimum Compatibility Scaffolding

The smallest compatibility layer later sprints will likely need is:

### 1. One-shot wrapper preservation

Keep public convenience routines that callers already use, but route them to:

- explicit internal factor handles
- disposable internal working copies when needed

### 2. Bridge accessors/adapters

Temporary adapters will likely be needed so existing solve/refine/condest
paths can operate while internal factor payloads are changing.

### 3. Transitional factor wrapper

`sparse_factors_t` should probably be the first public wrapper preserved while
its internals change underneath.

### 4. Explicit documentation boundary

Compatibility docs should emphasize:

- stable semantic promises
- not internal field ownership

So callers do not learn future-unstable details during the migration.

## Major Design Decisions From Day 9

1. `SparseMatrix` should remain the coefficient/value object, not the long-term
   universal factor state carrier.
2. LU and Cholesky should move toward the same explicit factor-handle family
   that LDLT, QR, SVD, and preconditioner builders already approximate.
3. The analysis pipeline is the best existing public bridge and should remain a
   first-class architecture anchor.
4. Iterative solvers and eigensolvers should remain operator-consumer APIs,
   improved primarily through composition boundaries and workspaces.
5. The public contract should preserve semantics, while internal field-level
   lifecycle machinery becomes progressively less visible.

## Open Questions For Day 10

Day 10 should finish the migration strategy around:

- how to stage LU/Cholesky internal handle landing without destabilizing
  current call patterns
- whether some identity-permutation/original-state rules should stay explicit
  or be internalized behind protected working copies
- what compatibility/deprecation surfaces are actually required versus merely
  convenient
- which bridge objects need to land first so later implementation work stays
  incremental
