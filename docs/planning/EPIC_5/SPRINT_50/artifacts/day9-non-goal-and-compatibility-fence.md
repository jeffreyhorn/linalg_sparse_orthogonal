# Sprint 50 Day 9 Artifact: Non-Goal and Compatibility Fence

## Purpose

Record the explicit scope fence that keeps the new direct-solver lifecycle work
bounded: what Sprint 50-52 are allowed to change, what they must preserve, and
what remains a conscious non-goal for Epic 5.

## Why Day 9 Matters

After Day 8, the public repeated-run direct contract is concrete enough to
implement. That is exactly the point where scope drift becomes most likely.

Without a written fence, later work could easily widen into:

- a broad direct-solver API rewrite
- accidental weakening of one-shot compatibility
- exposure of internal CSC/native structures
- benchmark/framework churn that does not close the actual lifecycle gap

Day 9 prevents that by turning the current boundary from implication into
explicit policy.

## Allowed Change Set For Sprint 50-52

The later direct-lifecycle implementation sprints are allowed to:

### 1. Clarify and strengthen the public analysis/factor/refactor story

- make the existing public repeated direct workflow easier to discover
- align header wording with the Day 8 contract
- make the repeated-run caller story explicit in the highest-value examples and
  benchmarks

### 2. Add bounded lifecycle-supporting public API refinements where justified

- small additive helper/API refinements around `sparse_analysis_t` and
  `sparse_factors_t`
- bounded wording or helper changes that improve zero/init/free clarity
- bounded wrapper or entry-point adjustments that reinforce the explicit
  repeated-run story without replacing the current public model

### 3. Preserve and improve stable-pattern repeated-run behavior

- better alignment between public docs and the real analysis/factor/refactor
  lifecycle
- direct test and benchmark coverage for the intended repeated-run path
- improved compatibility checks and wording around same-pattern refactor use

### 4. Keep family-specific behavior explicit while unifying the lifecycle story

- LU, Cholesky, and LDL^T may share the same repeated-run caller framing
- family-specific option/result behavior may remain distinct where it reflects
  real mathematical or storage differences

## Explicit Non-Goals

The following are out of scope for Sprint 50-52 unless a later Epic explicitly
reopens them.

### 1. No broad public factor-container redesign everywhere at once

Epic 5 is not trying to replace every direct public API with a new unified
factor container hierarchy.

Not allowed:

- sweeping replacement of family-specific public types
- new generic direct solver framework abstractions as the main Sprint 50-52
  deliverable

### 2. No removal or demotion of one-shot direct APIs

The one-shot direct APIs remain first-class supported public paths.

Not allowed:

- deprecation framing for LU / Cholesky / LDL^T one-shot entry points
- migration language that implies forced conversion
- implementation choices that make one-shot paths second-class by design

### 3. No raw internal storage exposure

Epic 5 does not expose:

- CSC supernodal internal factor containers
- linked-list/private factor storage layout
- analysis-aware CSC helper names as public API
- backend-private scratch or telemetry objects

### 4. No unrelated solver-family expansion

Sprint 50’s lifecycle design is centered on:

- LU
- Cholesky
- LDL^T

Not allowed:

- broad QR lifecycle redesign inside this sprint slice
- unrelated iterative/eigensolver revisiting
- new solver-family feature expansion under the cover of lifecycle work

### 5. No broad benchmark-framework redesign

Benchmarks may be updated where they are the highest-signal compatibility
surface for the repeated direct-run story, but Epic 5 is not a benchmark
infrastructure epic.

Not allowed:

- broad benchmark CLI redesign
- benchmark framework consolidation work unrelated to the direct lifecycle gap
- backend-heavy benchmark churn presented as lifecycle progress

### 6. No structural-pattern verifier redesign in Sprint 50-52

The current refactor contract remains:

- same-pattern structural compatibility is a caller precondition

Not allowed:

- broad new structural-equality validation layer as part of the first direct
  lifecycle landing

## Compatibility Contract That Must Be Preserved

### 1. One-shot compatibility is a conscious contract, not an accident

Sprint 50-52 must preserve that callers can still use:

- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`
- `sparse_cholesky_factor(...)`
- `sparse_cholesky_factor_opts(...)`
- `sparse_ldlt_factor(...)`
- `sparse_ldlt_factor_opts(...)`

These are not fallback leftovers. They remain:

- supported
- documented
- appropriate for one-off or low-context solves

### 2. Mutable `SparseMatrix` behavior remains an accepted tradeoff

For LU and Cholesky, the public compatibility story still includes:

- factorization on a copied matrix when the original matrix view matters
- mutation of matrix-carried factor/reorder state in the one-shot path

Epic 5 may clarify this more explicitly, but it does not remove that behavior.

### 3. Family-specific semantics remain real API differences

Sprint 50-52 must preserve explicit differences such as:

- symmetry requirements
- pivoting behavior
- option-struct shape
- reorder/backend nuances
- one-shot mutation versus factor-object behavior

The lifecycle story should unify the repeated-run framing without pretending
those differences are cosmetic.

### 4. Reuse semantics stay narrow

Sprint 50-52 must preserve the Day 8 truth:

- reuse preserves symbolic/permutation setup, not old numeric factor state

That sentence is part of the compatibility fence because later wording or
implementation choices must not blur it.

## Accepted Epic 5 Tradeoffs

These are not bugs Day 9 expects Sprint 50-52 to “solve.”

### 1. The direct public surface remains mixed by design

Even after the lifecycle work lands, the project still keeps:

- one-shot family APIs
- explicit analysis/factor/refactor workflow

That mixed model is an accepted compatibility tradeoff rather than a failure.

### 2. The one-shot direct story remains mutation-aware

Epic 5 may reduce surprise through clearer lifecycle framing, but it does not
turn the one-shot direct path into an immutable-matrix model.

### 3. Same-pattern structural compatibility remains caller-owned

The repeated direct lifecycle will be clearer, but it will still rely on the
caller to respect the same-pattern precondition for refactorization.

## Sprint 50 Design vs Sprint 51+ Implementation Boundary

### Sprint 50 design owns

- lifecycle contract wording
- non-goals
- compatibility fence
- adoption-boundary decisions
- validation/landing planning

### Sprint 51+ implementation owns

- header edits
- source integration
- targeted test additions
- example/benchmark adoption where justified
- validation execution

This is an important fence because the current sprint should not try to
pre-solve code-shape or test-layout decisions that only become concrete once
implementation begins.

## Adoption Boundary

The explicit contract and scope fence together imply:

### Early adopters

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`
- the public headers most directly tied to analysis/factor/refactor wording

### Intentional lagging surfaces

- small one-shot examples
- `examples/README.md` one-shot teaching surfaces
- `benchmarks/bench_refactor_csc.c`
- broader README/tutorial reshaping

This keeps the first implementation slices aligned with the highest-value
repeated-run surfaces instead of forcing broad documentation churn.

## Highest-Value Day 9 Conclusions

### 1. Epic 5 is an additive lifecycle-centering effort, not a direct-solver rewrite

That is now a written scope rule rather than an assumption.

### 2. One-shot direct APIs remain first-class by explicit contract

The repeated-run lifecycle work is not a migration-away-from-one-shot project.

### 3. Internal storage exposure is now clearly fenced off

This protects Sprint 51-52 from widening into CSC/native storage API work.

### 4. The remaining work can now proceed with a stable scope boundary

Sprint 50 no longer needs more contract-shape discovery. It needs validation
and landing planning inside a fixed compatibility fence.
