# Sprint 50 Day 6 Artifact: Public Direct-Solver Lifecycle API Design Batch I

## Purpose

Turn the Day 5 ranked gap list into a bounded first-pass public lifecycle
design for direct solvers, with explicit decisions on abstraction shape,
lifecycle stages, naming, and first-model coverage.

## Design Goal

Sprint 50 does not need a brand-new public direct-solver framework. It needs a
clearer and more intentional repeated direct-run contract that:

- centers the existing analysis/factor/refactor workflow
- preserves one-shot compatibility paths
- reduces hidden-state surprise in the public caller story
- stays small enough to implement without reopening broad family redesign

## Main Shape Decision

### Decision: use an analysis-centric bounded hybrid, not a brand-new generic direct handle

The preferred first-pass public lifecycle model is:

- keep `sparse_analysis_t` as the public prepare/analyze object
- keep `sparse_factors_t` as the public numeric-state object
- treat those two objects together as the primary explicit repeated direct-run
  lifecycle
- allow small additive helper/API refinements later if the lifecycle contract
  needs to be made easier to initialize, validate, or document

Rejected Day 6 alternatives:

- pure “do nothing, docs only”:
  - too small; it leaves the repeated direct workflow under-centered
- brand-new generic `sparse_direct_handle_t`:
  - too broad; it duplicates an already-real direct lifecycle instead of
    clarifying it
- family-specific new public handle types for LU / Cholesky / LDL^T:
  - too large for Sprint 50 and too likely to widen public-shape drift

Interpretation:

- the direct-solver side already has a meaningful public lifecycle object model
- Sprint 50 should make that model feel first-class rather than replacing it

## Target Public Lifecycle Stages

The first-pass direct repeated-run lifecycle should read publicly as:

1. initialize / zero
2. analyze / prepare
3. factor
4. solve
5. refactor / reuse
6. free

### 1. Initialize / zero

Caller expectations:

- `sparse_analysis_t` and `sparse_factors_t` may start as zeroed structs
- the first successful prepare/factor call owns populating them
- free must remain safe on zeroed or partially empty state

Why this stage matters:

- it matches the existing direct public precedent
- it matches the broader repo lifecycle safety contract from Epic 4

### 2. Analyze / prepare

Public meaning:

- choose factor family
- choose reorder policy
- compute reusable symbolic structure and permutation state
- establish the structural contract for later numeric factor/refactor work

Day 6 design rule:

- “prepare” should remain direct-solver vocabulary layered on top of
  `sparse_analyze(...)`, not a separate generic opaque phase with new storage

Interpretation:

- the direct-solver prepare step is analysis-specific, not just generic buffer
  reservation

### 3. Factor

Public meaning:

- compute numeric factors for one analyzed structural pattern
- materialize the factor state in `sparse_factors_t`

Day 6 design rule:

- the repeated direct lifecycle should center `sparse_factor_numeric(...)` as
  the explicit factor step
- one-shot LU / Cholesky / LDL^T remain valid public alternatives for simple
  or compatibility-driven use

### 4. Solve

Public meaning:

- solve using already-computed factor state
- preserve analysis/factor state for later solves or refactorization

Day 6 design rule:

- `sparse_factor_solve(...)` is the primary explicit repeated-run solve step
- solve is read-only on prepared analysis/factor state

### 5. Refactor / reuse

Public meaning:

- reuse the analyzed structural pattern
- replace numeric factor state for new values with the same sparsity pattern
- preserve setup investment, not old numeric state

Day 6 design rule:

- reuse must mean:
  - same-pattern repeated numeric work
  - preserved symbolic/permutation setup
  - overwritten factor state on success
- reuse must not mean:
  - preserving old triangular data as an incremental update contract
  - structural-pattern validation beyond the stated caller precondition
  - public exposure of backend-specific CSC/native workspace

### 6. Free

Public meaning:

- explicit teardown of analysis and factor lifecycle state
- safe on zeroed objects
- no hidden ownership of the source matrix survives free

Interpretation:

- the repeated direct lifecycle remains explicit and caller-owned end to end

## First-Model Coverage Decision

### Explicitly covered in the first public lifecycle model

- LU
- Cholesky
- LDL^T

Reason:

- these are the direct families already connected to the public
  analysis/factor/refactor bridge
- they represent the strongest current direct repeated-run value
- leaving any of the three out would keep the lifecycle story fragmented

### Not a first-model target

- QR

Reason:

- QR is still useful as a lifecycle contrast surface
- but Sprint 50 does not need QR to define the direct repeated-run baseline
- pulling QR into the first contract would broaden scope without closing the
  highest-value gap

## Relationship To Existing One-Shot APIs

### One-shot APIs stay first-class

Sprint 50 should explicitly preserve:

- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`
- `sparse_cholesky_factor(...)`
- `sparse_cholesky_factor_opts(...)`
- `sparse_ldlt_factor(...)`
- `sparse_ldlt_factor_opts(...)`

Caller-facing interpretation:

- one-shot APIs remain the simple/default path for single-run or low-context
  direct solves
- the analysis/factor/refactor lifecycle becomes the explicit opt-in path for
  stable-pattern repeated runs

### Day 6 compatibility rule

Do not describe the one-shot APIs as deprecated, legacy-only, or second-class.
The correct relationship is:

- repeated direct lifecycle:
  - explicit performance-oriented stable-pattern path
- one-shot APIs:
  - compatibility-preserving, simple, first-class peer entry points

## Naming and Terminology Decisions

### 1. Prefer “analysis” and “factor” over generic “handle”

Use direct domain vocabulary first:

- analysis
- factors
- factorization type
- refactor
- repeated direct run

Avoid re-centering the design around generic names like:

- handle
- workspace
- context

unless a later Sprint 50 design pass proves a small additive helper is needed.

Reason:

- `sparse_analysis.h` is already the strongest public direct precedent
- generic naming would flatten real direct-solver semantics that callers
  actually need to reason about

### 2. Prefer “analyze once, factor/refactor many” as the public repeated-run story

This should become the direct-solver equivalent of the repeated-run value
statement, because it is both true and already implemented enough to support.

### 3. Keep family differences explicit

Do not pretend LU / Cholesky / LDL^T are interchangeable under one abstract
story. The shared lifecycle should coexist with explicit differences in:

- symmetry requirements
- mutation behavior of one-shot paths
- pivoting behavior
- reorder/backend details

## Public Contract Boundaries

### In scope for the lifecycle model

- zero/init expectations
- analyze/factor/refactor/solve/free stages
- same-pattern reuse framing
- first-class repeated-run guidance
- relationship to one-shot APIs

### Out of scope for Sprint 50 Day 6

- raw CSC/native factor storage exposure
- backend-specific lifecycle objects
- broad new family-specific direct handles
- structural-pattern verifier redesign
- removing mutable-`SparseMatrix` one-shot behavior
- tutorial/example rewrite details

## Highest-Value Day 6 Conclusions

### 1. The repeated direct workflow should be centered, not reinvented

The right first move is to promote the existing analysis/factor/refactor path
into the clearly intended repeated-run direct contract.

### 2. The first public lifecycle model must explicitly cover LU, Cholesky, and LDL^T

Anything smaller would leave the project with the same public-shape split that
Day 5 identified as the main maintainability risk.

### 3. The lifecycle is direct-specific even when it borrows generic pattern rules

Sprint 49 handle semantics help with init/reuse/free safety, but the direct
side still needs analysis-centric vocabulary and same-pattern refactor
semantics.

### 4. Day 7 can now audit a concrete contract instead of a generic ambition

The remaining open questions are no longer “should direct solvers get a public
lifecycle?” They are narrower:

- does the first-pass analysis-centric shape fully cover the repeated-run story
- what should stay one-shot-first
- what tiny additive helper surface, if any, is justified later
