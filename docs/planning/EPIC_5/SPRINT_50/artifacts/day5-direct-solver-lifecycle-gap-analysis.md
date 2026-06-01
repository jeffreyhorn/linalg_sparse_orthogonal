# Sprint 50 Day 5 Artifact: Direct-Solver Lifecycle Gap Analysis

## Purpose

Turn the Day 3 public-surface inventory and Day 4 precedent inventory into a
ranked lifecycle gap map for usability, correctness, efficiency, and
maintainability, and fix the smallest credible public-exposure target for the
later Sprint 50 design days.

## Target Qualities

Sprint 50 is trying to move the direct-solver side closer to:

- explicit lifecycle
- reduced hidden mutable state
- stronger factor-many guidance
- compatibility preservation

The Day 5 question is therefore not “what could be redesigned?” but “what
small set of lifecycle gaps blocks those qualities today?”

## Ranked Gap List

### 1. Highest: the repeated direct workflow is real but still under-centered publicly

The repo already exposes a public repeated direct workflow:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`
- `sparse_factor_solve(...)`

Yet the dominant direct-solver caller story still reads as one-shot-first.

Why it matters:

- users can miss the best repeated-run path even though it already exists
- the direct-solver public story looks less explicit than the iterative and
  eigensolver repeated-run story after Epic 4
- public docs, examples, and benchmarks naturally keep over-centering the
  one-shot path when the core contract is not clearly centered

### 2. High: one-shot LU / Cholesky still depend too heavily on hidden mutable matrix-state knowledge

The strongest remaining correctness-risk seam is not a known numerical defect.
It is misuse potential:

- callers need to know when to `sparse_copy()`
- factor state and permutations are stored inside the mutated `SparseMatrix`
- preserving the original matrix view is still largely a documentation rule

Why it matters:

- the public surface is easy to use correctly once the user understands the
  model
- but it is still too easy to use wrong without that model already in mind

### 3. High: factor-many efficiency is public but still not clearly the default performance story

The analysis/refactor bridge already gives the project a factor-many path, but
it is not yet obviously the public answer to stable-pattern repeated direct
solves.

Why it matters:

- the system already has the mechanism for repeated direct reuse
- users still see a specialist bridge rather than a first-class repeated-run
  contract
- that leaves both usability and performance value under-realized

### 4. Medium: the public direct lifecycle is split across three models that are each reasonable alone but not yet reconciled together

Current models:

- matrix-mutating one-shot
- factor-object one-shot
- analysis/factor/refactor

Why it matters:

- each model is individually defensible
- together they create long-term documentation, example, and test drift unless
  Sprint 50 defines how they relate

### 5. Medium: docs and examples still over-center the one-shot path

This is real, but secondary:

- examples intentionally stay simple
- docs currently mirror the dominant public contract
- as long as the repeated direct contract stays under-centered, docs and
  examples will keep reflecting that imbalance

Why it matters:

- documentation cleanup alone will not solve the lifecycle problem
- but the final API design has to leave behind a clearer story for later docs
  work

## Strongest Constraints

### Compatibility constraints

Sprint 50 must preserve:

- one-shot direct public APIs as first-class supported paths
- existing mutable-`SparseMatrix` compatibility behavior where already public
- family-specific differences between LU, Cholesky, and LDL^T

### Structural constraints

Sprint 50 should not expose:

- CSC/native internal layout
- backend-specific storage contracts
- analysis-aware CSC helper names as public API
- broad new direct-solver framework machinery

### Precedent constraints

Sprint 50 should borrow from:

- `sparse_analysis.h` for domain-specific repeated direct workflow
- Sprint 49 public handles for generic prepare/run/free and reuse semantics

## Minimum-Credible Public Lifecycle Exposure

The smallest credible public move that would materially improve the project is:

1. center the analysis/factor/refactor path as the explicit repeated direct-run
   story
2. clarify or extend its lifecycle ownership semantics where needed
3. preserve one-shot LU / Cholesky / LDL^T entries as compatibility-first
   public paths
4. avoid exposing raw internal CSC/native state or broad new public solver
   frameworks

This rules out both extremes:

- too small:
  - docs-only cleanup with no contract clarification
- too large:
  - broad new direct-solver API redesign

## Highest-Value Day 5 Conclusions

### 1. Sprint 50’s direct-lifecycle problem is now specific, not abstract

It is no longer “direct solver state is awkward.” It is:

- repeated direct workflow exists but is under-centered
- one-shot mutable-state dependence remains the main misuse seam
- factor-many efficiency is public but not yet obviously first-class

### 2. The right design target is bounded and additive

Sprint 50 should not remove or deprecate the one-shot paths. It should make the
explicit repeated direct lifecycle easier to see, easier to reason about, and
easier to build later implementation work around.

### 3. Day 6 can now design against a narrow ranked queue instead of a generic backlog

The Day 6 design batch should primarily address:

1. centering the repeated direct workflow
2. reducing hidden mutable-state dependence at the public-contract level
3. making the factor-many path feel intentional rather than specialist
