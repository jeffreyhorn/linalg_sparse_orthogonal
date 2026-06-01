# Sprint 50 Day 7 Artifact: Post-Design Audit

## Purpose

Audit the Day 6 public direct-solver lifecycle design against the Day 5 ranked
gap list and the inherited Epic 4 compatibility boundary, then reduce the
remaining Sprint 50 queue to a small set of true public-contract decisions.

## Audit Question

Day 6 produced a concrete first-pass contract:

- analysis-centric bounded hybrid
- explicit lifecycle stages
- LU / Cholesky / LDL^T coverage
- one-shot APIs preserved as first-class peers

Day 7 therefore asks a narrower question:

- what still needs a public decision in Sprint 50
- what should stay explicitly out of scope until Sprint 51 implementation work

## Gap-Closure Audit

### 1. The Day 6 design substantially closes the “repeated direct workflow is under-centered” gap

Day 5’s highest-ranked issue was that repeated direct workflow was real but not
the dominant public caller story.

Day 6 improves that materially by making the intended repeated-run path
explicit:

- `sparse_analysis_t`
- `sparse_factors_t`
- analyze once
- factor / solve
- refactor / solve
- free

Audit result:

- mostly resolved at the design level

Residual question:

- Sprint 50 still needs to finalize how strongly that path is framed as the
  default repeated direct-run story in the final contract wording

### 2. The Day 6 design narrows, but does not erase, the hidden mutable-state gap

Day 6 did not try to redesign one-shot LU / Cholesky mutation semantics.
That is correct for scope, but it means the public contract still needs one
more explicit decision:

- what should remain one-shot-first because mutation is part of the
  compatibility story

Audit result:

- intentionally only partially resolved

Interpretation:

- Sprint 50 should clarify the relationship between explicit repeated-run
  analysis/factors and the one-shot mutable matrix path
- Sprint 50 should not attempt to eliminate the mutable-matrix model itself

### 3. The Day 6 design substantially closes the “factor-many efficiency is under-centered” gap

The analysis/factor/refactor story is now clearly positioned as:

- the stable-pattern repeated-run path
- the factor-many performance path

Audit result:

- resolved at the design level

Residual question:

- only the caller-facing phrasing still needs to be finalized so the
  performance story reads intentional rather than incidental

### 4. The Day 6 design materially narrows the multiple-model maintainability gap

Before Day 6, the direct public surface read as three loosely related models:

- one-shot mutating LU / Cholesky
- one-shot factor-object LDL^T
- analysis/factor/refactor bridge

Day 6 now gives them a cleaner relationship:

- analysis/factor/refactor = explicit repeated-run path
- family-specific one-shot APIs = simple/default peer paths

Audit result:

- mostly resolved

Residual question:

- Sprint 50 still needs to decide the exact wording for “peer entry point”
  versus “simple/default path” so later docs and examples do not drift again

### 5. The docs/examples gap should lag the contract on purpose

Day 5 identified docs/example imbalance, but Day 6 correctly did not try to
rewrite those surfaces yet.

Audit result:

- still intentionally open

Interpretation:

- Sprint 50 should first finish the contract
- examples and benchmark docs should only adopt the final wording after the
  contract language is stable

## What Should Stay One-Shot-First

Day 7’s strongest compatibility conclusion is that some caller stories should
remain explicitly one-shot-first even after the repeated lifecycle is centered.

### Keep one-shot-first

- simple single-solve LU usage on copied matrices
- simple single-solve Cholesky usage on copied matrices
- small examples whose main job is to teach basic public factor-and-solve flow
- family-local option behavior that only matters during one-shot factorization

Reason:

- these paths are simpler
- they are already publicly supported
- forcing them through a repeated-run framing would make the API look more
  abstract than the caller needs

## What Should Stay Internal-Only

The Day 6 model is cleaner specifically because it does not expose several
tempting implementation details.

### Keep internal-only

- CSC/native factor storage layout
- analysis-aware CSC helper names
- backend-selection plumbing details beyond current public option structs
- structural-pattern validation machinery beyond the current caller
  precondition
- generic “direct handle” storage abstractions
- benchmark-oriented helper seams

Reason:

- these are implementation details or performance plumbing
- exposing them now would broaden the public surface without solving the main
  lifecycle gap

## Example and Benchmark Adoption Boundary

### Should adopt the final repeated-run story early

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

Reason:

- both already model the analysis/factor/refactor workflow directly
- they are the highest-signal compatibility surfaces for the explicit repeated
  direct-run story

### Can lag intentionally

- small one-shot examples in `examples/README.md`
- family-specific one-shot examples
- `benchmarks/bench_refactor_csc.c`

Reason:

- the small one-shot examples are intentionally simple
- the CSC benchmark is primarily an analysis-aware backend/perf surface, not
  the main public repeated-run teaching surface

## Must Decide In Sprint 50

These are the remaining questions that still affect the public contract itself.

### 1. Final lifecycle wording for zero/init expectations

Sprint 50 still needs to decide whether the final caller story is:

- zero-init only
- zero-init plus optional additive init helpers later

What must be fixed now:

- zeroed structs are valid starting state
- free is safe on zeroed state
- the public docs should read as intentionally designed that way

### 2. Final public wording for analyze-once / factor-refactor-many

Sprint 50 must decide how explicitly to state that this is the preferred
stable-pattern repeated direct-run path.

### 3. Final relationship wording between one-shot APIs and the repeated lifecycle

Sprint 50 must lock down whether one-shot APIs are described as:

- peer entry points
- simple/default path
- or both, with different emphasis by caller context

### 4. Final reuse meaning

Sprint 50 must explicitly define that reuse preserves:

- symbolic/permutation setup

and does not preserve:

- old numeric factor state as an incremental-update guarantee

## Should Wait For Sprint 51+

These questions are real, but they are implementation or follow-on validation
detail rather than Sprint 50 contract design.

### 1. Exact code-shape changes in headers and source files

- additive helper names
- wrapper routing details
- implementation layering

### 2. Direct regression-test expansion shape

- which exact tests should cover the new contract
- where parity checks belong

### 3. Broader documentation rewrite

- top-level README reshaping
- tutorial restructuring
- broader example modernization beyond the highest-signal surfaces

### 4. Benchmark framework adjustments

- broad refactor benchmark redesign
- exposing backend-specific repeated-run knobs publicly

## Bounded Target Set For Day 8

Day 8 should finalize only these contract details:

1. zero/init/free expectations
2. analyze/prepare wording
3. solve/refactor/reuse wording
4. one-shot API relationship wording
5. caller-facing reuse meaning and non-meaning

Day 8 should not try to solve:

- implementation helper naming
- CSC/internal storage exposure
- large docs/example rewrites
- Sprint 51 validation details

## Highest-Value Day 7 Conclusions

### 1. The analysis-centric shape still holds after audit

No stronger alternative surfaced. A new generic direct handle would still be
too broad, and a docs-only centering move would still be too weak.

### 2. The remaining Sprint 50 queue is now mostly wording and boundary work, not architecture search

The major architecture decision is already made. The next step is to finalize
the exact public contract language.

### 3. Example and benchmark adoption should be selective, not universal

`example_analysis.c` and `bench_refactor.c` are the right early adopters.
Small one-shot examples and backend-heavy performance surfaces should lag
deliberately.

### 4. Sprint 50 remains bounded away from Sprint 51 implementation planning

The real remaining work is still public-contract finalization, not code-shape
or regression-suite design.
