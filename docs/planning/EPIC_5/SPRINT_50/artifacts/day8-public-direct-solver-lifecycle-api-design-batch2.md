# Sprint 50 Day 8 Artifact: Public Direct-Solver Lifecycle API Design Batch II

## Purpose

Finalize the caller-facing public contract for direct repeated-run lifecycle
work: zero/init expectations, analyze/factor/refactor/solve/free semantics,
reuse meaning, result/option-struct expectations, and the relationship between
the explicit repeated-run path and existing one-shot APIs.

## Final Contract Summary

Sprint 50’s final public direct repeated-run contract should read as:

- zero-initialize `sparse_analysis_t` and `sparse_factors_t`
- analyze once for a chosen direct factor family and reorder policy
- factor and solve as needed
- refactor on new values with the same sparsity pattern
- free both lifecycle objects explicitly

This is the intended stable-pattern repeated direct-run path for:

- LU
- Cholesky
- LDL^T

The one-shot family APIs remain first-class peer paths for simple or
low-context solves.

## Final Lifecycle Contract

### 1. Zero / init expectations

The final caller story should be:

- `sparse_analysis_t` may begin as a zeroed struct
- `sparse_factors_t` may begin as a zeroed struct
- zeroed structs are the normative initial state for first use
- free functions are safe on zeroed structs

Sprint 50 decision:

- zero-init is sufficient for the public contract
- optional additive init helpers may be considered later, but they are not
  required to make the lifecycle coherent

Why this is the right final rule:

- it matches the existing direct public precedent
- it matches the broader repo lifecycle safety norm
- it avoids introducing extra public ceremony before Sprint 51 proves any
  helper need concretely

### 2. Analyze / prepare semantics

The final caller-facing meaning of the prepare step is:

- choose direct factor family
- choose reorder policy
- compute reusable symbolic/permutation state
- establish the same-pattern contract for later numeric work

Caller-facing wording should prefer:

- analyze
- prepare
- analysis object

It should avoid implying:

- generic buffer reservation
- opaque workspace-only setup
- backend-specific storage commitments

Final Day 8 rule:

- “prepare” is descriptive caller vocabulary layered on top of
  `sparse_analyze(...)`
- `sparse_analyze(...)` remains the real public analysis/prepare entry point

### 3. Factor semantics

The final factor step should be described as:

- build numeric factor state for one analyzed structure
- materialize that state in `sparse_factors_t`
- preserve the analysis object for later solve or refactor calls

Final Day 8 rule:

- `sparse_factor_numeric(...)` is the explicit repeated-run factor step
- it is not a second symbolic-analysis phase

### 4. Solve semantics

The final solve step should be described as:

- solve with already-prepared numeric state
- read analysis/factor state without consuming it
- permit multiple solves per factorization

Final Day 8 rule:

- `sparse_factor_solve(...)` is the explicit repeated-run solve step
- solve is logically read-only on prepared lifecycle state

### 5. Refactor semantics

The final refactor step should be described as:

- reuse the analyzed symbolic/permutation structure
- replace numeric factor state using new values on the same sparsity pattern
- preserve previous numeric factors on failure

This should remain explicit because it is one of the highest-value direct-side
behavior differences from the one-shot path.

### 6. Reuse and reset semantics

The final public meaning of reuse is:

- preserve setup investment
- preserve symbolic/permutation structure
- permit repeated factor/solve cycles for stable-pattern matrices

The final public meaning of reset is:

- there is no special public “reset to warm numeric state” contract
- reuse is driven by re-analysis when the structure changes, or refactor when
  the structure is unchanged

### 7. Free semantics

The final teardown story should be:

- callers own `sparse_analysis_t`
- callers own `sparse_factors_t`
- free is explicit
- free is safe on zeroed objects
- free leaves the object reusable as a zeroed state

## Final Reuse Meaning

Sprint 50 needs one sentence of behavioral truth that is short enough to stay
stable across headers, docs, examples, and benchmarks:

- reuse preserves symbolic/permutation setup, not old numeric factor state

That sentence should be treated as the direct-solver equivalent of the Epic 4
repeated-run truth.

### What reuse explicitly means

- same-pattern repeated numeric work
- preserved analysis investment
- overwritten factor state on successful refactor
- multiple solves per factorization

### What reuse explicitly does not mean

- incremental-update guarantee on prior triangular numeric data
- backend-specific CSC/native storage persistence contract
- automatic structural-pattern validation beyond the caller precondition
- promise that one-shot mutating factor paths become state-preserving

## One-Shot API Relationship

Day 7 narrowed this to a wording decision. Day 8 should finalize it as:

- one-shot APIs are first-class peer entry points
- for single-run or low-context direct solves, they are also the simple/default
  caller path
- the analysis/factor/refactor lifecycle is the explicit opt-in path for
  stable-pattern repeated direct runs

This lets Sprint 50 say both truths at once without contradiction:

- one-shot APIs remain fully supported
- repeated direct lifecycle is the intended performance-oriented repeated-run
  story

### Final wording recommendation

Use language like:

- “simple/default path” for one-off or occasional solves
- “explicit repeated-run path” for stable-pattern repeated work
- “first-class peer entry points” when describing compatibility/support status

Avoid language like:

- deprecated
- legacy-only
- replacement path
- mandatory migration

## Result and Option Struct Story

Sprint 50 also needs the caller story for structs to be explicit enough that
Sprint 51 implementation can align headers and examples cleanly.

### Analysis and factor objects

Caller expectations:

- lifecycle objects are explicit user-owned structs
- zero-init is valid
- free is explicit
- they can be stack-allocated by callers and passed through the repeated-run
  workflow

### Option structs

Caller expectations should follow the current repo norm:

- designated initializers are the preferred public style
- zero-initialization should preserve safe defaults where already part of the
  public contract
- direct solver option structs remain family-specific where that reflects real
  semantics

### Result/factor state ownership

Caller expectations:

- `sparse_analysis_t` owns symbolic/permutation state
- `sparse_factors_t` owns numeric factor state
- neither object should be described as owning or preserving the source matrix

## Final Coverage Statement

The final Sprint 50 design should explicitly say:

- the direct repeated-run lifecycle is centered on the public
  analysis/factor/refactor path
- it is the intended repeated-run path for LU, Cholesky, and LDL^T
- QR is not part of the first implementation target
- one-shot LU / Cholesky / LDL^T remain supported peer paths

## Caller-Facing Story

The final high-level caller story should be short enough to survive later docs
adoption without drift:

1. for one-off direct solves, keep using the one-shot family APIs
2. for stable-pattern repeated solves, analyze once and factor/refactor many
3. reuse preserves symbolic/permutation setup, not old numeric factor state
4. free analysis and factors explicitly when done

## Important Day 8 Non-Decisions

This final contract still does **not** commit Sprint 50 to:

- new generic direct handles
- public CSC/native factor containers
- structural-pattern verifier redesign
- broad example conversion
- broad README/tutorial rewrite
- exact Sprint 51 code-shape choices

Those remain outside the Day 8 design boundary.

## Highest-Value Day 8 Conclusions

### 1. Zero-init plus explicit free is the final lifecycle baseline

Sprint 50 does not need new init helpers to make the contract coherent.
Zeroed lifecycle objects and safe explicit free are enough for the final design.

### 2. Analyze-once / factor-refactor-many is the final repeated-run direct story

This is no longer just an implementation capability or benchmark pattern. It is
the intended stable-pattern repeated-run public contract.

### 3. The one-shot versus repeated-run relationship is now explicit rather than implied

One-shot family APIs remain:

- first-class
- simple/default for one-off solves

The explicit lifecycle path becomes:

- the opt-in repeated-run path
- the factor-many performance path

### 4. Sprint 50’s remaining design work should now be about fences, not contract shape

The public contract is now concrete enough to drive Sprint 51 implementation.
