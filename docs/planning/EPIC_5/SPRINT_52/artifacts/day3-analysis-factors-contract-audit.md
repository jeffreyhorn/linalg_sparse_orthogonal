# Sprint 52 Day 3 - analysis/factors contract audit

Date: 2026-06-01
Branch: `sprint-52`

## Scope

Audit the live `sparse_analysis_t` / `sparse_factors_t` repeated-run direct
path against the Sprint 52 Phase 2 goal and reduce the remaining work to named
integration seams.

## Touched surfaces reviewed

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `tests/test_integration.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `examples/example_analysis.c`

## Findings

### 1. The public repeated-run direct contract is already stable enough to keep

The live public contract still centers on:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

It still says the right compatibility things:

- one-shot LU / Cholesky / LDL^T remain first-class peer entry points
- reuse preserves symbolic/permutation setup, not old numeric factor state
- same-pattern structural compatibility is a caller precondition

Interpretation:

- Sprint 52 does not need a new public abstraction
- the public model is stable enough to deepen rather than redesign

### 2. The main Phase 2 problem is still fallback inside the shared numeric path

The live shared `sparse_factor_numeric(...)` implementation still:

- materializes a fresh working copy each time
- reapplies the analysis permutation by building a permuted matrix copy
- delegates into family one-shot implementations

The header still says the symbolic structures are:

- "available for future optimizations"
- "not currently used to bypass internal symbolic work"

Interpretation:

- this is the strongest true Sprint 52 target
- the repeated-run direct path is correct and public, but still not deeply
  integrated

### 3. Cholesky and LDL^T are the cleanest deepening candidates

The shared repeated-run path for:

- `SPARSE_FACTOR_CHOLESKY`
- `SPARSE_FACTOR_LDLT`

already routes through the corresponding public one-shot options entry with:

- `.reorder = SPARSE_REORDER_NONE`

after the analysis path has already materialized the permutation.

Interpretation:

- these two families provide the cleanest shared-path deepening surface
- Sprint 52 should likely start here rather than chasing LU uniformity first

### 4. LU remains the strongest bounded family-specific seam

The shared LU path still:

- uses a fresh permuted working copy
- calls `sparse_lu_factor(...)`
- hardcodes:
  - `SPARSE_PIVOT_PARTIAL`
  - tolerance `1e-12`

Interpretation:

- LU remains the strongest family-specific special case
- Sprint 52 should treat it as a bounded deeper-integration seam rather than
  forcing symmetry for its own sake

### 5. Refactor is still shallow relative to the public repeated-run story

The live `sparse_refactor_numeric(...)` implementation still:

- creates a temporary new factor object
- calls `sparse_factor_numeric(...)`
- swaps the result in on success

It explicitly does not:

- validate structural compatibility
- reuse prior numeric structure
- perform a tighter incremental refresh

Interpretation:

- the Sprint 52 "Refactor Path Tightening" item is real and well-scoped
- the current refactor path is safe but shallow

### 6. Solve-path allocation churn is real but secondary

The live `sparse_factor_solve(...)` path still allocates temporary buffers for:

- permuted RHS storage
- temporary solution storage

Interpretation:

- this is a real repeated-run efficiency seam
- but it is not the strongest Sprint 52 target compared with shared numeric
  fallback and refactor tightening

### 7. The best proof/adoption surfaces are already obvious

The strongest later Sprint 52 proof/adoption surfaces remain:

- `tests/test_integration.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `examples/example_analysis.c`

Interpretation:

- proof and adoption work should stay concentrated here
- no broad docs/example spread is needed to justify the phase-2 story

## Ranked Phase 2 target list

1. reduce avoidable family-local fallback inside `sparse_factor_numeric(...)`
2. tighten `sparse_refactor_numeric(...)` so the refactor story is less shallow
3. deepen Cholesky/LDL^T first because they are the cleanest shared-path cases
4. treat LU as a bounded special-case seam
5. refresh factor-many benchmark proof in `bench_refactor*`
6. expand lifecycle regression proof in `tests/test_integration.c`
7. keep solve-path allocation churn as a secondary seam

## Explicit non-targets

- new public generic direct-handle abstraction
- raw CSC/native storage exposure
- broad factor-container redesign
- structural-pattern verifier redesign
- sweeping tutorial rewrite
- broad example conversion outside the strongest repeated-run surfaces

## Conclusion

Day 3 reduces Sprint 52’s problem to a concrete Phase 2 target:

- the public repeated-run direct contract is stable enough to keep
- the strongest remaining gap is internal fallback and shallow refactor
  behavior
- Cholesky/LDL^T are the cleanest deepening candidates
- LU remains the strongest bounded family-specific seam
- the strongest benchmark/test/example proof surfaces are already clear

That is enough to start the Day 4 numeric-reuse integration batch from a real
audit rather than a generic integration theme.
