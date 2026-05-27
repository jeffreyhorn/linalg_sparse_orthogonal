# Sprint 42 Day 2 Artifact: Lifecycle Seam Refresh Inventory

## Purpose

Refresh the Sprint 40 lifecycle inventory against the current live headers,
implementations, and user-facing docs so Sprint 42's first implementation
batches are grounded in the actual remaining hidden-mutation and
matrix-eligibility seams.

## Authoritative Inputs

Primary inputs used for this refresh:

- `docs/planning/EPIC_4/SPRINT_42/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day5-lifecycle-inventory-lu-cholesky-ldlt.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day6-lifecycle-inventory-qr-svd-analysis-iterative-eigs.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day8-lifecycle-contract-map.md`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `README.md`
- `docs/tutorial.md`

## Refreshed Seam Summary

### 1. LU and Cholesky remain the strongest hidden-lifecycle overloads

The current live contract still makes LU and Cholesky the two clearest
matrix-as-factor-handle outliers:

- factorization mutates the same `SparseMatrix` that later becomes the solve
  handle
- the matrix still owns factor-state telemetry and solve-readiness state
- permutation state remains matrix-local
- cancellation can leave the matrix non-original before the first callback
  returns

This means the most important remaining lifecycle ambiguity is still not
“separate handles with strict preconditions.” It is “one object carrying too
many roles over time.”

### 2. LDLT and analysis already expose handles, but they still carry bridge-state debt

The live code still confirms the bridge shape:

- LDLT already separates input matrix from factor object:
  - `const SparseMatrix *A`
  - `sparse_ldlt_t`
- `sparse_analyze()` + `sparse_factor_numeric()` + `sparse_factor_solve()`
  already expose explicit analysis/factor handles
- but `sparse_factors_t` still wraps a matrix-centric payload:
  - `SparseMatrix *F`
  - plus LDLT-specific side arrays

This puts LDLT and the analyze-once workflow in the bridge class rather than
the direct mutation class.

### 3. QR and SVD are structurally cleaner, but still rely on strict caller discipline

QR and SVD remain read-only on the input matrix and store result state in:

- `sparse_qr_t`
- `sparse_svd_t`

Their remaining lifecycle pressure is not hidden mutation. It is repeated
strict eligibility rules:

- identity permutations required
- original/unfactored matrix view required in practice
- cancellation is clean, but usability still depends on callers remembering the
  matrix-state rules

These are therefore prime targets for shared matrix-state guard helpers rather
than early payload-separation work.

## Immediate Classification Map

| Seam family | Matrix mutation boundary | Factor/result payload ownership | Cancellation risk | Public compatibility exposure | Day 2 class |
|---|---|---|---|---|---|
| LU | In-place on `SparseMatrix` | Matrix owns factor payload and solve state | High | High | Immediate internal-handle target |
| Cholesky | In-place on `SparseMatrix` | Matrix owns factor payload and solve state | High | High | Immediate internal-handle target |
| LDLT | Input preserved | `sparse_ldlt_t` owns factor payload | Low on input matrix | Moderate | Bridge / guard adoption target |
| Analyze-once workflow | Input preserved, explicit handles | `sparse_analysis_t` and `sparse_factors_t`, but `sparse_factors_t` still wraps matrix-centric payload | Mixed by delegated factor path | Moderate / high | Bridge normalization target |
| QR | Input preserved | `sparse_qr_t` | Low | Moderate | Guard-helper adoption target |
| SVD | Input preserved | `sparse_svd_t` | Low | Moderate | Guard-helper adoption target |

## What Needs New Internal Boundaries vs Shared Preconditions

### Needs new internal object / payload boundaries

- LU internal numeric payload ownership
- Cholesky internal numeric payload ownership
- `sparse_factors_t` payload normalization as a preserve-and-evolve bridge

### Primarily needs shared lifecycle/precondition helpers

- original-state required checks
- identity-permutation required checks
- factored-state required checks
- compatibility-consistent error returns across:
  - LDLT
  - analysis
  - QR
  - SVD

### Should remain mostly local in Sprint 42

- algorithm-specific symbolic logic
- detailed numerical workspace composition
- broader public-handle API design beyond compatibility scaffolding

## Immediate vs Later Separation

### Immediate Sprint 42 landing seams

1. LU internal ownership seam
2. Cholesky internal ownership seam
3. shared state-guard helper layer
4. bounded guard adoption in LDLT / analysis / QR / SVD
5. initial `sparse_factors_t` bridge normalization

### Later lifecycle-phase seams

- broader public explicit-handle rollout
- README/tutorial/header reconciliation for user-visible lifecycle changes
- wider workspace/context API enrichment
- larger subsystem decomposition work unrelated to the first lifecycle seams

## Landing Order For Days 3-10

The refreshed landing order is:

1. internal handle scaffolding design for LU / Cholesky / `sparse_factors_t`
2. shared guard-helper design for original-state / identity-permutation /
   factored-state checks
3. first LU/Cholesky ownership-boundary implementation batch
4. shared guard-helper implementation
5. normalized adoption in LDLT / analysis / QR / SVD paths
6. bounded cancellation / mutation-contract cleanup
7. focused misuse / cancellation tests

## Day 2 Conclusions

1. The current lifecycle queue still reduces cleanly to two major classes:
   - hidden mutable lifecycle overloading
   - explicit handles with strict eligibility burden
2. LU and Cholesky remain the strongest immediate internal-handle insertion
   targets.
3. `sparse_factors_t` remains the most important bridge object for
   compatibility-preserving normalization.
4. QR and SVD are already structurally handle-oriented enough that Sprint 42
   should treat them mainly as shared-guard adoption targets.
5. Sprint 42 now has a concrete implementation landing order rooted in the
   live code and header contract rather than only in Sprint 40's design model.
