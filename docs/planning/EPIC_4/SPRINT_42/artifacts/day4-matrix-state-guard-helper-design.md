# Sprint 42 Day 4 Artifact: Matrix-State Guard Helper Design

## Purpose

Define the shared internal helper layer for lifecycle-sensitive matrix-state
validation so Sprint 42 can stop carrying repeated bespoke checks for
factored-state rejection, identity-permutation requirements, and
original-matrix eligibility.

## Design Inputs

This design is derived from:

- `docs/planning/EPIC_4/SPRINT_42/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day2-lifecycle-seam-refresh-inventory.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day3-internal-handle-scaffolding-design.md`
- current duplicated guard implementations in:
  - `src/sparse_analysis.c`
  - `src/sparse_ic.c`
  - `src/sparse_ilu.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`

## Design Goals

The Sprint 42 guard-helper layer should:

1. centralize the repeated lifecycle-state checks that already agree
   semantically across the codebase
2. preserve current public-facing error semantics
3. remain small enough to avoid hiding algorithm-specific validation
4. support both:
   - matrix-mutating factor builders
   - already handle-oriented result/factor families
5. reduce wording and implementation drift without changing the public
   lifecycle contract

## Scope of the Shared Guard Layer

The shared layer should cover only the repeated lifecycle-state seam:

### Shared lifecycle-state checks

- matrix must not already be factored
- matrix must have identity row permutations
- matrix must have identity column permutations
- matrix must be in the original/original-eligible state
- matrix must already be factored where a solve path requires it

### Out of scope for the shared layer

These checks should remain local:

- symmetry / SPD validation
- square vs rectangular shape validation
- reorder-enum and option validation
- rank/numerical-threshold logic
- algorithm-specific structural assumptions
- larger cancellation cleanup behavior

## Proposed Private Helper Surface

Sprint 42 should add a small private helper seam in `src/`, likely following
the same private-layer pattern as Sprint 41:

- private lifecycle/guard header
- optional small private source file only if a source-backed helper adds value

### Candidate helper responsibilities

#### 1. Identity permutation predicate

Purpose:

- answer whether a matrix still has identity row and column permutations

Value:

- replaces repeated handwritten loops across QR/SVD/analysis/ILU/IC/LDLT

#### 2. Original-state validator

Purpose:

- implement the common lifecycle contract:
  - matrix must not be factored
  - matrix must have identity row/column permutations

Value:

- gives one compatibility-consistent way to express the “original matrix view
  required” rule

#### 3. Factored-state validator

Purpose:

- centralize the common “solve requires already-factored matrix” gate for the
  matrix-mutating factor families

Value:

- reduces repeated direct `mat->factored` checks and keeps solve-entry guard
  semantics aligned

#### 4. Optional family-oriented wrapper helpers

Purpose:

- provide narrow wrappers if needed to keep call sites readable in families
  that need a specific state gate repeatedly

Value:

- improves readability without widening the helper layer into a general
  validation framework

## Shared vs Local Validation Boundary

### Shared checks

- original-state required
- identity permutations required
- factored-state required

### Local checks

- LU / Cholesky numerical/singularity/SPD logic
- LDLT pivoting/symmetry logic
- QR/SVD dimensional and option validation
- analysis-mode factor-type / reorder-mode interpretation
- ILU/IC matrix-class assumptions

## Adoption Matrix

### Primary initial adoption set

| Family | Shared guard use |
|---|---|
| LU | factored-state checks on solve side; original-state helpers where wrapper paths benefit |
| Cholesky | factored-state checks on solve side; original-state helpers where factor-entry wrappers benefit |
| LDLT | original-state / identity-permutation helper |
| Analysis | original-state / identity-permutation helper |
| QR | original-state / identity-permutation helper |
| SVD | original-state / identity-permutation helper |

### Near-adjacent follow-ons once proven

| Family | Likely later use |
|---|---|
| ILU / ILUT / IC | original-state / identity-permutation helper |
| CSC backend entry guards | original-state helper where current entry checks mirror the same logic |

## Error-Semantics Preservation

The helper layer should preserve current outward behavior:

- lifecycle-state violations still return `SPARSE_ERR_BADARG`
- shape/class/numerical errors remain separate and local
- original-state requirements remain semantic contract, even when implemented
  through factored/permutation checks internally

This means the helper layer is an implementation normalization seam, not a new
public policy layer.

## Relationship To Day 3 Handle Design

Day 3 and Day 4 solve different lifecycle problems:

- Day 3 handle scaffolding:
  - ownership ambiguity
  - hidden matrix-as-factor-handle overloading
- Day 4 guard layer:
  - repeated eligibility checks
  - lifecycle wording/implementation drift

Sprint 42 should keep them separate:

- handle seams own payload separation
- guard seams own matrix-state validation normalization

## Day 6 Implementation Guidance

The first implementation batch should:

1. add the small private guard seam
2. migrate one local `has_identity_perms()` implementation into it
3. adopt it in the highest-value lifecycle-sensitive families first:
   - analysis
   - LDLT
   - QR
   - SVD
4. expand into ILU / IC only if the batch remains bounded

The implementation batch should not:

- fold in unrelated numerical validation
- rewrite public headers broadly
- mix handle-payload work and guard-helper work into one abstraction

## Day 4 Conclusions

1. The shared lifecycle validation seam is small and clearly bounded.
2. The right abstraction is a private matrix-state guard layer, not a broad
   generic validation framework.
3. The main shared contracts are:
   - original-state required
   - identity permutations required
   - factored-state required
4. Algorithm-specific checks should remain local to their current families.
5. Day 6 now has a concrete helper-layer target and adoption matrix aligned to
   the Day 2 seam inventory and Day 3 handle design.
