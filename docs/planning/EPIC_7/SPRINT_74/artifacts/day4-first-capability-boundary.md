# Sprint 74 Day 4: First Capability Boundary

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Refine the Day 3 capability ranking and freeze the first bounded Sprint 74
modernization fence before implementation design begins.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day3-capability-ceiling-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day6-capability-modernization-fence.md`
- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

## Day 4 Boundary Conclusions

### 1. The strongest first Sprint 74 fence is the width contract

The Day 4 rerank confirms the best first bounded lane is:

- index-width modernization centered on the public `idx_t` contract and the
  highest-value matrix/product shell size boundary

That lane has the strongest combination of:

- broad user-facing capability payoff
- bounded cleanup scope
- real compatibility-path value
- acceptable first-pass proof and migration risk

The scalar and eigensolver-family ceilings remain real, but they are not the
right first landing because they widen more quickly into public API families,
callbacks, result structs, and much broader proof cost.

### 2. The scalar-preparation seam is support context for later work, not the
first landing

The real-only `double` ceiling remains the strongest second contradiction.

But the rerank shows it should be treated as:

- the strongest second batch
- not the first landing

because:

- the width contract is narrower and easier to make real end-to-end first
- scalar-surface work widens immediately into iterative, eigensolver, and SVD
  public contracts
- the migration and proof burden is much larger than the first width lane

### 3. The first-batch landing surfaces are now explicit

Required first landing:

- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Likely support only if the first landing forces it:

- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

Deferred or explicitly later:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- examples and benchmark surfaces beyond support-only wording follow-through
- package/install workflow changes beyond truthful width-contract wording

### 4. The strongest non-goal fence is now explicit

Sprint 74 Day 4 fixes the first-lane non-goals:

- no repo-wide `int64_t` conversion in one batch
- no scalar-type genericity campaign hidden inside width cleanup
- no fake complex-readiness or broader precision-product claims
- no unsymmetric eigensolver expansion as part of the first lane
- no widened packaging/platform/install claims beyond the actual landed width
  seam

## Exit State

Sprint 74 Day 4 closes with:

1. one explicit first modernization boundary around the width contract
2. one fixed support-only map for proof and maintained-surface follow-through
3. one explicit deferred map for scalar-surface and later algorithm-family
   work
4. one clear starting point for Day 5 implementation design
