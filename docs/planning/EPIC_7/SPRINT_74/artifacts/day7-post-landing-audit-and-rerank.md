# Sprint 74 Day 7: Post-Landing Audit and Rerank

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Audit the Day 6 width-contract landing and rerank the remaining Sprint 74
capability seams from the live post-landing state.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day5-index-scalar-architecture-design.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day6-index-width-integration-batch1.md`
- `include/sparse_types.h`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`

## Day 7 Audit Conclusions

### 1. Day 6 closed the strongest width-first contradiction

The Day 6 landing materially changed the queue:

- the width contract no longer reads like a fixed hand-edited typedef
- the checked `idx_t` <-> `size_t` bridge now has a clearer owner
- the matrix shell no longer reads like the strongest remaining capability
  contradiction

So the next move should not be another same-lane width batch.

### 2. The strongest remaining seam is now the real-only scalar contract

The strongest remaining capability contradiction is now concentrated in the
public real-only callback and result surfaces, centered on:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

with likely implementation support in:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

That is the densest remaining public ceiling because the current contracts still
hard-code dense real buffers through:

- `sparse_precond_fn`
- `sparse_matvec_fn`
- iterative one-shot and block solve argument/result signatures
- `sparse_eigs_t` result buffers

### 3. SVD and broader algorithm breadth remain later, not next

These lanes remain real but are not the strongest next move:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- broader eigensolver-family expansion beyond `sparse_eigs_sym(...)`

Why they stay later:

- SVD is narrower and more family-local than the iterative/eigs callback and
  result contracts
- unsymmetric eigensolver breadth is still product expansion, not the strongest
  current contract contradiction
- reopening the width lane before the scalar lane would widen Sprint 74 for
  less value

### 4. The exact Day 8 target set is now fixed

Required next design center:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Likely implementation center if the design proves it:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Likely proof homes:

- `tests/test_iterative.c`
- `tests/test_eigs.c`

Support only if wording truly forces it:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `README.md`
- `docs/maintainer_guide.md`

Explicitly not next:

- another broad width-contract batch
- another broad matrix-shell batch
- unsymmetric eigensolver expansion
- fake complex-readiness or broad scalar-generic claims

## Exit State

Sprint 74 Day 7 exits with:

1. the Day 6 width-first lane closed as the strongest first contradiction
2. one new strongest remaining seam fixed to the real-only scalar surface
3. one exact Day 8 design center around iterative and eigensolver public
   contracts
4. one explicit non-center list that keeps later width and algorithm-breadth
   work deferred
