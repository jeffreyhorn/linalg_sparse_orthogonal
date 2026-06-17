# Sprint 74 Day 8: Scalar Surface Preparation Design

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Convert the Day 7 rerank into one explicit Day 9 implementation fence for the
strongest remaining real-only scalar contract, without widening Sprint 74 into
fake scalar genericity or broader algorithm-family work.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day7-post-landing-audit-and-rerank.md`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`

## Day 8 Design Conclusions

### 1. The strongest scalar contradiction is concentrated in iterative and eigs public contracts

The strongest remaining capability seam is not the repo's general use of
`double`.

It is the densest caller-facing real-only callback and result layer centered
on:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Those two headers still carry the strongest public scalar ceiling through:

- `sparse_precond_fn`
- `sparse_matvec_fn`
- iterative one-shot and block solve dense RHS/result signatures
- iterative residual-history and progress fields
- `sparse_eigs_opts_t`
- `sparse_eigs_t`

That makes them the best bounded Sprint 74 scalar-preparation center.

### 2. The right next move is contract preparation, not broad scalar genericity

The correct Day 9 goal is not "make iterative and eigs type generic now."

It is:

- prepare the strongest public real-only callback/result seam
- make the current real-only promise read more deliberately and coherently
- reduce later scalar-widening friction without widening today's shipped
  capability claim

So the next batch should favor:

- tighter contract wording on the strongest public real-only seams
- bounded implementation support only where the contract cleanup truly forces
  it
- proof in the maintained iterative/eigs owners only if behavior or public
  compatibility interpretation actually moves

It should explicitly avoid:

- repo-wide scalar abstraction
- fake complex-readiness language
- broad implementation churn across unrelated solver families

### 3. SVD remains support-only for this batch

`include/sparse_svd.h` and `src/sparse_svd.c` remain real-only surfaces, but
they are not the strongest next center because:

- SVD is narrower and more family-local than the iterative/eigs callback and
  result contracts
- its public result carriers are important but not as central to the broad
  caller-facing scalar ceiling
- touching it now would widen Sprint 74 for less value than the denser
  iterative/eigs seam

So SVD remains support-only if wording truly forces it, not a required Day 9
center.

### 4. The exact Day 9 target set is now fixed

Required Day 9 center:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Likely implementation center if the design proves it is needed:

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
- repo-wide scalar-generic conversion
- fake complex-readiness or broader precision-product claims
- unsymmetric eigensolver expansion
- reopening broad matrix-shell or configuration lanes

## Exit State

Sprint 74 Day 8 exits with:

1. one exact scalar-preparation center fixed to iterative and eigensolver
   public contracts
2. one bounded Day 9 implementation lane that stays narrower than full scalar
   genericity
3. one support-only classification for SVD and broader public follow-through
4. one explicit non-goal fence keeping fake capability expansion out of Sprint
   74
