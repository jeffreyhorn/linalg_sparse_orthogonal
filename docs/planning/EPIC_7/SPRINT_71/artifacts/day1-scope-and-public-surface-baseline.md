# Sprint 71 Day 1: Scope and Public-Surface Baseline

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Freeze the Sprint 71 starting point before cleanup work begins by
reconfirming the inherited Sprint 70 architecture contract, the preserved
reviewed baseline, the strongest live public/reference contradiction centers,
and the most important support, proof, and planning surfaces that Sprint 71
will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_71/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_70/RETROSPECTIVE.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `make -n quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
- current live public/reference/support/proof/planning surfaces measured
  directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 71 starts from the Sprint 70 architecture contract, not from another broad Epic 7 review pass

Sprint 70 already froze the starting contract that Sprint 71 is supposed to
obey:

- public-surface cleanup is now ranked ahead of product-model and capability
  implementation work
- the state-of-the-art target and non-goal fence are already explicit
- reviewed, supplemental, and deferred platform/package claims are already
  separated
- the Sprint 71-79 carry-forward queue is already ranked

That means Sprint 71 is not rediscovering the library. It is taking the live
post-Sprint-70 tree plus the explicit carry-forward queue and freezing the
exact cleanup contract for the next sprint.

The opening Sprint 71 work should therefore be treated as:

- public-surface history audit
- front-door and install cleanup
- public header narrative cleanup
- tutorial/example/benchmark support-surface reconciliation
- maintainer-guide re-centering
- truth-surface review and closeout

### 2. The strongest local reviewed baseline remains the authoritative Sprint 71 truth surface

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The Day 1 pass rechecked `make -n quality-review-full`, so Sprint 71 starts
from the same reviewed local truth surface and rerun guidance that Sprint 70
preserved.

### 3. The highest-value Sprint 71 problem is now public/reference drift, not capability or implementation drift

The live repo still has the deeper Epic 7 structural ceilings, but Sprint 71
is intentionally not the sprint that should move them.

The real Day 1 pressure is concentrated in:

1. top-level public front-door cleanup
2. install/release surface cleanup
3. public header narrative cleanup
4. support-surface teaching/proof reconciliation
5. policy-authority recentering
6. truth-surface review before handoff to Sprint 72

Sprint 71 therefore should not widen into feature work. It should sharpen the
exact public/reference and non-goal fence around those six lanes.

### 4. The strongest live Sprint 71 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- all counts below are raw Day 1 `wc -l` newline counts captured from the live
  tree
- editor or GitHub line numbers may read one higher when they count displayed
  lines rather than trailing newlines

- maintained public/reference surfaces:
  - `README.md` = `1034`
  - `INSTALL.md` = `237`
  - `include/sparse_cholesky.h` = `232`
- likely support and policy surfaces:
  - `docs/tutorial.md` = `479`
  - `examples/README.md` = `166`
  - `benchmarks/README.md` = `370`
  - `docs/maintainer_guide.md` = `578`
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest proof-owner surfaces most sensitive to wording drift:
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_integration.c` = `2411`
  - `tests/test_fuzz.c` = `651`
- project-level planning and handoff surfaces:
  - `docs/planning/EPIC_7/PROJECT_PLAN.md` = `356`
  - `docs/planning/EPIC_7/SPRINT_70/RETROSPECTIVE.md` = `252`

These are not all immediate edit targets, but they are the real Day 1 map for
where Sprint 71 cleanup pressure and truth-surface sensitivity now live.

### 5. The current tree still confirms the strongest Sprint 71 contradiction centers from the Sprint 70 queue

The Day 1 pass directly re-confirmed the strongest Sprint 70 carry-forward
centers:

- front-door and install contradiction centers:
  - `README.md`
  - `INSTALL.md`
- strongest header/reference contradiction center:
  - `include/sparse_cholesky.h`
- strongest support-only surfaces unless first-batch cleanup forces them:
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- strongest proof-owner surfaces that Sprint 71 must not let public wording
  steal authority from:
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`

So the Sprint 70 queue remains directionally correct against the live repo.
Day 1 did not uncover a different first-order Sprint 71 problem than the
Sprint 70 contract already named.

## Preserved Day 1 Non-Goal Fence

Sprint 71 Day 1 confirms the following non-goals before deeper work begins:

- no reopening of the Sprint 70 architecture contract
- no implementation or behavior work disguised as public cleanup
- no widening of platform, packaging, benchmark, or capability claims
- no repo-wide chronology cleanup campaign detached from ranked contradiction
  centers
- no public-header cleanup wave wider than the strongest ranked center
- no benchmark, example, or install wording that steals test- or policy-owned
  guarantees

## Day 1 Exit State

Sprint 71 now starts from one explicit cleanup baseline:

- the Sprint 70 architecture contract and carry-forward queue are both active
  and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor remains explicit at `53`
- the broad Sprint 71 goal has already narrowed to front-door/install cleanup,
  public-header cleanup, support-surface reconciliation, maintainer-guide
  recentering, and truth-surface review
- the next step is to recheck the exact docs-only validation and truth-surface
  contract before the deeper public/reference audit begins
