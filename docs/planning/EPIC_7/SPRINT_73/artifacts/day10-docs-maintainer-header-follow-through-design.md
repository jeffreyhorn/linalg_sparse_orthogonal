# Sprint 73 Day 10: Docs / Maintainer / Header Follow-Through Design

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Decide the smallest maintained-surface follow-through actually required by the
Day 6 and Day 9 landed configuration work.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day6-fm-graph-policy-integration-batch1.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day9-debug-profile-rationalization-batch.md`
- `docs/maintainer_guide.md`
- `include/sparse_analysis.h`
- `src/sparse_reorder_amd_qg.c`

## Day 10 Design Result

### 1. The live follow-through target is the maintainer policy surface

The only maintained surface that now clearly needs follow-through is:

- `docs/maintainer_guide.md`

Why:

- the live code no longer treats the FM lane as an unstructured deferred env
  surface
- recognized `SPARSE_FM_*` compatibility env vars are now parsed once at the
  graph orchestration boundary
- they lower into one internal typed FM policy/runtime contract
- `src/sparse_graph_refine.c` no longer behaves like a second independent
  configuration parser
- the developer-only/debug lane is now narrower after the Day 9 precedence
  cleanup for `SPARSE_HCC_DEBUG` and `SPARSE_ND_PROFILE`

That means the maintainer guide now needs a more precise residual queue and
ownership explanation.

### 2. `include/sparse_analysis.h` is already truthful

The public analysis header currently says:

- lower-level FM tuning and debug/profile env vars remain internal or
  compatibility-only for now

That remains accurate after the Day 6 and Day 9 landings because:

- Sprint 73 did not add a new public typed FM option family
- Sprint 73 did not add a public typed debug/profile option family
- the FM lane is better integrated internally, but it still is not a caller
  front door
- `SPARSE_ND_PROFILE`, `SPARSE_HCC_DEBUG`, and `SPARSE_QG_PROFILE` still read
  as internal or developer-only surfaces rather than supported typed policy

So `include/sparse_analysis.h` should stay support-only unless Day 11 wording
forces a small consistency edit.

### 3. `SPARSE_QG_PROFILE` stays explicitly deferred

`src/sparse_reorder_amd_qg.c` still carries the support-only profile lane:

- `SPARSE_QG_PROFILE`

Day 10 does not treat that as a required Day 11 touch surface because:

- the Day 9 batch explicitly left it deferred
- the maintained contradiction is in policy wording, not in the quotient-graph
  implementation file
- widening into QG profile commentary now would blur the Sprint 73 boundary

## Exact Day 11 Touch Set

Required:

- `docs/maintainer_guide.md`

Support only if wording truly forces it:

- `include/sparse_analysis.h`

Explicit non-touch set:

- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `src/sparse_analysis.c`
- `src/sparse_svd.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

## Preserved Truthfulness Checklist

Day 11 must preserve:

- no new public typed FM option family
- no public typed debug/profile option family
- `SPARSE_FM_*` compatibility env vars are still compatibility-first, even if
  they now lower through one internal typed policy owner
- `SPARSE_HCC_DEBUG`, `SPARSE_ND_PROFILE`, and `SPARSE_QG_PROFILE` remain
  internal or developer-only surfaces
- `SPARSE_QG_PROFILE` remains a deferred support-only lane
- the Sprint 70 truthfulness fence stays intact:
  - no broadened platform claims
  - no silent product-surface widening
  - no fake configuration-modernization completion story

## Exit State

Sprint 73 Day 10 closes with:

1. one exact Day 11 target: `docs/maintainer_guide.md`
2. one support-only header surface: `include/sparse_analysis.h`
3. the `SPARSE_QG_PROFILE` implementation lane still explicitly deferred
4. a bounded follow-through batch instead of a broad docs/header cleanup spill
