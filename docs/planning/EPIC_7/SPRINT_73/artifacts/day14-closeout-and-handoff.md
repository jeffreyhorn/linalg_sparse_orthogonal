# Sprint 73 Day 14: Closeout and Handoff

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Close Sprint 73 from the Day 13 validated baseline with one explicit
configuration-modernization package and a ranked carry-forward queue for
Sprint 74.

## Sprint 73 Package Summary

Sprint 73 now closes as one coherent residual-configuration package across:

- residual env-var inventory and rerank
- FM/graph policy integration
- debug/profile precedence rationalization
- docs/maintainer/header follow-through
- proof-owner alignment
- validated close state

## What Landed

### 1. The strongest FM compatibility lane is now narrower and more coherent

- recognized `SPARSE_FM_*` compatibility env vars now parse once at the graph
  orchestration boundary
- they lower into one internal typed FM policy/runtime contract
- `src/sparse_graph_refine.c` no longer behaves like a second independent FM
  parser

### 2. The strongest developer-only precedence seams now have explicit owners

- `SPARSE_HCC_DEBUG` now resolves through one internal current/override seam
- `SPARSE_ND_PROFILE` now resolves through one internal current/override seam
- both lanes have focused proof in the correct owners

### 3. The maintained policy/doc surfaces now tell the smaller truthful story

- `docs/maintainer_guide.md` now distinguishes:
  - compatibility-first FM policy overrides
  - deferred developer-only debug/profile controls
  - the explicit proof-owner map for the touched lanes
- `include/sparse_analysis.h` remained support-only and already truthful, so
  Sprint 73 did not widen into unnecessary header churn

## Preserved Fence

Sprint 73 preserved the intended Epic 7 boundary:

- no new public typed FM option family
- no new public typed debug/profile option family
- no widened reviewed/platform claim
- no fake “all env-var debt is gone” story
- `SPARSE_QG_PROFILE` stayed explicitly deferred as support-only context

## Validated Close State

Sprint 73 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- reviewed CMake parity `53`
- Makefile/CMake parity `53 vs 53`
- reviewed CMake `ctest` `53 / 53`
- `Total Test time (real) = 421.78 sec`

Representative retained anchors:

- `test_graph`: `bcsstk14 under SPARSE_ND_COARSENING=hcc: sep=97`
- `test_reorder_nd`: `Pres_Poisson ND/AMD = 0.923`
- `test_fuzz`: `large-n CSC lifecycle property: 3/3 passed`
- `bench_amd_qg`: `Kuu: qg 1252.7 ms vs bitset 3195.9 ms`
- install/package regressions: installed `pkg-config` version `2.2.0`

## Ranked Carry-Forward Queue

1. ND compatibility/default-policy convergence beyond the already-landed
   `SPARSE_ND_PROFILE` precedence seam
2. residual support-only control cleanup where `SPARSE_QG_PROFILE` or later
   SVD-routing wording truly moves
3. capability modernization starting with the Sprint 70 index-width fence
4. later backend/performance maturity only where benchmark-governed ownership
   seams actually justify it
5. later permanent-surface cleanup only after higher-value configuration and
   capability lanes move

## Project Plan Recheck

`docs/planning/EPIC_7/PROJECT_PLAN.md` does not need a Sprint 73 correction.
The landed work still matches the sprint contract.

## Exit State

Sprint 73 closes with:

1. a smaller and more coherent residual configuration surface
2. explicit compatibility vs internal-policy vs developer-only ownership
3. named proof owners for the touched precedence and compatibility lanes
4. a validated handoff package for Sprint 74
