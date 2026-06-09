# Sprint 61 Day 1: Scope and Configuration Baseline

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Freeze the Sprint 61 starting point before implementation work begins by
reconfirming the inherited Sprint 60 contract, the preserved reviewed
baseline, the strongest live configuration/env-var seams, and the most
important public/implementation/proof surfaces the sprint will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_61/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_60/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_60/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live configuration/control surfaces measured directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 61 starts from a frozen Epic 6 contract, not from renewed target-definition work

Sprint 60 already froze the productization ranking, state-of-the-art target,
control-placement rule, and validation/platform truthfulness contract. That
means Sprint 61 is not reopening those decisions. It is the first bounded Epic
6 implementation sprint centered on typed configuration surfaces, precedence
rules, and compatibility behavior.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 61 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 61 should not invent a new validation story. It should inherit the
existing one and make later control-plane edits obey it.

### 3. The highest-value Phase 1 configuration problem is concentrated in graph/reorder and analysis-time env-var seams

The live repo now shows a clear concentration:

- strongest Phase 1 control families:
  - `SPARSE_ND_*`
  - `SPARSE_FM_*`
  - `SPARSE_SUPERNODAL_POSTORDER`
- lower-priority adjacent seams:
  - `SPARSE_QG_PROFILE`
  - `SPARSE_SVD_LOWRANK_OUTER`

So the opening Epic 6 implementation batch should not pretend every env-var
surface is equally urgent. The highest-value work is the graph/reorder plus
selected analysis-time control plane.

### 4. Sprint 61 reduces cleanly to seven bounded workstreams

The project-plan scope collapses to:

1. env-var inventory
2. typed option design
3. reorder/ND integration
4. analysis/postorder integration
5. compatibility behavior
6. regression/docs updates
7. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 productization
goal into a smaller implementation contract.

### 5. The strongest live Sprint 61 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- public/control surfaces:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest implementation/control seams:
  - `src/sparse_analysis.c` = `780`
  - `src/sparse_graph.c` = `801`
  - `src/sparse_reorder_nd.c` = `642`
  - `src/sparse_reorder_amd_qg.c` = `611`
- docs/truthfulness surfaces:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `315`
- strongest proof surfaces likely to matter in Phase 1:
  - `tests/test_integration.c` = `1976`
  - `tests/test_graph.c` = `2900`

These are not all immediate edit targets, but they are the real Day 1 map for
where Phase 1 control-plane pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 61 Day 1 confirms the following non-goals before deeper work begins:

- no reopening the repeated-run workflow fence
- no broad backend/AUTO-policy rewrite in the same batch
- no packaging/platform widening disguised as configuration work
- no fake removal of all legacy env-var behavior in Phase 1
- no broad migration of lower-priority debug/profile seams unless they prove
  to block the Phase 1 typed-control landing

## Day 1 Exit State

Sprint 61 now starts from one explicit Phase 1 implementation baseline:

- the Sprint 60 contract is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 “too env-var driven” claim has already narrowed to a ranked
  graph/reorder and analysis-time control seam
- the next step is to inventory and rank those live controls precisely before
  writing the typed-option and precedence contract
