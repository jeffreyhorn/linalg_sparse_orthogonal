# Sprint 44 Day 1 Artifact: Scope and Phase-2 Baseline

## Purpose

Capture the Sprint 44 starting baseline before Phase-2 graph extraction,
runtime/orchestration cleanup, and the first large-test maintainability batch
begin.

## Starting Truth

Sprint 44 starts from a stable preserved Sprint 40/41/42/43 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code surfaces already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized because `deadcode*` still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- Sprint 41 already left a reusable internal safety/helper layer:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 42 already left a compatibility-preserving structural-refactor model:
  - internal-first scaffolding
  - shared matrix-state guard adoption
  - focused misuse-regression reinforcement
- Sprint 43 already completed Phase 1 of graph decomposition:
  - `src/sparse_graph_core.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - narrowed residual `src/sparse_graph.c`

This means Sprint 44 is not opening with quality repair, helper redesign, or
public-contract churn. It is opening with residual graph decomposition plus a
bounded large-test maintainability batch on top of a preserved reviewed
baseline and an already-written Epic 4 execution contract.

## Day 1 Workstreams

Sprint 44 Day 1 confirms the sprint's seven bounded workstreams:

1. FM refinement extraction
2. separator lifting extraction
3. runtime strategy parsing cleanup
4. final graph orchestration cleanup
5. large-test helper audit
6. first test-helper consolidation batch
7. validation closeout

These come directly from the Sprint 44 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and are consistent with Sprint 43's
closeout, which explicitly narrowed the residual graph queue to FM refinement,
separator lifting, and deeper runtime simplification.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_44/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_43/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 residual graph inputs

- `src/sparse_graph.c`
- `src/sparse_graph_fm_buckets.h`
- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

### Highest-risk Day 1 large-test maintainability inputs

- `tests/test_chol_csc.c`
- `tests/test_svd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 44 is a bounded graph Phase-2 plus test-maintainability sprint, not a behavior-rewrite sprint

The preserve-not-reopen boundary is explicit:

- keep work internal-first
- preserve current public API stability
- preserve Sprint 40 validation-anchor truth
- preserve Sprint 41 shared-helper rules where generic safety work is needed
- preserve Sprint 42 compatibility-preserving refactor style
- preserve Sprint 43's extracted graph ownership boundaries
- avoid opportunistic algorithm changes or broad documentation churn while
  finishing the graph split and improving large-test structure

### 2. The residual graph queue is explicit before code changes begin

The strongest Sprint 44 graph target set is:

- FM refinement extraction
- separator lifting extraction
- runtime strategy parsing cleanup
- top-level orchestration cleanup after those moves land

This gives the sprint a bounded residual graph target set before the
large-test maintainability work begins.

### 3. The residual graph hotspot is real, but the ownership picture is already better than Sprint 43 Day 1

The live repo now shows:

- `src/sparse_graph.c` = `2153` lines
- extracted supporting files:
  - `src/sparse_graph_core.c` = `264`
  - `src/sparse_graph_coarsen.c` = `597`
  - `src/sparse_graph_bisect.c` = `521`

That means Sprint 44 can start from a narrowed residual orchestration file
rather than a whole-subsystem monolith. The Phase-2 queue is now about the
remaining high-value seams, not about re-finding the module boundaries Sprint
43 already established.

### 4. The large-test maintainability queue is already justified by live size concentration

The strongest inherited test-hotspot set is:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_svd.c` = `3746`
- `tests/test_ldlt_csc.c` = `3637`
- `tests/test_qr.c` = `3291`

That means Sprint 44 can begin helper/fixture consolidation from a real and
already-measured maintainability hotspot cluster rather than needing another
preparatory sprint.

### 5. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and scope confirmation
2. residual graph seam inventory
3. FM boundary design
4. separator/runtime/test design
5. bounded FM and separator extraction batches

That ordering preserves Sprint 40's core rule: structural refactors should be
guided by measured seams and explicit ownership boundaries before code movement
lands.
