# Sprint 43 Day 1 Artifact: Scope and Graph Baseline

## Purpose

Capture the Sprint 43 starting baseline before graph-subsystem seam inventory,
module-boundary design, and Phase-1 extraction work begins.

## Starting Truth

Sprint 43 starts from a stable preserved Sprint 40/41/42 baseline:

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
- Sprint 40 already identified the top source-level structural hotspot:
  - `src/sparse_graph.c` = `3555` lines
- Sprint 41 already left a reusable internal safety/helper layer:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 42 already left a compatibility-preserving structural-refactor model:
  - internal-first scaffolding
  - shared guard-helper adoption
  - focused regression reinforcement

This means Sprint 43 is not opening with quality repair, helper redesign, or
public-contract churn. It is opening with graph/ND subsystem decomposition on
top of a preserved reviewed baseline and an already-written Epic 4 execution
contract.

## Day 1 Workstreams

Sprint 43 Day 1 confirms the sprint's seven bounded workstreams:

1. graph-module boundary design
2. graph ownership / construction extraction
3. hierarchy / coarsening extraction
4. coarse-bisection extraction
5. build/include cleanup
6. focused graph tests
7. validation closeout

These come directly from the Sprint 43 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and are consistent with Sprint 40's
Day 4 hotspot baseline, which already identified graph decomposition as the
highest-value structural refactor target in `src/`.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_43/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day4-hotspot-allocation-baseline.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 graph seam inputs

- `src/sparse_graph.c`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 43 is a bounded internal decomposition sprint, not a graph-behavior rewrite sprint

The preserve-not-reopen boundary is explicit:

- keep work internal-first
- preserve current public API stability
- preserve Sprint 40 validation and hotspot-baseline truth
- preserve Sprint 41's shared-helper rules where generic safety work is needed
- preserve Sprint 42's compatibility-preserving refactor style
- avoid opportunistic algorithm changes or broad documentation churn while
  splitting files

### 2. The first Phase-1 extraction cluster is explicit before code changes begin

The strongest Sprint 43 Phase-1 extraction cluster is:

- graph construction / ownership
- hierarchy / coarsening
- coarse-level bisection
- shared internal graph declarations and build/include wiring

This gives the sprint a bounded first-wave target set before later FM
refinement and separator-lifting work begins.

### 3. The graph hotspot and graph-focused test surfaces already justify starting implementation

The live repo still matches the structural case for Sprint 43:

- the graph monolith remains the largest source hotspot:
  - `src/sparse_graph.c` = `3555` lines
- the current graph-focused regression surface already exists in four major
  files:
  - `tests/test_graph.c`
  - `tests/test_graph_fm_buckets.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder_amd_qg.c`

That means Sprint 43 can begin extraction work from a real hotspot with a real
existing test base, rather than needing another exploratory pre-sprint.

### 4. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and scope confirmation
2. monolith seam inventory
3. module-boundary design
4. build/include strategy design
5. first bounded extraction batch

That ordering preserves Sprint 40's core rule: structural refactors should be
guided by measured seams and explicit ownership boundaries before code movement
lands.
