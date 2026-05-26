# Sprint 41 Day 1 Artifact: Scope and Safety Baseline

## Purpose

Capture the Sprint 41 starting baseline before helper-pattern inventory and
shared internal utility work begins.

## Starting Truth

Sprint 41 starts from a stable preserved Epic 3/Sprint 40 baseline:

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
- Sprint 40 already left a concrete architecture contract:
  - lifecycle/state taxonomy
  - lifecycle contract map
  - handle-model migration strategy
  - quality-truth ownership map
  - public migration-risk audit
  - validation anchor

This means Sprint 41 is not opening with quality-baseline repair work. It is
opening with internal helper/safety consolidation on top of a stable reviewed
baseline and an explicit architecture contract.

## Day 1 Workstreams

Sprint 41 Day 1 confirms the sprint's seven bounded workstreams:

1. helper-pattern inventory
2. shared utility design
3. first core-module migration
4. broader `src/` migration
5. auxiliary-surface alignment
6. prep-rule documentation
7. validation closeout

These come directly from the Sprint 41 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and from the Epic 4 remediation plan's
allocation/overflow-helper consolidation queue.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_41/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day14-architecture-contract-synthesis.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day11-quality-contract-ownership-map.md`
- `docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### First helper-consolidation hotspot inputs

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 41 is an internal consolidation sprint, not a public-contract sprint

The preserve-not-reopen boundary is now explicit:

- keep work internal-first
- preserve Sprint 40 ownership and validation contracts
- avoid early public lifecycle/API churn
- avoid opportunistic cross-platform, dead-code-topology, or reviewed-wrapper
  changes

### 2. The first migration cluster is explicit before code changes begin

The first helper-consolidation cluster is:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

This gives the sprint a bounded first-wave target set before broader `src/`
migration work starts.

### 3. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and scope confirmation
2. helper-pattern inventory
3. shared utility design
4. first bounded implementation batch

That ordering preserves Sprint 40's core rule: architecture and ownership
truth should guide implementation, not be inferred after the fact.
