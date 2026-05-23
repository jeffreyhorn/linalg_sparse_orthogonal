# Sprint 40 Day 1 Artifact: Baseline Setup

## Purpose

Capture the Sprint 40 starting baseline before lifecycle inventory and
architecture-definition work begins.

## Starting Truth

Sprint 40 starts from a stable post-Epic-3 quality baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity is still explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- dead-code reporting/checking is already a maintained workflow surface:
  - `make deadcode-report`
  - `make deadcode-check`
- cross-platform reviewed/dead-code limits are already documented rather than
  implicit

This means Sprint 40 is not starting by rebuilding missing quality
infrastructure. It is starting by defining the architecture and inventory
contract for Epic 4.

## Day 1 Workstreams

Sprint 40 Day 1 confirms the sprint’s seven bounded workstreams:

1. baseline capture
2. lifecycle inventory
3. state-model taxonomy
4. future handle-model design
5. quality-contract ownership map
6. public migration-risk audit
7. validation anchor

These come directly from the Sprint 40 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and match the remediation plan rather
than inventing new scope.

## Highest-Value Authoritative Inputs

### Epic 4 planning inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_40/PLAN.md`
- `docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
- `docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`

### Inherited baseline/contract inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `build/deadcode/report.md`
- `build/deadcode/report.tsv`
- `build/deadcode/coverage-notes.txt`
- `build/quality-review-cmake/` CTest registration state

## Highest-Value Day 1 Conclusions

### 1. Sprint 40 is an audit-and-architecture sprint, not a repair sprint

The inherited Epic 3 baseline remains live and documented. That shifts Sprint
40’s job toward architecture definition and risk inventory rather than quality
system repair.

### 2. The core structural risk themes remain exactly the ones from the Epic 4 review

Day 1 did not surface a new top-priority risk class. The leading themes remain:

- hidden mutable matrix/factor lifecycle state
- monolithic subsystem hotspots
- fragmented allocation/overflow helper patterns
- duplicated quality-contract ownership
- weaker benchmark/tooling interface consistency than the library surface

### 3. The first Sprint 40 audit order is already clear

The front half of the sprint should proceed in this order:

1. baseline and scope confirmation
2. hotspot/allocation-density inventory
3. lifecycle inventory
4. state-model taxonomy
5. handle-model design

That preserves the key Sprint 40 rule: no implementation-heavy refactor should
begin before the lifecycle and ownership contracts are explicit.
