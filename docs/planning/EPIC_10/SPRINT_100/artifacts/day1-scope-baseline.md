# Sprint 100 Day 1 Scope Baseline

## Purpose

Day 1 opens Sprint 100 by converting the Epic 10 project-plan section into a
bounded execution package. The sprint is a baseline and claim-contract sprint,
not an implementation sprint.

## Day 1 Deliverables

| deliverable | status | location |
|---|---|---|
| Sprint 100 workstream inventory | complete | this artifact |
| working-notes baseline | complete | `../WORKING_NOTES.md` |
| initial artifacts directory | complete | `../artifacts/` |
| authoritative input list | complete | `day1-authoritative-inputs.txt` |
| validation expectation list | complete | this artifact and working notes |

## Sprint 100 Workstreams

| # | workstream | project-plan item | day ownership | expected output |
|---:|---|---|---|---|
| 1 | Baseline quality recheck | Item 1 | Days 2-3 | reviewed quality, build, package, CI, and platform baseline artifacts |
| 2 | State-of-the-art definition | Item 2 | Day 6 | target definition, capability categories, comparison dimensions, disallowed claims |
| 3 | Residual queue conversion | Item 3 | Days 7-8 | residual-to-claim map, sprint dependency model, earned/candidate/non-goal classification |
| 4 | Evidence templates | Item 4 | Days 9-11 | solver, benchmark, coverage, package, ABI, and platform evidence templates |
| 5 | Baseline metrics artifact | Item 5 | Days 4-5 | source/test metrics, comparison lane inventory, benchmark and coverage baseline |
| 6 | Public claim audit | Item 6 | Day 12 | supported/candidate/unsupported public claim table |
| 7 | Sprint closeout | Item 7 | Days 13-14 | integrated handoff, final validation notes, Sprint 101 requirements |

## Landing Order

1. Scope and artifact setup.
2. Reviewed quality, build, package, CI, platform, source, test, comparison,
   benchmark, and coverage baselines.
3. State-of-the-art target definition.
4. Epic 9 residual conversion into Epic 10 claims and non-goals.
5. Evidence template creation for future implementation sprints.
6. Public claim audit.
7. Integrated handoff and closeout.

This order is deliberate. It prevents later sprints from writing public claims
or starting implementation before the evidence contract exists.

## Day-Level Ownership

| day | title | owned scope |
|---:|---|---|
| 1 | Scope Baseline | workstreams, inputs, artifact structure, validation expectations |
| 2 | Quality Baseline | reviewed Make/CMake/install/export/source-list quality baseline |
| 3 | Build Evidence | build, package, CI, and platform proof map |
| 4 | Metrics Baseline | source/test size, large-file, and maintainability metrics |
| 5 | Comparison Baseline | external comparison, benchmark, coverage, and reporting inventory |
| 6 | Target Draft | state-of-the-art definition and disallowed broad claims |
| 7 | Residual Map | Epic 9 residual conversion |
| 8 | Claim Model | Sprint 101-109 claim map and dependency model |
| 9 | Solver Template | solver comparison evidence template |
| 10 | Benchmark Template | benchmark, coverage, and performance sentinel templates |
| 11 | Package Template | packaging, ABI, consumer, and platform-tier templates |
| 12 | Claim Audit | public/support claim audit |
| 13 | Handoff Package | integrated Sprint 100 handoff package |
| 14 | Closeout | validation, artifact index, closeout notes, Sprint 101 handoff |

## Initial Risk Register

| risk | why it matters | Day 1 handling |
|---|---|---|
| overclaiming state-of-the-art status | the review explicitly says the project is not yet state of the art | require a target definition and disallowed claim list before implementation |
| treating non-claims as hidden failures | Epic 9 deliberately preserved non-claims | carry non-claims forward explicitly into Day 7 and Day 8 |
| running implementation before evidence contracts exist | later sprints need consistent templates and claim rules | make Sprint 100 a baseline and contract sprint |
| mixing reviewed and supplemental checks | quality surfaces have different authority levels | capture reviewed/supplemental status during Days 2-5 |
| bloating planning docs without handoff value | Epic 10 needs actionable artifacts, not generic process text | require each day to produce an artifact with a later-sprint consumer |

## Validation Expectations

Day 1 changed planning documentation only:

- `docs/planning/EPIC_10/SPRINT_100/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day1-authoritative-inputs.txt`
- `docs/planning/EPIC_10/SPRINT_100/artifacts/day1-scope-baseline.md`

Required Day 1 validation:

- `git diff --check`
- trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_100`

Not required for Day 1:

- `make format && make lint && make test`, because no `.c` or `.h` files were
  modified
- build-system, workflow, package, benchmark, or script-specific proof, because
  none of those surfaces changed

## Day 1 Exit Criteria

- Sprint 100 work is bounded before collection begins.
- All Sprint 100 project-plan items have day-level ownership.
- Validation expectations are visible in working notes and artifacts.
- Day 2 can start reviewed quality baseline collection without redefining
  Sprint 100 scope.

