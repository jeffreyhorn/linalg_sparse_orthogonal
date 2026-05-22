# Sprint 39 Day 8 Artifact: Standards & Maintainer-Doc Audit

## Purpose

Map the final Epic 3 maintainer-standard ownership surfaces so closeout work
can consolidate only the durable guidance that should survive the sprint, while
leaving historical sprint narrative in artifacts.

## Day 8 Bottom Line

The repo does **not** have a missing-standards problem. It has an ownership and
duplication problem.

The main long-term standards already have natural homes:

- `README.md`
  - operator command map
  - reviewed-quality contract
  - dead-code contract
  - cross-platform CI contract
  - readiness checklist
  - test-category policy
- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`
  - warning-clean authority model
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
  - reproducible warning-workflow execution model
- `tests/test_framework.h`
  - executable skip/slow/experimental semantics

## Topic Ownership Map

### Warning cleanliness and warning proof

Authoritative home:

- Sprint 30 playbook + rebuild workflow docs

Why:

- they define the authoritative Apple Clang CMake full-tree model
- they explain the difference between authoritative warning proof and narrower
  local cross-checks

Closeout implication:

- Day 9 should cross-reference these more explicitly from the current
  operator-facing surfaces rather than rewriting them elsewhere

### Reviewed local quality and cross-platform contract

Authoritative home:

- `README.md`

Why:

- this is the operator-facing command map and CI-contract surface
- it already owns `quality-review-full`, dead-code, and platform staging text

Closeout implication:

- keep it concise
- add only the smallest needed maintainer-facing clarifications

### Dead-code workflow contract

Authoritative homes:

- `README.md`
- generated `build/deadcode/report.md`
- `Makefile` operator messages

Why:

- they are the surfaces operators actually run and inspect
- they already expose the staged serialized-execution limit and residual-bucket
  meaning

Closeout implication:

- preserve this split; do not create another parallel dead-code policy file

### Test truthfulness and dormant/historical evidence policy

Authoritative homes:

- `README.md` test-category policy
- `tests/test_framework.h`

Why:

- README states the policy
- `test_framework.h` is the executable implementation of skip/slow/experimental
  semantics

Closeout implication:

- Day 9 should make the dormant/historical-evidence rule more explicit as a
  maintainer expectation, but it should not create a large new test-policy doc

### Designated-initializer guidance

Current durable homes:

- public-facing docs from Sprint 35
- headers/tutorial/README examples

Why:

- the rule is already reflected in shipped public examples
- the residual need is a concise maintainer-facing reminder, not another large
  narrative document

Closeout implication:

- Day 9 should add a short stable expectation rather than promoting Sprint 35’s
  full reasoning into permanent top-level prose

## What Should Stay Sprint-Artifact-Only

Do **not** promote these into permanent top-level standards:

- day-by-day sprint design notes
- old debt triage inventories
- pre-cleanup review narratives
- retrospective rationale that is only historically interesting

Those remain valuable references, but they are not the stable operator or
maintainer contract.

## Day 9 Likely Implementation Shape

The expected standards/documentation closeout batch is narrow:

1. strengthen README cross-references to Sprint 30 warning authority docs
2. add one concise maintainer-facing designated-initializer expectation
3. add one concise maintainer-facing dormant/historical-test expectation
4. avoid broad README restructuring or a new standalone standards document
