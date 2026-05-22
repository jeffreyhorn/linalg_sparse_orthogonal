# Sprint 39 Day 9 Artifact: Standards & Documentation Closeout Batch 1

## Purpose

Land the narrow standards/documentation closeout batch identified on Day 8:
consolidate the durable Epic 3 maintainer expectations into the existing
top-level contract without creating a new standalone standards document.

## Shipped Batch

Touched surface:

- `README.md`

Changes shipped:

1. Added a compact `Maintainer Standards` subsection to the README.
2. Pointed warning-clean authority directly at the Sprint 30 authoritative
   sources:
   - `COMPILE_HYGIENE_PLAYBOOK.md`
   - `REBUILD_WORKFLOW.md`
3. Added a concise maintainer-facing designated-initializer expectation for
   public non-default option examples.
4. Added a concise maintainer-facing dormant/historical-test expectation that
   keeps retired evidence in `docs/planning/` artifacts instead of dormant
   active-suite scaffold.
5. Linked non-default test semantics back to `tests/test_framework.h` as the
   executable truth for:
   - `RUN_TEST_SLOW(...)`
   - `RUN_TEST_EXPERIMENTAL(...)`
   - `SKIP_TEST(...)`

## Why This Was The Right Batch

Day 8 showed the repo did not need another top-level standards document. It
needed:

- clearer cross-references to the existing authoritative homes
- one concise maintainer-facing anchor in the README
- less inference across multiple sprint artifacts

This batch does that without turning the README into another long-form Epic 3
history file.

## Validation

Focused doc-surface validation:

- `rg -n "Maintainer Standards|Compile Hygiene Playbook|Rebuild Workflow|designated initializers|historical measurements|RUN_TEST_SLOW|RUN_TEST_EXPERIMENTAL|SKIP_TEST" README.md`
- `sed -n '796,835p' README.md`

## Residual Standards Queue

After Day 9, the residual standards/documentation queue is minimal:

- preserve these references and expectations in the final Sprint 39 summary and
  closeout docs
- avoid duplicating sprint-artifact narrative into permanent repo-level docs

There is no current evidence that another permanent standards file is needed.
