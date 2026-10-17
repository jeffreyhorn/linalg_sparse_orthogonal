# Sprint 213 Day 12: Maintainer Documentation And Residuals

## Summary

Day 12 completes maintainer-facing documentation for the Sprint 213 stronger
local-only generated API policy. The branch now documents how to diagnose and
repair generated API validation failures and records which generated API
publication options remain unclaimed.

## Maintainer Guide Changes

| Area | Change |
| --- | --- |
| Repair workflow | Added a generated API local-only repair workflow that starts with `make api-docs-freshness` and then isolates Doxygen/coverage, local-only, or routing failures. |
| Expected artifacts | Added expected repair artifacts: regenerated local `docs/api/html/` output plus passing `api-docs-coverage`, `api-docs-local-only`, and `api-docs-routing`. |
| Residual options | Added a residual list naming hosted HTML, retained generated-doc artifacts, and committed generated HTML as unclaimed future options. |

## Project Plan Change

`docs/planning/EPIC_19/PROJECT_PLAN.md` now records the current Sprint 213
branch direction: stronger local-only generated API closure is selected, while
hosted generated API HTML, retained generated-doc artifacts, and committed
generated HTML remain future options unless exact publication evidence is
selected later.

## Guard Coverage

| Surface | Regression |
| --- | --- |
| Repair workflow marker | `test_missing_maintainer_repair_workflow_text_fails_clearly()` |
| Expected repair artifact marker | `test_missing_maintainer_expected_repair_artifact_text_fails_clearly()` |
| Residual option marker | `test_missing_unclaimed_publication_options_text_fails_clearly()` |

The routing guard enforces these markers through `REQUIRED_TEXT`.

## Preserved Boundaries

Day 12 does not claim:

- hosted API publication;
- retained generated-doc artifacts;
- committed generated HTML;
- release evidence;
- package-manager distribution;
- shared-library or dynamic ABI support;
- broad platform parity;
- completeness beyond checked-in public headers selected by `Doxyfile`.

## Validation

Ran for Day 12 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

`python3 tests/test_api_docs_routing.py` and `make api-docs-freshness` passed.
Final hygiene checks are recorded in the final turn summary.

No `.c` or `.h` files are modified, so the full C quality gate is not required
by the sprint instruction.

## Outcome

Sprint item 213.5 is complete for maintainer and planning documentation.
Maintainers have exact commands and triage surfaces for generated API
local-only failures, and residual status names only the future publication
options that remain unclaimed.
