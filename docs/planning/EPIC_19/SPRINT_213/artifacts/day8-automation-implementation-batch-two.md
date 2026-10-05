# Sprint 213 Day 8: Automation Implementation Batch Two

## Summary

Day 8 completes the selected stronger local-only automation batch for workflow
staging and archive detection. The implementation closes a folded multiline
workflow-command gap while preserving the no-publication decision from Day 5.

## Implementation

| Surface | Change | Rationale |
| --- | --- | --- |
| Local-only workflow guard | `scripts/check_api_docs_local_only.sh` now checks staging and archive command patterns against flattened workflow text as well as line-normalized text. | A YAML folded `run: >` block can split `tar`, `cp`, or `docs/` across lines; the guard now sees the rendered command shape. |
| Local-only tests | `tests/test_api_docs_local_only_guard.py` adds a folded `cp -R docs artifact/` upload regression. | Proves staged docs cannot be uploaded through multiline shell syntax. |
| Local-only tests | `tests/test_api_docs_local_only_guard.py` adds a folded `tar -czf artifact.tgz docs/` upload regression. | Proves archived docs cannot be uploaded through multiline shell syntax. |

## Preserved Boundaries

Day 8 does not:

- add a generated API upload or deployment workflow;
- allow a retained generated-doc artifact;
- commit generated Doxygen HTML;
- add a hosted generated API URL;
- weaken the ignored `docs/api/` staging contract;
- change public C headers or implementation files.

## Regression Coverage

| Regression | Expected failure |
| --- | --- |
| Folded YAML command stages `docs/` with `cp -R` before `actions/upload-artifact` | `api-docs-local-only` reports docs staging for publication. |
| Folded YAML command archives `docs/` with `tar` before `actions/upload-artifact` | `api-docs-local-only` reports docs archiving for publication. |

## Validation

Ran for Day 8 closeout:

```sh
python3 tests/test_api_docs_local_only_guard.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

`python3 tests/test_api_docs_local_only_guard.py` and `make
api-docs-freshness` passed. Final hygiene checks are recorded in the final turn
summary.

No `.c` or `.h` files are modified, so the full C quality gate is not required
by the sprint instruction.

## Outcome

Sprint item 213.3 is functionally complete for the selected local-only
automation path. Sprint item 213.4 has additional workflow staging/archive
coverage for one of the remaining bypass classes identified in the Day 6
fixture plan.
