# Sprint 213 Day 7: Automation Implementation Batch One

## Summary

Day 7 implements the first guard hardening batch for the Sprint 213 stronger
local-only generated API policy. The implementation preserves `docs/api/` as
ignored local generated output and adds no hosted generated API route, retained
artifact upload, committed generated HTML, or publication allowlist.

## Implementation

| Surface | Change | Rationale |
| --- | --- | --- |
| Routing required text | `scripts/check_api_docs_routing.py` now requires maintainer-guide wording that future generated API publication must reopen the product decision and validate the selected publication policy. | Keeps future hosted HTML, retained CI artifacts, and committed generated output from becoming implicit claims. |
| Routing tests | `tests/test_api_docs_routing.py` adds a GitHub suites artifact URL regression. | Covers the retained-artifact URL shape already rejected by policy and implementation. |
| Routing tests | `tests/test_api_docs_routing.py` adds a missing reopening-criteria marker regression. | Proves the maintainer guide cannot lose the future-publication decision boundary silently. |

## Preserved Boundaries

Day 7 does not:

- add hosted generated API HTML;
- retain generated API docs as a CI artifact;
- commit generated Doxygen HTML;
- add workflow upload or deploy steps for `docs/api/`;
- create an exception or allowlist for generated API publication;
- change public C headers or implementation files.

## Regression Coverage

| Regression | Expected failure |
| --- | --- |
| `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/suites/123/artifacts/456` in user-facing docs | `api-docs-routing` reports an unsupported generated or hosted API publication target. |
| Maintainer guide removes `reopen the product decision and validate the selected publication policy` | `api-docs-routing` reports missing required local-only API routing text. |

## Validation

Ran for Day 7 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

All commands passed. The C/header diff check returned no files.

No `.c` or `.h` files are modified, so the full C quality gate is not required
by the sprint instruction.

## Outcome

Sprint items 213.3 and 213.4 now have their first implementation batch: a
stronger required-text contract for future generated API publication reopening
and a missing retained-artifact URL fixture.
