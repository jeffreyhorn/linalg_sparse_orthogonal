# Sprint 213 Day 11: User Documentation Update

## Summary

Day 11 updates user-facing generated API documentation for the selected
stronger local-only policy. The docs now state that durable API documentation
links should use the source-controlled API reference and checked-in public
headers, while generated Doxygen HTML is only an on-demand local view for the
current checkout.

## Documentation Changes

| Path | Change |
| --- | --- |
| `README.md` | Clarifies that CI artifacts, release downloads, Pages deployments, and repository `docs/api/` paths are not the API documentation route. |
| `INSTALL.md` | Clarifies that generated API HTML is an on-demand local view, not an install, release, hosted, or artifact publication surface. |
| `docs/api_reference.md` | Adds durable-link guidance and tells users to regenerate local HTML with `make api-docs-freshness` when they need the generated view. |

## Guard Coverage

| Surface | Regression |
| --- | --- |
| API reference marker | `test_missing_api_reference_durable_route_text_fails_clearly()` |
| README marker | `test_missing_readme_no_artifact_route_text_fails_clearly()` |
| INSTALL marker | `test_missing_install_local_view_text_fails_clearly()` |

The routing guard now requires the new markers through `REQUIRED_TEXT`, so
future edits cannot silently replace source-controlled routes with generated or
published-output routes.

## Preserved Boundaries

Day 11 does not claim:

- hosted API publication;
- retained generated-doc artifacts;
- committed generated HTML;
- release evidence;
- package-manager distribution;
- shared-library or dynamic ABI support;
- broad platform parity;
- completeness beyond checked-in public headers selected by `Doxyfile`.

## Validation

Ran for Day 11 closeout:

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

Sprint item 213.5 is complete for README, INSTALL, and API reference
documentation. Users now have a clearer route: use source-controlled API docs
for durable links and run `make api-docs-freshness` only for local generated
Doxygen output.
