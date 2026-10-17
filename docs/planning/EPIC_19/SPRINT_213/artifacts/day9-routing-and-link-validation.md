# Sprint 213 Day 9: Routing And Link Validation

## Summary

Day 9 validates the routing/link behavior for the selected stronger local-only
generated API policy. The day adds encoded HTML-anchor regressions for both
the source-controlled API route and generated-output rejection path.

## Implementation

| Surface | Change | Rationale |
| --- | --- | --- |
| Source-controlled API route | `tests/test_api_docs_routing.py` adds `test_html_entity_encoded_required_route_is_allowed()`. | Proves an HTML anchor with `href="docs&#x2F;api_reference.md"` satisfies the required README API route after entity decoding. |
| Generated-output rejection | `tests/test_api_docs_routing.py` adds `test_html_percent_encoded_href_generated_api_link_fails_clearly()`. | Proves an HTML anchor with `href="docs/%61pi/html/index.html"` is percent-decoded and rejected as generated API output. |

## Route Policy Evidence

| Category | Evidence |
| --- | --- |
| Required source route | README route to `docs/api_reference.md` remains required and can be expressed as Markdown, reference-style Markdown, normalized local path, or HTML anchor. |
| Generated output | Local links resolving to `docs/api/` or `docs/api/html/` remain forbidden after case, backslash, entity, percent, root-relative, and fragment/query normalization. |
| External docs | Unrelated external documentation links remain allowed, including unrelated docs providers and incidental API-like path text. |
| Publication routes | Hosted/project publication URLs, repository release artifacts, Actions artifacts, suites artifacts, and generated API repository paths remain forbidden. |

## Preserved Boundaries

Day 9 does not:

- publish generated Doxygen HTML;
- add generated API hosted URL allowlists;
- add generated-doc artifact retention;
- commit `docs/api/`;
- change workflow publication policy;
- change public C headers or implementation files.

## Validation

Ran for Day 9 closeout:

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

Sprint item 213.4 now covers the main generated-output link and
source-controlled API route categories expected by the Day 9 plan.
