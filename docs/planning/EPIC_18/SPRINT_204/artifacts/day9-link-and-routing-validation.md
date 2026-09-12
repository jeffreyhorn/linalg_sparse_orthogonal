# Sprint 204 Day 9: Link And Routing Validation

## Purpose

Day 9 finishes item 204.3 by adding link and routing validation appropriate to
the selected stronger local-only generated API policy. The guard validates the
source-controlled API entry points and rejects links that would imply generated
HTML is hosted, committed, retained, or otherwise published.

## Implemented Changes

| Surface | Change | Reason |
| --- | --- | --- |
| `scripts/check_api_docs_routing.py` | Added a focused API routing guard for README, INSTALL, `docs/api_reference.md`, and `docs/maintainer_guide.md`. | Preserve local-only generated API semantics while validating the supported source-controlled routes. |
| `tests/test_api_docs_routing.py` | Added current-tree and fixture regression coverage for required routes, missing route targets, forbidden generated/hosted API links, and missing local-only wording. | Makes route and publication-boundary failures executable. |
| `Makefile` | Added `api-docs-routing` and included it in `api-docs-validate`. | Ensures `make api-docs-freshness` runs Doxygen generation, coverage, local-only staging, and route validation together. |

## Routing Contract

The Day 9 guard freezes these selected local-only routes:

| Document | Required API routing behavior |
| --- | --- |
| README | Routes users to `docs/api_reference.md`, `include/`, and `INSTALL.md#support-readiness-matrix`; retains generated API local-only wording. |
| INSTALL | Keeps the support/readiness matrix row for local generated API HTML and the no-hosted/no-broad-completeness boundary. |
| `docs/api_reference.md` | Routes to checked-in public headers, `Doxyfile`, support/readiness status, tutorial, cookbook, solver selection, and maintainer interpretation. |
| `docs/maintainer_guide.md` | Keeps maintainer interpretation for `docs/api_reference.md`, `docs/api/html/`, and local-only generated output. |

The guard rejects Markdown links to:

- `docs/api/` or generated HTML under `docs/api/html/`;
- hosted API, Doxygen, Pages, or GitHub Pages-style publication URLs;
- missing required API route targets.

## Regression Coverage

| Test | Covered behavior |
| --- | --- |
| `test_current_tree_passes_routing_guard` | The real repository satisfies the selected API routing contract. |
| `test_fixture_passes_routing_guard` | A minimal copied fixture satisfies the same contract. |
| `test_missing_api_reference_route_fails_clearly` | Removing the README API reference Markdown route fails clearly. |
| `test_missing_route_target_fails_clearly` | Removing a required route target fails clearly. |
| `test_generated_html_publication_link_fails_clearly` | Linking to generated HTML under `docs/api/html/` fails as unsupported publication. |
| `test_hosted_api_publication_link_fails_clearly` | Adding a hosted API publication URL fails as unsupported publication. |
| `test_missing_local_only_text_fails_clearly` | Removing local-only routing text fails clearly. |

## Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, confirmed no generated API publication links, and reported `docs/api_reference.md` as the source-controlled API entry point. |
| `python3 tests/test_api_docs_routing.py` | Passed | Regression suite completed without failures. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py` | Passed | New routing script and tests compile. |
| `make api-docs-freshness` | Passed | Doxygen generation, coverage, local-only guard, and API routing guard all passed. |
| `git diff --check` | Passed | No whitespace errors. |

## Boundary Review

Day 9 does not publish generated API HTML. It adds route validation only:

- no hosted API docs URL was added;
- no generated API HTML link was added;
- no retained workflow artifact or Pages deployment was added;
- no files under `docs/api/` were made trackable;
- no Doxygen input beyond checked-in public headers was added;
- no API completeness, ABI, package-manager, release, platform-parity, or
  state-of-the-art claim was added.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 204.3 has link and routing coverage appropriate to the selected policy. | Met. `api-docs-routing` validates the selected source-controlled routes and local-only generated-output wording. |
| User docs do not point to unavailable generated API locations. | Met. Links to `docs/api/`, generated HTML, hosted API docs, Doxygen, Pages, or GitHub Pages-style API publication fail. |
| Generated output is described as rendered documentation, not the API source of truth. | Met. The guard requires local-only wording and keeps `docs/api_reference.md` plus checked-in public headers as the source-controlled entry point. |

## Day 9 Disposition

Item 204.3 is complete. Day 10 can now focus on user-facing wording cleanup or
confirmation without needing to invent a new publication route.
