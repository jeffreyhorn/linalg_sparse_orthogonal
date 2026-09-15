# Sprint 204 Day 8: Freshness And Coverage Checks

## Purpose

Day 8 strengthens generated API freshness and generated-page coverage for the
selected stronger local-only generated API policy. The goal is to make missing
or stale local Doxygen output fail clearly without converting generated HTML
into hosted, retained-artifact, release, or committed evidence.

## Reviewed Surfaces

| Surface | Day 8 finding |
| --- | --- |
| `scripts/check_api_docs_coverage.py` | Already checked for generated index, reference pages, source pages, and the checked-in public-header input set. It did not reject stale generated pages after a public header changed. |
| `scripts/check_api_docs_local_only.sh` | Day 7 already covers ignore state, tracking/staging absence, Doxyfile output policy, local-only wording, and workflow publication/path absence. |
| `Makefile` | `docs-check` and `api-docs-freshness` already run Doxygen before coverage; `api-docs-coverage` had no direct regression owner. |
| `Doxyfile` | Still correctly limits generated API input to checked-in public headers under `include/` and output to `docs/api/html/`. No change required. |

## Implemented Changes

| Surface | Change | Diagnostic outcome |
| --- | --- | --- |
| `scripts/check_api_docs_coverage.py` | Added stale-page detection by comparing each checked-in public header mtime with its generated reference and source page mtimes. | A stale reference or source page fails with the specific header, generated page path, and `rerun make docs-check` guidance. |
| `tests/test_api_docs_coverage.py` | Added fixture regressions for complete coverage, missing HTML directory, missing index, missing reference page, missing source page, stale reference page, and stale source page. | Missing and stale generated-output failures are executable and reviewable. |
| `Makefile` | Added `python3 tests/test_api_docs_coverage.py` to `api-docs-coverage`. | `make docs-check` and `make api-docs-freshness` now run the coverage checker and its regression suite. |

## Regression Coverage

| Test | Covered behavior |
| --- | --- |
| `test_complete_fixture_passes_with_checked_in_headers_only` | A complete fixture reports coverage for checked-in `.h` files only and ignores `sparse_version.h.in`. |
| `test_missing_html_directory_fails_clearly` | Missing `docs/api/html` fails with the generated HTML directory diagnostic. |
| `test_missing_index_fails_clearly` | Missing `index.html` fails with the generated API index diagnostic. |
| `test_missing_reference_page_identifies_header` | Missing reference page reports the owning checked-in public header. |
| `test_missing_source_page_identifies_header` | Missing source page reports the owning checked-in public header. |
| `test_stale_reference_page_identifies_header` | Stale reference page reports the owning checked-in public header and generated page. |
| `test_stale_source_page_identifies_header` | Stale source page reports the owning checked-in public header and generated page. |

## Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 tests/test_api_docs_coverage.py` | Passed | Coverage diagnostic regression suite completed successfully. |
| `python3 -m py_compile scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Modified Python docs tooling compiled. |
| `make api-docs-freshness` | Passed | Doxygen regenerated local HTML; coverage reported 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages; local-only guard passed. |
| `git diff --check` | Passed | No whitespace errors. |

## Policy Boundary

The Day 8 changes strengthen local freshness and coverage only:

- no generated API HTML is committed;
- no generated API HTML is uploaded as a workflow artifact;
- no Pages or hosted documentation deployment is added;
- no Doxygen input beyond checked-in public headers under `include/` is added;
- no API, ABI, package-manager, release, or broad completeness claim is added;
- `docs/api/` remains ignored local generated output.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 204.3 has concrete freshness or coverage enforcement. | Met for freshness/coverage. Stale generated pages now fail, and fixture diagnostics cover missing and stale output. |
| Missing or stale generated API output fails clearly. | Met. Missing directory/index/page and stale reference/source pages identify the failing path and header. |
| Checks remain aligned with the chosen policy rather than implying broader API completeness. | Met. The check remains scoped to checked-in public headers under `include/` and local generated HTML only. |

## Day 8 Disposition

The freshness and generated-page coverage portion of item 204.3 is complete.
Day 9 should finish item 204.3 by validating local-only API routing and links
from user-facing documentation.
