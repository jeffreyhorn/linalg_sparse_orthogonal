# Sprint 204 Day 10: User-Facing API Docs Update

## Purpose

Day 10 updates user-facing generated API wording to match the selected
stronger local-only policy and the implemented guard stack. This is a
documentation alignment pass only; it does not publish generated HTML or
change public API behavior.

## Updated User-Facing Surfaces

| Surface | Update |
| --- | --- |
| README | Describes `make api-docs-freshness` as selected local Doxygen freshness plus local-only staging and routing guard. The API docs paragraph now states that the command checks generated page coverage, local-only staging, and API routing. |
| INSTALL | Updates the `Local generated API HTML` support/readiness row to name coverage/local-only/routing scripts and exclude hosted API publication, retained generated-doc artifacts, committed generated HTML, and broad API completeness. |
| `docs/api_reference.md` | Clarifies generated page coverage/freshness, local-only staging enforcement, API routing validation, and rejection of generated HTML or hosted API publication links. |
| `scripts/check_api_docs_routing.py` | Updates the guarded INSTALL wording marker so the user-facing support row cannot drift from the Day 10 claim boundary. |

## User-Facing Policy Vocabulary

The public docs now use one consistent generated API policy:

- source-controlled API truth: `docs/api_reference.md` and checked-in public
  headers under `include/`;
- local rendered view: generated Doxygen HTML under ignored `docs/api/html/`;
- freshness command: `make api-docs-freshness`;
- proof surface: Doxygen generation, page coverage/freshness, local-only
  staging, and API routing validation;
- unsupported paths: hosted API docs, retained generated-doc artifact,
  committed generated HTML, release evidence, and broad completeness beyond
  checked-in public headers selected by `Doxyfile`.

## Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, confirmed no generated API publication links, and reported `docs/api_reference.md` as the source-controlled API entry point. |
| `python3 tests/test_api_docs_routing.py` | Passed | Regression suite accepted the updated user-facing wording. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Local-only wording and no-publication workflow checks remain satisfied. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Python docs tooling compiled. |
| `make api-docs-freshness` | Passed | Doxygen generation, coverage/freshness, local-only staging, and routing guards all passed. |
| `git diff --check` | Passed | No whitespace errors. |

## Non-Changes

- No hosted generated API publication was added.
- No retained generated-doc artifact was added.
- No committed generated HTML was added.
- No workflow, Pages, `.gitignore`, or `Doxyfile` changes were made.
- No public headers or C sources changed.
- No broad API completeness, ABI, package-manager, platform-parity,
  performance, release, or state-of-the-art claim was added.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 204.4 is implemented for user-facing docs. | Met. README, INSTALL, and `docs/api_reference.md` now describe the selected local-only policy and guard stack. |
| Public docs use one consistent generated API policy vocabulary. | Met. The docs consistently use source-controlled API entry point, checked-in public headers, ignored local generated HTML, and `make api-docs-freshness`. |
| No user-facing doc implies unsupported ABI, package, hosted, or completeness guarantees. | Met. The Day 10 wording explicitly excludes those claims and the routing/local-only guards passed. |

## Day 10 Disposition

Item 204.4 is complete. Day 11 should align maintainer guidance and
claim-boundary guard coverage with the same local-only vocabulary.
