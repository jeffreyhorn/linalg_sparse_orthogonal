# Sprint 204 Day 11: Maintainer And Claim Boundary Docs

## Purpose

Day 11 aligns maintainer guidance and claim-boundary guards with the selected
stronger local-only generated API policy. The work makes the Day 7-Day 10
guard stack explicit for maintainers and adds regression coverage for the
maintainer markers that protect the policy.

## Maintainer Guidance Updates

| Surface | Update |
| --- | --- |
| `docs/maintainer_guide.md` | Describes `make api-docs-freshness` as Doxygen generation, page coverage/freshness, local-only generated-output validation, and API routing validation. |
| `docs/maintainer_guide.md` | Defines `api-docs-local-only` as the guard for ignored/untracked/unstaged generated output and workflow publication semantics. |
| `docs/maintainer_guide.md` | Defines `api-docs-routing` as the guard for user-facing API routes and generated/hosted API publication link rejection. |
| `docs/maintainer_guide.md` | Adds explicit rejection rules for generated API workflow publication, hosted URLs, committed `docs/api/`, generated HTML release evidence, package-manager evidence, ABI evidence, broad platform evidence, portable performance evidence, and state-of-the-art evidence. |

## Guard Updates

| Surface | Update |
| --- | --- |
| `scripts/check_api_docs_routing.py` | Requires maintainer-guide markers for `api-docs-freshness`, `api-docs-routing`, retained generated-doc artifact non-claims, and routing-guard preservation. |
| `tests/test_api_docs_routing.py` | Adds a maintainer claim-boundary mutation that fails if the routing-guard preservation marker is removed. |

## Non-Claim Marker Inventory

| Non-claim | Maintained status |
| --- | --- |
| Hosted API docs | Not claimed; generated HTML remains local-only. |
| Retained generated-doc artifact | Not claimed; maintainer and INSTALL wording reject it. |
| Committed generated HTML | Not claimed; `docs/api/` remains ignored and guarded. |
| Release evidence | Not claimed; generated HTML is local freshness evidence only. |
| Broad API completeness | Not claimed beyond checked-in public headers selected by `Doxyfile`. |
| Dynamic ABI compatibility | Not claimed from generated API docs. |
| Shared-library support | Not claimed from generated API docs. |
| Package-manager distribution | Not claimed from generated API docs. |
| Broad platform parity | Not claimed from generated API docs. |
| External-library parity | Not claimed from generated API docs. |
| Portable performance | Not claimed from generated API docs. |
| State-of-the-art evidence | Not claimed from generated API docs. |

## Validation

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, confirmed no generated API publication links, and retained source-controlled API routing. |
| `python3 tests/test_api_docs_routing.py` | Passed | Regression suite includes the new maintainer marker mutation. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Local-only generated-output, wording, Doxyfile, and workflow publication checks passed. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Python docs tooling compiled. |
| `make api-docs-freshness` | Passed | Doxygen generation, page coverage/freshness, local-only staging, and routing guards all passed. |
| `git diff --check` | Passed | No whitespace errors. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 204.5 has guard-backed documentation boundaries. | Met. Maintainer markers are required by `check_api_docs_routing.py` and covered by `tests/test_api_docs_routing.py`. |
| Maintainer instructions describe exactly how to validate or reject generated API output. | Met. The maintainer guide names `make api-docs-freshness`, `api-docs-local-only`, and `api-docs-routing`, with reject conditions for publication and overclaim paths. |
| Documentation cannot drift into unsupported publication or completeness claims without guard failures. | Met for the selected local-only generated API surface. The routing/local-only guards reject generated/hosted links, workflow publication semantics, missing local-only wording, and missing maintainer markers. |

## Day 11 Disposition

Item 204.5 is complete. Day 12 should run the integrated validation set and
record the final command evidence before review hardening and closeout.
