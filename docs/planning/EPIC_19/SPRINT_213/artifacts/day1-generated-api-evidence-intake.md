# Sprint 213 Day 1: Generated API Evidence Intake

## Summary

Day 1 establishes the Sprint 213 generated API evidence surface before any
publication or local-only implementation changes. The current policy keeps
generated Doxygen HTML under `docs/api/html/` as ignored local output. The
supported source-controlled route is `docs/api_reference.md` plus checked-in
public headers under `include/`, and `make api-docs-freshness` is the current
selected validation command.

Sprint 213 reopens the publication decision deliberately. Local-only, hosted
generated API HTML, retained generated-doc artifacts, and committed generated
HTML remain open options until the Day 5 product policy decision.

## Sprint Scope Mapping

| Item | Day 1 owner interpretation |
| --- | --- |
| 213.1 Publication Option Review | Inventory the current local-only evidence surface and preserve all candidate publication options for later comparison. |
| 213.2 Policy Decision | Defer policy selection until Day 5 after baseline, option review, criteria, and stop-condition evidence. |
| 213.3 Automation Implementation | Defer workflow, retention, routing, staging, or stronger local-only implementation until the selected policy is recorded. |
| 213.4 Routing And Guard Tests | Identify existing API docs coverage, local-only, and routing guard owners for later fixture expansion. |
| 213.5 User And Maintainer Docs | Identify current generated API wording in README, INSTALL, API reference, maintainer guide, and related docs. |
| 213.6 Validation And Closeout | Use documentation-only validation on Day 1; reserve generated API validation commands for baseline and implementation days. |

## Current Evidence Sources

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 213 is a 166-hour sprint to decide whether generated API HTML remains local-only or is published, then implement matching automation. |
| `Doxyfile` | Doxygen reads checked-in `include/*.h` headers and writes HTML under `docs/api/html/`. |
| `.gitignore` | `docs/api/` is ignored, preserving generated HTML as local output; generated `include/sparse_version.h` is ignored separately. |
| `Makefile` | `docs` generates Doxygen output; `docs-check`, `api-docs-local-only`, `api-docs-routing`, and `api-docs-freshness` provide the current validation chain. |
| `scripts/check_api_docs_coverage.py` | Validates generated reference/source pages for checked-in public headers and excludes generated `sparse_version.h`. |
| `scripts/check_api_docs_local_only.sh` | Guards ignored/untracked/unstaged generated output and workflow publication/staging bypasses. |
| `scripts/check_api_docs_routing.py` | Validates source-controlled API routes and rejects generated-output or unsupported hosted API publication links. |
| `tests/test_api_docs_coverage.py` | Regression owner for generated page coverage behavior. |
| `tests/test_api_docs_local_only_guard.py` | Regression owner for local-only, workflow, archive, broad path, and publication semantics. |
| `tests/test_api_docs_routing.py` | Regression owner for Markdown, HTML, reference-link, route, and hosted-link parsing. |
| `docs/api_reference.md` | Source-controlled API reference path; states generated HTML is local-only generated output, not hosted or source-controlled publication. |
| `README.md` | User-facing quick-start route for `make docs-check`, `make api-docs-freshness`, and `docs/api_reference.md`. |
| `INSTALL.md` | Support/readiness matrix classifies local generated API HTML as local-only and excludes hosted publication, retained artifacts, and committed generated HTML. |
| `docs/maintainer_guide.md` | Current maintainer policy owner; names Sprint 204 as the active local-only generated API policy and rejects publication drift without a reopened decision. |

## Current Generated API Boundary

| Boundary field | Current evidence |
| --- | --- |
| Source-controlled entry point | `docs/api_reference.md` |
| Exact declaration owners | Checked-in public headers under `include/` |
| Generated root | `docs/api/` |
| Generated HTML | `docs/api/html/` |
| Doxygen input set | `include/`, `*.h`, non-recursive |
| Validation command | `make api-docs-freshness` |
| Support tier | `local_only` |
| Current publication status | No hosted API publication, retained generated-doc artifact, committed generated HTML, or release evidence |
| Completeness boundary | Checked-in public headers selected by `Doxyfile`; generated install headers remain separate |
| Current policy owner | Sprint 204 stronger local-only generated API policy |

## Initial Publication Option Notes

| Option | Day 1 observation |
| --- | --- |
| Stronger local-only | Lowest publication risk and matches current policy, but may leave external API discoverability weaker. |
| Hosted generated API HTML | Improves discoverability, but requires deployment ordering, freshness proof, hosted URL policy, stale-output handling, and source/generated route clarity. |
| Retained generated-doc artifact | Preserves CI-produced output without a public docs site, but needs artifact retention, naming, and non-release-evidence wording. |
| Committed generated HTML | Makes output reviewable and browsable from the repository, but introduces generated-output review noise, repository growth, and staleness risk. |

## Initial Guard And Test Owners

| Guard owner | Likely Sprint 213 use |
| --- | --- |
| `tests/test_api_docs_coverage.py` | Header/page coverage and generated-header exclusion fixtures. |
| `tests/test_api_docs_local_only_guard.py` | Workflow publication, broad docs path, archive staging, upload/deploy command, dynamic path, and committed-output bypass fixtures. |
| `tests/test_api_docs_routing.py` | Source route, generated-output route, hosted publication route, unrelated external docs, Markdown, and HTML anchor fixtures. |
| `scripts/check_api_docs_coverage.py` | Generated page freshness and checked-in public-header coverage enforcement. |
| `scripts/check_api_docs_local_only.sh` | Local-only or selected publication staging/workflow policy enforcement. |
| `scripts/check_api_docs_routing.py` | Source-controlled API route and publication-link policy enforcement. |

## Day 1 Outcome

Day 1 closes the intake setup. The sprint now has:

- a working-notes ledger for item status, evidence, risks, validation, and
  changed surfaces;
- an initial generated API evidence inventory;
- a current local-only policy boundary table;
- a preserved set of publication options for Day 3-5 comparison;
- a validation plan for later generated API automation, routing, workflow, and
  documentation changes.

Day 2 should capture current local-only baseline behavior with focused command
results and concrete guard coverage gaps before publication options are scored.

## Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

Day 1 validation command:

```sh
git diff --check
```

