# Sprint 213 Day 14: Closeout Review

## Summary

Sprint 213 closes with stronger local-only generated API closure. Generated
Doxygen HTML remains ignored local output under `docs/api/html/`;
`docs/api_reference.md` and checked-in public headers remain the
source-controlled API route.

The sprint does not publish generated API HTML, retain generated-doc CI
artifacts, or commit generated HTML.

## Final Policy

| Field | Final Sprint 213 policy |
| --- | --- |
| Selected policy | Stronger local-only generated API closure |
| Source-controlled API route | `docs/api_reference.md` and checked-in public headers |
| Generated output root | `docs/api/` |
| Generated HTML path | `docs/api/html/` |
| Freshness command | `make api-docs-freshness` |
| Coverage boundary | Checked-in public headers selected by `Doxyfile` |
| Publication status | No hosted generated API HTML, retained generated-doc artifact, or committed generated HTML |
| Residual status | Future publication requires exact hosting, retention, freshness, routing, rollback, and claim-boundary evidence |

## Item Reconciliation

| Epic item | Final status | Evidence |
| --- | --- | --- |
| 213.1 Publication Option Review | Complete | Days 1-3 inventory the current local-only baseline and compare stronger local-only, hosted Pages, retained artifact, and committed generated HTML options. |
| 213.2 Policy Decision | Complete | Days 4-5 define criteria and select stronger local-only generated API closure. |
| 213.3 Automation Implementation | Complete | Days 7-8 harden routing required-text checks and folded workflow staging/archive detection for the selected policy. |
| 213.4 Routing And Guard Tests | Complete | Days 7-10 add generated-output, hosted/publication, encoded-link, folded-command, quote-aware workflow, and `.yaml` workflow regressions. |
| 213.5 User And Maintainer Docs | Complete | Days 11-12 update README, INSTALL, API reference, maintainer guide, and Epic 19 status/residual wording. |
| 213.6 Validation And Closeout | Complete | Days 13-14 record integrated validation, final status reconciliation, and residual options; post-closeout C test changes also passed the full C gate. |

## Final Evidence

| Surface | Closeout evidence |
| --- | --- |
| Sprint plan | `docs/planning/EPIC_19/SPRINT_213/PLAN.md` |
| Sprint notes | `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` |
| Day artifacts | `docs/planning/EPIC_19/SPRINT_213/artifacts/day1-generated-api-evidence-intake.md` through `day14-closeout-review.md` |
| Local-only guard | `scripts/check_api_docs_local_only.sh` |
| Routing guard | `scripts/check_api_docs_routing.py` |
| Local-only regressions | `tests/test_api_docs_local_only_guard.py` |
| Routing regressions | `tests/test_api_docs_routing.py` |
| User documentation | `README.md`, `INSTALL.md`, `docs/api_reference.md` |
| Maintainer documentation | `docs/maintainer_guide.md` |
| Epic status | `docs/planning/EPIC_19/PROJECT_PLAN.md` |

## Validation

| Command | Closeout result | Purpose |
| --- | --- | --- |
| `make docs-check` | Passed | Regenerates local Doxygen output and checks generated page coverage. |
| `python3 tests/test_api_docs_coverage.py` | Passed | Standalone checked-in public-header coverage regression suite. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Local-only workflow, staging, publication, archive, and generated-output regression suite. |
| `python3 tests/test_api_docs_routing.py` | Passed | Source-controlled route, generated/hosted publication link, encoded-link, and docs-marker regression suite. |
| `make api-docs-freshness` | Passed | Aggregate generated API freshness, local-only, and routing validation. |
| `git diff --check` | Passed | Whitespace validation. |
| `make format && make lint && make test` | Passed after closeout | Validates the two C test changes added in the PR CI fix. |

At the Day 14 snapshot, no C/header files had changed. The final PR later
changed `tests/test_ldlt.c` and `tests/test_reorder_nd.c`; the full C quality
gate passed for those changes in the post-closeout CI fix (`a5e4692e`).

## Residual Risks And Non-Claims

Sprint 213 does not claim:

- hosted generated API HTML;
- retained generated-doc CI artifacts;
- committed generated HTML;
- generated API release evidence;
- generated API package-manager evidence;
- broad API completeness beyond checked-in public headers selected by
  `Doxyfile`;
- package, ABI, shared-library, or runtime-loader support;
- broad platform support;
- portable performance;
- external-library parity;
- state-of-the-art sparse linear algebra status.

Future generated API publication still requires exact hosting, retention,
freshness, routing, rollback, access-control, stale-output, documentation, and
claim-boundary evidence before changing the local-only policy.

## Retrospective Handoff

The retrospective should treat Sprint 213 as a completed policy-closure sprint.
The selected outcome is stronger local-only generated API closure, not
publication. The main implementation surfaces are workflow/staging guard
hardening, routing/link guard hardening, user documentation, maintainer repair
guidance, and Epic 19 residual alignment.

## Outcome

Sprint 213 is ready for retrospective preparation. The branch closes the
generated API publication decision with stronger local-only automation,
documentation, and validation guards without overstating current generated API
publication support.
