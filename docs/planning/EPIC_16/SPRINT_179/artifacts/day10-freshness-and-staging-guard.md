# Sprint 179 Day 10: Freshness And Staging Guard

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Complete freshness and staging guard hardening for the strengthened local-only
generated API HTML decision. Day 10 closes the remaining configuration drift
gap by requiring Doxygen to keep generating local HTML from the expected
public-header input set into the ignored `docs/api/html/` tree.

## Guard Changes

| File | Change |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | Added `require_doxyfile_setting()` for fail-closed Doxyfile contract checks. |
| `scripts/check_api_docs_local_only.sh` | Added `check_doxyfile_contract()` and wired it into the local-only guard before tracked/staged output checks. |

## Doxyfile Contract Checks

The local-only guard now requires:

| Setting | Required value | Reason |
| --- | --- | --- |
| `INPUT` | `include/` | Generated API HTML remains derived from checked-in public headers. |
| `FILE_PATTERNS` | `*.h` | Generated API input remains the public header set. |
| `RECURSIVE` | `NO` | Current Sprint 179 scope remains top-level checked-in public headers. |
| `OUTPUT_DIRECTORY` | `docs/api` | Generated output remains under the ignored local tree. |
| `GENERATE_HTML` | `YES` | Local HTML remains generated for inspection. |
| `HTML_OUTPUT` | `html` | Generated HTML remains under `docs/api/html/`. |

If a future product decision changes generated API status, these checks should
be updated with that decision. Until then, configuration drift fails the
local-only guard.

## Existing Guard Coverage Preserved

Day 10 preserves the Day 8/Day 9 checks:

| Guard area | Current behavior |
| --- | --- |
| ignored output paths | Requires `docs/api`, `docs/api/html`, and `docs/api/html/index.html` to be ignored. |
| tracked output | Rejects tracked generated API files under `docs/api`. |
| staged output | Rejects staged generated API files under `docs/api`. |
| non-ignored output | Rejects visible non-ignored generated API files under `docs/api`. |
| product-status wording | Requires local-only wording in README, API reference, and maintainer guide. |
| workflow publication paths | Rejects `.github/workflows` references to `docs/api/html` and `docs/api/`. |

## Missing Page And Stale Output Ownership

| Requirement | Owner |
| --- | --- |
| Regenerate output before checking pages | `make docs-check` through `make api-docs-freshness`. |
| Require generated index | `scripts/check_api_docs_coverage.py`. |
| Require reference/source pages for checked-in public headers | `scripts/check_api_docs_coverage.py`. |
| Enforce Doxygen configured input/output contract | `scripts/check_api_docs_local_only.sh`. |
| Enforce local-only output status | `scripts/check_api_docs_local_only.sh`. |

This split keeps page coverage in the Python coverage checker and local-only
product-status enforcement in the shell guard.

## Validation Evidence

Direct guard output included:

```text
api-docs-local-only: Doxyfile INPUT local-only contract ok
api-docs-local-only: Doxyfile FILE_PATTERNS local-only contract ok
api-docs-local-only: Doxyfile RECURSIVE local-only contract ok
api-docs-local-only: Doxyfile OUTPUT_DIRECTORY local-only contract ok
api-docs-local-only: Doxyfile GENERATE_HTML local-only contract ok
api-docs-local-only: Doxyfile HTML_OUTPUT local-only contract ok
api-docs-local-only: passed
```

Full freshness output included:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
api-docs-local-only: Doxyfile INPUT local-only contract ok
api-docs-local-only: Doxyfile FILE_PATTERNS local-only contract ok
api-docs-local-only: Doxyfile RECURSIVE local-only contract ok
api-docs-local-only: Doxyfile OUTPUT_DIRECTORY local-only contract ok
api-docs-local-only: Doxyfile GENERATE_HTML local-only contract ok
api-docs-local-only: Doxyfile HTML_OUTPUT local-only contract ok
api-docs-local-only: passed
```

## Remaining Risks

| Risk | Day 10 disposition |
| --- | --- |
| Doxygen warnings are visible but not fatal through `WARN_AS_ERROR`. | Accepted for Sprint 179 because Day 3 observed no warning output; future warning-policy work should be a separate decision. |
| Workflow scan remains string-based. | Accepted because generated API publication is rejected; any future publication workflow should replace this with a structured guard. |
| No hosted or artifact metadata exists. | Intentional under strengthened local-only status. |
| Generated HTML can be stale between local command runs. | Documented limitation; supported freshness exists only immediately after `make api-docs-freshness`. |

## Day 10 Deliverables

- stale-output guard
- missing-page guard
- staged-file guard
- publication metadata guard
- Day 10 freshness and staging artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Required generated API artifacts cannot silently drift. | Complete | Coverage script requires index/reference/source pages; local-only guard now requires matching Doxyfile input/output contract. |
| Staged generated files follow the selected product policy. | Complete | Guard rejects staged generated API files under `docs/api`. |
| Guard failures are actionable. | Complete | Doxyfile, wording, workflow, ignore, tracked, staged, and non-ignored checks each produce scoped failure messages. |
