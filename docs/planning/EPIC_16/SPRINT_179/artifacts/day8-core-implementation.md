# Sprint 179 Day 8: Core Implementation Batch

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Implement the first strengthened local-only enforcement change after the Day 6
product decision and Day 7 implementation design. Day 8 keeps generated API
HTML local-only and adds checked-in wording enforcement to the existing
local-only guard.

## Code And Documentation Changes

| File | Change | Reason |
| --- | --- | --- |
| `scripts/check_api_docs_local_only.sh` | Added `require_file_contains()` and `check_product_status_wording()`. | The guard now checks both generated-output state and source-controlled product-status wording. |
| `docs/maintainer_guide.md` | Replaced older Sprint 158 policy wording with Sprint 179 product-decision wording. | Maintainer docs now point at the current strengthened local-only decision. |

No generated files under `docs/api/html/` were edited or staged.

## Guard Behavior Added

`api-docs-local-only` now fails if the checked-in docs no longer preserve core
local-only product-status wording:

| Checked file | Required wording intent |
| --- | --- |
| `README.md` | Names selected local Doxygen freshness plus local-only staging guard. |
| `docs/api_reference.md` | Says generated HTML is local-only generated output. |
| `docs/api_reference.md` | Says generated HTML is not hosted or source-controlled. |
| `docs/maintainer_guide.md` | Names the Sprint 179 product decision. |
| `docs/maintainer_guide.md` | Says local generated output is not hosted, artifact-published, or release evidence. |

## Failure Message Contract

The new wording check reports failures as:

```text
api-docs-local-only: FAIL: <path> must state <label> for the strengthened local-only generated API HTML product decision
```

This keeps failures deterministic and tied to the product decision rather than
silently letting navigation drift toward hosted, artifact-published, or
source-controlled claims.

## Generated Output Policy

Day 8 preserves the selected output policy:

| State | Behavior |
| --- | --- |
| `docs/api/` ignored | Required and checked. |
| `docs/api/html/` ignored | Required and checked. |
| `docs/api/html/index.html` ignored | Required and checked. |
| tracked generated API files | Rejected. |
| staged generated API files | Rejected. |
| non-ignored untracked generated API files | Rejected. |
| hosted/artifact/committed generated output | Not implemented and still unsupported. |

## Early Validation Output

Direct guard:

```text
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: README.md local-only freshness wording ok
api-docs-local-only: docs/api_reference.md local-only generated output wording ok
api-docs-local-only: docs/api_reference.md not hosted or source-controlled wording ok
api-docs-local-only: docs/maintainer_guide.md Sprint 179 product decision wording ok
api-docs-local-only: docs/maintainer_guide.md not hosted, artifact-published, or release evidence wording ok
api-docs-local-only: passed
```

Full freshness target:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: README.md local-only freshness wording ok
api-docs-local-only: docs/api_reference.md local-only generated output wording ok
api-docs-local-only: docs/api_reference.md not hosted or source-controlled wording ok
api-docs-local-only: docs/maintainer_guide.md Sprint 179 product decision wording ok
api-docs-local-only: docs/maintainer_guide.md not hosted, artifact-published, or release evidence wording ok
api-docs-local-only: passed
```

## Remaining Implementation Work

Day 9 should reconcile whether additional path assumptions or wording checks
are needed. Day 10 should harden freshness/staging behavior if the Day 8 guard
still leaves a concrete false-positive or bypass risk.

## Day 8 Deliverables

- core implementation changes
- initial test or guard coverage
- generated output policy behavior
- early validation notes
- Day 8 implementation artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected path is implemented at a usable first pass. | Complete | `api-docs-local-only` now checks product-status wording in checked-in docs. |
| Failure behavior is explicit. | Complete | `require_file_contains()` emits deterministic file/label failures. |
| Generated output handling matches the product decision. | Complete | The guard still rejects tracked, staged, or non-ignored generated API files and preserves ignored local output. |
