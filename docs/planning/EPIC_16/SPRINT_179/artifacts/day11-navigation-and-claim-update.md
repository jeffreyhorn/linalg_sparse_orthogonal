# Sprint 179 Day 11: Navigation And Claim Update

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Align public and maintainer documentation with the Sprint 179 strengthened
local-only generated API HTML decision. Day 11 makes the supported API
documentation path explicit while preserving non-claims for hosted,
artifact-published, source-controlled, and release generated HTML.

## Documentation Updates

| File | Update |
| --- | --- |
| `README.md` | Adds a short API documentation paragraph after the command list, naming `docs/api_reference.md` and public headers as the supported source-controlled path. |
| `README.md` | States that generated Doxygen HTML is a local-only convenience view and not hosted documentation, a retained CI artifact, source-controlled output, or release evidence. |
| `docs/api_reference.md` | Adds a Sprint 179 product-decision note that keeps generated HTML local-only rather than hosted, artifact-published, or committed. |
| `docs/api_reference.md` | Reaffirms that the API reference page and public headers are the source-controlled API reference path. |
| `docs/maintainer_guide.md` | Adds generated API support-tier wording: `local_only`. |
| `docs/maintainer_guide.md` | Replaces historical Sprint 158 policy wording with the Sprint 179 product-decision wording. |

## Supported Navigation Path

After Day 11, users and maintainers should follow this path:

1. Use README, tutorial, cookbook, and solver-selection docs for workflow
   selection.
2. Use `docs/api_reference.md` as the compact API reference index.
3. Use checked-in public headers under `include/` for exact declarations and
   call-site contracts.
4. Run `make api-docs-freshness` before inspecting local generated Doxygen HTML
   under `docs/api/html/`.

## Claim Alignment

| Claim | Day 11 status |
| --- | --- |
| Maintained local generated API HTML path | Supported after `make api-docs-freshness`. |
| Source-controlled API reference path | Supported through `docs/api_reference.md` and public headers. |
| Generated API support tier | `local_only`. |
| Hosted generated API documentation | Unsupported. |
| Retained CI generated API artifact | Unsupported. |
| Source-controlled generated API HTML | Unsupported. |
| Generated API release evidence | Unsupported. |
| Generated API completeness beyond configured Doxygen inputs | Unsupported. |

## Guard Interaction

The Day 8-Day 10 strengthened local-only guard now protects the Day 11 wording:

| Guard check | Day 11 result |
| --- | --- |
| README local-only freshness wording | Pass |
| API reference local-only generated output wording | Pass |
| API reference not hosted/source-controlled wording | Pass |
| Maintainer guide Sprint 179 product-decision wording | Pass |
| Maintainer guide not hosted/artifact-published/release evidence wording | Pass |
| Workflow generated API output path references | Pass: none found |

## Support-Tier Interpretation

Generated API HTML is `local_only` evidence. That means:

- it is generated and validated in a maintainer checkout;
- it is current only immediately after `make api-docs-freshness` passes;
- it is not a hosted CI proof;
- it is not an artifact-retained publication;
- it is not part of a release bundle;
- it does not promote broader platform, package, ABI, external-library, or
  state-of-the-art claims.

## Validation Evidence

Day 11 validation used:

```bash
bash -n scripts/check_api_docs_local_only.sh
bash scripts/check_api_docs_local_only.sh
make api-docs-freshness
git diff --check
```

The direct guard passed after the documentation updates and reported all
local-only wording checks as ok.

## Day 11 Deliverables

- README navigation update
- API reference update
- maintainer guide update
- support-tier wording update
- Day 11 navigation artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Users can find the supported API documentation path. | Complete | README and API reference name `docs/api_reference.md`, public headers, and `make api-docs-freshness`. |
| Maintainers can reproduce or enforce the selected behavior. | Complete | Maintainer guide names the Sprint 179 local-only decision and `local_only` support tier; guard checks enforce wording. |
| Docs claims stay within validated evidence. | Complete | Unsupported hosted/artifact/source-controlled/release generated HTML claims are explicitly rejected. |
