# Day 5: Product Decision

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Choose the Sprint 204 generated API policy and freeze the
implementation boundary.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`

## Decision

Sprint 204 selects **stronger local-only generated API policy**.

Generated API HTML remains local generated output under `docs/api/html/`,
ignored by the repository, refreshed with `make api-docs-freshness`, and
interpreted only as a checkout-local rendered view of the configured Doxygen
input set.

The source-controlled API reference path remains:

- `docs/api_reference.md`;
- checked-in public headers under `include/`;
- install/package validation surfaces for generated installed headers such as
  `sparse_version.h`.

## Rationale

The selected policy is the only Day 3 option that fully fits the Day 4
acceptance gate without adding new durable publication infrastructure.

| Input | Finding |
| --- | --- |
| Day 2 baseline | `make docs-check` and `make api-docs-freshness` passed; 18 checked-in public headers produced 18 generated reference pages and 18 generated source pages; `docs/api/` remained ignored with no tracked, staged, or visible non-ignored generated files. |
| Day 3 option matrix | Stronger local-only had the lowest implementation and maintenance cost, highest reviewability, and strongest claim fit. |
| Day 4 acceptance gate | Hosted, retained-artifact, and committed-output paths require additional metadata, link, workflow, retention, drift, or review controls before they can be responsibly claimed. |
| Prior decisions | Sprint 158, Sprint 179, and Sprint 186 already made and preserved local-only generated API semantics, with `R186-HOSTED-API` kept open only for a future product decision that funds publication infrastructure. |

This decision improves Sprint 204 closure by hardening the existing supported
path instead of adding a weak publication path that would create new ambiguity.

## Rejected Or Deferred Paths

| Path | Disposition | Reason |
| --- | --- | --- |
| Hosted generated API HTML | Deferred | High discoverability value, but not selected because it requires a named hosting surface, deployment workflow, freshness metadata, link validation, workflow/publication guard coverage, access semantics, and rollback ownership. |
| Retained CI artifact | Deferred | Useful for reviewer and maintainer evidence, but not selected because artifact name, retention, branch/event scope, upload behavior, metadata, and guard coverage are not yet owned. |
| Committed generated output | Rejected for Sprint 204 | Conflicts with the ignored-output policy, increases generated-diff review burden, and requires drift detection, cleanup rules, and partial-staging enforcement to avoid stale source-controlled HTML. |

Deferred hosted and retained-artifact paths remain legitimate future work only
if a later sprint explicitly reopens the product decision and implements the
Day 4 acceptance gate for that path.

## Selected Implementation Boundary

Days 6-11 may change only the surfaces needed to strengthen local-only
generated API policy:

| Surface | Allowed Day 6-Day 11 purpose |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | Strengthen local-only ignore, staging, workflow-publication, wording, rollback, or diagnostic checks. |
| `scripts/check_api_docs_coverage.py` | Clarify generated-page coverage diagnostics only if needed for the selected local-only policy. |
| Makefile docs/API targets | Adjust guard composition or naming only if needed to keep `make api-docs-freshness` authoritative. |
| Focused docs/guard tests | Add regression coverage for local-only policy, workflow path rejection, or routing checks if useful. |
| README | Clarify user-facing generated API route and local-only non-claims. |
| INSTALL | Keep support/readiness matrix aligned with local-only generated API policy. |
| `docs/api_reference.md` | Keep source-of-truth and generated-output semantics precise. |
| `docs/maintainer_guide.md` | Document local-only ownership, validation, rollback, and future-publication residuals. |
| Sprint 204 artifacts | Record decisions, validation, residuals, and closeout evidence. |

## Explicitly Disallowed Changes

The selected policy does not authorize:

- hosted generated API publication;
- generated API artifact upload or retention workflow;
- GitHub Pages or any other generated API deployment path;
- committing generated files under `docs/api/`;
- broadening Doxygen input beyond checked-in public headers under `include/`;
- treating generated API HTML as release evidence;
- treating generated API HTML as package-manager or ABI proof;
- solver behavior changes;
- unrelated `.c` or public `.h` behavior changes.

If later work needs any disallowed change, the Day 5 decision must be reopened
and the relevant Day 4 acceptance gate must be satisfied before implementation.

## Day 6-Day 11 Implementation Plan

| Day | Planned focus under selected policy |
| --- | --- |
| Day 6 | Design stronger local-only guard and tracking changes. Decide whether current string-based workflow path rejection needs structured workflow inventory support. |
| Day 7 | Implement selected local-only guard or staging diagnostics with minimal surface area. |
| Day 8 | Strengthen freshness and coverage diagnostics only where the selected policy needs clearer failure modes. |
| Day 9 | Validate API routing and links for source-controlled docs plus local Doxygen generation instructions. |
| Day 10 | Update user-facing API docs if wording or routing gaps remain. |
| Day 11 | Update maintainer guidance and claim-boundary guards for the selected local-only policy. |

## Required Closeout Validation

Minimum validation for the selected path:

```bash
make docs-check
make api-docs-freshness
git diff --check
```

Additional required validation depends on later edits:

| Later edit | Required validation |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | `bash -n scripts/check_api_docs_local_only.sh`; direct guard run; `make api-docs-freshness`. |
| `scripts/check_api_docs_coverage.py` | Direct Python coverage run after Doxygen output exists; `make docs-check`; `make api-docs-freshness`. |
| Makefile docs/API targets | The changed target plus `make docs-check` and `make api-docs-freshness`. |
| Workflow-path guard coverage | Focused workflow guard test or direct guard run demonstrating generated API workflow publication remains rejected. |
| User-facing docs only | `make docs-check` and `make api-docs-freshness` if guard wording is affected; otherwise `git diff --check` plus any relevant docs guard. |
| Any `.c` or `.h` edit | `make format && make lint && make test`. |

## Claim Boundary

The selected policy supports only this claim:

Generated API HTML is a local-only rendered Doxygen view for the checked-in
public headers selected by `Doxyfile`; it is current only for a checkout where
`make api-docs-freshness` has just passed, and generated output remains
ignored, untracked, unstaged, and unpublished.

The selected policy does not claim:

- broad API completeness beyond the configured Doxygen input set;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- Homebrew/core, bottles, Linuxbrew, public tap, vcpkg, Conan, pkgsrc, or
  system package support;
- broad platform parity;
- Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable performance;
- release evidence;
- hosted generated API publication;
- retained generated API artifacts;
- committed generated HTML;
- state-of-the-art status.

## Completion Criteria Review

| Day 5 criterion | Status |
| --- | --- |
| Item 204.1 is complete with one selected policy and documented rationale. | Complete; stronger local-only generated API policy is selected above. |
| Unselected publication paths remain blocked unless a future sprint reopens them. | Complete; hosted, retained-artifact, and committed-output paths are deferred or rejected and require the Day 4 gate if reopened. |
| Implementation can proceed without ambiguity about supported claims. | Complete; allowed surfaces, disallowed changes, validation, and claim boundaries are explicit. |

## Changed Surfaces

- Added this Day 5 product-decision artifact.
- Updated `WORKING_NOTES.md` with the Day 5 decision, implementation boundary,
  open-question cleanup, and validation notes.

No source, public header, workflow, Makefile, Doxyfile, user-facing
documentation behavior, or generated-output policy changed on Day 5.
