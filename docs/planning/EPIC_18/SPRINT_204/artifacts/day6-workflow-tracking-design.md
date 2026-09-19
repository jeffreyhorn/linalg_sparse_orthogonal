# Day 6: Workflow And Tracking Design

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Design workflow, artifact, Pages, ignore, or staging changes needed
by the selected policy.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`

## Purpose

Day 6 converts the Day 5 stronger local-only generated API decision into a
narrow implementation design. It does not make implementation edits. It defines
which files should remain unchanged, which guard surfaces may change, how
generated API paths are owned, and what validation must cover the Day 7-Day 11
work.

## Selected Policy Restatement

Sprint 204 selected stronger local-only generated API policy:

- generated API HTML remains ignored local output under `docs/api/html/`;
- `make api-docs-freshness` remains the supported local Doxygen freshness and
  staging command;
- `docs/api_reference.md` and checked-in public headers under `include/`
  remain the source-controlled API reference path;
- no hosted generated API publication, retained generated API artifact, GitHub
  Pages deployment, or committed generated HTML is selected.

## Implementation Surface Map

| Surface | Day 6 design decision | Day 7-Day 11 implication |
| --- | --- | --- |
| `.gitignore` | Keep unchanged. `docs/api/` remains ignored. | No edit unless a later check proves the ignore rule is ambiguous. |
| `Doxyfile` | Keep unchanged. Doxygen output remains `docs/api`; input remains checked-in public headers under `include/`. | No edit unless Day 8 finds a coverage/freshness blocker. |
| Makefile | Keep target names and composition unchanged by default. | Edit only if guard composition or diagnostics need a clearer target. |
| `.github/workflows/*.yml` | Keep workflows unchanged and free of generated API publication paths. | Do not add upload, deploy, artifact, Pages, or generated API output steps. |
| `scripts/check_api_docs_local_only.sh` | Primary Day 7 implementation candidate. | Strengthen diagnostics and workflow-publication rejection while preserving local-only behavior. |
| `scripts/check_api_docs_coverage.py` | Secondary Day 8 candidate only. | Adjust only for concrete coverage diagnostic gaps. |
| README, INSTALL, `docs/api_reference.md`, maintainer guide | User/maintainer routing surfaces. | Update on Days 10-11 only if Day 9 or guard hardening finds wording gaps. |
| Sprint 204 artifacts | Evidence surface. | Continue updating as days close. |

## Generated API Path Ownership

| Path | Ownership | Required behavior |
| --- | --- | --- |
| `include/*.h` | Source-controlled public API declarations and Doxygen input set. | Checked in; reviewed as source; changes require appropriate docs/API validation and full C gate if headers change. |
| `include/sparse_version.h.in` | Template for generated installed version header. | Install/package validation owns this behavior; not an expected Doxygen generated page under current input scope. |
| `docs/api_reference.md` | Source-controlled API reference index and routing page. | Must state source-of-truth and local generated-output semantics. |
| `docs/api/` | Ignored local generated output root. | Must not be tracked, staged, or visible as non-ignored untracked output. |
| `docs/api/html/` | Ignored local Doxygen HTML view. | Current only after `make api-docs-freshness`; must not be published, uploaded, deployed, or committed under selected policy. |
| `docs/api/html/index.html` | Local generated entry point. | Must remain ignored and local-only. |
| `.github/workflows/*.yml` | CI orchestration surface. | Must not reference generated API output paths while local-only policy is selected. |

## Guard Design

The current local-only guard already checks:

- `docs/api`, `docs/api/html`, and `docs/api/html/index.html` are ignored;
- no generated API files are tracked;
- no generated API files are staged;
- no generated API files are visible as non-ignored untracked files;
- `Doxyfile` keeps expected local-only input/output settings;
- README, API reference, and maintainer guide keep selected local-only wording;
- workflows do not reference `docs/api/html` or `docs/api/`.

Day 7 should harden this guard without widening scope into publication:

| Candidate hardening | Reason | Validation |
| --- | --- | --- |
| Name exact workflow files scanned. | Makes diagnostics clearer if future workflow publication references appear. | Direct guard run. |
| Reject common generated API publication terms near generated API paths if path checks are refactored. | Prevents upload/deploy/Pages semantics from sneaking in under local-only wording. | Direct guard run or focused mutation test if practical. |
| Improve rollback diagnostics for staged/tracked generated output. | Helps maintainers restore local-only state after Doxygen runs. | Direct guard run; optional fixture if added. |
| Keep Doxyfile contract checks explicit. | Prevents accidental input/output drift from looking like a publication decision. | `make api-docs-freshness`. |

Day 7 should not add hosted, artifact, Pages, or committed-output support.

## Workflow Design

Under the selected policy, workflow behavior is intentionally negative:

- no workflow should upload `docs/api/`;
- no workflow should upload `docs/api/html/`;
- no workflow should deploy generated API HTML;
- no workflow should configure GitHub Pages for generated API HTML;
- no workflow should publish generated API metadata as release evidence.

The Day 6 design therefore selects workflow **guarding**, not workflow
publication. If a later day discovers an existing workflow generated API path,
the correct action is to remove or reclassify it, not to treat it as selected
publication.

## Staging And Cleanup Design

Generated output after Doxygen runs should be handled this way:

| State | Required local-only behavior |
| --- | --- |
| Ignored generated files exist under `docs/api/` | Acceptable local state after `make docs-check` or `make api-docs-freshness`. |
| Generated files are tracked | Guard failure; remove from index or reopen committed-output decision. |
| Generated files are staged | Guard failure; unstage before closeout. |
| Generated files are visible as non-ignored untracked files | Guard failure; fix ignore rules or output path. |
| Workflow references generated output | Guard failure unless a future product decision selects hosted/artifact publication and satisfies the Day 4 gate. |

No separate staging directory is selected for Sprint 204. Local Doxygen output
continues to use `docs/api/html/`.

## Validation Mapping

| Planned change | Required validation |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` diagnostics or workflow rejection | `bash -n scripts/check_api_docs_local_only.sh`; direct guard run; `make api-docs-freshness`; `git diff --check`. |
| `scripts/check_api_docs_coverage.py` diagnostics | Direct Python coverage run after Doxygen output exists; `make docs-check`; `make api-docs-freshness`; `git diff --check`. |
| Makefile docs/API target composition | Changed target directly; `make docs-check`; `make api-docs-freshness`; `git diff --check`. |
| README, INSTALL, API reference, or maintainer wording | Relevant docs guard; `make api-docs-freshness` if wording markers are affected; `git diff --check`. |
| Any `.c` or `.h` file | `make format && make lint && make test`. |

## Day 7 Implementation Target

Day 7 should inspect `scripts/check_api_docs_local_only.sh` and implement only
the highest-value local-only hardening that is still missing. The expected
first choice is diagnostic and workflow-publication rejection hardening because
it directly supports the selected stronger local-only policy without changing
publication semantics.

Day 7 should leave these surfaces unchanged unless concrete evidence says
otherwise:

- `.gitignore`;
- `Doxyfile`;
- `.github/workflows/*.yml`;
- generated files under `docs/api/`;
- public headers under `include/`;
- C source files.

## Completion Criteria Review

| Day 6 criterion | Status |
| --- | --- |
| Item 204.2 has a narrow implementation design. | Complete; the design selects local-only guard hardening and rejects publication implementation. |
| Generated output tracking semantics are explicit before file changes. | Complete; path ownership, staging behavior, and cleanup states are defined. |
| Workflow or ignore changes have matching guard coverage planned. | Complete; no workflow or ignore changes are planned, and workflow path rejection remains a required guard behavior. |

## Changed Surfaces

- Added this Day 6 workflow/tracking design artifact.
- Updated `WORKING_NOTES.md` with the Day 6 status, decision-log entry, path
  ownership, and validation mapping.

No source, public header, workflow, Makefile, Doxyfile, user-facing
documentation behavior, or generated-output policy changed on Day 6.
