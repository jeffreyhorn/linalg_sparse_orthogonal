# Sprint 179 Day 1: Sprint Intake And Evidence Baseline

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Establish the Sprint 179 baseline before deciding whether generated API HTML
should be hosted, retained as a CI artifact, committed, or kept local-only with
stronger enforcement. Day 1 records source-plan authority, prior evidence,
current Doxygen configuration, current generated-output policy, current
navigation wording, and open decision options.

## Source Authority

The active Sprint 179 project-plan section is:

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- section: `Sprint 179: Generated API HTML Publication Decision`

The sprint artifact path is:

- `docs/planning/EPIC_16/SPRINT_179/`

## Starting Snapshot

| Field | Value |
| --- | --- |
| Branch | `sprint-179` |
| Starting commit | `17754f05face0b7a9c82810790c11f430ef2d6c8` |
| Source project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Sprint plan path | `docs/planning/EPIC_16/SPRINT_179/PLAN.md` |
| Working notes path | `docs/planning/EPIC_16/SPRINT_179/WORKING_NOTES.md` |
| Artifact directory | `docs/planning/EPIC_16/SPRINT_179/artifacts/` |

## Recent Prior PR Context

| Commit | Context |
| --- | --- |
| `17754f05` | Merged PR #198 from Sprint 178. |
| `a7d58196` | Completed Sprint 178 allocation-failure proof. |
| `3907e754` | Merged PR #197 from Sprint 177. |
| `4bca0a10` | Addressed PR #197 review comments. |
| `aad776d9` | Moved Sprint 177 planning artifacts to Epic 16. |

## Sprint 179 Scope

Sprint 179 must close generated API HTML product status by selecting and
enforcing exactly one of these options:

| Option | Day 1 status |
| --- | --- |
| Hosted publication | Open for Day 5/Day 6 decision. |
| Retained CI artifact | Open for Day 5/Day 6 decision. |
| Committed generated output | Open for Day 5/Day 6 decision, but conflicts with current `docs/api/` ignore policy unless explicitly selected. |
| Stronger local-only status | Open for Day 5/Day 6 decision and matches the current baseline policy. |

## Prior Evidence Baseline

| Source | Day 1 finding |
| --- | --- |
| Sprint 177 ESM-005 | Generated API HTML is currently local-only evidence: `docs/api/html/` is generated, ignored, untracked, unstaged, and validated by `make api-docs-freshness`. |
| Sprint 177 ESM-011 | Public headers and API reference are maintained through local docs/API checks; no generated HTML hosting, package ABI, or dynamic ABI guarantee is implied. |
| Sprint 177 Gate 2 | Sprint 179 must select one product status and align docs navigation, Doxygen inputs, generated-output behavior, freshness checks, ignored/staged-file policy, and public support wording with that status. |
| Sprint 177 handoff | First actions are to audit Doxygen inputs, output location, ignored paths, staging guard, and API navigation wording before selecting a product status. |
| Sprint 178 closeout | Sprint 178 left no generated API HTML implementation changes and handed off Sprint 179 as the next generated API publication/status decision. |

## Current Doxygen Configuration

| Setting | Current value | Baseline implication |
| --- | --- | --- |
| `INPUT` | `include/` | Generated API coverage is scoped to checked-in public headers. |
| `FILE_PATTERNS` | `*.h` | Header files are the generated API source input. |
| `RECURSIVE` | `NO` | Nested include subdirectories are not part of the current configured input set. |
| `OUTPUT_DIRECTORY` | `docs/api` | Generated output is under the ignored `docs/api/` tree. |
| `HTML_OUTPUT` | `html` | Generated HTML lands under `docs/api/html/`. |
| `GENERATE_HTML` | `YES` | HTML generation is enabled locally. |
| `WARN_IF_UNDOCUMENTED` | `YES` | Missing documentation warnings are enabled. |
| `WARN_IF_DOC_ERROR` | `YES` | Doxygen documentation-error warnings are enabled. |
| `WARN_NO_PARAMDOC` | `YES` | Missing parameter documentation warnings are enabled. |
| `WARN_AS_ERROR` | `NO` | Warnings are visible but not promoted to Doxygen process failures. |

## Current Command And Guard Baseline

| Command or script | Current role |
| --- | --- |
| `make docs` | Runs `doxygen Doxyfile` and writes generated HTML under `docs/api/html/`. |
| `make docs-check` | Runs Doxygen and checks generated page coverage with `scripts/check_api_docs_coverage.py`. |
| `make api-docs-local-only` | Runs `scripts/check_api_docs_local_only.sh`. |
| `make api-docs-validate` | Combines `docs-check` and `api-docs-local-only`. |
| `make api-docs-freshness` | Selected local generated API freshness proof. |
| `scripts/check_api_docs_coverage.py` | Requires generated reference and source pages for checked-in public headers under `include/`; excludes generated `sparse_version.h` from expected Doxygen pages. |
| `scripts/check_api_docs_local_only.sh` | Requires `docs/api`, `docs/api/html`, and `docs/api/html/index.html` to be ignored and rejects tracked, staged, or visible untracked generated API files under `docs/api/`. |

## Current Generated Output Policy

| Evidence | Day 1 result |
| --- | --- |
| `.gitignore` | `docs/api/` is ignored. |
| `git check-ignore -v docs/api docs/api/html docs/api/html/index.html` | All three paths are ignored by `.gitignore:40:docs/api/`. |
| `git ls-files docs/api` | No tracked generated API files. |
| Local generated tree | `docs/api/html/` exists on disk in this checkout as ignored generated output. |

## Current Navigation And Claim Surface

| File | Current generated API wording |
| --- | --- |
| `README.md` | Lists `make docs`, `make docs-check`, `make api-docs-freshness`, and says the API reference entry point is `docs/api_reference.md`. |
| `docs/api_reference.md` | Says checked-in public headers under `include/` are the source of truth; generated HTML is local-only generated output, ignored by the repository, and not hosted or source-controlled. |
| `docs/maintainer_guide.md` | Says `docs/api_reference.md` is the user-facing entry point; `docs/api/html/` is generated Doxygen output kept local-only and ignored; `make api-docs-freshness` refreshes and validates the local view. |

## Protected Non-Claims

Sprint 179 must not imply these claims unless the selected product decision and
implementation explicitly prove them:

- hosted generated API documentation publication;
- source-controlled generated API HTML;
- artifact-published generated API HTML;
- release evidence for generated API HTML;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable runtime or performance guarantees;
- generated API completeness beyond the configured Doxygen input set.

## Day 1 Decisions

- Treat the Epic 16 project plan as Sprint 179 source authority.
- Treat Sprint 177 Gate 2 as the acceptance gate for the sprint.
- Treat the current local-only generated API behavior as baseline, not as the
  final Sprint 179 decision.
- Defer product status selection until the Doxygen audit, guard audit, and
  option matrix are complete.
- Do not edit generated HTML directly.

## Day 1 Deliverables

- `docs/planning/EPIC_16/SPRINT_179/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_179/artifacts/day1-evidence-baseline.md`
- Sprint 179 artifact directory
- generated API claim baseline
- Doxygen and local-only guard status baseline

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 179 scope is tied to the Epic 16 project plan. | Complete | Source authority and scope sections above. |
| Current generated API status is recorded before changes begin. | Complete | Doxygen, command, generated-output, and navigation baselines above. |
| Publication and local-only options remain open pending the audit. | Complete | Option table keeps all four product statuses open for later decision. |
