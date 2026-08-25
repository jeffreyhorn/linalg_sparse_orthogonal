# Sprint 179 Day 4: Current Guard And CI Audit

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Audit the current generated API docs targets, scripts, tests, and workflow
evidence paths before selecting a product status. Day 4 makes the current guard
coverage explicit and records CI artifact/publication gaps that must be closed
if hosted or retained generated API HTML is selected later.

## Docs Target Inventory

| Target | Owner | Behavior | Day 4 finding |
| --- | --- | --- | --- |
| `make docs` | `Makefile` | Runs `doxygen Doxyfile` and writes HTML to `docs/api/html/`. | Local generation only; no guard by itself. |
| `make api-docs-coverage` | `Makefile`, `scripts/check_api_docs_coverage.py` | Checks generated reference and source pages for checked-in public headers. | Enforces required page presence after generation. |
| `make api-docs-local-only` | `Makefile`, `scripts/check_api_docs_local_only.sh` | Checks ignored, untracked, unstaged, and non-visible untracked status for `docs/api/`. | Enforces current local-only generated-output policy. |
| `make docs-check` | `Makefile` | Runs `docs` plus `api-docs-coverage`. | Proves generation and configured header page coverage locally. |
| `make api-docs-validate` | `Makefile` | Runs `docs-check` plus `api-docs-local-only`. | Combines coverage and local-only staging behavior. |
| `make api-docs-freshness` | `Makefile` | Alias for `api-docs-validate`. | Current selected generated API freshness proof. |

## Script And Test Inventory

| Surface | File | Current responsibility | Gap or boundary |
| --- | --- | --- | --- |
| Coverage script | `scripts/check_api_docs_coverage.py` | Requires `docs/api/html/index.html`, reference pages, and source pages for checked-in public headers under `include/`. | Does not check hosted publication metadata, Doxygen warning policy, or non-input adoption docs. |
| Local-only script | `scripts/check_api_docs_local_only.sh` | Requires `docs/api`, `docs/api/html`, and `docs/api/html/index.html` to be ignored; rejects tracked, staged, and visible untracked generated API files. | Correct for current policy; would need redesign if committed or hosted generated output is selected. |
| Python tests | `tests/` | No generated API workflow guard was found by Day 4 search. | Publication or workflow changes would need a checked-in guard if selected. |

## Workflow And Artifact Inventory

Day 4 inspected `.github/workflows/*.yml` for generated API docs validation,
artifact retention, and publication keywords.

| Workflow | Current generated API docs role | Artifact/publication note |
| --- | --- | --- |
| `.github/workflows/ci.yml` | No generated API docs target found. | Uploads existing oracle, selected comparison, selected performance, dead-code, and coverage artifacts; none are generated API HTML. |
| `.github/workflows/macos-ci.yml` | No generated API docs target found. | Uploads selected comparison artifacts; no generated API HTML artifact. |
| `.github/workflows/windows-ci.yml` | No generated API docs target found. | No generated API HTML publication or artifact path found. |

No workflow currently runs `make docs-check`, `make api-docs-freshness`, or an
equivalent generated API docs target. No workflow currently uploads
`docs/api/html/`, deploys Pages, or writes generated API publication metadata.

## Stale-Output Guard Findings

| Question | Current answer |
| --- | --- |
| Can a maintainer regenerate and validate local generated API HTML? | Yes, with `make api-docs-freshness`. |
| Does local validation rebuild output before checking page coverage? | Yes, `docs-check` depends on `docs`, then runs coverage. |
| Does the local guard prove generated output was freshly produced in hosted CI? | No. It is local command evidence only. |
| Does CI currently prevent generated API HTML drift? | No generated API docs CI lane was found. |
| Does the current guard detect Doxygen warnings as fatal errors? | No. Day 3 observed no warning lines, but `WARN_AS_ERROR = NO`. |

## Staged Generated-File Guard Findings

The current local-only guard checks three generated-output states:

| State | Current check | Day 4 result |
| --- | --- | --- |
| Tracked generated files | `git ls-files docs/api` | Empty output. |
| Staged generated files | `git diff --cached --name-only -- docs/api` | Empty output. |
| Non-ignored generated files | `git ls-files --others --exclude-standard docs/api` | Empty output. |

The guard intentionally rejects staged generated API files unless a future
product decision selects committed generated output.

## CI Artifact Retention And Metadata Gaps

| Gap | Impact if hosted or retained artifact is selected |
| --- | --- |
| No generated API artifact upload exists. | A retained artifact decision would need upload paths, retention policy, and fail-closed `if-no-files-found` behavior. |
| No generated API publication metadata exists. | Hosted or artifact status would need source commit, branch, generation command, support tier, and claim-boundary metadata. |
| No Pages deployment path exists. | Hosted publication would need an explicit deployment mechanism and reviewable workflow contract. |
| No generated API workflow guard exists. | Workflow drift could bypass intended docs publication or freshness behavior unless a guard is added. |
| No hosted stale-output check exists. | Hosted/artifact claims would need proof that output was generated from the checked-out source. |

## Product-Decision Implications

| Product status option | Current guard fit | Required follow-through if selected |
| --- | --- | --- |
| Stronger local-only status | Strongest match to current code. | Tighten docs/guards as needed; possibly add CI or workflow guard if desired. |
| Retained CI artifact | Not currently implemented. | Add CI docs generation, artifact upload, metadata, and fail-closed upload checks. |
| Hosted publication | Not currently implemented. | Add publication workflow, metadata, freshness proof, and deployment guard. |
| Committed generated output | Conflicts with current ignored-output policy. | Change ignore/tracking policy and replace local-only guard semantics. |

## Day 4 Deliverables

- docs target inventory
- CI artifact and workflow inventory
- stale-output guard findings
- staged generated-file guard findings
- Day 4 guard and CI audit artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Current generated API guard coverage is explicit. | Complete | Target/script inventory and stale/staged guard tables above. |
| Publication metadata gaps are visible before the decision. | Complete | CI artifact retention and metadata gap table above. |
| Workflow behavior is tied to checked-in scripts or tests. | Complete | Current local behavior is tied to checked-in Make/script owners; workflow behavior currently has no generated API docs owner. |
