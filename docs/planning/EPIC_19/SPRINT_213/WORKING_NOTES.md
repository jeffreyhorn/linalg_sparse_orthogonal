# Sprint 213 Working Notes: Generated API Publication Decision

## Sprint Goal

Decide whether generated API HTML remains local-only or is published, then
implement the selected policy with matching automation, routing guards,
documentation, and validation evidence.

## Scope Boundary

Sprint 213 is a generated API publication policy sprint. It may select hosted
generated API HTML, retained generated-doc artifacts, committed generated HTML,
or a stronger local-only policy, but only after the current local-only evidence
surface and publication risks are documented.

The sprint must not claim broad API stability, package-manager distribution,
shared-library or dynamic ABI support, release artifacts, broad Windows parity,
portable performance, external-library parity, or state-of-the-art coverage.
Generated API publication, if selected, must be narrow enough to have explicit
freshness, routing, retention, and stale-output controls.

## Item Checklist

| Epic item | Sprint 213 interpretation | Status |
| --- | --- | --- |
| 213.1 Publication Option Review | Compare local-only, hosted Pages, retained artifact, and committed generated HTML policies against current Sprint 204 local-only guards. | Complete for option inventory; Day 1 identifies the current local-only evidence surface, Day 2 captures command-backed baseline behavior, and Day 3 compares local-only, hosted Pages, retained artifact, and committed generated HTML policy implications. |
| 213.2 Policy Decision | Select one generated API policy and record support, retention, routing, and claim implications. | Complete; Day 5 selects stronger local-only closure and rejects hosted, retained-artifact, and committed generated HTML publication for this sprint. |
| 213.3 Automation Implementation | Implement workflow, retention, route, staging, or stronger local-only guard behavior for the selected policy. | Functionally complete for selected automation path; Day 7 hardens routing required text and Day 8 hardens folded workflow staging/archive detection. |
| 213.4 Routing And Guard Tests | Add tests for publication links, generated-output staging, workflow references, and source-controlled route coverage. | Complete for main bypass categories; Day 7 adds routing regressions, Day 8 adds folded staging/archive regressions, Day 9 adds encoded HTML-anchor route/link regressions, and Day 10 adds quote-aware workflow literal and `.yaml` coverage. |
| 213.5 User And Maintainer Docs | Update API reference, README, INSTALL, maintainer guide, and generated API evidence docs. | Complete; Day 11 completes user-facing docs and Day 12 completes maintainer repair/residual guidance for stronger local-only generated API closure. |
| 213.6 Validation And Closeout | Run docs-check, api-docs-freshness, routing/local-only tests, workflow checks, and full C gate if headers changed. | Complete; Days 13-14 pass `make docs-check`, focused API docs regression suites, `make api-docs-freshness`, closeout reconciliation, and whitespace/status checks. Full C gate is skipped because no `.c` or `.h` files changed. |

## Day 1: Generated API Evidence Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial evidence |
| --- | --- | --- |
| 213.1 Publication Option Review | Identify the current generated API policy, automation owners, docs owners, and historical publication context before option comparison. | Current policy is local-only generated Doxygen HTML under `docs/api/html/`, guarded by `make api-docs-freshness`. |
| 213.2 Policy Decision | Preserve all policy branches until criteria and baseline evidence are complete. | Day 1 records local-only, hosted Pages, retained artifact, and committed generated HTML as open options for Day 3-5 analysis. |
| 213.3 Automation Implementation | No implementation on Day 1. | Automation changes deferred until policy decision. |
| 213.4 Routing And Guard Tests | Identify current regression-test owners and likely future fixture classes. | Current owners include API docs coverage, local-only, and routing tests plus workflow publication checks. |
| 213.5 User And Maintainer Docs | Identify user-facing and maintainer-facing generated API wording surfaces. | README, INSTALL, API reference, tutorial/cookbook/solver docs, maintainer guide, and planning artifacts inventoried. |
| 213.6 Validation And Closeout | Plan documentation-only validation for Day 1 and generated API validation for later days. | Day 1 uses `git diff --check`; later days need focused API docs and workflow guard validation. |

### Evidence Ledger

| Surface | Current owner | Day 1 finding | Sprint 213 relevance |
| --- | --- | --- | --- |
| Sprint source plan | `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 213 is a 166-hour sprint to decide generated API publication policy and implement matching automation. | Source of item scope and deliverables. |
| Day plan | `docs/planning/EPIC_19/SPRINT_213/PLAN.md` | Day 1 is evidence intake; Day 5 is the policy decision; Days 7-12 implement and document the selected policy. | Controls sequencing so implementation does not outrun decision evidence. |
| Doxygen configuration | `Doxyfile` | Input is checked-in `include/` headers, output directory is `docs/api`, and HTML output is `html`. | Defines current generated-output root and source header scope. |
| Ignore policy | `.gitignore` | `docs/api/` is ignored; generated `include/sparse_version.h` is ignored separately. | Current local-only and generated-header boundary. |
| Make targets | `Makefile` | `docs` runs Doxygen; `docs-check` depends on `api-docs-coverage`; `api-docs-freshness` serializes coverage, local-only, and routing validation through `api-docs-validate`. | Main validation wiring surface. |
| Coverage checker | `scripts/check_api_docs_coverage.py` | Checks generated Doxygen reference/source pages for checked-in public headers and excludes generated `sparse_version.h`. | Current freshness and page-coverage owner. |
| Local-only guard | `scripts/check_api_docs_local_only.sh` | Proves `docs/api/` remains ignored/untracked/unstaged and workflows do not combine generated API output paths with publication semantics. | Main guard for local-only or future publication bypass fixtures. |
| Routing guard | `scripts/check_api_docs_routing.py` | Keeps user-facing API links on `docs/api_reference.md`, checked-in headers, `Doxyfile`, workflow guides, and INSTALL; rejects generated API output and unsupported hosted publication links. | Main source/generated route contract. |
| Coverage tests | `tests/test_api_docs_coverage.py` | Regression owner for generated page coverage and excluded generated header behavior. | Candidate test owner for header/page freshness changes. |
| Local-only tests | `tests/test_api_docs_local_only_guard.py` | Regression owner for ignored output, workflow publication, broad path, archive, and staging bypasses. | Candidate owner for selected local-only hardening or publication exceptions. |
| Routing tests | `tests/test_api_docs_routing.py` | Regression owner for Markdown/HTML/reference route parsing and forbidden generated/hosted link detection. | Candidate owner for source route and publication-link behavior. |
| API reference | `docs/api_reference.md` | User-facing API reference path; states generated HTML is local-only, ignored, not hosted, not source-controlled, and not publication evidence. | Primary user-facing policy surface. |
| README | `README.md` | Routes users to `docs/api_reference.md`, `include/`, and `make api-docs-freshness`; states generated HTML is local-only ignored output. | High-visibility user entry point. |
| INSTALL support matrix | `INSTALL.md` | Lists local generated API HTML as `local-only` with no hosted API publication, retained artifact, committed HTML, or completeness beyond Doxyfile-selected headers. | Public support truth for generated API status. |
| Maintainer guide | `docs/maintainer_guide.md` | Names Sprint 204 as current policy owner and instructs reviewers to reject hosted URLs, artifact uploads, Pages deployment, or committed `docs/api/` without reopening the product decision. | Maintainer repair and review surface. |
| Workflow files | `.github/workflows/*.yml` | Current local-only guard scans workflows for generated API publication semantics. | Main publication/staging risk surface. |
| Historical Epic 19 review | `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md` | Identifies generated API publication decision as an Epic 19 gap. | Rationale for reopening the policy in Sprint 213. |

### Current Generated API Boundary

| Field | Current value or policy |
| --- | --- |
| Source-controlled entry point | `docs/api_reference.md` |
| Exact declarations | Checked-in public headers under `include/` |
| Generated output root | `docs/api/` |
| Generated HTML path | `docs/api/html/` |
| Doxygen input | `include/`, `*.h`, non-recursive |
| Generated installed header boundary | `include/sparse_version.h` remains generated/ignored and outside expected Doxygen pages |
| Support tier | `local_only` |
| Freshness command | `make api-docs-freshness` |
| Validation chain | `docs` -> `api-docs-coverage` -> `docs-check` -> `api-docs-local-only` -> `api-docs-routing` -> `api-docs-validate` -> `api-docs-freshness` |
| Current publication status | Not hosted, not retained as generated-doc artifact, not committed, not release evidence |
| Current completeness boundary | Checked-in public headers selected by `Doxyfile`, not every installed/generated header |
| Current policy owner | Sprint 204 stronger local-only generated API policy |

### Publication Options Preserved For Decision

| Option | Day 1 status | Key decision questions |
| --- | --- | --- |
| Stronger local-only | Open | Are current guards sufficient, or should Sprint 213 harden workflow, routing, and staging checks while keeping generated HTML unhosted? |
| Hosted generated API HTML | Open | What host, route, freshness proof, deployment workflow, retention, stale-output rollback, and user docs are required? |
| Retained generated-doc artifact | Open | What retention period, artifact naming, artifact freshness proof, and user/maintainer wording prevent release-evidence overclaims? |
| Committed generated HTML | Open | Is review noise, repository size, staleness control, and generated-source ownership acceptable? |

### Decision Log

| Day | Decision | Rationale |
| --- | --- | --- |
| 1 | No publication decision yet. | Day 1 is intake only. The current policy is intentionally local-only, so Days 2-5 must compare options and define stop conditions before implementation. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only Day 1 changes. |
| `python3 tests/test_api_docs_coverage.py` | Planned for later days. | Use when coverage checker or generated page expectations change. |
| `python3 tests/test_api_docs_local_only_guard.py` | Planned for later days. | Use when local-only or publication workflow guard behavior changes. |
| `python3 tests/test_api_docs_routing.py` | Planned for later days. | Use when route/link policy changes. |
| `make docs-check` | Planned for baseline/implementation days. | Requires Doxygen and generated local output. |
| `make api-docs-freshness` | Planned for baseline/implementation days. | Full selected generated API validation chain. |
| `make format && make lint && make test` | Not required for Day 1. | No `.c` or `.h` files changed. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Accidental publication through workflow drift | Broad docs paths, archives, release uploads, or Pages actions can expose `docs/api/html/` without a product decision. | Keep or strengthen workflow publication/staging guards; add regression fixtures before allowing any publication path. |
| Source/generated route confusion | Users may bookmark generated output instead of the source-controlled API reference path. | Keep routing guard strict unless a hosted publication path is selected and documented. |
| Stale hosted generated HTML | Hosted docs can become stale relative to public headers and look authoritative. | Require freshness gate, deployment ordering, stale-output rollback, and maintainer repair docs if hosted publication is selected. |
| Retained artifact overclaim | CI artifacts can be mistaken for release evidence or supported hosted docs. | Require explicit retention wording, artifact naming, and docs guard coverage if retained artifacts are selected. |
| Committed generated HTML review noise | Generated output can obscure code review and drift from Doxyfile/header inputs. | Require generated-output ownership, size/noise assessment, and freshness checks if committed output is selected. |
| Generated header confusion | `sparse_version.h` is installed/generated but not part of Doxygen checked-in header coverage. | Preserve the existing generated-header exclusion unless the Doxygen input policy changes deliberately. |
| Documentation overclaim | README, INSTALL, API reference, and maintainer guide are public support surfaces. | Add docs guard markers and forbidden-claim tests after the Day 5 policy decision. |

### Changed Surface Tracker

| Path | Day | Change type | Notes |
| --- | --- | --- | --- |
| `docs/planning/EPIC_19/SPRINT_213/PLAN.md` | 0 | Added | Day-by-day Sprint 213 plan created before Day 1 execution. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 1 | Added | Sprint ledger, evidence inventory, risk register, validation matrix, and changed-surface tracker. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day1-generated-api-evidence-intake.md` | 1 | Added | Day 1 intake artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 2 | Updated | Current local-only baseline commands, generated page counts, documentation wording, guard coverage, and Day 3 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day2-current-local-only-baseline.md` | 2 | Added | Day 2 baseline artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 3 | Updated | Publication option matrix, workflow/routing/retention implications, risk inventory, and Day 4 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day3-publication-option-inventory.md` | 3 | Added | Day 3 publication option inventory artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 4 | Updated | Decision rule, per-policy acceptance criteria, stop conditions, and evidence-to-test mapping. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day4-decision-criteria.md` | 4 | Added | Day 4 decision criteria artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 5 | Updated | Product policy decision, rejected alternatives, selected implementation scope, non-claims, and Day 6 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day5-product-policy-decision.md` | 5 | Added | Day 5 generated API policy decision artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 6 | Updated | Automation ownership map, path policy, fixture plan, validation ordering, and Day 7 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day6-automation-design.md` | 6 | Added | Day 6 stronger local-only automation design artifact. |
| `scripts/check_api_docs_routing.py` | 7 | Updated | Requires maintainer-guide future publication reopening criteria wording as part of the local-only generated API routing contract. |
| `tests/test_api_docs_routing.py` | 7 | Updated | Adds regressions for repository suites artifact URLs and missing future-publication reopening wording. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 7 | Updated | Records implementation batch one, validation, and Day 8 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day7-automation-implementation-batch-one.md` | 7 | Added | Day 7 implementation artifact. |
| `scripts/check_api_docs_local_only.sh` | 8 | Updated | Extends staging/archive command checks to independently folded `run` scalar text so multiline commands cannot publish staged `docs/` output without merging unrelated workflow fields. |
| `tests/test_api_docs_local_only_guard.py` | 8 | Updated | Adds folded `cp -R docs artifact/` and folded `tar ... docs/` artifact-upload regressions. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 8 | Updated | Records implementation batch two, validation, and Day 9 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day8-automation-implementation-batch-two.md` | 8 | Added | Day 8 implementation artifact. |
| `tests/test_api_docs_routing.py` | 9 | Updated | Adds encoded HTML-anchor source route and generated-output route regressions. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 9 | Updated | Records routing/link validation, source/generated route evidence, and Day 10 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day9-routing-and-link-validation.md` | 9 | Added | Day 9 routing and link validation artifact. |
| `scripts/check_api_docs_local_only.sh` | 10 | Updated | Reuses quote-aware YAML comment stripping for direct generated API workflow-reference scans. |
| `tests/test_api_docs_local_only_guard.py` | 10 | Updated | Adds quoted-`#` generated path and `.yaml` workflow extension regressions. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 10 | Updated | Records workflow/staging validation, current workflow audit evidence, and Day 11 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day10-workflow-and-staging-validation.md` | 10 | Added | Day 10 workflow and staging validation artifact. |
| `README.md` | 11 | Updated | Clarifies that CI artifacts, release downloads, Pages deployments, and repository `docs/api/` paths are not the API documentation route. |
| `INSTALL.md` | 11 | Updated | Clarifies local generated API HTML as an on-demand local view, not install/release/hosted/artifact publication. |
| `docs/api_reference.md` | 11 | Updated | Adds durable-link guidance that favors the source-controlled API reference and public headers over generated HTML. |
| `scripts/check_api_docs_routing.py` | 11 | Updated | Enforces the new README, INSTALL, and API reference local-only user documentation markers. |
| `tests/test_api_docs_routing.py` | 11 | Updated | Adds regressions for missing Day 11 user documentation markers. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 11 | Updated | Records user documentation update, guard markers, validation, and Day 12 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day11-user-documentation-update.md` | 11 | Added | Day 11 user documentation artifact. |
| `docs/maintainer_guide.md` | 12 | Updated | Adds generated API local-only repair workflow, expected repair artifacts, and remaining unclaimed publication options. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 12 | Updated | Records current Sprint 213 branch direction and remaining unclaimed generated API publication options. |
| `scripts/check_api_docs_routing.py` | 12 | Updated | Enforces maintainer repair workflow and residual-option markers. |
| `tests/test_api_docs_routing.py` | 12 | Updated | Adds missing-marker regressions for maintainer repair and residual wording. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 12 | Updated | Records maintainer documentation update, residuals, validation, and Day 13 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day12-maintainer-documentation-and-residuals.md` | 12 | Added | Day 12 maintainer documentation artifact. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 13 | Updated | Records integrated validation evidence, skipped C-gate rationale, and Day 14 handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day13-integrated-validation.md` | 13 | Added | Day 13 integrated validation artifact. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 14 | Updated | Marks Sprint 213 closed with stronger local-only generated API closure and narrows the pending Epic 19 range to Sprints 214-216. |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 14 | Updated | Records final item reconciliation, residuals, validation summary, and retrospective handoff. |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day14-closeout-review.md` | 14 | Added | Day 14 closeout review artifact. |

### Open Questions For Day 2

1. Does `make api-docs-freshness` pass on the current branch with local Doxygen
   available, and what exact generated-page/header counts does it report?
2. Which current local-only guard regressions are most relevant to the Sprint
   213 policy options?
3. Are there any existing workflow comments, examples, or planning references
   that imply generated API publication despite the current local-only policy?
4. What evidence would make hosted Pages safer than retained artifacts or
   committed generated HTML?
5. What stop condition should force the sprint to ask for user input instead
   of selecting a publication path?

### Day 1 Validation

Commands planned for Day 1 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 1 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 2: Current Local-Only Baseline

### Baseline Commands

| Command | Result | Evidence captured |
| --- | --- | --- |
| `make api-docs-freshness` | Pass | Doxygen generated local HTML, coverage checked 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages; local-only and routing guards passed. |
| `python3 tests/test_api_docs_coverage.py && python3 tests/test_api_docs_local_only_guard.py && python3 tests/test_api_docs_routing.py` | Pass | Standalone regression suites for coverage, local-only workflow/staging behavior, and routing all completed without failure. |
| `git check-ignore -v docs/api docs/api/html docs/api/html/index.html include/sparse_version.h` | Pass | `docs/api/` and descendants are ignored by `.gitignore:44`; generated `include/sparse_version.h` is ignored by `.gitignore:48`. |
| `git status --short --ignored docs/api include/sparse_version.h` | Pass | Local generated Doxygen output appears only as ignored `!! docs/api/`. |
| `rg -n "docs/api\|api/html\|upload-artifact\|deploy-pages\|gh-pages\|pages" .github/workflows` | Informational | Workflows contain existing `actions/upload-artifact@v4` steps, but no generated API output path references; `api-docs-local-only` also reports no workflow generated API publication semantics. |

The generated `docs/api/` tree is ignored local output and was not added to the
branch. It is recorded here as baseline evidence only.

### Local Generated Output Snapshot

| Field | Day 2 baseline |
| --- | --- |
| Doxygen command | `doxygen Doxyfile` via `make docs` |
| Output root | `docs/api/` |
| HTML directory | `docs/api/html/` |
| Generated index | `docs/api/html/index.html` present |
| Checked-in public headers | 18 |
| Generated reference pages | 18 |
| Generated source pages | 18 |
| Generated installed header policy | `sparse_version.h` remains separate installed-header policy row, not an expected Doxygen page |
| Git status | `docs/api/` is ignored output only |
| Support interpretation | Local current-output proof for this checkout only |

### Current Local-Only Behavior

| Surface | Current behavior | Day 2 interpretation |
| --- | --- | --- |
| `Doxyfile` | Reads checked-in `include/*.h`, writes HTML under `docs/api/html/`. | Doxygen input/output contract is narrow and local. |
| `api-docs-coverage` | Requires generated reference/source pages for checked-in public headers and rejects stale/missing/obsolete header pages. | Freshness is branch-local and tied to checked-in public headers. |
| `api-docs-local-only` | Requires no staged/tracked/non-ignored generated API files, ignore rules for `docs/api/`, Doxyfile output settings, required local-only docs wording, and no workflow publication semantics. | Current guard fails closed for accidental generated API publication. |
| `api-docs-routing` | Checks seven routing docs, Makefile wiring, required source-controlled routes, required local-only wording, and absence of generated API publication links. | Current docs route users to source-controlled API docs, not generated HTML. |
| Workflows | Existing upload-artifact steps serve other evidence lanes, not generated API output. | Publication risk is workflow drift, not current generated API upload. |
| Ignored output | `docs/api/` is ignored by `.gitignore`; generated files are not source-controlled. | Committed generated HTML is currently unsupported. |

### Current Documentation Wording

| Document | Day 2 wording baseline |
| --- | --- |
| `README.md` | Lists `make docs-check`, `make api-docs-freshness`, and `docs/api_reference.md`; states generated HTML is local-only ignored output, not hosted documentation, retained artifact, source-controlled output, or release evidence. |
| `INSTALL.md` | Support/readiness matrix lists local generated API HTML as `local-only` and explicitly excludes hosted API publication, retained generated-doc artifacts, committed generated HTML, and completeness beyond Doxyfile-selected public headers. |
| `docs/api_reference.md` | Defines checked-in public headers as source of truth; generated HTML is local-only output current only after `make api-docs-freshness`; generated links are not hosted or source-controlled publication surfaces. |
| `docs/maintainer_guide.md` | States Sprint 204 owns the current local-only policy; review guidance rejects hosted API URLs, workflow uploads, Pages deployment, or committed `docs/api/` without reopening the product decision. |
| `docs/tutorial.md` and `docs/cookbook.md` | Route exact declarations to `docs/api_reference.md` rather than generated HTML. |
| `docs/solver_selection.md` | Does not create a generated API publication route. |

### Guard Coverage And Known Gaps

| Area | Current coverage | Known Day 2 gap or decision dependency |
| --- | --- | --- |
| Header/page freshness | Coverage checker validates 18 checked-in public headers and generated reference/source pages. | Publication options still need stale hosted-output rollback criteria. |
| Generated-header exclusion | `sparse_version.h` remains excluded from expected Doxygen pages and ignored as generated install header. | Any expanded Doxygen input policy would need a separate decision. |
| Local-only staging | Guard rejects staged/tracked/non-ignored generated API files. | Committed generated HTML would require intentionally changing this guard and replacing local-only wording. |
| Workflow publication | Guard rejects generated API paths combined with publication semantics, broad docs paths, archive staging, upload/deploy actions, and dynamic publication paths. | Hosted or retained-artifact publication would need narrow allowlisted workflow behavior and regression coverage. |
| Routing | Guard rejects generated-output and unsupported hosted publication links while preserving source-controlled API routes. | Hosted publication would need a deliberately allowed project-owned API URL shape and source/generated route wording. |
| User docs | Required local-only wording is present in README, INSTALL, API reference, and maintainer guide. | Any publication selection must update docs and add guard markers to prevent contradictory non-claims. |

### Day 2 Outcome

Item 213.1 now has command-backed current-state data. The baseline confirms
that current generated API HTML is locally fresh for the checked-in Doxygen
input set, ignored by Git, absent from workflow publication paths, and routed
through source-controlled API docs rather than generated HTML.

Day 3 should compare the four publication options against this baseline and
identify which workflow, retention, routing, freshness, and documentation
changes each option would require.

### Day 2 Validation

Day 2 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 2 closeout:

```sh
make api-docs-freshness
python3 tests/test_api_docs_coverage.py && python3 tests/test_api_docs_local_only_guard.py && python3 tests/test_api_docs_routing.py
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 3: Publication Option Inventory

### Option Definitions

| Option | Definition | Current-branch delta |
| --- | --- | --- |
| Stronger local-only | Keep generated Doxygen HTML ignored under `docs/api/`, keep `docs/api_reference.md` as the public source-controlled route, and use Sprint 213 to close the publication residual with stronger guards/docs. | Least disruptive; extends current Sprint 204 policy instead of changing publication behavior. |
| Hosted Pages publication | Generate Doxygen HTML in CI and deploy a project-owned hosted documentation site, likely with GitHub Pages or a dedicated docs workflow. | Largest workflow/routing/docs change; requires new hosted URL policy and stale-output controls. |
| Retained generated-doc artifact | Generate Doxygen HTML in CI and upload a narrowly named artifact containing generated docs for a fixed retention window. | Requires upload path exceptions and artifact retention wording while preserving no public docs site. |
| Committed generated HTML | Commit generated `docs/api/` output to the repository and review it as source-controlled generated content. | Requires changing ignore/staging policy and accepting generated-output review noise. |

### Option Comparison Matrix

| Criteria | Stronger local-only | Hosted Pages publication | Retained generated-doc artifact | Committed generated HTML |
| --- | --- | --- | --- | --- |
| User discoverability | Lowest external discoverability; users must run `make api-docs-freshness` or use source-controlled API docs. | Highest discoverability through a stable URL. | Medium for reviewers with artifact access; low for general users. | Medium; browsable in repository but not a polished docs site. |
| Freshness proof | Current `make api-docs-freshness` already passes locally. | Must prove generated output is produced after Doxygen/coverage checks and deployed only from fresh output. | Must prove uploaded artifact is generated after freshness checks. | Must prove committed output is regenerated from current headers before merge. |
| Routing policy | Keep rejecting generated-output and hosted-publication links. | Add explicit allowlist for the selected hosted API URL while rejecting stale/unowned hosts. | Usually keep user docs on source-controlled route; maintainer docs may mention artifact retrieval. | Decide whether user docs may link to committed generated pages or still route to `docs/api_reference.md`. |
| Workflow changes | Optional guard hardening only. | New deployment job, permissions, Pages artifact, branch/environment rules, and deployment status handling. | New upload step with narrow `docs/api/html/**` or archive path and retention controls. | No CI publication workflow required, but CI must check generated tree freshness. |
| Retention | Not applicable; local output is not retained. | Hosted until overwritten or disabled; needs stale site handling. | Artifact retention days must be explicit. | Retained indefinitely in Git history. |
| Review noise | Low. | Medium; workflow and generated deployment metadata changes. | Medium; workflow and artifact policy changes. | High; generated HTML is currently 214 files and about 3.1 MB locally. |
| Claim-boundary risk | Low if current guards are hardened. | High; hosted docs can be mistaken for release/API stability evidence. | Medium-high; artifacts can be mistaken for release evidence. | Medium; committed generated files can be mistaken for authoritative support beyond Doxyfile input. |
| Guard changes | Strengthen existing local-only and routing guards. | Replace or conditionally relax local-only guard; add hosted URL/routing/deploy tests. | Replace or conditionally relax local-only guard for one upload path; add artifact tests. | Replace ignore/staging guard and add generated-tree freshness tests. |

### Workflow And Routing Impact Map

| Area | Stronger local-only | Hosted Pages publication | Retained generated-doc artifact | Committed generated HTML |
| --- | --- | --- | --- | --- |
| `.github/workflows` | Continue rejecting generated API output paths mixed with publication semantics. Existing upload-artifact steps remain for non-API evidence lanes. | Add a dedicated docs publication job with least-privilege Pages permissions and deployment only after `make docs-check` or `make api-docs-freshness`. | Add one narrow upload step after `make api-docs-freshness`; forbid broad `docs/` or repository-root upload paths. | No publication upload required; optionally add CI check that committed `docs/api/` matches regenerated output. |
| `scripts/check_api_docs_local_only.sh` | Harden fail-closed workflow/staging scans and documentation non-claims. | Convert to selected publication guard or add a separate publication-policy guard; allow only the selected deploy path. | Convert to selected artifact guard or add a retained-artifact policy guard; allow only exact artifact paths. | Replace ignore requirement with source-controlled generated-output freshness/ownership checks. |
| `scripts/check_api_docs_routing.py` | Continue rejecting generated-output and hosted publication links. | Allow only selected project-owned hosted docs route; reject release/artifact/custom stale routes. | Keep user routes source-controlled; optionally allow maintainer artifact instructions only. | Decide whether committed generated pages become valid source routes; update required routes accordingly. |
| `docs/api_reference.md` | Preserve current local-only wording and explain why no hosted docs exist. | Add hosted URL and freshness/retention caveats while preserving source header ownership. | Add artifact retrieval/retention caveats if user-visible. | Explain committed generated pages and source/header precedence. |
| `README.md` and `INSTALL.md` | Keep source-controlled route and local-only support tier. | Promote selected hosted generated API support tier with non-claims. | Promote retained-artifact tier only if intended for users/maintainers. | Promote source-controlled generated HTML only if accepted. |
| `docs/maintainer_guide.md` | Expand repair workflow for local generation and guard failures. | Add deployment repair, stale hosted site handling, and rollback workflow. | Add artifact retention, naming, and retrieval workflow. | Add generated tree regeneration, diff review, and stale committed-output workflow. |

### Retention And Staleness Risk List

| Risk | Stronger local-only | Hosted Pages publication | Retained generated-doc artifact | Committed generated HTML |
| --- | --- | --- | --- | --- |
| Stale output visible to users | Low; output is local and regenerated on demand. | High unless deployment is gated and stale site rollback is defined. | Medium; stale artifacts remain downloadable during retention. | High unless every merge enforces regeneration. |
| Unsupported API stability inference | Low with current non-claims. | High; public docs sites look official. | Medium; retained artifacts can look like release deliverables. | Medium; committed generated files look source-owned. |
| Broad route bypass | Low if routing guard stays strict. | High unless hosted URL allowlist is narrow. | Medium if artifact URLs are mentioned. | Medium if docs link directly into generated tree. |
| Workflow path broadening | Low; guards currently reject broad docs publication shapes. | High; deploy actions often consume directories broadly. | High; upload-artifact paths can accidentally include all `docs/`. | Low for workflow publication, higher for stale committed output. |
| Review burden | Low. | Medium. | Medium. | High because current generated HTML is many files. |

### Day 3 Observations

- The current generated tree is small enough to publish technically, but it is
  still generated output: the local `docs/api/` tree contains 214 files and is
  about 3.1 MB after Day 2 generation.
- Current workflows already use `actions/upload-artifact@v4` for selected
  comparison, benchmark, dead-code, and coverage evidence, but Day 2 and Day 3
  inspection found no generated API output paths in those upload steps.
- Hosted Pages publication gives the clearest user-facing discoverability
  improvement, but it has the highest stale-output and claim-boundary risk.
- Retained generated-doc artifacts are less public than Pages, but they still
  require publication exceptions and retention wording.
- Committed generated HTML would make output visible in the repository, but it
  conflicts most directly with the current ignore/local-only guard and would
  add persistent generated review noise.
- Stronger local-only is the smallest policy change and likely the safest
  closure path if Day 4 criteria do not require public generated docs.

### Day 3 Outcome

Item 213.1 now covers all four project-plan publication options. No option is
selected yet. Day 4 should convert this inventory into decision criteria and
stop conditions so Day 5 can choose a policy without relying on unstated
preferences.

### Day 3 Validation

Day 3 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 3 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 4: Decision Criteria And Stop Conditions

### Decision Rule

Day 5 must select exactly one of these policies:

1. **Stronger local-only**: allowed when current local-only validation remains
   sufficient and publication options fail one or more acceptance criteria.
2. **Hosted generated API publication**: allowed only when hosted URL,
   deployment, freshness, routing, stale-site, and non-claim criteria are all
   satisfiable inside this sprint.
3. **Retained generated-doc artifact**: allowed only when artifact path,
   retention, freshness, retrieval, and non-release wording criteria are all
   satisfiable inside this sprint.
4. **Committed generated HTML**: allowed only when generated-output review
   noise, repository-size impact, freshness, and source/generated ownership
   criteria are all acceptable inside this sprint.

If no publication path satisfies every required criterion, Sprint 213 should
select stronger local-only closure and explicitly reject the publication
options for now.

### Stronger Local-Only Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Existing validation passes | `make api-docs-freshness` passes and current Doxygen coverage remains 18 checked-in public headers, 18 reference pages, and 18 source pages unless headers intentionally change. |
| Generated output remains ignored | `docs/api/`, `docs/api/html/`, and generated `include/sparse_version.h` remain ignored and untracked. |
| Workflow publication remains rejected | Workflow scans reject generated API output paths combined with artifact, Pages, deploy, release, archive, staging, broad docs, root, or dynamic publication semantics. |
| Routing remains source-controlled | User-facing docs continue routing to `docs/api_reference.md`, checked-in public headers, `Doxyfile`, workflow guides, and INSTALL support matrix. |
| Non-claim wording remains explicit | README, INSTALL, API reference, and maintainer guide state no hosted API publication, retained generated-doc artifact, committed generated HTML, release evidence, package/ABI proof, or completeness beyond Doxyfile-selected checked-in headers. |
| Publication residual is closed deliberately | Documentation explains why generated HTML remains local-only and what evidence would be needed to reopen hosted/artifact/committed output. |

### Hosted Publication Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Project-owned hosted URL | A single project-owned hosted generated API URL is selected and documented; unrelated hosts remain rejected. |
| Deployment ordering | The workflow deploys only after Doxygen generation, coverage/freshness, local staging checks adapted for publication, and routing checks pass. |
| Deployment permissions | Workflow permissions, branch/environment behavior, and deploy action are least-privilege and explicit. |
| Stale-site rollback | Maintainer docs define how to disable, roll back, or mark stale hosted docs when freshness fails or Doxygen inputs change. |
| Routing exception is narrow | Routing guard allows only the selected hosted route while still rejecting `docs/api/`, release artifacts, generic Pages/API/Doxygen hosts, and unrelated generated-output links. |
| Claim boundary is guarded | Docs and tests prevent hosted generated docs from implying API stability, release readiness, package-manager distribution, ABI support, broad platform parity, or completeness beyond Doxyfile input. |
| Retention semantics are clear | Hosted docs overwrite or retain content according to documented policy; stale historical docs are not accidentally treated as current. |

### Retained Artifact Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Exact artifact scope | Artifact name and uploaded paths include only the selected generated API output and required metadata, not broad `docs/` or repository root. |
| Retention duration | `retention-days` is explicit and documented as generated-doc evidence, not release evidence. |
| Upload ordering | Artifact upload happens only after Doxygen generation, coverage/freshness, and adapted local-only/publication checks pass. |
| Retrieval semantics | User or maintainer docs state who should use the artifact, when it expires, and why it is not a supported hosted docs site. |
| Local-only guard adaptation is narrow | Guard permits only the selected artifact path or archive and continues rejecting broad upload/deploy/publication bypasses. |
| Routing stays safe | User-facing docs do not route general users to expiring artifacts unless the product decision explicitly selects that behavior. |

### Committed Generated HTML Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Review-noise acceptance | The sprint explicitly accepts committing about 214 generated files and about 3.1 MB of output, or records updated counts if generation changes. |
| Freshness enforcement | CI or a local guard proves committed `docs/api/` output matches current checked-in headers and `Doxyfile`. |
| Ignore/staging policy updated | `.gitignore`, local-only guard, and docs wording are deliberately updated to treat `docs/api/` as source-controlled generated output. |
| Source/generated ownership is clear | Docs state whether users should prefer generated pages, `docs/api_reference.md`, or public headers for exact declarations. |
| Generated-output diffs are reviewable | Maintainer docs define how reviewers distinguish intended generated changes from stale or unrelated Doxygen churn. |
| Claim boundary remains explicit | Committed generated HTML still does not imply package, ABI, release, broad platform, performance, or state-of-the-art support. |

### Stop Conditions

Stop and ask for user direction instead of implementing if:

- the user explicitly prefers a publication option that fails one or more
  acceptance criteria;
- hosted publication requires external DNS, repository Pages settings, secrets,
  or admin changes that are unavailable in this branch;
- retained artifact policy requires a retention period, audience, or access
  model that is not specified;
- committed generated HTML is selected without accepting generated-file review
  noise and repository growth;
- any publication path requires weakening routing/local-only guards without a
  replacement guard;
- Doxygen generation or `make api-docs-freshness` fails before policy
  selection;
- documentation cannot preserve package, ABI, release, platform, performance,
  and state-of-the-art non-claims;
- required quality checks fail.

### Evidence-To-Test Mapping

| Criterion | Enforcement path |
| --- | --- |
| Header/page coverage | `scripts/check_api_docs_coverage.py`; `tests/test_api_docs_coverage.py`; `make docs-check`. |
| Local-only staging and workflow scanning | `scripts/check_api_docs_local_only.sh`; `tests/test_api_docs_local_only_guard.py`; `make api-docs-freshness`. |
| Source-controlled routing | `scripts/check_api_docs_routing.py`; `tests/test_api_docs_routing.py`; `make api-docs-freshness`. |
| Hosted URL exception if selected | New or updated routing tests for the exact selected hosted route and forbidden near-misses. |
| Retained artifact exception if selected | New or updated local-only/workflow guard fixtures for exact upload path, retention, archive shape, and broad-path rejection. |
| Committed generated HTML if selected | New or updated coverage/freshness guard proving generated tree matches checked-in headers and `Doxyfile`. |
| Documentation claim boundary | Required text markers and forbidden-link/claim regressions in API docs routing/local-only tests. |

### Day 4 Outcome

Item 213.2 now has explicit decision criteria before the Day 5 decision. The
criteria make publication paths testable rather than aspirational and preserve
strong stop conditions for cases where branch-local automation cannot safely
support a publication claim.

Day 5 should evaluate each option against these criteria, select one policy,
record rejected alternatives, and define the implementation scope for Days
6-12.

### Day 4 Validation

Day 4 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 4 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 5: Product Policy Decision

### Decision

Sprint 213 will implement **stronger local-only generated API closure**.

Generated Doxygen HTML remains ignored local output under `docs/api/html/`.
Sprint 213 will not publish hosted generated API HTML, retain generated-doc CI
artifacts, or commit generated HTML. The implementation work should strengthen
the existing local-only policy with additional guard coverage, clearer repair
workflow, and explicit future reopening criteria.

### Selected Scope

| Field | Selected policy |
| --- | --- |
| Policy | Stronger local-only generated API closure |
| Source-controlled API route | `docs/api_reference.md` plus checked-in public headers under `include/` |
| Generated output root | `docs/api/` |
| Generated HTML path | `docs/api/html/` |
| Support tier | `local_only` |
| Freshness command | `make api-docs-freshness` |
| Coverage scope | Checked-in public headers selected by `Doxyfile` |
| Generated installed header boundary | `sparse_version.h` remains separate installed-header policy, not an expected generated Doxygen page |
| Publication status | No hosted API publication, no retained generated-doc artifact, no committed generated HTML |
| Routing policy | User-facing docs continue routing through source-controlled API docs and headers |

### Rationale

The Day 4 criteria allow hosted, retained-artifact, or committed generated
HTML only when the branch can satisfy freshness, routing, retention, stale
output, workflow, and claim-boundary requirements without external ambiguity.
The current branch satisfies the stronger local-only criteria:

- `make api-docs-freshness` passes;
- generated output covers 18 checked-in public headers with 18 generated
  reference pages and 18 generated source pages;
- `docs/api/` remains ignored, untracked, and unstaged;
- current workflows do not publish generated API paths;
- routing docs keep the source-controlled entry point at
  `docs/api_reference.md`;
- README, INSTALL, API reference, and maintainer guide already carry the core
  local-only non-claims.

The publication options do not satisfy the criteria inside the current branch:

- hosted publication would require a selected hosted URL, deployment settings,
  permissions, stale-site handling, and a routing exception;
- retained artifacts would require a selected retention/audience policy and a
  narrow upload exception;
- committed generated HTML would require accepting persistent generated output
  churn for the current 214-file, about 3.1 MB generated tree and replacing the
  ignore/staging contract.

Closing with stronger local-only policy therefore resolves the Epic 19
generated API ambiguity without creating a hosted or retained publication
surface that the branch cannot fully govern.

### Rejected Alternatives

| Alternative | Reason rejected for Sprint 213 |
| --- | --- |
| Hosted Pages publication | No selected project-owned hosted URL, deployment permission model, stale-site rollback policy, or route allowlist exists yet. Publishing would weaken the current guard before replacement evidence exists. |
| Retained generated-doc artifact | No artifact audience, retention purpose, retrieval workflow, or exact upload exception is selected. Retained generated-doc artifacts could be mistaken for release evidence without stronger policy. |
| Committed generated HTML | The generated tree is currently 214 files and about 3.1 MB. Committing it would introduce persistent generated review noise and require replacing the current ignore/local-only guard. |
| Broad generated API publication | Out of scope; Sprint 213 owns one generated API policy decision, not API stability, package, release, ABI, or broad platform support. |

### Required Non-Claims

The implementation must not claim:

- hosted API publication;
- retained generated-doc artifacts;
- committed generated HTML;
- release evidence;
- package-manager distribution;
- package, ABI, shared-library, dynamic-loader, or runtime compatibility;
- broad Windows parity or broad platform parity;
- external-library parity;
- portable performance;
- state-of-the-art coverage;
- completeness beyond checked-in public headers selected by `Doxyfile`;
- generated installed-header Doxygen coverage for `sparse_version.h`.

### Implementation Direction

Days 6-12 should implement and document stronger local-only closure rather
than publication exceptions.

| Surface | Direction |
| --- | --- |
| Automation design | Keep `docs/api/` ignored and local-only; design additional checks for future publication bypass shapes if Day 6 finds gaps. |
| Local-only guard | Preserve staged/tracked/non-ignored generated-output rejection and workflow publication scanning; strengthen only where bypass fixtures reveal gaps. |
| Routing guard | Preserve source-controlled route requirements and forbidden generated/hosted publication links; add fixtures for local-only residual closure wording if needed. |
| Coverage checker | Preserve checked-in public-header coverage and `sparse_version.h` installed-header boundary. |
| Documentation | Add or sharpen wording that explains why no hosted/generated artifact path is selected and what evidence would be needed to reopen publication. |
| Maintainer guide | Add repair guidance for local-only failures and future publication decision prerequisites. |
| Project plan/residuals | Mark Sprint 213 as closing generated API publication by selecting stronger local-only policy, while preserving hosted/retained/committed publication as future options only if acceptance criteria are met. |

### Day 5 Outcome

Item 213.2 is complete. Sprint 213 now has a documented product policy:
stronger local-only generated API closure. The implementation surface is
bounded to automation design, guard hardening, routing coverage, workflow
publication checks, and documentation calibration. Unsupported generated API
publication claims are explicitly rejected.

### Day 5 Validation

Day 5 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 5 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 6: Automation Design

### Selected Policy Automation Goal

The selected policy is stronger local-only generated API closure. Automation
should continue treating `docs/api/` as ignored local generated output, while
making future publication bypasses harder to introduce accidentally and making
the route/non-claim contract easier to verify.

Day 6 does not change code. It defines the implementation targets for Days 7
and 8.

### Automation Ownership Map

| Layer | Owner | Day 6 responsibility |
| --- | --- | --- |
| Generation | `Doxyfile`; `make docs` | Preserve checked-in `include/*.h` input and `docs/api/html/` output. |
| Coverage/freshness | `scripts/check_api_docs_coverage.py`; `tests/test_api_docs_coverage.py`; `make docs-check` | Preserve 18 checked-in public-header page coverage and `sparse_version.h` exclusion unless headers change. |
| Local-only staging/workflow policy | `scripts/check_api_docs_local_only.sh`; `tests/test_api_docs_local_only_guard.py`; `api-docs-local-only` | Preserve ignored/untracked/unstaged checks and strengthen publication bypass fixtures where current tests are weakest. |
| Routing policy | `scripts/check_api_docs_routing.py`; `tests/test_api_docs_routing.py`; `api-docs-routing` | Preserve source-controlled routes and reject generated-output, hosted, retained artifact, and release/artifact publication links. |
| Validation ordering | `Makefile` | Preserve `docs` -> `api-docs-coverage` -> `docs-check` -> `api-docs-local-only` -> `api-docs-routing` -> `api-docs-validate` -> `api-docs-freshness`. |
| User docs | `README.md`; `INSTALL.md`; `docs/api_reference.md` | Preserve local-only support tier and add stronger explanation of why no hosted/artifact/committed generated output is selected if needed. |
| Maintainer docs | `docs/maintainer_guide.md` | Preserve repair workflow and add future reopening criteria after implementation. |
| Planning/residuals | Epic 19 plan and Sprint 213 artifacts | Record that Sprint 213 closes generated API publication by selecting stronger local-only policy. |

### Path Policy

| Path or route | Policy | Guard owner |
| --- | --- | --- |
| `docs/api/` | Ignored local generated output; must not be staged, tracked, or non-ignored. | `check_api_docs_local_only.sh`; local-only tests. |
| `docs/api/html/` | Local Doxygen HTML current only after `make api-docs-freshness` passes. | Coverage checker; local-only guard. |
| `docs/api/html/index.html` | Ignored generated index; not a source-controlled or hosted route. | Local-only guard; routing tests. |
| `docs/api_reference.md` | Source-controlled API reference entry point. | Routing guard. |
| `include/*.h` | Exact declaration source of truth and Doxygen input. | Coverage checker; routing guard. |
| `include/sparse_version.h` | Generated installed header; ignored and not expected as Doxygen page. | Coverage checker and docs wording. |
| Hosted generated API URLs | Rejected for Sprint 213. | Routing guard and docs wording. |
| CI retained generated-doc artifacts | Rejected for Sprint 213. | Local-only workflow guard and docs wording. |
| Committed generated HTML | Rejected for Sprint 213. | Local-only guard and `.gitignore`. |

### Fixture Plan For Days 7-8

| Fixture class | Purpose | Candidate owner | Priority |
| --- | --- | --- | --- |
| Local-only residual wording marker | Ensure docs explain that hosted, retained, and committed generated API outputs were rejected by decision, not omitted accidentally. | `tests/test_api_docs_routing.py` or local-only docs assertions. | High |
| Future reopening criteria wording | Ensure maintainer docs state the evidence needed before hosted/artifact/committed output can be selected. | Routing/local-only docs assertions. | High |
| Hosted URL near-miss | Ensure project-looking hosted generated API links remain rejected unless a future decision adds a narrow allowlist. | `tests/test_api_docs_routing.py` | High |
| Repository release/artifact URL | Ensure release/download or Actions artifact URLs for generated API docs are rejected. | `tests/test_api_docs_routing.py` | High |
| Workflow upload exact generated root | Ensure `actions/upload-artifact` of `docs/api` or `docs/api/html` remains rejected. | `tests/test_api_docs_local_only_guard.py` | Existing/confirm |
| Workflow broad docs upload | Ensure broad `docs/`, repository root, archive, or dynamic docs upload paths remain rejected when publication semantics exist. | `tests/test_api_docs_local_only_guard.py` | Existing/confirm; add gaps only |
| Committed generated HTML simulation | Ensure tracked or non-ignored `docs/api/` content fails local-only guard. | `tests/test_api_docs_local_only_guard.py` | Existing/confirm |
| Source route preservation | Ensure `docs/api_reference.md`, `include/`, `Doxyfile`, workflow guides, and INSTALL routes remain accepted. | `tests/test_api_docs_routing.py` | Existing/confirm |

### Implementation Priorities

Day 7 should start with the highest-value local-only closure gap:

1. inspect current local-only and routing regression suites for existing
   coverage of hosted publication URLs, release/artifact URLs, generated root
   uploads, broad docs uploads, and required non-claim wording;
2. add only missing fixtures or markers;
3. avoid changing publication policy or adding any allowlist;
4. run the focused API docs tests touched by the changes.

Day 8 should finish any remaining fixture categories and Makefile/docs wiring
checks:

1. confirm validation order is asserted strongly enough;
2. add any missing broad-path/archive/dynamic workflow regressions only if Day
   7 finds a gap;
3. keep implementation scoped to stronger local-only closure.

### Validation Ordering

The selected policy relies on this order:

```text
docs
  -> api-docs-coverage
  -> docs-check
  -> api-docs-local-only
  -> api-docs-routing
  -> api-docs-validate
  -> api-docs-freshness
```

The order matters because routing and local-only checks need generated output
to exist before proving it is fresh, ignored, untracked, unstaged, and not used
as a user-facing or hosted publication route.

### Day 6 Outcome

Item 213.3 now has an implementation design before code changes begin.
Selected-policy automation can be tested locally through existing API docs
coverage, local-only, routing, and `make api-docs-freshness` targets.
Workflow, routing, freshness, staging, and documentation responsibilities are
explicitly owned.

Day 7 should inspect existing regression coverage, add the highest-priority
missing local-only closure fixtures, and run the focused validation commands.

### Day 6 Validation

Day 6 changed planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

Commands for Day 6 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

## Day 7: Automation Implementation Batch One

### Scope

Day 7 implements the first stronger local-only guard hardening batch selected
by the Day 6 design. The batch keeps generated API HTML local-only and adds no
publication path, hosted route, artifact upload, or generated-output allowlist.

### Coverage Inspection

| Surface | Day 7 finding | Action |
| --- | --- | --- |
| Routing publication URLs | Existing tests covered hosted project URLs, GitHub release downloads, GitHub Actions run artifacts, generated `docs/api` links, Doxygen paths, provider hosts, custom domains, and unrelated external docs allow cases. | Added a missing regression for GitHub suites artifact URLs because the routing pattern already rejected that retained-artifact shape but had no direct fixture. |
| Maintainer reopening criteria | `docs/maintainer_guide.md` already states that future hosted HTML, retained CI artifacts, or committed generated output must reopen the product decision and validate selected publication policy. | Added that wording to the required routing text contract and added a regression that fails if it is removed. |
| Local-only workflow guard | Existing tests already cover generated API uploads, broad docs publication paths, archive staging, release uploads, dynamic docs paths, tracked/staged generated output, and ignored-output behavior. | No Day 7 local-only shell guard change needed. |

### Implementation

| Path | Change |
| --- | --- |
| `scripts/check_api_docs_routing.py` | Added the future publication reopening criteria phrase to `REQUIRED_TEXT["docs/maintainer_guide.md"]`. |
| `tests/test_api_docs_routing.py` | Added `test_repository_suite_artifact_fails_clearly()` for GitHub suites artifact URLs. |
| `tests/test_api_docs_routing.py` | Added `test_missing_future_publication_reopening_text_fails_clearly()` for maintainer-guide reopening wording drift. |

### Guard Behavior

The routing guard now fails if maintainer documentation stops saying that
future hosted HTML, retained CI artifacts, or committed generated output must
reopen the product decision and validate the selected publication policy before
docs may claim publication.

Repository-hosted retained artifact routes are also covered across release,
Actions run, and suites artifact URL forms.

### Validation

Commands run for Day 7 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

All commands passed. The C/header diff check returned no files.

Day 7 changes Python guard/test code and planning documentation only. No `.c`
or `.h` files are modified, so `make format && make lint && make test` is not
required by the sprint instruction.

### Day 7 Outcome

Items 213.3 and 213.4 are in progress with the first implementation batch
complete. The stronger local-only generated API decision is now backed by an
additional routing contract marker and a retained-artifact URL regression.

Day 8 should continue with any remaining missing local-only fixtures after
rechecking the local-only workflow suite, then preserve the same no-publication
boundary.

## Day 8: Automation Implementation Batch Two

### Scope

Day 8 completes the selected-policy automation surface for workflow staging and
archive checks. The selected policy remains stronger local-only generated API
closure: generated Doxygen HTML stays ignored under `docs/api/`, and no hosted
route, retained artifact, committed generated HTML, workflow upload, or deploy
allowlist is introduced.

### Gap Closed

The local-only workflow guard already rejected single-line staging and archive
commands such as `cp -R docs artifact/`, `mv docs artifact/`, `tar ... docs/`,
and `7z ... docs/` when paired with publication or artifact semantics. Day 8
extends the same checks to each independently folded `run` scalar, preserving
step and field boundaries while preventing folded multiline YAML commands from
splitting the command and `docs/` operand across lines to bypass the guard.

### Implementation

| Path | Change |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | Checks `docs_staging_command_regex` and `docs_archive_command_regex` against normalized line text and independently folded `run` scalar text. |
| `tests/test_api_docs_local_only_guard.py` | Adds `test_workflow_folded_staged_docs_artifact_upload_fails_clearly()`. |
| `tests/test_api_docs_local_only_guard.py` | Adds `test_workflow_folded_archived_docs_artifact_upload_fails_clearly()`. |

### Makefile And Workflow Wiring

No Makefile or real workflow wiring change is needed on Day 8. The existing
`api-docs-freshness` chain remains the selected validation path and current
project workflows still contain no generated API publication route.

### Validation

Commands run for Day 8 closeout:

```sh
python3 tests/test_api_docs_local_only_guard.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

The focused local-only guard suite and `make api-docs-freshness` passed. Final
hygiene checks are recorded in the Day 8 artifact and final turn summary.

Day 8 changes shell/Python guard code and planning documentation only. No `.c`
or `.h` files are modified, so `make format && make lint && make test` is not
required by the sprint instruction.

### Day 8 Outcome

Item 213.3 is functionally complete for the selected stronger local-only
automation path. Item 213.4 now covers the main bypass categories across
routing links, retained artifact URLs, broad workflow paths, dynamic paths,
staging commands, archive commands, and source-controlled API route
preservation.

Day 9 should focus on routing and link validation rather than changing the
selected local-only workflow policy.

## Day 9: Routing And Link Validation

### Scope

Day 9 validates the source-controlled API route and forbidden generated-output
route behavior across Markdown and HTML link forms. The selected policy remains
stronger local-only generated API closure, so no generated API publication URL
or generated-output allowlist is added.

### Route And Link Evidence

| Link class | Current behavior | Day 9 action |
| --- | --- | --- |
| Source-controlled route | `README.md` must route API readers to `docs/api_reference.md`; `docs/api_reference.md` must route to checked-in headers, `Doxyfile`, support matrix, tutorial, cookbook, solver selection, and maintainer guide. | Added an HTML-anchor fixture proving an entity-encoded `href="docs&#x2F;api_reference.md"` still satisfies the required source route. |
| Generated-output local route | Links resolving to `docs/api/` or `docs/api/html/` are rejected after normalization. | Added an HTML-anchor fixture proving `href="docs/%61pi/html/index.html"` is decoded and rejected. |
| Markdown/reference routes | Existing tests cover inline Markdown, reference-style definitions/usages, nested labels, escaped labels, angle-wrapped targets, entities, code fences, comments, image-only route exclusions, and URL-encoded fragments. | No Day 9 change needed. |
| External documentation | Existing tests allow unrelated external docs, Python docs, unrelated ReadTheDocs links, unrelated `pages/usage`, and incidental `api`/`docs/apiary` substrings. | No Day 9 change needed. |
| Hosted/generated publication | Existing tests reject project-looking hosted URLs, provider-hosted publication URLs, release downloads, Actions artifacts, suites artifacts, protocol-relative URLs, and generated API repository paths. | No Day 9 change needed beyond preserving Day 7 suite artifact coverage. |

### Implementation

| Path | Change |
| --- | --- |
| `tests/test_api_docs_routing.py` | Added `test_html_entity_encoded_required_route_is_allowed()`. |
| `tests/test_api_docs_routing.py` | Added `test_html_percent_encoded_href_generated_api_link_fails_clearly()`. |

### Validation

Commands run for Day 9 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

The focused routing suite and `make api-docs-freshness` passed. Final hygiene
checks are recorded in the Day 9 artifact and final turn summary.

Day 9 changes Python regression tests and planning documentation only. No
`.c` or `.h` files are modified, so `make format && make lint && make test` is
not required by the sprint instruction.

### Day 9 Outcome

Item 213.4 is complete for the main routing and guard-test bypass categories.
Required source-controlled API routes remain valid across Markdown and HTML
forms, generated-output targets are rejected after entity/percent decoding, and
valid unrelated external documentation links remain allowed.

Day 10 should move to workflow and staging validation evidence without changing
the selected no-publication policy.

## Day 10: Workflow And Staging Validation

### Scope

Day 10 validates workflow publication, artifact staging, archive commands, and
generated-output references for the selected stronger local-only policy. The
policy remains no hosted generated API HTML, no retained generated-doc
artifact, and no committed generated HTML.

### Workflow Audit

| Audit command | Result |
| --- | --- |
| `find .github/workflows -maxdepth 1 -type f -print | sort` | Current workflow files are `ci.yml`, `macos-ci.yml`, and `windows-ci.yml`. |
| `rg -n "docs/api|api/html|upload-artifact|deploy-pages|upload-pages-artifact|gh-pages|pages|release upload|rclone|aws s3|tar .*docs|zip .*docs|7z .*docs" .github/workflows` | Found existing `actions/upload-artifact@v4` steps, but no generated API paths, Pages deployment steps, generated-doc release uploads, rclone/aws generated-doc publication commands, or docs archive commands. |

Existing artifact uploads remain non-API evidence lanes: selected comparison
freshness, selected performance freshness, dead-code, and coverage artifacts.
None upload `docs/`, `docs/api/`, or `docs/api/html/`.

### Gap Closed

The workflow publication scanner already used quote-aware YAML comment
stripping, but the direct generated-output reference scans still used `sed`.
That meant a quoted `#` before `docs/api/html` could hide a generated API
literal from the direct reference check. Day 10 centralizes the quote-aware
stripping helper and reuses it for direct generated API path scans.

### Implementation

| Path | Change |
| --- | --- |
| `scripts/check_api_docs_local_only.sh` | Added `strip_yaml_comments()` and reused it for direct workflow generated-path scans and publication semantics scans. |
| `tests/test_api_docs_local_only_guard.py` | Added `test_workflow_quoted_hash_before_generated_api_path_fails_clearly()`. |
| `tests/test_api_docs_local_only_guard.py` | Added `test_yaml_workflow_generated_api_path_fails_clearly()` to confirm `.yaml` files are scanned like `.yml`. |

### Validation

Commands run for Day 10 closeout:

```sh
python3 tests/test_api_docs_local_only_guard.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

The focused local-only guard suite and `make api-docs-freshness` passed. Final
hygiene checks are recorded in the Day 10 artifact and final turn summary.

Day 10 changes shell/Python guard code and planning documentation only. No
`.c` or `.h` files are modified, so `make format && make lint && make test` is
not required by the sprint instruction.

### Day 10 Outcome

Workflow behavior matches the selected Day 5 policy. Broad or ambiguous
generated API publication paths remain rejected, direct generated-output
workflow references are checked with quote-aware parsing, `.yaml` workflows are
covered by regression tests, and current project workflows do not publish or
archive generated API output.

Day 11 should update user and maintainer documentation wording for the
stronger local-only closure and repair workflow without adding publication
claims.

## Day 11: User Documentation Update

### Scope

Day 11 updates user-facing documentation for the selected stronger local-only
generated API policy. The docs now say more explicitly where users should link,
when they should run local Doxygen generation, and which generated-output
surfaces remain unsupported.

### Documentation Updates

| Path | User-facing update |
| --- | --- |
| `README.md` | Adds durable-route guidance that rejects CI artifacts, release downloads, Pages deployments, and repository `docs/api/` paths as API documentation routes. |
| `INSTALL.md` | Adds support-matrix follow-up wording that generated API HTML is an on-demand local view, not an install, release, hosted, or artifact publication surface. |
| `docs/api_reference.md` | Adds durable-link guidance: use the API reference and checked-in public headers, regenerate local HTML only for the current checkout, and do not substitute CI artifacts, release assets, or hosted pages. |

### Guard Markers

| Marker owner | New required wording |
| --- | --- |
| `scripts/check_api_docs_routing.py` | `docs/api_reference.md` must retain durable-link wording for source-controlled API reference/public headers. |
| `scripts/check_api_docs_routing.py` | `README.md` must retain wording that rejects CI artifacts, release downloads, Pages deployments, and repository paths as the API docs route. |
| `scripts/check_api_docs_routing.py` | `INSTALL.md` must retain wording that generated API HTML is an on-demand local view, not install/release/hosted/artifact publication. |
| `tests/test_api_docs_routing.py` | Missing-marker regressions cover all three new user documentation markers. |

### Validation

Commands run for Day 11 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

The focused routing suite and `make api-docs-freshness` passed. Final hygiene
checks are recorded in the Day 11 artifact and final turn summary.

Day 11 changes Markdown documentation and Python routing guard/tests only. No
`.c` or `.h` files are modified, so `make format && make lint && make test` is
not required by the sprint instruction.

### Day 11 Outcome

Item 213.5 is complete for user-facing documentation. Users can tell that
`docs/api_reference.md` and checked-in public headers are the durable API
route, `make api-docs-freshness` creates only a local current-checkout Doxygen
view, and generated HTML is not hosted, retained, committed, or release
evidence.

Day 12 should finish maintainer-facing repair workflow and planning/residual
status updates.

## Day 12: Maintainer Documentation And Residuals

### Scope

Day 12 completes maintainer-facing documentation for the selected stronger
local-only generated API policy. It adds repair workflow, expected artifacts,
and residual status without changing the no-publication decision.

### Maintainer Guide Update

| Topic | Day 12 wording |
| --- | --- |
| Repair workflow | Maintainers should reproduce with `make api-docs-freshness`, isolate Doxygen/coverage/local-only/routing failures, and repair source docs, Doxygen comments, workflow paths, or guard fixtures first. |
| Expected artifacts | Repair artifacts are regenerated local `docs/api/html/` output plus passing `api-docs-coverage`, `api-docs-local-only`, and `api-docs-routing`; they are not retained generated-doc artifacts. |
| Residual options | Hosted HTML, retained generated-doc artifacts, and committed generated HTML remain unclaimed future options requiring exact hosting, retention, freshness, routing, rollback, and claim-boundary evidence. |

### Project Plan Update

The Sprint 213 project-plan section now records the current branch direction:
stronger local-only generated API closure is selected. It also names the
remaining unclaimed future options and the evidence needed before a later
sprint can claim them.

### Guard Markers

| Marker owner | New required wording |
| --- | --- |
| `scripts/check_api_docs_routing.py` | `docs/maintainer_guide.md` must retain `Generated API local-only repair workflow:`. |
| `scripts/check_api_docs_routing.py` | `docs/maintainer_guide.md` must retain expected repair artifacts wording for regenerated local `docs/api/html/` output and passing checks. |
| `scripts/check_api_docs_routing.py` | `docs/maintainer_guide.md` must retain remaining unclaimed generated API publication options wording. |
| `tests/test_api_docs_routing.py` | Missing-marker regressions cover all three maintainer/residual markers. |

### Validation

Commands run for Day 12 closeout:

```sh
python3 tests/test_api_docs_routing.py
make api-docs-freshness
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

The focused routing suite and `make api-docs-freshness` passed. Final hygiene
checks are recorded in the Day 12 artifact and final turn summary.

Day 12 changes Markdown documentation and Python routing guard/tests only. No
`.c` or `.h` files are modified, so `make format && make lint && make test` is
not required by the sprint instruction.

### Day 12 Outcome

Item 213.5 is complete for maintainer and planning documentation. Residual
status now names only genuine future generated API publication work, and
maintainers have exact commands and triage surfaces for selected-policy
failures.

Day 13 should focus on integrated validation and review hardening without
changing the selected local-only policy.

## Day 13: Integrated Validation

### Scope

Day 13 validates the selected stronger local-only generated API policy as an
integrated chain. It does not change the Day 5 product decision: generated API
HTML remains ignored local output, and hosted generated API HTML, retained
generated-doc artifacts, and committed generated HTML remain unclaimed future
options.

### Integrated Validation Matrix

| Command | Result | Evidence |
| --- | --- | --- |
| `make docs-check` | Pass | Doxygen regenerated `docs/api/html/`; `api-docs-coverage` passed with 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` retained as a separate generated-header policy row. |
| `python3 tests/test_api_docs_coverage.py` | Pass | Standalone coverage regression suite completed without failure. |
| `python3 tests/test_api_docs_local_only_guard.py` | Pass | Local-only workflow, staging, publication, folded-command, quoted-comment, and `.yaml` regressions completed without failure. |
| `python3 tests/test_api_docs_routing.py` | Pass | Routing, required-route, publication-link, encoded-link, documentation-marker, and residual-option regressions completed without failure. |
| `make api-docs-freshness` | Pass | Serialized docs generation, coverage, local-only, and routing validation completed; routing checked seven documents and confirmed generated API publication links absent. |
| `git diff --check` | Pass | No whitespace errors in the final Day 13 diff after writing the notes and artifact. |
| `git diff --name-only -- '*.c' '*.h'` | Pass | No C source or header files are modified. |
| `git status --short --branch` | Informational | Branch remains `sprint-213` with Sprint 213 documentation, generated API guard, and regression-test changes pending. |

### Changed-Surface Summary

Day 13 adds validation evidence only. The functional and documentation surfaces
validated by the integrated pass are the Day 7-12 changes:

| Surface | Day 13 validation relevance |
| --- | --- |
| `scripts/check_api_docs_routing.py` and `tests/test_api_docs_routing.py` | Required source-controlled route markers, generated/hosted publication link rejection, encoded link handling, and maintainer residual markers remain covered by standalone and aggregate routing checks. |
| `scripts/check_api_docs_local_only.sh` and `tests/test_api_docs_local_only_guard.py` | Folded workflow staging/archive checks, quote-aware workflow scanning, `.yaml` workflow coverage, and generated-output publication rejection remain covered by standalone and aggregate local-only checks. |
| `README.md`, `INSTALL.md`, `docs/api_reference.md`, and `docs/maintainer_guide.md` | User and maintainer documentation keeps the stronger local-only route and non-publication boundary required by the routing and local-only guards. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 213 planning status records stronger local-only closure and leaves hosted HTML, retained generated-doc artifacts, and committed generated HTML as unclaimed future options. |

### Full C Gate Decision

`git diff --name-only -- '*.c' '*.h'` produced no files. Day 13 therefore does
not run `make format && make lint && make test` under the sprint instruction,
which requires that full gate only when C source or public/internal header files
are modified.

### Day 13 Outcome

Item 213.6 is complete for integrated validation. The selected local-only
generated API policy passes documentation generation, page coverage, workflow
local-only checks, route/link checks, and aggregate freshness validation.

Day 14 should perform final closeout review, reconcile the Sprint 213 notes and
artifacts, and prepare retrospective inputs without changing the selected
generated API publication policy.

## Day 14: Closeout Review

### Final Policy Status

Sprint 213 closes with **stronger local-only generated API closure**.
Generated Doxygen HTML remains ignored local output under `docs/api/html/`;
`docs/api_reference.md` and checked-in public headers remain the
source-controlled API route.

The sprint does not publish generated API HTML, retain generated-doc CI
artifacts, or commit generated HTML.

### Final Item Reconciliation

| Epic item | Final status | Evidence |
| --- | --- | --- |
| 213.1 Publication Option Review | Complete | Days 1-3 inventory the current local-only baseline and compare stronger local-only, hosted Pages, retained artifact, and committed generated HTML options. |
| 213.2 Policy Decision | Complete | Days 4-5 define criteria and select stronger local-only generated API closure. |
| 213.3 Automation Implementation | Complete | Days 7-8 harden routing required-text checks and folded workflow staging/archive detection for the selected policy. |
| 213.4 Routing And Guard Tests | Complete | Days 7-10 add generated-output, hosted/publication, encoded-link, folded-command, quote-aware workflow, and `.yaml` workflow regressions. |
| 213.5 User And Maintainer Docs | Complete | Days 11-12 update README, INSTALL, API reference, maintainer guide, and Epic 19 status/residual wording. |
| 213.6 Validation And Closeout | Complete | Days 13-14 record integrated validation, final status reconciliation, residual options, and the no-C/header full-gate decision. |

### Final Evidence

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

### Final Validation Summary

Closeout validation passed:

- `make docs-check`;
- `python3 tests/test_api_docs_coverage.py`;
- `python3 tests/test_api_docs_local_only_guard.py`;
- `python3 tests/test_api_docs_routing.py`;
- `make api-docs-freshness`;
- `git diff --check`;
- `git diff --name-only -- '*.c' '*.h'`.

No `.c` or `.h` files changed during Sprint 213, so the full C quality gate
`make format && make lint && make test` is not required by the sprint rule.

### Residual Risks And Non-Claims

Sprint 213 leaves these claims unearned:

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

### Retrospective Handoff

The retrospective should treat Sprint 213 as a completed policy-closure sprint.
The selected outcome is stronger local-only generated API closure, not
publication. The main implementation surfaces are workflow/staging guard
hardening, routing/link guard hardening, user documentation, maintainer repair
guidance, and Epic 19 residual alignment.

### Day 14 Outcome

Sprint 213 is ready for retrospective preparation. The branch closes the
generated API publication decision with stronger local-only automation,
documentation, and validation guards without overstating current generated API
publication support.
