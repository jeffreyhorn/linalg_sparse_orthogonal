# Day 8: Validation Scope Design

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Define the focused and broad validation matrix required by the
changed Sprint 206 surfaces before running final gates.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 8 designs the validation path for Sprint 206. It does not promote any new
support claim and does not replace the Day 9-Day 10 validation runs. The goal
is to make the remaining checks explicit, proportional to the current branch
diff, and tied to the claim surfaces changed by Days 5 through 7.

The current branch changes are documentation and planning only:

- `README.md`
- `INSTALL.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_18/PROJECT_PLAN.md`
- `docs/planning/EPIC_18/SPRINT_206/PLAN.md`
- `docs/planning/EPIC_18/SPRINT_206/WORKING_NOTES.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day1-closeout-intake.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day2-outcome-reconciliation.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day3-claim-surface-audit.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day4-project-plan-status-design.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day5-public-claim-update.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day6-maintainer-claim-update.md`
- `docs/planning/EPIC_18/SPRINT_206/artifacts/day7-project-plan-status-implementation.md`
- this Day 8 artifact

No `.c` or `.h` files, workflow files, guard scripts, manifests, schemas,
Makefile rules, CMake files, benchmark sources, examples, or tests are changed
by Sprint 206 through Day 8.

## Changed-Surface Validation Map

| Changed surface | Validation command | Run day | Requirement |
| --- | --- | --- | --- |
| `README.md` support/package/API wording | `make support-docs-guard` | Day 9 focused validation | Required because README carries public support and adoption boundaries. |
| `README.md` package-manager non-claim wording | `bash scripts/package_manager_deferral_check.sh` | Day 9 focused validation | Required because Sprint 198 local Homebrew proof must remain a local proof only. |
| `INSTALL.md` support/readiness matrix and package evidence owner | `make support-docs-guard` | Day 9 focused validation | Required because `INSTALL.md#support-readiness-matrix` is the public support truth. |
| `INSTALL.md` static/package/ABI boundary wording | `bash scripts/static_package_deferral_check.sh` | Day 9 focused validation | Required because install text must not imply dynamic ABI or package-provider support. |
| `docs/api_reference.md` generated API local-only wording | `make api-docs-freshness` | Day 9 focused validation; repeat on Day 10 if needed | Required because the API route and local-only generated HTML policy changed. |
| `docs/maintainer_guide.md` generated API, package, selected-comparison, and support ownership wording | `make api-docs-freshness`; `make support-docs-guard`; package deferral scripts | Day 9 focused validation | Required because the maintainer guide is a claim-boundary owner for these surfaces. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` current status snapshot | `git diff --check`; stale-wording `rg` checks | Day 9 focused validation | Required because current Epic 18 status changed and must not preserve stale Sprint 197-only closeout wording. |
| Sprint 206 planning artifacts | `git diff --check`; artifact existence checks | Day 9 focused validation | Required because they are new source-controlled evidence. |

## Focused Validation Commands

Day 9 should run these focused commands in this order:

```sh
git diff --check
make support-docs-guard
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
make api-docs-freshness
```

Day 9 should also run focused stale-wording checks against
`docs/planning/EPIC_18/PROJECT_PLAN.md`:

```sh
rg -n 'Sprint 197 Day 8 Interim Status Snapshot|Requested final-validation evidence recorded through `SPRINT_197`|In progress with numbering caveat' docs/planning/EPIC_18/PROJECT_PLAN.md
```

That `rg` command is expected to return no matches. A non-zero exit caused by
no matches is a pass for this specific stale-wording check.

## Broad Validation Commands

Day 10 should run the broad documentation and project hygiene gates:

```sh
make docs-check
make api-docs-freshness
git status --short
git status --ignored --short docs/api
```

`make api-docs-freshness` already depends on the API docs routing and
local-only checks, so it is the broad generated API policy gate. It is safe for
the command to leave ignored generated output under `docs/api/`; the required
closeout check is that generated output remains ignored and unstaged.

## Conditional Checks

| Trigger | Required command | Day 8 disposition |
| --- | --- | --- |
| Any `.c` or `.h` file changes before closeout | `make format && make lint && make test` | Not currently triggered; no C or header files are changed through Day 8. |
| Any Makefile, CMake, workflow, guard script, manifest, schema, benchmark, example, or test changes before closeout | Re-map focused checks to the changed owner surface and run the relevant standalone tests. | Not currently triggered through Day 8. |
| Any selected report manifest or selected comparison wording change before closeout | `python3 scripts/validate_corpus_schema.py`; `python3 tests/test_selected_report_targets_manifest.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_selected_performance_docs.py`; `make report-index-comparison-freshness` if generated selected comparison freshness is affected. | Not currently triggered by the Day 5-Day 8 diff. |
| Any Windows support or PowerShell claim wording change before closeout | `make windows-powershell-guard` | Not currently triggered by the Day 5-Day 8 diff. |
| Any QR public-header documentation change before closeout | `make qr-header-docs-guard` | Not currently triggered by the Day 5-Day 8 diff. |

## Environment Blocker Rules

A command may be skipped only if the blocker is explicit and reproducible.
Acceptable blocker records must include:

- exact command attempted;
- local platform/toolchain detail relevant to the failure;
- whether the failure is environmental, dependency-related, or caused by the
  branch diff;
- fallback evidence that still checks the claim boundary where possible;
- the follow-up owner day or residual queue entry.

Convenience, long runtime alone, or lack of immediate need is not a blocker.
If `.c` or `.h` files are changed later in Sprint 206, the full
`make format && make lint && make test` gate becomes mandatory before closeout.

## Generated Output Hygiene

`make docs-check` and `make api-docs-freshness` may regenerate local Doxygen
HTML under `docs/api/`. That output is intentionally ignored. Day 9-Day 14
validation should verify:

```sh
git status --ignored --short docs/api
```

Expected acceptable result:

```text
!! docs/api/
```

Any staged or unignored generated API file is a failure unless a later sprint
explicitly changes the generated API publication decision.

## Claim Boundaries Preserved By This Plan

This validation plan does not claim:

- broad package-manager distribution, Homebrew/core readiness, bottles,
  Linuxbrew, public tap maintenance, or binary package support;
- broad Windows support, selected Windows Cholesky promotion, or selected
  Windows QR incompatible promotion;
- broad allocation-failure guarantees;
- repository-wide review-surface cleanup;
- hosted generated API publication, retained generated-doc artifacts, or
  committed generated HTML;
- shared-library or dynamic ABI support;
- portable performance, release readiness, external-library parity, or
  state-of-the-art status.

## Day 8 Validation And Hygiene

| Check | Day 8 result |
| --- | --- |
| Changed surface inventory | Complete. Current branch changes are documentation and Sprint 206 planning artifacts only. |
| C/header quality gate trigger review | Complete. No `.c` or `.h` files are changed through Day 8, so the full C gate is not required yet. |
| Focused validation matrix | Complete. Day 9 commands are mapped to public support, package deferral, static package/ABI deferral, generated API freshness, and project-plan stale wording. |
| Broad validation matrix | Complete. Day 10 commands are mapped to docs generation/API freshness and generated-output hygiene. |
| Environment blocker rules | Complete. Skips require explicit blocker evidence and fallback handling. |
| `git diff --check` | Passed after Day 8 edits. |
| Stale project-plan wording search | Passed. The old Sprint 197 interim heading, old Sprint 206 `SPRINT_197`-only status row, and old `In progress with numbering caveat` project-plan row are absent from `PROJECT_PLAN.md`. |
| Generated-output status | Existing ignored `docs/api/` output remains ignored and unstaged. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 206.4 has a complete validation plan before final gates run. | Met. This artifact defines focused, broad, conditional, and blocker-driven validation. |
| Validation scope is proportional to changed surfaces. | Met. Current required checks are documentation, package-deferral, API-docs, and planning-status checks; full C gates are conditional on code/header edits. |
| Any skipped command requires an explicit blocker, not convenience. | Met. Blocker evidence requirements are documented above. |

## Day 8 Disposition

Day 8 is complete after the Day 8 artifact and working notes pass patch
hygiene. Day 9 should run the focused validation commands listed above and fix
any failures within Sprint 206 scope.
