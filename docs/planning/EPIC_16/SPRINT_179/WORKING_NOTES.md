# Sprint 179 Working Notes

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Source Artifact Note

The Sprint 179 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 179: Generated API HTML
Publication Decision". Sprint 179 artifacts in this directory follow the Epic
16 scope.

## Sprint Goal

Close generated API HTML status with either a hosted publication path or a
stronger enforced local-only product decision.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day6-populated-matrix.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day8-gate-templates.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day12-handoff-package.md`
- `docs/planning/EPIC_16/SPRINT_178/artifacts/day14-closeout.md`
- `Doxyfile`
- `Makefile`
- `scripts/check_api_docs_coverage.py`
- `scripts/check_api_docs_local_only.sh`
- `docs/api_reference.md`
- `README.md`
- `docs/maintainer_guide.md`
- `.gitignore`

## Starting Branch Snapshot

- Branch: `sprint-179`
- Starting commit: `17754f05face0b7a9c82810790c11f430ef2d6c8`
- Recent base context:
  - `17754f05` Merge pull request #198 from `sprint-178`
  - `a7d58196` Complete Sprint 178 allocation-failure proof
  - `3907e754` Merge pull request #197 from `sprint-177`
  - `4bca0a10` Address PR #197 review comments
  - `aad776d9` Move Sprint 177 planning artifacts to Epic 16

## Sprint 179 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 179.1 | Doxygen Surface Audit | Complete | Day 2 completes configured input/output inventory; Day 3 completes warning and page-coverage readiness evidence. |
| 179.2 | Publication Decision | Complete | Day 6 formally selects strengthened local-only generated API HTML status. |
| 179.3 | Implementation | Complete | Day 9 completes local-only enforcement implementation with wording, file-existence, and workflow-publication checks. |
| 179.4 | Freshness and Staging Guard | Complete | Day 10 adds Doxyfile input/output contract checks to the local-only freshness guard. |
| 179.5 | Navigation Update | Complete | Day 11 updates README, API reference, and maintainer guide wording for the Sprint 179 local-only decision. |
| 179.6 | Verification | Complete | Day 13 completes integrated validation and claim reconciliation. |

## Current Generated API HTML Baseline

| Surface | Current state |
| --- | --- |
| Doxygen input | `Doxyfile` reads `include/` with `FILE_PATTERNS = *.h` and `RECURSIVE = NO`. |
| Doxygen output | `OUTPUT_DIRECTORY = docs/api`, `GENERATE_HTML = YES`, and `HTML_OUTPUT = html`, so local HTML lands under `docs/api/html/`. |
| Generated output policy | `.gitignore` ignores `docs/api/`; `git ls-files docs/api` is empty at Day 1 baseline. |
| Make docs target | `make docs` runs `doxygen Doxyfile` and reports generated output in `docs/api/html/`. |
| Coverage check | `make docs-check` runs `docs` and `scripts/check_api_docs_coverage.py`. |
| Local-only check | `make api-docs-local-only` runs `scripts/check_api_docs_local_only.sh`. |
| Selected freshness target | `make api-docs-freshness` runs `api-docs-validate`, which combines docs generation, coverage, and local-only staging checks. |
| User entry point | `docs/api_reference.md` remains the compact user-facing API reference entry point. |
| Maintainer wording | `docs/maintainer_guide.md` says `docs/api/html/` is local-only generated output and not committed, hosted, artifact-published, or release evidence. |

## Doxygen Surface Inventory

Day 2 confirms the configured Doxygen surface is narrow and header-owned:

- configured input path: `include/`;
- configured file pattern: `*.h`;
- configured recursion: `NO`;
- checked-in public headers under the configured input path: 18;
- nested public headers under `include/`: none;
- configured output directory: `docs/api`;
- configured HTML output directory: `docs/api/html`;
- local generated files currently present under `docs/api/html/`: 214;
- generated header reference pages currently present: 18;
- generated header source pages currently present: 18;
- examples, tutorials, cookbook pages, maintainer docs, planning docs, and
  generated install headers are not configured Doxygen inputs.

The checked-in public headers that currently own the generated API surface are:

| Header | Generated reference page | Generated source page |
| --- | --- | --- |
| `include/sparse_analysis.h` | `sparse__analysis_8h.html` | `sparse__analysis_8h_source.html` |
| `include/sparse_bidiag.h` | `sparse__bidiag_8h.html` | `sparse__bidiag_8h_source.html` |
| `include/sparse_cholesky.h` | `sparse__cholesky_8h.html` | `sparse__cholesky_8h_source.html` |
| `include/sparse_csr.h` | `sparse__csr_8h.html` | `sparse__csr_8h_source.html` |
| `include/sparse_dense.h` | `sparse__dense_8h.html` | `sparse__dense_8h_source.html` |
| `include/sparse_eigs.h` | `sparse__eigs_8h.html` | `sparse__eigs_8h_source.html` |
| `include/sparse_ic.h` | `sparse__ic_8h.html` | `sparse__ic_8h_source.html` |
| `include/sparse_ilu.h` | `sparse__ilu_8h.html` | `sparse__ilu_8h_source.html` |
| `include/sparse_iterative.h` | `sparse__iterative_8h.html` | `sparse__iterative_8h_source.html` |
| `include/sparse_ldlt.h` | `sparse__ldlt_8h.html` | `sparse__ldlt_8h_source.html` |
| `include/sparse_lu.h` | `sparse__lu_8h.html` | `sparse__lu_8h_source.html` |
| `include/sparse_lu_csr.h` | `sparse__lu__csr_8h.html` | `sparse__lu__csr_8h_source.html` |
| `include/sparse_matrix.h` | `sparse__matrix_8h.html` | `sparse__matrix_8h_source.html` |
| `include/sparse_qr.h` | `sparse__qr_8h.html` | `sparse__qr_8h_source.html` |
| `include/sparse_reorder.h` | `sparse__reorder_8h.html` | `sparse__reorder_8h_source.html` |
| `include/sparse_svd.h` | `sparse__svd_8h.html` | `sparse__svd_8h_source.html` |
| `include/sparse_types.h` | `sparse__types_8h.html` | `sparse__types_8h_source.html` |
| `include/sparse_vector.h` | `sparse__vector_8h.html` | `sparse__vector_8h_source.html` |

## Current Claim Baseline

README currently lists:

- `make docs` for Doxygen API reference generation;
- `make docs-check` for local Doxygen page coverage;
- `make api-docs-freshness` for selected local Doxygen freshness plus
  local-only staging guard;
- `docs/api_reference.md` as the API reference entry point.

`docs/api_reference.md` currently says:

- checked-in public headers under `include/` are the source of truth for API
  declarations and call-site contracts;
- generated HTML under `docs/api/html/` is local-only generated output;
- the generated tree is ignored and not a hosted or source-controlled
  publication surface;
- generated install headers such as `sparse_version.h` are not expected
  Doxygen pages under the current configured input set.

`docs/maintainer_guide.md` currently says:

- `docs/api_reference.md` is the user-facing API reference entry point;
- generated API HTML is refreshed with `make api-docs-freshness`;
- generated HTML is current only for the branch and checkout where the selected
  freshness command just passed;
- local generated output under `docs/api/html/` is not source-controlled,
  hosted, artifact-published, or release evidence.

## Day 1 Decisions

- Treat `docs/planning/EPIC_16/PROJECT_PLAN.md` as the Sprint 179 source
  authority.
- Use Sprint 177 Gate 2 as the closeout acceptance contract for generated API
  HTML product status.
- Use Sprint 177 Day 12 and Sprint 178 Day 14 as the immediate handoff trail.
- Do not select hosted publication, retained CI artifact, committed generated
  output, or stronger local-only status on Day 1.
- Keep the current local-only Doxygen behavior as baseline evidence, not as the
  final Sprint 179 product decision.

## Open Risks

- Hosted publication may require repository settings, credentials, or
  infrastructure that are not represented in the current source tree.
- Committed generated output would conflict with the current ignored
  `docs/api/` policy unless the product decision explicitly changes that
  policy.
- Retained CI artifact publication would require workflow updates and
  fail-closed artifact metadata checks.
- Keeping local-only status without stronger wording or guards may leave the
  discoverability gap identified in the Epic 16 review.

## Daily Log

### Day 1 - Sprint Intake And Evidence Baseline

Status: Complete

Completed:

- Re-read the Sprint 179 project-plan section.
- Reviewed Sprint 177 evidence rows ESM-005 and ESM-011.
- Reviewed Sprint 177 Gate 2 and Day 12 Sprint 179 handoff.
- Created Sprint 179 working notes and artifact directory structure.
- Recorded current generated API claim surface in README, API reference, and
  maintainer guide.
- Recorded current Doxygen configuration, output location, ignored output path,
  and local-only guard behavior.
- Created the Day 1 evidence-baseline artifact.

Validation:

- `git diff --check`

### Day 2 - Doxygen Input And Output Audit

Status: Complete

Completed:

- Inspected `Doxyfile` input, output, extraction, warning, and HTML settings.
- Mapped all configured public header inputs under `include/`.
- Confirmed no nested include headers are currently included or excluded by the
  non-recursive Doxygen configuration.
- Mapped current generated reference and source pages under `docs/api/html/`.
- Confirmed examples, tutorial/cookbook docs, maintainer docs, planning docs,
  and generated install headers are not Doxygen inputs.
- Confirmed `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` are
  ignored by `.gitignore:40:docs/api/`.
- Confirmed generated API files remain untracked, unstaged, and invisible as
  non-ignored untracked files.
- Created the Day 2 Doxygen surface audit artifact.

Validation:

- `python3 scripts/check_api_docs_coverage.py`
- `bash scripts/check_api_docs_local_only.sh`
- `git diff --check`

### Day 3 - Warning And Coverage Audit

Status: Complete

Completed:

- Ran `make docs-check` and captured the generated API command path.
- Ran `make api-docs-freshness` to validate the combined generation,
  coverage, and local-only staging path.
- Recorded that the captured Doxygen run emitted no warning lines.
- Confirmed coverage for all 18 configured public headers, with 18 generated
  reference pages and 18 generated source pages.
- Recorded that examples, tutorial/cookbook docs, solver-selection docs,
  maintainer docs, and install/version generated headers remain outside the
  current generated HTML input set.
- Separated publication blockers from polish and non-input risks.
- Created the Day 3 warning and coverage audit artifact.

Validation:

- `make docs-check`
- `make api-docs-freshness`
- `git diff --check`

### Day 4 - Current Guard And CI Audit

Status: Complete

Completed:

- Inspected Make targets that generate and validate generated API docs.
- Inspected `scripts/check_api_docs_coverage.py` and
  `scripts/check_api_docs_local_only.sh`.
- Inspected GitHub Actions workflows for generated API docs validation,
  artifact upload, Pages deployment, and publication metadata.
- Recorded that current generated API freshness is local-only through
  `make api-docs-freshness`, not hosted CI evidence.
- Recorded that staged generated files are rejected by the local-only guard.
- Recorded that CI currently has no generated API HTML upload, retained
  artifact, hosted publication, Pages deployment, or publication metadata path.
- Created the Day 4 guard and CI audit artifact.

Validation:

- `make api-docs-freshness`
- `git diff --check`

### Day 5 - Publication Option Decision Matrix

Status: Complete

Completed:

- Defined decision criteria for user value, maintenance cost, reviewability,
  freshness, reproducibility, and CI complexity.
- Evaluated hosted generated API HTML publication.
- Evaluated retained CI artifact publication.
- Evaluated committed generated HTML output.
- Evaluated stronger local-only status.
- Recommended stronger local-only status as the Day 6 decision candidate.
- Recorded rejected-option rationale and required follow-through if Day 6
  accepts the recommendation.
- Created the Day 5 publication decision matrix artifact.

Validation:

- `git diff --check`

### Day 6 - Product Decision Record

Status: Complete

Completed:

- Formally selected strengthened local-only generated API HTML status.
- Documented rejected alternatives: hosted publication, retained CI artifact,
  and committed generated output.
- Defined implementation acceptance requirements for freshness, staging,
  navigation, and guard behavior.
- Defined supported claims after the decision.
- Defined unsupported claims that documentation must not imply.
- Created the Day 6 product decision record artifact.

Validation:

- `git diff --check`

### Day 7 - Implementation Design

Status: Complete

Completed:

- Identified implementation owner files for the strengthened local-only path.
- Confirmed workflow edits are out of scope unless a future sprint selects
  hosted or artifact publication.
- Defined command names, no artifact names, and no hosted metadata paths for
  the selected local-only status.
- Designed the local freshness and staging verification path around
  `make api-docs-freshness`.
- Designed documentation navigation updates for README, API reference, and the
  maintainer guide.
- Designed a focused local-only product-status guard for later implementation.
- Created the Day 7 implementation design artifact.

Validation:

- `git diff --check`

### Day 8 - Core Implementation Batch

Status: Complete

Completed:

- Extended `scripts/check_api_docs_local_only.sh` with
  `require_file_contains()` and product-status wording checks.
- Made `api-docs-local-only` fail if README, API reference, or maintainer guide
  wording stops preserving the strengthened local-only generated API HTML
  product decision.
- Updated `docs/maintainer_guide.md` from the historical Sprint 158 wording to
  the Sprint 179 product decision.
- Preserved `docs/api/` as ignored generated output and did not edit generated
  HTML.
- Captured early validation output for the direct guard and full freshness
  target.
- Created the Day 8 implementation artifact.

Validation:

- `bash scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `git diff --check`

### Day 9 - Enforcement Completion

Status: Complete

Completed:

- Added explicit checked-file existence failures to
  `scripts/check_api_docs_local_only.sh`.
- Added workflow-publication path checks that reject `.github/workflows`
  references to `docs/api/html` or `docs/api/` while Sprint 179 keeps generated
  API HTML strengthened local-only.
- Confirmed `make api-docs-freshness` still runs from the repository root
  through checked-in `Makefile` and script owners.
- Confirmed generated API output remains ignored, untracked, unstaged, and not
  referenced by workflows as a publication/upload path.
- Recorded remaining risks and deferrals for final guard hardening.
- Created the Day 9 enforcement completion artifact.

Validation:

- `bash -n scripts/check_api_docs_local_only.sh`
- `bash scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `git diff --check`

### Day 10 - Freshness And Staging Guard

Status: Complete

Completed:

- Added Doxyfile contract checks to `scripts/check_api_docs_local_only.sh`.
- Made the local-only guard fail if Doxygen input/output configuration drifts
  away from the strengthened local-only product decision.
- Confirmed the guard checks `INPUT`, `FILE_PATTERNS`, `RECURSIVE`,
  `OUTPUT_DIRECTORY`, `GENERATE_HTML`, and `HTML_OUTPUT`.
- Confirmed missing-page coverage remains owned by
  `scripts/check_api_docs_coverage.py` through `make api-docs-freshness`.
- Confirmed staged/tracked/non-ignored generated API file rejection still
  passes.
- Created the Day 10 freshness and staging guard artifact.

Validation:

- `bash -n scripts/check_api_docs_local_only.sh`
- `bash scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `git diff --check`

### Day 11 - Navigation And Claim Update

Status: Complete

Completed:

- Updated `README.md` to name `docs/api_reference.md` plus public headers as
  the supported source-controlled API documentation path.
- Updated `README.md` to state that generated Doxygen HTML is a Sprint 179
  local-only convenience view, not hosted documentation, retained CI artifact,
  source-controlled output, or release evidence.
- Updated `docs/api_reference.md` with the Sprint 179 local-only product
  decision and source-controlled API reference path.
- Updated `docs/maintainer_guide.md` with generated API `local_only` support
  tier wording.
- Confirmed the strengthened local-only guard still passes after the docs
  changes.
- Created the Day 11 navigation and claim update artifact.

Validation:

- `bash -n scripts/check_api_docs_local_only.sh`
- `bash scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `git diff --check`

### Day 12 - Focused Verification

Status: Complete

Completed:

- Ran Doxygen generation through `make docs`.
- Ran generated API docs coverage through `make docs-check`.
- Ran the strengthened local-only shell guard syntax check.
- Ran the strengthened local-only shell guard directly.
- Ran the aggregate `make api-docs-freshness` target.
- Confirmed generated API files are not tracked, staged, or visible as
  non-ignored untracked files.
- Ran whitespace validation with `git diff --check`.
- Created the Day 12 focused verification artifact.

Validation:

- `make docs`
- `make docs-check`
- `bash -n scripts/check_api_docs_local_only.sh`
- `bash scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `python3 scripts/check_api_docs_coverage.py`
- `git ls-files docs/api`
- `git diff --cached --name-only -- docs/api`
- `git ls-files --others --exclude-standard docs/api`
- `git diff --check`

### Day 13 - Integrated Validation And Reconciliation

Status: Complete

Completed:

- Re-ran the selected generated API validation chain after documentation
  updates.
- Inspected changed files and confirmed no generated API HTML files are
  tracked, staged, or visible as non-ignored untracked output.
- Reconciled README, API reference, maintainer guide, script, and workflow
  wording against the Sprint 179 strengthened local-only decision.
- Confirmed all Sprint 179 project-plan items have evidence artifacts.
- Recorded residual risks and explicit deferrals before Day 14 closeout.
- Created the Day 13 integrated validation artifact.

Validation:

- `make api-docs-freshness`
- `bash -n scripts/check_api_docs_local_only.sh`
- `git ls-files docs/api`
- `git diff --cached --name-only -- docs/api`
- `git ls-files --others --exclude-standard docs/api`
- `git diff --check`

### Day 14 - Sprint Closeout And Handoff

Status: Complete

Completed:

- Finalized Sprint 179 working notes and marked the sprint complete.
- Summarized the completed Sprint 179 deliverables and validation evidence.
- Captured the generated API HTML product decision in closeout-ready form.
- Recorded follow-up work and explicit deferrals for later Epic 16 sprints.
- Prepared retrospective inputs covering completed work, residual risks, and
  lessons.
- Created the Day 14 closeout and handoff artifact.

Validation:

- `bash -n scripts/check_api_docs_local_only.sh`
- `make api-docs-freshness`
- `git diff --check`
