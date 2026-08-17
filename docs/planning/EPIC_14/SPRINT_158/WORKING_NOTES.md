# Sprint 158 Working Notes

## Goal

Sprint 158 closes the generated API reference residual with either committed
generated HTML and coverage evidence or an explicit no-commit product decision
with a recurring freshness guard.

## Starting Evidence

- Sprint 157 selected T157-01/C157-01 as the generated API reference target.
- Sprint 157 Day 9 defines the API documentation publication evidence
  contract: `make docs`, Doxygen warning log, generated-page inventory,
  page-coverage check, generated `sparse_version.h` policy, and publication
  decision.
- Sprint 157 Day 10 defines generated API docs validation: `make docs`,
  warning triage, page coverage, `git diff --check`, and the full C/header gate
  if public headers change.
- Sprint 157 Day 12 and Day 14 define the Sprint 158 stop conditions and
  handoff.
- Current repository policy keeps `docs/api/` ignored unless Sprint 158
  explicitly changes the generated-output publication decision.

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-158` |
| Starting commit | `ab98ce4d32173617de3dc009e0a07e446d157042` |
| Starting commit summary | `ab98ce4d Merge pull request #175 from jeffreyhorn/sprint-157` |
| Upstream state | created from current `master` after PR #175 merge |
| Sprint plan path | `docs/planning/EPIC_14/SPRINT_158/PLAN.md` |
| Project-plan source | `docs/planning/EPIC_14/PROJECT_PLAN.md`, Sprint 158 |

## Day 1 Tool And Generated-Doc State

| Surface | Day 1 state |
| --- | --- |
| Doxygen binary | `/usr/local/bin/doxygen` |
| Doxygen version | `1.16.1` |
| Documentation target | `make docs` runs `doxygen Doxyfile` and reports `docs/api/html/`. |
| Doxygen input | `include/` with `FILE_PATTERNS = *.h` and `RECURSIVE = NO`. |
| Doxygen output | `OUTPUT_DIRECTORY = docs/api`, `HTML_OUTPUT = html`, `GENERATE_HTML = YES`. |
| Warning policy | `WARN_IF_UNDOCUMENTED = YES`, `WARN_IF_DOC_ERROR = YES`, `WARN_NO_PARAMDOC = YES`, `WARN_AS_ERROR = NO`, `QUIET = YES`. |
| Generated output tracking | `git status --ignored=matching --short docs/api` reports `!! docs/api/`. |
| Existing local generated output | `docs/api/html/` exists locally with 100 top-level files, but remains ignored local context. |

## API Documentation Inputs

| Input | Sprint 158 role |
| --- | --- |
| `Doxyfile` | Defines Doxygen input set, warning policy, and generated output path. |
| `Makefile` `docs` target | Owned command for generating `docs/api/html/`. |
| `include/*.h` | Checked-in public headers and source-header-first declaration authority. |
| `include/sparse_version.h.in` | Generated installed version-header template; Sprint 158 must decide how it appears in generated API docs. |
| `docs/api_reference.md` | User-facing API reference entry point and current generated HTML boundary. |
| `docs/maintainer_guide.md` | Maintainer policy for Doxygen freshness, generated output, and source-header-first ownership. |
| `README.md` | User-facing discovery links and quality command summary. |
| `docs/tutorial.md` | First-use workflow and API reference routing. |
| `.gitignore` | Current generated-output tracking policy for `docs/api/`. |
| Sprint 157 Day 9, 10, 12, and 14 artifacts | Evidence contract, validation map, risk register, and Sprint 158 handoff. |

## Public Header Source Set

The checked-in public header source set currently contains 18 files:

1. `include/sparse_analysis.h`
2. `include/sparse_bidiag.h`
3. `include/sparse_cholesky.h`
4. `include/sparse_csr.h`
5. `include/sparse_dense.h`
6. `include/sparse_eigs.h`
7. `include/sparse_ic.h`
8. `include/sparse_ilu.h`
9. `include/sparse_iterative.h`
10. `include/sparse_ldlt.h`
11. `include/sparse_lu.h`
12. `include/sparse_lu_csr.h`
13. `include/sparse_matrix.h`
14. `include/sparse_qr.h`
15. `include/sparse_reorder.h`
16. `include/sparse_svd.h`
17. `include/sparse_types.h`
18. `include/sparse_vector.h`

Generated install behavior for `sparse_version.h` is intentionally separate
from the checked-in source set and must be addressed explicitly during the
coverage and publication decision.

## Stop Conditions

- `make docs` cannot run and the tooling blocker cannot be resolved locally.
- Doxygen warnings are present without triage before generated HTML is treated
  as fresh or published evidence.
- Generated page coverage misses intended public headers without recorded
  exclusions.
- Generated output is committed while `.gitignore`, review guidance, or docs
  still describe it as ignored local-only output.
- Generated output remains local-only while docs imply it is checked in,
  published, complete, or fresh.
- Public header comments change without declaration-preservation proof and
  `make format && make lint && make test`.
- API reference wording implies dynamic ABI, shared-library support,
  package-manager distribution, broad platform parity, external parity,
  portable performance, or state-of-the-art coverage.

## Daily Log

### Day 1: API Docs Intake

- Re-read the Sprint 158 plan and verified the authoritative project-plan
  source is `docs/planning/EPIC_14/PROJECT_PLAN.md`.
- Reviewed Sprint 157 Day 9 evidence contract, Day 10 quality surface map, Day
  12 risk register and Sprint 158 handoff, and Day 14 final handoff.
- Recorded branch baseline: `sprint-158` at
  `ab98ce4d32173617de3dc009e0a07e446d157042`, created from current `master`
  after PR #175.
- Confirmed Doxygen is available at `/usr/local/bin/doxygen`, version
  `1.16.1`.
- Captured generated-doc tracking state before generation:
  `git status --ignored=matching --short docs/api` reports `!! docs/api/`.
- Captured that `docs/api/html/` already exists locally with 100 top-level
  generated files, but it remains ignored local context and is not pass
  evidence.
- Recorded Doxygen configuration: input `include/`, pattern `*.h`,
  non-recursive, output `docs/api/html/`, HTML enabled, and warnings enabled but
  not errors.
- Captured the 18 checked-in public headers and the separate generated
  `sparse_version.h` policy question.
- Day 2 handoff: run `make docs`, capture Doxygen version/command/exit status,
  stdout/stderr, warnings, generated output inventory, and any local path or
  stale-output risks.

### Day 2: Doxygen Baseline

- Ran `make docs 2>&1`; it completed with exit code `0`.
- Confirmed the command path is `make docs` -> `doxygen Doxyfile`, with
  Doxygen `1.16.1`.
- Captured 10 warnings:
  - five `Found unknown command '\U'` warnings in
    `include/sparse_lu_csr.h` at lines 105, 152, 268, 286, and 288;
  - four undocumented macro/typedef warnings in `include/sparse_types.h` for
    `idx_t`, `IDX_MAX`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX`;
  - one undocumented struct-member warning in `include/sparse_iterative.h` for
    `sparse_gmres_opts_t::progress_user`.
- Confirmed generated output remains under ignored `docs/api/html/`; this is
  still local context and not published or source-controlled pass evidence.
- Captured generated output inventory: 212 files, 2 directories, 87 top-level
  `.html` files, 18 header reference pages, and 18 header source pages.
- Confirmed `index.html`, `files.html`, `annotated.html`, and `globals.html`
  are present.
- Confirmed generated header reference pages exist for all 18 checked-in
  public headers.
- Confirmed no generated page exists for `sparse_version.h` or
  `sparse_version.h.in` under the current Doxygen input configuration.
- Searched generated HTML for absolute local path fragments such as
  `/Users/jeff`, `/private/`, `/tmp/`, and `/var/folders`; no matches were
  found.
- Timestamp-like matches are limited to vendored JavaScript comments in
  `jquery.js`, not per-run generated timestamps.
- Day 3 handoff: build a public-header coverage matrix from the 18 checked-in
  headers, record generated version-header treatment, and classify the 10
  warnings for fix/defer/exclude/blocker disposition.

### Day 3: Header Coverage Map

- Reconfirmed the Doxygen input set is `include/` with `*.h` and
  `RECURSIVE = NO`.
- Reconfirmed the checked-in public header source set contains 18
  `include/*.h` files.
- Built the header-to-generated-page coverage map:
  - all 18 checked-in public headers have matching `*_8h.html` reference
    pages;
  - all 18 checked-in public headers have matching `*_8h_source.html` source
    pages.
- Recorded that Day 3 page coverage should count checked-in `include/*.h`
  inputs only.
- Recorded generated version-header treatment:
  - Make derives `build/include/sparse_version.h` from `VERSION` and
    `include/sparse_version.h.in`, then installs it beside public headers;
  - CMake derives `${CMAKE_CURRENT_BINARY_DIR}/include/sparse_version.h` from
    the same template and installs it beside public headers;
  - current Doxygen input does not include `include/sparse_version.h.in` and
    does not generate a `sparse_version` page;
  - `sparse_types.h` includes `sparse_version.h`, so generated HTML references
    the include but not the generated header's macro definitions as a separate
    page.
- Day 3 found no missing generated pages for the checked-in public header
  source set.
- Day 3 did not change headers or generated-doc policy; it leaves the 10 Day 2
  warnings for Day 4 triage.
- Day 4 handoff: normalize the warning categories and decide which warnings
  are fix/defer/exclude/blocker, with full C/header gates required for any
  public header edits.

### Day 4: Warning Triage Policy

- Inspected the Day 2 warning locations in `include/sparse_lu_csr.h`,
  `include/sparse_types.h`, and `include/sparse_iterative.h`.
- Normalized the 10 Doxygen warnings into three categories:
  - unknown Doxygen command `\U` from public-header prose containing `L\U`;
  - undocumented `idx_t` and index-format macros in `sparse_types.h`;
  - undocumented `sparse_gmres_opts_t::progress_user` in
    `sparse_iterative.h`.
- Classified all three warning categories as selected for Sprint 158 warning
  closure, not as exclusions.
- Recorded publication gating: these warnings block any generated API HTML
  publication or freshness claim until fixed and revalidated or deliberately
  reclassified.
- Recorded implementation scope: expected fixes are public-header
  documentation/comment edits only; declaration or code behavior changes are
  not selected by Day 4.
- Recorded quality escalation: any public-header edit, including comment-only
  Doxygen cleanup, requires `make format && make lint && make test` per Sprint
  157 Day 10 policy.
- Day 5 handoff: compare publication options while assuming warning closure is
  required before committed or hosted generated HTML can be presented as fresh
  evidence.

### Day 5: Publication Options

- Compared three generated API HTML publication paths:
  - source-controlled `docs/api/html/`;
  - CI-published artifact or hosted pages;
  - guarded local-only generated output with a recurring freshness/page/warning
    guard.
- Recorded current generated-output cost: ignored `docs/api/` is approximately
  3.1 MB and contains 212 generated files after `make docs`.
- Confirmed `.gitignore` ignores `docs/api/` and no existing workflow currently
  publishes Doxygen HTML.
- Recorded that the current maintainer guide describes generated HTML as fresh
  only when generated output is committed with the corresponding source/header
  change; Day 5 recommends replacing that with a guarded local-only policy
  unless Day 6 deliberately selects a different publication path.
- Recommended Sprint 158 product direction: keep generated API HTML
  local-only and ignored, but add or document recurring checks for `make docs`,
  warning closure, and public-header page coverage.
- Rejected committed HTML for this sprint because it adds large generated
  review churn without improving source-header authority.
- Rejected CI-published HTML for this sprint because it requires new hosted
  artifact/retention/support-tier policy beyond the current sprint's immediate
  closure target.
- Day 6 handoff: convert the guarded local-only recommendation into a concrete
  file-change checklist, stale-output prevention rules, and support-tier
  wording for API reference and maintainer documentation.

### Day 6: Publication Decision

- Finalized the Sprint 158 publication decision: keep generated API HTML
  local-only and ignored under `docs/api/`, with a recurring guard for
  `make docs`, Doxygen warning closure, and public-header page coverage.
- Confirmed Sprint 158 will not commit `docs/api/html/` and will not add a CI
  publication or hosted-pages lane.
- Recorded files that should remain unchanged for the selected path:
  `.gitignore` continues to ignore `docs/api/`; CI workflows do not publish
  Doxygen HTML; `Doxyfile` output remains `docs/api/html/` unless a later
  guard implementation needs a focused non-output change.
- Recorded likely implementation files for Days 7-11:
  `docs/api_reference.md`, `docs/maintainer_guide.md`, a page-coverage guard
  script or Make target, and public headers touched only for selected Doxygen
  warning cleanup.
- Drafted support-tier wording: generated API HTML is reproducible local
  Doxygen output, not source-controlled or hosted evidence; source-controlled
  API truth remains `docs/api_reference.md` plus checked-in public headers.
- Recorded stale-output prevention rules:
  - ignored `docs/api/html/` is never cited as checked-in freshness evidence;
  - generated HTML is fresh only for the local branch/run that executed the
    guard;
  - public docs must say how to regenerate and validate local output;
  - warning closure and page coverage are prerequisites for describing a local
    generated tree as current for the configured input set.
- Day 7 handoff: design the deterministic page-coverage guard for the 18
  checked-in public headers and the generated version-header policy row.

### Day 7: Page Coverage Check Design

- Designed the page-coverage guard as a small script-owned check with a Make
  wrapper:
  - proposed script: `scripts/check_api_docs_coverage.py`;
  - proposed Make target: `api-docs-coverage`;
  - optional combined target: `docs-check` or equivalent follow-up wrapper if
    Day 8 keeps the scope focused.
- Defined required guard inputs:
  - checked-in public headers from `include/*.h`;
  - generated HTML directory `docs/api/html/`;
  - Doxygen configuration and Day 3 generated version-header policy.
- Defined deterministic generated page naming by applying the Doxygen filename
  convention observed on Day 2/3:
  - replace `_` with `__`;
  - replace `.` with `_`;
  - append `.html` for the reference page;
  - append `_source.html` for the supplemental source page.
- Defined pass/fail behavior:
  - fail if `docs/api/html/` or `index.html` is missing;
  - fail if any checked-in public header lacks its reference page;
  - optionally fail or warn on missing source pages depending on Day 8 scope;
  - report missing pages with both source header and expected generated path.
- Confirmed generated `sparse_version.h` remains a policy row and is not part
  of expected checked-in public-header page coverage under current Doxygen
  inputs.
- Defined freshness behavior: the guard validates the current local generated
  tree only; it must be run after `make docs` for a fresh local result and does
  not convert ignored generated HTML into source-controlled evidence.
- Day 8 handoff: implement the script and Make target, run `make docs` followed
  by the coverage guard, and record results.

### Day 8: Page Coverage Check Implementation

- Added `scripts/check_api_docs_coverage.py`.
- Added Make targets:
  - `api-docs-coverage` runs the coverage script against the current generated
    local Doxygen tree;
  - `docs-check` runs `make docs` followed by `api-docs-coverage`.
- Implemented the guard to derive expected pages from checked-in
  `include/*.h` files rather than hard-coding the 18 current headers.
- Implemented required checks for:
  - `docs/api/html/`;
  - `docs/api/html/index.html`;
  - one generated reference page per checked-in public header;
  - one generated source page per checked-in public header.
- Preserved generated `sparse_version.h` as a separate installed-header policy
  row and not an expected generated page.
- Initial local coverage run exposed a mapping bug: the script expected `_h`
  page suffixes, while Doxygen emits `_8h`; fixed the mapping and corrected the
  Day 7 design note.
- Ran `python3 -m py_compile scripts/check_api_docs_coverage.py`; passed.
- Ran `make api-docs-coverage`; passed with 18 checked-in public headers, 18
  generated reference pages, and 18 generated source pages.
- Ran `make docs-check`; passed coverage after regenerating docs. Doxygen still
  emits the 10 Day 4 warnings, which remain selected for Day 9 closure.
- Ran a missing-directory negative-path check with
  `--html-dir /tmp/lso-missing-api-html`; the script returned exit status `1`
  and reported that generated API HTML is missing.
- Confirmed generated HTML remains ignored under `docs/api/` and is not staged.
- Day 9 handoff: fix the three selected Doxygen warning categories, then run
  `make format && make lint && make test`, `make docs-check`, and docs hygiene
  because public headers will be edited.

### Day 9: Warning Fix Batch

- Applied only the Day 4 selected warning fixes.
- Updated `include/sparse_lu_csr.h` comment prose from Doxygen-problematic
  `L\U` wording to equivalent `L and U` wording.
- Added Doxygen comments for `idx_t`, `IDX_MAX`, `SPARSE_PRIDX`, and
  `SPARSE_SCNIDX` in both 32-bit and 64-bit branches of
  `include/sparse_types.h`.
- Added member documentation for
  `sparse_gmres_opts_t::progress_user` in `include/sparse_iterative.h`.
- Kept Day 9 public-header edits comment-only; no declarations, macro values,
  struct layout, or behavior changed.
- Ran `make docs-check`; passed with no Doxygen warnings and complete API page
  coverage for the 18 checked-in public headers.
- Ran the required full public-header gate:
  `make format && make lint && make test`; passed.
- Confirmed generated `docs/api/html/` remains ignored local validation output
  and is not source-controlled publication evidence.
- Day 10 handoff: align public and maintainer documentation with the local-only
  generated API HTML decision and the new `docs-check` guard.

### Day 10: Policy Alignment

- Updated `docs/api_reference.md` to name `make docs-check` as the local
  generated API HTML freshness and page-coverage command.
- Clarified that `docs/api/html/` remains ignored local generated output, not a
  hosted or source-controlled publication surface.
- Clarified that generated Doxygen HTML is current only for the branch and
  checkout where `make docs-check` has just passed.
- Added generated `sparse_version.h` ownership wording: version macro behavior
  remains owned by installed artifacts, `VERSION`, and install-validation tests
  rather than an expected Doxygen page under the current input set.
- Updated `docs/maintainer_guide.md` to replace committed-output freshness
  rules with the selected Sprint 158 local-only generated HTML policy.
- Reviewed README and tutorial API-reference routing; no route change was
  needed because both already point declaration-oriented users to
  `docs/api_reference.md` and public headers.
- Ran `make docs-check`; passed with no Doxygen warnings and complete coverage
  for the 18 checked-in public headers.
- Ran `git diff --check`; passed.
- Ran a stale wording scan across the live docs and Sprint 158 artifacts; the
  only committed-output phrase left is historical Day 5 context describing the
  pre-Day-10 maintainer-guide wording.
- Day 10 handoff: Day 11 should verify `.gitignore`, generated paths,
  `make docs-check`, and public/maintainer wording all agree with the
  local-only publication path.

### Day 11: Publication Finalization

- Finalized the generated API HTML publication path as local-only and ignored.
- Confirmed `.gitignore` still excludes `docs/api/`.
- Confirmed `git status --ignored=matching docs/api` reports ignored generated
  output as `!! docs/api/`, with no staged generated HTML.
- Confirmed the recurring freshness guard is implemented through
  `make docs-check`, which composes Doxygen generation and
  `api-docs-coverage`.
- Updated README's local command inventory to list `make docs-check` beside
  `make docs`.
- Reconfirmed `docs/api_reference.md` and `docs/maintainer_guide.md` describe
  the same local-only generated HTML policy and source-header-first ownership.
- Ran `make docs-check`; passed with complete generated page coverage for 18
  checked-in public headers.
- Ran `git diff --check`; passed.
- Ran a trailing-whitespace scan over README, API reference, maintainer guide,
  and Sprint 158 planning artifacts; passed.
- Day 12 handoff: run the broader validation evidence pass, including
  `make docs-check`, documentation hygiene, and the full C/header quality gate
  because Sprint 158 has public-header edits earlier in the branch.

### Day 12: Validation Evidence

- Ran `make docs-check`; passed with no Doxygen warnings and complete generated
  page coverage for 18 checked-in public headers.
- Ran the required public-header gate:
  `make format && make lint && make test`; passed.
- The full gate completed formatting, strict warning compile, clang-tidy,
  cppcheck, and the maintained test suite, ending with `All tests passed.`
- Checked for formatter side effects; no additional source changes appeared
  beyond the intended Sprint 158 public-header comment changes.
- Confirmed generated API HTML remains ignored local output:
  `git status --ignored=matching docs/api` reports `!! docs/api/`.
- No required Day 12 checks were skipped.
- Day 13 handoff: reconcile claims and artifacts against the local-only
  generated API HTML publication path and prepare the Sprint 159 hosted-report
  handoff boundaries.

### Day 13: Claim Reconciliation

- Audited README, API reference, maintainer guide, tutorial, Makefile, and
  `.gitignore` against the selected local-only generated API HTML decision.
- Confirmed live API-reference and maintainer-guide wording treats
  `make docs-check` as the local freshness/page-coverage guard and keeps
  `docs/api/html/` ignored and local-only.
- Confirmed README and tutorial routing remains appropriate: exact declaration
  readers go to `docs/api_reference.md` and checked-in public headers.
- Ran a claim-sensitive scan for hosted generated API freshness,
  source-controlled generated HTML, release evidence, dynamic ABI,
  shared-library, package-manager, broad platform parity, external-library
  parity, portable performance, and state-of-the-art wording.
- Reconciled Sprint 158 project-plan items:
  Doxygen baseline, publication decision, coverage check, warning triage, docs
  alignment, and validation are closed; closeout remains for Day 14.
- Recorded residuals: hosted Doxygen publication, committed generated HTML,
  generated `sparse_version.h` Doxygen page, broad generated-reference
  completeness, and hosted generated report promotion.
- Drafted Sprint 159 handoff prerequisites for hosted generated report work,
  keeping it separate from generated API HTML publication.
- Ran `git diff --check`; passed.
- Ran a trailing-whitespace scan over README, API reference, maintainer guide,
  tutorial, and Sprint 158 planning artifacts; passed.
- Ran a claim-sensitive scan over README, API reference, maintainer guide,
  tutorial, and the Day 13 artifact; matches were explicit non-claims,
  existing unrelated bounded-evidence language, or the Day 13 audit terms
  themselves.
- Day 14 handoff: produce final Sprint 158 closeout and confirm all artifacts,
  working notes, ignored generated output, and validation evidence are ready
  for retrospective/PR packaging.

### Day 14: Closeout Handoff

- Verified all Day 1 through Day 13 artifacts are present.
- Wrote the Day 14 closeout artifact with final outcome, delivered changes,
  artifact inventory, final validation evidence, final tracking state,
  residuals, Sprint 159 handoff prerequisites, and retrospective inputs.
- Reconfirmed the Sprint 158 final decision: generated API HTML remains
  local-only and ignored, with `make docs-check` as the maintained freshness
  and page-coverage guard.
- Reconfirmed generated HTML is not hosted, committed, staged, or treated as
  release/package/ABI/platform/performance/parity/state-of-the-art evidence.
- Ran final `make docs-check`; passed with complete generated page coverage for
  18 checked-in public headers.
- Ran final `git diff --check`; passed.
- Ran final trailing-whitespace scan across touched docs/planning/source
  surfaces; passed.
- Final closeout handoff: Sprint 159 can start from the settled API-doc policy
  and focus on hosted generated report promotion without re-deciding Doxygen
  API HTML publication.
