# Sprint 173 Working Notes

## Sprint Goal

Decide and implement the supported generated API HTML publication path.

## Source Artifact Note

The Sprint 173 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`,
but the active merged Sprint 173 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 173: Generated API HTML Publication Closure".

## Branch Baseline

- Branch: `sprint-173`
- Starting point: current `master` after PR #191 merge.
- Sprint 172 status: complete and merged, with LU public-header/docs cleanup
  and a local LU drift guard.
- Sprint 173 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_173/PLAN.md`.

## Prior Evidence Carried Forward

| Input | Source | Sprint 173 use |
| --- | --- | --- |
| LU public-header cleanup and guard | `docs/planning/EPIC_15/SPRINT_172/RETROSPECTIVE.md`, `scripts/check_lu_header_docs_guard.sh` | Treats `include/sparse_lu.h` as improved generated-doc input, but not as generated API HTML freshness proof. |
| API reference entry point | `docs/api_reference.md` | Current user-facing source for exact public declarations and generated HTML interpretation. |
| Generated API maintainer policy | `docs/maintainer_guide.md` | Records current local-only `docs/api/html/` policy and unsupported hosted/source-controlled claims. |
| Doxygen configuration | `Doxyfile` | Defines checked-in `include/*.h` as the current generated API input set and `docs/api/html/` as the HTML output path. |
| Docs Make targets | `Makefile` | Defines `make docs`, `make api-docs-coverage`, and `make docs-check`. |
| API page coverage guard | `scripts/check_api_docs_coverage.py` | Checks generated Doxygen reference/source pages for checked-in public headers. |
| Ignore rules | `.gitignore` | Keeps `docs/api/` local generated output out of source control. |
| Static package/shared ABI deferral | `docs/planning/EPIC_15/SPRINT_170/RETROSPECTIVE.md`, `scripts/static_package_deferral_check.sh` | Keeps generated API docs from becoming a shared-library, dynamic ABI, runtime-loader, or package support claim. |
| Package-manager deferral | `docs/planning/EPIC_15/SPRINT_171/RETROSPECTIVE.md`, `scripts/package_manager_deferral_check.sh` | Keeps generated API docs from implying provider package-manager availability. |

## Current Generated API Surface

- `Doxyfile` reads checked-in headers from `include/` with `FILE_PATTERNS =
  *.h` and `RECURSIVE = NO`.
- `Doxyfile` writes generated HTML under `docs/api/html/`.
- `make docs` runs `doxygen Doxyfile`.
- `make docs-check` runs `make docs` and then
  `python3 scripts/check_api_docs_coverage.py`.
- `scripts/check_api_docs_coverage.py` expects generated reference and source
  pages for each checked-in public header under `include/`.
- `docs/api_reference.md` and `docs/maintainer_guide.md` describe generated
  HTML as local-only generated output, current only after `make docs-check`
  passes in the active checkout.
- `.gitignore` ignores `docs/api/`, so generated API HTML is not
  source-controlled by default.

## Publication Choices To Preserve For Decision

Sprint 173 must decide exactly one supported path before changing publication
behavior:

- hosted generated API HTML;
- committed generated API HTML;
- CI artifact-only generated API HTML;
- local-only generated API HTML with explicit enforcement.

Until the decision lands, the inherited policy remains local-only and ignored.

## Retained Claim Non-Claims

Sprint 173 starts with no support claim for:

- hosted generated API HTML publication;
- source-controlled generated API HTML;
- generated API HTML freshness without a just-passed check;
- generated installed-header Doxygen coverage for `sparse_version.h`;
- dynamic ABI stability;
- shared-library build/install support;
- runtime-loader behavior;
- package-manager provider availability;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance superiority;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

Generated API docs may improve API discoverability. They must not widen
package, ABI, runtime-loader, platform, performance, external-parity, or
state-of-the-art claims.

## Sprint 173 Stop Conditions

Stop and revise before proceeding if a change:

- changes generated API publication behavior before a decision artifact exists;
- stages generated files under `docs/api/` without selecting a committed-output
  publication path;
- claims generated API HTML is hosted or source-controlled before that path is
  implemented and checked;
- treats `make docs-check` as proof beyond the configured checked-in public
  header input set;
- expects generated installed headers such as `sparse_version.h` to appear in
  Doxygen coverage without changing the input policy;
- weakens Sprint 170 shared-library/dynamic ABI/runtime-loader deferral;
- weakens Sprint 171 package-manager provider deferral;
- changes `.c` or `.h` files without running `make format && make lint &&
  make test`;
- updates adoption/package/ABI/platform wording without running targeted claim
  scans and relevant deferral guards.

## Working Assumptions

- Day 1 is planning and intake only.
- If only planning files change on a given day, `git diff --check` is
  sufficient for that day.
- If generated API commands or docs are changed later, run focused generator,
  staging, freshness, and claim checks.
- If `.c` or `.h` files change, run the full C quality gate.
- If package/adoption/ABI wording changes, run
  `bash scripts/package_manager_deferral_check.sh` and/or
  `bash scripts/static_package_deferral_check.sh` as appropriate.

## Daily Log

### Day 1: Sprint Intake And Generated API Boundary

- Re-read the active Sprint 173 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Confirmed the prompt path points at an older Epic 12 planning file, while
  the active Sprint 173 section lives in Epic 15.
- Reviewed Sprint 172 closeout, retrospective, working notes, and LU
  header/docs guard handoff.
- Reviewed current generated API HTML surface in `Doxyfile`, `Makefile`,
  `scripts/check_api_docs_coverage.py`, `docs/api_reference.md`,
  `docs/maintainer_guide.md`, README command guidance, and `.gitignore`.
- Confirmed existing local generated HTML files are present under ignored
  `docs/api/html/`, with `docs/api/` ignored by source control.
- Created Sprint 173 artifact directory structure.
- Recorded generated API publication choices and retained non-claim
  boundaries before publication implementation.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-api-docs-intake.md`.

### Day 2: Generator And Output Inventory

- Inventoried the generated API HTML configuration, commands, documentation
  references, CI references, tracked inputs, ignored outputs, and freshness
  assumptions.
- Confirmed the current generator is Doxygen via `Doxyfile`, invoked by
  `make docs`, with output under ignored `docs/api/html/`.
- Confirmed `make docs-check` is the current local validation target: it runs
  Doxygen and `scripts/check_api_docs_coverage.py`.
- Confirmed there are no `.github` workflow references to `docs-check`,
  `api-docs-coverage`, Doxygen, or `docs/api/html/`.
- Confirmed the current tracked Doxygen input set is the 18 checked-in public
  headers under `include/*.h`; `include/sparse_version.h.in` is tracked but
  does not generate an expected Doxygen page under the current policy.
- Confirmed ignored local generated HTML is present under `docs/api/html/`
  with 214 files at Day 2 inspection time.
- Ran `python3 scripts/check_api_docs_coverage.py`; it passed with 18 checked
  public headers, 18 generated reference pages, and 18 generated source pages.
- Recorded the key freshness gap: page coverage proves expected generated
  pages exist, but it does not prove the generated HTML is fresh relative to
  public headers, `Doxyfile`, the coverage script, Makefile targets, or
  documentation navigation.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-generator-inventory.md`.

### Day 3: Publication Decision Matrix

- Compared four generated API HTML publication paths: hosted site, committed
  HTML, CI artifact-only HTML, and guarded local-only generation.
- Re-read Sprint 158 publication options, publication decision, and
  retrospective to avoid reopening settled local-only policy without new
  evidence.
- Confirmed Day 2 found no new `.github` Doxygen/docs-check publication lane
  and no tracked generated API output.
- Rated committed HTML as high repository/review churn because generated
  output is currently 214 ignored files under `docs/api/html/`.
- Rated hosted HTML as useful but currently unfunded because it requires
  deployment permissions, retention policy, URL ownership, branch semantics,
  and support-tier wording not present in the repo.
- Rated artifact-only HTML as lower risk than hosting but still incomplete for
  user discoverability and absent from current CI.
- Recommended guarded local-only generation for the Day 4 decision, with
  Sprint 173 implementation focused on freshness/staging enforcement rather
  than a new publication surface.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-publication-options.md`.

### Day 4: Publication Decision Record

- Converted the Day 3 recommendation into the formal Sprint 173 generated API
  HTML publication decision.
- Selected guarded local-only generated API HTML with stronger
  freshness/staging enforcement.
- Confirmed `docs/api/` remains ignored, generated HTML remains uncommitted,
  and no hosted or artifact-only Doxygen publication lane is selected.
- Defined the supported user surface as checked-in public headers plus
  `docs/api_reference.md`, with local Doxygen HTML available only after
  `make docs-check` passes in the active checkout.
- Preserved non-claims for hosted docs, committed generated HTML,
  artifact-only generated HTML, stale generated HTML freshness, generated
  installed-header Doxygen coverage, package-manager support,
  shared-library/dynamic ABI support, runtime-loader behavior, broad platform
  parity, performance, external parity, and state-of-the-art coverage.
- Defined required implementation checks for Days 5 through 9: generator
  command review, ignored-output staging enforcement, freshness/staging guard
  design and implementation, docs navigation review, claim scans, and relevant
  deferral guards if wording touches package/ABI/platform surfaces.
- Day 4 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day4-publication-decision.md`.

### Day 5: Generator Command Normalization Design

- Reviewed the selected Day 4 guarded local-only path against the current
  Doxygen, Makefile, ignore, and coverage-check surfaces.
- Ran `make -n docs-check` and confirmed the selected command expands to:
  `doxygen Doxyfile` followed by
  `python3 scripts/check_api_docs_coverage.py`.
- Confirmed `git check-ignore -v docs/api docs/api/html
  docs/api/html/index.html` reports the `.gitignore` `docs/api/` rule.
- Confirmed `git ls-files docs/api` reports zero tracked generated API files.
- Confirmed `git ls-files --others --exclude-standard docs/api` reports zero
  non-ignored untracked generated API files.
- Designed Day 6 implementation around a focused local-only guard script plus
  optional Make target integration, not around changing `make docs-check`,
  `Doxyfile`, or `.gitignore`.
- Defined the intended guard behavior: prove `docs/api/` remains ignored, no
  generated API files are tracked, no generated API files are staged, and local
  generated HTML is not represented as source-controlled evidence.
- Day 5 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day5-generator-design.md`.

### Day 6: Generator Command Implementation

- Implemented the Day 5 local-only enforcement path with
  `scripts/check_api_docs_local_only.sh`.
- Added Make targets:
  - `api-docs-local-only` for the focused ignored/tracked/staged generated API
    HTML guard;
  - `api-docs-validate` for `docs-check` plus the local-only guard.
- Preserved existing `make docs`, `make api-docs-coverage`, and
  `make docs-check` behavior.
- Kept `Doxyfile`, `.gitignore`, public docs, public headers, and generated
  `docs/api/html/` tracking policy unchanged.
- Ran `make api-docs-local-only`; it passed.
- Ran `make api-docs-validate`; it regenerated local Doxygen HTML, passed page
  coverage for 18 checked-in public headers, and passed the local-only guard.
- Confirmed generated HTML remains ignored local output and is not tracked,
  staged, or visible as non-ignored untracked output.
- Day 6 changed `Makefile`, one shell script, and planning artifacts. No `.c`
  or `.h` files were modified, so the full C quality gate is not required for
  this day.
- Created `artifacts/day6-generator-implementation.md`.

### Day 7: Freshness Gate Design

- Designed the generated API HTML freshness gate around the selected Day 4
  guarded local-only path.
- Defined `make api-docs-validate` as the current selected local freshness
  proof because it regenerates Doxygen output, checks expected public-header
  pages, and verifies ignored/tracked/staged generated-output boundaries in one
  command.
- Classified freshness inputs: checked-in public headers, `Doxyfile`,
  Makefile docs targets, coverage/local-only guard scripts, README/API
  reference/maintainer navigation, `.gitignore`, and the Day 4 decision
  record.
- Explicitly rejected persisted source-to-output metadata for Sprint 173
  unless Day 8 finds a concrete stale-output failure that command regeneration
  cannot cover.
- Defined unselected-output exclusion rules for hosted, committed, and
  artifact-only Doxygen HTML.
- Recommended Day 8 add an explicit `api-docs-freshness` Make alias to the
  selected local gate so generated API docs have a command name consistent
  with other maintained freshness targets.
- Day 7 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day7-freshness-design.md`.

### Day 8: Freshness Gate Implementation

- Implemented the Day 7 freshness naming recommendation by adding
  `api-docs-freshness` as a Make alias to `api-docs-validate`.
- Preserved existing `docs`, `api-docs-coverage`, `api-docs-local-only`,
  `docs-check`, and `api-docs-validate` behavior.
- Ran `make api-docs-freshness`; it regenerated Doxygen HTML, passed page
  coverage for 18 checked-in public headers, and passed the local-only guard.
- Ran an isolated fail-mode proof in a temporary Git repository without a
  `docs/api/` ignore rule; `scripts/check_api_docs_local_only.sh` failed with
  the intended local-only publication message.
- Confirmed no generated HTML was staged or tracked by the Sprint 173 working
  tree.
- Day 8 changed `Makefile` and planning artifacts. No `.c` or `.h` files were
  modified, so the full C quality gate is not required for this day.
- Created `artifacts/day8-freshness-implementation.md`.

### Day 9: Documentation Navigation Design

- Reviewed README front-door/adoption map, README command list,
  `docs/api_reference.md`, `docs/maintainer_guide.md`, `docs/tutorial.md`,
  cookbook, solver-selection guide, install docs, and benchmark docs for API
  reference and generated API HTML navigation.
- Confirmed the existing user-facing navigation already routes exact API
  declarations to `docs/api_reference.md` and checked-in public headers.
- Identified the main Day 10 update need: replace or supplement generated API
  validation wording that still names only `make docs-check` with the Sprint
  173 selected freshness command, `make api-docs-freshness`.
- Designed README command-list wording to keep `make docs` as raw generation,
  keep `make docs-check` as local generation plus page coverage, and add
  `make api-docs-freshness` as the selected local generated API freshness and
  local-only staging proof.
- Designed `docs/api_reference.md` wording so generated HTML remains
  local-only and current only after `make api-docs-freshness` passes.
- Designed `docs/maintainer_guide.md` wording so maintainers prefer
  `make api-docs-freshness` for generated API freshness and local-only
  staging enforcement.
- Preserved non-claims for hosted docs, committed generated HTML, artifact-only
  generated HTML, package-manager support, shared-library support, dynamic ABI,
  runtime-loader behavior, platform parity, performance, external parity, and
  state-of-the-art coverage.
- Day 9 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day9-navigation-design.md`.

### Day 10: Documentation Navigation Update

- Updated README command guidance to list `make api-docs-freshness` as the
  selected local Doxygen freshness plus local-only staging guard.
- Updated `docs/api_reference.md` so generated HTML is current only after
  `make api-docs-freshness` passes, while `docs-check` remains the page
  coverage layer.
- Updated `docs/maintainer_guide.md` so maintainers use
  `make api-docs-freshness` for local generated API freshness and local-only
  staging enforcement.
- Preserved the source-controlled API reference hierarchy:
  `docs/api_reference.md` plus checked-in public headers under `include/`.
- Preserved local-only generated HTML wording: no hosted, committed,
  artifact-published, source-controlled, or release-evidence claim.
- Ran `make api-docs-freshness`; it passed.
- Ran the targeted generated-doc claim scan over README, API reference, and
  maintainer guide; inspected matches as expected non-claim or unrelated
  bounded-evidence wording.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Day 10 changed README, `docs/api_reference.md`, `docs/maintainer_guide.md`,
  and planning artifacts. No `.c` or `.h` files were modified, so the full C
  quality gate is not required for this day.
- Created `artifacts/day10-navigation-update.md`.

### Day 11: Integrated Generator Validation

- Ran `make api-docs-freshness`; it passed, proving local Doxygen generation,
  page coverage for 18 checked-in public headers, and local-only
  ignored/tracked/staged generated-output boundaries.
- Ran `make api-docs-local-only`; it passed as a direct staging and
  local-only guard proof.
- Ran the targeted claim scan over README, API reference, and maintainer guide.
  Matches were reviewed as expected non-claim language, pre-existing bounded
  evidence wording, or unrelated planning/report artifact references.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Confirmed generated API HTML remains ignored local output under `docs/api/`.
- Day 11 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day11-generator-validation.md`.

### Day 12: CI And Maintenance Surface Review

- Reviewed Makefile API docs targets, local-only guard script, README,
  `docs/api_reference.md`, `docs/maintainer_guide.md`, `.github` workflow
  references, and report-family metadata for generated API publication impact.
- Confirmed `make api-docs-freshness` owns selected local generated API
  freshness by delegating to `api-docs-validate`, which runs `docs-check` plus
  `api-docs-local-only`.
- Confirmed `scripts/check_api_docs_coverage.py` owns generated page coverage
  for checked-in public headers.
- Confirmed `scripts/check_api_docs_local_only.sh` owns local-only ignore,
  tracking, staging, and non-ignored-untracked generated-output checks.
- Confirmed README, API reference, and maintainer guide name the selected
  freshness command and preserve local-only generated HTML wording.
- Confirmed no `.github` workflow currently publishes or uploads Doxygen HTML.
- Confirmed no report-family metadata row currently promotes generated API HTML
  as a hosted, committed, artifact-only, package, ABI, platform, performance,
  external-parity, or state-of-the-art proof surface.
- Recorded residuals: no hosted Doxygen docs, no artifact-only Doxygen lane,
  no committed generated HTML, no `sparse_version.h` Doxygen page, and no
  CI docs-check lane.
- Day 12 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day12-maintenance-review.md`.

### Day 13: Integrated Claim Review

- Reviewed the Sprint 173 artifact chain from publication decision through
  implementation, freshness naming, documentation navigation, validation, and
  maintenance ownership.
- Reconciled the supported claim: generated API HTML is a local Doxygen view
  that is current only after `make api-docs-freshness` passes in the active
  checkout.
- Confirmed source-controlled API truth remains the checked-in public headers,
  `docs/api_reference.md`, and maintainer policy.
- Confirmed hosted generated HTML, committed generated HTML, artifact-only
  generated HTML, release-evidence generated HTML, package-manager provider
  support, shared-library support, dynamic ABI, runtime-loader behavior,
  platform parity, performance, external parity, and state-of-the-art coverage
  remain non-claims.
- Ran `make api-docs-freshness`; it passed with 18 checked-in public headers,
  18 generated reference pages, 18 generated source pages, and local-only
  generated-output enforcement.
- Ran `make api-docs-local-only`; it passed.
- Ran the generated API claim scan over README, API reference, maintainer
  guide, Makefile, and the local-only guard script; matches were expected
  selected local-only wording or unrelated pre-existing report surfaces.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Confirmed `git status --ignored --short docs/api` reports `!! docs/api/`,
  which is the intended ignored local-output state.
- Day 13 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day13-claim-review.md`.

### Day 14: Sprint Closeout And Sprint 174 Handoff

- Reconciled Sprint 173 against Epic 15 project-plan items 173.1 through
  173.6 and confirmed each item is complete.
- Confirmed the final supported generated API state is guarded local-only
  Doxygen HTML, current only after `make api-docs-freshness` passes.
- Confirmed `Makefile`, README, API reference, maintainer guide, and
  `scripts/check_api_docs_local_only.sh` consistently preserve the selected
  local-only path and unselected publication-mode non-claims.
- Ran `make api-docs-freshness`; it passed with 18 checked-in public headers,
  18 generated reference pages, 18 generated source pages, and local-only
  generated-output enforcement.
- Ran `make api-docs-local-only`; it passed.
- Ran the generated API claim scan over README, API reference, maintainer
  guide, Makefile, and the local-only guard script; matches were expected
  selected local-only wording or unrelated pre-existing report surfaces.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed.
- Confirmed `git status --ignored --short docs/api` reports `!! docs/api/`,
  which is the intended ignored local-output state.
- Recorded Sprint 174 handoff guidance: continue using
  `make api-docs-freshness` for local generated HTML or create a new
  publication decision before hosting, committing, or uploading generated API
  HTML.
- Day 14 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day14-sprint-closeout.md`.
