# Day 5 Publication Options

## Scope

Day 5 compares the viable generated API HTML publication paths before Sprint
158 implements a concrete policy. The options are:

1. commit generated `docs/api/html/` into source control;
2. publish generated HTML through CI artifacts or hosted pages;
3. keep generated HTML local-only while adding a recurring freshness, warning,
   and page-coverage guard.

This artifact recommends a product direction but does not edit `.gitignore`,
CI workflows, public docs, public headers, or generated HTML.

## Current Baseline

| Field | Current state |
| --- | --- |
| Generation command | `make docs` -> `doxygen Doxyfile` |
| Generated output path | `docs/api/html/` |
| Tracking policy | `.gitignore` ignores `docs/api/` |
| Generated output size | Approximately 3.1 MB under `docs/api/` |
| Generated file count | 212 files under `docs/api/html/` after Day 2 `make docs` |
| Header page coverage | 18 of 18 checked-in public headers have reference and source pages |
| Doxygen warnings | 10 warnings remain selected for Sprint 158 closure |
| Existing docs workflow | No current workflow publishes generated Doxygen HTML |
| Source of truth | Checked-in public headers remain exact declaration authority |

## Option 1: Commit `docs/api/html/`

| Evaluation field | Assessment |
| --- | --- |
| Description | Remove or narrow the `.gitignore` rule for `docs/api/`, regenerate Doxygen HTML, and commit the generated tree. |
| Evidence strength | Strongest source-controlled review visibility for generated HTML at a specific commit. |
| Cost | Adds approximately 3.1 MB and 212 generated files to the review surface today, with future diffs dominated by generator churn. |
| Freshness behavior | Fresh only when regenerated and committed with matching source/header changes. |
| Review risk | High. Generated output can obscure source/header edits and make ordinary API comment changes noisy. |
| Maintenance burden | High. Every relevant header/comment change must include regenerated output or explicitly mark it stale. |
| Required validation | `make docs`, warning closure, public-header page coverage, generated-output diff review, docs hygiene, and full C/header gate if headers change. |
| Claim boundary | Committed HTML is a generated view of configured headers only; it is not broader API, ABI, platform, package, parity, performance, or state-of-the-art evidence. |
| Day 5 disposition | Rejected for Sprint 158 unless Day 6 overrides with a dedicated generated-reference refresh policy. |

### Rejection Rationale

Committed HTML would close the "where is the generated output" question, but it
does so by making generated artifacts part of every relevant review. The
project's stronger and more maintainable authority is the checked-in public
header set plus a reproducible generator and coverage guard. Source-controlled
HTML is not needed to prove Day 3 page coverage or Day 4 warning closure.

## Option 2: CI-Published Or Hosted Generated HTML

| Evaluation field | Assessment |
| --- | --- |
| Description | Add a workflow that runs `make docs`, validates warnings/page coverage, and uploads or publishes generated HTML. |
| Evidence strength | Strong hosted freshness signal when tied to a specific commit and retained artifact/page URL. |
| Cost | Requires workflow design, artifact retention policy, branch trigger policy, publication permissions, and support-tier wording. |
| Freshness behavior | Fresh for the hosted run or published branch context, subject to retention and deployment semantics. |
| Review risk | Medium. Avoids generated file churn but moves evidence into hosted logs/artifacts. |
| Maintenance burden | Medium to high. Requires CI failure semantics, retention policy, and public docs explaining where generated HTML lives. |
| Required validation | Local `make docs`, warning/page checks, workflow dry-run or hosted result, artifact upload/deploy verification, docs alignment. |
| Claim boundary | Hosted HTML is generated docs evidence for the selected branch/run only; not broad hosted report promotion or package/platform proof. |
| Day 5 disposition | Deferred. Candidate for a future hosted-docs/product-docs sprint, not selected for immediate Sprint 158 closure. |

### Deferral Rationale

CI publication is valuable but introduces infrastructure and retention
questions that are broader than this sprint's immediate residual: the project
needs explicit generated API policy now. Sprint 159 already owns hosted
generated-report promotion; adding hosted Doxygen publication here would blur
that boundary unless the team deliberately funds hosted docs separately.

## Option 3: Guarded Local-Only Generated HTML

| Evaluation field | Assessment |
| --- | --- |
| Description | Keep `docs/api/` ignored, keep generated HTML local-only, and add/document recurring checks for `make docs`, Doxygen warnings, and public-header page coverage. |
| Evidence strength | Strong local reproducibility and reviewable policy; no source-controlled generated output claim. |
| Cost | Low repository churn. Requires warning closure, page-coverage guard, and docs wording updates. |
| Freshness behavior | Fresh only for the local branch/run that executed the guard. Public docs must say generated HTML is local-only unless independently published. |
| Review risk | Low. Reviews focus on source headers, Doxygen config, guards, and docs policy. |
| Maintenance burden | Moderate. Maintainers must run the guard for relevant header/docs changes, but they do not review generated HTML churn. |
| Required validation | `make docs`, zero or triaged warnings, public-header page-coverage guard, docs hygiene, and full C/header gate if headers change. |
| Claim boundary | Generated HTML is a local convenience view. Public headers and `docs/api_reference.md` remain the source-controlled API reference. |
| Day 5 disposition | Recommended for Sprint 158. |

### Recommendation Rationale

Guarded local-only output best matches the current repository shape:

- `.gitignore` already treats `docs/api/` as generated output;
- the current generated tree is nontrivial review noise at 212 files and about
  3.1 MB;
- Day 3 already proves page coverage can be checked from generated output
  without committing it;
- Day 4 makes warning closure a prerequisite for any freshness claim;
- source-header-first API ownership remains clearer when generated HTML is a
  reproducible local view rather than a parallel reviewed artifact.

## Recommended Product Direction

Sprint 158 should select **guarded local-only generated API HTML** unless Day 6
finds a blocking reason to override this recommendation.

The implementation should:

1. keep `docs/api/` ignored;
2. add or document a deterministic page-coverage guard for the 18 checked-in
   public headers;
3. close the selected Doxygen warnings or keep publication/freshness claims
   blocked;
4. update `docs/api_reference.md` and `docs/maintainer_guide.md` so they no
   longer require committed generated HTML as the only freshness path;
5. describe generated HTML as local-only unless a future hosted/publication
   sprint adds CI publication or committed output;
6. preserve source-header-first authority for exact declarations and call-site
   contracts.

## Claim Boundaries

| Claim surface | Day 5 boundary |
| --- | --- |
| Generated HTML freshness | Fresh only for a specific local `make docs` run plus warning/page checks. |
| Source-controlled API reference | `docs/api_reference.md` and checked-in public headers remain source-controlled truth. |
| Generated HTML completeness | Bounded to configured `Doxyfile` input and page-coverage guard. |
| Generated version header | Separate installed-header/version metadata policy row; no missing-page claim under current config. |
| Hosted evidence | Not selected by Day 5. |
| Broader product claims | No dynamic ABI, shared-library, package-manager, broad platform, external parity, portable performance, or state-of-the-art claim. |

## Day 6 Handoff

Day 6 should turn this recommendation into an implementation checklist:

1. final decision: guarded local-only generated API HTML;
2. files likely to change: `docs/api_reference.md`,
   `docs/maintainer_guide.md`, and possibly a page-coverage check script or
   Make target;
3. files likely not to change: `.gitignore` should continue ignoring
   `docs/api/`; CI workflows should not publish Doxygen HTML unless Day 6
   overrides the recommendation;
4. stale-output rule: generated HTML is local-only and never cited as
   source-controlled freshness evidence;
5. validation rule: `make docs`, warning closure, page-coverage guard,
   `git diff --check`, direct whitespace scan for new docs, and full C/header
   gate for any public header edit.

## Completion Check

- Each viable publication path has explicit cost and evidence implications.
- Committed and CI-published HTML paths are rejected or deferred with reasons.
- Guarded local-only generated HTML is recommended without overclaiming
  generated documentation freshness.
- Day 6 has a concrete policy direction to implement.
