# Sprint 179 Day 5: Publication Option Decision Matrix

**Sprint:** 179 - Generated API HTML Publication Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_179/`
**Status:** Complete

## Purpose

Compare the four generated API HTML product-status options before Day 6 makes
the formal product decision. Day 5 evaluates hosted publication, retained CI
artifact, committed generated output, and stronger local-only status against
the current evidence from Days 1-4.

## Decision Criteria

| Criterion | What good looks like |
| --- | --- |
| User value | Users can find the supported API path without confusing local-only output for published release evidence. |
| Maintenance cost | The path fits current Doxygen inputs, generated-output policy, and reviewer workflow without large permanent churn. |
| Reviewability | Source-controlled diffs stay focused on source-of-truth files, scripts, workflows, and metadata rather than large generated output. |
| Freshness | The supported path has a command or workflow that regenerates output and checks required pages. |
| Reproducibility | Maintainers can reproduce the supported state from checked-in commands and configuration. |
| CI complexity | Any hosted or artifact path has fail-closed workflow behavior and metadata, without depending on unstated infrastructure. |
| Claim fit | Public wording stays bounded to configured Doxygen inputs and does not imply package, ABI, platform, release, or state-of-the-art proof. |

## Evidence Inputs

| Evidence | Current finding |
| --- | --- |
| Day 1 baseline | Generated API HTML is currently local-only; all product-status options remain open. |
| Day 2 surface audit | Doxygen input is `include/*.h`, non-recursive, with ignored HTML output under `docs/api/html/`. |
| Day 3 warning/coverage audit | `make docs-check` and `make api-docs-freshness` pass; 18 of 18 configured public headers have generated reference and source pages; no Doxygen warning lines were observed. |
| Day 4 guard/CI audit | Current local-only guard is strong locally; no workflow runs or uploads generated API HTML, deploys Pages, or writes publication metadata. |
| Epic 16 review | The gap is discoverability: generated HTML exists locally but has no stable hosted URL, artifact-retention policy, or publication workflow. |

## Option Matrix

Scores use `High`, `Medium`, and `Low` for fit against the Day 5 criteria.

| Option | User value | Maintenance cost | Reviewability | Freshness | Reproducibility | CI complexity | Claim fit | Day 5 assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hosted publication | High | Medium | Medium | Medium | Medium | High complexity | Medium | Best user discoverability, but requires new publication workflow, metadata, deployment guard, and hosted-status wording. Not a small closure from current state. |
| Retained CI artifact | Medium | Medium | High | High if implemented | High if implemented | Medium complexity | High | Good evidence path without permanent generated diffs, but still requires CI lane, upload policy, metadata, and workflow guard. |
| Committed generated output | Medium | High cost | Low | Medium | High | Low complexity | Low to medium | Makes HTML directly browsable in source, but conflicts with current ignored-output policy and creates noisy generated diffs. |
| Stronger local-only status | Medium | Low | High | High locally | High locally | Low | High | Fits current policy and evidence, closes ambiguity by strengthening guards/docs, but explicitly leaves no hosted browsable generated API. |

## Option Details

### Hosted Publication

Hosted generated API HTML provides the strongest user-facing discoverability:
users would have a stable published generated reference instead of running
Doxygen locally. However, Day 4 found no existing Pages deployment, generated
API artifact path, publication metadata, or workflow guard.

Required implementation if selected:

- add a hosted publication workflow or deployment path;
- generate HTML from the checked-out source;
- upload or deploy only freshly generated `docs/api/html/`;
- write source commit, branch, command, support tier, and claim-boundary
  metadata;
- add fail-closed checks for missing pages and missing publication artifacts;
- update README, API reference, maintainer guide, and support-tier wording to
  distinguish hosted generated API docs from release or completeness proof.

Day 5 rationale:

- good long-term user value;
- too much new infrastructure for a low-risk Sprint 179 closure unless Day 6
  deliberately chooses to spend the rest of the sprint on workflow and metadata
  hardening.

### Retained CI Artifact

A retained generated API artifact is a middle path: CI can produce a browsable
artifact with commit metadata without requiring a public hosted site. It would
give reviewers and maintainers a CI-produced generated view, but users still
would not have a stable documentation URL.

Required implementation if selected:

- add a generated API docs CI lane;
- run `make docs-check` or `make api-docs-freshness` in that lane;
- upload `docs/api/html/` with `if-no-files-found: error`;
- define retention days and artifact name;
- include source commit, branch, generator command, support tier, and claim
  boundary metadata;
- add a workflow guard so the artifact lane cannot drift silently.

Day 5 rationale:

- more reviewable than committed output;
- less user-facing than hosted publication;
- requires workflow and metadata work that the current repository does not yet
  own.

### Committed Generated Output

Committing `docs/api/html/` would make generated HTML visible in source
checkout and reviewable by file presence, but it conflicts directly with the
current local-only guard and `.gitignore` policy.

Required implementation if selected:

- remove or narrow the `docs/api/` ignore rule;
- replace `scripts/check_api_docs_local_only.sh` with committed-output
  freshness checks;
- add generated-output drift detection;
- decide how to review large generated HTML/CSS/JS diffs;
- prevent source comments and generated HTML from diverging.

Day 5 rationale:

- highest diff noise and review burden;
- weak fit for the repository's existing source-of-truth model;
- not recommended unless a future project explicitly values source-controlled
  generated docs over small, reviewable diffs.

### Stronger Local-Only Status

Stronger local-only status keeps public headers and Markdown docs as the
source-controlled API surface while making the generated HTML status
unambiguous. It matches the existing `docs/api/` ignore policy and current
`make api-docs-freshness` guard.

Required implementation if selected:

- write a Day 6 product decision that explicitly chooses local-only generated
  API HTML;
- tighten local-only wording in README, API reference, maintainer guide, and
  support-tier docs if needed;
- preserve `docs/api/` as ignored generated output;
- strengthen or document the local-only guard contract;
- optionally add a lightweight checked-in guard for docs wording or generated
  output policy drift;
- keep non-claims for hosted publication, retained artifact publication,
  committed output, release evidence, and completeness beyond configured
  Doxygen inputs.

Day 5 rationale:

- strongest fit to current evidence and guard behavior;
- lowest implementation risk;
- closes ambiguous status cleanly;
- leaves the discoverability residual explicit instead of pretending local HTML
  is a hosted publication.

## Recommended Decision Candidate

Day 5 recommends **stronger local-only status** as the Day 6 product-decision
candidate.

Reasons:

1. It aligns with the current working guard: `make api-docs-freshness` already
   regenerates, checks configured header page coverage, and enforces ignored,
   untracked, unstaged generated output.
2. It preserves reviewability by keeping generated HTML out of source diffs.
3. It avoids introducing a partially governed hosted or artifact path without
   metadata, retention policy, workflow guard, and deployment ownership.
4. It keeps public headers and Markdown docs as source-of-truth surfaces.
5. It can still improve user clarity by making the supported path explicit:
   use `docs/api_reference.md` and public headers for checked-in declarations;
   use `make api-docs-freshness` for local generated HTML inspection.

## Rejected-Option Rationale For Day 6 Consideration

| Option | Rejection rationale if stronger local-only is selected |
| --- | --- |
| Hosted publication | Valuable but not currently owned by workflow, metadata, or deployment infrastructure. Select only if Sprint 179 is willing to add and guard those surfaces now. |
| Retained CI artifact | Useful for reviewer evidence, but still requires a new CI lane, artifact naming/retention, metadata, and workflow guard. |
| Committed generated output | Conflicts with current ignore policy and would create large generated diffs while weakening the current source-of-truth model. |

## Implementation Planning Notes If Recommendation Is Accepted

If Day 6 accepts stronger local-only status, Days 7-12 should focus on:

- documenting the product decision and non-claims;
- making the local-only guard contract easy to find and hard to bypass;
- preserving `docs/api/` ignored generated-output behavior;
- checking README/API reference/maintainer wording for any hosted or
  source-controlled implication;
- adding a focused drift guard only if it materially improves enforcement
  without widening scope into hosted publication.

## Day 5 Deliverables

- publication option criteria
- option-by-option tradeoff matrix
- recommended decision candidate
- rejected-option rationale
- Day 5 publication decision matrix artifact

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every project-plan option is evaluated. | Complete | Hosted, retained artifact, committed output, and stronger local-only options are each evaluated above. |
| Tradeoffs are concrete enough for implementation planning. | Complete | Each option lists required implementation follow-through and decision rationale. |
| Rejected options have evidence-backed rationale. | Complete | Rejection rationale is tied to Days 1-4 evidence and current workflow/guard gaps. |
