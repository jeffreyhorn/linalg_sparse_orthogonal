# Day 3: Publication Option Inventory

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Compare hosted, retained-artifact, committed-output, and stronger
local-only generated API policy options.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`

## Purpose

Day 3 compares all generated API policy paths named by Sprint 204 item 204.1
before the sprint defines an acceptance gate or makes a product decision. This
artifact uses the Day 2 baseline as the current evidence floor:

- `make docs-check` passed;
- `make api-docs-freshness` passed;
- Doxygen `1.16.1` generated local HTML under `docs/api/html/`;
- coverage reported 18 checked-in public headers, 18 generated reference
  pages, and 18 generated source pages;
- `docs/api/` remained ignored, untracked, unstaged, and invisible as
  non-ignored untracked output.

## Decision Criteria

| Criterion | What good looks like |
| --- | --- |
| User value | Users can find and trust the supported API path without mistaking generated output for broader release, package, ABI, or completeness proof. |
| Implementation cost | The policy can be implemented with bounded changes to existing docs, guards, workflows, and generated-output ownership. |
| Maintenance cost | The policy does not create recurring manual cleanup, stale-output triage, deployment repair, or generated-diff review burden beyond its value. |
| Reviewability | Pull requests remain focused on source-of-truth inputs, scripts, workflows, and policy docs instead of large generated HTML churn. |
| Freshness | The supported generated-output path has a deterministic command or workflow that regenerates output and fails clearly when stale or incomplete. |
| Link/routing integrity | User-facing entry points route to available docs, local generation instructions, retained artifacts, or hosted output with clear source-of-truth semantics. |
| Claim fit | The policy preserves non-claims for ABI, package-manager distribution, broad platform parity, release evidence, broad API completeness, and state-of-the-art coverage. |

## Evidence Inputs

| Evidence | Day 3 relevance |
| --- | --- |
| Sprint 204 Day 1 | Identified current policy surfaces and kept all four product options open. |
| Sprint 204 Day 2 | Proved the current local-only Doxygen/freshness path passes and generated output remains ignored local state. |
| Sprint 158 | Established local-only generated API HTML with page coverage and no committed/hosted generated HTML. |
| Sprint 179 | Strengthened local-only generated API status with staging and workflow-publication guards. |
| Sprint 186 | Kept `R186-HOSTED-API` open for any future hosted, retained-artifact, or committed-output product decision. |
| Epic 18 retrospective | Reopened generated API publication policy as a future closure candidate without changing the existing local-only support tier. |

## Option Matrix

Scores use `High`, `Medium`, and `Low` as fit ratings for the criterion. For
implementation and maintenance cost, `Low` means less cost.

| Option | User value | Implementation cost | Maintenance cost | Reviewability | Freshness | Link/routing integrity | Claim fit | Day 3 assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hosted generated API HTML | High | High | Medium to high | High if generated output is not committed | Medium until guarded | Medium until link checks exist | Medium until wording/guards exist | Strongest discoverability, but requires new publication ownership, deployment/freshness metadata, link validation, workflow guards, and support-tier wording. |
| Retained CI artifact | Medium | Medium | Medium | High | High if artifact lane runs `make api-docs-freshness` | Medium | High if scoped carefully | Useful reviewer/maintainer evidence without generated diffs, but still requires artifact naming, retention, metadata, and guard coverage. |
| Committed generated output | Medium | Medium | High | Low | Medium if drift checks exist | High for source browsing | Low to medium | Technically straightforward, but conflicts with the ignored-output model and creates large stale generated diffs. |
| Stronger local-only policy | Medium | Low | Low | High | High locally | Medium to high if routing docs improve | High | Best fit to current evidence; can close ambiguity with stronger guard/docs/link discipline while preserving generated output as local-only. |

## Hosted Generated API HTML

Hosted publication would provide a stable browsable generated API reference for
users who do not want to run Doxygen locally.

Required implementation if selected:

- choose a hosted surface, such as Pages or another explicit deployment path;
- add a workflow that generates HTML from the checked-out source;
- run `make docs-check` or an equivalent freshness/coverage command before
  publication;
- publish only freshly generated `docs/api/html/`;
- record source commit, branch, Doxygen version, command, generated path,
  support tier, and non-claim metadata;
- add fail-closed checks for missing pages, missing `index.html`, stale output,
  unexpected source sets, and broken entry links;
- update README, INSTALL, `docs/api_reference.md`, and
  `docs/maintainer_guide.md` to distinguish hosted generated HTML from API
  source of truth, release evidence, package proof, ABI proof, or broad API
  completeness.

Main risks:

- hosted URLs can outlive the branch evidence that generated them;
- workflow publication can fail silently without structured guard coverage;
- public docs may imply stronger support than the generated output proves;
- Pages or deployment permissions add operational ownership.

Preliminary Day 3 disposition: viable only if Day 4 acceptance criteria require
and Day 5 selects a full publication workflow plus freshness, metadata, and
link validation. It should not be implemented as a lightweight docs-only
change.

## Retained CI Artifact

A retained artifact would make generated API HTML available from CI without
publishing a stable public documentation site.

Required implementation if selected:

- add a workflow lane that runs `make api-docs-freshness` or equivalent;
- upload `docs/api/html/` with `if-no-files-found: error`;
- define artifact name, retention days, branch/event scope, and access
  expectations;
- include or generate metadata for source commit, branch, Doxygen version,
  command, and support tier;
- add workflow guard coverage for artifact path, retention, and selected
  generated API output semantics;
- update public and maintainer docs so users understand artifact output is
  retained evidence, not source-controlled API truth or a release guarantee.

Main risks:

- artifacts are less discoverable than a stable hosted URL;
- retention policy may create disappearing links;
- artifact access differs for forks, private repositories, and historical
  workflow runs;
- reviewers may treat artifact existence as broader publication or release
  evidence.

Preliminary Day 3 disposition: viable if Sprint 204 values reviewer and
maintainer evidence more than public discoverability. It still needs workflow
and metadata ownership, so it is not equivalent to the existing local-only
policy.

## Committed Generated Output

Committed output would place generated API HTML directly in the repository.

Required implementation if selected:

- remove or narrow the `docs/api/` ignore rule;
- replace or substantially revise `scripts/check_api_docs_local_only.sh`;
- add generated-output drift detection so public headers and generated HTML
  cannot diverge silently;
- decide whether generated CSS, JavaScript, images, search data, and all
  Doxygen support files are committed;
- document regeneration discipline and review expectations;
- update docs to explain generated HTML is source-controlled rendered output,
  while checked-in public headers remain the declaration source of truth.

Main risks:

- large generated diffs obscure source-of-truth changes;
- generated HTML can go stale when comments or Doxygen configuration change;
- merge conflicts and review noise increase permanently;
- the current local-only guard and ignore policy would need to be unwound;
- committed rendered docs may be misread as API completeness or release proof.

Preliminary Day 3 disposition: technically possible but weakest fit. It should
be selected only if the project explicitly prefers browsable source-tree HTML
over review clarity and stale-output containment.

## Stronger Local-Only Policy

Stronger local-only policy would keep the existing product shape: public
headers and Markdown docs are source-controlled, and generated Doxygen HTML is
local ignored output refreshed with `make api-docs-freshness`.

Required implementation if selected:

- record a Day 5 decision that explicitly keeps generated API HTML local-only;
- preserve `docs/api/` ignore behavior;
- keep `make api-docs-freshness` as the supported freshness/staging command;
- strengthen local-only guard coverage if gaps are found, especially around
  workflow publication references or link/routing drift;
- improve README, INSTALL, API reference, and maintainer wording if the
  current policy is still hard to find;
- document residual hosted/artifact/committed publication as intentionally
  deferred unless a future sprint selects that path.

Main risks:

- users still do not have a stable hosted generated API URL;
- local Doxygen remains a user prerequisite for rendered HTML;
- link/routing documentation must be clear enough that local-only output does
  not look missing or unsupported by accident.

Preliminary Day 3 disposition: strongest fit to the Day 2 baseline and current
repository architecture. It preserves reviewability, minimizes new workflow
surface area, and can still improve usability by making source-of-truth and
local generation routes clearer.

## Preliminary Rejection Notes

No option is formally rejected on Day 3. The following notes identify what
would have to be true for rejection on Day 5:

| Option | Rejection basis if not selected |
| --- | --- |
| Hosted generated API HTML | Reject if the sprint does not fund deployment ownership, freshness metadata, link checks, and structured workflow guard coverage. |
| Retained CI artifact | Reject if artifact retention/discoverability adds maintenance burden without enough user value or if workflow metadata cannot be guarded cleanly. |
| Committed generated output | Reject if preserving small reviewable diffs and source-of-truth headers remains more important than checked-in rendered HTML. |
| Stronger local-only policy | Reject only if Sprint 204 explicitly values public browsability or retained CI evidence enough to own the extra publication infrastructure. |

## Day 4 Inputs

Day 4 should convert this inventory into an acceptance gate. Minimum inputs:

- exact pass/fail criteria for selecting any publication path;
- required metadata fields for hosted or retained artifacts;
- required drift checks for committed generated output;
- required guard/doc improvements for stronger local-only policy;
- link validation expectations for user-facing generated API routes;
- non-claim wording for ABI, package-manager distribution, platform parity,
  release evidence, broad API completeness, and state-of-the-art coverage.

## Completion Criteria Review

| Day 3 criterion | Status |
| --- | --- |
| Each viable policy path has concrete implementation and validation needs. | Complete; every option lists required implementation follow-through and validation needs. |
| Publication choices are separated from API completeness and ABI claims. | Complete; claim-fit scoring and non-claim requirements keep generated output separate from ABI/package/completeness evidence. |
| Review burden and stale-output risks are documented for every option. | Complete; each option records reviewability and stale-output risks. |

## Changed Surfaces

- Added this Day 3 option-inventory artifact.
- Updated `WORKING_NOTES.md` with the Day 3 status and summary.

No source, public header, workflow, Makefile, Doxyfile, user-facing
documentation behavior, or generated-output policy changed on Day 3.
