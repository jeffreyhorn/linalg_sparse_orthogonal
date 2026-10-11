# Sprint 213 Day 3: Publication Option Inventory

## Summary

Day 3 compares the four generated API policy options from the Epic 19 plan:
keep generated HTML local-only, publish hosted generated API HTML, retain a CI
generated-doc artifact, or commit generated HTML. This is an option inventory,
not the policy decision. Day 5 remains the decision point.

The current Day 2 baseline is a passing local-only chain with 18 checked-in
public headers, 18 generated reference pages, 18 generated source pages, and
ignored local output under `docs/api/`.

## Candidate Policies

| Option | Definition |
| --- | --- |
| Stronger local-only | Keep `docs/api/` ignored and untracked, keep `docs/api_reference.md` as the public source-controlled route, and close the publication residual with stronger guards/docs. |
| Hosted Pages publication | Generate Doxygen HTML in CI and publish it to a project-owned hosted documentation URL. |
| Retained generated-doc artifact | Generate Doxygen HTML in CI and upload a narrow retained artifact with explicit retention and non-release wording. |
| Committed generated HTML | Commit generated `docs/api/` output to the repository and enforce freshness against headers/Doxyfile. |

## Option Comparison

| Criteria | Stronger local-only | Hosted Pages publication | Retained generated-doc artifact | Committed generated HTML |
| --- | --- | --- | --- | --- |
| Discoverability | Lowest; users rely on source docs or local generation. | Highest; users can browse hosted generated docs. | Medium for maintainers/reviewers with artifact access. | Medium; browsable in the repository. |
| Freshness proof | Existing `make api-docs-freshness` already proves local current output. | Needs deployment-after-freshness proof and stale-site rollback. | Needs upload-after-freshness proof and retention controls. | Needs committed-output freshness proof on every change. |
| Routing | Keep routing to `docs/api_reference.md` and headers. | Add a narrow hosted URL exception. | Usually keep user routing source-controlled; artifact route is maintainer-only if selected. | Decide whether generated pages become valid source-controlled routes. |
| Workflow impact | Minimal guard hardening. | New Pages/deploy job and permissions. | New exact upload-artifact path and retention policy. | Possible CI freshness check; no upload/deploy path. |
| Review impact | Low. | Medium. | Medium. | High; local generated tree is 214 files and about 3.1 MB. |
| Claim risk | Low. | High because hosted docs look official. | Medium-high because retained artifacts can look like release evidence. | Medium because committed generated files look source-owned. |

## Required Changes By Option

| Area | Stronger local-only | Hosted Pages publication | Retained generated-doc artifact | Committed generated HTML |
| --- | --- | --- | --- | --- |
| Workflow | Keep rejecting generated API output in uploads/deployments. | Add dedicated deploy job after freshness checks. | Add exact generated-doc upload path after freshness checks. | No publication workflow required; optionally check generated tree freshness. |
| Local-only guard | Harden current fail-closed behavior. | Replace or conditionally relax for selected deployment path. | Replace or conditionally relax for selected artifact path. | Replace ignore/staging requirement with committed-output freshness checks. |
| Routing guard | Continue rejecting generated/hosted links. | Allow only selected hosted project URL. | Keep user routes source-controlled; maybe allow maintainer artifact wording. | Decide whether committed generated pages are valid routes. |
| Docs | Explain no hosted docs and exact local generation path. | Document hosted route, freshness semantics, and non-claims. | Document artifact name, retention, retrieval, and non-claims. | Document committed generated tree ownership and source/header precedence. |
| Tests | Add stronger bypass regressions if needed. | Add deployment URL, workflow path, and stale-output regressions. | Add artifact path, retention, and broad upload regressions. | Add generated-tree freshness and review-noise guard tests. |

## Risks

| Risk | Affected options | Notes |
| --- | --- | --- |
| Stale generated output | Hosted, retained artifact, committed HTML | Publication makes stale output visible beyond the local checkout. |
| Unsupported support inference | Hosted, retained artifact, committed HTML | Users may infer API stability, release readiness, package support, or ABI promises from published generated docs. |
| Workflow broad path upload | Hosted, retained artifact | Existing guard coverage must stay fail-closed for broad `docs/` paths unless a narrow path is selected. |
| Route confusion | Hosted, committed HTML | Users can bypass `docs/api_reference.md` if generated pages become primary routes. |
| Review noise | Committed HTML | Current generated tree is many files and would persist in normal diffs. |

## Day 3 Outcome

All four publication options have documented automation, routing, retention,
freshness, documentation, and risk implications. No policy is selected yet.

Day 4 should turn this inventory into acceptance criteria and stop conditions.
Day 5 should then choose one policy and reject the alternatives with explicit
rationale.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

