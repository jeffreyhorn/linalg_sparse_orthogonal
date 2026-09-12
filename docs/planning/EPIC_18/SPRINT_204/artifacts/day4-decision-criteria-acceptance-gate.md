# Day 4: Decision Criteria And Acceptance Gate

**Sprint:** 204 - Generated API Publication Decision  
**Theme:** Convert product and maintainer tradeoffs into an explicit acceptance
gate for the selected policy.  
**Time estimate:** 12 hours  
**Branch:** `sprint-204`

## Purpose

Day 4 defines the gate that the Day 5 product decision must satisfy before
Sprint 204 selects hosted generated API publication, retained generated API
artifacts, committed generated output, or a stronger local-only policy.

The gate is deliberately policy-specific. It allows different implementation
paths, but it does not allow any selected path to bypass freshness, routing,
reviewability, rollback, or claim-boundary evidence.

## Universal Acceptance Criteria

Any selected generated API policy must satisfy all universal criteria:

| Criterion | Required evidence |
| --- | --- |
| Source-of-truth preservation | `docs/api_reference.md` and checked-in public headers under `include/` remain the declaration and call-site contract authority. |
| Freshness command or workflow | The selected generated-output path has a deterministic command or workflow that regenerates Doxygen output and checks the configured public-header source set. |
| Generated page coverage | The policy preserves or replaces the current 18 checked-in public header coverage check with an equal or stronger check. |
| Generated `sparse_version.h` treatment | Generated installed header behavior remains tied to `VERSION`, `include/sparse_version.h.in`, and install/package validation unless the policy explicitly changes Doxygen input scope and validates the result. |
| Link/routing clarity | README, INSTALL, `docs/api_reference.md`, and maintainer guide route users to an available API reference path and state whether generated HTML is local, retained, hosted, or committed. |
| Stale-output failure | Missing, stale, or unexpectedly tracked generated output fails clearly for the selected policy. |
| Claim boundary | Generated API docs do not imply dynamic ABI compatibility, shared-library support, package-manager distribution, broad platform parity, release evidence, broad API completeness, external-library parity, portable performance, or state-of-the-art status. |
| Review hygiene | Generated-output files are either intentionally absent from review diffs or intentionally present with a documented review and drift policy. |
| Rollback path | The selected policy has a documented way to remove stale generated output, disable accidental publication, or restore local-only behavior. |

## Hosted Publication Gate

Hosted generated API HTML may be selected only if all hosted-specific criteria
are accepted for Days 6-12 implementation:

| Area | Acceptance requirement |
| --- | --- |
| Hosted surface | The exact hosting mechanism is named, such as GitHub Pages or another explicit deployment target. |
| Workflow ownership | A workflow generates Doxygen HTML from the checked-out source and fails if generation, coverage, metadata, or upload/deploy steps fail. |
| Freshness metadata | Published output includes or is paired with source commit, branch/ref, Doxygen version, generation command, generated path, timestamp or run id, and support-tier/non-claim wording. |
| Link validation | Public docs link only to the selected hosted entry point and have a check or recorded validation that the link target is valid. |
| Artifact scope | The workflow publishes only the selected generated API tree and not unrelated build, report, benchmark, package, or private artifacts. |
| Access and retention | Public/private access expectation and retention/lifecycle behavior are documented. |
| Guard coverage | A workflow/publication guard rejects accidental hosted docs drift, wrong paths, missing metadata, or publication without freshness checks. |
| Rollback | The plan names how to disable publication and remove or supersede stale hosted output. |

Hosted publication fails the gate if it is only a README link, only a generated
local tree, or only a workflow upload without freshness, metadata, link, and
claim-boundary enforcement.

## Retained CI Artifact Gate

Retained generated API artifacts may be selected only if all artifact-specific
criteria are accepted for Days 6-12 implementation:

| Area | Acceptance requirement |
| --- | --- |
| Artifact lane | A workflow lane runs `make api-docs-freshness` or an equivalent generation, coverage, and policy check before upload. |
| Artifact naming | Artifact name, event scope, branch/ref scope, and retention days are documented and guarded. |
| Upload behavior | Upload uses fail-closed behavior for missing files and uploads only `docs/api/html/` plus any selected metadata. |
| Metadata | Artifact evidence identifies source commit, branch/ref, Doxygen version, command, support tier, and retained non-claims. |
| User wording | Public docs describe artifacts as retained generated views, not source-controlled API truth or release evidence. |
| Guard coverage | Workflow checks reject missing upload paths, missing retention metadata, broad artifact scopes, or uploads that bypass freshness. |
| Rollback | The plan names how to remove the artifact lane and restore local-only semantics. |

Retained artifacts fail the gate if they lack retention/metadata semantics, if
they are described as stable public docs, or if they upload generated output
without running the freshness/coverage path.

## Committed Generated Output Gate

Committed generated output may be selected only if all committed-output
criteria are accepted for Days 6-12 implementation:

| Area | Acceptance requirement |
| --- | --- |
| Ignore policy | `.gitignore` is deliberately changed so only the selected generated files become trackable. |
| Drift detection | A reproducible command fails when checked-in generated HTML differs from regenerated output for the same source. |
| Review policy | The sprint documents which generated Doxygen support files are committed and how reviewers should evaluate generated diffs. |
| Cleanup policy | The sprint documents how stale, obsolete, or removed generated files are deleted when public headers change. |
| Source-of-truth wording | Public docs state generated HTML is rendered output while public headers remain declaration truth. |
| Staging guard | Guards reject partial generated-output staging or source changes without matching regenerated output. |
| Rollback | The plan names how to restore ignored local-only generated output if committed docs prove too noisy. |

Committed generated output fails the gate if it simply commits `docs/api/html/`
without drift detection, review policy, cleanup rules, and changed claim
wording.

## Stronger Local-Only Gate

A stronger local-only policy may be selected only if all local-only criteria
are accepted for Days 6-12 implementation:

| Area | Acceptance requirement |
| --- | --- |
| Ignore preservation | `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` remain ignored generated output. |
| Freshness command | `make api-docs-freshness` remains the supported local Doxygen freshness and staging command. |
| Tracking guard | The local-only guard rejects tracked, staged, or visible non-ignored generated API files. |
| Workflow guard | Workflows do not publish, upload, deploy, or otherwise reference generated API output paths unless a future product decision changes policy. |
| Routing docs | README, INSTALL, `docs/api_reference.md`, and maintainer guide clearly route users to source-controlled docs and local Doxygen regeneration. |
| Residual wording | Hosted publication, retained artifacts, and committed generated output remain explicit residuals or rejected paths, not implied support. |
| Link check | Any links added for local generation or source-of-truth routing are checked or manually validated in Day 9/Day 12 evidence. |
| Rollback | If generated output is accidentally staged, tracked, or linked as hosted output, the guard explains how to restore local-only state. |

Stronger local-only policy fails the gate if it leaves ambiguous hosted,
artifact, committed-output, or release wording in user or maintainer docs.

## Minimum Validation By Policy

| Selected policy | Minimum validation before closeout |
| --- | --- |
| Hosted generated API HTML | `make docs-check`; publication workflow/static validation; generated-output metadata check; link check; workflow guard; `make api-docs-freshness` if local-only staging guard remains applicable or its replacement if not. |
| Retained CI artifact | `make docs-check`; artifact workflow/static validation; artifact metadata and retention check; link/routing check; workflow guard; `make api-docs-freshness` or replacement policy guard. |
| Committed generated output | `make docs-check`; generated-output drift check; partial-staging guard; `git diff --check`; docs routing check; full C gate only if `.c` or `.h` files change. |
| Stronger local-only policy | `make docs-check`; `make api-docs-freshness`; direct local-only guard if changed; link/routing check if added; workflow path guard; `git diff --check`. |

If any `.c` or `.h` file changes, the sprint must also run:

```bash
make format && make lint && make test
```

## Required Claim Boundaries

The selected policy must keep these boundaries visible in README, INSTALL,
`docs/api_reference.md`, maintainer guide, and closeout artifacts as
applicable:

- generated API HTML is rendered documentation, not the source of API truth;
- checked-in public headers under `include/` own declarations and call-site
  contracts;
- no dynamic ABI compatibility claim;
- no shared-library support claim;
- no package-manager distribution claim;
- no Homebrew/core, bottle, Linuxbrew, public tap, vcpkg, Conan, pkgsrc, or
  system package claim;
- no broad platform parity claim;
- no broad API completeness beyond the configured Doxygen input set;
- no release evidence claim unless a future release process explicitly owns
  generated API publication;
- no external-library parity, portable performance, or state-of-the-art claim.

## Rollback Rules

| Failure mode | Required rollback or stop rule |
| --- | --- |
| Doxygen generation fails | Stop policy implementation until generation is fixed or the blocker is recorded; do not publish or commit generated output. |
| Page coverage fails | Stop and fix Doxygen input/configuration/header docs or record a blocker; do not treat partial generated output as current. |
| Hosted publication fails | Disable or withhold hosted publication, keep docs pointing to source-controlled API reference/local generation, and record residual hosted evidence. |
| Retained artifact upload fails | Withhold artifact claims, keep docs local-only/source-controlled, and record the artifact lane as blocked. |
| Committed generated output drifts | Regenerate and commit matching output or revert to local-only generated output before closeout. |
| Generated output is accidentally staged under local-only policy | Unstage/remove generated output and rerun the local-only guard before closeout. |
| Workflow references generated API paths under local-only policy | Remove the workflow reference or explicitly reopen the product decision with hosted/artifact acceptance criteria. |
| Public docs overclaim ABI/package/platform/completeness support | Correct wording and rerun the relevant docs/claim guard before closeout. |
| Link validation fails | Fix or remove the link before closeout; do not document unavailable hosted/artifact paths. |

## Day 5 Decision Rule

Day 5 should select exactly one policy path. The default tie-breaker is:

1. prefer the policy that fully satisfies its acceptance gate with the smallest
   durable maintenance burden;
2. prefer source-of-truth clarity over generated-output convenience when user
   value is similar;
3. do not select hosted publication, retained artifacts, or committed generated
   output unless the sprint can also implement the required guards and
   validation evidence;
4. preserve stronger local-only policy if publication value is not high enough
   to justify new workflow, retention, deployment, or generated-diff ownership.

## Completion Criteria Review

| Day 4 criterion | Status |
| --- | --- |
| Item 204.1 has a concrete decision framework before the decision is made. | Complete; this artifact defines universal and policy-specific acceptance gates. |
| The selected policy cannot pass without matching validation coverage. | Complete; each policy has minimum validation and fail-closed requirements. |
| Generated API docs cannot imply unsupported ABI, package, or completeness claims. | Complete; required claim boundaries and rollback rules preserve those non-claims. |

## Changed Surfaces

- Added this Day 4 acceptance-gate artifact.
- Updated `WORKING_NOTES.md` with the Day 4 status, decision-log entry, and
  validation notes.

No source, public header, workflow, Makefile, Doxyfile, user-facing
documentation behavior, or generated-output policy changed on Day 4.
