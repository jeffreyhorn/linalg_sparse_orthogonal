# Day 3: Publication Decision Matrix

## Purpose

Compare generated API HTML publication modes before Sprint 173 formalizes a
Day 4 decision and starts implementation.

## Baseline From Day 2

| Field | Current Day 2 state |
| --- | --- |
| Generator | Doxygen via `Doxyfile` |
| Local generator target | `make docs` |
| Local validation target | `make docs-check` |
| Coverage check | `scripts/check_api_docs_coverage.py` |
| Generated output path | `docs/api/html/` |
| Generated output tracking | ignored by `.gitignore` through `docs/api/` |
| Local generated file count | 214 files under `docs/api/html/` |
| Checked-in public headers covered locally | 18 reference pages and 18 source pages |
| CI publication references | none found for Doxygen, `docs-check`, or `docs/api/html/` |
| Current docs policy | local-only generated HTML; exact API declarations owned by checked-in headers and `docs/api_reference.md` |

Sprint 158 already selected guarded local-only generated API HTML. Sprint 173
should only move beyond that policy if the new publication path has explicit
freshness, ownership, staging, retention, and claim-boundary gates.

## Candidate Modes

| Mode | Description | Current repository fit |
| --- | --- | --- |
| Hosted site | Generate Doxygen HTML and publish it to a stable web location. | Not wired; needs deployment, permissions, retention, URL, and support-tier policy. |
| Committed HTML | Regenerate `docs/api/html/` and commit generated files. | Conflicts with current ignore policy; adds large generated review surface. |
| CI artifact-only | Generate Doxygen HTML in CI and upload an artifact for reviewed runs. | Not wired; less discoverable than hosted docs and still needs retention/branch semantics. |
| Guarded local-only | Keep `docs/api/` ignored, use `make docs-check`, and strengthen freshness/staging enforcement. | Matches current policy and repo shape. |

## Decision Matrix

| Evaluation field | Hosted site | Committed HTML | CI artifact-only | Guarded local-only |
| --- | --- | --- | --- | --- |
| Repository size impact | none | high: generated tree becomes tracked | none | none |
| Reviewability | source diffs stay clean, but evidence moves to hosted logs/deploys | poor for ordinary header/doc edits because generated churn dominates | source diffs stay clean, evidence in CI artifacts | best source review, no generated churn |
| User discoverability | strongest if URL is stable and documented | moderate through repository browser | weak to moderate; artifacts are run-scoped | moderate; users run local command and use source docs |
| CI reliability need | high | low to moderate | high | optional/local unless promoted later |
| Freshness enforcement | must bind URL/deploy to commit and branch | must require regenerated committed output with source changes | must bind artifact to commit/run | must require local regeneration/check before claiming current output |
| Release-process burden | high | high | medium | low |
| Current implementation work | large | medium to large | medium | small to medium |
| Claim risk | high if URL retention or branch semantics are vague | high if generated output drifts or is treated as release evidence | medium if artifacts are transient or unavailable | low if local-only wording remains explicit |
| Fits Sprint 173 scope | only if publication infrastructure is intentionally selected | only if source-controlled generated output is intentionally selected | possible, but not present today | best fit |

## Mode-Specific Requirements

### Hosted Site

Required before support:

- workflow or deployment command that runs `make docs-check`;
- artifact or deployed content tied to commit SHA and branch;
- stable URL ownership and publication permissions;
- retention or replacement policy;
- public docs naming exactly what the hosted docs prove;
- failure semantics for missing deploys, stale deploys, and warning output;
- claim scan preventing package, ABI, platform, performance, external-parity,
  and state-of-the-art overreach.

Day 3 disposition: defer. No hosted Doxygen lane exists today.

### Committed HTML

Required before support:

- deliberate `.gitignore` change or narrow exception for selected generated
  files;
- `make docs-check` before every generated HTML update;
- generated-output diff review rule;
- source-to-output freshness check or explicit generated-output sync policy;
- documentation explaining that committed HTML is only a generated view of the
  configured checked-in public-header input set;
- guard preventing accidental unrelated generated files.

Day 3 disposition: reject for Sprint 173. The generated tree is currently 214
ignored files and would add high review churn without solving broader API,
package, ABI, platform, or performance claims.

### CI Artifact-Only

Required before support:

- workflow lane that installs Doxygen and runs `make docs-check`;
- artifact upload step for `docs/api/html/`;
- artifact naming that includes commit, branch, and generator context;
- retention policy;
- documentation explaining artifact lookup and expiration;
- fallback path for users when artifacts expire;
- claim boundary wording that artifact-only output is not hosted docs or
  release evidence.

Day 3 disposition: possible future path, but defer for Sprint 173 unless Day 4
chooses to add CI scope. It does not improve source-controlled discoverability
enough to justify new CI surface today.

### Guarded Local-Only

Required before support:

- preserve `docs/api/` ignored status;
- keep `docs/api_reference.md` and public headers as source-controlled API
  reference truth;
- require `make docs-check` before saying local generated HTML is current;
- add or strengthen a freshness/staging check so ignored generated HTML is not
  accidentally staged or represented as source-controlled;
- keep `sparse_version.h` outside Doxygen page expectations under current
  input policy;
- update docs only if the selected freshness/staging behavior changes.

Day 3 disposition: recommended for Day 4 decision.

## Maintenance And CI Risk Assessment

| Risk | Hosted site | Committed HTML | CI artifact-only | Guarded local-only |
| --- | --- | --- | --- | --- |
| Stale generated docs | medium unless deploy freshness is strict | medium unless generated diffs are required | medium unless artifact freshness is strict | low when docs say current only after local check |
| Accidental overclaim | high | medium | medium | low |
| Review noise | low | high | low | low |
| Infra failure | high | low | medium | low |
| User confusion | low if URL is stable; high if stale | medium because generated files look authoritative | medium because artifacts expire | low if docs keep local-only wording clear |
| Implementation complexity | high | medium | medium | low |

## Claim Risks To Block

Any selected path must keep these non-claims explicit:

- no generated API HTML freshness without the selected check passing;
- no generated installed-header Doxygen coverage for `sparse_version.h` under
  current input policy;
- no package-manager provider support;
- no shared-library support;
- no dynamic ABI stability;
- no runtime-loader behavior support;
- no Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- no broad platform parity;
- no portable performance guarantee;
- no external-library parity;
- no state-of-the-art sparse linear algebra claim.

## Recommended Day 4 Decision

Recommend that Day 4 formally select **guarded local-only generated API HTML
with strengthened freshness/staging enforcement**.

This means Sprint 173 should:

1. keep `docs/api/` ignored;
2. keep `docs/api_reference.md` plus checked-in public headers as
   source-controlled API truth;
3. keep `make docs-check` as the local generator plus coverage command;
4. add or document a focused freshness/staging gate that confirms generated
   API HTML is local-only and not staged unintentionally;
5. preserve `sparse_version.h` installed-header ownership outside current
   Doxygen page coverage;
6. avoid adding hosted, committed, or artifact-only publication claims.

## Day 4 Handoff

Day 4 should convert this recommendation into a formal decision record with:

- selected mode: guarded local-only;
- supported claims: local generated Doxygen HTML is a convenience view for the
  active checkout after `make docs-check` passes;
- unsupported claims: hosted, committed, artifact-only, package, ABI,
  platform, performance, external-parity, and state-of-the-art claims;
- implementation scope for Days 5 through 9: freshness/staging gate design,
  enforcement, and docs navigation alignment if needed.

## Completion Check

Day 3 completion criteria are met:

- all viable publication modes have explicit tradeoffs;
- no publication support claim is made before a decision record exists;
- selected-mode prerequisites are clear for Day 4.

No `.c` or `.h` files changed on Day 3, so the full C quality gate is not
required for this day.
