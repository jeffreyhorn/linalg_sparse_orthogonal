# Sprint 204 Working Notes: Generated API Publication Decision

**Sprint:** 204 - Generated API Publication Decision  
**Branch:** `sprint-204`  
**Base commit:** `cb4371b3`  
**Plan:** [PLAN.md](./PLAN.md)  
**Epic source:** [EPIC_18/PROJECT_PLAN.md](../PROJECT_PLAN.md)

## Sprint Goal

Decide and implement either hosted generated API publication or a stronger
local-only generated API policy, then align freshness, routing documentation,
claim-boundary guards, and validation evidence with the selected policy.

## Item Checklist

| Item | Description | Status | Evidence path |
| --- | --- | --- | --- |
| 204.1 | Product Decision | Complete | Day 1 intake; Day 3 option matrix; Day 4 acceptance gate; [Day 5 decision artifact](./artifacts/day5-product-decision.md) |
| 204.2 | Publication Or Guard Implementation | Complete | [Day 6 implementation design](./artifacts/day6-workflow-tracking-design.md); [Day 7 implementation batch](./artifacts/day7-policy-implementation-batch.md) |
| 204.3 | Freshness And Link Checks | Complete | [Day 8 freshness/coverage artifact](./artifacts/day8-freshness-and-coverage-checks.md); [Day 9 routing/link artifact](./artifacts/day9-link-and-routing-validation.md) |
| 204.4 | API Routing Docs | Complete | [Day 10 user-facing API docs artifact](./artifacts/day10-user-facing-api-docs-update.md) |
| 204.5 | Claim Boundary Guard | Complete | [Day 11 maintainer/claim-boundary artifact](./artifacts/day11-maintainer-and-claim-boundary-docs.md) |
| 204.6 | Validation | Complete | [Day 12 integrated validation](./artifacts/day12-integrated-validation.md); [Day 13 hardening](./artifacts/day13-review-hardening.md); [Day 14 closeout](./artifacts/day14-closeout-review.md) |

## Day Status Ledger

| Day | Title | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Generated API Intake | Complete | [day1-generated-api-intake.md](./artifacts/day1-generated-api-intake.md) |
| 2 | Current Doxygen Baseline | Complete | [day2-current-doxygen-baseline.md](./artifacts/day2-current-doxygen-baseline.md) |
| 3 | Publication Option Inventory | Complete | [day3-publication-option-inventory.md](./artifacts/day3-publication-option-inventory.md) |
| 4 | Decision Criteria And Acceptance Gate | Complete | [day4-decision-criteria-acceptance-gate.md](./artifacts/day4-decision-criteria-acceptance-gate.md) |
| 5 | Product Decision | Complete | [day5-product-decision.md](./artifacts/day5-product-decision.md) |
| 6 | Workflow And Tracking Design | Complete | [day6-workflow-tracking-design.md](./artifacts/day6-workflow-tracking-design.md) |
| 7 | Policy Implementation Batch | Complete | [day7-policy-implementation-batch.md](./artifacts/day7-policy-implementation-batch.md) |
| 8 | Freshness And Coverage Checks | Complete | [day8-freshness-and-coverage-checks.md](./artifacts/day8-freshness-and-coverage-checks.md) |
| 9 | Link And Routing Validation | Complete | [day9-link-and-routing-validation.md](./artifacts/day9-link-and-routing-validation.md) |
| 10 | User-Facing API Docs Update | Complete | [day10-user-facing-api-docs-update.md](./artifacts/day10-user-facing-api-docs-update.md) |
| 11 | Maintainer And Claim Boundary Docs | Complete | [day11-maintainer-and-claim-boundary-docs.md](./artifacts/day11-maintainer-and-claim-boundary-docs.md) |
| 12 | Integrated Validation | Complete | [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md) |
| 13 | Review Hardening | Complete | [day13-review-hardening.md](./artifacts/day13-review-hardening.md) |
| 14 | Closeout Package | Complete | [day14-closeout-review.md](./artifacts/day14-closeout-review.md) |

## Current Generated API Policy Inventory

| Surface | Current state | Sprint 204 relevance |
| --- | --- | --- |
| `docs/api_reference.md` | Source-controlled API reference entry point. Checked-in public headers under `include/` are the declaration source of truth. Generated HTML is local-only, ignored, and current only after `make api-docs-freshness`. | Primary user-facing routing surface for any policy decision. |
| README | Lists `make docs`, `make docs-check`, and `make api-docs-freshness`; states generated API HTML is not hosted documentation, retained CI artifact, source-controlled output, or release evidence. | Must be updated if publication semantics change or if local-only wording is strengthened. |
| INSTALL | Support/readiness matrix lists `Local generated API HTML` as `local-only` with no hosted publication or completeness beyond checked-in public headers selected by `Doxyfile`. | Public support-tier authority for install/package/API readiness claims. |
| `docs/maintainer_guide.md` | Maintainer interpretation says `docs/api/html/` is generated Doxygen output from `Doxyfile`, kept local-only and ignored; `make api-docs-freshness` is the validation command. | Main maintainer policy and claim-boundary surface. |
| `.gitignore` | Ignores `docs/api/`, including `docs/api/html/` and `docs/api/html/index.html`. | Controls committed-output and staging policy. |
| `Doxyfile` | `OUTPUT_DIRECTORY = docs/api`; input scope is checked-in public headers under `include/`. | Defines generated-output location and source set. |
| Makefile | Defines `docs`, `api-docs-coverage`, `api-docs-local-only`, `api-docs-routing`, `docs-check`, `api-docs-validate`, and `api-docs-freshness`. | Central validation and freshness command surface. |
| `scripts/check_api_docs_coverage.py` | Checks generated Doxygen page coverage for checked-in public headers. | Freshness/coverage enforcement candidate. |
| `scripts/check_api_docs_local_only.sh` | Enforces ignore rules, no tracked/staged/non-ignored generated files, Doxyfile local-only contract, product-status wording, and absence of workflow publication paths. | Current policy guard and likely Day 7/Day 8 owner. |
| `.github/workflows/*.yml` | Current workflows must not reference `docs/api/` while local-only status is active. | Any hosted, artifact, or Pages path would require deliberate workflow changes and new guards. |

## Current Public Header Source Set

The current checked-in public header source set has 18 headers:

- `include/sparse_analysis.h`
- `include/sparse_bidiag.h`
- `include/sparse_cholesky.h`
- `include/sparse_csr.h`
- `include/sparse_dense.h`
- `include/sparse_eigs.h`
- `include/sparse_ic.h`
- `include/sparse_ilu.h`
- `include/sparse_iterative.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu.h`
- `include/sparse_lu_csr.h`
- `include/sparse_matrix.h`
- `include/sparse_qr.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_types.h`
- `include/sparse_vector.h`

Generated install header behavior for `sparse_version.h` remains owned by
`VERSION`, `include/sparse_version.h.in`, and install/package validation, not
by the current Doxygen input set.

## Prior Evidence Summary

| Prior evidence | Relevant conclusion for Sprint 204 |
| --- | --- |
| Sprint 158 | Selected local-only generated API HTML, page coverage, no selected Doxygen warnings, and no committed/hosted generated HTML. |
| Sprint 179 | Reconfirmed and strengthened local-only generated API HTML; added local-only staging and workflow-publication guard behavior. |
| Sprint 186 | Kept `R186-HOSTED-API` open as a residual; generated API evidence remained local freshness and staging proof, not hosted or retained publication. |
| Epic 18 retrospective | Lists generated API publication policy as a future closure candidate while preserving local-only evidence and no hosted API publication claim. |

## Decision Log

| Date | Decision | Rationale |
| --- | --- | --- |
| Sprint 204 Day 1 | No publication decision made yet. | Day 1 is intake only. Hosted publication, retained artifacts, committed generated output, and stronger local-only policy remain open for Day 3-Day 5 evaluation. |
| Sprint 204 Day 4 | Acceptance gate established; no product option selected yet. | Day 4 defines the minimum evidence, validation, rollback, and claim-boundary requirements that Day 5 must apply before selecting hosted publication, retained artifacts, committed output, or stronger local-only policy. |
| Sprint 204 Day 5 | Selected stronger local-only generated API policy. | Day 2 proved the current local Doxygen/freshness path, Day 3 showed stronger local-only has the best maintenance/review fit, and Day 4 showed hosted, retained-artifact, and committed-output paths require new publication infrastructure not justified for this closure. |
| Sprint 204 Day 6 | Designed the stronger local-only implementation path without changing policy files. | Day 6 maps generated API input/output ownership, keeps `.gitignore`, `Doxyfile`, workflows, and Makefile behavior unchanged by default, and selects targeted local-only guard/diagnostic hardening for Day 7-Day 9. |
| Sprint 204 Day 7 | Implemented the stronger local-only guard batch. | The guard now rejects workflows that combine generated API output paths with artifact, Pages, or publication semantics; `api-docs-local-only` now runs a Python regression suite so this boundary is executable evidence rather than shell-only behavior. |
| Sprint 204 Day 8 | Strengthened generated API freshness and coverage checks. | `scripts/check_api_docs_coverage.py` now fails when generated reference or source pages are older than their checked-in public headers, and `api-docs-coverage` now runs fixture regressions for missing and stale generated pages. |
| Sprint 204 Day 9 | Added local-only API routing validation. | `api-docs-routing` now validates required API reference entry-point links, local-only generated API wording, missing route targets, and absence of generated/hosted API publication links. |
| Sprint 204 Day 10 | Updated user-facing generated API wording. | README, INSTALL, and `docs/api_reference.md` now describe the selected local-only policy using the same freshness, coverage, staging, routing, and non-publication vocabulary enforced by the Day 7-Day 9 guards. |
| Sprint 204 Day 11 | Aligned maintainer guidance and claim-boundary guards. | `docs/maintainer_guide.md` now names `api-docs-routing`, workflow publication rejection, stale/partial generated-output interpretation, and unsupported publication/ABI/package/platform/performance/state-of-the-art boundaries; the routing regression suite now fails if those maintainer markers are removed. |
| Sprint 204 Day 12 | Ran integrated generated API validation. | `make docs-check`, `make api-docs-freshness`, all focused routing/local-only/coverage regressions, Python compile checks, and whitespace checks passed; no `.c` or `.h` files changed, so the full C gate was not required. |
| Sprint 204 Day 13 | Added Makefile routing-wiring hardening. | Review audit found `api-docs-routing` was validated directly but the routing guard did not prove `api-docs-validate` still depended on it; Day 13 made that wiring guard-backed and regression-covered. |
| Sprint 204 Day 14 | Closed Sprint 204 with stronger local-only generated API policy. | The sprint completed the product decision, guard implementation, freshness/coverage/routing checks, public and maintainer docs, claim-boundary guard coverage, and final validation evidence; hosted publication, retained artifacts, committed generated HTML, package/ABI/platform/performance/state-of-the-art claims remain explicit residual non-claims. |

## Initial Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Generated HTML gets staged or committed accidentally. | Review noise, stale docs, and source-of-truth confusion. | Preserve or strengthen `scripts/check_api_docs_local_only.sh`; verify git status and ignore behavior after docs runs. |
| Hosted or artifact publication is added without freshness/link checks. | Users may rely on stale generated docs. | Require Day 4 acceptance gate before workflow or publication changes. |
| Public docs imply ABI/package completeness from generated API output. | Overstates supported surface. | Keep claim-boundary wording and guards aligned across README, INSTALL, API reference, and maintainer guide. |
| Doxygen input set drifts from checked-in public headers. | Missing or unexpected generated pages. | Reuse and, if needed, strengthen `scripts/check_api_docs_coverage.py` and Doxyfile checks. |
| Workflow publication scans remain string-based. | Guard could miss a new publication path. | Day 6-Day 8 should decide whether structured workflow validation is required for the selected policy. |
| Hosted Pages or artifact retention creates a support expectation. | Increases maintenance burden and stale-output risk. | Document retention, branch, URL, and freshness semantics before implementation. |

## Validation Matrix

| Command | Current owner | When required | Status |
| --- | --- | --- | --- |
| `make docs-check` | Doxygen generation and public-header page coverage | Day 2 baseline; after docs/Doxygen changes | Passed on Day 2, Day 12, and Day 14; coverage owner strengthened on Day 8 |
| `make api-docs-freshness` | Doxygen generation, coverage, local-only staging guard, and API routing guard | Day 2 baseline; after policy/docs/guard changes | Passed on Day 2, Day 7, Day 8, Day 9, Day 10, Day 11, Day 12, Day 13, and Day 14 |
| `bash scripts/check_api_docs_local_only.sh` | Local-only generated-output guard | After local-only guard or wording changes | Passed directly on Day 7 and Day 12 |
| `python3 tests/test_api_docs_local_only_guard.py` | Regression coverage for local-only generated API guard failures | After local-only guard logic changes | Passed on Day 7 and Day 12 |
| `python3 scripts/check_api_docs_coverage.py` | Generated page coverage and stale-page freshness script | After coverage logic changes | Strengthened on Day 8; covered through `make api-docs-freshness` |
| `python3 tests/test_api_docs_coverage.py` | Regression coverage for missing and stale generated API pages | After coverage logic changes | Passed on Day 8, Day 12, and Day 14 |
| `python3 -m py_compile scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Python syntax/import sanity check | After Python script/test edits | Passed on Day 8, Day 12, and Day 14 |
| Workflow publication/path guard | `scripts/check_api_docs_local_only.sh` and `tests/test_api_docs_local_only_guard.py` | If hosted/artifact/Pages or local-only workflow policy changes | Day 7 guard and regression passed; publication remains absent |
| `python3 scripts/check_api_docs_routing.py` | Local-only API routing and generated API claim-boundary guard | After generated API routing/docs changes | Added on Day 9; Makefile routing-wiring checks added on Day 13; passed on Day 14 |
| `python3 tests/test_api_docs_routing.py` | Regression coverage for API routing and maintainer claim-boundary guard | After generated API routing guard changes | Added on Day 9; maintainer marker regressions passed on Day 11-Day 12; Makefile dependency/target regressions added on Day 13; passed on Day 14 |
| Markdown/link validation | `scripts/check_api_docs_routing.py` for the selected API routing surface | Required if the selected policy adds hosted, retained, committed, or new local routing links | Day 9 local-only routing guard passed |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py` | Python syntax/import sanity check | After Python routing script/test edits | Passed on Day 9, Day 10, Day 11, Day 12, Day 13, and Day 14 |
| `git diff --check` | Whitespace sanity check | After script/test/docs edits | Passed on Day 7, Day 8, Day 9, Day 10, Day 11, Day 12, Day 13, and Day 14 |
| `make format && make lint && make test` | Full C quality gate | Required if `.c` or `.h` files change | Not required through Day 14; no `.c` or `.h` files changed |

## Open Questions

No open Sprint 204 questions remain at closeout. Sprint 205 should consume the
remaining generated API publication residual only if it deliberately reopens
hosted publication, retained generated artifacts, or committed generated HTML.

## Explicit Non-Goals

Sprint 204 does not claim or implement:

- broad API completeness beyond checked-in public headers selected by
  `Doxyfile`;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- Homebrew/core, bottles, Linuxbrew, public tap, vcpkg, Conan, pkgsrc, or
  system package support;
- broad platform parity;
- Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable performance or benchmark publication;
- solver behavior changes;
- release evidence from generated API HTML;
- hosted generated API publication unless Day 5 explicitly selects it and
  Days 6-12 implement matching guards;
- committed generated HTML unless Day 5 explicitly selects it and Days 6-12
  implement matching freshness and staging rules.

## Day 1 Notes

Day 1 completed intake and scaffolding only. No generated output, workflow,
guard, public docs, or source code behavior changed beyond Sprint 204 planning
artifacts.

## Day 2 Notes

Day 2 reproduced the current generated API baseline without changing policy.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `doxygen --version` | Passed | Reported `1.16.1`. |
| `git check-ignore -v docs/api docs/api/html docs/api/html/index.html` | Passed | All three paths match `.gitignore:44:docs/api/`. |
| `make docs-check` | Passed | Generated local Doxygen HTML; coverage reported 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` as a separate installed-header policy row. |
| `make api-docs-freshness` | Passed | Re-ran Doxygen and coverage, then passed local-only ignore, Doxyfile contract, no tracked/staged/non-ignored generated files, wording, and workflow non-publication checks. |
| `git status --short --ignored docs/api docs/planning/EPIC_18/SPRINT_204` | Passed | Reported `!! docs/api/` and untracked Sprint 204 planning files only. |

Generated-output inventory after validation:

| Output | Day 2 observation |
| --- | --- |
| `docs/api/html/` files at max depth 1 | 156 files |
| `docs/api/html/*.html` files at max depth 1 | 88 HTML files |
| Tracked `docs/api` files | None |
| Staged `docs/api` files | None |
| Visible non-ignored untracked `docs/api` files | None |

No source, public header, workflow, Makefile, Doxyfile, user-facing docs, or
generated-output policy files changed on Day 2.

## Day 3 Notes

Day 3 compared four generated API policy options without selecting one:

| Option | Day 3 fit | Main reason |
| --- | --- | --- |
| Hosted generated API HTML | Viable but highest infrastructure burden | Best user discoverability, but requires deployment ownership, freshness metadata, link checks, workflow guards, and explicit support-tier wording. |
| Retained CI artifact | Viable as reviewer/maintainer evidence | Avoids permanent generated diffs, but still needs artifact naming, retention, metadata, and workflow guard coverage. |
| Committed generated output | Technically possible but weakest fit | Conflicts with the current ignored-output model and would add large generated diffs and stale-output risk. |
| Stronger local-only policy | Strongest current fit | Matches Day 2 passing evidence, preserves reviewability, and can close ambiguity with guard/docs hardening. |

Day 3 does not decide item 204.1. It supplies the option inventory for the
Day 4 acceptance gate and Day 5 product decision.

No source, public header, workflow, Makefile, Doxyfile, user-facing docs, or
generated-output policy files changed on Day 3.

## Day 4 Notes

Day 4 converted the Day 3 option inventory into an acceptance gate for the
Day 5 product decision. The gate requires:

- every selected policy to preserve source-of-truth headers and API reference
  semantics;
- hosted publication to include deployment ownership, metadata, freshness,
  link validation, workflow guards, and rollback rules;
- retained artifacts to include artifact naming, retention, metadata,
  freshness, and workflow guard coverage;
- committed generated output to include drift detection, review discipline,
  changed ignore/staging semantics, and generated-output cleanup rules;
- stronger local-only policy to preserve ignored output, no workflow
  publication, freshness/staging checks, and explicit residual wording;
- all paths to retain ABI, package-manager, broad platform, release,
  broad-completeness, and state-of-the-art non-claims.

Day 4 does not decide item 204.1. It supplies the pass/fail framework for the
Day 5 decision.

No source, public header, workflow, Makefile, Doxyfile, user-facing docs, or
generated-output policy files changed on Day 4.

## Day 5 Notes

Day 5 selected **stronger local-only generated API policy** for Sprint 204.

Selected path:

- keep checked-in public headers under `include/` and `docs/api_reference.md`
  as the source-controlled API reference path;
- keep `docs/api/` and `docs/api/html/` ignored generated output;
- keep `make api-docs-freshness` as the supported local Doxygen freshness and
  staging command;
- strengthen local-only guard, routing, and claim-boundary coverage where Day
  6-Day 11 finds concrete gaps.

Rejected paths:

| Path | Day 5 disposition |
| --- | --- |
| Hosted generated API HTML | Deferred. It has high user value, but requires deployment ownership, freshness metadata, link checks, workflow guards, access/retention semantics, and rollback ownership. |
| Retained CI artifact | Deferred. It is useful reviewer evidence, but still requires artifact naming, retention, metadata, upload-scope guard coverage, and public wording. |
| Committed generated output | Rejected for this sprint. It conflicts with the current ignored-output model and would create durable generated-diff review and stale-output burden. |

Allowed implementation surfaces for the selected path:

- `scripts/check_api_docs_local_only.sh`
- `scripts/check_api_docs_coverage.py` only if coverage diagnostics need
  local-only clarification
- Makefile docs/API targets only if guard composition needs clarification
- docs tests or focused guard tests if added for local-only policy
- README, INSTALL, `docs/api_reference.md`, and `docs/maintainer_guide.md`
- Sprint 204 planning artifacts

Explicitly disallowed for this sprint unless the product decision is reopened:

- workflow publication, upload, deploy, or Pages changes for generated API
  HTML;
- committing files under `docs/api/`;
- broadening the Doxygen input set beyond checked-in public headers under
  `include/`;
- `.c` or public `.h` behavior changes unrelated to generated API policy.

No source, public header, workflow, Makefile, Doxyfile, user-facing docs, or
generated-output policy files changed on Day 5.

## Day 6 Notes

Day 6 designed the selected stronger local-only implementation path.

Design decisions:

- keep `.gitignore` unchanged and preserve `docs/api/` as ignored output;
- keep `Doxyfile` unchanged and preserve checked-in public headers under
  `include/` as the generated API input set;
- keep workflows free of generated API publication, upload, deploy, Pages, and
  `docs/api/` references;
- keep Makefile target names unchanged unless later implementation needs a
  clearer aggregate guard;
- focus Day 7 on hardening `scripts/check_api_docs_local_only.sh` diagnostics
  and workflow-publication rejection rather than adding publication support;
- reserve `scripts/check_api_docs_coverage.py` changes for Day 8 only if a
  concrete coverage diagnostic gap is found.

Path ownership:

| Path | Owner under selected policy |
| --- | --- |
| `include/*.h` | Source-controlled API declaration and Doxygen input truth. |
| `docs/api_reference.md` | Source-controlled API reference entry point. |
| `docs/api/` | Ignored local generated output root. |
| `docs/api/html/` | Ignored local generated Doxygen HTML view. |
| `.github/workflows/*.yml` | Must not publish, upload, deploy, or reference generated API output while local-only policy is selected. |

Day 6 does not implement item 204.2 yet; it narrows the Day 7 implementation
target and validation plan.

No source, public header, workflow, Makefile, Doxyfile, user-facing docs, or
generated-output policy files changed on Day 6.

## Day 7 Notes

Day 7 implemented the selected stronger local-only policy guard batch.

Implementation changes:

- `scripts/check_api_docs_local_only.sh` now scans top-level workflow YAML
  files deterministically and rejects any workflow that combines generated API
  output paths with artifact upload, Pages, `gh-pages`, or publication
  semantics.
- `tests/test_api_docs_local_only_guard.py` adds executable regression
  coverage for the current tree, a minimal passing fixture, a raw generated API
  workflow path, a generated API publication/artifact path, and missing
  local-only wording.
- `Makefile` wires the Python regression suite into `api-docs-local-only`, so
  `make api-docs-freshness` now exercises the shell guard and its mutation
  tests.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `bash -n scripts/check_api_docs_local_only.sh` | Passed | Shell syntax validated before execution. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Verified ignore rules, Doxyfile contract, untracked/staged absence, local-only wording, and no workflow generated API publication semantics. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Regression suite covered passing and failing local-only guard fixtures. |
| `make api-docs-freshness` | Passed | Regenerated local Doxygen HTML, confirmed 18 public headers/pages/source pages, and ran the strengthened local-only guard path. |
| `git diff --check` | Passed | No whitespace errors in the Day 7 diff. |

Generated output remains local and ignored:

| Path | Day 7 state |
| --- | --- |
| `docs/api/` | Ignored generated output (`!! docs/api/`). |
| `.github/workflows/*.yml` | No generated API output path, artifact upload, Pages, or publication semantics were added. |
| `.gitignore` | Unchanged; `docs/api/` remains ignored. |
| `Doxyfile` | Unchanged; Doxygen input and output policy remain local-only. |

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 7.

## Day 8 Notes

Day 8 strengthened generated API freshness and generated-page coverage for the
selected stronger local-only policy.

Implementation changes:

- `scripts/check_api_docs_coverage.py` now treats a generated reference or
  source page as stale if it is older than its checked-in public header and
  reports the specific header/page pair with a `rerun make docs-check`
  diagnostic.
- `tests/test_api_docs_coverage.py` adds fixture coverage for complete
  checked-in public-header pages, missing HTML directory, missing index,
  missing reference page, missing source page, stale reference page, and stale
  source page.
- `Makefile` wires the new coverage regression suite into
  `api-docs-coverage`, so `make docs-check` and `make api-docs-freshness`
  exercise the regression owner after local Doxygen generation.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_api_docs_coverage.py` | Passed | Covered fixture diagnostics for missing and stale generated pages. |
| `python3 -m py_compile scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Python syntax/import sanity check for modified docs tooling. |
| `make api-docs-freshness` | Passed | Regenerated local Doxygen HTML, confirmed 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and passed local-only guard regression coverage. |
| `git diff --check` | Passed | No whitespace errors in the Day 8 diff. |

Coverage-policy boundary:

| Boundary | Day 8 disposition |
| --- | --- |
| Checked-in public headers under `include/` | Still the only generated API coverage input set. |
| `include/sparse_version.h.in` and generated installed header behavior | Still excluded from expected Doxygen page coverage and owned by install/version policy. |
| `docs/api/html/` | Still ignored local generated output; freshness is local evidence only. |
| Hosted/API publication | Still absent; no workflow, Pages, artifact, or committed generated HTML path was added. |

Day 8 completes the freshness and generated-page coverage half of item 204.3.
The link/routing half remains planned for Day 9.

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 8.

## Day 9 Notes

Day 9 completed local-only API link and routing validation for item 204.3.

Implementation changes:

- `scripts/check_api_docs_routing.py` validates the selected API routing
  surface across README, INSTALL, `docs/api_reference.md`, and
  `docs/maintainer_guide.md`.
- The routing guard requires the source-controlled API entry point,
  public-header route, Doxyfile route, support/readiness matrix route, and
  local workflow-guide links that are part of the selected API reference path.
- The guard rejects Markdown links to generated API HTML, `docs/api/`, hosted
  API/Doxygen/Pages-style publication targets, and missing required API route
  targets.
- `tests/test_api_docs_routing.py` adds current-tree and fixture regressions
  for missing API-reference routes, missing route targets, generated HTML
  publication links, hosted API publication links, and missing local-only
  routing text.
- `Makefile` adds `api-docs-routing` and wires it into `api-docs-validate`, so
  `make api-docs-freshness` now runs generation, coverage, local-only staging,
  and routing validation.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, found no generated API publication links, and confirmed `docs/api_reference.md` as the source-controlled API entry point. |
| `python3 tests/test_api_docs_routing.py` | Passed | Covered passing fixtures and failing route/publication/text mutations. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py` | Passed | Python syntax/import sanity check for new routing tooling. |
| `make api-docs-freshness` | Passed | Regenerated local Doxygen HTML, ran coverage/local-only checks, and passed the new `api-docs-routing` target. |
| `git diff --check` | Passed | No whitespace errors in the Day 9 diff. |

Routing inventory:

| Route | Day 9 disposition |
| --- | --- |
| README API entry point | Guard requires `docs/api_reference.md`, `include/`, and `INSTALL.md#support-readiness-matrix` links. |
| `docs/api_reference.md` source-of-truth route | Guard requires links to `../include/`, `../Doxyfile`, and `../INSTALL.md#support-readiness-matrix`. |
| `docs/api_reference.md` workflow routes | Guard requires local links to tutorial, cookbook, solver-selection, and maintainer-guide docs. |
| INSTALL support/readiness row | Guard requires local-only generated API HTML wording and no hosted/completeness overclaim. |
| Maintainer generated API section | Guard requires source-controlled entry-point, generated-output, and local-only interpretation wording. |

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 9. Generated `docs/api/`,
`scripts/__pycache__/`, and `tests/__pycache__/` remain ignored local output.

## Day 10 Notes

Day 10 aligned user-facing generated API docs with the selected stronger
local-only policy and the Day 7-Day 9 guard stack.

Documentation changes:

- README now describes `make api-docs-freshness` as the selected local Doxygen
  freshness command with local-only staging and routing guards.
- README now states that the freshness command checks generated page coverage,
  local-only staging, and API routing so user docs keep pointing at the
  source-controlled API entry point rather than unavailable generated or hosted
  publication.
- INSTALL now names the coverage/local-only/routing scripts as the proof
  surface for the local generated API HTML support row and explicitly excludes
  hosted API publication, retained generated-doc artifacts, committed generated
  HTML, and completeness beyond checked-in public headers selected by
  `Doxyfile`.
- `docs/api_reference.md` now describes generated page coverage/freshness,
  local-only staging enforcement, API routing validation, and the routing guard
  rejection of generated HTML or hosted API publication links.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Confirmed four routing documents, no generated API publication links, and `docs/api_reference.md` as source-controlled API entry point. |
| `python3 tests/test_api_docs_routing.py` | Passed | Routing regression suite still passed after the wording update. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Local-only wording, Doxyfile, workflow, ignore, and tracking checks still pass. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Python syntax/import sanity check for current docs tooling. |
| `make api-docs-freshness` | Passed | Regenerated local Doxygen HTML and passed coverage, local-only, and routing guards. |
| `git diff --check` | Passed | No whitespace errors in the Day 10 diff. |

User-facing claim boundary:

| Boundary | Day 10 wording |
| --- | --- |
| Source-controlled API entry point | `docs/api_reference.md` plus checked-in public headers under `include/`. |
| Generated HTML status | Local-only convenience view under ignored `docs/api/html/`. |
| Freshness command | `make api-docs-freshness` covers generation, page coverage/freshness, local-only staging, and routing validation. |
| Unsupported publication paths | No hosted API docs, retained generated-doc artifact, committed generated HTML, or release evidence. |
| Unsupported support claims | No broad API completeness beyond checked-in public headers selected by `Doxyfile`; no ABI, package-manager, broad platform, performance, or state-of-the-art claim. |

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 10. Generated `docs/api/`,
`scripts/__pycache__/`, and `tests/__pycache__/` remain ignored local output.

## Day 11 Notes

Day 11 aligned maintainer guidance and guard-backed claim boundaries with the
selected stronger local-only generated API policy.

Maintainer updates:

- `docs/maintainer_guide.md` now describes `make api-docs-freshness` as
  Doxygen generation, page coverage/freshness, local-only generated-output
  validation, and API routing validation.
- The maintainer guide now names `api-docs-routing` as the guard that keeps
  user-facing docs routed to `docs/api_reference.md`, checked-in public
  headers, `Doxyfile`, workflow guides, and INSTALL support/readiness status
  rather than generated HTML or hosted API publication links.
- The guide now explicitly rejects workflow artifact upload, Pages deployment,
  hosted API URLs, committed `docs/api/` content, generated HTML release
  evidence, package-manager evidence, ABI evidence, broad platform evidence,
  portable performance evidence, and state-of-the-art evidence unless the
  product decision is reopened.
- The guide now states that removing `api-docs-routing` from
  `make api-docs-freshness` must be replaced by an equivalent
  claim-boundary guard.

Guard updates:

- `scripts/check_api_docs_routing.py` now requires the maintainer-guide
  markers for `api-docs-freshness`, `api-docs-routing`, retained generated-doc
  artifact non-claims, and routing-guard preservation.
- `tests/test_api_docs_routing.py` now includes a maintainer claim-boundary
  mutation that fails if the routing-guard preservation marker is removed.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Confirmed four routing documents, no generated API publication links, and maintainer claim-boundary markers. |
| `python3 tests/test_api_docs_routing.py` | Passed | Regression suite covered the new maintainer claim-boundary marker. |
| `bash scripts/check_api_docs_local_only.sh` | Passed | Local-only wording, Doxyfile, workflow, ignore, and tracking checks still pass. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py scripts/check_api_docs_coverage.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py` | Passed | Python syntax/import sanity check for current docs tooling. |
| `make api-docs-freshness` | Passed | Regenerated local Doxygen HTML and passed coverage, local-only, and routing guards. |
| `git diff --check` | Passed | No whitespace errors in the Day 11 diff. |

Claim-boundary inventory:

| Boundary | Guarded or documented Day 11 state |
| --- | --- |
| Hosted API publication | Rejected by maintainer guidance, routing guard, and local-only workflow publication checks. |
| Retained generated-doc artifact | Explicit maintainer and INSTALL non-claim; guarded by routing markers and workflow publication semantics check. |
| Committed generated HTML | Rejected by `.gitignore`, local-only guard, and maintainer guidance. |
| Release evidence | Explicit maintainer and API-reference non-claim for generated HTML. |
| API completeness | Limited to checked-in public headers selected by `Doxyfile`; no broad completeness claim. |
| ABI/shared-library support | Explicit maintainer non-claim for generated API evidence. |
| Package-manager support | Explicit maintainer and INSTALL non-claim for generated API evidence. |
| Broad platform parity | Explicit maintainer non-claim for generated API evidence. |
| Performance or state-of-the-art evidence | Explicit maintainer non-claim for generated API evidence. |

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 11. Generated `docs/api/`,
`scripts/__pycache__/`, and `tests/__pycache__/` remain ignored local output.

## Day 12 Notes

Day 12 ran the integrated generated API validation set for the selected
stronger local-only policy.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `make docs-check` | Passed | Doxygen generated local HTML; coverage reported 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` as a separate installed-header policy row. |
| `make api-docs-freshness` | Passed | Ran Doxygen generation, coverage/freshness, local-only staging guard, local-only guard regressions, routing guard, and routing regressions. |
| `python3 tests/test_api_docs_coverage.py` | Passed | Missing and stale generated-page fixture regressions passed. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Local-only staging/workflow-publication fixture regressions passed. |
| `python3 tests/test_api_docs_routing.py` | Passed | API route, generated/hosted publication-link, and maintainer marker regressions passed. |
| `bash -n scripts/check_api_docs_local_only.sh && bash scripts/check_api_docs_local_only.sh` | Passed | Shell syntax and direct local-only guard checks passed. |
| `python3 scripts/check_api_docs_routing.py` | Passed | Direct routing guard checked four routing documents and confirmed generated API publication links are absent. |
| `python3 -m py_compile scripts/check_api_docs_coverage.py scripts/check_api_docs_routing.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py tests/test_api_docs_routing.py` | Passed | Python docs tooling compiled. |
| `git diff --check` | Passed | No whitespace errors. |
| `git diff --name-only -- '*.c' '*.h' && git ls-files --others --exclude-standard -- '*.c' '*.h'` | Passed | No changed or untracked `.c`/`.h` files were reported. |

Residual ledger:

| Residual | Day 12 status |
| --- | --- |
| Hosted generated API HTML | Still intentionally absent; not selected by Sprint 204. |
| Retained generated-doc artifact | Still intentionally absent; not selected by Sprint 204. |
| Committed generated HTML | Still intentionally absent; `docs/api/` remains ignored. |
| Full C quality gate | Not required because no `.c` or `.h` files changed. |
| Generated runtime output | `docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/` remain ignored local output. |

No blockers were found on Day 12.

No `.c`, `.h`, public API, Doxyfile, workflow, `.gitignore`, or generated API
HTML files were changed or committed by Day 12.

## Day 13 Notes

Day 13 audited the selected local-only generated API policy for review
hardening gaps, unrelated surface expansion, generated-output accidents, and
claim-boundary drift.

Reviewed surfaces:

- `README.md`
- `INSTALL.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `scripts/check_api_docs_coverage.py`
- `scripts/check_api_docs_local_only.sh`
- `scripts/check_api_docs_routing.py`
- `tests/test_api_docs_coverage.py`
- `tests/test_api_docs_local_only_guard.py`
- `tests/test_api_docs_routing.py`
- Sprint 204 planning artifacts and working notes

Hardening result:

| Finding | Resolution |
| --- | --- |
| `api-docs-routing` validated the local-only routing policy directly, but the routing guard did not prove that `api-docs-validate` still depended on the routing target. | `scripts/check_api_docs_routing.py` now validates Makefile routing target wiring, and `tests/test_api_docs_routing.py` adds regressions for both a missing `api-docs-routing` target and a missing `api-docs-validate` dependency. |

Day 13 review did not find unrelated production edits, public API edits,
workflow publication paths, committed generated API HTML, Doxyfile scope
changes, `.gitignore` changes, or `.c`/`.h` changes.

Validation:

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, confirmed generated API publication links are absent, and confirmed Makefile `api-docs-routing` wiring is present. |
| `python3 tests/test_api_docs_routing.py` | Passed | Existing route/publication/claim-boundary regressions passed; new missing Makefile target/dependency regressions passed. |
| `python3 -m py_compile scripts/check_api_docs_routing.py tests/test_api_docs_routing.py` | Passed | Python routing script and regression suite compiled. |
| `make api-docs-freshness` | Passed | Doxygen generation, coverage/freshness, local-only staging guard, local-only guard regressions, routing guard, and routing regressions all passed. |
| `git diff --check` | Passed | No whitespace errors. |
| `git diff --name-only -- '*.c' '*.h' && git ls-files --others --exclude-standard -- '*.c' '*.h'` | Passed | No changed or untracked `.c`/`.h` files were reported. |

Residual ledger:

| Residual | Day 13 status |
| --- | --- |
| Hosted generated API HTML | Still intentionally absent; not selected by Sprint 204. |
| Retained generated-doc artifact | Still intentionally absent; not selected by Sprint 204. |
| Committed generated HTML | Still intentionally absent; `docs/api/` remains ignored local output. |
| Makefile routing integration drift | Closed by Day 13 routing guard and regression coverage. |
| Full C quality gate | Not required because no `.c` or `.h` files changed. |
| Closeout package | Remains for Day 14. |

Generated `docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/`
remain ignored local output after Day 13 validation.

## Day 14 Notes

Day 14 packaged Sprint 204 for retrospective creation and PR review.

Final item status:

| Item | Closeout status | Evidence |
| --- | --- | --- |
| 204.1 Product Decision | Complete | Stronger local-only generated API policy selected on Day 5 after Day 3 option scoring and Day 4 acceptance-gate review. |
| 204.2 Publication Or Guard Implementation | Complete | Day 7 strengthened local-only staging/workflow-publication guards and integrated guard regressions. |
| 204.3 Freshness And Link Checks | Complete | Day 8 added generated-page stale checks; Day 9 added source-controlled API routing validation; Day 13 hardened Makefile routing wiring. |
| 204.4 API Routing Docs | Complete | Day 10 aligned README, INSTALL, and `docs/api_reference.md` with local-only generated API semantics. |
| 204.5 Claim Boundary Guard | Complete | Day 11 aligned maintainer guidance and regression-covered unsupported publication, package, ABI, platform, performance, and state-of-the-art boundaries. |
| 204.6 Validation | Complete | Day 12 integrated validation, Day 13 hardening validation, and Day 14 closeout validation passed without `.c` or `.h` changes. |

Final selected policy:

- Generated Doxygen HTML remains local-only output under `docs/api/html/`.
- `docs/api_reference.md`, checked-in public headers under `include/`,
  `Doxyfile`, README, INSTALL, and maintainer guidance are the
  source-controlled API documentation route.
- `make api-docs-freshness` is the aggregate freshness path for generated page
  coverage, stale generated pages, local-only staging, workflow
  non-publication, API routing, and routing guard regressions.
- Generated API HTML is not hosted documentation, a retained CI artifact,
  source-controlled output, release evidence, ABI evidence, package-manager
  evidence, broad platform evidence, performance evidence, or
  state-of-the-art evidence.

Final validation:

| Command | Result | Notes |
| --- | --- | --- |
| `make docs-check` | Passed | Generated local Doxygen HTML and confirmed 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| `make api-docs-freshness` | Passed | Ran Doxygen generation, coverage/freshness, local-only staging guard, local-only guard regressions, routing guard, and routing regressions. |
| `python3 tests/test_api_docs_coverage.py` | Passed | Missing and stale generated-page fixture regressions passed. |
| `python3 tests/test_api_docs_local_only_guard.py` | Passed | Local-only staging and workflow-publication fixture regressions passed. |
| `python3 tests/test_api_docs_routing.py` | Passed | Routing, publication-link, maintainer marker, and Makefile wiring regressions passed. |
| `python3 scripts/check_api_docs_routing.py` | Passed | Checked four routing documents, Makefile routing wiring, generated API publication-link absence, and the source-controlled API entry point. |
| `python3 -m py_compile scripts/check_api_docs_coverage.py scripts/check_api_docs_routing.py tests/test_api_docs_coverage.py tests/test_api_docs_local_only_guard.py tests/test_api_docs_routing.py` | Passed | Python docs tooling compiled. |
| `git diff --check` | Passed | No whitespace errors. |
| `git diff --name-only -- '*.c' '*.h' && git ls-files --others --exclude-standard -- '*.c' '*.h'` | Passed | No changed or untracked `.c`/`.h` files were reported. |

Changed source-controlled surfaces:

- `README.md`
- `INSTALL.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `scripts/check_api_docs_coverage.py`
- `scripts/check_api_docs_local_only.sh`
- `scripts/check_api_docs_routing.py`
- `tests/test_api_docs_coverage.py`
- `tests/test_api_docs_local_only_guard.py`
- `tests/test_api_docs_routing.py`
- `docs/planning/EPIC_18/SPRINT_204/PLAN.md`
- `docs/planning/EPIC_18/SPRINT_204/WORKING_NOTES.md`
- `docs/planning/EPIC_18/SPRINT_204/RETROSPECTIVE.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day1-generated-api-intake.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day2-current-doxygen-baseline.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day3-publication-option-inventory.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day4-decision-criteria-acceptance-gate.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day5-product-decision.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day6-workflow-tracking-design.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day7-policy-implementation-batch.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day8-freshness-and-coverage-checks.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day9-link-and-routing-validation.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day10-user-facing-api-docs-update.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day11-maintainer-and-claim-boundary-docs.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day12-integrated-validation.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day13-review-hardening.md`
- `docs/planning/EPIC_18/SPRINT_204/artifacts/day14-closeout-review.md`

Retrospective inputs:

| Category | Closeout note |
| --- | --- |
| Completed work | Stronger local-only generated API policy, stale generated-page checks, source-controlled API routing guard, local-only workflow-publication guard, public docs alignment, maintainer claim-boundary alignment, Makefile routing-wiring hardening, and final validation. |
| Deferred work | Hosted generated API publication, retained generated-doc artifacts, committed generated HTML, package-manager evidence, ABI/shared-library evidence, broad platform evidence, performance evidence, release evidence, and state-of-the-art evidence. |
| Lesson | The selected local-only path is reviewable only when documentation, generated-output ignore behavior, workflow non-publication checks, and aggregate Makefile wiring are validated together. |
| Risk | Future hosted publication work must not reuse local-only generated HTML freshness as hosted, retained-artifact, release, ABI, package, or broad platform evidence without new workflow and link validation. |
| Handoff | Sprint 205 can rely on `make api-docs-freshness` as the Sprint 204 aggregate API-docs guard; any reopened publication path should start by replacing or extending that guard. |

Generated `docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/`
remain ignored local output at Sprint 204 closeout.
