# Epic 18 Retrospective

**Epic:** 18 - Selected Productization, Reliability & Evidence Promotion
**Sprints:** 197-206
**Status:** Complete through Sprint 206 Day 14

## Epic Objective

Epic 18 started from the Epic 17 closeout and targeted a small number of
productization, platform freshness, reliability, reviewability, benchmark,
comparison, generated API, support-matrix, and final validation gaps. The plan
intentionally preferred complete closure of selected gaps over partial progress
across every remaining state-of-the-art shortcoming.

The Epic 18 closeout also preserves a historical numbering caveat: Sprint 197
contains requested final-validation evidence for scope that the Epic project
plan later identifies as Sprint 206. Sprint 206 is now the explicit current
closeout branch and records the final reconciliation, claim recalibration,
validation, retrospective, residual queue, consistency hardening, and final
closeout path.

## Sprint Outcomes

| Sprint | Current outcome |
| --- | --- |
| 197 | Historical final-validation evidence with numbering caveat. The branch contains a requested final-validation plan, working notes, evidence ledger, claim audits, no-promotion records, project-plan interim status snapshot, validation logs, retrospective draft, residual queue, and final closeout review, but it is no longer the current explicit Sprint 206 closeout path. |
| 198 | Closed with developer-mode local Homebrew static source proof. Root MIT metadata, formula metadata, archive/checksum proof, temporary local tap render, source install, installed static surface validation, downstream `brew test`, uninstall, cleanup, and proof exit `0` are recorded; Homebrew/core readiness, bottles, Linuxbrew, public tap maintenance, and broad package-manager distribution remain unclaimed. |
| 199 | Closed with Windows Cholesky promotion re-deferred. Hosted Windows evidence for `cholesky-spd-tridiag-5` was reviewed and retained as guarded workflow evidence; selected Windows freshness promotion remains re-deferred until selected metadata, generated support tier, generated non-claim wording, and final claim contract are promoted together. |
| 200 | Closed with selected symbolic LU allocation-failure proof. `sparse_symbolic_lu()` was selected as the additional allocation-failure owner; invariants, harness reachability, failed-allocation, cleanup, retry, focused gate, registration guard, claim-documentation, integrated-validation, review-hardening, and closeout artifacts are present. Broader allocation-failure and state-of-the-art reliability claims remain unearned. |
| 201 | Closed for selected SVD helper review-surface reduction. The selected `tests/test_svd.c` rank, pseudoinverse, and dense low-rank cluster is helper-owned by `tests/test_svd_selected_helpers.h`; focused SVD guard coverage and validation evidence are recorded, while broader review-surface cleanup remains unclaimed. |
| 202 | Closed with macOS hosted selected benchmark freshness evidence. The selected macOS hosted performance-freshness lane for `SRT-BENCH-REFACTOR-CSC-NOS4` is recorded with PR run, job, artifact, digest, manifest, workflow, and validator evidence; portable performance, timing thresholds, broad benchmark publication, and state-of-the-art performance remain unclaimed. |
| 203 | Closed with Windows QR incompatible promotion re-deferred. Local `qr-incompatible-ls` generator and selected freshness proof passed, Windows-style artifact path and diagnostic guards were added, and workflow/selected manifest promotion remained intentionally absent because hosted Windows/MSVC proof and hosted artifact inspection were not available. |
| 204 | Closed with stronger local-only generated API policy. Generated Doxygen HTML remains local-only ignored output under `docs/api/html/`; `make api-docs-freshness` covers generated-page freshness, local-only staging, workflow non-publication, API routing, and Makefile routing wiring; hosted generated API publication, retained generated-doc artifacts, committed generated HTML, package-manager evidence, ABI/shared-library evidence, broad platform evidence, performance evidence, release evidence, and state-of-the-art evidence remain unclaimed. |
| 205 | Closed with support matrix and adoption quick-reference consolidation. Public support truth remains `INSTALL.md#support-readiness-matrix`; the compact problem-shape quick reference routes users to existing solver/workflow owners, diagnostics wording was normalized across selected docs, and focused claim guards were aligned with simplified wording while package-manager, Windows, ABI/shared-library, hosted generated API, portable performance, release, and state-of-the-art claims remain unclaimed. |
| 206 | Closed on the explicit closeout branch. Evidence reconciliation, public and maintainer claim recalibration, project-plan status, focused validation, broad documentation/API validation, this retrospective, residual queue refresh, consistency hardening, and final closeout review are recorded. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence reconciliation | Sprint 206 Days 1-2 reconciled Sprint 197-205 plans, working notes, retrospectives, artifacts, PR review follow-ups, and current project-plan evidence into one closeout ledger. |
| Public claim calibration | Sprint 206 Days 3 and 5 updated README/INSTALL wording so the retained Sprint 198 Homebrew proof is represented as developer-mode local static source formula proof only, not a user-facing package-manager install path. |
| Maintainer/API calibration | Sprint 206 Day 6 aligned `docs/maintainer_guide.md` and `docs/api_reference.md` around Sprint 204 as the current generated API local-only policy owner, while preserving Sprint 179 and Sprint 186 historical context. |
| Project-plan status | Sprint 206 Day 7 updated `PROJECT_PLAN.md` so Sprint 197 is historical numbering-caveat evidence and Sprint 206 is the explicit closeout branch; Days 8-10 kept the row current as validation evidence landed. |
| Focused validation | Sprint 206 Day 9 passed support-doc, package-manager deferral, static-package deferral, generated API freshness, project-plan stale wording, and generated-output hygiene checks after reflowing one README guard marker. |
| Broad documentation/API validation | Sprint 206 Day 10 passed `make docs-check` and `make api-docs-freshness`; generated Doxygen HTML remained ignored under `docs/api/`. |
| Residual queue | Sprint 206 Day 12 refreshed `EPIC_18_RESIDUAL_QUEUE.md` so selected closures and re-deferrals are not described as unstarted work and broader claims remain explicit residuals. |
| Consistency hardening | Sprint 206 Day 13 cross-checked public, maintainer, planning, retrospective, residual, working-note, and artifact surfaces for stale current-status wording and claim-boundary drift. |
| Final closeout | Sprint 206 Day 14 closed the Sprint 206 day ledger, item dispositions, current-status docs, PR-ready notes, residual handoff, and generated-output hygiene plan. |
| Claim governance | Sprints 198-205 closed selected scopes or explicit re-deferrals without promoting broad package, Windows, ABI, hosted API, portable performance, release, or state-of-the-art claims. |

## Project-Plan Status

| Status | Current count | Rows |
| --- | ---: | --- |
| Historical final-validation evidence with numbering caveat | 6 | 197.1-197.6. |
| Closed selected scopes and explicit re-deferrals | 48 | 198.1-205.6, with Sprints 199 and 203 closed as re-deferrals rather than promotions. |
| Sprint 206 project-plan items complete | 6 | 206.1-206.6: evidence reconciliation, claim recalibration, project-plan status, integrated validation, retrospective, and residual queue. |
| Sprint 206 pending | 0 project-plan items | Day 14 final closeout review is recorded. |
| Residual queue refreshed | 1 document | `EPIC_18_RESIDUAL_QUEUE.md` is updated as the final residual handoff. |

The current status snapshot lives in [PROJECT_PLAN.md](./PROJECT_PLAN.md).
Historical Sprint 197 item-level evidence remains in
[SPRINT_197/artifacts/day8-project-plan-status.md](./SPRINT_197/artifacts/day8-project-plan-status.md).
Current Sprint 206 reconciliation and validation evidence begins with
[SPRINT_206/artifacts/day2-outcome-reconciliation.md](./SPRINT_206/artifacts/day2-outcome-reconciliation.md)
and continues through
[SPRINT_206/artifacts/day14-closeout-review.md](./SPRINT_206/artifacts/day14-closeout-review.md).

## Validation Evidence

| Evidence | Current result | Boundary |
| --- | --- | --- |
| Patch hygiene | Sprint 206 Days 7-10 `git diff --check` passed after each edit batch. | Whitespace and patch hygiene only. |
| Support/adoption guard | Sprint 206 Day 9 `make support-docs-guard` passed. | Support-truth and adoption wording guard only. |
| Package-manager non-claims | Sprint 206 Day 9 `bash scripts/package_manager_deferral_check.sh` passed after README marker reflow. | Deferral and non-claim enforcement only; no Homebrew/package-manager support claim. |
| Shared-library/dynamic ABI non-claims | Sprint 206 Day 9 `bash scripts/static_package_deferral_check.sh` passed. | Static package boundary and dynamic ABI deferral only. |
| Docs/API generation | Sprint 206 Day 10 `make docs-check` passed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. | Documentation generation and coverage only; no hosted API publication claim. |
| Generated API local-only policy | Sprint 206 Day 10 `make api-docs-freshness` passed, including Doxygen generation, coverage, local-only generated-output checks, workflow non-publication checks, and API routing checks. | Local-only generated API freshness, routing, staging, and non-publication guard only. |
| Generated output hygiene | Sprint 206 Day 10 `git status --ignored --short docs/api` reported `!! docs/api/`. | Confirms generated API output remains ignored, not source-controlled. |
| Final closeout hygiene | Sprint 206 Day 14 reruns patch hygiene, current-status stale wording, generated-output status, and C/header diff trigger checks after final documentation edits. | Documentation/planning closeout validation only. |
| Full C quality gate | Sprint 206 Days 8-14 recorded `make format && make lint && make test` as not required. | No `.c` or `.h` files changed through Day 14. |

## Changed Surface

| Metric | Day 14 evidence |
| --- | --- |
| Public docs edited | `README.md`, `INSTALL.md`. |
| Maintainer/API docs edited | `docs/maintainer_guide.md`, `docs/api_reference.md`. |
| Planning docs edited | `PROJECT_PLAN.md`, `EPIC_18_RESIDUAL_QUEUE.md`, this retrospective, and Sprint 206 plan/working notes/artifacts. |
| Sprint 206 artifacts present | Day 1 through Day 14 artifacts exist on this branch. |
| C source/header files edited | None through Day 14. |
| Workflows, guard scripts, manifests, schemas, tests, benchmarks, examples edited | None through Day 14. |
| Generated API/build artifacts tracked | None; generated output remains ignored. |

## Earned Claims

Epic 18 earns these narrow claims:

- Selected developer-mode local Homebrew static source formula proof exists
  with MIT metadata and proof records, but broad package-manager support
  remains unclaimed.
- Selected Windows Cholesky and Windows QR incompatible work received review,
  guards, and re-deferral decisions, but selected Windows freshness promotion
  remains unearned.
- One additional allocation-failure owner, selected `sparse_symbolic_lu()`,
  has bounded proof.
- One selected SVD test review surface was reduced into helper ownership with
  guards.
- One additional hosted selected benchmark freshness lane exists for macOS for
  the selected `SRT-BENCH-REFACTOR-CSC-NOS4` row.
- Generated API policy is stronger and remains local-only, ignored generated
  output rather than hosted or committed generated HTML.
- Support/readiness and adoption quick-reference wording are more coherent and
  guarded without expanding support claims.
- Sprint 206 reconciled the evidence, recalibrated claims, updated current
  status, passed focused plus broad documentation/API validation, and completed
  final closeout review.

## Non-Claims

Epic 18 does not claim:

- Homebrew/core readiness, bottles, Linuxbrew, public tap maintenance, vcpkg,
  Conan, pkgsrc, distro/system packages, binary packages, or broad
  package-manager distribution;
- shared-library packaging, dynamic ABI compatibility, runtime-loader behavior,
  SONAME/install-name/RPATH, DLL/import-library behavior, or static/shared
  selectors;
- promoted selected Windows Cholesky freshness, selected Windows QR
  incompatible freshness, Windows selected benchmark freshness, Windows
  Makefile parity, Windows `pkg-config` execution parity, or broad Windows
  parity;
- broad allocation-failure cleanup coverage beyond selected proof owners;
- repository-wide review-surface cleanup;
- hosted generated API HTML, artifact-published generated API HTML, or
  committed generated API HTML;
- release readiness;
- portable performance, speedup, scalability, backend superiority, or
  state-of-the-art performance;
- broad external-library parity against SuiteSparse, PETSc, Trilinos, Eigen,
  SciPy, NumPy, LAPACK, or other ecosystems;
- unqualified state-of-the-art sparse linear algebra library status.

## Residual Queue

`EPIC_18_RESIDUAL_QUEUE.md` is refreshed as of Sprint 206 Day 14. It keeps the
residual themes visible without treating them as current closure evidence:

| Priority theme | Closure target |
| --- | --- |
| Package-manager distribution | Promote only after provider-ready metadata, formula/tap policy, supported platform tier, install/test proof, and non-claim guard updates exist. |
| Windows selected freshness | Promote only after hosted Windows/MSVC evidence, selected manifest metadata, generated support tier, and non-claim wording align. |
| Additional reliability owners | Select one owner at a time, define invariants, add deterministic failure/retry proof, and keep broad reliability claims out of scope. |
| Review-surface reduction | Reduce one selected high-risk surface at a time with helper ownership and guard coverage. |
| Benchmark/platform evidence | Add hosted evidence one selected lane at a time with threshold-free methodology unless thresholds are explicitly designed and reviewed. |
| Generated API publication | Decide hosted/artifact/committed/local-only policy before any publication claim. |
| Release and ABI readiness | Define release, shared-library, dynamic ABI, loader, and package policies before any public readiness claim. |
| State-of-the-art evidence | Define external baselines, methodology, platform matrix, reliability semantics, package provenance, and acceptance thresholds before any broad claim. |

## State-Of-The-Art Assessment

Epic 18 does not earn an unqualified state-of-the-art sparse linear algebra
claim.

The defensible Day 14 assessment is that Epic 18 improved selected evidence
quality, support-boundary clarity, and closeout governance. It added or
retained selected proof for package metadata, allocation-failure ownership, SVD
test review-surface ownership, macOS selected benchmark freshness, generated
API local-only policy, and adoption/support routing. Those improvements are
useful product and maintenance evidence, but they do not establish broad
external-library parity, portable performance superiority, complete platform
parity, broad package distribution, release readiness, dynamic ABI policy, or
state-of-the-art numerical scope.

A future state-of-the-art claim would require exact external baselines,
versions, fixtures, matrix suites, workloads, metrics, tolerances, compilers,
platforms, package provenance, ABI policy, reliability semantics, performance
methodology, and reviewed hosted evidence for every claim.

## What Went Well

1. **Selected closure stayed bounded.** Sprints 198-205 closed selected scopes
   or explicit re-deferrals without converting evidence into broad support
   claims.

2. **Claim calibration became clearer.** README, INSTALL, maintainer guide,
   API reference, project-plan status, and Sprint 206 artifacts now agree on
   package, generated API, support, and state-of-the-art boundaries.

3. **Validation scope matched the diff.** Sprint 206 Days 8-10 separated
   focused documentation/package/API checks from the conditional full C gate.

4. **Generated output hygiene stayed intact.** Doxygen output was regenerated
   for validation but remained ignored under `docs/api/`.

5. **The numbering mismatch is explicit.** Sprint 197 remains historical
   final-validation evidence, while Sprint 206 is the current closeout branch.

## Could Be Better

1. **The Sprint 197/Sprint 206 overlap creates review overhead.** Future
   closeout work should use the project-plan sprint number from the start.

2. **Many high-value product gaps remain broad residuals.** Package-manager
   support, Windows freshness promotion, ABI/release readiness, hosted API
   publication, and state-of-the-art evidence need future implementation and
   hosted proof.

3. **Line-sensitive guards require source-text discipline.** Day 9 exposed
   that some shell guards rely on contiguous source-line markers; future docs
   edits should preserve guarded phrases or strengthen guards intentionally.

## Handoff

- Use `PROJECT_PLAN.md` as the current Sprint 197-206 status snapshot.
- Use `SPRINT_206/WORKING_NOTES.md` and Day 1-Day 14 artifacts as the current
  closeout evidence.
- Use `EPIC_18_RESIDUAL_QUEUE.md` as the current residual handoff.
- Keep `make format && make lint && make test` mandatory if any later Sprint
  206 day edits C source or headers.
