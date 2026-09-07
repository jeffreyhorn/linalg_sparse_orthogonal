# Epic 18 Retrospective

**Epic:** 18 - Selected Productization, Reliability & Evidence Promotion
**Sprints:** 197-206
**Status:** Sprint 197 Day 14 closeout draft with explicit residuals

## Epic Objective

Epic 18 starts from the Epic 17 closeout and targets a small number of
productization, platform freshness, reliability, reviewability, benchmark,
comparison, generated API, support-matrix, and final validation gaps. The plan
intentionally prefers complete closure of selected gaps over partial progress
across every remaining state-of-the-art shortcoming.

This Day 14 draft is based on the current `SPRINT_197` final-validation
artifacts. Those artifacts were requested under `SPRINT_197`, but they execute
the cited final-validation scope that `PROJECT_PLAN.md` labels as Sprint 206.
At Day 14, Sprints 198 through 205 have no branch-local implementation artifact
directories, validation records, or PR evidence. This retrospective therefore
records interim evidence and residuals without promoting future sprint work.

## Sprint Outcomes

| Sprint | Current outcome |
| --- | --- |
| 197 | Complete for the requested final-validation branch path, with numbering caveat. The branch contains the requested final-validation plan, working notes, evidence ledger, claim audits, no-promotion records, project-plan interim status snapshot, validation logs, retrospective draft, residual queue, and final closeout review. |
| 198 | Pending future execution. Homebrew/package-manager license metadata, proof execution, guard promotion, docs promotion, and validation evidence are not present on this branch. |
| 199 | Pending future execution. Selected Windows Cholesky hosted evidence review, manifest promotion/re-deferral, normalizer hardening, workflow guard updates, docs calibration, and validation evidence are not present on this branch. |
| 200 | Pending future execution. Additional allocation-failure owner selection, invariants, harness work, regressions, focused gate, docs, and validation evidence are not present on this branch. |
| 201 | Pending future execution. Additional review-surface ranking, extraction, guard, focused regression, docs, and validation evidence are not present on this branch. |
| 202 | Pending future execution. Additional hosted selected benchmark platform/row evidence, methodology metadata, workflow lane, freshness tests, docs, and validation evidence are not present on this branch. |
| 203 | Pending future execution. Windows QR incompatible comparison MSVC/CMake proof or re-deferral, generator fixes, manifest decision, tests, docs, and validation evidence are not present on this branch. |
| 204 | Pending future execution. Generated API publication/local-only product decision, guard implementation, freshness/link checks, routing docs, claim guard, and validation evidence are not present on this branch. |
| 205 | Pending future execution. Support/adoption quick reference, support truth consolidation, diagnostics vocabulary, claim guards, docs, and validation evidence are not present on this branch. |
| 206 | Requested final-validation evidence is recorded through the `SPRINT_197` artifacts. Evidence reconciliation, claim recalibration, project-plan status, focused/full gate decisions, retrospective draft, residual queue, claim decision, and final closeout review are complete for this branch state. |

## Major Outcomes

| Area | Day 14 outcome |
| --- | --- |
| Evidence reconciliation | Sprint 197 Day 1-3 artifacts inventoried Epic 18 evidence sources, created an outcome ledger, classified evidence conflicts, and separated future sprint evidence from completed proof. |
| Public claim calibration | Day 4 audited public claim surfaces and Day 6 recorded a no-promotion decision because current public docs already retain package, Windows, benchmark, API, ABI, release, and state-of-the-art non-claims. |
| Maintainer/API calibration | Day 5 audited maintainer/API/corpus/schema surfaces and Day 7 recorded a no-promotion decision because current owner docs already identify evidence gates and retained boundaries. |
| Project-plan status | Day 8 added an interim `PROJECT_PLAN.md` status snapshot and a full item-level ledger that marks Sprints 198-205 as pending future execution rather than complete. |
| Focused validation | Day 10 ran focused docs/API, Windows ownership, package-manager deferral, static-package deferral, and source-list confidence gates. |
| Full-gate decision | Day 11 verified the branch remains docs/planning-only, ran required docs/planning checks, and recorded that the full C gate is not required without `*.c` or `*.h` changes. |
| Claim governance | Day 13 publishes the residual queue and records that no stronger support, package, Windows freshness, performance, API publication, release, ABI, or state-of-the-art claim has been earned on this branch. |
| Final closeout review | Day 14 verifies artifact completeness, internal consistency, claim calibration, validation currency, generated-artifact hygiene, and PR summary inputs. |

## Project-Plan Status

| Status | Current count | Rows |
| --- | ---: | --- |
| In progress with numbering caveat | 6 | 197.1-197.6. |
| Pending future execution | 48 | 198.1-205.6. |
| Partial final-validation evidence | 4 | 206.1-206.4. |
| Requested final-validation evidence recorded | 2 | 206.5-206.6. |
| Pending final-validation work | 0 | Day 14 final closeout review is recorded for the current branch state. |
| Complete | 0 | No Epic 18 implementation item is finally complete at Day 14. |
| Narrowed | 0 | No Epic 18 implementation item has been narrowed at Day 14. |
| Deferred | 0 | No Epic 18 implementation item has been formally deferred at Day 14. |
| Residualized | 10 residual entries | `EPIC_18_RESIDUAL_QUEUE.md` publishes prioritized residuals E18-RQ-001 through E18-RQ-010. |
| Superseded | 0 | No Epic 18 item has been superseded at Day 14. |

The evidence-linked interim status snapshot lives in
[PROJECT_PLAN.md](./PROJECT_PLAN.md). The full item-level Day 8 ledger lives in
[SPRINT_197/artifacts/day8-project-plan-status.md](./SPRINT_197/artifacts/day8-project-plan-status.md).

## Validation Evidence

| Evidence | Current result | Boundary |
| --- | --- | --- |
| Patch hygiene | Day 14 `git diff --check` rerun passed. | Whitespace and patch hygiene only. |
| Docs/API generation | Day 14 `make docs-check` rerun passed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. | Documentation generation and coverage only; no hosted API publication claim. |
| Generated API local-only policy | Day 10 `make api-docs-freshness` passed and confirmed generated API HTML remains ignored, untracked, unstaged, and not a workflow publication path. | Local-only generated API freshness and staging guard only. |
| Windows PowerShell ownership | Day 10 `make windows-powershell-guard` passed. | Workflow/snippet ownership and claim-boundary validation only; local `pwsh` unavailable remains an environment residual. |
| Package-manager non-claims | Day 10 `bash scripts/package_manager_deferral_check.sh` passed. | Deferral and non-claim enforcement only; no Homebrew/package-manager support claim. |
| Shared-library/dynamic ABI non-claims | Day 10 `bash scripts/static_package_deferral_check.sh` passed. | Static package boundary and dynamic ABI deferral only. |
| Source-list confidence | Day 10 `make source-list-check` passed with 49 library sources. | Source registration confidence only; no implementation behavior proof. |
| Full C quality gate | Day 11 recorded `make format && make lint && make test` as not required. | No `.c` or `.h` files changed through Day 14. |
| Generated report freshness | Skipped with documented reasons on Days 10-11. | No report generator, manifest, normalizer, or report-doc source changed; local regeneration would not create hosted evidence. |
| Benchmark freshness | Skipped with documented reasons on Days 10-11. | No benchmark code, manifest, methodology, docs, or checker source changed; local output would not create portable performance evidence. |
| Final closeout review | Day 14 artifact completeness and consistency review completed. | Review packaging only; no new implementation proof. |

## Changed Surface

| Metric | Day 14 evidence |
| --- | --- |
| Epic project plan | `PROJECT_PLAN.md` has an interim status snapshot. |
| Sprint plan | `SPRINT_197/PLAN.md` exists. |
| Sprint working notes | `SPRINT_197/WORKING_NOTES.md` records Days 1-14. |
| Sprint artifacts | Day 1-14 artifacts exist, including the retrospective draft record, residual queue/claim decision, and final closeout review. |
| Public docs edited | None through Day 14. |
| Maintainer/API/corpus/schema docs edited | None through Day 14. |
| C source/header files edited | None through Day 14. |
| Generated API/build artifacts tracked | None; generated output remains ignored. |

## Earned Claims

At Day 14, Epic 18 earns only these narrow claims:

- The requested `SPRINT_197` final-validation path has a day-by-day plan,
  working notes, evidence intake, outcome ledger, evidence conflict review,
  public and maintainer/API claim audits, no-promotion records, project-plan
  interim status snapshot, validation matrix, focused validation log, full-gate
  decision log, this retrospective draft, a prioritized residual queue, and a
  final closeout review package.
- Current public and maintainer/API docs were reviewed against the available
  evidence, and no stronger claim was promoted.
- Focused documentation, package-boundary, Windows ownership, API local-only,
  and source-list confidence gates passed for the current branch state.
- The branch remains docs/planning-only through Day 14, so the full C gate is
  not required unless later days edit `*.c` or `*.h` files.

## Non-Claims

Epic 18 does not currently claim:

- completed Homebrew/package-manager support;
- approved standalone license metadata for Homebrew proof;
- Homebrew/core, Linuxbrew, bottles, taps, vcpkg, Conan, pkgsrc, or
  distro/system package availability;
- shared-library packaging, dynamic ABI compatibility, runtime-loader behavior,
  SONAME/install-name/RPATH, DLL/import-library behavior, or static/shared
  selectors;
- promoted Windows selected Cholesky freshness;
- broad Windows report freshness, selected oracle freshness on Windows,
  selected benchmark freshness on Windows, Windows QR incompatible freshness,
  Windows Makefile parity, Windows `pkg-config` execution parity, or broad
  Windows parity;
- additional allocation-failure owner proof beyond prior selected owners;
- additional review-surface reduction beyond prior Epic 17 work;
- hosted selected benchmark freshness on an additional platform;
- hosted or artifact-published generated API HTML;
- release readiness;
- portable performance, speedup, scalability, or state-of-the-art performance;
- broad external-library parity against SuiteSparse, PETSc, Trilinos, Eigen,
  SciPy, NumPy, LAPACK, or other ecosystems;
- unqualified state-of-the-art sparse linear algebra library status.

## Residual Queue

The prioritized residual handoff with owner surfaces, expected evidence,
validation commands, and claim boundaries is published in
[EPIC_18_RESIDUAL_QUEUE.md](./EPIC_18_RESIDUAL_QUEUE.md).

| Priority | Residual | Closure target |
| ---: | --- | --- |
| 1 | Homebrew/package-manager support | Add approved license metadata, run the selected Homebrew proof, update guards, and promote only the earned support tier. |
| 2 | Selected Windows Cholesky freshness | Review hosted Windows evidence, align manifest metadata, rerun normalizer/workflow/PowerShell guards, and promote or re-defer explicitly. |
| 3 | Additional allocation-failure owner | Select one owner, record invariants, extend deterministic harness coverage, add regressions, and create a focused gate. |
| 4 | Additional review-surface reduction | Select one high-risk surface, extract or refactor behavior-preserving helpers, add ownership guards, and rerun focused/full validation. |
| 5 | Additional hosted benchmark freshness | Select one platform/row, add methodology metadata, create hosted artifact evidence, and keep performance claims methodology-bound. |
| 6 | Windows QR incompatible comparison | Add MSVC/CMake generation evidence, path/normalizer tests, exact manifest metadata, and calibrated selected-claim docs. |
| 7 | Generated API publication policy | Decide hosted/artifact/committed/local-only policy and implement matching guards, link checks, and docs. |
| 8 | Adoption/support simplification | Consolidate support truth and diagnostics vocabulary without weakening support/package/platform/performance boundaries. |
| 9 | Release and ABI readiness | Define release, shared-library, dynamic ABI, loader, and package policies before any public readiness claim. |
| 10 | State-of-the-art evidence | Define external baselines, methodology, platform matrix, reliability semantics, package provenance, and acceptance thresholds before any broad claim. |

## State-Of-The-Art Assessment

Epic 18 does not currently earn an unqualified state-of-the-art sparse linear
algebra claim.

The defensible Day 14 assessment is that the branch improves closeout
discipline: it reconciles evidence, prevents premature support promotion,
records validation ownership, and makes pending work visible. Those are useful
governance improvements, but they do not add new solver capability, broad
external comparison evidence, portable performance data, platform parity,
package-manager distribution, release readiness, shared-library ABI support, or
hosted API publication.

A future state-of-the-art claim would require exact external baselines,
versions, fixtures, matrix suites, workloads, metrics, tolerances, compilers,
platforms, package provenance, ABI policy, reliability semantics, performance
methodology, and reviewed hosted evidence for every claim.

## What Went Well

1. **Evidence stayed separated from intent.** Future Sprint 198-205 goals were
   not counted as completed evidence.

2. **Claim calibration stayed conservative.** Public and maintainer docs were
   audited without promoting unsupported package, Windows, benchmark, API,
   release, ABI, or state-of-the-art claims.

3. **Validation ownership became explicit.** Day 9-11 artifacts map changed
   surfaces to focused gates and full-gate triggers.

4. **Environment residuals stayed visible.** Local PowerShell unavailability,
   hosted Windows evidence, Homebrew proof prerequisites, benchmark hosting,
   and generated report locality are recorded as boundaries.

5. **The numbering mismatch is documented.** The branch preserves the user's
   requested `SPRINT_197` path while retaining traceability to the project
   plan's Sprint 206 final-validation item numbers.

## Could Be Better

1. **The branch executes final validation before the implementation sprints.**
   That makes most Epic 18 outcomes necessarily pending rather than complete.

2. **The project-plan numbering mismatch creates review overhead.** Future
   closeout work should either use the plan's Sprint 206 path or explicitly
   revise the project plan before execution.

3. **Most high-value product gaps remain open.** Package-manager support,
   Windows freshness, additional reliability proof, review-surface reduction,
   benchmark platform freshness, Windows QR comparison, API publication,
   release readiness, ABI support, and state-of-the-art evidence still require
   implementation sprints.

4. **No new implementation evidence exists yet.** The current branch improves
   planning and claim governance only.

## Handoff

- Use `EPIC_18_RESIDUAL_QUEUE.md` as the prioritized next-epic handoff.
- Keep the Sprint 197/Sprint 206 numbering caveat visible in any PR review.
- Keep `make format && make lint && make test` mandatory if any later day edits
  C source or headers.
