# Epic 17 Retrospective

**Epic:** 17 - Product Proof, Platform Freshness & Reviewability
**Sprints:** 187-196
**Status:** Complete

## Epic Objective

Epic 17 started from the Epic 16 closeout and focused on closing a smaller set
of evidence, productization, platform, reviewability, and reliability gaps with
bounded proof instead of broad claims. The epic converted the post-Epic-16
review into a selected closure ledger, hardened package proof boundaries,
added PowerShell validation ownership, narrowed one Windows report freshness
path, added one external comparison family, promoted one methodology-bound
performance evidence lane, reduced one large QR review surface, simplified
adoption/API guidance, and added one selected allocation-failure proof.

The epic deliberately did not try to claim broad package-manager support,
broad Windows parity, portable performance, release readiness,
shared-library/dynamic ABI support, broad external ecosystem parity, broad
allocation-failure coverage, or unqualified state-of-the-art sparse linear
algebra status. Those remain future product, platform, methodology, or
research claims that require stronger evidence than Epic 17 produced.

## Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 187 | Established the Epic 17 baseline, gap ledger, selected closures, acceptance gates, quality map, and handoff records. |
| 188 | Hardened Homebrew proof material and package guards while keeping package-manager/Homebrew support blocked by missing approved standalone license metadata. |
| 189 | Added source-controlled PowerShell validation ownership with local unavailable semantics and hosted validation wiring. |
| 190 | Added one guarded selected Windows Cholesky workflow path while leaving selected manifest promotion dependent on hosted evidence review. |
| 191 | Added one bounded local-only `qr-incompatible-ls` external comparison family with fixture, runner, manifest, docs, and freshness diagnostics. |
| 192 | Added one selected Linux hosted performance evidence lane for `bench_refactor_csc` with methodology metadata and threshold-free interpretation. |
| 193 | Reduced one selected QR external-reference review surface through helper extraction and ownership guards. |
| 194 | Simplified adoption/API guidance with a compact support/readiness matrix, installed-consumer routing, diagnostics wording, and selected header narrative cleanup. |
| 195 | Added deterministic allocation-failure cleanup, stale-output suppression, and retry evidence for selected `sparse_symbolic_cholesky()` output allocation. |
| 196 | Reconciled Epic 17 evidence, recalibrated public and maintainer claims, updated project-plan status, published this retrospective and the residual queue, completed focused/full-quality validation, and packaged the final closeout evidence. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence reconciliation | Sprint 196 Day 2 reconciled Sprint 187-195 outcomes, and Day 7 added item-level status tables to the Epic 17 project plan. |
| Package proof | Local Homebrew proof material, package deferral guards, static-package guards, and install checks improved; package-manager support remains explicitly unclaimed pending approved license metadata. |
| Windows/PowerShell | PowerShell validation ownership exists with local unavailable semantics and hosted workflow wiring; one selected Windows Cholesky path exists as guarded workflow evidence. |
| Selected comparisons | The selected comparison set gained bounded QR incompatible least-squares evidence while retaining fixture, target, dependency, and platform boundaries. |
| Selected performance | One Linux hosted selected performance evidence lane exists with methodology metadata, exact hosted bundle scope, and threshold-free interpretation. |
| Review surface | One QR external-reference rank/nullspace/threshold cluster was extracted and guarded without broad QR behavior claims. |
| Adoption/API coherence | Support/readiness truth is easier to find, installed-consumer routing is clearer, diagnostics wording is more coherent, and selected header narrative was reduced without public declaration changes. |
| Reliability | One selected symbolic Cholesky output allocation path has deterministic failure, cleanup, stale-output suppression, and retry proof. |
| Claim governance | README, INSTALL, maintainer guide, corpus docs, benchmark docs, API docs, project-plan status, residual queue, and this retrospective now align around earned evidence and retained non-claims. |

## Project-Plan Status

| Status | Current count | Rows |
| --- | ---: | --- |
| Complete | 50 | Selected complete rows from Sprints 187-196 as recorded in `PROJECT_PLAN.md`. |
| Complete with guarded residual | 2 | 188.5, 188.6. |
| Complete with hosted evidence pending at closeout | 1 | 189.3. |
| Complete with residual narrowed | 2 | 190.5, 190.6. |
| Narrowed | 2 | 190.2, 192.4. |
| Deferred | 1 | 188.2. |
| Residualized | 2 | 188.1, 190.3. |

These are final Epic 17 project-plan closeout counts after Sprint 196 Day 14.

The evidence-linked project-plan status tables live in
[PROJECT_PLAN.md](./PROJECT_PLAN.md). The Sprint 196 Day 7 status artifact
records the status vocabulary; the Day 10 project-plan update records residual
queue completion.

## Validation Evidence

| Evidence | Current result | Boundary |
| --- | --- | --- |
| Package/Homebrew proof guards | Sprint 188 package-manager and static-package guards passed for the guarded unavailable state; Sprint 196 Day 5 repeated package claim guards. | Proof material and non-claim enforcement only; no Homebrew or package-manager support claim. |
| Source install checks | Sprint 188 Make and CMake install validation passed. | Static/source install confidence only; no package-manager provider claim. |
| Windows PowerShell ownership | Sprint 189 validation command, guard tests, and hosted workflow wiring landed; Sprint 196 Day 5-6 `make windows-powershell-guard` passed. | PowerShell workflow ownership and claim-boundary validation only, not broad Windows parity. |
| Selected Cholesky workflow/freshness checks | Sprint 190 selected workflow and target-specific freshness tests passed locally. | Guarded selected Windows Cholesky workflow evidence pending hosted evidence review and manifest promotion. |
| Selected QR incompatible comparison | Sprint 191 direct generator, QR owner test, runner tests, target freshness, normalizer tests, manifest checks, and workflow guards passed. | One local-only selected comparison family; no QR or ecosystem parity. |
| Selected performance evidence | Sprint 192 canonical benchmark freshness, selected-performance docs guard, selected workflow guard, selected report schema, normalizer tests, and freshness tests passed. | One Linux selected methodology-bound lane; no timing threshold, portability, or performance superiority claim. |
| QR review-surface guard | Sprint 193 source-list, QR helper regression tests, QR helper guard, reviewed CMake path, full C quality gate, and focused C tests passed. | One selected QR helper/test cluster only. |
| Adoption/API coherence | Sprint 194 docs/API coverage, generated API local-only guard, QR header docs guard, install proofs, selected freshness checks, and full C quality gate passed. | User-doc routing, selected public-header cleanup, and local generated API freshness only. |
| Symbolic allocation-failure proof | Sprint 195 focused registration guard, focused Make gate, CMake selector, CTest selector, claim-boundary checks, and full C quality gate passed. | Selected `sparse_symbolic_cholesky()` output allocation only. |
| Sprint 196 focused integrated validation | Day 11 passed package/install guards, Windows PowerShell guard, selected report target and normalizer tests, selected comparison/oracle freshness, selected performance freshness, QR/LDLT review-surface guards, symbolic allocation-failure registration and focused gate, corpus schema, docs/API freshness, and final whitespace. | Focused evidence-owner confidence for current Sprint 196 documentation and planning changes. |
| Sprint 196 full-quality decision | Day 12 `make format` passed with no `.c` or `.h` diff, `make lint` passed strict warning builds, clang-tidy, and cppcheck, and `make test` was not required because Sprint 196 changed documentation/planning surfaces only. | Full-quality confidence for the changed surface; not a new solver behavior proof. |
| Sprint 196 final claim review | Day 13 re-read public, maintainer, planning, retrospective, residual, benchmark, corpus, and API-adjacent docs; high-risk claim grep found only bounded claims or explicit non-claims; `make docs-check` passed after the review updates. | Claim consistency and retrospective readiness for closeout packaging; no new implementation proof. |
| Sprint 196 closeout package | Day 14 finalized the item-to-evidence traceability checklist, residual/non-claim handoff, and review package. | Review packaging only; no new implementation proof. |

## Changed Surface

| Metric | Current evidence |
| --- | --- |
| Epic sprint plans | Sprint 187-196 each have a `PLAN.md`. |
| Epic sprint retrospectives | Sprint 187-195 retrospectives exist; Sprint 196 working notes and artifacts support the final sprint retrospective. |
| Project-plan status tables | Sprints 187-196 now have closeout-status tables in `PROJECT_PLAN.md`. |
| Library source count in source-list checks | Sprint 193-195 retrospectives report 49 library sources. |
| Generated API coverage | Sprint 194 and Sprint 196 docs checks report 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| Selected oracle freshness row count | Sprint 194 retrospective reports 54 selected oracle freshness rows. |
| Selected comparison freshness row count | Sprint 191 and Sprint 194 retrospectives report 46 selected comparison freshness rows. |
| Hosted selected performance lanes added or hardened | Sprint 192 hardened one Linux selected performance lane. |
| Selected review-surface clusters reduced | Sprint 193 reduced one selected QR external-reference cluster. |
| Selected allocation-failure owners proved in Epic 17 | Sprint 195 proved one selected symbolic Cholesky owner. |
| C implementation edits in Epic 17 selected closures | Sprint 195 reports 1 C implementation file changed. |
| Public header edits in Epic 17 selected closures | Sprint 194 reports 6 public header files changed. |

No Day 14 closeout packaging changed the implementation, public header, report,
or benchmark surface.

## Earned Claims

Epic 17 earns these claims with qualifiers:

- Epic 17 has a source-controlled baseline, gap ledger, acceptance-gate map,
  quality-surface map, status pass, and selected closure evidence.
- Local Homebrew proof material and package guards are stricter, but
  package-manager support remains unavailable until license metadata is
  approved and proof succeeds.
- PowerShell validation has an owned source-controlled command, local
  unavailable semantics, guard coverage, and hosted workflow wiring.
- One selected Windows Cholesky workflow path exists as guarded evidence.
- One bounded `qr-incompatible-ls` comparison family exists with fixture,
  runner, manifest, docs, and freshness diagnostics.
- One selected Linux hosted performance evidence lane exists with exact bundle
  scope and threshold-free methodology metadata.
- One selected QR external-reference review surface was reduced with helper
  ownership guards.
- Adoption/API guidance and support-readiness routing are clearer without
  public API/ABI expansion.
- One selected symbolic Cholesky output allocation path has deterministic
  allocation-failure cleanup, stale-output suppression, and retry proof.
- Public and maintainer documentation now states the earned support level and
  retained non-claims more consistently.

## Non-Claims

Epic 17 does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against SuiteSparse, PETSc,
  Trilinos, Eigen, SciPy, LAPACK, NumPy, or package-manager ecosystems;
- broad solver correctness beyond named fixtures, maintained corpus rows, and
  selected proof owners;
- portable performance superiority, backend superiority, OpenMP speedup proof,
  thresholded performance regression policy, or state-of-the-art performance;
- package-manager provider availability, Homebrew/core readiness, bottles,
  Linuxbrew support, or public tap support;
- broad Windows package, Makefile, `pkg-config`, generated-report, selected
  oracle, selected benchmark, or platform parity;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- release readiness;
- hosted generated API HTML publication or committed generated API HTML;
- broad allocation-failure cleanup coverage beyond selected proof owners.

## Residual Queue

The detailed prioritized residual handoff with owner surfaces, expected
evidence, validation commands, and deferral horizons is published in
[EPIC_17_RESIDUAL_QUEUE.md](./EPIC_17_RESIDUAL_QUEUE.md).

| Priority | Residual | Closure target |
| ---: | --- | --- |
| 1 | Package-manager/Homebrew support | Add approved standalone license metadata and exact Homebrew license identifier, rerun proof and guards, then calibrate docs. |
| 2 | Selected Cholesky Windows freshness promotion | Review hosted evidence, inspect bundle, promote exact manifest metadata, and rerun coupled guards. |
| 3 | Additional allocation-failure owner | Select one owner and repeat the Sprint 195 invariant, harness, regression, gate, and docs pattern. |
| 4 | Additional QR review-surface cluster | Select one cluster, extract with behavior-preserving guard coverage, and rerun focused/full validation. |
| 5 | Windows/macOS selected benchmark freshness | Add hosted platform lanes and selected artifact validation without broadening Linux selected claims. |
| 6 | Windows QR incompatible freshness | Add MSVC/CMake proof and exact selected metadata before any promotion. |

Long-horizon residuals remain shared-library/dynamic ABI support, broad
Windows parity, portable performance, release benchmark claims, OS OOM and
concurrency semantics, hosted generated API publication, optional package
baselines, and unqualified state-of-the-art positioning.

## State-Of-The-Art Assessment

Epic 17 does not earn an unqualified state-of-the-art sparse linear algebra
claim.

The defensible assessment is narrower: Epic 17 improves evidence discipline,
selected hosted evidence, package and Windows claim honesty, comparison
breadth for one selected QR family, methodology-bound performance evidence for
one Linux lane, one review-surface boundary, adoption/API coherence, and one
selected allocation-failure proof. Those are meaningful engineering
improvements, but they do not establish comparative state-of-the-art status
against mature sparse linear algebra ecosystems.

A future state-of-the-art claim would require named external libraries,
versions, fixtures, metrics, tolerances, platforms, compilers, package
provenance, performance methodology, reliability semantics, ABI/package
support policy, support-tier boundaries, and reviewed hosted evidence for each
specific claim.

## What Went Well

1. **The epic stayed selected instead of sprawling.** Sprints closed one proof
   path, one workflow path, one comparison family, one performance lane, one
   review surface, one adoption/API simplification pass, and one reliability
   owner rather than partially broadening every gap.

2. **Claim boundaries survived implementation pressure.** Package proof,
   Windows workflow evidence, hosted performance rows, local-only comparison
   families, and selected reliability proof all retained explicit non-claims.

3. **Evidence ownership became more concrete.** Manifests, workflows,
   Makefile targets, Python guards, docs guards, maintainer guidance, and
   sprint artifacts now identify which surface owns each selected claim.

4. **Windows and package work avoided accidental promotion.** Missing license
   metadata and pending hosted/manifest evidence stayed visible as blockers
   instead of being hidden behind adjacent proof material.

5. **Performance and comparison evidence remained reviewable.** The work used
   exact target keys, fixtures, artifact names, rows, dependency policies, and
   methodology fields.

6. **The reliability proof pattern is reusable.** Sprint 195's invariant-first
   approach gives future allocation-failure owners a practical template:
   select one owner, record invariants, add deterministic failure tests, add a
   focused gate, calibrate docs, and run the full quality gate when C changes.

## Could Be Better

1. **Several residuals are product decisions, not implementation chores.**
   Package-manager support, shared-library/dynamic ABI support, hosted API
   publication, broad Windows parity, and release readiness all need explicit
   future product choices before code alone can close them.

2. **Claim governance is still distributed.** README, INSTALL, maintainer
   guide, corpus docs, benchmark docs, scripts, workflows, manifests, headers,
   and planning artifacts all carry parts of the support story.

3. **Hosted evidence remains selected-lane scoped.** Epic 17 improved hosted
   ownership for selected lanes, but broad generated reports, broad benchmark
   publication, and broad Windows/macOS freshness are still not covered.

4. **Performance evidence remains statistically thin.** The selected hosted
   lane is useful for methodology and artifact freshness, but it is still
   threshold-free and not portable performance proof.

5. **Review-surface reduction is incremental.** One QR cluster is smaller and
   guarded, but other large helper/test/source surfaces remain future work.

6. **Reliability proof remains narrow by design.** One selected symbolic
   Cholesky output allocation path is proved; other symbolic, analysis, direct
   solver, matrix construction, OS OOM, and concurrency paths remain outside
   the claim.

## Key Deliverables

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [EPIC_17_RESIDUAL_QUEUE.md](./EPIC_17_RESIDUAL_QUEUE.md)
- [EPIC_17_RETROSPECTIVE.md](./EPIC_17_RETROSPECTIVE.md)
- [SPRINT_187/RETROSPECTIVE.md](./SPRINT_187/RETROSPECTIVE.md)
- [SPRINT_188/RETROSPECTIVE.md](./SPRINT_188/RETROSPECTIVE.md)
- [SPRINT_189/RETROSPECTIVE.md](./SPRINT_189/RETROSPECTIVE.md)
- [SPRINT_190/RETROSPECTIVE.md](./SPRINT_190/RETROSPECTIVE.md)
- [SPRINT_191/RETROSPECTIVE.md](./SPRINT_191/RETROSPECTIVE.md)
- [SPRINT_192/RETROSPECTIVE.md](./SPRINT_192/RETROSPECTIVE.md)
- [SPRINT_193/RETROSPECTIVE.md](./SPRINT_193/RETROSPECTIVE.md)
- [SPRINT_194/RETROSPECTIVE.md](./SPRINT_194/RETROSPECTIVE.md)
- [SPRINT_195/RETROSPECTIVE.md](./SPRINT_195/RETROSPECTIVE.md)
- [SPRINT_196/WORKING_NOTES.md](./SPRINT_196/WORKING_NOTES.md)
- [SPRINT_196/artifacts/day11-integrated-focused-validation.md](./SPRINT_196/artifacts/day11-integrated-focused-validation.md)
- [SPRINT_196/artifacts/day12-full-quality-decision.md](./SPRINT_196/artifacts/day12-full-quality-decision.md)
- [SPRINT_196/artifacts/day13-final-claim-retrospective-review.md](./SPRINT_196/artifacts/day13-final-claim-retrospective-review.md)
- [SPRINT_196/artifacts/day14-closeout-review-package.md](./SPRINT_196/artifacts/day14-closeout-review-package.md)
