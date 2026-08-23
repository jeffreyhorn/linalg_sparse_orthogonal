# Epic 15 Retrospective

**Epic:** 15 - Evidence Publication, ABI Decision & Adoption Hardening
**Sprints:** 167-176
**Status:** Complete with explicit residuals

## Epic Objective

Epic 15 started from the Epic 14 residual queue and focused on closing a
bounded set of credibility, productization, and claim-governance gaps. The
epic promoted selected performance and comparison evidence, hardened
methodology metadata, closed shared-library and package-manager product
decisions, continued public-header/API cleanup, clarified generated API HTML
status, and added one targeted allocation-failure proof.

The epic deliberately did not attempt an unqualified state-of-the-art sparse
linear algebra claim. Its working standard was narrower: every public claim
must identify the evidence owner, support tier, validation gate, and retained
non-claims. Unsupported surfaces remain visible instead of being implied by
adjacent source-controlled artifacts.

## Source Artifact Note

Several Sprint 167-176 prompts referenced older Epic 12 project-plan sections
while the active merged Epic 15 planning source lived in
`docs/planning/EPIC_15/PROJECT_PLAN.md`. Each sprint recorded the mismatch in
working notes and tied execution to the active Epic 15 plan.

## Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 167 | Established the Epic 15 baseline, evidence ledger, selected gap list, acceptance gates, quality surfaces, and non-claim register. |
| 168 | Promoted one selected Linux hosted performance freshness lane for `bench_refactor_csc` on `nos4.mtx --repeat 1`, with methodology-bound wording. |
| 169 | Hardened selected performance methodology with stable metadata, repeat/warmup/variance policy, threshold-free interpretation, and local sentinel tests. |
| 170 | Closed the shared-library ABI decision as static-first-only continuation with explicit shared/dynamic ABI rejection and package metadata guards. |
| 171 | Closed package-manager readiness as formal provider deferral with source-install guidance and an executable non-claim guard. |
| 172 | Cleaned one public-header family, `include/sparse_lu.h`, with declaration preservation, tutorial/API alignment, and focused docs guardrails. |
| 173 | Closed generated API HTML publication as local-only with freshness and staging guards rather than hosted, committed, or artifact-only publication. |
| 174 | Added one bounded external comparison family: linked-list LU square solve on `lu_nonsym_square_5` against a source-controlled dense reference. |
| 175 | Promoted selected comparison freshness on Linux and macOS for the selected QR, partial-SVD, and LU comparison targets, with workflow guards. |
| 176 | Added a deterministic allocation-failure proof for iterative repeated-run handles, documented cleanup invariants, recalibrated claims, and ran integrated validation. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence ledger | Epic 15 begins from a source-controlled ledger and ends with explicit earned claims, support tiers, residuals, and non-claims. |
| Hosted performance | One selected Linux hosted performance lane exists for a single methodology-bound benchmark row. |
| Performance methodology | Selected performance rows have stable methodology fields, variance/warmup policy, and threshold-free interpretation. |
| Shared-library ABI | Static-first-only remains the product decision; shared-library, dynamic ABI, symbol/export, and runtime-loader behavior remain rejected or deferred. |
| Package-manager readiness | Provider package-manager support is formally deferred and guarded; source install through Make/CMake remains the maintained path. |
| Public headers/API | One additional high-impact public header family has clearer lifecycle, ownership, and usage wording without declaration drift. |
| Generated API HTML | Generated HTML is explicitly local-only, with freshness and staging checks preserving source-header-first API authority. |
| External comparison | The selected comparison surface now includes bounded QR, partial-SVD, and LU fixture-local families. |
| Cross-platform report freshness | Linux and macOS selected comparison freshness are reviewed hosted evidence lanes for the named selected artifacts. |
| Allocation failure | One selected subsystem, iterative repeated-run handle prepare/growth cleanup, has deterministic allocation-failure proof and focused validation. |
| Claim governance | README, maintainer guidance, report/index docs, and planning artifacts distinguish hosted evidence, local evidence, advisory rows, deferred products, and non-claims. |

## Validation Evidence

| Evidence | Result | Boundary |
| --- | --- | --- |
| Sprint 168 hosted selected performance lane | Local and hosted workflow wiring validated for the selected Linux `bench_refactor_csc` row. | One selected benchmark family and methodology only, not portable performance superiority. |
| Sprint 169 performance methodology hardening | Benchmark canonical freshness, sentinel, schema, and report guard checks passed. | Methodology-bound rows only. |
| Sprint 170 shared-library ABI decision | Static package deferral, install metadata, and shared-library rejection guards passed. | Static-first package support only; no shared-library or dynamic ABI support. |
| Sprint 171 package-manager deferral | Package-manager deferral guard passed and docs retained provider non-claims. | No package-manager provider availability. |
| Sprint 172 public-header cleanup | Full C quality gate and declaration-preserving checks passed after `sparse_lu.h` cleanup. | Selected header family only. |
| Sprint 173 generated API HTML local-only decision | API docs freshness and local-only staging checks passed. | Local generated HTML status, not hosted publication. |
| Sprint 174 LU comparison family | Comparison runner, report freshness, and relevant solver tests passed. | One LU fixture-local comparison family. |
| Sprint 175 selected comparison freshness | Linux/macOS selected comparison workflow guards and report freshness checks passed. | Selected comparison artifacts only; not broad report freshness or selected oracle freshness on macOS. |
| Sprint 176 allocation-failure proof | `make iterative-allocation-failure-gate` passed with `test_iterative`: `85` tests, `0` failures, `743` assertions. | CG, GMRES, and MINRES repeated-run handle prepare/growth cleanup only. |
| Sprint 176 integrated validation | Passed package-manager deferral, static package deferral, report-index normalizer, selected comparison workflow, benchmark freshness, `make format && make lint && make test`, and `git diff --check`. | Local validation plus selected hosted-lane structure; final PR CI remains hosted activation evidence. |

## Earned Claims

Epic 15 earns these claims with qualifiers:

- The project has a source-controlled Epic 15 evidence ledger and closeout
  plan that maps selected claims to evidence owners and non-claims.
- The project has one selected Linux hosted performance freshness lane for
  `bench_refactor_csc` on `nos4.mtx --repeat 1`, with methodology metadata
  and threshold-free interpretation.
- Selected performance rows have stronger repeat, warmup, variance, runner,
  and methodology semantics, but they remain bounded evidence rows.
- Static-first package support is the maintained product path; unsupported
  shared-library and dynamic ABI requests are rejected or guarded.
- Package-manager provider support is explicitly deferred and mechanically
  guarded; source install through Make/CMake remains the supported package
  path.
- `include/sparse_lu.h` has declaration-preserving public-header cleanup and
  aligned user documentation.
- Generated API HTML has a maintained local-only freshness path and local-only
  staging guard.
- The selected comparison family includes QR minimum-norm, QR compatible
  least-squares, partial-SVD diagonal top-k, and linked-list LU nonsymmetric
  square-solve rows.
- Linux and macOS hosted workflows provide selected comparison artifact
  freshness for the selected comparison targets.
- The iterative repeated-run handle APIs have deterministic allocation-failure
  cleanup evidence for selected CG, GMRES, and MINRES prepare/growth paths.

## Non-Claims

Epic 15 does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against SuiteSparse, PETSc,
  Trilinos, Eigen, SciPy, LAPACK, NumPy, or package-manager ecosystems;
- broad solver correctness beyond named fixtures, maintained corpus rows, and
  selected proof owners;
- portable performance superiority, backend superiority, OpenMP speedup proof,
  portable runtime proof, or state-of-the-art performance;
- broad benchmark publication or broad generated report freshness;
- broad allocation-failure cleanup coverage across all solvers, matrix
  construction, package/install flows, generated-report tooling, or allocation
  paths;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selector UX;
- package-manager provider availability;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broad Windows package or platform parity;
- Windows generated report freshness;
- selected oracle freshness on macOS;
- hosted publication of all generated reports;
- hosted generated API HTML publication;
- release evidence.

## Residual Queue

| Priority | Residual | Why it remains | Next-epic closure target |
| ---: | --- | --- | --- |
| 1 | Broader allocation-failure coverage | Sprint 176 proves one selected iterative handle family only. | Select one additional allocation-heavy subsystem and repeat the deterministic proof pattern with cleanup invariants and a focused gate. |
| 2 | Hosted generated API HTML | Sprint 173 chose local-only generated HTML. | Decide hosted URL/artifact/retention policy, then add publication and freshness evidence if hosted support is selected. |
| 3 | Package-manager provider support | Sprint 171 formally deferred providers. | Select exactly one provider and add recipe, provenance, install proof, cleanup proof, and support-tier wording. |
| 4 | Shared-library and dynamic ABI | Sprint 170 selected static-first-only continuation. | Reopen only with symbol visibility, export/import policy, SONAME/install-name/DLL metadata, ABI tests, and loader validation. |
| 5 | Windows report freshness | Sprint 175 did not promote Windows report generation/freshness. | Design Windows-safe report generation or record a stronger product deferral. |
| 6 | Selected oracle freshness beyond Linux | macOS gained selected comparison freshness, not selected oracle freshness. | Add a separate macOS oracle lane only if runtime and dependency constraints are acceptable. |
| 7 | Broad external comparison parity | Epic 15 added one LU fixture-local family on top of existing selected rows. | Add one bounded comparison family at a time with exact fixtures, tolerances, metrics, and non-parity wording. |
| 8 | Portable performance publication | Epic 15 published one selected Linux lane, not portable performance evidence. | Broaden only after a multi-platform fixture/methodology policy exists. |
| 9 | Public-header coherence breadth | Sprint 172 cleaned only `sparse_lu.h`. | Select the next high-risk public header family and run the declaration-preserving cleanup workflow. |
| 10 | Workflow target-list duplication | Selected comparison workflow inventories remain explicit and repetitive. | Factor selected target inventory before adding more hosted comparison targets. |

Long-horizon residuals remain broad state-of-the-art positioning, broad
external ecosystem parity, broad Windows package/platform parity, broad
generated report hosting, release packaging evidence, and package-provider
upgrade behavior.

## State-Of-The-Art Assessment

Epic 15 does not earn an unqualified state-of-the-art sparse linear algebra
claim.

The defensible assessment is narrower: Epic 15 improves the project's
evidence discipline, product honesty, selected hosted evidence, static-first
package confidence, API/adoption clarity, selected comparison breadth, and
failure-path proof for one subsystem. Those are meaningful engineering
improvements, but they are not comparative proof against mature sparse linear
algebra ecosystems.

A future state-of-the-art or broad external-parity claim would require named
libraries, versions, fixtures, metrics, tolerances, platforms, compilers,
package provenance, performance methodology, failure semantics, support-tier
boundaries, and reviewed hosted evidence for each selected claim.

## What Went Well

1. **The epic closed narrow gaps completely.** Performance publication,
   shared-library ABI posture, package-manager support, generated API HTML,
   comparison freshness, and allocation-failure evidence each ended with
   explicit evidence or explicit deferral.

2. **Claim wording stayed evidence-bound.** The work repeatedly separated
   hosted evidence, local generated evidence, source-controlled metadata,
   advisory rows, supplemental checks, and retained non-claims.

3. **Static-first product boundaries became clearer.** Shared-library,
   dynamic ABI, runtime-loader, static/shared selector, package-manager, and
   Windows package parity claims are now explicit product decisions or
   deferrals, not accidental implications.

4. **Comparison work remained fixture-local.** New selected comparison rows
   are useful because they name exact fixtures, metrics, tolerances, support
   tiers, and non-parity boundaries.

5. **The allocation-failure gap became actionable.** Sprint 176 added one
   deterministic proof, public cleanup invariant docs, a focused Make gate,
   and CTest labeling without claiming broad allocator reliability.

6. **Validation matched changed surfaces.** Source/header work ran full C
   quality gates, while documentation-only closeout days used docs hygiene and
   focused guards.

## Could Be Better

1. **Prompt/source mismatches added overhead.** Many sprint prompts pointed at
   stale Epic 12 sections, requiring repeated source-artifact notes.

2. **Claim governance remains distributed.** README, maintainer docs,
   report-index manifests, workflows, scripts, package docs, benchmark docs,
   and planning artifacts all carry pieces of the support-tier story.

3. **Hosted evidence remains selected-lane scoped.** Epic 15 improved selected
   hosted performance and comparison evidence, but broad generated report,
   generated API, performance, and oracle hosting remain limited or deferred.

4. **Windows still has important retained gaps.** Windows CMake-first package
   evidence exists from prior epics, but Windows report freshness, Makefile
   parity, and `pkg-config` execution parity remain non-claims.

5. **Allocation-failure proof is narrow by design.** It is valuable, but most
   direct solvers, eigensolvers, matrix construction paths, and generated
   tooling still lack deterministic allocation-failure proof.

## Key Deliverables

- [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- [SPRINT_167/RETROSPECTIVE.md](./SPRINT_167/RETROSPECTIVE.md)
- [SPRINT_168/RETROSPECTIVE.md](./SPRINT_168/RETROSPECTIVE.md)
- [SPRINT_169/RETROSPECTIVE.md](./SPRINT_169/RETROSPECTIVE.md)
- [SPRINT_170/RETROSPECTIVE.md](./SPRINT_170/RETROSPECTIVE.md)
- [SPRINT_171/RETROSPECTIVE.md](./SPRINT_171/RETROSPECTIVE.md)
- [SPRINT_172/RETROSPECTIVE.md](./SPRINT_172/RETROSPECTIVE.md)
- [SPRINT_173/RETROSPECTIVE.md](./SPRINT_173/RETROSPECTIVE.md)
- [SPRINT_174/RETROSPECTIVE.md](./SPRINT_174/RETROSPECTIVE.md)
- [SPRINT_175/RETROSPECTIVE.md](./SPRINT_175/RETROSPECTIVE.md)
- [SPRINT_176/artifacts/day5-harness-implementation.md](./SPRINT_176/artifacts/day5-harness-implementation.md)
- [SPRINT_176/artifacts/day7-regression-gate.md](./SPRINT_176/artifacts/day7-regression-gate.md)
- [SPRINT_176/artifacts/day8-invariant-docs.md](./SPRINT_176/artifacts/day8-invariant-docs.md)
- [SPRINT_176/artifacts/day10-claim-recalibration.md](./SPRINT_176/artifacts/day10-claim-recalibration.md)
- [SPRINT_176/artifacts/day12-integrated-validation.md](./SPRINT_176/artifacts/day12-integrated-validation.md)
- [SPRINT_176/artifacts/day13-retrospective-finalization.md](./SPRINT_176/artifacts/day13-retrospective-finalization.md)

## Completion

Epic 15 is complete with explicit residuals. The completed work improves
selected hosted evidence, performance methodology, static-first package and
ABI honesty, package-manager deferral clarity, public-header coherence,
generated API status, bounded comparison coverage, cross-platform selected
comparison freshness, and one allocation-failure proof. The remaining work is
visible, prioritized, and bounded by promotion gates rather than implied by
existing evidence.
