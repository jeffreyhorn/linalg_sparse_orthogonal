# Epic 13 Retrospective

**Epic:** 13 - Platform Closure, Maintained Corpus Expansion, Report
Freshness, Static-First ABI Boundaries & Adoption Coherence
**Sprints:** 147-156
**Status:** Complete

## Epic Objective

Epic 13 started from the Epic 12 closeout residual queue and selected a
bounded set of complete-gap closures instead of trying to widen every sparse
linear algebra claim at once. The epic focused on reviewed Windows CMake
coverage, Windows CMake install/downstream validation, maintained QR and
partial-SVD corpus-family expansion, selected generated report freshness,
static-first package and shared-library deferral proof, one narrow external
comparison study, tutorial/API adoption coherence, and final claim
recalibration.

The epic deliberately did not attempt an unqualified state-of-the-art sparse
linear algebra claim. The working standard was evidence-bound closure:
source-controlled artifacts, focused proof-owner tests, generated-local report
semantics where selected, hosted evidence where required, and explicit
non-claims for unsupported surfaces.

## Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 147 | Converted Epic 12 residuals into selected Epic 13 claim targets, evidence gates, quality surfaces, public claim freeze rules, and handoffs for Windows, corpus, report, ABI/package, comparison, adoption, and closeout work. |
| 148 | Ported and promoted the remaining Windows-staged CMake tests: `test_threads`, `test_sprint4_integration`, and `test_fuzz`; Windows CTest registration moved to `59` while non-CMake Windows parity remained explicit non-claims. |
| 149 | Promoted the Windows CMake install/downstream lane for the maintained static-first package surface, adding stronger installed CMake metadata, `sparse.pc` metadata, generated and maintained CMake consumers, exact-version proof, and mismatch rejection. |
| 150 | Expanded the maintained QR corpus family with bounded rank-deficient rectangular and underdetermined minimum-norm fixtures, expected rows, proof-owner tests, generated oracle/report rows, and documentation caveats. |
| 151 | Expanded the maintained partial-SVD corpus family with clustered/repeated, rank-deficient projector, sparse low-rank output, and fail-closed recovery fixtures, generated oracle/report rows, focused proof-owner tests, and documentation caveats. |
| 152 | Published selected generated report freshness for the combined QR and partial-SVD local oracle path, stabilized report-index normalization, and kept other generated families advisory or deferred. |
| 153 | Made the shared-library ABI product decision explicit: preserve and strengthen static-first package support while rejecting shared-library requests with exact blocker diagnostics and metadata guards. |
| 154 | Implemented the external comparison harness and first narrow QR minimum-norm study for `qr_underdetermined_minnorm_2x4` against a source-controlled dense reference helper, with normalized comparison freshness and explicit non-parity wording. |
| 155 | Simplified tutorial and adoption flow, added `docs/api_reference.md`, cleaned selected public-header comments with declaration-preservation proof, and documented generated API HTML freshness boundaries. |
| 156 | Reconciled final local validation, package proof, hosted master platform evidence, corpus/report evidence, comparison evidence, adoption/API surfaces, public claims, project-plan status, and the final residual queue. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Windows reviewed CMake coverage | Prior staged exclusions for threads, Sprint 4 integration, and fuzz/property coverage are now part of the reviewed Windows MSVC CMake subset. |
| Windows package confidence | Windows now has a reviewed CMake install/downstream validation lane for the maintained static-first package surface, while Makefile and `pkg-config` execution parity remain non-claims. |
| QR corpus | The maintained QR corpus expanded from a narrow prior fixture to a bounded family covering selected rank/nullity/nullspace and minimum-norm behavior. |
| Partial-SVD corpus | The maintained partial-SVD corpus expanded to selected clustered/repeated, rank-deficient projector, sparse-output, fail-closed, and recovery behavior. |
| Report freshness | The selected combined QR plus partial-SVD oracle freshness path is executable and normalized; non-selected generated families remain advisory or deferred. |
| Static-first package/ABI | Static package support is stronger and shared-library support is rejected with exact product blockers instead of vague unsupported behavior. |
| Comparison | The project has one maintained local comparison lane and one source-controlled comparison study snapshot for a selected QR fixture. |
| Adoption/API | First-use docs route through a clearer build, example, cookbook, solver-selection, install, benchmark/report, and API-reference path. |
| Claim governance | Public claims are evidence-bound, and unsupported state-of-the-art, parity, performance, package-manager, shared-library, dynamic ABI, loader, and broad Windows claims remain explicit non-claims. |

## Validation Evidence

| Evidence | Result | Boundary |
| --- | --- | --- |
| Sprint 148 full C quality gate | Passed: `make format && make lint && make test`. | Required because `.c` and `.h` portability files changed. |
| Sprint 148 Windows CMake surface | Locally enumerated and focused-promoted targets passed; hosted proof was pending until PR CI. | Supports promoted test registration after hosted CI, not Windows Makefile or package parity. |
| Sprint 149 local package checks | Passed local `tests/test_cmake_install.sh`, `tests/test_install.sh`, and static deferral guard. | Local static package proof; Windows hosted proof depends on workflow run. |
| Sprint 150 QR proof-owner tests | Passed focused QR corpus proof and integrated full C gate. | Fixture-local QR family evidence, not broad QR correctness. |
| Sprint 151 partial-SVD proof-owner tests | Passed focused partial-SVD corpus proof and integrated full C gate. | Fixture-local partial-SVD evidence, not broad SVD parity. |
| Sprint 152 selected oracle freshness | Passed `make report-index-oracle-freshness` with `52` generated rows and required freshness. | Local-only selected oracle evidence. |
| Sprint 153 package/ABI validation | Passed static package deferral, install, report-index, docs, and stale wording checks. | Static-first support and shared-library rejection proof only. |
| Sprint 154 comparison freshness | Passed `make report-index-comparison-freshness` and harness self-check. | One local QR minimum-norm comparison fixture only. |
| Sprint 155 full C quality gate | Passed `make format && make lint && make test`; declaration-preservation normalized diffs were `0` bytes. | Selected public-header comments only, not API redesign. |
| Sprint 156 Day 4 local baseline | Passed `make quality-review-full`; CMake and Makefile both registered `59` tests and CTest passed `59/59`. | Local macOS evidence only. |
| Sprint 156 Day 5 package validation | Passed static deferral, Make install/`pkg-config` with `23` checks, CMake install/export with `27` checks, package report-index checks, and runtime-backend freshness check. | Local static-first package proof. |
| Sprint 156 Day 6 hosted master platform reconciliation | Latest inspected master merge baseline for PR #172 passed Linux CI, macOS CI, and Windows CI on commit `c7981c6ef7fa887e87575279009113b1dcf3a630`. | Master hosted baseline, not branch-specific Sprint 156 PR proof. |
| Sprint 156 Day 7 corpus/report validation | Schema passed; QR corpus passed `14` tests; partial-SVD corpus passed `10` tests; selected oracle freshness passed. | Local-only fixture/report evidence. |
| Sprint 156 Day 8 comparison reconciliation | Harness self-check and selected comparison freshness passed; all six generated comparison rows passed. | One local-only comparison fixture. |
| Sprint 156 Day 9 adoption/API reconciliation | Link targets and declaration-preservation evidence passed; generated HTML residuals recorded. | Docs/API surface coherence, not generated HTML completeness. |
| Sprint 156 Day 10 claim audit | Public claim scans found evidence-bound wording or explicit non-claims only. | Claim audit proof, not new capability proof. |
| Sprint 156 Day 11 residual queue | Published `18` final residuals with owners, blockers, prerequisites, and promotion gates. | Planning/closeout artifact. |
| Sprint 156 Day 13 project-plan reconciliation | Project-plan status matched real artifacts, with narrowed/deferred work mapped to the residual queue. | Planning/closeout artifact. |
| Sprint 156 Day 14 docs-only checks | `git diff --check` and final claim scan passed. | Documentation hygiene only; no code gate was required. |

Sprint 156 changed documentation and planning artifacts only. No new `.c` or
public `.h` edits were made during Sprint 156, so the final full C quality
gate was not required for Day 14.

## Earned Claims

Epic 13 earns these claims with qualifiers:

- Windows reviewed MSVC CMake coverage now includes the formerly staged
  `test_threads`, `test_sprint4_integration`, and `test_fuzz` targets.
- Windows has reviewed CMake install/downstream validation for the maintained
  static-first package surface, including installed static `.lib`, headers,
  CMake package metadata, `sparse.pc` metadata, downstream CMake consumers,
  exact-version behavior, mismatch rejection, and no shared/DLL metadata.
- The maintained QR corpus includes selected fixture-local rank-deficient
  rectangular and underdetermined minimum-norm evidence with focused proof
  ownership and generated local oracle/report rows.
- The maintained partial-SVD corpus includes selected fixture-local
  clustered/repeated, rank-deficient projector, sparse-output, fail-closed, and
  recovery evidence with focused proof ownership and generated local
  oracle/report rows.
- The selected local oracle freshness gate covers the combined QR plus
  partial-SVD generated report family and fails closed on missing or stale
  required selected evidence.
- The maintained package posture remains static-first and is backed by local
  Make install, `pkg-config`, CMake install/export, exact-version,
  mismatch-version, downstream consumer, static metadata, and shared deferral
  checks.
- Shared-library support is intentionally deferred with exact blocker
  diagnostics for export/import policy, symbol visibility, dynamic ABI policy,
  platform loader metadata, installed shared consumers, and runtime-loader
  validation.
- The project has one maintained local comparison lane for
  `qr_underdetermined_minnorm_2x4` against a source-controlled dense reference
  helper.
- The adoption surface is clearer across README, tutorial, examples, cookbook,
  solver selection, install docs, benchmark/report docs, API reference, and
  maintainer evidence guidance.
- Selected public-header comment cleanup preserved declarations according to
  normalized before/after/current declaration diffs.
- Public claim wording audited during Sprint 156 maps to evidence or remains
  an explicit non-claim.

## Non-Claims

Epic 13 does not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library or ecosystem parity against LAPACK, NumPy, SciPy,
  SuiteSparse, Eigen, PETSc, Trilinos, or package-manager ecosystems;
- broad QR, SVD, or partial-SVD correctness beyond selected fixture-local
  evidence;
- raw QR basis identity, QR sign/orientation/order parity, raw singular-vector
  identity, phase, or basis-order parity;
- broad rank-threshold policy, broad rank-deficient least-squares,
  minimum-norm, nullspace, economy-mode, sparse-mode, reorder, or COLAMD QR
  behavior;
- broad partial-SVD repeated-spectrum, sparse-output/drop-tolerance
  optimality, convergence-rate, partial-result, pseudoinverse, or
  minimum-norm behavior;
- portable performance, backend superiority, portable runtime, or portable
  iteration-count guarantees;
- generated report pass evidence from source-controlled rows alone;
- hosted CI proof for local-only corpus, oracle, and comparison rows until a
  reviewed hosted lane is added;
- generated API HTML completeness or freshness beyond the documented
  source-header and `make docs` boundary;
- shared-library support, dynamic ABI compatibility, runtime-loader behavior,
  static/shared package selectors, or package-manager distribution;
- Windows Makefile parity, Windows `pkg-config` execution parity, Windows
  package-manager support, or broad Windows platform parity;
- branch-specific Sprint 156 hosted PR success until PR workflows run and are
  reconciled.

## Residuals And Next-Epic Handoff

The final Epic 13 residual queue is published in
[SPRINT_156/artifacts/day11-residual-queue-publication.md](./SPRINT_156/artifacts/day11-residual-queue-publication.md).

Highest-priority next-epic candidates:

| Priority | Candidate | Closure target |
| --- | --- | --- |
| 1 | Generated API HTML refresh | Run `make docs`, triage warnings, close or explicitly document missing generated pages, and preserve source-header-first API wording. |
| 2 | Hosted promotion for selected local-only oracle/comparison rows | Move already-selected freshness gates into reviewed hosted evidence without broadening solver semantics. |
| 3 | One bounded QR comparison expansion | Extend the existing comparison harness with one controlled fixture family, metrics, tolerances, provenance, and freshness gate. |
| 4 | One bounded partial-SVD comparison publication | Add subspace-safe comparison rows and freshness for a selected partial-SVD fixture family. |
| 5 | Windows package parity decision | Decide and either implement or explicitly retain non-claims for Windows Makefile and/or Windows `pkg-config` parity. |
| 6 | Next public-header cleanup batch | Continue declaration-preserving documentation cleanup with the Sprint 155 guardrails. |

Long-horizon deferrals:

- package-manager distribution;
- shared-library product support;
- dynamic ABI compatibility policy;
- broad ecosystem parity;
- portable performance superiority;
- broad state-of-the-art positioning;
- typed runtime/backend API promotion unless tied to selected API and ABI
  scope.

## State-Of-The-Art Assessment

Epic 13 did not earn an unqualified state-of-the-art claim.

The strongest defensible assessment is narrower: Epic 13 improved the
project's platform evidence, Windows CMake coverage, static-first package
confidence, QR and partial-SVD fixture-local corpus breadth, report freshness
discipline, comparison infrastructure, adoption clarity, and claim hygiene.
Those changes improve engineering maturity and user trust, but they are not a
direct comparative proof against mature sparse linear algebra ecosystems.

A future state-of-the-art or broad external-parity claim would require direct
comparative evidence naming libraries, versions, fixtures, metrics,
tolerances, platforms, compilers, package provenance, performance methodology,
failure semantics, and caveats.

## What Went Well

1. **Windows staged coverage closed cleanly.** Sprint 148 selected exact tests,
   ported them, updated CMake/CI registration, and kept all broader Windows
   package and platform claims out of scope.

2. **Windows install confidence widened without overclaiming.** Sprint 149
   turned the CMake install/downstream lane into reviewed package evidence
   while preserving Makefile and `pkg-config` non-claims.

3. **Corpus expansion stayed fixture-local.** Sprints 150 and 151 added
   meaningful QR and partial-SVD families without converting them into broad
   numerical correctness, external parity, platform, package, ABI, or
   performance claims.

4. **Generated report freshness became more executable.** Sprint 152 provided
   a selected local freshness target for combined QR plus partial-SVD oracle
   rows while keeping other generated families advisory.

5. **Static-first package posture became more honest.** Sprint 153 rejected
   shared-library requests with exact blockers, making the absence of dynamic
   support explicit and test-backed.

6. **The first comparison lane is real and bounded.** Sprint 154 delivered a
   reproducible comparison harness and one narrow study without claiming
   NumPy/SciPy/LAPACK or ecosystem parity.

7. **Adoption docs now have a coherent front door.** Sprint 155 improved the
   tutorial/API path and public-header comments while preserving declarations.

8. **Sprint 156 kept closeout disciplined.** Final validation, platform
   reconciliation, claim audit, project-plan reconciliation, and residual
   publication did not inflate support claims beyond the evidence.

## What Did Not Go Well

1. **Hosted branch evidence remains PR-time.** Sprint 156 can cite latest green
   master hosted baselines, but the current branch still needs PR workflow
   reconciliation before final hosted branch claims.

2. **Generated artifacts remain split across source-controlled policy and
   ignored local output.** This is intentional, but it creates recurring
   interpretation work for maintainers and reviewers.

3. **API HTML refresh was deferred.** Sprint 155 documented the generated API
   reference boundary, but users still do not have complete generated pages
   for all checked-in public headers.

4. **Comparison breadth is still small.** The first comparison study is useful
   infrastructure proof, not meaningful ecosystem coverage.

5. **Shared-library, dynamic ABI, and package-manager support remain product
   decisions.** Static-first proof is stronger, but dynamic-linking and
   packaging maturity remain substantial future work.

6. **Broad numerical claims remain far away.** QR and partial-SVD fixture
   families improved, but broad rank, basis, convergence, sparse-output, and
   ecosystem parity claims still require a much larger evidence program.

## Final Handoff

Day 14 leaves Epic 13 ready for review with:

- a complete Sprint 156 artifact package;
- a final Epic 13 retrospective;
- a reconciled Epic 13 project-plan status;
- a final residual queue with owner roles, blockers, prerequisites, and
  promotion gates;
- explicit support boundaries for platform, package, report, corpus,
  comparison, adoption, ABI, performance, and state-of-the-art claims.

Future work should start from the Day 11 residual queue and prefer bounded
complete-gap closure over broad partial progress.
