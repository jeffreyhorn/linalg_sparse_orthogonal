# Sprint 137 Day 5 - Residual Owner & Non-Goal Map

## Purpose

Day 5 assigns the Day 4 residual intake queue to Epic 12 owner workstreams,
defines dependency order, records explicit non-goals, and establishes
promotion gates for any residual that may become a future claim.

This artifact does not select final Epic 12 implementation scope. Day 6 will
define the scoring rubric, and Day 7 will choose the specific gaps for Sprints
138-146. Day 5 only makes ownership and claim gates explicit.

## Owner Workstreams

| Owner workstream | Responsibility |
| --- | --- |
| Corpus/oracle owner | Maintained fixture taxonomy, deterministic generated matrices, optional-data policy, expected-result metadata, oracle row semantics, and corpus skip behavior. |
| QR owner | Selected QR residual closure, QR fixtures, QR oracle semantics, QR proof-owner placement, and QR public/maintainer non-claims. |
| Partial-SVD owner | Selected partial-SVD residual closure, edge-case fixtures, convergence-budget semantics, vector/subspace comparison policy, and SVD non-claims. |
| Report-index owner | Cross-report metadata normalization, freshness fields, stale-report checks, report-family row-meaning preservation, and report non-claims. |
| Runtime/backend owner | Backend/runtime precedence, sentinel fixture choices, local metric semantics, variance policy, optional backend availability semantics, and performance non-claims. |
| Package/ABI owner | Static-first optional modes, shared-library product decision, ABI policy, package metadata, downstream consumer proof, and package non-claims. |
| Platform owner | macOS/Windows promotion decisions, hosted-runner history, Windows staged test portability, CTest count changes, and platform support-tier wording. |
| Adoption/docs owner | README, INSTALL, cookbook, tutorial, solver-selection, algorithm reference/history, docs link checks, and first-use simplification after evidence lands. |
| Maintainability owner | Focused helper/test/source movement needed to support selected gap closure without broad giant-test decomposition. |
| Closeout owner | Final validation, claim recalibration, residual publication, retrospective, and next-epic handoff. |

## Residual Owner Map

| Active residual | Primary owner | Supporting owners | Dependency order | Promotion gate |
| --- | --- | --- | --- | --- |
| Numerical corpus index | Corpus/oracle owner | Report-index owner, QR owner, partial-SVD owner | Must precede QR/partial-SVD corpus promotion and report normalization. | Fixture taxonomy, row-level metadata, skip policy, expected-result semantics, support tier, validation command, and docs interpretation exist. |
| SuiteSparse rank-deficient QR corpus evidence | Corpus/oracle owner | QR owner, Report-index owner | Depends on corpus row semantics; may feed selected QR residual. | Independent expected-rank metadata, provenance, optional-data behavior, runtime budget, support tier, and reviewed validation are present. |
| SuiteSparse and optional-large minimum-norm expansion | Corpus/oracle owner | QR owner, Partial-SVD owner, Report-index owner | Depends on corpus row semantics and QR/SVD comparison semantics. | Extraction rules, RHS policy, rank/nullity, residual/norm metrics, skip behavior, support tier, and claim boundary are present. |
| External-reference helper generated index | Corpus/oracle owner | Report-index owner, solver-family owners | Depends on corpus/oracle row model; informs report normalization. | Helper-specific output class, fixture key, tolerance policy, skip behavior, assertion class, and non-parity wording are preserved. |
| QR compatible zero-residual lane | QR owner | Corpus/oracle owner, Maintainability owner | Depends on QR selection criteria; may depend on corpus fixture model. | Distinct proof value, named fixture, exact residual expectation, solution-selection note, and no-minimum-norm boundary exist. |
| QR wide residual-only behavior | QR owner | Corpus/oracle owner, Maintainability owner | Depends on QR selection criteria and output semantics. | Rank/nullity, residual metric, raw-Q/economy boundary, and no-minimum-norm/no-basis-parity wording exist. |
| QR rank-deficient nullspace/subspace expansion | QR owner | Corpus/oracle owner, Report-index owner | Depends on corpus metadata and QR comparison policy. | Projector or two-way projection metric, sign/orientation policy, rank/nullity metadata, fixture-local tolerance, support tier, and docs non-claims exist. |
| Additional QR-vs-SVD minimum-norm cross-checks | QR owner | Partial-SVD owner, Corpus/oracle owner | Depends on QR and SVD comparison semantics. | Fixture keys, QR residual/norm metrics, SVD tolerance, and explicit consistency-not-global-oracle wording exist. |
| Partial-SVD repeated/clustered spectra | Partial-SVD owner | Corpus/oracle owner, Report-index owner | Depends on partial-SVD scoring and fixture/gap policy. | Gap/order/tolerance policy, deterministic fixture, convergence behavior, validation, and vector/subspace non-claims exist. |
| Partial-SVD rank-deficient subspace expansion | Partial-SVD owner | Corpus/oracle owner, Maintainability owner | Depends on vector/subspace comparison policy. | Range/subspace metrics, rank/nullity metadata, orientation/sign policy, fixture-local tolerances, and docs boundaries exist. |
| Partial-SVD low-rank optimality expansion | Partial-SVD owner | Corpus/oracle owner | Depends on selecting one claim class rather than many. | One claim class is selected, metric is defined, fixtures and validation pass, and sparse-output/drop-tolerance/corpus non-claims remain fenced. |
| Partial-SVD convergence reporting semantics | Partial-SVD owner | Runtime/backend owner, Report-index owner | Depends on convergence-budget definition and report row semantics. | Achieved tolerance, iteration count, non-convergence, stagnation, partial-result, runtime budget, and docs semantics are implemented. |
| Cross-report normalized index | Report-index owner | Corpus/oracle owner, Runtime/backend owner, Package/ABI owner | Depends on report-family inventory and corpus/oracle row model. | Shared schema preserves row meaning, freshness, support tier, failure class, claim boundary, and skip/defer reason without flattening report families. |
| Automated stale-report scanner | Report-index owner | Corpus/oracle owner, Runtime/backend owner | Depends on common metadata contract. | Scanner can detect stale commit/branch/time/support-tier context for selected maintained reports and reports failures without implying release proof. |
| Runtime/backend sentinel expansion | Runtime/backend owner | Report-index owner, solver-family owners | Depends on runtime precedence and report row semantics. | Fixture, metric, tolerance, runtime budget, variance policy, backend-state semantics, support tier, and performance non-claims are defined. |
| LDLT report-only sentinel | Runtime/backend owner | direct-solver owner, Report-index owner | Depends on sentinel selection criteria. | KKT fixture, backend CSV field semantics, threshold-free status, support tier, and no-hard-timing wording exist. |
| Iterative convergence or BiCGSTAB sentinel | Runtime/backend owner | iterative owner, Report-index owner | Depends on sentinel selection criteria and variance policy. | One bounded fixture, metric, tolerance, runtime budget, variance policy, and local-only claim boundary exist. |
| Eigensolver backend/preconditioner sentinel | Runtime/backend owner | eigensolver owner, Report-index owner | Depends on sentinel selection criteria and backend-state vocabulary. | Narrow backend/preconditioner slice, fixture, metric, support tier, backend fallback semantics, and no-backend-parity wording exist. |
| SVD/bidiag report rows | Runtime/backend owner | Partial-SVD owner, Report-index owner | Depends on selected partial-SVD scope. | Fixture and metric semantics exist and rows are explicitly local report evidence only. |
| Optional backend availability rows | Runtime/backend owner | Package/ABI owner, Report-index owner | Depends on optional-backend semantics and report schema. | Unsupported/unavailable/probed/fallback meanings, non-portability policy, and docs boundaries exist. |
| Static-first optional package mode matrix | Package/ABI owner | Runtime/backend owner, Platform owner | Depends on package/ABI decision criteria. | `SPARSE_MUTEX` and/or `SPARSE_OPENMP` install/downstream proof, package metadata, support tier, and default-package boundary exist. |
| Shared-library packaging | Package/ABI owner | Platform owner, Adoption/docs owner | Depends on Day 7 product decision. | Build rules, artifact naming, export/import policy, install/export metadata, downstream consumers, platform proof, and docs exist. |
| Dynamic ABI compatibility | Package/ABI owner | public-header owners, Platform owner | Depends on shared-library product decision. | ABI epoch, symbol inventory, public layout policy, export macros, soname/install-name policy, compatibility tests, and docs exist. |
| Package-manager support | Package/ABI owner | Platform owner, Adoption/docs owner | Depends on ABI/release mechanics unless explicitly narrowed. | Manager recipes, dependency metadata, install roots, upgrade/uninstall proof, and downstream tests exist. |
| macOS reviewed install/export parity | Platform owner | Package/ABI owner | Depends on package decision and hosted-runner evidence. | Promotion decision, hosted-runner history, runtime budget, failure triage ownership, CI proof, and reviewed-platform wording exist. |
| Windows reviewed install-validation parity | Platform owner | Package/ABI owner | Depends on package decision and hosted-runner evidence. | Exact CMake-first scope, hosted-runner history, failure triage ownership, CI proof, and reviewed-platform wording exist. |
| Windows staged pthread/POSIX test promotion | Platform owner | Maintainability owner, test owners | Depends on source portability work. | Windows-native equivalents or portability wrappers, expected CTest count update, MSVC configure/build/execute proof, and support-tier docs exist. |
| Documentation-link automation | Adoption/docs owner | Maintainability owner | Optional after docs volume and Day 12 claim freeze. | Maintained target is deterministic, scoped, documented, and does not alter product claims. |
| Algorithm reference continued slimming | Adoption/docs owner | solver-family owners | Should follow evidence/package decisions. | Current-reference content remains accurate, historical material moves safely, links pass, and claim boundaries remain unchanged. |
| Cookbook and adoption navigation maintenance | Adoption/docs owner | package, runtime, solver owners | Should follow selected implementation changes. | New examples/workflows are earned, support tiers are correct, and navigation remains aligned. |
| Generic QR/SVD/minimum-norm helper consolidation | Maintainability owner | QR owner, Partial-SVD owner | Only if selected QR/partial-SVD work needs it. | Behavior-specific helpers move with tolerance preservation, focused tests, and no new numerical claims. |

## Dependency Order

1. **Baseline evidence:** Days 2-3 source/test/build/package/CI/report metrics.
2. **Residual intake:** Day 4 candidate, duplicate, optional, closed, and
   non-claim classification.
3. **Owner assignment:** Day 5 primary/supporting owners, dependencies,
   promotion gates, non-goals, and stop conditions.
4. **Selection criteria:** Day 6 complete-closure rubric and anti-goals.
5. **Gap decision:** Day 7 selected Sprint 138-146 targets.
6. **Evidence templates:** Days 8-10 corpus/oracle, report/freshness,
   package/ABI, platform, and claim templates.
7. **Quality and claim gates:** Days 11-12 validation map and public claim
   freeze.
8. **Implementation handoff:** Days 13-14 Sprint 138 readiness, residuals, and
   closeout.

## Epic 12 Non-Goal Register

| Non-goal | Reason |
| --- | --- |
| Unqualified state-of-the-art claim | Requires broad external comparison, reproducible performance, package/ABI/platform proof, and ecosystem evidence beyond Sprint 137 planning. |
| Broad SuiteSparse, LAPACK, NumPy, SciPy, PETSc, Trilinos, Eigen, ARPACK, GraphBLAS, oneMKL, or vendor parity | Current evidence is fixture-local and support-tier bounded. |
| GPU or distributed-memory support | Outside Epic 12 selected high-value closure candidates and not supported by current architecture/proof. |
| Full package-manager distribution across ecosystems | Depends on ABI/release mechanics and manager-specific proof; likely deferred unless Day 7 explicitly narrows scope. |
| Portable performance superiority | Current benchmark/sentinel evidence is local and threshold-bounded only where explicitly documented. |
| Full decomposition of every giant test/source file | Epic 12 should split only where needed for selected gap closure. |
| Reviewed macOS/Windows parity by wording alone | Platform promotion requires hosted CI proof, failure semantics, support-tier docs, and source portability where applicable. |
| Coverage percentage as behavioral completeness | Coverage remains supplemental and tree-mutating. |
| Dead-code report as removal-ready proof | Dead-code output remains triage/report-completeness evidence until API owner review exists. |
| Generated report index as broad correctness or release proof | Report indexes provide row interpretation and freshness context only. |

## Promotion Gate Table

| Claim family | Minimum promotion gate |
| --- | --- |
| Corpus/oracle claim | Corpus row schema, fixture provenance, expected-result semantics, skip/defer policy, support tier, validation command, and docs interpretation. |
| QR claim | Named fixture, residual/output semantics, tolerance/rank/nullity metadata where relevant, focused tests, oracle or consistency boundary, and public/maintainer non-claims. |
| Partial-SVD claim | Selected edge class, singular-value/vector/subspace comparison policy, convergence-budget semantics where relevant, focused tests, report/docs interpretation, and external-parity non-claims. |
| Report/freshness claim | Shared metadata contract, row-meaning preservation, support tier, freshness fields, stale detection semantics, and report-family non-claims. |
| Runtime/backend claim | Typed or documented runtime precedence, fixture and metric semantics, variance/runtime budget, backend-state interpretation, focused validation, and portable-performance non-claims. |
| Package/ABI claim | Product decision, build/install/export metadata, downstream consumer proof, unsupported artifact checks, platform proof, and README/INSTALL/maintainer alignment. |
| Platform claim | Hosted CI evidence, expected test/package counts, failure semantics, support-tier wording, and no supplemental-to-reviewed promotion by implication. |
| Adoption claim | Earned implementation or proof behind each workflow, examples/docs alignment, link validation, and no unsupported package/platform/performance wording. |

## Stop Conditions

- A residual requires broad ecosystem parity rather than a bounded fixture,
  report, package, or platform proof.
- A candidate cannot be assigned one primary owner workstream.
- A promotion gate lacks validation, documentation, or support-tier evidence.
- A generated report schema would flatten row meanings across families.
- A package/ABI proposal would imply shared-library or dynamic ABI support
  without build rules, symbol policy, loader behavior, and downstream proof.
- A platform promotion relies on local or supplemental evidence without hosted
  CI proof.
- A QR or partial-SVD claim depends on raw basis equality, sign/orientation, or
  global external-library parity.
- A docs-only wording change would widen support without implementation and
  validation.

## Day 5 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Each active residual has one owner workstream. | Complete | Residual owner map assigns every active candidate one primary owner plus supporting owners. |
| Non-goals are explicit rather than hidden. | Complete | Epic 12 non-goal register lists broad claims and unsupported surfaces that remain out of scope. |
| Promotion gates require implementation, validation, and documentation proof. | Complete | Promotion gate table and stop conditions require evidence, validation, support-tier wording, and docs before claims widen. |

