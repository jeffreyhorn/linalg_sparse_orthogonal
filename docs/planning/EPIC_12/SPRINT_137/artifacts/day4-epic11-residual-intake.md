# Sprint 137 Day 4 - Epic 11 Residual Intake

## Purpose

Day 4 converts Epic 11 residuals into an Epic 12 intake queue before owners,
dependencies, non-goals, and gap-selection decisions are assigned on Days 5-7.

This artifact considers the Epic 11 retrospective, Sprint 136 residual queue
publication, and recent Sprint 130-136 retrospectives. It does not select final
Epic 12 scope. It classifies residuals as candidates, duplicates/consolidated
items, already-covered context, optional-local work, or explicit non-claims so
Day 5 can assign owners without reopening closed Epic 11 work.

## Intake Classification

| Classification | Meaning |
| --- | --- |
| Candidate | Active Epic 12 intake item that may receive owner assignment and gap-selection scoring. |
| Duplicate / consolidated | Covered by a broader candidate row; retain detail as evidence, but do not count as separate scope. |
| Already covered | Useful context from Sprint 130-136, but the relevant product decision or documentation outcome already landed in Epic 11. |
| Optional-local | Useful maintainer cleanup that does not change support claims by itself. |
| Explicit non-claim | Must remain absent from public/support wording unless future implementation, validation, and docs earn it. |

## Intake Sources

| Source | Residual value |
| --- | --- |
| `docs/planning/EPIC_11/EPIC_11_RETROSPECTIVE.md` | Provides final future-epic candidate categories and high-level non-claims. |
| `docs/planning/EPIC_11/SPRINT_136/artifacts/day12-residual-queue-publication.md` | Provides the canonical post-Epic-11 residual queue, QR residual subqueue, owner surfaces, promotion criteria, and claim boundaries. |
| `docs/planning/EPIC_11/SPRINT_130/RETROSPECTIVE.md` | Adds partial-SVD detail around repeated/clustered spectra, rank-deficient owners, low-rank optimality, SuiteSparse corpus metadata, and convergence reporting. |
| `docs/planning/EPIC_11/SPRINT_131/RETROSPECTIVE.md` | Adds corpus/report detail around row metadata, coverage index, dead-code freshness, stale-report scanner, and external-helper index boundaries. |
| `docs/planning/EPIC_11/SPRINT_132/RETROSPECTIVE.md` | Adds runtime/backend detail around canonical metadata, direct backend fields, LDLT sentinel candidates, iterative/eigensolver/SVD sentinel candidates, and optional backend rows. |
| `docs/planning/EPIC_11/SPRINT_133/RETROSPECTIVE.md` | Adds package/ABI detail around shared-library design, ABI policy, package-manager recipes, static/shared selectors, `Libs.private`, optional package modes, and install proof promotion. |
| `docs/planning/EPIC_11/SPRINT_134/RETROSPECTIVE.md` | Adds platform detail around hosted-runner history, macOS/Windows install promotion, Windows staged test portability, and CTest count changes. |
| `docs/planning/EPIC_11/SPRINT_135/RETROSPECTIVE.md` | Adds adoption detail around algorithm-reference slimming, docs link-check target, cookbook maintenance, and navigation alignment. |
| `docs/planning/EPIC_11/SPRINT_136/RETROSPECTIVE.md` | Confirms the consolidated closeout residual set and final non-claim boundaries. |

## Epic 12 Intake Queue

| Intake item | Source detail | Classification | Initial Epic 12 group | Day 5 owner-assignment note |
| --- | --- | --- | --- | --- |
| QR compatible zero-residual lane | Separate proof value, output semantics, solution-selection rules, residual metric, and no-minimum-norm interpretation are missing. | Candidate | QR residual closure | Assign to QR owner; likely compare against selected Sprint 139 QR priority scope. |
| QR wide residual-only behavior | Needs output semantics, rank/nullity, raw-Q/economy boundaries, residual-only proof value, and no-minimum-norm boundary. | Candidate | QR residual closure | Assign to QR owner; score with compatible zero-residual and nullspace work. |
| QR rank-deficient nullspace/subspace expansion | Needs rank/nullity metadata, projector/two-way projection metrics, sign/orientation policy, support tier, and fixture-local tolerances. | Candidate | QR residual closure | Assign to QR owner; likely high state-of-the-art relevance if corpus metadata lands first. |
| Near-threshold QR rank behavior | Needs threshold family, perturbation scale, tolerance, and rank-model policy. | Duplicate / consolidated | QR residual closure | Keep as subcase under QR rank-deficient selection rather than standalone scope unless Day 7 selects it. |
| SuiteSparse rank-deficient QR corpus evidence | Needs independent expected-rank metadata, optional-data behavior, support tiers, provenance, runtime budget, and reviewed validation. | Candidate | Corpus/oracle plus QR residual closure | Assign jointly to corpus and QR owners; cannot be promoted before corpus row semantics exist. |
| SuiteSparse and optional-large minimum-norm expansion | Needs extraction rules, RHS policy, rank/nullity, residual/norm metrics, skip behavior, optional-data policy, and support tier per row. | Candidate | Corpus/oracle plus QR/SVD helper boundary | Assign to corpus and QR/SVD helper owners; likely dependent on Sprint 138 corpus decisions. |
| Additional QR-vs-SVD minimum-norm cross-checks | Needs fixture keys, SVD tolerance, QR residual/norm metrics, and explicit non-oracle boundary. | Candidate | QR residual closure plus partial-SVD/SVD semantics | Assign as consistency evidence only; block global-oracle interpretation. |
| Generic QR/SVD/minimum-norm helper consolidation | Move behavior-specific helpers only with tolerance preservation and focused validation. | Optional-local | Maintainability | Keep as enabling work only if selected QR/partial-SVD scope needs it. |
| Partial-SVD repeated/clustered spectra | Sprint 130 left repeated leading-block and clustered-spectrum policy unresolved. | Candidate | Partial-SVD residual closure | Assign to partial-SVD owner; score high only if fixture/gap/tolerance semantics can be closed. |
| Partial-SVD rank-deficient subspace expansion | Range-only evidence exists; null-space/subspace, duplicate-column, zero-crossing, and pseudoinverse owners remain split. | Candidate | Partial-SVD residual closure | Assign to partial-SVD owner with explicit vector/subspace non-claim boundaries. |
| Partial-SVD low-rank optimality expansion | Frobenius lane exists, but spectral norm, sparse output, drop tolerance, and corpus claims remain unearned. | Candidate | Partial-SVD residual closure | Assign to partial-SVD owner; likely defer broad low-rank classes unless one can close fully. |
| Partial-SVD convergence reporting semantics | Needs achieved tolerance, iteration count, convergence rate, stagnation handling, and partial-result semantics. | Candidate | Partial-SVD residual closure | Assign to partial-SVD owner; dependency for any convergence-budget claim. |
| Numerical corpus index | Needs row-level SuiteSparse/integration/product-observed/expected-error/oracle/tolerance/runtime/support metadata. | Candidate | Corpus/oracle architecture | Assign to corpus owner; likely Sprint 138 center of gravity. |
| External-reference helper generated index | Needs helper-specific output classes, fixture keys, skip behavior, tolerance policy, and assertion class. | Candidate | Corpus/oracle architecture plus report index | Assign to external-oracle/report owners; avoid parity overclaim. |
| Cross-report normalized index | Needs shared schema preserving row meaning, freshness, support tier, failure class, and claim boundary. | Candidate | Report normalization/freshness | Assign to report-index owner; likely Sprint 141 center of gravity. |
| Coverage index and coverage gap follow-through | Needs backend, threshold, tree-mutating behavior, source filters, freshness, reset policy, support tier, and owner labels. | Optional-local | Report normalization/freshness | Keep optional unless selected as part of normalized report scope. |
| Dead-code freshness and public-surface review | Needs freshness metadata and API owner review before public-surface interpretation or removal. | Optional-local | Report normalization/freshness | Keep optional unless stale-report schema covers it cheaply. |
| Automated stale-report scanner | Needs common metadata across report families before detecting stale branch, commit, generated time, support tier, and row meaning. | Candidate | Report normalization/freshness | Assign to report-index owner after normalization prerequisites. |
| Runtime/backend sentinel expansion | Needs fixtures, metrics, tolerances, runtime budget, variance policy, backend-state semantics, and claim gates. | Candidate | Runtime/backend governance | Assign to runtime/benchmark owner; likely Sprint 142 target. |
| Canonical `support_tier` and `claim_boundary` generation | Sprint 132 left open whether canonical rows should generate these fields or remain documentation-backed. | Duplicate / consolidated | Report normalization plus runtime/backend governance | Fold into report normalization and sentinel expansion decisions. |
| Direct backend fields in canonical reports | Backend fields remain inside benchmark CSVs rather than a companion report index. | Duplicate / consolidated | Report normalization plus runtime/backend governance | Fold into normalized report index decision. |
| LDLT report-only sentinel | Candidate using existing KKT backend CSV fields without hard timing thresholds. | Candidate | Runtime/backend governance | Assign to runtime/benchmark owner; score against iterative/eigensolver/SVD sentinel candidates. |
| Iterative convergence or BiCGSTAB sentinel | Needs one bounded fixture, metric, tolerance, runtime budget, and variance policy. | Candidate | Runtime/backend governance | Assign to runtime/iterative owner; likely higher user value than broad sentinel expansion. |
| Eigensolver backend/preconditioner sentinel | Needs narrow backend/preconditioner slice before adding a report lane. | Candidate | Runtime/backend governance | Assign to eigensolver/runtime owner; preserve no-backend-parity claim. |
| SVD/bidiag report rows | Needs fixture and metric semantics before local report rows. | Candidate | Runtime/backend governance plus partial-SVD | Assign only if it supports selected partial-SVD closure. |
| Optional backend availability rows | Needs unsupported/unavailable semantics, probe contract, fallback meaning, and non-portability policy. | Candidate | Runtime/backend governance | Assign to runtime owner; likely depends on Day 10 package/runtime claim templates. |
| Static-first optional package mode matrix | Needs install/downstream consumer proof for `SPARSE_MUTEX` and `SPARSE_OPENMP` modes across supported static package routes. | Candidate | Package/ABI productization | Assign to package owner; may close a bounded package gap without shared ABI. |
| Shared-library packaging | Requires product decision, build rules, artifact naming, export/import policy, install/export metadata, downstream consumers, and platform proof. | Candidate / explicit non-claim until selected | Package/ABI productization | Assign to package/ABI owner for Day 5; Day 7 must decide implement versus preserve deferral. |
| Dynamic ABI compatibility | Requires ABI epoch, public layout policy, symbol inventory, export/import macros, soname/install-name policy, compatibility tests, and docs. | Candidate / explicit non-claim until selected | Package/ABI productization | Assign to ABI owner; cannot be incidental to shared-library build work. |
| Runtime-loader behavior | Requires shared-library product decision plus platform-specific loader/runtime validation. | Duplicate / consolidated | Package/ABI plus platform | Fold under shared-library/ABI path if selected. |
| Package-manager support | Requires manager-specific recipes, dependency metadata, install roots, upgrade/uninstall proof, and downstream consumer tests. | Candidate / explicit non-claim until selected | Package/ABI productization | Assign to distribution owner; likely non-goal unless ABI/release mechanics close first. |
| CMake static/shared selector semantics | Relevant only if shared support is selected later. | Duplicate / consolidated | Package/ABI productization | Fold into shared-library package decision. |
| `pkg-config` `Libs.private` split | Relevant only if dependency visibility changes. | Duplicate / consolidated | Package/ABI productization | Fold into package metadata decision. |
| macOS reviewed install/export parity | Needs promotion decision, hosted-runner history, runtime budget, failure triage ownership, and reviewed-platform scope. | Candidate | Platform promotion | Assign to platform owner; compare against Windows promotion candidates. |
| Windows reviewed install-validation parity | Needs promotion decision, hosted-runner history, exact CMake-first scope, failure triage ownership, and reviewed-platform scope. | Candidate | Platform promotion | Assign to platform owner; preserve CMake-first boundary. |
| Windows staged pthread/POSIX test promotion | Needs Windows-native equivalents or portability wrappers, CTest count updates, and hosted MSVC configure/build/execute proof. | Candidate | Windows staged tests | Assign to Windows/test portability owner; likely a complete platform lane if selected. |
| Hosted-runner history for supplemental macOS/Windows lanes | Required before promotion from supplemental to reviewed. | Duplicate / consolidated | Platform promotion | Fold under macOS/Windows promotion candidates. |
| Documentation-link automation | Add maintained target only if docs volume continues to justify it. | Optional-local | Adoption/maintainability | Assign optional docs tooling owner; do not treat as product support. |
| Algorithm reference continued slimming | Move historical/high-friction sections only with link validation and claim-boundary review. | Optional-local | Adoption simplification | Assign adoption docs owner; likely Sprint 145 candidate if evidence decisions settle. |
| Cookbook and adoption navigation maintenance | Keep current as examples, workflows, package surfaces, or reports land. | Optional-local | Adoption simplification | Assign adoption owner; should follow implementation, not lead claims. |
| Public solver-selection wording refresh | Sprint 130 kept this blocked until future evidence supports user-facing claims. | Duplicate / consolidated | Adoption plus QR/partial-SVD | Fold into selected QR/partial-SVD claim gates. |

## Initial Grouping by Epic 12 Workstream

| Workstream | Candidate residuals |
| --- | --- |
| Corpus/oracle architecture | Numerical corpus index; SuiteSparse rank-deficient QR corpus; SuiteSparse/optional-large minimum-norm rows; external-reference helper generated index. |
| QR residual closure | Compatible zero-residual lane; wide residual-only behavior; rank-deficient nullspace/subspace; near-threshold rank as subcase; QR-vs-SVD minimum-norm cross-checks. |
| Partial-SVD residual closure | Repeated/clustered spectra; rank-deficient subspace; low-rank optimality class decision; convergence reporting semantics; SVD/bidiag report rows if selected. |
| Report normalization/freshness | Cross-report normalized index; stale-report scanner; canonical support-tier/claim-boundary fields; direct backend fields; optional coverage/dead-code freshness. |
| Runtime/backend governance | Runtime/backend sentinel expansion; LDLT report-only sentinel; iterative/BiCGSTAB sentinel; eigensolver backend/preconditioner sentinel; optional backend availability rows. |
| Package/ABI productization | Static-first optional package mode matrix; shared-library packaging; dynamic ABI policy; runtime-loader behavior; package-manager support; CMake/pkg-config selector decisions. |
| Platform promotion and staged tests | macOS reviewed install/export parity; Windows reviewed install-validation parity; Windows pthread/POSIX test promotion; hosted-runner history and CTest count updates. |
| Adoption and documentation | Algorithm reference slimming; documentation-link automation; cookbook/navigation maintenance; public solver-selection wording after evidence lands. |
| Maintainability support | Generic QR/SVD/minimum-norm helper consolidation and focused proof-owner movement only when needed by selected gaps. |

## Duplicate, Closed, and Obsolete Fences

| Item | Fence |
| --- | --- |
| Sprint 130 rectangular/nonsymmetric partial-SVD accepted lanes | Already covered as bounded Epic 11 evidence; only residual edge cases and convergence semantics remain active. |
| Sprint 131 corpus taxonomy and first large-matrix index path | Already covered as policy/first-index evidence; normalized corpus/report implementation remains active. |
| Sprint 132 backend/runtime vocabulary and existing sentinel metadata | Already covered as governance baseline; expansion and common scanner remain active. |
| Sprint 133 static-first package decision and default package proof | Already covered as maintained static-first baseline; optional package modes, shared ABI, and distribution remain active. |
| Sprint 134 Linux package-contract CI promotion | Already covered; macOS/Windows promotion and staged Windows tests remain active. |
| Sprint 135 cookbook/navigation/algo split baseline | Already covered; maintenance and further simplification remain optional/follow-on. |
| Sprint 136 claim audit/unsupported cleanup | Already covered; public claim freeze remains a Sprint 137 validation step, not a new implementation residual. |

## Unresolved Questions for Day 5

1. Should the package/ABI owner treat shared-library support as an Epic 12
   implementation candidate or explicitly choose the bounded optional static
   mode matrix instead?
2. Which QR residual can be closed completely after Sprint 138 corpus metadata
   without creating broad QR/minimum-norm/nullspace parity wording?
3. Which partial-SVD residual can be closed completely without requiring broad
   LAPACK/SciPy/NumPy/SuiteSparse parity?
4. Should report normalization include coverage/dead-code freshness in Epic 12
   or limit initial implementation to benchmark/sentinel/guardrail/oracle
   report families?
5. Is the highest-value platform promotion macOS install/export, Windows
   CMake-first install/downstream, or Windows staged pthread/POSIX portability?
6. Which adoption simplification should wait until package/platform/runtime
   decisions settle, and which can proceed as documentation-only cleanup?

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All Epic 11 closeout residuals have been considered. | Complete | Intake queue covers Sprint 136 residual publication plus Sprint 130-136 retrospective residual details. |
| Duplicates and already-closed items are fenced. | Complete | Duplicate, closed, and obsolete fences table separates active work from Epic 11 completed evidence. |
| Candidate residuals are grouped for owner assignment. | Complete | Initial grouping by Epic 12 workstream provides Day 5 owner-assignment inputs. |

