# Sprint 136 Day 12 Residual Queue Publication

## Purpose

Day 12 publishes the post-Epic-11 residual queue. It consolidates unresolved
work from Sprint 118-135 retrospectives and handoffs, then classifies each
item so future work is visible without widening Epic 11 earned claims.

This artifact does not reopen implementation scope. It preserves deferred QR
residual work, report/index work, package/platform work, runtime governance,
and competitive non-claims as future or optional work with promotion criteria.

## Classification Model

| Classification | Meaning |
|---|---|
| Future-epic candidate | Large enough to need explicit planning, owners, validation, and claim gates. |
| Evidence-blocked | Needs oracle, metadata, fixture, tolerance, runtime, or hosted-runner evidence before implementation or promotion. |
| Metadata-blocked | Needs row-level support-tier, freshness, ownership, or output-class metadata before it can be promoted. |
| Optional-local work | Useful maintainer or developer improvement that does not change support claims by itself. |
| Explicit non-claim | Must remain absent from public/support wording until a future product decision and proof stack exists. |

## Post-Epic-11 Residual Queue

| Residual | Classification | Owner surface | Promotion criteria | Claim boundary |
|---|---|---|---|---|
| QR residual expansion for compatible zero-residual and wide residual-only lanes | Evidence-blocked future-epic candidate | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | Define distinct trust value, output semantics, solution-selection rules, tolerance, residual metric, and no-minimum-norm interpretation before adding fixtures. | No broad QR solve, rank-deficient solve, nullspace, or minimum-norm claim. |
| QR rank-deficient nullspace/subspace expansion | Evidence-blocked future-epic candidate | `tests/test_qr.c`, `tests/test_qr_helpers.h`, QR external helper, maintainer guide | Add rank/nullity metadata, projector or two-way projection metrics, sign/orientation policy, support tier, and fixture-local tolerances. | No raw Q/nullspace basis equality or broad subspace parity claim. |
| SuiteSparse rank-deficient QR corpus evidence | Metadata-blocked future-epic candidate | corpus taxonomy owner, QR owner, optional-data/report owners | Add independent expected-rank metadata, availability/missing-data behavior, support tiers, matrix provenance, runtime budget, and reviewed validation. | No broad SuiteSparse corpus coverage claim. |
| SuiteSparse and optional-large minimum-norm expansion | Metadata-blocked future-epic candidate | `tests/test_colamd.c`, QR/SVD helper owners, corpus taxonomy owner | Add extraction rules, RHS policy, rank/nullity, residual/norm metrics, skip behavior, optional-data policy, and support tier per row. | No broad minimum-norm, corpus, platform, or optional-data claim. |
| Additional QR-vs-SVD minimum-norm cross-checks | Evidence-blocked | QR solve owner, COLAMD owner, SVD helper owner | Add bounded fixture keys, QR residual/norm metrics, SVD tolerance, and explicit non-oracle boundary. | Cross-checks remain consistency evidence, not a global oracle. |
| Generic QR/SVD/minimum-norm helper consolidation | Optional-local work | QR/SVD test helper owners | Move only behavior-specific helpers with call-site tolerance preservation and focused validation. | Helper movement does not create new numerical claims. |
| Partial-SVD residual expansion beyond accepted lanes | Evidence-blocked future-epic candidate | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, SVD external helper, maintainer guide | Add rectangular, repeated/clustered, rank-deficient subspace, low-rank optimality, convergence-budget, and corpus fixtures only with metric, gap, tolerance, and failure semantics. | No broad partial-SVD vector, subspace, convergence-rate, sparse-output, or performance claim. |
| Numerical corpus index | Metadata-blocked future-epic candidate | corpus taxonomy owner, solver-family owners, report-index owner | Add row-level SuiteSparse/integration/product-observed/expected-error/oracle/tolerance/runtime/support metadata. | No broad Matrix Market, SuiteSparse, optional-data, or ecosystem coverage claim. |
| External-reference helper generated index | Metadata-blocked | external-oracle owner, report-index owner | Preserve helper-specific output classes, fixture keys, skip behavior, tolerance policy, and assertion class. | Value helpers do not imply vector, projector, basis, or parity claims. |
| Cross-report normalized index | Metadata-blocked future-epic candidate | report-index owner | Define common schema that preserves each report family's row meaning, freshness, support tier, failure class, and claim boundary. | No normalized cross-report correctness, coverage, release, or performance proof claim. |
| Coverage index and coverage gap follow-through | Metadata-blocked optional-local work | coverage workflow owners, source-family owners | Preserve backend, threshold, tree-mutating behavior, source filters, freshness, reset policy, support tier, and owner labels. | Coverage percentages remain supplemental, not reviewed behavior completeness. |
| Dead-code freshness and public-surface review | Metadata-blocked optional-local work | dead-code workflow owner, public-header owners | Add freshness metadata and require API owner review before any removal or public-surface interpretation. | Dead-code reports remain report-completeness and triage evidence, not removal-ready proof. |
| Runtime/backend sentinel expansion | Evidence-blocked future-epic candidate | benchmark owners, runtime governance owner | Define bounded iterative/eigensolver/SVD/backend fixtures, metrics, tolerances, runtime budget, variance policy, backend-state semantics, and claim gates. | No portable performance, backend parity, OpenMP speedup, scalability, or memory claim. |
| Automated stale-report scanner | Metadata-blocked optional-local work | report-index owner | Wait until report families share enough metadata to detect stale branch, commit, generated time, support tier, and row meaning consistently. | Freshness remains traceability context, not CI/release/support proof. |
| Optional backend availability rows | Evidence-blocked | runtime governance owner, benchmark/report owners | Define unsupported/unavailable semantics, probe contract, fallback meaning, and non-portability policy. | No optional backend availability guarantee or builtin/optional backend parity claim. |
| Static-first optional package mode matrix | Evidence-blocked future-epic candidate | package-validation owner, Make/CMake install owners | Add install/downstream consumer proof for `SPARSE_MUTEX` and `SPARSE_OPENMP` modes across supported static package routes. | Optional modes are not part of the default package contract today. |
| Shared-library packaging | Explicit non-claim future-epic candidate | package/ABI product owner, CMake/package owners | Requires product decision, build rules, artifact naming, export/import policy, install/export metadata, downstream consumers, and platform proof. | No shared-library support claim. |
| Dynamic ABI compatibility | Explicit non-claim future-epic candidate | ABI policy owner, public-header owners | Requires ABI epoch, public layout policy, symbol inventory, export/import macros, soname/install-name policy, compatibility tests, and docs. | No dynamic ABI compatibility claim. |
| Runtime-loader behavior | Explicit non-claim future-epic candidate | package/ABI/platform owners | Requires shared-library product decision plus platform-specific loader/runtime validation. | No runtime-loader behavior claim. |
| Package-manager support | Explicit non-claim future-epic candidate | distribution/package owners | Requires manager-specific recipes, dependency metadata, install roots, upgrade/uninstall proof, and downstream consumer tests. | No Homebrew, apt/deb, rpm/dnf, pacman, vcpkg, Conan, or equivalent support claim. |
| macOS reviewed install/export parity | Evidence-blocked future-epic candidate | platform CI owner, macOS workflow owner | Requires promotion decision, hosted-runner history, runtime budget, failure triage ownership, and reviewed-platform scope. | macOS package confidence remains supplemental. |
| Windows reviewed install-validation parity | Evidence-blocked future-epic candidate | platform CI owner, Windows workflow owner | Requires promotion decision, hosted-runner history, exact CMake-first scope, failure triage ownership, and reviewed-platform scope. | Windows install/downstream confidence remains supplemental. |
| Windows staged pthread/POSIX test promotion | Evidence-blocked future-epic candidate | Windows CMake/CTest owner, affected test owners | Requires Windows-native equivalents or portability wrappers, intentional CTest count updates, and hosted MSVC configure/build/execute proof. | Staged tests remain outside reviewed Windows subset. |
| Documentation-link automation | Optional-local work | docs tooling owner | Add maintained target only if docs volume continues to justify it; keep local link checks scoped and deterministic. | Tooling does not change product support claims. |
| Algorithm reference continued slimming | Optional-local work | `docs/algorithm.md`, `docs/algorithm_history.md` owners | Move historical/high-friction sections only with link validation and claim-boundary review. | Documentation organization does not create new solver behavior. |
| Cookbook and adoption navigation maintenance | Optional-local work | `docs/cookbook.md`, `examples/README.md`, README/tutorial owners | Update when new examples, workflows, package surfaces, or report targets land. | Adoption docs do not imply new package, platform, report-schema, or solver behavior. |

## Deferred QR Residual Queue

| QR residual item | Current baseline | Blocker | Promotion criteria |
|---|---|---|---|
| Compatible zero-residual rank-deficient solve | Existing deterministic compatible solve evidence. | Need proof that a separate zero-residual lane adds trust beyond deterministic tests and cannot be misread as minimum-norm evidence. | Add named fixture, exact residual expectation, solution-selection note, and maintainer non-claim wording. |
| Wide residual-only behavior | Existing wide/underdetermined exact-value and minimum-norm lanes. | Need output semantics, rank/nullity, raw-Q/economy boundaries, and residual-only proof value. | Add bounded wide fixture with residual metric and explicit no-minimum-norm/no-basis-parity boundary. |
| Rank-deficient nullspace/subspace beyond current projector lanes | Existing duplicate/dependent-row projector fixtures. | Need rank/nullity, sign/orientation, projector or principal-angle policy, and tolerance rules. | Add projector/subspace metrics and support-tier note before public or maintainer wording expands. |
| Near-threshold QR rank behavior | Existing threshold-family evidence. | Need threshold family, perturbation scale, tolerance, and rank-model policy. | Add threshold metadata and fixture-local rank interpretation. |
| SuiteSparse rank-deficient QR | Existing SuiteSparse controls and smoke baselines. | Missing independent expected-rank metadata, support tier, optional-data policy, and runtime/freshness rules. | Add corpus row metadata and reviewed validation before promotion. |
| SuiteSparse minimum-norm | Existing `west0067` submatrix smoke plus small exact-value lanes. | Missing extraction rules, rank/nullity, residual/norm metrics, and optional/report-only support tiers. | Add per-fixture owner records and skip behavior before adding checked-in or optional lanes. |
| Additional QR-vs-SVD minimum-norm cross-checks | Existing bounded 2 x 4 cross-check baseline. | Need fixture keys, SVD tolerance, QR residual/norm metrics, and non-oracle boundary. | Add only as bounded consistency evidence with no global SVD oracle claim. |

## Future-Epic Candidate List

The following work is too large or claim-sensitive for incidental cleanup:

- QR residual and SuiteSparse/corpus expansion;
- partial-SVD edge-case residual and convergence-budget expansion;
- corpus/report normalized indexing with row-meaning preservation;
- runtime/backend sentinel expansion and stale-report automation;
- shared-library, ABI, runtime-loader, and package-manager productization;
- platform promotion for macOS/Windows package install/export confidence;
- Windows staged pthread/POSIX test portability and promotion.

## Explicit Non-Claim Register

These remain non-claims after Epic 11:

- no unqualified state-of-the-art claim;
- no broad ecosystem replacement or external-library parity claim;
- no every-solver-family external oracle coverage claim;
- no broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  or dense-library parity claim;
- no raw Q-basis, raw nullspace-basis, singular-vector, eigenvector, sign,
  orientation, or unique-basis parity claim;
- no broad QR, SVD, partial-SVD, minimum-norm, nullspace, subspace, corpus,
  optional-data, sparse-output, drop-tolerance, convergence-rate, or
  partial-result claim;
- no portable performance, scalability, memory, runtime, OpenMP speedup,
  backend parity, optional-backend availability, or universal reorder/fill
  superiority claim;
- no coverage percentage as reviewed behavioral completeness;
- no dead-code report as removal-ready proof;
- no generated report/index row as broad correctness, release, coverage, or
  performance proof;
- no shared-library packaging claim;
- no dynamic ABI compatibility claim;
- no runtime-loader behavior claim;
- no package-manager support claim;
- no reviewed macOS install/export parity claim;
- no reviewed Windows install-validation parity claim;
- no Windows staged pthread/POSIX test promotion claim.

## Validation And Claim Recalibration Links

| Evidence source | Residual use |
|---|---|
| Day 5-7 Sprint 136 validation records | Establish local validation and report/package confidence used to avoid reclassifying residuals as immediate blockers. |
| Day 8 competitive baseline | Separates earned local/support evidence from unsupported broad claims. |
| Day 9 claim recalibration | Defines where earned claims may appear and which claims must remain local, supplemental, deferred, or unsupported. |
| Day 10 unsupported-claim audit | Confirms public/support surfaces had no P0 unsupported wording before cleanup. |
| Day 11 unsupported-claim cleanup | Confirms no public-doc edits were required and preserves non-claims for residual publication. |

## Completion Criteria

| Criterion | Status | Evidence |
|---|---|---|
| Residuals are visible, classified, and actionable. | Complete | Consolidated queue includes classification, owner surface, promotion criteria, and claim boundary. |
| QR residual work is preserved without becoming immediate sprint scope. | Complete | Deferred QR residual queue records current baselines, blockers, and promotion criteria. |
| Future work is separated from earned Epic 11 claims. | Complete | Future-epic candidates and explicit non-claim register prevent residual work from being read as completed support. |
