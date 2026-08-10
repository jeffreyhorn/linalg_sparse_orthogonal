# Sprint 147 Day 6 Claim Target Register

## Purpose

Day 6 converts the selected Epic 13 gaps into candidate earned claims, required
evidence, rejected claims, promotion rules, and rollback rules. These targets
are planning contracts for later sprints. They are not current product claims.

## Claim States

| State | Meaning |
| --- | --- |
| Candidate | Epic 13 may earn the claim if all required evidence lands. |
| Conditional | The claim may be earned only after a product decision chooses a supported path. |
| Rejected | The claim must not appear in public or support docs. |
| Deferred | Valuable but outside selected Epic 13 closure scope. |
| Rollback | A previously candidate claim must be removed or narrowed because required evidence failed. |

## Candidate Earned Claims

| ID | Candidate Claim | Sprint Owner | Required Evidence | Wording Boundary |
| --- | --- | --- | --- | --- |
| C1 | Windows staged pthread/POSIX test coverage is closed or intentionally replaced in the reviewed Windows CMake lane. | Sprint 148 | Source changes or test splits for `test_threads`, `test_sprint4_integration`, and/or `test_fuzz`; CMake registration; hosted Windows run IDs; updated expected-count policy; docs/report rows. | Claim only the promoted tests and the reviewed CMake lane. Do not imply Windows Makefile or `pkg-config` parity. |
| C2 | Windows CMake install/downstream support has a reviewed parity decision. | Sprint 149 | Product decision artifact; workflow change or explicit rejection; hosted Windows install/downstream proof if promoted; package metadata checks; docs/report rows. | Claim reviewed Windows CMake install/downstream parity only if promoted. Preserve Unix-only Make/`pkg-config` boundaries. |
| C3 | QR has a broader maintained corpus family beyond the Sprint 139 fixture. | Sprint 150 | Multiple source-controlled QR fixture rows; generator/expected rows; focused proof-owner tests; oracle/report integration; schema checks; C quality gate; bounded docs. | Claim only the named QR families, metrics, tolerances, commands, commit/platform context, and support tier. |
| C4 | Partial-SVD has a broader maintained corpus family beyond the Sprint 140 fixture. | Sprint 151 | Multiple source-controlled partial-SVD fixture rows; subspace-safe comparison contract; generator/expected rows; focused proof-owner tests; oracle/report integration; schema checks; C quality gate; bounded docs. | Claim only the named partial-SVD families and comparison semantics. Never claim raw singular-vector identity. |
| C5 | Selected generated report families have freshness gates tied to claim-bearing rows. | Sprint 152 | Stable generation commands; normalized report-index rows; `--require-generated <family> --check-freshness` checks for selected families; artifact policy; docs. | Claim freshness only for selected generated local families and the checked commit/platform/configuration. Do not claim hosted proof unless hosted artifacts are added. |
| C6 | Shared-library ABI support is either implemented with proof or explicitly deferred with stronger tested blockers. | Sprint 153 | ABI/symbol/header audit; product decision; shared build/install/export/loader tests if implemented, or stronger static-first rejection/diagnostics/tests if deferred; package docs. | If implemented, claim only the supported platforms and loader/package paths. If deferred, claim improved static-first deferral only. |
| C7 | The project has a first narrow external comparison harness and study. | Sprint 154 | Selected target; pinned external library/tool versions; fixture set; metrics; tolerances; skip/defer semantics; local or hosted run artifact; report rows; caveats. | Claim only the narrow compared target and conditions. Do not claim broad ecosystem parity. |
| C8 | Tutorial, selected headers, and API reference guidance align with earned support. | Sprint 155 | Tutorial audit/update; selected header cleanup; declaration-preservation scan; API reference plan; docs validation; C quality gate if headers change. | Claim documentation coherence and selected-header cleanup only. Do not create new solver/platform/package support claims. |
| C9 | Epic 13 final claims are reconciled against local validation, hosted CI, reports, docs, and residuals. | Sprint 156 | Final evidence inventory; full local validation package; hosted Linux/macOS/Windows reconciliation; public/support claim audit; residual queue; retrospective. | Claim only evidence that survived final validation and hosted reconciliation. |

## Required Evidence Map

| Evidence Type | Required For | Minimum Bar |
| --- | --- | --- |
| Implementation or test changes | C1, C3, C4, C6, C7, C8 when headers change | Changes are focused, reviewed by owner surface, and preserve existing support boundaries. |
| C quality gate | C1, C3, C4, C6, C7, and C8 if `.c` or `.h` files change | `make format && make lint && make test` passes before claim promotion. |
| Hosted Windows proof | C1, C2 | GitHub Actions run IDs, commit SHA, job names, conclusions, and expected CTest/install proof are recorded. |
| Corpus schema proof | C3, C4 | `python3 scripts/validate_corpus_schema.py` passes after fixture/generator/expected row changes. |
| Focused proof-owner tests | C3, C4 | New or extended tests execute the selected fixture families without expanding broad solver claims. |
| Oracle/report proof | C3, C4, C5, C7 | Generated rows preserve family, command, commit, platform, compiler, configuration, support tier, claim scope, and non-claims. |
| Package/downstream proof | C2, C6 | Install/export, exact-version, mismatch-version, unsupported-artifact, and downstream consumer checks match the selected support tier. |
| External comparison proof | C7, C9 | External library/tool versions, fixture set, metrics, tolerances, statuses, skips, and caveats are documented. |
| Documentation alignment | All candidate claims | README, INSTALL, maintainer guide, benchmark/report docs, solver docs, tutorial, and headers use the same boundaries. |
| Final claim audit | C9 | Public and support wording contains no unsupported state-of-the-art, broad parity, package, ABI, platform, performance, or freshness claims. |

## Rejected And Deferred Claim Register

| Claim | State | Reason |
| --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra status | Rejected | No direct comparative evidence currently supports a broad claim against mature sparse linear algebra ecosystems. |
| Broad external-library parity against LAPACK, NumPy, SciPy, SuiteSparse, ARPACK, PETSc, Trilinos, Eigen, or vendor stacks | Rejected | Sprint 154 can target only one narrow study; broad parity would require a much larger comparison matrix. |
| Broad QR correctness, global rank-threshold policy, broad rank-deficient solve, broad minimum-norm, broad reorder, or SuiteSparse QR parity | Rejected | Sprint 150 can earn only selected maintained QR fixture-family claims. |
| Broad partial-SVD correctness, repeated-spectrum generality, rank-deficient null-space, sparse-output optimality, convergence-rate, or partial-result guarantee | Rejected | Sprint 151 can earn only selected maintained partial-SVD fixture-family claims. |
| Raw QR basis identity or raw singular-vector identity parity | Rejected | QR and SVD comparisons must use residual, rank/nullity, projector, subspace, value, and status semantics. |
| Portable performance or benchmark superiority | Rejected | Generated benchmark and sentinel rows remain local/advisory unless a future comparative performance methodology is built. |
| Generated report freshness from source-controlled rows alone | Rejected | Source-controlled rows define metadata and expected values; generated freshness requires generated artifacts and freshness checks. |
| Shared-library ABI support | Conditional | It can be claimed only if Sprint 153 implements and validates supported shared-library paths. Otherwise only stronger static-first deferral can be claimed. |
| Dynamic ABI compatibility and runtime-loader compatibility | Conditional | These require explicit ABI policy, symbol/version strategy, loader tests, and platform proof. |
| Package-manager distribution | Deferred | R14 remains behind the shared/static product decision, release/versioning policy, recipes, and update/uninstall proof. |
| Runtime/backend typed control API promotion | Deferred | R10 is outside selected scope unless a complete typed-control API/ABI gate is added. |
| Expanded runtime/backend sentinel coverage | Deferred | R11 is outside selected scope unless Sprint 152 selects sentinel freshness as claim-bearing. |
| Windows Makefile parity and Windows `pkg-config` parity | Rejected | Sprint 149 targets Windows CMake install/downstream parity decision only. |

## State-Of-The-Art Decision Boundary

Epic 13 starts with state-of-the-art status rejected. Sprint 156 may revisit
only a narrow state-of-practice statement if Sprint 154 produces direct,
auditable comparison evidence. The minimum decision package is:

- named external library or tool;
- pinned version and installation method;
- exact fixture family and fixture keys;
- metrics and tolerances;
- platform, compiler, build type, optional dependency, and hardware context;
- pass/fail/skip/defer semantics;
- generated or hosted report rows;
- public wording that includes caveats and non-claims.

Anything less keeps broad state-of-the-art, broad parity, and portable
performance wording rejected.

## Promotion Rules

1. A claim may be promoted only by the sprint that owns it or by Sprint 156
   final closeout.
2. A promoted claim must cite implementation, validation, report, CI,
   documentation, package, or comparison evidence appropriate to its surface.
3. Source-controlled metadata alone can promote metadata ownership, not pass
   evidence.
4. Generated local rows can support only their command, commit, platform,
   compiler, configuration, family, and support tier.
5. Hosted platform claims require hosted run IDs and conclusions.
6. Package and ABI claims require downstream consumer proof.
7. External parity claims require direct comparison evidence and caveats.
8. Documentation updates may describe earned support but cannot earn support by
   themselves.

## Rollback Rules

| Failure | Required Rollback |
| --- | --- |
| Required local quality gate fails | Do not promote the claim; record blocker and keep previous non-claim wording. |
| Hosted platform proof is absent or red | Do not promote platform support; preserve staged, supplemental, local-only, or deferred wording. |
| Corpus schema or proof-owner test fails | Remove or mark the candidate corpus rows as unpromoted; preserve fixture-local non-claims. |
| Generated freshness check fails | Remove `--require-generated` promotion or classify the generated family as advisory/deferred. |
| Shared-library implementation fails package/downstream/loader proof | Fall back to stronger static-first deferral wording and tests. |
| External comparison is unavailable, skipped, or inconclusive | Keep broad parity and state-of-the-art rejected; publish the skip/defer reason only. |
| Documentation scan finds unsupported widened wording | Fix wording before closeout or mark the associated claim unearned. |

## Later-Sprint Wording Boundaries

Reusable language for later sprint docs:

- "reviewed Windows CMake lane" means only the named hosted CMake workflow
  surface and registered tests.
- "Windows install-validation parity" means the chosen Windows CMake
  install/downstream proof, not Windows Makefile or `pkg-config` support.
- "maintained QR corpus family" means named source-controlled fixtures,
  expected rows, proof-owner tests, and oracle/report rows with bounded
  metrics.
- "maintained partial-SVD corpus family" means named source-controlled
  fixtures with subspace-safe comparisons and bounded convergence/status
  semantics.
- "generated freshness" means selected local generated artifacts matched to
  current inputs by `normalize_report_index.py`; it is not hosted proof unless
  a hosted artifact policy is added.
- "shared-library support" must not appear unless Sprint 153 implements and
  validates shared build/install/export/downstream/loader behavior.
- "external comparison" means the exact named study only, not ecosystem
  parity.

## Day 7 Handoff

Day 7 should use C1 and C2 to define the Windows evidence gate in detail:

- current reviewed, supplemental, staged, and deferred Windows support tiers;
- promotion requirements for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`;
- reviewed Windows CMake install/downstream promotion or rejection criteria;
- CTest expected-count policy;
- hosted log requirements;
- required documentation and report-row updates.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every candidate claim has concrete required evidence. | Complete | Candidate claim table and required evidence map define implementation, validation, report, CI, package, comparison, and docs requirements. |
| Unsupported broad claims remain rejected. | Complete | Rejected/deferred claim register preserves state-of-the-art, broad parity, broad solver, performance, package-manager, and unsupported Windows claims. |
| Later sprint docs have wording boundaries to reuse. | Complete | Promotion rules, rollback rules, and wording boundaries provide reusable claim language for Sprints 148-156. |
