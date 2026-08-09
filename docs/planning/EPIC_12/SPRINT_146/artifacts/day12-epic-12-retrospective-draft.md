# Day 12 Epic 12 Retrospective Draft

## Scope

Day 12 drafts the Epic 12 retrospective from the Sprint 146 evidence package.
This is not the final closeout file. Day 13 must still reconcile the Epic 12
project plan, and Day 14 must finalize the retrospective after any remaining
validation, CI, or wording evidence is available.

This draft is grounded in:

- [day2-corpus-solver-evidence-inventory.md](./day2-corpus-solver-evidence-inventory.md)
- [day3-support-evidence-inventory.md](./day3-support-evidence-inventory.md)
- [day5-final-local-validation-command-log.md](./day5-final-local-validation-command-log.md)
- [day6-ci-evidence-intake.md](./day6-ci-evidence-intake.md)
- [day7-cross-platform-reconciliation.md](./day7-cross-platform-reconciliation.md)
- [day8-public-claim-audit.md](./day8-public-claim-audit.md)
- [day9-support-maintainer-claim-audit.md](./day9-support-maintainer-claim-audit.md)
- [day11-published-residual-queue.md](./day11-published-residual-queue.md)

## Draft Epic Objective

Epic 12 started from the completed Epic 11 closeout with a broad sparse linear
algebra library that had improved product discipline but still carried several
large gaps: maintained corpus evidence, bounded QR and partial-SVD residuals,
report freshness semantics, runtime/backend governance, static-first package
proof, platform support-tier clarity, and adoption-surface complexity.

The epic objective was to close selected high-value gaps completely where
possible and to publish conservative residuals where the evidence still did
not justify broader claims.

## Draft Major Outcomes

| Area | Draft Outcome |
| --- | --- |
| Evidence contract | Sprint 137 froze the post-Epic-11 baseline, selected complete-gap targets, and established evidence templates and non-goal boundaries. |
| Corpus architecture | Sprint 138 created maintained corpus structure, manifests, source-controlled metadata, skip/defer semantics, and reproducible local oracle/report generation. |
| QR residual closure | Sprint 139 closed the selected QR rank-deficient residual for `qr_rank_deficient_6x4_nullspace_v1` with fixture-local proof ownership and bounded documentation. |
| Partial-SVD residual closure | Sprint 140 closed the selected clustered/repeated-spectrum partial-SVD residual for `partial_svd_clustered_repeated_diag8x6_k3_v1` with projector, residual, convergence, and fail-closed checks. |
| Report governance | Sprint 141 normalized report indexes and freshness semantics while preserving the distinction between source-controlled rows and generated local evidence. |
| Runtime/backend governance | Sprint 142 documented runtime/backend control boundaries and sentinel interpretation without promoting portable performance or backend-superiority claims. |
| Package/ABI posture | Sprint 143 converted the maintained package contract into a static-first install/export surface with Make, CMake, `pkg-config`, version, deferral, and downstream proof. |
| Platform tiers | Sprint 144 promoted platform lanes without flattening support tiers: Linux remains strongest, macOS has reviewed static-first install/export proof, and Windows remains CMake-first with staged exclusions. |
| Adoption surface | Sprint 145 simplified the first-use route through README, INSTALL, examples, cookbook, solver-selection, and selected headers while preserving evidence boundaries. |
| Final closeout evidence | Sprint 146 inventoried final evidence, ran local validation, reconciled hosted master CI, audited claims, and published the residual queue. |

## Draft Validation Evidence

| Evidence | Draft Result | Boundary |
| --- | --- | --- |
| Corpus schema | Passed on Sprint 146 Day 5. | Supports manifest/schema validity, not broad solver correctness. |
| Report normalization and freshness | Passed on Sprint 146 Day 5 for source-controlled rows and selected support families. | Supports report navigation and diagnostics, not generated pass proof. |
| Static package deferral | Passed on Sprint 146 Day 5. | Supports static-first package posture and shared-library deferral. |
| Make install and `pkg-config` | Passed locally on Sprint 146 Day 5: 23 passed, 0 failed. | Local static archive install/downstream proof. |
| CMake install/export | Passed locally on Sprint 146 Day 5: 26 passed, 0 failed, 0 skipped. | Local static CMake package/downstream proof. |
| Maintained examples | Passed locally on Sprint 146 Day 5: 14 example binaries built. | Build proof only, not complete example-output proof. |
| QR corpus proof | Passed locally on Sprint 146 Day 5: 4 tests, 0 failures, 83 assertions, residual `2.220e-16`. | Fixture-local proof for the named QR residual. |
| Partial-SVD corpus proof | Passed locally on Sprint 146 Day 5: 6 tests, 0 failures, 140 assertions. | Fixture-local proof for the named partial-SVD residual. |
| Local oracle/report refresh | Passed on Sprint 146 Day 5 with ignored `build/` outputs. | Reproducible local generated evidence only. |
| Hosted Linux baseline | Latest inspected `master` CI run passed on commit `daac9a85d516f72100c34b90b92ec78941a72200`. | Hosted master baseline, not branch-specific Sprint 146 proof. |
| Hosted macOS baseline | Latest inspected `master` macOS CI run passed on commit `daac9a85d516f72100c34b90b92ec78941a72200`. | Hosted master baseline, not branch-specific Sprint 146 proof. |
| Hosted Windows baseline | Latest inspected `master` Windows CI run passed with `56` expected CTest registrations. | Hosted master baseline, Windows CMake-first support only. |

## Draft Earned Claims

Epic 12 appears to earn the following bounded claims:

- The project has a maintained corpus/report architecture with
  source-controlled manifests, expected rows, schemas, report-family metadata,
  and reproducible local generation commands.
- The selected QR residual is closed for
  `qr_rank_deficient_6x4_nullspace_v1` with fixture-local nullspace and
  residual proof.
- The selected partial-SVD residual is closed for
  `partial_svd_clustered_repeated_diag8x6_k3_v1` with top-k value,
  subspace/projector, residual, orthogonality, convergence-budget, and
  fail-closed proof.
- Report indexes normalize heterogeneous report-family rows and expose
  freshness diagnostics without treating generated local rows as release or
  hosted pass proof.
- Runtime/backend controls and sentinels are documented as local governance and
  local measurement surfaces, not portable performance or backend-superiority
  evidence.
- The maintained package posture is static-first, with local Make install,
  `pkg-config`, CMake install/export, exact-version, mismatch-version, and
  downstream consumer proof.
- Platform support tiers are clearer: Linux is the strongest reviewed
  source-of-truth lane, macOS has reviewed static-first install/export proof,
  and Windows is reviewed CMake-first with supplemental CMake
  install/downstream confidence.
- The adoption surface is easier to enter through local build, first solve,
  data input, solver choice, diagnostics, static-first install/downstream use,
  and advanced controls.
- Public and support documentation scanned during Sprint 146 did not require
  additional wording fixes for unsupported state-of-the-art, broad parity,
  shared-library ABI, package-manager, generated report freshness, Windows
  parity, or portable performance claims.

## Draft Non-Claims

Epic 12 still should not claim:

- unqualified state-of-the-art sparse linear algebra status;
- broad external-library parity against LAPACK, NumPy, SciPy, SuiteSparse,
  ARPACK, PETSc, Trilinos, or other ecosystems;
- broad QR, SVD, or partial-SVD correctness beyond the reviewed corpus
  fixtures;
- raw QR basis identity or singular-vector identity parity;
- broad rank-threshold, minimum-norm, null-space, repeated-spectrum,
  convergence-rate, sparse-output, partial-result, platform, or performance
  generality;
- portable performance, benchmark superiority, backend superiority, optional
  backend availability, or runtime scaling guarantees;
- generated report freshness from source-controlled rows alone;
- coverage completeness or zero-dead-code status;
- shared-library support, dynamic ABI compatibility, runtime-loader
  compatibility, package-manager distribution, or static/shared package
  selector support;
- Windows Makefile parity, Windows `pkg-config` parity, reviewed Windows
  install-validation parity, or Windows staged pthread/POSIX test closure;
- branch-specific hosted Sprint 146 CI success until branch/PR hosted runs are
  available and reconciled.

## Draft State-Of-The-Art Assessment

Epic 12 did not earn an unqualified state-of-the-art claim.

The strongest truthful assessment is that Epic 12 improved the project's
engineering maturity, evidence ownership, bounded numerical proof, report
governance, static-first package confidence, platform-tier clarity, and
first-use documentation. Those are meaningful library-quality improvements,
but they are not a direct comparative study against established sparse linear
algebra ecosystems.

A future state-of-the-art or external-parity claim would require direct
comparative evidence that names libraries, versions, fixtures, metrics,
tolerances, platforms, compilers, optional dependencies, memory behavior,
performance methodology, failure semantics, and caveats. Until that evidence
exists, state-of-the-art remains an explicit non-claim.

## Draft Lessons Learned

1. Evidence contracts kept the epic from overclaiming.
   The corpus, report, package, platform, and adoption work stayed useful
   because each claim had a source-controlled owner, validation command, and
   non-claim boundary.

2. Fixture-local numerical closure is valuable but narrow.
   The QR and partial-SVD residuals are genuinely closed for named fixtures,
   yet the remaining broad numerical coverage gap is still large enough to
   need a dedicated future corpus expansion program.

3. Report freshness needs disciplined wording.
   Normalized source-controlled rows are useful navigation metadata, but they
   must not be confused with regenerated local outputs or hosted proof.

4. Static-first packaging became stronger by rejecting adjacent claims.
   The package surface is clearer because shared-library ABI, loader behavior,
   package-manager support, and static/shared selectors remain explicitly
   deferred.

5. Platform promotion works best when each lane keeps its own support tier.
   Linux, macOS, and Windows evidence improved, but Windows staged exclusions
   and install-validation parity remain real residuals.

6. Adoption simplification cannot close product or numerical gaps.
   Better docs reduce first-use friction, but they do not create broad solver
   parity, portable performance, ABI support, or competitive evidence.

## Draft Residual Summary

The published residual queue remains the planning source of truth:

- R1: branch-specific hosted CI reconciliation for Sprint 146;
- R2-R3: Windows staged portability and reviewed install-validation parity;
- R4 and R14: shared-library ABI and package-manager productization;
- R5-R6 and R12: broader QR, partial-SVD, and external-library corpus work;
- R7: generated benchmark, sentinel, coverage, dead-code, and guardrail
  refresh package;
- R8-R9: tutorial alignment and broader public-header cleanup;
- R10-R11: runtime/backend typed-control and sentinel follow-through;
- R13: state-of-the-art competitive decision.

## Draft Next-Epic Recommendations

The highest-value next epic should choose one complete gap closure instead of
partially advancing many queues. The best candidates are:

| Candidate | Why |
| --- | --- |
| Windows platform closure | It would close the largest remaining platform support gap by promoting staged tests and deciding reviewed install-validation parity with hosted proof. |
| Numerical corpus expansion | It would turn the bounded QR and partial-SVD fixture successes into broader maintained corpus families without overclaiming isolated cases. |
| Shared-library and ABI productization | It would resolve a major product/distribution gap only if ABI policy, loader behavior, symbol checks, package metadata, and cross-platform validation land together. |
| Report evidence refresh | It would make generated report families fresh and reviewable for selected claims without blurring local and hosted evidence. |
| Competitive positioning | It should wait until direct external comparison evidence exists; otherwise state-of-the-art remains a non-claim. |

## Day 13 Handoff

Day 13 should reconcile the Sprint 137-146 project-plan items against this
draft. Any incomplete item must either be tied to a closed evidence artifact or
carried into the residual queue without changing public claims.
