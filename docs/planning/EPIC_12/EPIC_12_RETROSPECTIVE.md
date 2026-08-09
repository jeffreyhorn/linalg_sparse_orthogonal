# Epic 12 Retrospective

**Epic:** 12 - Corpus-Grade Evidence, Report Governance & Productization
Follow-Through
**Sprints:** 137-146
**Status:** Complete

## Epic Objective

Epic 12 started from the completed Epic 11 closeout with a broad sparse linear
algebra library that had improved product discipline but still carried several
large gaps: maintained corpus evidence, bounded QR and partial-SVD residuals,
report freshness semantics, runtime/backend governance, static-first package
proof, platform support-tier clarity, and adoption-surface complexity.

The objective was to close selected high-value gaps completely where evidence
could support closure and to publish explicit residuals where the work remained
too broad for honest claims.

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence contract | Sprint 137 froze the post-Epic-11 baseline, selected complete-gap targets, and established evidence templates, quality maps, and non-goal boundaries. |
| Corpus architecture | Sprint 138 created maintained corpus structure, manifests, source-controlled metadata, skip/defer semantics, and reproducible local oracle/report generation. |
| QR residual closure | Sprint 139 closed the selected QR rank-deficient residual for `qr_rank_deficient_6x4_nullspace_v1` with fixture-local proof ownership and bounded documentation. |
| Partial-SVD residual closure | Sprint 140 closed the selected clustered/repeated-spectrum partial-SVD residual for `partial_svd_clustered_repeated_diag8x6_k3_v1` with projector, residual, convergence, and fail-closed checks. |
| Report governance | Sprint 141 normalized report indexes and freshness semantics while preserving the distinction between source-controlled rows and generated local evidence. |
| Runtime/backend governance | Sprint 142 documented runtime/backend control boundaries and sentinel interpretation without promoting portable performance or backend-superiority claims. |
| Package/ABI posture | Sprint 143 converted the maintained package contract into a static-first install/export surface with Make, CMake, `pkg-config`, version, deferral, and downstream proof. |
| Platform tiers | Sprint 144 promoted platform lanes without flattening support tiers: Linux remains strongest, macOS has reviewed static-first install/export proof, and Windows remains CMake-first with staged exclusions. |
| Adoption surface | Sprint 145 simplified the first-use route through README, INSTALL, examples, cookbook, solver-selection, and selected headers while preserving evidence boundaries. |
| Final closeout | Sprint 146 inventoried final evidence, ran local validation, reconciled hosted master CI, audited claims, published residuals, and reconciled the project plan. |

## Validation Evidence

Sprint 146 provides the final Epic 12 validation snapshot:

| Evidence | Result | Boundary |
| --- | --- | --- |
| Corpus schema | Passed on Sprint 146 Day 5. | Supports manifest/schema validity, not broad solver correctness. |
| Report-index unit tests | Passed on Sprint 146 Day 5. | Supports normalization behavior and freshness semantics. |
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

The full local C quality gate was not required during Sprint 146 because the
sprint closeout changed documentation only and introduced no `.c` or `.h`
changes. Earlier Epic 12 implementation sprints ran their required gates for
their touched surfaces.

## Earned Claims

Epic 12 earns these claims, with the stated boundaries:

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

## Non-Claims

Epic 12 does not claim:

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

## State-Of-The-Art Assessment

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

## Residuals And Future-Epic Candidates

The post-Epic-12 residual queue is published in
[day11-published-residual-queue.md](./SPRINT_146/artifacts/day11-published-residual-queue.md).
The highest-value future-epic candidates are:

- Windows platform closure: branch-specific hosted CI publication, Windows
  staged pthread/POSIX test portability, and reviewed Windows install-validation
  parity.
- Numerical corpus expansion: broader QR and partial-SVD fixture families plus
  external-library comparison semantics.
- Shared-library and ABI productization: shared install/export, loader tests,
  symbol policy, ABI rules, package metadata, and package-manager distribution.
- Report evidence refresh: selected generated benchmark, sentinel, coverage,
  dead-code, and guardrail families with freshness gates.
- Adoption/documentation completion: tutorial alignment and broader public
  header cleanup without widening support claims.
- Runtime/backend follow-through: typed-control promotion review and additional
  sentinel rows.
- Competitive positioning: state-of-the-art or external-parity decision only
  after direct comparative evidence exists.

## What Went Well

1. **The epic closed selected gaps instead of spreading effort thinly.**
   QR and partial-SVD work ended with named fixture-local closures and
   explicit residuals for broader coverage.

2. **Evidence boundaries stayed visible.**
   Corpus rows, generated report outputs, CI lanes, package support, platform
   tiers, and adoption docs kept separate proof meanings.

3. **Static-first packaging became executable product truth.**
   Make install, `pkg-config`, CMake install/export, version checks,
   downstream consumers, and shared-library deferral are now guarded by tests
   and documentation.

4. **Platform language became more precise.**
   Linux, macOS, and Windows lanes now have clearer reviewed, supplemental,
   staged, deferred, local-only, and hosted-only meanings.

5. **The final closeout rejected unsupported competitive claims.**
   State-of-the-art status remained a non-claim because the epic did not
   produce direct comparative evidence.

## What Didn't Go Well

1. **Broad numerical coverage remains large.**
   The selected QR and partial-SVD residuals closed, but broad solver-family
   parity still requires a larger corpus and comparison program.

2. **Hosted branch evidence is still external to final local closeout.**
   Sprint 146 could inspect green `master` hosted baselines, but branch-specific
   Sprint 146 hosted CI remains a residual until PR workflows run.

3. **Windows parity remains incomplete.**
   Windows CMake-first support improved, but staged pthread/POSIX tests,
   reviewed install-validation parity, Makefile parity, and `pkg-config` parity
   remain unpromoted.

4. **Generated report freshness is intentionally selective.**
   Source-controlled rows and local generated artifacts are coherent, but broad
   generated report refreshes should be tied to concrete claims.

5. **Adoption work did not finish every documentation surface.**
   The high-value first-use path improved, but tutorial alignment and all-header
   cleanup remain future work.

## Closeout Assessment

Epic 12 is complete. It materially improved the library's evidence ownership,
fixture-local numerical proof, report governance, runtime/backend support
boundaries, static-first package confidence, platform-tier clarity, and
adoption surface.

It deliberately leaves broad competitive, numerical, package-manager,
shared-library ABI, Windows parity, generated-report freshness, and portable
performance claims as explicit non-claims or future work.
