# Epic 12 Retrospective

**Epic:** 12 - Corpus-Grade Evidence, Report Governance & Productization
Follow-Through
**Sprints:** 137-146
**Status:** Complete

## Epic Objective

Epic 12 started from the completed Epic 11 closeout with a broad sparse linear
algebra library that had useful coverage but still needed stronger evidence
ownership before it could make larger product or competitive claims. The epic
focused on closing selected gaps completely: maintained corpus architecture,
bounded QR and partial-SVD residuals, report-index normalization, runtime and
backend governance, static-first package proof, platform support-tier clarity,
adoption simplification, and final claim recalibration.

The epic deliberately did not try to claim broad state-of-the-art status. Its
standard was narrower and stricter: if a claim did not have source-controlled
evidence, local validation, hosted evidence where required, and explicit
support boundaries, it stayed a non-claim or moved to the residual queue.

## Sprint Outcomes

| Sprint | Outcome |
| --- | --- |
| 137 | Froze the post-Epic-11 baseline, selected complete-gap targets, created evidence contract templates, mapped quality gates, and reaffirmed public non-claims. |
| 138 | Built the maintained numerical corpus architecture with fixture taxonomy, corpus layout, oracle/report row semantics, skip/defer handling, and first durable validation lane. |
| 139 | Closed the selected QR rank-deficient residual for `qr_rank_deficient_6x4_nullspace_v1` with corpus-backed fixture evidence, focused proof ownership, docs, and validation. |
| 140 | Closed the selected partial-SVD clustered/repeated-spectrum residual for `partial_svd_clustered_repeated_diag8x6_k3_v1` with value, projector, residual, convergence, and fail-closed checks. |
| 141 | Normalized report indexes and freshness gates across report families while preserving generated-output boundaries and source-controlled row meaning. |
| 142 | Clarified runtime/backend governance, typed-control decisions, sentinel interpretation, and performance non-claims. |
| 143 | Made the package/ABI product decision explicit: static-first support was strengthened, while shared-library ABI and package-manager support remained deferred. |
| 144 | Promoted platform support lanes without flattening tiers: Linux remained strongest, macOS gained reviewed static-first install/export proof, and Windows stayed CMake-first with staged exclusions. |
| 145 | Simplified the first-use adoption surface across README, INSTALL, examples, cookbook, solver selection, and selected public headers without widening support claims. |
| 146 | Inventoried final evidence, ran final local validation, reconciled hosted master CI, audited claims, published residuals, reconciled the project plan, and closed Epic 12. |

## Major Outcomes

| Area | Outcome |
| --- | --- |
| Evidence ownership | Evidence contracts and sprint artifacts now tie claims to owners, source paths, validation commands, freshness rules, and non-claim boundaries. |
| Corpus architecture | Corpus manifests, schemas, expected rows, optional-data semantics, skip/defer behavior, and generated-local oracle/report outputs have a maintained shape. |
| QR evidence | The selected QR rank-deficient residual is closed for a named fixture with focused compiled proof and bounded documentation. |
| Partial-SVD evidence | The selected partial-SVD clustered-spectrum residual is closed for a named fixture with subspace-safe comparison and convergence/failure checks. |
| Report governance | Report-family rows normalize across heterogeneous sources, preserve row meaning, and expose freshness diagnostics without converting generated local rows into hosted proof. |
| Runtime/backend governance | Runtime controls, backend decisions, and sentinels are documented as local governance and local measurement surfaces, not ABI or portable performance promises. |
| Package/ABI | The maintained product contract is static-first, backed by Make install, `pkg-config`, CMake install/export, version checks, downstream consumer proof, and shared-library deferral guards. |
| Platform support | Linux, macOS, and Windows now have clearer reviewed, supplemental, staged, local-only, hosted-only, and deferred meanings. |
| Adoption | The public front door is shorter and more coherent while retaining links to deeper support-tier and evidence documentation. |
| Final closeout | Epic 12 ends with a published retrospective, project-plan reconciliation, residual queue, and next-epic handoff. |

## Validation Evidence

The final Sprint 146 validation package provides the Epic 12 closeout snapshot:

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

Sprint 146 changed documentation only and introduced no `.c` or `.h` changes,
so its final full C quality gate was not required. Earlier Epic 12 sprints ran
their required gates according to the surfaces they touched.

## Earned Claims

Epic 12 earns these claims with their qualifiers:

- The project has a maintained corpus/report architecture with
  source-controlled manifests, expected rows, schemas, report-family metadata,
  skip/defer rules, and reproducible local generation commands.
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

## Residuals And Future-Epic Candidates

The post-Epic-12 residual queue is published in
[SPRINT_146/artifacts/day11-published-residual-queue.md](./SPRINT_146/artifacts/day11-published-residual-queue.md).
The highest-value future-epic candidates are:

| Candidate | Residuals | Closure Target |
| --- | --- | --- |
| Windows platform closure | R1, R2, R3 | Branch-specific hosted CI publication, Windows staged pthread/POSIX portability, and reviewed Windows install-validation parity. |
| Numerical corpus expansion | R5, R6, R12 | Broader QR and partial-SVD fixture families plus external-library comparison semantics. |
| Shared-library and ABI productization | R4, R14 | Shared install/export, ABI policy, loader tests, symbol checks, package metadata, and package-manager distribution. |
| Report evidence refresh | R1, R7 | Branch/PR hosted evidence and selected generated benchmark, sentinel, coverage, dead-code, and guardrail freshness gates. |
| Adoption/documentation completion | R8, R9 | Tutorial alignment and broader public-header cleanup without widened support claims. |
| Runtime/backend follow-through | R10, R11 | Typed-control promotion review and additional sentinel rows without portable performance wording. |
| Competitive positioning | R12, R13 | State-of-the-art or external-parity decision only after direct comparative evidence exists. |

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

## What Went Well

1. **The epic closed selected gaps instead of spreading effort thinly.** QR and
   partial-SVD work ended with named fixture-local closures and explicit
   residuals for broader coverage.

2. **Evidence boundaries stayed visible.** Corpus rows, generated report
   outputs, CI lanes, package support, platform tiers, and adoption docs kept
   separate proof meanings.

3. **Static-first packaging became executable product truth.** Make install,
   `pkg-config`, CMake install/export, version checks, downstream consumers,
   and shared-library deferral are now guarded by tests and documentation.

4. **Platform language became more precise.** Linux, macOS, and Windows lanes
   now have clearer reviewed, supplemental, staged, deferred, local-only, and
   hosted-only meanings.

5. **Adoption work improved the front door without inflating claims.** The
   README, INSTALL, examples, cookbook, solver-selection docs, and selected
   public headers now route first-use workflows more cleanly while preserving
   proof boundaries.

6. **The final closeout rejected unsupported competitive claims.**
   State-of-the-art status remained a non-claim because the epic did not
   produce direct comparative evidence.

## What Didn't Go Well

1. **Broad numerical coverage remains large.** The selected QR and partial-SVD
   residuals closed, but broad solver-family parity still requires a larger
   corpus and comparison program.

2. **Hosted branch evidence is still external to final local closeout.** Sprint
   146 could inspect green `master` hosted baselines, but branch-specific
   Sprint 146 hosted CI remains a residual until PR workflows run.

3. **Windows parity remains incomplete.** Windows CMake-first support improved,
   but staged pthread/POSIX tests, reviewed install-validation parity, Makefile
   parity, and `pkg-config` parity remain unpromoted.

4. **Generated report freshness is intentionally selective.** Source-controlled
   rows and local generated artifacts are coherent, but broad generated report
   refreshes should be tied to concrete claims.

5. **Adoption work did not finish every documentation surface.** The high-value
   first-use path improved, but tutorial alignment and all-header cleanup
   remain future work.

6. **Shared-library and package-manager work remained product decisions, not
   implementation work.** Static-first support is stronger, but dynamic ABI,
   runtime-loader behavior, and package-manager distribution need a dedicated
   productization effort.

## Key Evidence Package

- [SPRINT_137/RETROSPECTIVE.md](./SPRINT_137/RETROSPECTIVE.md)
- [SPRINT_138/RETROSPECTIVE.md](./SPRINT_138/RETROSPECTIVE.md)
- [SPRINT_139/RETROSPECTIVE.md](./SPRINT_139/RETROSPECTIVE.md)
- [SPRINT_140/RETROSPECTIVE.md](./SPRINT_140/RETROSPECTIVE.md)
- [SPRINT_141/RETROSPECTIVE.md](./SPRINT_141/RETROSPECTIVE.md)
- [SPRINT_142/RETROSPECTIVE.md](./SPRINT_142/RETROSPECTIVE.md)
- [SPRINT_143/RETROSPECTIVE.md](./SPRINT_143/RETROSPECTIVE.md)
- [SPRINT_144/RETROSPECTIVE.md](./SPRINT_144/RETROSPECTIVE.md)
- [SPRINT_145/RETROSPECTIVE.md](./SPRINT_145/RETROSPECTIVE.md)
- [SPRINT_146/RETROSPECTIVE.md](./SPRINT_146/RETROSPECTIVE.md)
- [SPRINT_146/artifacts/day5-final-local-validation-command-log.md](./SPRINT_146/artifacts/day5-final-local-validation-command-log.md)
- [SPRINT_146/artifacts/day8-public-claim-audit.md](./SPRINT_146/artifacts/day8-public-claim-audit.md)
- [SPRINT_146/artifacts/day9-support-maintainer-claim-audit.md](./SPRINT_146/artifacts/day9-support-maintainer-claim-audit.md)
- [SPRINT_146/artifacts/day11-published-residual-queue.md](./SPRINT_146/artifacts/day11-published-residual-queue.md)
- [SPRINT_146/artifacts/day13-final-project-plan-reconciliation.md](./SPRINT_146/artifacts/day13-final-project-plan-reconciliation.md)
- [SPRINT_146/artifacts/day14-final-closeout-package.md](./SPRINT_146/artifacts/day14-final-closeout-package.md)

## Closeout Assessment

Epic 12 is complete. It materially improved evidence ownership, fixture-local
numerical proof, report governance, runtime/backend support boundaries,
static-first package confidence, platform-tier clarity, adoption flow, claim
discipline, and residual transparency.

It deliberately leaves broad competitive, numerical, package-manager,
shared-library ABI, Windows parity, generated-report freshness, and portable
performance claims as explicit non-claims or future work.
