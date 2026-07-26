# Epic 11 Retrospective

**Epic:** 11 - Final Evidence Ownership, Product Truth, Package/Platform
Support, Adoption Surface, and Closeout
**Sprints:** 118-136
**Status:** Complete

## Epic Objective

Epic 11 converted inherited residual debt into owned implementation,
validation, documentation, package, platform, performance, and claim-boundary
work. Its central goal was not to claim broad market or state-of-the-art
superiority. It was to make the project's evidence, support tiers, install
surface, report interpretation, adoption docs, and remaining residuals clear
enough to close the epic honestly.

## Major Outcomes

| Area | Outcome |
|---|---|
| Source/test ownership | Eigensolver, direct, iterative, QR, SVD, partial-SVD, graph/reorder, helper, and validation owners were inventoried and repeatedly reconciled. |
| Oracle/helper evidence | External-reference helper evidence expanded while staying fixture-specific and output-class-specific. |
| QR/SVD/partial-SVD residuals | Multiple bounded QR, minimum-norm, nullspace/subspace, SVD, and partial-SVD lanes landed or were explicitly deferred with promotion gates. |
| Corpus/report architecture | Corpus taxonomy, report-index requirements, coverage/dead-code architecture, freshness policy, and generated report interpretation were documented. |
| Runtime/performance governance | Benchmark and sentinel rows gained support-tier and local-measurement interpretation without becoming portable performance claims. |
| Package/ABI decision | Static-first install/export became the maintained package shape; shared-library, dynamic ABI, runtime-loader, and package-manager support remain deferred. |
| Platform tiers | Linux package-contract ownership, macOS supplemental confidence, Windows supplemental CMake-first confidence, and Windows staged exclusions are explicit. |
| Adoption surface | README, tutorial, cookbook, examples, algorithm docs, benchmark docs, install docs, and maintainer docs now have clearer owner roles. |
| Final closeout | Sprint 136 validated local evidence, recalibrated claims, confirmed no unsupported public wording cleanup was needed, and published residual queues. |

## Validation Evidence

Epic 11 used sprint-local validation matched to touched surfaces. The final
Sprint 136 validation package provides the closeout snapshot:

| Evidence | Result | Boundary |
|---|---|---|
| Package/static checks | Passed on Day 5, including static deferral proof. | Supports static-first deferral boundary only. |
| Local CMake configure/build | Passed on Day 6. | Local Darwin/AppleClang confidence only. |
| Local CTest | 57/57 passed on Day 6. | Does not promote hosted platform parity. |
| CMake install/export proof | 21 checks passed, 0 failures, 0 skips. | Static installed CMake consumer proof only. |
| Canonical benchmark report | 4 rows generated on Day 7. | Threshold-free local measurement snapshot. |
| Performance sentinels | 11 rows generated on Day 7. | Local wall-check/report evidence only. |
| Large-matrix guardrails | 6 rows generated on Day 7; 4 reviewed pass, 2 supplemental skip. | Bounded structural/report evidence. |
| Make install/`pkg-config` proof | 22 checks passed, 0 failures. | Static package route only. |
| Day 10-11 claim audit/cleanup | Passed; no P0 public-doc cleanup required. | Confirms wording is bounded, not that deferred claims are earned. |
| Day 12 residual publication | Complete. | Future work remains future work. |

## Earned Claims

The following claims are earned only with their qualifiers:

- Epic 11 strengthened product discipline, evidence ownership, validation
  discipline, package decisions, platform-tier clarity, report governance,
  adoption navigation, and residual transparency.
- Local Sprint 136 validation passed for the selected local command package.
- The maintained package surface is static-first, with Make install,
  `pkg-config`, CMake install/export, and downstream consumer proof.
- Linux is the strongest reviewed package-contract owner; macOS and Windows
  package/install confidence remains supplemental where documented.
- Generated report metadata is fresh for the Day 7 Sprint 136 branch/commit
  context.
- Solver/oracle evidence is broader and better owned than the Epic 10
  baseline, but remains fixture/helper-specific.
- Adoption documentation is clearer and better routed after Sprint 135.

## Non-Claims

Epic 11 does not claim:

- unqualified state-of-the-art status;
- broad ecosystem replacement or external-library parity;
- every-solver-family external oracle coverage;
- broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  dense-library parity;
- broad QR, SVD, partial-SVD, minimum-norm, nullspace, subspace, corpus,
  optional-data, convergence-rate, sparse-output, or partial-result coverage;
- portable performance, scalability, memory, runtime, OpenMP speedup, backend
  parity, optional-backend availability, or universal reorder/fill
  superiority;
- coverage percentage as reviewed behavioral completeness;
- dead-code output as removal-ready proof;
- generated report/index rows as broad correctness, release, coverage, or
  performance proof;
- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager support;
- reviewed macOS install/export parity;
- reviewed Windows install-validation parity;
- Windows staged pthread/POSIX test promotion.

## Residuals And Future-Epic Candidates

The post-Epic-11 residual queue is published in
[day12-residual-queue-publication.md](./artifacts/day12-residual-queue-publication.md).
The highest-value future-epic candidates are:

- QR residual and SuiteSparse/corpus expansion;
- partial-SVD edge-case residual and convergence-budget expansion;
- corpus/report normalized indexing with row-meaning preservation;
- runtime/backend sentinel expansion and stale-report automation;
- shared-library, ABI, runtime-loader, and package-manager productization;
- platform promotion for macOS/Windows package install/export confidence;
- Windows staged pthread/POSIX test portability and promotion.

## What Went Well

1. **Epic 11 turned residual debt into owned decisions.**
   Even when work stayed deferred, it ended with a reason, owner surface,
   blocker, support tier, and promotion criteria.

2. **The static-first package decision became enforceable.**
   The project now rejects `BUILD_SHARED_LIBS=ON` under the maintained static
   contract and has package proofs that guard the selected support surface.

3. **Public adoption docs became more coherent.**
   Sprint 135 produced a cookbook, algorithm/history split, report-index
   handoff, and navigation alignment without implying new behavior.

4. **Final closeout kept support tiers separate.**
   Sprint 136 preserved local, reviewed, supplemental, staged, deferred, and
   unsupported evidence rather than flattening them into broad claims.

## What Didn't Go Well

1. **Some residual families are still large.**
   QR corpus expansion, partial-SVD edge cases, report normalization, and
   platform promotion each need dedicated future planning.

2. **Report metadata is useful but still fragmented.**
   Canonical, sentinel, guardrail, coverage, dead-code, and oracle-helper rows
   do not yet share a safe normalized schema.

3. **Hosted-platform confidence remains tiered.**
   macOS and Windows package/install confidence improved, but the project did
   not earn equal reviewed package parity across platforms.

4. **State-of-the-art comparison remains intentionally unclaimed.**
   Epic 11 improved discipline and maturity, but did not run the external,
   statistical, cross-platform, cross-library comparison needed for broad
   competitive superiority.

## Closeout Assessment

Epic 11 is complete. It materially improved the project's maturity,
evidence ownership, local validation, static-first packaging confidence,
report governance, platform tier clarity, adoption navigation, and residual
transparency. It deliberately leaves broad competitive, platform, package,
runtime, ABI, corpus, and performance claims as explicit non-claims or future
work.
