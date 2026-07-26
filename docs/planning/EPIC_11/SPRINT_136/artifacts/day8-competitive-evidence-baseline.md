# Sprint 136 Day 8 - Competitive Evidence Baseline

## Purpose

Day 8 compares final Epic 11 evidence against the Epic 11 goals,
state-of-the-art target language, and explicit non-claims.

This artifact is the input to Day 9 claim recalibration. It classifies what is
earned, local-only, supplemental, deferred, or unsupported before any public or
support wording is edited.

## Epic 11 Goal Baseline

The Epic 11 project plan says the library is broad, well-tested, and more
product-disciplined after Epic 10, but still should not claim unqualified
state-of-the-art status.

Epic 11's intended maturity gains are:

- reduce remaining source/test ownership risk;
- widen numerical oracle architecture;
- improve local performance governance;
- make package, ABI, and platform decisions;
- simplify the adoption surface.

The final comparison must therefore distinguish product maturity and bounded
evidence from broad competitive superiority.

## Goal-To-Evidence Comparison

| Epic 11 goal | Final evidence | Classification | Claim impact |
| --- | --- | --- | --- |
| Reduce source/test ownership risk | Sprints 119-130 moved or clarified eigensolver, direct/iterative, QR, SVD, partial-SVD, residual, and helper ownership; Day 6 local CMake build and 57/57 CTest pass validated current local registration/execution. | Earned internal/local evidence | Supports internal closeout wording that ownership and local validation improved; does not imply complete solver-family or platform coverage. |
| Widen numerical oracle architecture | Direct, iterative, LU, Cholesky, LDLT, QR, SVD, partial-SVD, and cross-solver oracle/helper surfaces are inventoried with bounded external-reference fixtures. | Earned bounded evidence | Supports named fixture and helper evidence; does not support broad LAPACK, NumPy, SciPy, SuiteSparse, backend, or ecosystem parity. |
| Improve local performance governance | Sprint 132 governance plus Day 7 canonical, sentinel, and large-matrix report generation with fresh manifests and row counts. | Local-only/freshness-scoped evidence | Supports local report freshness and governance; does not support portable performance, scalability, memory, or state-of-the-art claims. |
| Make package/ABI decisions | Sprint 133 static-first package decision; Day 5 static deferral proof; Day 6 CMake install/export proof; Day 7 Make install/`pkg-config` proof. | Earned static-first evidence | Supports maintained static-first package/install/export story; shared-library, dynamic ABI, runtime-loader, and package-manager support remain deferred. |
| Make platform decisions | Sprint 134 platform-tier closeout; Day 6 local CMake/CTest evidence; Day 7 local install proof. | Mixed reviewed/local/supplemental evidence | Supports tiered platform wording only: Linux reviewed package CI owner, macOS/Windows supplemental package confidence, Windows staged tests. |
| Simplify adoption surface | Sprint 135 adoption docs, cookbook, algorithm/history split, report-index discovery, and Day 5-7 validation package. | Earned documentation/navigation evidence | Supports clearer adoption and documentation ownership; does not imply new solver behavior, package support, report schema, or platform parity. |
| Close Epic 11 with residual clarity | Day 2 residual grouping and Sprint 136 deferred QR residual plan. | Deferred/residual evidence | Supports publication of future work with blockers and promotion criteria, not implementation closure. |

## Claim Classification

| Claim class | Classification | Evidence basis | Boundary |
| --- | --- | --- | --- |
| Local source/test validation is healthy at closeout | Earned local | Day 6 CMake build and 57/57 CTest pass; Day 5 source-list check; no C/header changes in Sprint 136. | Local platform evidence only; hosted CI still required for hosted claims. |
| Solver evidence breadth improved across Epic 11 | Earned bounded | Sprint 119-130 artifacts and Day 2 inventory. | Evidence is owner- and fixture-specific; no complete-family or external parity claim. |
| External-reference helper architecture is broader | Earned bounded | Direct, LU, Cholesky, LDLT, QR, SVD, and cross-solver helper surfaces. | Helper outputs do not imply broad dense-library or ecosystem parity. |
| Partial-SVD residual/subspace/optimality evidence supports broader public solver-selection wording | Deferred/unsupported for public wording | Sprint 130 accepted fixture-bounded evidence and explicitly made no public solver-selection update. | Keep public solver-selection workflow-oriented until broader evidence lands. |
| Local benchmark/report evidence is fresh | Earned local | Day 7 generated reports at commit `b178de48` on branch `sprint-136`. | Freshness metadata is not CI, release, performance, or correctness proof. |
| Portable performance or scalability improved | Unsupported | No multi-host, statistical, memory, variance, or platform matrix evidence. | Keep as explicit non-claim. |
| Static-first package/install/export support is maintained | Earned bounded | Day 5 static deferral proof; Day 6 CMake install/export proof; Day 7 Make install/`pkg-config` proof; Sprint 133 decision. | Static archive surface only. |
| Shared-library packaging or dynamic ABI compatibility exists | Deferred/unsupported | Sprint 133 intentionally deferred; static deferral proof confirms boundaries. | Do not claim. |
| Package-manager support exists | Deferred/unsupported | No package-manager recipes or install/upgrade validation. | Do not claim Homebrew, apt/deb, rpm, pacman, vcpkg, Conan, or similar support. |
| Linux reviewed package-contract CI owns package proof | Earned hosted-reviewed owner | Sprint 134 workflow decision. | Actual hosted pass is branch/PR CI evidence, not local proof. |
| macOS package install/export parity is reviewed | Unsupported | Sprint 134 keeps macOS install/export confidence supplemental. | Do not promote. |
| Windows install validation parity is reviewed | Unsupported | Sprint 134 keeps Windows install/downstream confidence supplemental and staged tests excluded. | Do not promote. |
| Adoption surface is clearer and productized | Earned documentation | Sprint 135 docs and Day 2 adoption inventory. | Documentation clarity is not new behavior or support expansion. |
| Generated report indexes are normalized cross-report proof | Deferred/unsupported | Sprint 131 accepted generated report indexes with row-family boundaries. | Do not claim normalized cross-report schema or broad report completeness. |
| Epic 11 achieved unqualified state-of-the-art status | Unsupported | Project-plan baseline rejects unqualified state-of-the-art claim; final evidence is bounded/local/tiered. | Keep as explicit non-claim. |

## Competitive Evidence Gap List

| Gap | Current evidence | Needed for promotion |
| --- | --- | --- |
| Unqualified state-of-the-art claim | Broad local tests and product discipline, but no external competitive study. | Independent competitive matrix, external baselines, statistical performance policy, correctness oracle breadth, and public claim gate. |
| External solver/library parity | Bounded helper fixtures and cross-checks. | Solver-family-specific external oracle matrix with support tiers, tolerances, fixture provenance, and failure semantics. |
| Portable performance | Local benchmark/sentinel/guardrail reports. | Multi-host, multi-compiler, repeat-count, variance, backend, OpenMP, and memory policy. |
| Platform parity | Linux reviewed package CI owner, macOS/Windows supplemental confidence. | Hosted runner history, failure triage policy, platform-specific proof, and explicit promotion decision. |
| Shared-library/dynamic ABI support | Static-first support and deferral proof. | Shared-library product decision, symbol/export policy, ABI epoch, soname/install-name/runtime-loader tests, and docs. |
| Package-manager support | Static install metadata only. | Manager-specific recipes, dependency metadata, install roots, upgrade/uninstall proof, and platform support. |
| Broad SuiteSparse/corpus proof | Named fixtures, generated report taxonomy, and residual queues. | Independent metadata, oracle provenance, expected ranks/properties, skip behavior, runtime budget, and support tier per corpus row. |
| Normalized cross-report indexing | Report-family-specific indexes and manifests. | Common schema that preserves row meanings, status semantics, support tiers, and freshness policy. |
| Partial-SVD public solver-selection expansion | Fixture-bounded residual, subspace, optimality, and convergence evidence. | Broader corpus and semantics for repeated spectra, clustered spectra, rank-deficient null-space, convergence diagnostics, and sparse-output optimality. |
| QR deferred residual closure | End-of-epic residual queue with blockers and promotion criteria. | Future owner implementation after metadata, semantics, runtime, diagnostics, and validation are pinned. |

## State-Of-The-Art Non-Claim Register

Sprint 136 should preserve these non-claims:

- no unqualified state-of-the-art claim;
- no broad ecosystem replacement claim;
- no broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  BLAS, or vendor-backend parity claim;
- no every-solver-family external oracle coverage claim;
- no portable performance, scalability, memory, or runtime superiority claim;
- no universal reorder/fill superiority claim;
- no normalized cross-report correctness, coverage, or release proof claim;
- no broad SuiteSparse corpus or optional-data coverage claim;
- no broad QR, SVD, partial-SVD, minimum-norm, nullspace, subspace,
  low-rank-optimality, convergence-rate, or partial-result claim beyond named
  bounded fixtures;
- no shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, or package-manager support claim;
- no equal Linux/macOS/Windows reviewed support claim;
- no reviewed macOS install/export parity claim;
- no reviewed Windows install-validation parity claim;
- no Windows pthread/POSIX staged-test promotion claim;
- no claim that adoption docs create new behavior, package support, report
  schemas, platform support, or performance evidence.

## Day 9 Recalibration Inputs

Day 9 should convert this baseline into final claim decisions:

1. Public claims may say Epic 11 strengthened evidence ownership, validation
   discipline, static-first package confidence, report freshness, and adoption
   navigation.
2. Public claims must keep solver and oracle wording fixture-bounded and
   workflow-oriented.
3. Package wording may continue to describe static-first install/export and
   downstream `pkg-config`/CMake consumers.
4. Platform wording must preserve reviewed Linux package CI, supplemental
   macOS/Windows package confidence, and staged Windows tests.
5. Benchmark/report wording must remain local, freshness-scoped, and
   non-portable.
6. State-of-the-art and ecosystem parity wording should remain explicit
   non-claims.
7. Deferred QR and corpus/report residuals should feed Day 12 publication, not
   Day 9 positive claims.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every major Epic 11 claim class has evidence classification. | Complete | Goal-to-evidence table and claim classification table classify source/test, oracle, reports, package, platform, docs, residuals, and competitive status. |
| State-of-the-art language is compared against actual evidence, not intent. | Complete | Project-plan baseline and state-of-the-art non-claim register preserve explicit non-claims. |
| Unsupported or overbroad claims are queued for cleanup. | Complete | Competitive gap list and Day 9 recalibration inputs identify public wording boundaries for Days 9-11. |
