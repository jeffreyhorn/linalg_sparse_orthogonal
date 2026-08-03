# Sprint 136 Day 9 - Competitive Claim Recalibration

## Purpose

Day 9 converts the Day 8 competitive evidence baseline into final claim
decisions. It decides which wording may appear publicly, which evidence should
remain maintainer/local, which claims are supplemental or deferred, and which
claims must remain explicit non-claims.

No public documentation is edited on Day 9. Day 10 audits public/support
surfaces against this decision package, and Day 11 performs cleanup if needed.

## Final Claim Decision Table

| Claim area | Decision | Approved owner surfaces | Evidence basis | Boundary |
| --- | --- | --- | --- | --- |
| Epic 11 maturity | May claim Epic 11 strengthened maturity through ownership, validation, package decisions, report governance, and adoption navigation. | Sprint 136 closeout, Epic 11 retrospective, maintainer docs if needed. | Sprints 118-136 artifacts and Day 5-7 validation. | Do not phrase as state-of-the-art superiority. |
| Source/test ownership | May claim source/test ownership and local validation discipline improved. | Planning closeout, maintainer guide if a future edit is needed. | Sprints 119-130 ownership work; Day 5 source-list; Day 6 CMake/CTest. | Do not claim complete solver-family coverage or platform parity. |
| Local CMake/test status | May claim local CMake build, 57-test registration, and 57/57 CTest pass for Sprint 136 validation. | Sprint 136 validation package and closeout. | Day 6 validation. | Local AppleClang/Darwin evidence only; not hosted CI parity. |
| Oracle/helper breadth | May claim broader bounded oracle/helper architecture. | Maintainer docs and closeout. | Day 2 evidence inventory and Sprint 120-130 artifacts. | Keep fixture/helper-specific; no broad external-library parity. |
| Public solver-selection wording | Keep existing workflow-oriented public guidance unless Day 10 finds drift. | Public solver-selection docs, README, cookbook. | Sprint 130 claim closeout and Day 8 classification. | No broader partial-SVD, QR, SVD, minimum-norm, corpus, convergence, or superiority claim. |
| Static-first package support | May claim maintained static-first install/export and downstream `pkg-config`/CMake consumer support. | README, INSTALL, maintainer guide, Sprint 136 closeout. | Day 5 static deferral proof; Day 6 CMake install proof; Day 7 Make install proof; Sprint 133 decision. | Static archive surface only. |
| Shared-library/dynamic ABI/package-manager support | Must remain explicit non-claims. | README, INSTALL, maintainer guide, closeout non-claim register. | Sprint 133 decision and Day 5 static deferral proof. | No shared library, ABI stability, runtime-loader, package-manager, or static/shared selector wording. |
| Platform support | May claim tiered platform support. | README, INSTALL, maintainer guide, closeout. | Sprint 134 platform truth and Day 6 local CMake evidence. | Linux package CI reviewed owner; macOS/Windows package confidence supplemental; Windows staged tests remain staged. |
| Benchmark/report freshness | May claim local generated report freshness and row counts for Sprint 136 validation. | Sprint 136 validation package and closeout; benchmark docs only if later edited. | Day 7 generated report metadata. | No portable performance, scalability, memory, release, or correctness claim. |
| Performance governance | May claim local performance governance improved. | Sprint 136 closeout, maintainer docs if needed. | Sprint 132 governance and Day 7 reports. | No speed superiority, backend parity, universal reorder/fill, or state-of-the-art performance wording. |
| Adoption documentation | May claim adoption/navigation docs were productized. | Sprint 136 closeout and public docs if later summarized. | Sprint 135 adoption closeout and Day 2 inventory. | Documentation clarity does not imply behavior, package, report-schema, or platform expansion. |
| Residual queue | May claim residuals are classified and queued with promotion criteria after Day 12. | Sprint 136 residual queue and closeout. | Day 2 grouping; Day 12 planned publication. | Do not claim deferred QR/corpus/report residuals are implemented or closed. |
| Competitive positioning | Must be bounded: stronger maturity and validation discipline, no unqualified state-of-the-art claim. | Sprint 136 closeout and Epic 11 retrospective. | Day 8 baseline. | No broad ecosystem replacement, external parity, or superiority claim. |

## Earned Claim Register

These claims are earned if phrased with their boundaries:

| Earned claim | Required qualifier |
| --- | --- |
| Epic 11 strengthened product discipline. | Tie to evidence ownership, validation, package/platform decisions, report governance, adoption docs, and residual queues. |
| Local validation package passed for Sprint 136. | Cite Day 5-7 exact commands and local/platform context. |
| CMake local build and CTest execution passed. | State 57/57 local tests on Darwin/AppleClang; do not imply hosted platform parity. |
| Static-first package proof passed locally. | State static archive install/export, `pkg-config`, and CMake installed consumers; no shared/dynamic/package-manager claims. |
| Generated benchmark/report metadata is fresh for Sprint 136 validation. | State branch `sprint-136`, commit `b178de48`, local platform/compiler, row counts, and report boundaries. |
| Adoption surfaces are clearer after Sprint 135. | State documentation/navigation improvement only. |
| Solver/oracle evidence is broader than the Epic 10 baseline. | State fixture/helper/owner boundaries and maintainer-facing evidence map. |

## Local-Only And Supplemental Claim Register

| Claim | Status | Required wording |
| --- | --- | --- |
| Local CMake/CTest confidence | Local-only | "Local CMake/CTest validation passed" with platform context. |
| Canonical benchmark report rows | Local-only | "Threshold-free local measurement snapshot." |
| Performance sentinel report rows | Local-only | "Local wall-check/report bundle; only existing wall-check rows are thresholded." |
| Large-matrix guardrail rows | Mixed reviewed/supplemental | "Four reviewed structural/report rows passed; two supplemental opt-in rows skipped." |
| macOS package install/export confidence | Supplemental | "Supplemental confidence" only. |
| Windows install/downstream confidence | Supplemental | "Supplemental CMake-first downstream confidence" only. |
| Coverage | Supplemental/deferred | Do not mention as final Day 8-9 evidence unless coverage is explicitly run later. |
| Dead-code | Report-completeness/deferred | Do not claim zero findings or removal readiness. |

## Non-Claim Register

The following must remain explicit non-claims or absent from positive wording:

- unqualified state-of-the-art status;
- broad ecosystem replacement;
- broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  BLAS, or vendor-backend parity;
- every-solver-family external oracle coverage;
- portable performance, scalability, memory, runtime, or speed superiority;
- universal reorder/fill superiority;
- normalized cross-report correctness, coverage, release, or performance
  proof;
- broad SuiteSparse corpus or optional-data coverage;
- broad QR, SVD, partial-SVD, minimum-norm, nullspace, subspace,
  low-rank-optimality, convergence-rate, achieved-tolerance, iteration-count,
  stagnation, or partial-result claim beyond named bounded fixtures;
- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager support;
- static/shared package selectors;
- equal Linux/macOS/Windows reviewed support;
- reviewed macOS install/export parity;
- reviewed Windows install-validation parity;
- Windows Makefile parity;
- Windows staged pthread/POSIX test promotion;
- public solver-selection expansion beyond workflow guidance;
- claim that adoption docs create new solver behavior, package support,
  report schemas, platform support, or performance evidence.

## Day 10 Unsupported-Claim Audit Queue

Day 10 should audit these surfaces and wording families:

| Surface | Wording to audit |
| --- | --- |
| `README.md` | State-of-the-art, parity, CI/platform tiers, package support, benchmark/report wording, generated report rows, adoption-map summaries. |
| `INSTALL.md` | Static-first package wording, shared-library/dynamic ABI/package-manager non-claims, platform support tiers, install proof language. |
| `docs/solver_selection.md` | Solver superiority, partial-SVD expansion, QR/SVD/minimum-norm/public workflow claims, benchmark comparison wording. |
| `docs/cookbook.md` | Compressed-first workflow wording that could imply behavior or performance expansion. |
| `docs/algorithm.md` | Algorithm-reference statements that could read as broad parity, optimality, or performance claims. |
| `docs/algorithm_history.md` | Historical measurement wording that could be mistaken for current portable performance. |
| `docs/maintainer_guide.md` | Support tiers, oracle/helper evidence, package/platform truth, report-index boundaries, non-claim registers. |
| `benchmarks/README.md` | Portable performance, scalability, memory, threshold, benchmark superiority, report-index wording. |
| `examples/README.md` | Example coverage or workflow wording that could imply full family support. |
| Sprint 136 artifacts | Ensure closeout artifacts themselves do not convert local/supplemental evidence into broad claims. |

## Competitive Recalibration Summary

Final Epic 11 positioning should be:

> Epic 11 materially improved product discipline, evidence ownership, local
> validation, static-first packaging confidence, report governance, platform
> tier clarity, adoption navigation, and residual transparency. It does not
> establish unqualified state-of-the-art status, broad ecosystem parity,
> portable performance, shared-library/dynamic ABI support, package-manager
> support, or equal reviewed platform parity.

This sentence is suitable as the Day 14 closeout posture if Day 10-11 audits
do not find conflicting public/support wording.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public/support wording decisions have evidence and owner surfaces. | Complete | Final claim decision table maps decisions to owner surfaces and evidence. |
| Unsupported or ambiguous claims have a cleanup path. | Complete | Day 10 unsupported-claim audit queue names surfaces and wording families. |
| Competitive comparison language is bounded and defensible. | Complete | Competitive recalibration summary preserves maturity gains and explicit non-claims. |
