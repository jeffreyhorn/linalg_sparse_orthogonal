# Sprint 147 Day 13 Public Claim Freeze Audit

## Purpose

Day 13 re-audits public and support-facing wording before Epic 13 implementation
begins. The audit uses the Day 12 quality map and freezes the current claim
baseline so Sprints 148-156 can add evidence without accidentally widening
platform, package, performance, corpus, ABI, external-comparison, or
state-of-practice language.

No source documentation fixes were applied on Day 13. The scanned surfaces
already use bounded wording, explicit non-claims, or residual language for the
claim-sensitive areas in scope.

## Scan Scope

| Surface | Files scanned | Reason |
| --- | --- | --- |
| Project front door | `README.md` | First public capability and workflow claims. |
| Install/package support | `INSTALL.md` | Static-first package, platform, downstream consumer, and non-claim wording. |
| Maintainer policy | `docs/maintainer_guide.md` | Reviewed baseline, support-tier, report, package, and evidence-owner interpretation. |
| Solver selection | `docs/solver_selection.md` | User-facing solver guidance and QR/SVD evidence boundaries. |
| Cookbook | `docs/cookbook.md` | Data-first adoption wording and benchmark/report handoffs. |
| Tutorial | `docs/tutorial.md` | Longer workflow guidance and partial-SVD explanation. |
| Benchmarks | `benchmarks/README.md` | Performance, sentinel, generated report, and benchmark interpretation claims. |
| Public headers | `include/*.h` | API-local caveats and claim-sensitive comments. |

Primary scan terms:

```text
state-of-the-art, external parity, external-library parity, package-manager,
Homebrew, apt, dnf, pacman, vcpkg, conan, shared-library, dynamic ABI,
Windows parity, portable performance, speedup guarantee, freshness,
generated report, benchmark, unsupported, deferred, non-claim
```

## Claim Classification Table

| Claim area | Current classification | Evidence or wording baseline | Day 13 action |
| --- | --- | --- | --- |
| State-of-the-art | Explicit non-claim | README, solver-selection, cookbook, tutorial, maintainer guide, and benchmark docs reject broad state-of-the-art wording unless bounded evidence exists. | No edit. Keep rejected wording frozen for implementation sprints. |
| External-library parity | Explicit non-claim with bounded fixture evidence | QR/SVD docs allow named fixture-local evidence and reject broad LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, or ecosystem parity. | No edit. Sprint 154 may add only a named dependency/version/fixture/metric claim. |
| Shared-library ABI | Explicit non-claim | README and INSTALL state shared-library packaging, dynamic ABI compatibility, and runtime-loader behavior are deferred; CMake rejects `BUILD_SHARED_LIBS=ON`. | No edit. Sprint 153 must choose implementation or strengthened static-first deferral before wording changes. |
| Package-manager distribution | Explicit non-claim | INSTALL and README separate maintained static install/export proof from package-manager distribution. | No edit. Do not add Homebrew/apt/dnf/pacman/vcpkg/conan support wording without recipe ownership and install proof. |
| Windows platform parity | Explicit tier boundary | INSTALL and maintainer guide distinguish reviewed Windows CMake subset, supplemental CMake install/downstream confidence, staged pthread/POSIX tests, and unsupported Windows Makefile/`pkg-config` parity. | No edit. Sprint 148-149 must preserve reviewed versus supplemental language. |
| Performance | Explicit local-measurement boundary | README, solver-selection, cookbook, headers, and benchmarks docs describe benchmark/sentinel rows as local, configuration-sensitive evidence, not portable guarantees. | No edit. Any new performance wording needs benchmark command, environment, and claim scope. |
| Generated report freshness | Explicit navigation/freshness boundary | README, cookbook, benchmarks docs, INSTALL, and maintainer guide treat normalized reports as freshness/navigation context, not pass evidence unless a selected gate requires generated rows. | No edit. Sprint 152 must choose required-generated rows before freshness wording can widen. |
| QR corpus proof | Supported narrow claim | README, solver-selection, cookbook, and maintainer guide identify `qr_rank_deficient_6x4_nullspace_v1` as fixture-local rank/nullity/nullspace evidence. | No edit. Sprint 150 can widen only through maintained corpus rows plus proof-owner tests. |
| Partial-SVD corpus proof | Supported narrow claim | README, tutorial, solver-selection, `include/sparse_svd.h`, and maintainer guide identify the generated 8x6 clustered/repeated fixture as fixture-local evidence. | No edit. Sprint 151 can widen only through maintained corpus rows plus generated oracle/freshness evidence. |
| Static package install/export | Supported narrow claim | INSTALL and README describe Make install, CMake install/export, `pkg-config`, `find_package(Sparse)`, and exact-version metadata as static archive proof. | No edit. Keep static-first wording unless Sprint 153 implements shared support. |
| Public header caveats | Supported or explicit caveat | Header comments for SVD, Cholesky threshold, eigensolver ABI breaks, callback behavior, and storage interop use API-local boundaries rather than broad product claims. | No edit. Future header updates must stay API-local. |

## Wording Fix List

No documentation wording fixes were applied.

Rationale:

- Claim-sensitive public docs already distinguish supported evidence from
  non-claims.
- The package wording stays static-first and does not imply dynamic ABI or
  package-manager support.
- Windows wording stays tiered and does not imply Makefile, `pkg-config`, or
  broad platform parity.
- Benchmark and generated-report wording stays local, advisory, or
  freshness-scoped.
- QR and partial-SVD wording names fixture-local evidence instead of broad
  solver or external-library parity.

## Implementation-Sprint Claim Warnings

Sprints 148-156 must not cross these boundaries without direct evidence:

| Future sprint | Warning |
| --- | --- |
| Sprint 148 | Promoting a Windows staged test requires source portability, CMake registration, hosted MSVC execution, expected-count updates, and docs/report wording that names only the promoted lane. |
| Sprint 149 | Windows install/downstream work must not become a reviewed install-validation parity claim unless the workflow and docs explicitly promote that support tier. |
| Sprint 150 | QR corpus expansion must not imply raw QR basis identity, global rank-threshold policy, broad rank-deficient solve support, platform proof, performance proof, or external-library parity. |
| Sprint 151 | Partial-SVD corpus expansion must not imply broad repeated-spectrum handling, convergence-rate guarantees, raw singular-vector identity, partial-result guarantees, platform proof, performance proof, or external-library parity. |
| Sprint 152 | Generated report freshness checks must not convert advisory rows into pass evidence. Required-generated rows need explicit family selection and freshness metadata. |
| Sprint 153 | Shared-library, dynamic ABI, loader, selector, or package-manager wording is blocked until the product decision and executable install/export/downstream proof exist. |
| Sprint 154 | External comparison wording must name the dependency, version, installation method, fixture set, metric, tolerance, platform, and support tier. |
| Sprint 155 | Documentation productization must preserve explicit non-claims while making supported routes easier to find. |
| Sprint 156 | Final validation wording must report the exact evidence gathered and keep residuals separate from completed claims. |

## Sprint 148-156 Wording Baseline

Use these phrases or equivalent bounded language when implementation sprints
touch public/support surfaces:

| Topic | Allowed baseline | Avoid |
| --- | --- | --- |
| QR corpus | "fixture-local QR evidence for named maintained corpus rows" | "broad QR parity", "global rank-threshold policy", "state-of-the-art QR" |
| Partial SVD | "fixture-local partial-SVD evidence for named generated/corpus rows" | "broad repeated-spectrum support", "external-library parity", "partial-result guarantee" |
| Performance | "local measurement on recorded environment and command" | "portable speedup", "faster across platforms", "benchmark-proven superiority" |
| Windows | "reviewed CMake subset" or "supplemental CMake install/downstream confidence" | "Windows parity", "Windows Makefile support", "Windows `pkg-config` support" |
| Package | "maintained static archive package surface" | "shared library support", "dynamic ABI compatibility", "package-manager support" |
| Reports | "freshness/navigation diagnostics for selected rows" | "release proof", "coverage proof", "platform proof" |
| External comparison | "bounded comparison against `<library> <version>` for `<fixtures>` and `<metric>`" | "ecosystem parity", "matches SciPy/LAPACK generally", "state-of-practice proof" |

## Day 14 Handoff

Day 14 should use this frozen claim baseline when producing the Sprint 147
closeout and Sprint 148 Windows prerequisite checklist. The closeout should
state that implementation sprints begin from a bounded public-claim surface,
with no unsupported wording fixes left open from Day 13.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| No unsupported claim is allowed to enter implementation sprints. | Complete | The classification table freezes unsupported areas as explicit non-claims or residuals. |
| Any wording fix is evidence-backed. | Complete | No fix was needed; the no-fix rationale records why existing wording is evidence-backed. |
| Explicit non-claims remain visible. | Complete | Static-first package, Windows tier, generated freshness, benchmark, QR/SVD, and external-parity boundaries remain documented. |
