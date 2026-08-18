# Sprint 165 Day 13: Evidence Review And Residual Register

## Purpose

Day 13 reviews the Sprint 165 evidence chain before closeout. The review checks
that the static-first package boundary hardening work is supported by artifacts,
tests, and documentation updates, and that remaining package/ABI work is
recorded as product decisions rather than hidden implementation debt.

## Sprint 165 Deliverable Evidence Checklist

| Sprint 165 deliverable | Status | Evidence |
| --- | --- | --- |
| Hardened static-first package boundary | Complete | Day 3 designed metadata absence rules; Day 4 strengthened `scripts/static_package_deferral_check.sh`; Day 12 reran the guard successfully. |
| Updated ABI/package non-claim audit | Complete | Day 5 separated source API, package metadata, and binary ABI; Day 6 removed stale ABI wording from `include/sparse_cholesky.h`; Day 12 claim-drift scan found no unsupported claims. |
| Refreshed downstream static package proof | Complete | Day 7 scoped Unix and Windows proof boundaries; Day 8 updated `tests/test_install.sh`; Day 10 and Day 11 validated Make install/`pkg-config` and CMake install/export downstream consumers. |
| Package documentation aligned with supported behavior | Complete | Day 9 updated README, INSTALL, maintainer guidance, and CMake comments; Day 12 checked package report rows and scanned for unsupported wording. |
| Sprint 166 closeout handoff | Complete for Sprint 165 | This artifact records residuals and Sprint 166 handoff items for final validation and Epic 14 closeout. |

## Project-Plan Item Evidence

| Item # | Item | Evidence | Result |
| --- | --- | --- | --- |
| 1 | Package Metadata Audit | Day 2 audited CMake metadata, `sparse.pc`, install/downstream scripts, CI package coverage, unsupported wording, and validation gaps. | Complete |
| 2 | Static Deferral Guard | Day 3 designed guard rules; Day 4 implemented stricter static-package guard checks; Day 12 confirmed the guard still passes. | Complete |
| 3 | ABI Non-Claim Audit | Day 5 documented the exact-package-version versus dynamic-ABI boundary; Day 6 cleaned stale public-header wording. | Complete |
| 4 | Downstream Proof Refresh | Day 7 scoped proof obligations; Day 8 added stronger `pkg-config` path and semantic output checks; Day 10/11 reran install and CMake downstream proofs. | Complete |
| 5 | Package Docs Alignment | Day 9 aligned README, INSTALL, maintainer guide, and CMake comments around static-first support and explicit non-claims. | Complete |
| 6 | Validation | Day 12 passed the full required quality gate, focused package scripts, package report-index checks, claim scan, and `git diff --check`. | Complete |
| 7 | Closeout | Day 13 residual register and Sprint 166 handoff are recorded here; Day 14 can convert this into final sprint closeout. | Complete for evidence-review stage |

## Evidence By Package Surface

| Surface | Sprint 165 evidence | Current supported reading |
| --- | --- | --- |
| CMake build configuration | `BUILD_SHARED_LIBS=ON` remains a configure-time failure and the rejection names missing shared-library, dynamic ABI, loader, and consumer-proof policies. | Static archive package surface only. |
| CMake install/export package | `tests/test_cmake_install.sh` checks installed static target metadata, install-prefix paths, no source/build path leaks, exact-version consumer success, and mismatched-version rejection. | Installed CMake package resolves the static archive for maintained downstream consumers. |
| Make install and `pkg-config` | `tests/test_install.sh` checks installed paths, exact version, static link flags, generated and maintained downstream consumers, no private dependency stanza, and unsupported wording absence. | Unix-like installed `pkg-config` proof for the maintained static archive package. |
| Windows package lane | Existing workflow remains CMake-first with installed `.lib`, headers, CMake package metadata, metadata-only `sparse.pc` inspection, exact-version checks, and explicit Make/pkg-config non-claims. | Reviewed Windows CMake install/downstream confidence, not Windows Makefile or `pkg-config` execution parity. |
| Public docs | README, INSTALL, and maintainer guide now point to the same static-first package evidence and retained non-claims. | Users get a coherent static package story without inferred shared-library, ABI, package-manager, or broad platform support. |
| Maintainer checks | `scripts/static_package_deferral_check.sh` owns local package-boundary drift detection. | Maintainers have a repeatable guard against accidental widening of unsupported package claims. |

## Package Residual Register

| Residual | Current state | Product decision needed | Promotion gate |
| --- | --- | --- | --- |
| True shared-library support | Deferred. `BUILD_SHARED_LIBS=ON` is intentionally rejected. | Decide whether the project will support shared libraries as a shipped product surface. | Export/import policy, symbol visibility policy, platform loader metadata, installed shared consumer tests, CI lanes, and docs that distinguish static and shared behavior. |
| Dynamic ABI compatibility | Deferred. Exact package-version metadata is not an ABI promise. | Decide compatibility scope, versioning rules, and what source/API or binary changes are allowed. | ABI policy document, compatibility tests or tooling, symbol/layout review process, release notes policy, and failure criteria for breaking changes. |
| Runtime-loader behavior | Deferred. No loader path or dynamic artifact behavior is claimed. | Decide whether runtime loader behavior belongs to this project and which platforms are supported. | Linux loader validation, macOS install-name/RPATH validation, Windows DLL/import-library validation, downstream run tests, and troubleshooting docs. |
| Package-manager distribution | Deferred. Package-manager names appear only in non-claims or prerequisite installation notes. | Decide whether to publish through Homebrew, apt/dnf/pacman, vcpkg, Conan, or another channel. | Provider-specific packaging files, package-manager install tests, upgrade/version behavior checks, ownership docs, and support tier wording. |
| Static/shared selector UX | Deferred beyond fail-closed `BUILD_SHARED_LIBS=ON` rejection. | Decide whether users should have a supported selector or whether static-only remains the product boundary. | CMake option design, package metadata split, CI for both modes if enabled, docs, and drift guards for both supported paths. |
| Windows Makefile install/uninstall parity | Deferred. Windows package proof is CMake-first. | Decide whether Windows Makefile installation is a supported workflow. | Windows shell/toolchain selection, Make install/uninstall tests, path normalization, CI lane, and public docs. |
| Windows `pkg-config` command execution parity | Deferred. Windows `sparse.pc` inspection is metadata-only. | Decide whether Windows `pkg-config` execution is a supported downstream path. | Selected Windows `pkg-config` provider, command execution tests, downstream compile/link/run proof, path semantics, CI lane, and docs. |
| Broader platform package parity | Deferred beyond selected reviewed/supplemental lanes. | Decide which platform/package surfaces are release-blocking versus confidence-only. | Platform support matrix, reviewed lane definitions, supplemental lane definitions, and claim-audit wording. |

## Sprint 166 Handoff Items

Sprint 166 should use this sprint’s artifacts as package-boundary inputs for
Epic 14 final validation:

1. Include the Sprint 165 package evidence in the final evidence inventory:
   Day 4 static guard, Day 8 downstream proof changes, Day 10/11 install proof,
   and Day 12 full quality gate.
2. Re-run or cite the strongest feasible package checks during final validation:
   `scripts/static_package_deferral_check.sh`, `tests/test_install.sh`,
   `tests/test_cmake_install.sh`, package report-index checks, and claim scan.
3. Reconcile hosted CI evidence separately from local evidence, especially for
   Windows CMake-first install/downstream proof and Windows metadata-only
   `sparse.pc` inspection.
4. Scan final public docs for unsupported state-of-the-art, package-manager,
   shared-library, dynamic ABI, runtime-loader, Windows Makefile, and Windows
   `pkg-config` wording.
5. Mark Sprint 165 project-plan items complete in the Epic 14 reconciliation,
   while preserving the residual register as future product scope.
6. Carry the residual register into the Epic 14 retrospective and final
   residual queue without presenting deferred product decisions as defects in
   the static-first package contract.

## Evidence Chain Summary

Sprint 165 now has a complete evidence chain for its selected scope:

- Day 1 established package surfaces and non-goals.
- Day 2 audited package metadata and validation owners.
- Day 3 designed guard hardening.
- Day 4 implemented the stricter static-package guard.
- Day 5 audited ABI-adjacent wording.
- Day 6 cleaned stale public-header ABI wording.
- Day 7 scoped downstream proof and stale-expectation risks.
- Day 8 strengthened Make install/`pkg-config` downstream validation.
- Day 9 aligned public and maintainer documentation.
- Day 10 inspected installed package metadata.
- Day 11 validated installed downstream consumers.
- Day 12 passed full quality and focused package gates.
- Day 13 records evidence coverage, residual product decisions, and Sprint 166
  handoff.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 165 deliverable has supporting evidence. | Complete | Deliverable checklist maps each end-state bullet to Day 1-12 artifacts and Day 13 handoff. |
| Residuals are product decisions, not hidden implementation gaps. | Complete | Residual register names the decision, current state, and promotion gate for each deferred package/ABI surface. |
| Sprint 166 receives a clear closeout handoff. | Complete | Handoff section lists final evidence inventory, validation, hosted CI reconciliation, claim audit, project-plan reconciliation, retrospective, and residual queue inputs. |
