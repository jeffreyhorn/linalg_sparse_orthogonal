# Sprint 165 Retrospective

**Sprint:** 165 - Static-First Package Boundary Hardening
**Duration:** 14 days (Days 1-14 landed on branch `sprint-165`)
**Status:** Complete

## Source Artifact Note

Sprint 165 was executed from the Epic 14 project-plan section for Sprint 165
and lives under `docs/planning/EPIC_14/SPRINT_165/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 project-plan path; `WORKING_NOTES.md` records that
path mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 165 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Inventoried CMake, Makefile, `pkg-config`, install scripts, CI package
      lanes, README, INSTALL, maintainer guide, and downstream examples.
- [x] Audited package metadata for static-first behavior, exact-version
      package metadata, no shared imported metadata, no loader metadata, and
      no package-manager or ABI wording in installed package files.
- [x] Strengthened the static package deferral guard so public headers cannot
      gain export/import or static/shared ABI macro scaffolding before a
      shared-library product decision.
- [x] Audited ABI-adjacent wording and replaced the stale public-header
      "`ABI break`" phrase in `include/sparse_cholesky.h` with source-rebuild
      options-layout wording.
- [x] Refreshed downstream Make install/`pkg-config` proof with filesystem
      identity path checks and semantic downstream output checks.
- [x] Revalidated CMake install/export proof, exact-version consumers,
      mismatched-version rejection, and installed package metadata boundaries.
- [x] Aligned README, INSTALL, maintainer guide, and CMake comments with the
      maintained static archive package surface.
- [x] Preserved explicit non-claims for shared-library support, dynamic ABI
      compatibility, runtime-loader behavior, package-manager distribution,
      Windows Makefile parity, Windows `pkg-config` execution parity, broad
      platform package parity, portable performance, and state-of-the-art
      claims.
- [x] Ran the required quality gate because a public header changed:
      `make format && make lint && make test`.
- [x] Ran focused package checks, package report-index checks, freshness
      checks, unsupported-claim scans, and final whitespace checks.
- [x] Published the Sprint 166 final-validation and Epic 14 closeout handoff.

## What Went Well

1. **The sprint kept static-first support as the selected product boundary.**
   It hardened evidence around the maintained static archive package without
   accidentally promoting shared-library, loader, package-manager, or dynamic
   ABI support.

2. **The guard work stayed targeted.** The only code-adjacent hardening was a
   narrow extension to `scripts/static_package_deferral_check.sh` so public
   headers reject `SPARSE_SHARED`, `SPARSE_STATIC`, and `SPARSE_ABI` alongside
   existing export/import macro checks.

3. **The ABI wording cleanup removed an over-readable phrase.** Replacing
   "`ABI break`" in `include/sparse_cholesky.h` with source-rebuild
   options-layout wording aligned Cholesky with newer public option/result
   structs without declaring a dynamic ABI policy.

4. **Install proof became less brittle.** `tests/test_install.sh` now compares
   installed `pkg-config` prefix, libdir, includedir, cflags, and libs paths by
   filesystem identity, preventing double-slash temp-prefix spelling from
   causing false package failures.

5. **Documentation now matches the proof lanes.** README, INSTALL, and the
   maintainer guide agree that Unix Make/`pkg-config` and CMake install/export
   proof cover the maintained static archive, while Windows remains
   CMake-first with metadata-only `sparse.pc` inspection.

6. **Closeout evidence is reviewable.** Day 13 maps every Sprint 165
   deliverable to evidence, and Day 14 records PR bullets, review risks,
   residuals, and Sprint 166 handoff items.

## What Didn't Go Well

1. **The sprint prompt path was stale.** The request referenced Epic 12 while
   the active Sprint 165 plan lives under Epic 14. The sprint recorded the
   mismatch and proceeded from the current Epic 14 plan.

2. **The package boundary remains deliberately narrow.** The sprint closed
   static-first hardening, not shared-library support, dynamic ABI
   compatibility, runtime-loader behavior, package-manager distribution, or
   Windows Make/`pkg-config` parity.

3. **Validation is expensive when a public header changes.** The public-header
   wording cleanup correctly required `make format && make lint && make test`.
   The gate passed, but it remains a long-running review cost.

4. **Windows package evidence is still hosted-only.** Local macOS validation
   can exercise Unix Make/`pkg-config` and CMake install/export paths, but
   Windows `.lib`, CMake-first install/downstream behavior, and metadata-only
   `sparse.pc` inspection remain hosted CI evidence.

5. **Claim-boundary wording requires repeated scans.** Terms such as ABI,
   shared library, package-manager, loader, and Windows parity appear in both
   guard checks and non-claims, so review needs to distinguish protective
   wording from accidental support claims.

## Final Metrics

### Validation

| Metric | Sprint 165 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | yes: `include/sparse_cholesky.h` |
| full C quality gate required | yes |
| full C quality gate | passed: `make format && make lint && make test` |
| static package deferral guard | passed |
| Make install/`pkg-config` validation | passed: 23 passes, 0 failures |
| CMake install/export validation | passed: 27 passes, 0 failures, 0 skips |
| package report-index check | passed: 6 rows |
| package report-index freshness | passed: 6 source-controlled advisory rows |
| unsupported-claim scan | passed; hits are guards, metadata rejection checks, artifacts, or explicit non-claims |
| stale Epic 12 reference check | passed; remaining mentions are source-note/context references |
| final `git diff --check` | passed |

### Selected Package Surface

| Metric | Sprint 165 close state |
| --- | ---: |
| checked package/install surfaces inventoried | 13 |
| public headers changed | 1 |
| C source files changed | 0 |
| shell/package validation scripts changed | 2 |
| public/maintainer docs changed | 3 |
| CMake package comments/config changed | 1 |
| installed headers validated | 19 |
| package report-index rows | 6 |
| shared-library support claims added | 0 |
| dynamic ABI support claims added | 0 |
| package-manager support claims added | 0 |
| Windows Make/`pkg-config` parity claims added | 0 |

### Artifact Package

| Metric | Sprint 165 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| generated build/install artifacts committed | 0 |

## Closed Claim

Sprint 165 closes this static-first package boundary hardening claim:

The project now has a stronger static-first package contract around the
maintained archive install surface. `BUILD_SHARED_LIBS=ON` remains fail-closed,
public headers are guarded against premature export/import and static/shared
ABI macro scaffolding, installed package metadata is validated against shared
library, loader, package-manager, static/shared selector, and ABI drift, and
downstream Make/`pkg-config` plus CMake install/export consumers are refreshed
with exact-version and installed-path checks. README, INSTALL, maintainer
guidance, and CMake comments describe the same bounded package surface.

No shared-library support, dynamic ABI compatibility, runtime-loader behavior,
package-manager distribution, Windows Makefile parity, Windows `pkg-config`
command execution parity, broad platform package parity, portable performance,
or state-of-the-art sparse linear algebra claim was added.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-package-metadata-audit.md](./artifacts/day2-package-metadata-audit.md);
- [day3-static-deferral-guard-design.md](./artifacts/day3-static-deferral-guard-design.md);
- [day4-static-deferral-guard-implementation.md](./artifacts/day4-static-deferral-guard-implementation.md);
- [day5-abi-non-claim-audit.md](./artifacts/day5-abi-non-claim-audit.md);
- [day6-abi-package-wording-cleanup.md](./artifacts/day6-abi-package-wording-cleanup.md);
- [day7-downstream-proof-scope.md](./artifacts/day7-downstream-proof-scope.md);
- [day8-downstream-proof-implementation.md](./artifacts/day8-downstream-proof-implementation.md);
- [day9-package-documentation-alignment.md](./artifacts/day9-package-documentation-alignment.md);
- [day10-package-metadata-validation.md](./artifacts/day10-package-metadata-validation.md);
- [day11-install-downstream-validation.md](./artifacts/day11-install-downstream-validation.md);
- [day12-quality-gate-drift-scan.md](./artifacts/day12-quality-gate-drift-scan.md);
- [day13-evidence-review-residual-register.md](./artifacts/day13-evidence-review-residual-register.md);
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md).

## Sprint 166 Readiness

Sprint 166 should begin from these settled Sprint 165 boundaries:

| Starting item | Required posture |
| --- | --- |
| Static package contract | Treat static archive install/export as the selected supported package surface. |
| Shared-library requests | Keep `BUILD_SHARED_LIBS=ON` fail-closed unless a product decision adds shared-library support and proof. |
| Package metadata | Keep installed CMake and `sparse.pc` metadata static archive scoped with no loader, shared selector, package-manager, or dynamic ABI claims. |
| Windows package proof | Treat Windows as CMake-first hosted evidence with metadata-only `sparse.pc` inspection. |
| Package validation | Reuse Day 12 checks as the strongest package-boundary baseline for Epic 14 closeout. |
| Public claims | Preserve static-first support and explicit package/ABI/platform non-claims during final claim recalibration. |

Recommended Sprint 166 first step:

Inventory Sprint 165's package-boundary evidence alongside generated API,
hosted oracle/comparison, partial-SVD, performance, and public-header evidence,
then run the strongest feasible final validation baseline before recalibrating
Epic 14 public claims.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 165 close:

- true shared-library support;
- dynamic ABI compatibility policy;
- runtime-loader behavior and platform loader metadata;
- Linux SONAME policy;
- macOS install-name/RPATH policy;
- Windows DLL/import-library behavior;
- package-manager distribution through Homebrew, apt, dnf, pacman, vcpkg,
  Conan, or similar systems;
- static/shared selector UX beyond fail-closed shared-library deferral;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broader platform package parity;
- provider-specific package upgrade/version behavior;
- release-quality package support matrix beyond the selected proof lanes.

Still consciously constrained rather than silently solved:

- no shared-library support claim;
- no dynamic ABI compatibility claim;
- no runtime-loader behavior claim;
- no package-manager distribution claim;
- no Windows Makefile parity claim;
- no Windows `pkg-config` execution parity claim;
- no broad platform package parity claim;
- no package evidence reused as performance evidence;
- no package evidence reused as state-of-the-art sparse linear algebra proof.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-package-metadata-audit.md](./artifacts/day2-package-metadata-audit.md)
- [day3-static-deferral-guard-design.md](./artifacts/day3-static-deferral-guard-design.md)
- [day4-static-deferral-guard-implementation.md](./artifacts/day4-static-deferral-guard-implementation.md)
- [day5-abi-non-claim-audit.md](./artifacts/day5-abi-non-claim-audit.md)
- [day6-abi-package-wording-cleanup.md](./artifacts/day6-abi-package-wording-cleanup.md)
- [day7-downstream-proof-scope.md](./artifacts/day7-downstream-proof-scope.md)
- [day8-downstream-proof-implementation.md](./artifacts/day8-downstream-proof-implementation.md)
- [day9-package-documentation-alignment.md](./artifacts/day9-package-documentation-alignment.md)
- [day10-package-metadata-validation.md](./artifacts/day10-package-metadata-validation.md)
- [day11-install-downstream-validation.md](./artifacts/day11-install-downstream-validation.md)
- [day12-quality-gate-drift-scan.md](./artifacts/day12-quality-gate-drift-scan.md)
- [day13-evidence-review-residual-register.md](./artifacts/day13-evidence-review-residual-register.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
