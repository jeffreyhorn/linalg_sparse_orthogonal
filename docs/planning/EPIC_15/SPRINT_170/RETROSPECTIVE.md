# Sprint 170 Retrospective

**Sprint:** 170 - Shared-Library ABI Product Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-170`)
**Status:** Complete

## Source Artifact Note

Sprint 170 was executed from the active Epic 15 project-plan section for
Sprint 170 and lives under `docs/planning/EPIC_15/SPRINT_170/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 170 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited public header ABI exposure across installed headers, generated
      version metadata, concrete structs, callbacks, constants, and lifecycle
      families.
- [x] Audited lifecycle and ownership semantics for matrix, dense, CSR/CSC,
      direct factor, repeated-run, callback, result-buffer, and error-state
      surfaces.
- [x] Audited static archive symbol visibility and identified internal-looking
      global symbol families that would need export policy before any
      shared-library support claim.
- [x] Reviewed Make and CMake package feasibility, including static archive
      install, `sparse.pc`, CMake package exports, exact version metadata, and
      Windows CMake-first package boundaries.
- [x] Audited public and maintainer documentation for package, ABI,
      runtime-loader, package-manager, platform, and state-of-the-art claim
      risk.
- [x] Created the Sprint 170 shared-library ABI product decision record and
      selected static-first-only continuation.
- [x] Designed and implemented static package deferral guard updates for the
      decision record, Makefile static archive contract, exact `sparse.pc.in`
      metadata, and documentation decision citations.
- [x] Updated README, INSTALL, and maintainer-guide wording so the canonical
      Sprint 170 decision record is discoverable from the public package
      support surface.
- [x] Ran integrated package, guard, documentation, and diff validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 170 edits.

## What Went Well

1. **The shared-library ABI question got a concrete answer.** Sprint 170 did
   not leave the project in an ambiguous "maybe shared" state. Day 9 accepted
   static-first-only continuation and listed the unsupported shared-library,
   dynamic ABI, loader, platform, and package-manager claims explicitly.

2. **The decision was evidence-based.** Days 1-8 reviewed headers, lifecycle
   ownership, concrete layout exposure, archive symbols, Make install behavior,
   CMake exports, CI lanes, package metadata, and claim wording before the
   product decision was recorded.

3. **Static-first support stayed real instead of rhetorical.** The retained
   claim maps directly to existing Make install/`pkg-config` proof, CMake
   install/export proof, Linux/macOS package lanes, and Windows CMake
   install/downstream validation.

4. **Unsupported ABI work is now mechanically guarded.** Day 11 extended
   `scripts/static_package_deferral_check.sh` so the Day 9 decision record,
   Makefile static archive contract, and exact static `sparse.pc.in` metadata
   cannot drift silently.

5. **Documentation now points to the canonical decision.** Day 12 added direct
   decision-record references in README, INSTALL, and the maintainer guide and
   guarded those references.

6. **Validation stayed scoped to the changed surface.** Day 13 reran the
   installed package proofs and static deferral guard, while correctly skipping
   the full C quality gate because no C or public header files changed.

7. **The next sprint has a clean handoff.** Day 14 makes package-manager
   readiness depend on provider-specific proof or explicit deferral, not on
   static install evidence alone.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 170 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **Shared-library support remains a large future product.** Sprint 170
   closed the decision, but it did not add export macros, hidden visibility,
   symbol allowlists, ABI version policy, SONAME/install-name/RPATH metadata,
   DLL/import-library support, or installed shared consumer proof.

3. **Public concrete structs remain an ABI blocker.** Many option/result/factor
   objects expose concrete layouts. That is acceptable for the selected
   source/static package posture, but it prevents an honest dynamic ABI claim.

4. **The archive symbol surface is still uncurated.** Local inspection found a
   large global symbol surface, including internal-looking helpers. This is
   guarded by rejecting shared-library configuration, but not solved for a
   future shared product.

5. **Claim scans remain interpretive.** Package and ABI terms appear in
   explicit non-claims, decision records, and validation logs, so targeted
   scans require human interpretation rather than simple zero-match rules.

6. **Package-manager readiness remains unselected.** The sprint deliberately
   avoided treating source install, CMake package discovery, or `pkg-config`
   metadata as package-manager support.

## Final Metrics

### Validation

| Metric | Sprint 170 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| shell syntax checks | passed: `scripts/static_package_deferral_check.sh` |
| static package deferral guard | passed |
| Make install and `pkg-config` proof | passed: 23 passed, 0 failed |
| CMake install/export proof | passed: 27 passed, 0 failed, 0 skipped |
| targeted documentation claim scan | passed by inspection; matches were scoped claims, non-claims, or decision references |
| final `git diff --check` | passed |
| generated build/report/cache artifacts committed | 0 |

### Changed Surface

| Metric | Sprint 170 close state |
| --- | ---: |
| shell guard scripts changed | 1 |
| public/maintainer docs changed | 3 |
| workflow files changed | 0 |
| Makefile behavior changed | 0 |
| CMake behavior changed | 0 |
| C source files changed | 0 |
| public header files changed | 0 |
| daily artifacts under `SPRINT_170/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 170 close state |
| --- | ---: |
| shared-library ABI product decisions recorded | 1 |
| selected maintained package posture | static-first-only |
| shared-library support claims added | 0 |
| dynamic ABI compatibility claims added | 0 |
| runtime-loader support claims added | 0 |
| package-manager distribution claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| broad platform parity claims added | 0 |
| state-of-the-art package/ABI claims added | 0 |
| new guard checks for decision/documentation/package drift | 4 |

## Closed Claim

Sprint 170 closes this Epic 15 package/ABI decision claim:

The project has a source-controlled shared-library ABI product decision. The
maintained package posture is static-first-only, backed by Make
install/`pkg-config` proof, CMake install/export proof, Linux/macOS static
package CI lanes, Windows CMake install/downstream validation, public and
maintainer documentation, and static package deferral guards.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-abi-intake.md](./artifacts/day1-abi-intake.md);
- [day2-header-abi-inventory.md](./artifacts/day2-header-abi-inventory.md);
- [day3-lifecycle-audit.md](./artifacts/day3-lifecycle-audit.md);
- [day4-symbol-visibility.md](./artifacts/day4-symbol-visibility.md);
- [day5-make-feasibility.md](./artifacts/day5-make-feasibility.md);
- [day6-cmake-feasibility.md](./artifacts/day6-cmake-feasibility.md);
- [day7-claim-surface-audit.md](./artifacts/day7-claim-surface-audit.md);
- [day8-decision-synthesis.md](./artifacts/day8-decision-synthesis.md);
- [day9-shared-library-abi-product-decision.md](./artifacts/day9-shared-library-abi-product-decision.md);
- [day10-guard-update-design.md](./artifacts/day10-guard-update-design.md);
- [day11-guard-implementation.md](./artifacts/day11-guard-implementation.md);
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No shared-library build/install support, dynamic ABI stability, stable dynamic
symbol list, runtime-loader behavior, Linux SONAME support, macOS
install-name/RPATH support, Windows DLL/import-library support, package-manager
distribution, static/shared selector, Windows Makefile parity, Windows
`pkg-config` execution parity, broad platform parity, performance claim,
external-library parity claim, or state-of-the-art sparse linear algebra claim
was added.

## Sprint 171 Readiness

| Future need | Sprint 170 handoff |
| --- | --- |
| Package-manager readiness | Start with provider selection or explicit deferral; do not treat source install, CMake discovery, or `pkg-config` metadata as package-manager support. |
| Static package contract | Keep the static archive package surface as the only maintained package product. |
| Shared-library exploration | Require ABI scope, layout policy, export policy, symbol allowlist, allocation policy, ABI version policy, platform loader policy, package metadata policy, and installed shared consumer proof before support can be claimed. |
| Windows package surface | Preserve CMake install/downstream validation as the reviewed Windows package path; do not infer Windows Makefile or Windows `pkg-config` execution parity. |
| Guard maintenance | Keep `scripts/static_package_deferral_check.sh`, `tests/test_install.sh`, and `tests/test_cmake_install.sh` as the minimum package-claim regression stack. |
| Claim-scope maintenance | Continue scanning README, INSTALL, maintainer docs, package metadata, and CI workflow text for shared-library, ABI, runtime-loader, package-manager, platform, and state-of-the-art overreach. |
