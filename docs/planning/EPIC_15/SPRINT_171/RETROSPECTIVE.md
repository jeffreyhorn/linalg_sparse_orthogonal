# Sprint 171 Retrospective

**Sprint:** 171 - Package-Manager Readiness First Provider
**Duration:** 14 days (Days 1-14 landed on branch `sprint-171`)
**Status:** Complete

## Source Artifact Note

Sprint 171 was executed from the active Epic 15 project-plan section for
Sprint 171 and lives under `docs/planning/EPIC_15/SPRINT_171/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 171 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reviewed Sprint 170 static-first package and shared-library ABI product
      decision as the package-manager readiness baseline.
- [x] Inventoried candidate package-manager providers: vcpkg, Homebrew, Conan,
      pkgsrc, distro/system packages, and formal deferral.
- [x] Selected formal package-manager deferral rather than promoting an
      unproven provider recipe.
- [x] Added the Sprint 171 formal package-manager deferral record.
- [x] Added `scripts/package_manager_deferral_check.sh` as the executable
      deferral guard for provider non-claims, provider recipe absence, package
      metadata neutrality, and public documentation wording.
- [x] Registered the package-manager deferral guard in normalized package
      proof-owner rows as `package_package_manager_deferral_v1`.
- [x] Updated README, INSTALL, and maintainer-guide wording so users and
      maintainers can distinguish source install, CMake/`pkg-config` package
      metadata, and package-manager provider support.
- [x] Ran focused package-manager deferral, static package deferral,
      report-index, package freshness, install proof, claim-scan, staging
      hygiene, and diff-hygiene validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 171 edits.

## What Went Well

1. **The package-manager question got a concrete answer.** Sprint 171 did not
   leave vcpkg, Homebrew, Conan, pkgsrc, distro packages, provider registries,
   taps, recipes, or binary packages in an ambiguous status. The sprint chose
   formal deferral and documented the evidence required before that decision
   can change.

2. **Source install stayed distinct from package-manager support.** The sprint
   preserved Make install, CMake install/export, and `pkg-config` metadata as
   maintained static archive package evidence without widening those facts
   into provider package-manager distribution support.

3. **The unsupported-provider decision is executable.** The new
   `scripts/package_manager_deferral_check.sh` guard verifies the deferral
   record, rejects unselected provider recipe artifacts outside allowed
   historical/planning locations, checks package metadata neutrality, and
   requires public non-claim wording.

4. **Package proof ownership became more complete.** The normalized package
   report-index surface now includes the package-manager deferral guard beside
   Make install, CMake install/export, `sparse.pc.in`,
   `cmake/SparseConfig.cmake.in`, and the Sprint 170 static package deferral
   guard.

5. **Documentation was tightened without creating a fake provider path.**
   README now gives a short user-facing deferral statement, INSTALL names the
   unsupported provider families in the support split, and the maintainer
   guide states when to run the package-manager deferral guard.

6. **Validation matched the changed surface.** The sprint ran the new
   package-manager guard, retained static package guard, package report-index
   checks, freshness checks, targeted claim scans, and the Unix
   Make install/`pkg-config` proof after install/package documentation changed.

7. **Sprint 172 has clean boundaries.** The handoff tells public-header
   coherence work not to imply package-manager availability, shared-library
   support, dynamic ABI guarantees, runtime-loader support, Windows Makefile
   parity, Windows `pkg-config` execution parity, or broad platform parity.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 171 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **No real provider package was added.** This was intentional after the
   provider inventory, but it means users still cannot install through vcpkg,
   Homebrew, Conan, pkgsrc, distro packages, registries, taps, recipes, or
   binary-package channels.

3. **Provider support remains evidence-heavy future work.** A future provider
   sprint still needs source/archive policy, checksums, license/version
   metadata, dependency policy, static/shared policy, provider recipe,
   isolated install proof, downstream consumer proof, cleanup proof, docs, and
   guard coverage.

4. **Claim scans remain interpretive.** Package-manager and package terms
   appear in valid non-claims, package proof descriptions, platform setup
   commands, and historical planning artifacts. Targeted scans are useful, but
   they still require human review of allowed matches.

5. **Package-manager deferral is not yet a CI lane.** The guard is local and
   source-controlled and appears in normalized package proof-owner rows, but no
   dedicated CI job was added in this sprint.

6. **Shared-library and dynamic ABI work remains outside this sprint.** Sprint
   171 depended on the Sprint 170 static-first decision and did not reopen
   export macros, symbol visibility, dynamic ABI policy, loader metadata, or
   installed shared consumer proof.

## Final Metrics

### Validation

| Metric | Sprint 171 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| shell syntax checks | passed: `scripts/package_manager_deferral_check.sh` |
| package-manager deferral guard | passed |
| static package/shared ABI deferral guard | passed |
| Make install and `pkg-config` proof | passed: 23 passed, 0 failed |
| normalized package report index | passed: 7 rows ok |
| package proof-owner freshness | passed: 7 source-controlled rows ok |
| targeted documentation/package claim scan | passed by inspection; matches were scoped claims, non-claims, setup commands, or guard terms |
| generated provider/package artifact staging check | passed: no generated provider recipes or package outputs found |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 171 close state |
| --- | ---: |
| shell guard scripts added | 1 |
| report-index scripts changed | 1 |
| public/maintainer docs changed | 3 |
| workflow files changed | 0 |
| Makefile behavior changed | 0 |
| CMake behavior changed | 0 |
| C source files changed | 0 |
| public header files changed | 0 |
| daily artifacts under `SPRINT_171/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 171 close state |
| --- | ---: |
| package-manager readiness decisions recorded | 1 |
| selected package-manager path | formal deferral |
| provider package-manager support claims added | 0 |
| provider registry readiness claims added | 0 |
| provider tap/recipe/binary-package claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI compatibility claims added | 0 |
| runtime-loader support claims added | 0 |
| Windows Makefile parity claims added | 0 |
| Windows `pkg-config` execution parity claims added | 0 |
| broad platform parity claims added | 0 |
| state-of-the-art package/distribution claims added | 0 |
| normalized package proof-owner rows | 7 |
| new guard scripts for package-manager non-claims | 1 |

## Closed Claim

Sprint 171 closes this Epic 15 package-manager readiness claim:

The project has a source-controlled package-manager product decision. Current
releases do not provide package-manager support. The maintained package
surface remains source install via Make or CMake with static archive
`pkg-config` and CMake package metadata, and package-manager provider support
is formally deferred behind explicit evidence requirements and an executable
guard.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-package-intake.md](./artifacts/day1-package-intake.md);
- [day2-provider-inventory.md](./artifacts/day2-provider-inventory.md);
- [day3-provider-selection.md](./artifacts/day3-provider-selection.md);
- [day4-deferral-artifact-design.md](./artifacts/day4-deferral-artifact-design.md);
- [day5-package-manager-deferral.md](./artifacts/day5-package-manager-deferral.md);
- [day6-proof-script-design.md](./artifacts/day6-proof-script-design.md);
- [day7-proof-script-implementation.md](./artifacts/day7-proof-script-implementation.md);
- [day8-claim-guard-design.md](./artifacts/day8-claim-guard-design.md);
- [day9-claim-guard-implementation.md](./artifacts/day9-claim-guard-implementation.md);
- [day10-documentation-design.md](./artifacts/day10-documentation-design.md);
- [day11-documentation-update.md](./artifacts/day11-documentation-update.md);
- [day12-provider-deferral-validation.md](./artifacts/day12-provider-deferral-validation.md);
- [day13-integrated-claim-review.md](./artifacts/day13-integrated-claim-review.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No vcpkg support, Homebrew formula, Conan recipe, pkgsrc package, Debian or
Fedora package, provider registry submission, tap, binary package,
package-manager dependency resolution, provider-managed version compatibility,
provider-managed checksum/license/source archive policy, shared-library
build/install support, dynamic ABI stability, runtime-loader behavior, Windows
Makefile parity, Windows `pkg-config` execution parity, broad platform parity,
portable performance claim, external-library parity claim, or state-of-the-art
sparse linear algebra claim was added.

## Sprint 172 Readiness

| Future need | Sprint 171 handoff |
| --- | --- |
| Public-header coherence | Do not imply package-manager availability, shared-library support, dynamic ABI guarantees, runtime-loader support, Windows Makefile parity, Windows `pkg-config` execution parity, or broad platform parity from public headers or API docs. |
| Package-manager wording changes | Run `bash scripts/package_manager_deferral_check.sh`. |
| Static package/shared ABI wording changes | Run `bash scripts/static_package_deferral_check.sh`. |
| Package proof-owner changes | Run `python3 scripts/normalize_report_index.py --family package --check` and `python3 scripts/normalize_report_index.py --family package --check-freshness`. |
| Install metadata or install-doc changes | Rerun relevant install proof, starting with `bash tests/test_install.sh` for Unix Make install/`pkg-config`. |
| Future provider support | Select exactly one provider and add recipe, isolated install proof, downstream consumer proof, cleanup proof, docs, and guards before claiming support. |
