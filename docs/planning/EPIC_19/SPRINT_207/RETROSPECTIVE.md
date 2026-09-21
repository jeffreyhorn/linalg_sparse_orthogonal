# Sprint 207 Retrospective

**Sprint:** 207 - Package Distribution Support Decision  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-207`)  
**Status:** Closed with continued package-provider deferral and stronger
guards; broad package-manager support remains unclaimed

## Source Artifact Note

Sprint 207 was executed from the Epic 19 project-plan section for Sprint 207
and lives under `docs/planning/EPIC_19/SPRINT_207/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint started from the Sprint 198 developer-mode local Homebrew static
source formula proof. It reviewed provider options, audited the existing
formula/proof metadata, re-ran the local proof, selected continued
package-provider deferral with stronger guards, updated user and maintainer
docs, added regression coverage for package-provider overclaims, and closed
with integrated validation. It did not promote a user-facing Homebrew install
path, Homebrew/core readiness, bottles, Linuxbrew, other package managers,
binary/release packages, shared-library packages, dynamic ABI behavior, or
broad package-manager distribution.

## Definition Of Done Checklist

- [x] Created Sprint 207 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Audited Sprint 198 Homebrew proof evidence, current package docs,
      package-manager guards, static-package guards, install checks, and
      maintainer guidance.
- [x] Compared public Homebrew tap/source formula, Homebrew/core readiness,
      and continued-deferral options with evidence requirements and claim risk.
- [x] Selected continued package-provider deferral with stronger guards after
      finding no stable public archive, provider formula ownership, non-local
      provider proof, or Homebrew/core audit evidence.
- [x] Preserved the local Homebrew proof as developer-mode local static source
      formula evidence only.
- [x] Updated `scripts/package_manager_deferral_check.sh` to enforce Sprint
      207 decision records, forbidden package-provider overclaims, local proof
      boundary, metadata neutrality, generated-output hygiene, and public
      non-claims.
- [x] Added `tests/test_package_manager_deferral_guard.py` with focused
      regressions for Homebrew/core readiness, public tap support, plural
      public taps support, package-manager support, binary packages, bottles,
      Linuxbrew, and release packages.
- [x] Updated README, INSTALL, `packaging/homebrew/README.md`, and maintainer
      guidance so users and maintainers can distinguish source install,
      local-only Homebrew proof, and retained provider non-claims.
- [x] Updated Epic 19 project-plan status to close Sprint 207 and leave
      Sprints 208-216 pending future execution.
- [x] Ran local Homebrew proof, package guards, static package guard, docs and
      support docs guards, package-manager regressions, Make install, CMake
      install/export, cleanup scans, and whitespace validation.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required by the sprint instruction.

## What Went Well

1. **The sprint made a product decision instead of stretching proof wording.**
   The local Homebrew proof remains valuable evidence, but the branch
   explicitly re-deferred public provider support because the missing evidence
   is provider and release provenance, not implementation syntax.

2. **Guard coverage now catches common package overclaims.** The package guard
   rejects broad package-manager support, public tap wording, Homebrew/core
   readiness, bottles, Linuxbrew, binary package, release package, and other
   unsupported provider wording before the expensive Homebrew proof runs.

3. **User and maintainer docs now agree on the package boundary.** README,
   INSTALL, the Homebrew proof notes, and maintainer guide all route users to
   source install while preserving Homebrew as developer-mode local proof only.

4. **Validation covered both proof and installed package behavior.** The
   sprint re-ran the standalone Homebrew proof, the aggregate package guard
   with embedded proof, Make install validation, CMake install/export
   validation, docs guards, support docs guard, and cleanup scans.

5. **Future package promotion now has an evidence checklist.** Maintainers have
   exact requirements for public tap/source formula, Homebrew/core readiness,
   bottles, Linuxbrew, other providers, release packages, shared-library
   packages, and dynamic ABI behavior.

## What Didn't Go Well

1. **Package-provider wording is easy to over-broaden.** Day 13 found that
   singular `public tap` checks did not explicitly cover plural `public taps`
   wording and that direct `package-manager support` overclaims needed their
   own guard pattern.

2. **The main package guard is intentionally expensive.** The guard embeds the
   local Homebrew proof, so repeated validation during Days 7-14 took time.
   That cost is acceptable for closeout but should stay visible to maintainers.

3. **Public package support remains a residual.** Sprint 207 improved decision
   clarity and guards, but it did not add stable public archive provenance,
   provider formula ownership, hosted/provider validation, or release
   readiness.

4. **The current Homebrew host is still limited evidence.** The proof was
   successful on macOS Intel x86_64 Tier 3 Homebrew with developer mode, which
   remains local proof rather than broad provider support.

## Final Metrics

### Validation

| Metric | Sprint 207 close state |
| --- | --- |
| standalone Homebrew local formula proof | passed with `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`: archive/checksum, temporary tap render, source install, installed static surface validation, downstream `brew test`, uninstall, cleanup, and local-proof scope |
| package-manager deferral guard | passed after Sprint 207 decision, overclaim, and public non-claim checks were added |
| static package deferral guard | passed |
| package-manager Python regression tests | passed |
| package-manager guard syntax check | passed |
| Make install validation | passed: 23 passed, 0 failed |
| CMake install validation | passed: 27 passed, 0 failed, 0 skipped |
| docs check | passed |
| support docs guard | passed |
| final `git diff --check` | passed |
| generated Homebrew proof output scan | passed with no generated proof outputs under `packaging/homebrew` |
| final `.c`/`.h` diff check | passed with no output |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 207 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 2 |
| Maintainer documentation files changed | 1 |
| Package documentation files changed | 1 |
| Package guard scripts changed | 1 |
| Package guard regression files added | 1 |
| Homebrew formula template files changed | 0 |
| Install metadata files changed | 0 |
| CI workflow files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Provider decision items completed | 1 |
| Formula/proof audit items completed | 1 |
| Selected deferral implementation items completed | 1 |
| Guard alignment items completed | 1 |
| User and maintainer documentation items completed | 1 |
| Validation and closeout items completed | 1 |
| Package-provider support claims promoted | 0 |
| Package-provider residuals retained | 7 |

The count covers Sprint 207 items 207.1 through 207.6.

## Closed Claim

Sprint 207 closes this bounded claim:

The current branch selects continued package-provider deferral with stronger
guards, preserves the Sprint 198 developer-mode local Homebrew static source
formula proof as local evidence only, adds package-provider overclaim
regressions, aligns user and maintainer documentation, records residual
evidence requirements for future package promotion, and validates the selected
local proof, package guards, docs/support guards, install surfaces, and cleanup
hygiene.

This claim does not include a user-facing Homebrew install path, public tap
maintenance, Homebrew/core readiness, bottles, Linuxbrew, vcpkg, Conan, pkgsrc,
distro/system packages, binary packages, release packages, package-manager
release readiness, shared-library packages, dynamic ABI behavior,
runtime-loader behavior, broad package-manager distribution, or package
ecosystem parity.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-package-intake.md](./artifacts/day1-package-intake.md);
- [day2-provider-scope-options.md](./artifacts/day2-provider-scope-options.md);
- [day3-formula-metadata-baseline.md](./artifacts/day3-formula-metadata-baseline.md);
- [day4-environment-proof-baseline.md](./artifacts/day4-environment-proof-baseline.md);
- [day5-provider-decision.md](./artifacts/day5-provider-decision.md);
- [day6-proof-deferral-design.md](./artifacts/day6-proof-deferral-design.md);
- [day7-deferral-guard-implementation.md](./artifacts/day7-deferral-guard-implementation.md);
- [day8-proof-regression-cleanup.md](./artifacts/day8-proof-regression-cleanup.md);
- [day9-package-guard-alignment.md](./artifacts/day9-package-guard-alignment.md);
- [day10-user-package-docs.md](./artifacts/day10-user-package-docs.md);
- [day11-maintainer-package-docs.md](./artifacts/day11-maintainer-package-docs.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Public Homebrew tap/source formula remains unclaimed | Future package/provider owner after product decision selects this path | Add stable source archive URL, SHA-256 provenance, provider formula ownership, non-local render/audit/install/test/uninstall proof, cleanup proof, docs, residual updates, and guard coverage. |
| Homebrew/core readiness remains unclaimed | Future package/provider owner after public tap evidence exists | Add all public tap/source formula evidence plus Homebrew/core-style formula audit evidence, release archive discipline, submission/maintenance ownership, and wording that does not imply acceptance, bottles, Linuxbrew, or binary distribution. |
| Bottles and Linuxbrew remain unclaimed | Future binary/provider owner | Add provider-specific bottle/Linuxbrew proof, platform policy, hosted or reproducible validation, cleanup/artifact policy, docs, and guard coverage. |
| vcpkg, Conan, pkgsrc, and distro/system packages remain unclaimed | Future provider owner for each selected provider | Add provider-specific recipe ownership, provider validation, downstream consumer proof, docs, residual updates, and overclaim guards. |
| Binary packages, release packages, and package-manager release readiness remain unclaimed | Future release/product owner | Add release/archive process, artifact provenance, install/test/uninstall validation, publication policy, docs, and release-claim guards. |
| Shared-library packages and dynamic ABI behavior remain unclaimed | Future ABI/package owner | Add separate package/ABI product decision, shared-library build/install proof, ABI/versioning policy, platform validation, docs, and guards. |
| Broad package-manager distribution remains unclaimed | Future product owner after multiple provider decisions | Add multiple provider proofs or explicit product policy with provider-specific evidence and non-claim boundaries for unsupported providers. |

## Next-Sprint Readiness

Sprint 207 leaves package distribution in a precise continued-deferral state.

| Future need | Sprint 207 handoff |
| --- | --- |
| Current Epic 19 status | Start from `docs/planning/EPIC_19/PROJECT_PLAN.md`, which marks Sprint 207 closed and Sprints 208-216 pending. |
| Package claim changes | Run `bash scripts/package_manager_deferral_check.sh`, `bash scripts/static_package_deferral_check.sh`, relevant install checks, `make support-docs-guard`, and `make docs-check`. |
| Public provider promotion | Use Day 5, Day 6, Day 11, and Day 14 evidence requirements before changing wording. |
| Local Homebrew proof | Keep `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` as local proof only. |
| Source or header changes | Run `make format && make lint && make test` before closeout. |
| Retrospective source material | Use `WORKING_NOTES.md` and Day 1-Day 14 artifacts under `SPRINT_207/artifacts/`. |

## Final Assessment

Sprint 207 is complete as a package-provider decision and guard-hardening
sprint. It deliberately does not promote package-manager support; it closes the
residual by proving that the current support tier remains source install plus
developer-mode local Homebrew proof, with stronger guard and documentation
coverage around every retained package-provider non-claim.

The branch is ready for review as planning, documentation, guard, regression,
and package-evidence governance work for Sprint 207.
