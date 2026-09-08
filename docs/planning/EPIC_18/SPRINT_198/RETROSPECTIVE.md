# Sprint 198 Retrospective

**Sprint:** 198 - Homebrew License Metadata and Formula Proof Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-198`)
**Status:** Closed with developer-mode local Homebrew static source formula
proof completed; broad package/Homebrew support remains unclaimed

## Source Artifact Note

Sprint 198 was executed from the Epic 18 project-plan section for Sprint 198
and lives under `docs/planning/EPIC_18/SPRINT_198/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint goal depended on an approved standalone root license metadata file,
an exact Homebrew formula license identifier, and a successful local formula
proof. The branch now includes root MIT metadata and the selected
`HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT` proof path. The recorded
macOS Intel x86_64 Tier 3 Homebrew run completed archive/checksum, temporary
tap render, source install, installed static package surface validation,
downstream `brew test`, uninstall, cleanup, and proof exit `0`.

## Definition Of Done Checklist

- [x] Created Sprint 198 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Audited root license metadata, Homebrew formula metadata, proof-script
      inputs, package guards, public docs, maintainer docs, install surfaces,
      and prior Sprint 188/Epic 18 residual evidence.
- [x] Recorded the initial missing-license blocker without guessing metadata,
      then added root MIT metadata after owner approval selected that path.
- [x] Preserved proof fail-safe behavior for missing or placeholder metadata
      while completing the MIT proof path through install, downstream test,
      uninstall, cleanup, and exit `0`.
- [x] Updated `scripts/package_manager_deferral_check.sh` to assert Sprint 198
      package proof artifacts and retained package-manager non-claims.
- [x] Updated README, INSTALL, Homebrew README, and maintainer guidance so the
      current status is bounded local proof wording, not broad support wording.
- [x] Ran package, static package, install, CMake install, docs, and whitespace
      validation for the bounded proof state.
- [x] Confirmed no `.c` or `.h` files changed, so the full C gate was not
      required by the user quality-check rule.
- [x] Confirmed no generated Homebrew proof outputs were staged and
      `scripts/__pycache__/` remains untracked generated cache.
- [x] Added root MIT license metadata, updated the proof to render through a
      temporary local tap, injected the local CMake bindir for Homebrew's
      formula environment, included the full CMake-declared source surface in
      the proof archive, and completed the developer-mode local static source
      formula proof without promoting broad package-manager support.

## What Went Well

1. **The sprint did not guess legal metadata.** Missing license inputs were
   recorded as a blocker rather than converted into a placeholder formula
   claim.

2. **The proof path stayed claim-safe.** The Homebrew proof exits before any
   archive/render/install/test stage when standalone root metadata is absent.

3. **Package-manager non-claims are now guarded by Sprint 198 evidence.** The
   package deferral guard checks MIT metadata, developer-mode proof records,
   temporary tap rendering, completed proof stages, and retained public
   non-claims.

4. **Public and maintainer docs now point at the exact proof scope.** README,
   INSTALL, Homebrew README, and maintainer guidance describe the
   developer-mode local static source formula proof without presenting
   Homebrew as an available install method.

5. **Validation matched the changed surface.** The sprint exercised package
   guards, install checks, CMake install checks, docs checks, and whitespace
   checks without running an unnecessary C implementation gate.

## What Didn't Go Well

1. **The proof required careful Homebrew environment classification.** The
   host is macOS Intel x86_64 Tier 3 for Homebrew, so the passing proof had to
   be recorded explicitly as developer-mode local evidence rather than broad
   Homebrew support.

2. **Several implementation days started as blocker-record days.** Days 3
   through 9 reviewed and documented the guarded proof path before the
   post-closeout MIT metadata addendum allowed archive/checksum/render stages
   to run.

3. **The success-state guard had to remain narrow.** The package guard now
   validates the completed local proof state, but still rejects broad provider
   claims outside the temporary local static source formula boundary.

4. **Package-manager support remains a product-scope residual.** The legal
   metadata blocker and local proof are resolved for MIT, but public tap,
   Homebrew/core, bottle, Linuxbrew, and other provider support remain outside
   the sprint evidence.

## Final Metrics

### Validation

| Metric | Sprint 198 close state |
| --- | --- |
| Homebrew local formula proof | passed with `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`: archive/checksum, temporary tap render, source install, installed static surface validation, downstream `brew test`, uninstall, cleanup, and exit `0` on macOS Intel x86_64 Tier 3 Homebrew |
| package-manager deferral guard | passed after Sprint 198 package proof checks were updated |
| static package deferral guard | passed |
| Make install validation | passed: 23 passed, 0 failed |
| CMake install validation | passed: 27 passed, 0 failed, 0 skipped |
| docs check | passed |
| final `git diff --check` | passed |
| generated Homebrew proof output scan | passed with no generated proof outputs under `packaging/homebrew` |
| final `.c`/`.h` diff check | passed with no output |
| final `make format && make lint && make test` | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 198 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 3 |
| Maintainer documentation files changed | 1 |
| Package guard scripts changed | 1 |
| Homebrew formula template files changed | 1 |
| Install metadata files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Blocked decision/proof items | 0 |
| Partial blocker-state guard/docs items | 0 |
| Complete current-surface validation items | 1 |
| Metadata/proof implementation items completed | 5 |
| Homebrew support items promoted | 0 |
| Package-manager support claims promoted | 0 |

The count covers Sprint 198 items 198.1 through 198.6.

## Closed Claim

Sprint 198 closes this bounded claim:

The current branch adds root MIT license metadata, selects
`HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`, renders the Homebrew proof
formula through a temporary local tap, injects the local CMake bindir into the
formula environment, includes CMake-declared benchmarks and tests in the proof
archive, preserves nonzero proof exit status during cleanup, completes the
developer-mode local static source formula proof, guards the bounded
package-manager state, aligns public and maintainer documentation with the
current proof boundary, and validates install/docs/package surfaces for that
state.

This claim does not include a user-facing Homebrew install path, Homebrew/core
readiness, bottles, Linuxbrew support, public tap maintenance, binary package
distribution, other package managers, shared-library package support, dynamic
ABI compatibility, runtime-loader behavior, broad package-manager support, or
state-of-the-art package ecosystem parity.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-package-metadata-intake.md](./artifacts/day1-package-metadata-intake.md);
- [day2-license-metadata-decision.md](./artifacts/day2-license-metadata-decision.md);
- [day3-root-metadata-implementation.md](./artifacts/day3-root-metadata-implementation.md);
- [day4-formula-metadata-wiring.md](./artifacts/day4-formula-metadata-wiring.md);
- [day5-archive-checksum-proof.md](./artifacts/day5-archive-checksum-proof.md);
- [day6-formula-render-validation.md](./artifacts/day6-formula-render-validation.md);
- [day7-install-surface-proof.md](./artifacts/day7-install-surface-proof.md);
- [day8-downstream-formula-test-proof.md](./artifacts/day8-downstream-formula-test-proof.md);
- [day9-end-to-end-proof-run.md](./artifacts/day9-end-to-end-proof-run.md);
- [day10-package-guard-promotion.md](./artifacts/day10-package-guard-promotion.md);
- [day11-public-package-docs.md](./artifacts/day11-public-package-docs.md);
- [day12-maintainer-planning-alignment.md](./artifacts/day12-maintainer-planning-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Public Homebrew tap or Homebrew/core support remains unclaimed | Package proof owner after a product decision selects a public provider path | Add public tap/Homebrew/core evidence, hosted validation, release metadata, and support docs before promoting any public install wording. |
| Bottle and Linuxbrew support remain unclaimed | Package proof owner after binary or Linux provider decisions | Add separate bottle/Linuxbrew evidence and guards. |
| Public package-manager support remains unclaimed | Documentation owner after broader provider evidence exists | Promote only the exact earned support tier and retain non-claims for unsupported providers, binary packages, shared libraries, dynamic ABI, and broad package-manager support. |

## Next-Sprint Readiness

Sprint 198 leaves package/Homebrew work in a precise bounded-proof state.

| Future need | Sprint 198 handoff |
| --- | --- |
| Public provider decision | Sprint 198 explicitly leaves Homebrew/core, public tap, bottles, Linuxbrew, and other provider support unclaimed. |
| Metadata implementation | Root MIT metadata and formula license injection are complete for the local proof path. |
| Formula proof execution | The developer-mode local static source formula proof completed through install, downstream test, uninstall, cleanup, and exit `0`. |
| Guard promotion | Day 10 plus the post-closeout update guard the bounded proof state and retained non-claims. |
| Documentation promotion | Public and maintainer docs now describe only the earned local static source formula proof. |
| Closeout validation | Day 13 and Day 14 record the package/install/docs validation baseline, with post-closeout proof evidence for the completed bounded state. |

## Final Assessment

Sprint 198 is complete as a bounded package/Homebrew evidence sprint. It
completes the developer-mode local static source formula proof for the MIT
path, but it does not complete broad Homebrew or package-manager support.

The branch is ready for review as documentation, guard, planning, and evidence
governance work for the package-manager blocker.
