# Sprint 198 Retrospective

**Sprint:** 198 - Homebrew License Metadata and Formula Proof Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-198`)
**Status:** Closed with metadata blocker; package/Homebrew support remains
unclaimed

## Source Artifact Note

Sprint 198 was executed from the Epic 18 project-plan section for Sprint 198
and lives under `docs/planning/EPIC_18/SPRINT_198/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint goal depended on an approved standalone root license metadata file
and an exact Homebrew formula license identifier. Those inputs were not
present in the repository, so the sprint closed the blocker safely instead of
inventing license terms or promoting Homebrew/package-manager support.

## Definition Of Done Checklist

- [x] Created Sprint 198 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Audited root license metadata, Homebrew formula metadata, proof-script
      inputs, package guards, public docs, maintainer docs, install surfaces,
      and prior Sprint 188/Epic 18 residual evidence.
- [x] Recorded that no approved standalone root `LICENSE`, `COPYING`, or
      `NOTICE` file exists and no exact Homebrew formula license identifier is
      selected.
- [x] Preserved proof fail-safe behavior: `homebrew_local_formula_proof.sh`
      exits `2` before archive, checksum, formula render, install,
      installed-surface validation, `brew test`, or uninstall work.
- [x] Updated `scripts/package_manager_deferral_check.sh` to assert Sprint 198
      metadata-blocker artifacts and retained package-manager non-claims.
- [x] Updated README, INSTALL, Homebrew README, and maintainer guidance so the
      current status is blocker/provenance wording, not support wording.
- [x] Ran package, static package, install, CMake install, docs, and whitespace
      validation for the blocker state.
- [x] Confirmed no `.c` or `.h` files changed, so the full C gate was not
      required by the user quality-check rule.
- [x] Confirmed no generated Homebrew proof outputs were staged and
      `scripts/__pycache__/` remains untracked generated cache.

## What Went Well

1. **The sprint did not guess legal metadata.** Missing license inputs were
   recorded as a blocker rather than converted into a placeholder formula
   claim.

2. **The proof path stayed claim-safe.** The Homebrew proof exits before any
   archive/render/install/test stage when standalone root metadata is absent.

3. **Package-manager non-claims are now guarded by Sprint 198 evidence.** The
   package deferral guard checks the Day 2 and Day 9 blocker records plus
   retained public non-claims.

4. **Public and maintainer docs now point at the current blocker.** README,
   INSTALL, Homebrew README, and maintainer guidance all describe the missing
   metadata prerequisite without presenting Homebrew as an available install
   method.

5. **Validation matched the changed surface.** The sprint exercised package
   guards, install checks, CMake install checks, docs checks, and whitespace
   checks without running an unnecessary C implementation gate.

## What Didn't Go Well

1. **The core support promotion could not happen.** Sprint 198 was planned to
   close Homebrew proof, but the required license decision was unavailable.

2. **Several implementation days became blocker-record days.** Days 3 through
   9 reviewed and documented the guarded proof path instead of producing a
   rendered formula, archive checksum, install run, or `brew test` result.

3. **Success-state guard promotion remains future work.** The package guard
   now validates blocker evidence, but it still cannot enforce proof-backed
   support wording until a successful proof exit `0` exists.

4. **Package-manager support remains a product/legal residual.** Local tool
   availability was not the blocker; the missing approved metadata is.

## Final Metrics

### Validation

| Metric | Sprint 198 close state |
| --- | --- |
| Homebrew local formula proof | expected exit `2` because no standalone root license metadata exists |
| package-manager deferral guard | passed after Sprint 198 blocker checks were added |
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
| Homebrew formula template files changed | 0 |
| Install metadata files changed | 0 |
| C implementation files changed | 0 |
| C test files changed | 0 |
| Public or internal header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Blocked decision/proof items | 3 |
| Partial blocker-state guard/docs items | 2 |
| Complete current-surface validation items | 1 |
| Homebrew support items promoted | 0 |
| Package-manager support claims promoted | 0 |

The count covers Sprint 198 items 198.1 through 198.6.

## Closed Claim

Sprint 198 closes this bounded claim:

The current branch records the exact Homebrew/package-manager metadata blocker,
keeps the Homebrew proof fail-safe before archive/render/install/test work,
guards the blocker evidence and retained package-manager non-claims, aligns
public and maintainer documentation with the current blocker state, and
validates install/docs/package surfaces for that state.

This claim does not include local Homebrew formula support, Homebrew/core
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
| Approved root license metadata is absent | Product/legal metadata owner | Add an approved standalone root `LICENSE`, `COPYING`, or `NOTICE` file. |
| Exact Homebrew formula license identifier is absent | Product/legal metadata owner | Select the exact formula license value matching the approved root metadata. |
| Formula render/install proof remains blocked | Package proof owner after metadata approval | Run source archive/checksum, render, install, installed-surface validation, `brew test`, uninstall, and cleanup to proof exit `0`. |
| Success-state package guard remains future work | Package guard owner after proof exit `0` | Update guards to require proof-backed support wording and reject stale blocker-only wording. |
| Public package-manager support remains unclaimed | Documentation owner after proof exit `0` | Promote only the exact earned support tier and retain non-claims for Homebrew/core, bottles, Linuxbrew, public taps, other package managers, binary packages, shared libraries, dynamic ABI, and broad package-manager support. |

## Next-Sprint Readiness

Sprint 198 leaves package/Homebrew work in a precise blocker state.

| Future need | Sprint 198 handoff |
| --- | --- |
| License approval | Day 2 records the missing root metadata and exact formula identifier inputs. |
| Metadata implementation | Day 3 and Day 4 identify the root and formula surfaces to update once approved inputs exist. |
| Formula proof execution | Day 5 through Day 9 describe the archive, checksum, render, install, downstream test, and cleanup path that must run after metadata approval. |
| Guard promotion | Day 10 shows the current blocker guard and the remaining success-state guard handoff. |
| Documentation promotion | Day 11 and Day 12 show the public and maintainer surfaces that must change only after proof exit `0`. |
| Closeout validation | Day 13 and Day 14 record the package/install/docs validation baseline for the current blocker state. |

## Final Assessment

Sprint 198 is complete as a blocker-safe package/Homebrew evidence sprint. It
does not complete Homebrew support because the required approved license
metadata is still absent.

The branch is ready for review as documentation, guard, planning, and evidence
governance work for the package-manager blocker.
