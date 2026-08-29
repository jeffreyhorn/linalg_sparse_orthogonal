# Sprint 188 Retrospective

**Sprint:** 188 - Homebrew Proof Completion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-188`)
**Status:** Complete with guarded residual

## Source Artifact Note

Sprint 188 was executed from the Epic 17 project-plan section for Sprint 188
and lives under `docs/planning/EPIC_17/SPRINT_188/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 188 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited the selected Homebrew local proof baseline, owner surfaces, local
      tool availability, current blocker, and support-claim risks.
- [x] Made a license strategy decision without inventing project license terms
      or choosing a guessed Homebrew license identifier.
- [x] Hardened `scripts/homebrew_local_formula_proof.sh` so missing or
      placeholder license metadata remains unavailable evidence and successful
      future paths validate archive contents, installed metadata, downstream
      test contract, uninstall, and cleanup behavior.
- [x] Updated package-manager and static-package guard coverage so support
      wording depends on the actual proof state.
- [x] Calibrated README, INSTALL, Homebrew package notes, and maintainer
      guidance to distinguish local proof material from user-facing Homebrew
      install support.
- [x] Ran the integrated package validation gate, including Homebrew proof
      blocker behavior, package guards, Make install proof, CMake install
      proof, documentation hygiene, generated-output scan, and C/header gate
      applicability review.
- [x] Recorded the final claim audit and closed the sprint as a guarded
      residual because approved standalone root license metadata is still
      absent.
- [x] Preserved explicit non-claims for Homebrew availability, Homebrew/core,
      bottles, Linuxbrew, public taps, other package managers, provider
      registry readiness, binary packages, shared-library package support,
      dynamic ABI compatibility, static/shared selectors, and broad
      package-manager distribution.

## What Went Well

1. **The blocker became sharper instead of fuzzier.** Days 1-3 separated
   environment availability from the real blocker: no approved standalone root
   `LICENSE`, `COPYING`, or `NOTICE` metadata exists, and no exact
   `SPARSE_HOMEBREW_LICENSE` value can be selected safely.

2. **The proof script now fails earlier and more clearly.** Day 5 moved
   license metadata validation before temporary archive creation, so the
   missing-license state no longer creates archive or checksum output that
   could be mistaken for partial support evidence.

3. **Future success paths are better guarded.** Days 5-7 added checks for
   required archive entries, installed static package metadata, unsupported
   provider/shared/ABI wording, exact-version downstream CMake consumers, and
   shared-artifact rejection in the formula test contract.

4. **Guard behavior now matches the actual proof state.** Day 9 taught the
   package-manager guard to accept the selected local Homebrew proof material
   while requiring exit `2`, unclaimed support wording, and no archive/render/
   install/test progress when root license metadata is absent.

5. **User and maintainer docs now say the same thing.** Days 10-11 aligned
   README, INSTALL, Homebrew README, and maintainer guidance around proof,
   blocker, failure, generated-output, and validation-command interpretation.

6. **Validation covered the changed surface, not just the narrow script diff.**
   Day 12 reran package guards plus Make and CMake install proofs because the
   sprint touched install-facing package guidance and installed metadata
   expectations.

## What Didn't Go Well

1. **The sprint goal could not be fully closed.** The Epic 17 plan expected a
   resolved Homebrew license metadata blocker and a passing local formula
   proof, but no authoritative license text or project-owner approval was
   present in the repository.

2. **The correct outcome is still awkward for users.** The project now has
   stronger Homebrew proof material, but users still must install from source
   via Make or CMake because Homebrew install support remains unclaimed.

3. **License metadata remains externally owned.** Code and documentation can
   keep the blocker safe, but they cannot decide project license terms or an
   accurate Homebrew license identifier.

4. **Claim wording remains distributed.** README, INSTALL, maintainer docs,
   package docs, proof scripts, guards, and sprint artifacts all carry part of
   the package support boundary.

5. **The local proof success path remains unexecuted.** The current expected
   exit is `2`; archive/render/install/`brew test` behavior is hardened and
   preflighted but cannot be fully proven until approved metadata exists.

## Final Metrics

### Validation

| Metric | Sprint 188 close state |
| --- | --- |
| expected unavailable Homebrew proof | passed: exit `2` |
| missing-license proof progress scan | passed: no archive/render/install/`brew test` work started |
| package-manager deferral guard | passed |
| static-package deferral guard | passed |
| Make install validation | passed: `bash tests/test_install.sh` |
| CMake install validation | passed: `bash tests/test_cmake_install.sh` |
| shell syntax validation | passed for changed shell scripts |
| final `git diff --check` | passed |
| trailing-whitespace scan | passed |
| Homebrew generated-output scan | passed |
| Sprint 188 markdown link check | passed |
| package report normalization checks | not required; package report metadata did not change |
| full C quality gate | not required; no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 188 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Public/maintainer/package docs changed | 4 |
| Shell scripts changed | 2 |
| C source files changed | 0 |
| Public header files changed | 0 |
| Package report files changed | 0 |
| Generated Homebrew outputs staged | 0 |

### Claim Governance

| Metric | Sprint 188 close state |
| --- | ---: |
| Homebrew install support claims added | 0 |
| Homebrew/core readiness claims added | 0 |
| bottle or hosted binary claims added | 0 |
| Linuxbrew claims added | 0 |
| public tap maintenance claims added | 0 |
| other package-manager claims added | 0 |
| provider registry readiness claims added | 0 |
| shared-library package claims added | 0 |
| dynamic ABI claims added | 0 |
| broad package-manager distribution claims added | 0 |

## Closed Claim

Sprint 188 closes this bounded implementation claim:

The selected local Homebrew proof path has been hardened, package guards and
documentation now reflect the actual proof state, the existing static package
install surface still passes Make and CMake install validation, and Homebrew
support remains explicitly unclaimed while approved standalone root license
metadata is absent.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-package-proof-intake.md](./artifacts/day1-package-proof-intake.md);
- [day2-license-strategy-decision.md](./artifacts/day2-license-strategy-decision.md);
- [day3-metadata-implementation.md](./artifacts/day3-metadata-implementation.md);
- [day4-formula-template-audit.md](./artifacts/day4-formula-template-audit.md);
- [day5-render-archive-hardening.md](./artifacts/day5-render-archive-hardening.md);
- [day6-install-surface-hardening.md](./artifacts/day6-install-surface-hardening.md);
- [day7-downstream-consumer-proof.md](./artifacts/day7-downstream-consumer-proof.md);
- [day8-end-to-end-proof-run.md](./artifacts/day8-end-to-end-proof-run.md);
- [day9-package-guard-alignment.md](./artifacts/day9-package-guard-alignment.md);
- [day10-user-facing-package-docs.md](./artifacts/day10-user-facing-package-docs.md);
- [day11-maintainer-package-docs.md](./artifacts/day11-maintainer-package-docs.md);
- [day12-integrated-package-validation.md](./artifacts/day12-integrated-package-validation.md);
- [day13-claim-audit.md](./artifacts/day13-claim-audit.md);
- [day14-closeout-summary.md](./artifacts/day14-closeout-summary.md).

No user-facing Homebrew install route, Homebrew/core readiness, bottle support,
Linuxbrew support, public tap support, other package-manager support, provider
registry readiness, binary package support, shared-library package support,
dynamic ABI guarantee, static/shared package selector, or broad
package-manager distribution claim was added.

## Residual

Sprint 188 leaves one explicit residual:

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Missing approved standalone root license metadata | Project/product decision | Add approved root `LICENSE`, `COPYING`, or `NOTICE`; set accurate `SPARSE_HOMEBREW_LICENSE`; rerun the local Homebrew proof to exit `0`; rerun package guards; update docs only to the exact proof level earned. |

## Next-Sprint Readiness

Sprint 189 can proceed without depending on Homebrew support promotion. The
Sprint 188 residual should remain bounded to package/provider proof work and
must not be folded into unrelated Windows, comparison, performance, or API
coherence claims.

| Future need | Sprint 188 handoff |
| --- | --- |
| Homebrew proof success | Resolve approved root license metadata and exact Homebrew license identifier before rerunning the proof for exit `0`. |
| Package claim updates | Run package-manager and static-package guards before promoting any support wording. |
| Install/package surface changes | Run Make and CMake install validation when installed metadata, examples, package docs, or downstream consumer paths change. |
| Broader provider support | Treat Homebrew/core, bottles, Linuxbrew, taps, vcpkg, Conan, pkgsrc, distro packages, and registries as separate product decisions with separate evidence. |

## Validation Retrospective

Sprint 188 changed shell scripts and documentation but no C source or public
header files. The selected validation set was therefore:

```sh
scripts/homebrew_local_formula_proof.sh
scripts/package_manager_deferral_check.sh
scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
```

The Homebrew proof command is expected to exit `2` until approved standalone
root license metadata exists. Any future `.c` or `.h` change must run:

```sh
make format
make lint
make test
```

## Carry Forward

- Resolve the project license metadata question before claiming a passing
  Homebrew formula proof.
- Keep placeholder Homebrew license identifiers as blocker evidence, not proof
  metadata.
- Keep generated formula, archive, tap, log, cache, install prefix, and bottle
  outputs out of source control.
- Promote only the exact local proof level earned by a passing proof and
  guards.
- Require separate evidence for Homebrew/core, bottles, Linuxbrew, public
  taps, other package managers, shared-library packages, dynamic ABI, and broad
  package-manager distribution.
