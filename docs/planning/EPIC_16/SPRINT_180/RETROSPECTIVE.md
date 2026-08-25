# Sprint 180 Retrospective

**Sprint:** 180 - Package-Manager Provider Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-180`)
**Status:** Complete

## Source Artifact Note

Sprint 180 was executed from the Epic 16 project-plan section for Sprint 180
and lives under `docs/planning/EPIC_16/SPRINT_180/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 180 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Reconciled the Sprint 171 package-manager deferral guard and Sprint 177
      package-manager acceptance gate before implementation.
- [x] Audited current package metadata, install/export proof, downstream
      proof, package-manager wording, and unsupported provider surfaces.
- [x] Evaluated vcpkg, Homebrew, Conan, and pkgsrc against static-first fit,
      CI feasibility, recipe complexity, maintenance cost, user value, proof
      completeness, and claim risk.
- [x] Selected exactly one provider implementation path: local Homebrew
      formula/tap proof.
- [x] Preserved explicit non-claims for Homebrew/core, bottles, Linuxbrew,
      public taps, provider-hosted binaries, broad package-manager support,
      shared-library support, dynamic ABI support, and static/shared selectors.
- [x] Designed and added the source-controlled Homebrew formula template.
- [x] Added Homebrew provider notes that keep generated formula, tap, archive,
      bottle, cache, log, and install output out of source control.
- [x] Designed and implemented `scripts/homebrew_local_formula_proof.sh`.
- [x] Implemented claim-safe unavailable behavior for the current missing
      standalone license metadata blocker.
- [x] Updated `scripts/package_manager_deferral_check.sh` from a pure
      deferral guard into a provider claim guard for the selected local proof
      boundary.
- [x] Updated README, INSTALL, and maintainer guidance to describe selected
      proof artifacts without claiming Homebrew or package-manager support.
- [x] Ran provider, package-manager, static package, install/export,
      report-index, generated-output hygiene, syntax, and whitespace checks.

## What Went Well

1. **The provider question ended with one bounded decision.** Sprint 180 did
   not leave vcpkg, Homebrew, Conan, and pkgsrc as parallel half-options. It
   selected local Homebrew formula/tap proof as the implementation path and
   recorded why the other providers were rejected as first-provider candidates.

2. **The selected artifact is useful without over-claiming support.** The
   source-controlled template and local proof script create a concrete path to
   Homebrew validation, while public docs still say package-manager support is
   unavailable.

3. **The proof fails at the right boundary.** The script can create a temporary
   source archive and checksum, but it stops before formula rendering when no
   standalone license metadata file exists. That makes the blocker visible
   without implying a successful Homebrew install.

4. **The guard now matches the product state.** The package-manager guard
   allows the exact selected Homebrew local-proof material while still
   rejecting unselected provider recipes, generated Homebrew output, provider
   metadata drift, and public support claims.

5. **The static-first package contract stayed intact.** Make and CMake install
   validation continued to prove static archive install/export, downstream
   consumption, exact-version behavior, pkg-config metadata, and cleanup.

6. **The sprint left a clear revisit path.** Sprint 181 or a later provider
   sprint can start from one blocker: approved standalone license metadata,
   followed by a successful render/install/`brew test`/uninstall/cleanup proof.

## What Didn't Go Well

1. **The selected Homebrew proof remains unclaimed.** The sprint selected and
   implemented a provider proof path, but the current repository cannot run the
   Homebrew formula install path until standalone license metadata exists.

2. **The local formula path is macOS-weighted.** Homebrew had the best
   immediate proofability, but it does not address Windows package-manager
   user value the way a future vcpkg proof might.

3. **The public claim boundary is still distributed.** README, INSTALL,
   maintainer guide, package metadata, provider notes, guard scripts, and
   sprint artifacts all carry pieces of the same package-manager support
   posture.

4. **The proof script has an unexecuted success branch.** Rendering, local
   Homebrew install, `brew test`, uninstall, and cleanup are implemented, but
   remain unproven in this repository state because the script correctly stops
   at the license gate.

5. **Guarded wording now requires maintenance discipline.** Future wording
   changes around package-manager support must update docs and guard checks
   together or the claim guard should fail.

## Final Metrics

### Validation

| Metric | Sprint 180 close state |
| --- | --- |
| Homebrew local formula proof | passed as claim-safe unavailable: exit `2` on missing standalone license metadata |
| package-manager provider claim guard | passed: `bash scripts/package_manager_deferral_check.sh` |
| static package deferral guard | passed: `bash scripts/static_package_deferral_check.sh` |
| Make install validation | passed: `bash tests/test_install.sh` with 23 checks |
| CMake install/export validation | passed: `bash tests/test_cmake_install.sh` with 27 checks |
| package report-index normalization | passed: `python3 scripts/normalize_report_index.py --family package --check` |
| proof script syntax | passed: `bash -n scripts/homebrew_local_formula_proof.sh` |
| provider guard syntax | passed: `bash -n scripts/package_manager_deferral_check.sh` |
| formula template syntax | passed: `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` |
| generated Homebrew output hygiene | passed: no committed formula, archive, log, bottle, or tap output |
| documentation whitespace hygiene | passed: `git diff --check` |
| C source/header quality gate | not required: no `*.c` or `*.h` files changed |

### Changed Surface

| Metric | Sprint 180 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 0 |
| package-manager guard scripts changed | 1 |
| provider proof scripts added | 1 |
| provider template files added | 1 |
| provider notes files added | 1 |
| public/maintainer docs changed | 3 |
| daily artifacts | 14 |
| closeout artifacts | 1 |
| retrospective files | 1 |
| project-plan items completed | 6 |

### Claim Governance

| Metric | Sprint 180 close state |
| --- | ---: |
| selected provider proof paths chosen | 1 |
| public Homebrew support claims added | 0 |
| broad package-manager support claims added | 0 |
| Homebrew/core claims added | 0 |
| bottle or binary-package claims added | 0 |
| Linuxbrew claims added | 0 |
| vcpkg, Conan, or pkgsrc support claims added | 0 |
| shared-library support claims added | 0 |
| dynamic ABI support claims added | 0 |
| generated Homebrew artifacts committed | 0 |

## Closed Claim

Sprint 180 closes this Epic 16 package-manager provider decision claim:

The first package-manager provider implementation path is local Homebrew
formula/tap proof. The repository now contains a source-controlled Homebrew
formula template, provider notes, a local proof script, guard behavior, and
public docs that describe this selected proof path without claiming provider
support.

This does not claim Homebrew support, Homebrew/core readiness, bottles,
Linuxbrew support, public tap support, hosted binary packages, broad
package-manager support, vcpkg support, Conan support, pkgsrc support,
shared-library support, dynamic ABI support, or static/shared selector support.
The local Homebrew proof remains unclaimed until standalone license metadata
exists and the proof script completes render, install, `brew test`, uninstall,
and cleanup successfully.

## Follow-Up Risks

1. **Standalone license metadata blocks proof completion.** Add an approved
   `LICENSE`, `COPYING`, or `NOTICE` file before expecting Homebrew formula
   rendering or install proof to pass.

2. **Homebrew success behavior still needs execution.** After license metadata
   exists, rerun `bash scripts/homebrew_local_formula_proof.sh` and require
   local formula render, source install, static installed-file checks,
   `brew test`, uninstall, and cleanup to pass.

3. **vcpkg remains the strongest rejected alternative.** If Windows
   package-manager user value becomes higher priority, revisit vcpkg with
   local tool availability, overlay recipe proof, source archive/checksum
   policy, and license metadata in place.

4. **Provider docs and guard strings can drift together.** Any future package
   wording changes should run the package-manager guard and update checked
   strings deliberately.

5. **Generated provider output must stay out of source control.** Temporary
   formula, tap, archive, bottle, cache, build, install, and log material
   should remain proof-script output only.

## Sprint 181 Readiness

Sprint 181 should begin from the Epic 16 project-plan section
`Sprint 181: Selected Report Target Manifest`.

The highest-value next action is to centralize selected report target metadata
for oracle, comparison, performance, artifact, expected-row, and support-tier
surfaces without widening package-manager claims. Sprint 180 leaves package
manager work in a stable state: one selected local Homebrew proof path, a
claim-safe license blocker, and guards that should remain green while Sprint
181 focuses on report-target manifest ownership.
