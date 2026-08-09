# Sprint 144 Day 3 Blocker Reproduction And Evidence Baseline

## Purpose

Establish the before-change evidence record for the selected Sprint 144 lane:
macOS reviewed static-first install/export parity. This artifact separates
actual source/script blockers from support-tier bookkeeping drift, captures
locally feasible validation, and defines expected post-promotion outputs before
design and implementation begin.

## Selected Lane

Sprint 144 selected macOS reviewed install/export parity on Day 2.

The selected lane is limited to:

- macOS static-first Make install/`pkg-config` proof;
- macOS static-first CMake install/export proof;
- static package deferral proof attached to the CMake install/export path;
- documentation and report alignment for that exact static-first package
  contract.

It does not include:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager distribution;
- static/shared selectors;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broad macOS platform parity beyond the named static-first install/export
  proof.

## Surfaces Inspected

| Surface | Current selected-lane signal |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Runs the correct package proof scripts, but comments and job names explicitly call them supplemental confidence paths. |
| `.github/workflows/ci.yml` | Linux remains the reviewed static-first package-contract source of truth and should stay the strongest overall reviewed baseline. |
| `.github/workflows/windows-ci.yml` | Windows remains reviewed CMake subset plus supplemental CMake install/downstream confidence; should remain unchanged unless backup lane is activated. |
| `tests/test_install.sh` | Local Unix-side Make install/`pkg-config` proof owner. |
| `tests/test_cmake_install.sh` | Local Unix-side CMake install/export and downstream `find_package(Sparse)` proof owner. |
| `scripts/static_package_deferral_check.sh` | Static-first package deferral guard and no-shared/no-selector proof owner. |
| `README.md` | Current CI summary says macOS has supplemental Homebrew GCC, static-first Make install/`pkg-config`, and CMake install/export confidence. |
| `INSTALL.md` | Supported-platform and validation sections say macOS package proof is supplemental and does not claim full reviewed install/export parity. |
| `docs/maintainer_guide.md` | Maintainer guidance says macOS package lanes are supplemental and not reviewed macOS install/export parity. |
| `tests/corpus/manifests/report_families.tsv` | Package and CI rows are source-controlled advisory rows; they identify proof owners but are not fresh hosted-run logs. |

## Current Workflow Baseline

The current macOS workflow top comment says:

- Apple Clang path is reviewed;
- Homebrew GCC is supplemental;
- Make install/`pkg-config` and CMake install/export jobs are supplemental;
- the package jobs strengthen static-first package confidence;
- the package jobs do not claim reviewed install/export parity.

Current selected-lane jobs:

| Job key | Current job name | Current command(s) | Current support tier |
| --- | --- | --- | --- |
| `install-and-pkgconfig` | `macOS supplemental static-first install and pkg-config confidence path` | `bash tests/test_install.sh` | Supplemental |
| `cmake-install-export` | `macOS supplemental CMake install/export confidence path` | `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | Supplemental |

## Locally Feasible Baseline Commands

The selected macOS lane uses portable shell scripts that can be run locally.
These local results are useful implementation evidence, but they do not replace
hosted GitHub Actions `macos-latest` proof.

| Command | Result | Evidence interpretation |
| --- | --- | --- |
| `bash tests/test_install.sh` | Passed: 23 passed, 0 failed | The Make install/`pkg-config` proof currently validates static archive install, no shared artifacts, 19 headers, `.pc` variables, cflags/libs, static metadata wording, downstream compile/link/run, and uninstall cleanup on this machine. |
| `bash tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped | The CMake install/export proof currently validates static archive install, package config files, static imported target metadata, no source/build path leaks, no shared metadata, installed example, exact-version consumer, mismatched-version rejection, and `.pc` version on this machine. |
| `bash scripts/static_package_deferral_check.sh` | Passed | The static-first package deferral guard currently verifies shared-build rejection, static target declaration, static install metadata, absence of shared export/ABI metadata, no static/shared package selector, and deferred support wording. |

## Current Documentation Baseline

| Document | Current wording to change later | Baseline interpretation |
| --- | --- | --- |
| `README.md` | macOS has supplemental Homebrew GCC, static-first Make install/`pkg-config`, and CMake install/export confidence. | Public summary does not yet claim reviewed macOS install/export proof. |
| `INSTALL.md` supported platforms | macOS row says reviewed Apple Clang lane plus supplemental static-first Make install/`pkg-config` and CMake install/export confidence. | User-facing install docs still treat macOS package proof as supplemental. |
| `INSTALL.md` validation section | macOS supplemental package confidence does not claim full reviewed install/export parity. | This explicit non-claim must be revised only if CI proof is promoted. |
| `docs/maintainer_guide.md` package guidance | macOS remains narrower with supplemental static-first install/export confidence. | Maintainer guidance still blocks reviewed macOS install/export parity claims. |
| `docs/maintainer_guide.md` Sprint 112/133/143 package interpretation | macOS package lanes do not become reviewed macOS install/export parity. | Historical interpretation must remain understandable after promotion; update current state without erasing history. |

## Source-Level Blockers

No source-level blocker is currently identified for the selected macOS package
promotion lane.

The existing proof scripts already pass locally and do not require public
header, library source, CMake target-shape, Make install, or package metadata
changes for the narrow static-first promotion.

Potential source-level work is out of scope unless a later hosted macOS run
reveals a real portability failure.

## CI-Only Or Bookkeeping Drift

The main blocker is support-tier bookkeeping drift:

| Drift area | Current state | Required post-fix state |
| --- | --- | --- |
| macOS workflow top comment | Package jobs are supplemental and do not claim reviewed install/export parity. | Package jobs are reviewed macOS static-first install/export proof, scoped to the static-first package contract. |
| macOS package job names | Job names include `supplemental` and `confidence path`. | Job names identify reviewed static-first install/`pkg-config` and CMake install/export proof. |
| macOS package step names | Steps include `maintained supplemental` wording. | Steps identify reviewed package proof commands without broadening claims. |
| README CI summary | macOS package proof is described as confidence, not reviewed install/export proof. | README names macOS reviewed static-first install/export proof while keeping Homebrew GCC supplemental. |
| INSTALL platform table | macOS package proof is supplemental. | INSTALL describes the promoted reviewed macOS static-first install/export lane. |
| maintainer guide | macOS package lanes remain supplemental and do not become reviewed parity. | Maintainer guide names the new reviewed macOS static-first install/export status and preserves non-claims. |
| report/freshness interpretation | Package/CI rows are proof-owner metadata, not hosted proof. | Keep this interpretation; add selected-lane references only if source-owned semantics require it. |

## Expected Post-Promotion Evidence Matrix

| Evidence owner | Expected post-fix output |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Top comment and job names state reviewed macOS static-first install/export proof for Make install/`pkg-config`, CMake install/export, and static deferral checks. |
| macOS `install-and-pkgconfig` job | Runs `bash tests/test_install.sh` and remains scoped to static archive package metadata and downstream `pkg-config` proof. |
| macOS `cmake-install-export` job | Runs `bash tests/test_cmake_install.sh` and `bash scripts/static_package_deferral_check.sh`, preserving no-shared/no-selector checks. |
| README | Cross-platform CI summary says macOS now has reviewed static-first install/export proof, while Homebrew GCC remains supplemental. |
| INSTALL | Supported-platform table and install validation interpretation distinguish macOS reviewed static-first install/export proof from full shared-library/package-manager parity. |
| Maintainer guide | Current support-tier guidance matches workflow comments and preserves Windows/Linux boundaries. |
| Report index | Existing package and CI rows remain source-controlled advisory proof-owner rows unless Day 8 identifies a source-owned row update. |

## Failure Message Expectations

If the selected macOS package lane fails after promotion, the failure should
point to the exact proof owner:

- Make install/`pkg-config` failure: `tests/test_install.sh`;
- CMake install/export or downstream CMake consumer failure:
  `tests/test_cmake_install.sh`;
- unsupported shared metadata or selector drift:
  `scripts/static_package_deferral_check.sh`;
- support-tier wording mismatch: macOS workflow comments, README, INSTALL, or
  maintainer guide.

Failure messages must not imply:

- shared libraries are supported;
- dynamic ABI compatibility is reviewed;
- package-manager distribution exists;
- Windows package parity changed;
- macOS has broad platform parity beyond the static-first install/export lane.

## Hosted Evidence Boundary

Local validation passed, but selected-lane promotion still depends on hosted
GitHub Actions behavior:

- runner image: `macos-latest`;
- workflow file: `.github/workflows/macos-ci.yml`;
- commands: `bash tests/test_install.sh`,
  `bash tests/test_cmake_install.sh`, and
  `bash scripts/static_package_deferral_check.sh`;
- status owner: macOS reviewed static-first install/export lane.

Day 7 and Day 12 should keep this distinction explicit: local package proof is
baseline evidence, and the reviewed macOS claim is earned only by the hosted CI
lane after workflow promotion.

## Day 4 Handoff

Day 4 should design the selected-lane portability and support-tier update with
these constraints:

1. Prefer workflow/docs/report wording changes over source changes unless a
   concrete blocker appears.
2. Keep the macOS promotion static-first and package-lane-specific.
3. Preserve Linux as the strongest reviewed source of truth.
4. Preserve Windows as reviewed CMake subset plus supplemental CMake
   install/downstream confidence.
5. Do not modify Make install, CMake install/export, `sparse.pc`, or public
   headers without a new blocker.

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation starts from concrete blockers or proof gaps. | Complete | The blocker is support-tier bookkeeping drift around already-passing macOS package proof scripts. |
| Source blockers and workflow bookkeeping issues are separated. | Complete | Source-level blockers section is empty for the selected lane; CI/bookkeeping drift is listed separately. |
| Expected evidence is specific enough to avoid broad platform overclaims. | Complete | Expected evidence matrix and failure-message expectations scope the promotion to static-first macOS install/export proof only. |
