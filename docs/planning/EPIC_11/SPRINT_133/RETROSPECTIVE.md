# Sprint 133 Retrospective

**Sprint:** 133 - Package, ABI & Shared-Library Product Decision
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 133 started from the Epic 11 project-plan package/ABI scope.
- [x] package and ABI source areas were inventoried before implementation:
      public headers, version metadata, Make install, CMake install/export,
      `pkg-config`, downstream consumers, support docs, and validation scripts.
- [x] public header, symbol, and macro exposure was audited before any shared
      library decision was made.
- [x] Make install, CMake install/export, `sparse.pc`, version metadata, and
      current downstream consumer behavior were audited against the live tree.
- [x] the product decision was recorded explicitly: preserve static-first
      package support and defer shared-library packaging, dynamic ABI
      compatibility, and package-manager support.
- [x] `BUILD_SHARED_LIBS=ON` was changed from warning-only behavior to
      configure-time rejection under the static-first contract.
- [x] README, INSTALL, and maintainer guide wording were aligned with the
      selected static-first package contract.
- [x] static-first deferral proof was implemented:
  - `scripts/static_package_deferral_check.sh`
  - CMake rejection proof
  - explicit static target declaration proof
  - unsupported shared ABI metadata/selector absence proof
  - support wording drift proof
- [x] CMake installed consumer proof was strengthened:
  - exact installed header count
  - static imported target metadata
  - installed-prefix include/archive checks
  - source/build-tree path leak scans
  - installed `find_package(Sparse)` configure/build/run
- [x] Make install and `pkg-config` consumer proof was strengthened:
  - exact `.pc` variables
  - installed cflags/libs
  - exact version resolution
  - `--libs --static` parity for the current self-contained link surface
  - no `Libs.private` claim
  - unsupported package/ABI wording scan
  - downstream compile/link/run consumers
- [x] integrated package validation passed:
  - `bash tests/test_install.sh` passed 22 checks, 0 failures
  - `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips
  - `bash scripts/static_package_deferral_check.sh` passed
- [x] no tracked or untracked `.c`/`.h` changes were present at closeout, so
      the full `make format && make lint && make test` gate was not required
      by the Sprint 133 validation rule.
- [x] `git diff --check` and trailing-whitespace scans passed.
- [x] final closeout, residual queue, Sprint 134 handoff, and PR summary
      material were written.

## What Went Well

1. **The product decision happened before implementation.**
   Day 5 made the support choice explicit before the sprint changed build
   behavior or tests. That avoided accidentally implementing shared-library
   fragments without the ABI, loader, symbol, metadata, and platform proof
   needed to support the claim.

2. **Static-first support became enforceable.**
   Changing `BUILD_SHARED_LIBS=ON` to a configure-time rejection made the
   static-first contract visible at the first point of misuse. The new
   deferral guard script keeps that behavior and related support wording from
   silently drifting.

3. **Consumer proof moved from broad smoke checks to contract checks.**
   `tests/test_cmake_install.sh` now proves the installed CMake package exports
   a static imported target, uses installed-prefix paths, and does not leak
   source or build paths. `tests/test_install.sh` now proves exact `pkg-config`
   variables, flags, version behavior, and current static/private dependency
   semantics.

4. **The sprint separated package metadata from ABI guarantees.**
   Version metadata remains useful package/source metadata, while dynamic ABI
   compatibility stays a non-claim. The docs, CMake package behavior, and
   validation artifacts now repeat that distinction consistently.

5. **Residual work is specific instead of vague.**
   The closeout queue names shared-library packaging, dynamic ABI policy,
   package-manager recipes, CMake selectors, `Libs.private`, optional
   thread/OpenMP package validation, and CI promotion as separate future work.

6. **Validation matched the touched surfaces.**
   The branch changed build-system behavior, shell validation scripts, and
   documentation. The sprint ran Make install, CMake install/export,
   downstream consumers, static deferral proof, syntax checks, and hygiene
   checks without inventing an unrelated C-source validation requirement.

## What Didn't Go Well

1. **The current public header surface is not ABI-ready.**
   Day 2 confirmed many layout-sensitive structs, callback payloads, typedefs,
   enums, and solver option/result structures. Shared-library support remains
   blocked until that surface gets an explicit ABI policy.

2. **The prior `BUILD_SHARED_LIBS` behavior was too easy to misread.**
   Warning-only behavior could let users believe shared-library mode was
   supported. Sprint 133 fixed that, but the fact it needed fixing shows the
   package contract was previously too implicit.

3. **The `pkg-config` exact-flag checks exposed portability detail.**
   Day 12 initially assumed a path form that did not match this platform's
   `pkg-config` normalization, and the PR CI run showed another valid path
   formatting. The final test separates raw `.pc` variables from semantic
   emitted flag checks, but the false starts are a reminder that package proof
   needs to validate real tool behavior.

4. **Install/export proofs are still local.**
   The strengthened tests are valuable, but they are not yet reviewed CI lanes
   across every platform. The closeout keeps reviewed, supplemental, local,
   deferred, and unsupported evidence tiers separate.

5. **Optional package modes remain unvalidated.**
   The default static package contract is covered. Optional `SPARSE_MUTEX` and
   `SPARSE_OPENMP` package metadata and consumer behavior remain a future
   matrix item.

## Final Metrics

### Validation

| Metric | Sprint 133 close state |
|---|---:|
| tracked `.c`/`.h` changes | 0 |
| untracked `.c`/`.h` changes | 0 |
| Make install/pkg-config proof | 22 passed, 0 failed |
| CMake install/export proof | 21 passed, 0 failed, 0 skipped |
| static deferral proof | passed |
| shell syntax checks | passed |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |
| full C quality gate | not required; no `.c`/`.h` changes |

### Sprint 133 Artifact Package

| Metric | Sprint 133 close state |
|---|---:|
| total artifact files under `SPRINT_133/artifacts/` | 14 |
| audit and intake artifacts | 4 |
| decision and design artifacts | 3 |
| implementation/proof artifacts | 5 |
| validation and closeout artifacts | 2 |

Notes:

- audit and intake artifacts:
  - `day1-package-abi-intake.md`
  - `day2-public-header-symbol-audit.md`
  - `day3-install-shape-package-metadata-audit.md`
  - `day4-downstream-consumer-expectation-audit.md`
- decision and design artifacts:
  - `day5-package-abi-product-decision.md`
  - `day6-selected-static-contract-design.md`
  - `day9-static-deferral-proof-design.md`
- implementation/proof artifacts:
  - `day7-build-install-contract-batch.md`
  - `day8-package-metadata-documentation-batch.md`
  - `day10-static-deferral-proof-implementation.md`
  - `day11-downstream-cmake-consumer-proof.md`
  - `day12-pkg-config-consumer-proof.md`
- validation and closeout artifacts:
  - `day13-integrated-package-validation.md`
  - `day14-closeout-package-abi-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- shared-library packaging design and validation
- dynamic ABI compatibility policy, symbol inventory, and soname/install-name
  policy
- package-manager recipes and manager-specific downstream consumer proof
- CMake static/shared selector semantics if shared support is selected later
- `pkg-config` `Libs.private` split if dependency visibility changes
- optional `SPARSE_MUTEX` and `SPARSE_OPENMP` package metadata/consumer matrix
- reviewed CI promotion analysis for the local install/export proof scripts
- platform install parity decisions for macOS and Windows beyond current
  reviewed and supplemental tiers

Still consciously constrained rather than silently solved:

- no shared-library artifact support
- no dynamic ABI stability guarantee
- no runtime-loader behavior claim
- no package-manager support claim
- no static/shared package selection contract
- no broad platform install/export parity claim
- no ABI compatibility implication from package/source version metadata
- no reviewed CI status for the full local install proof stack

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-package-abi-intake.md](./artifacts/day1-package-abi-intake.md)
- [day2-public-header-symbol-audit.md](./artifacts/day2-public-header-symbol-audit.md)
- [day3-install-shape-package-metadata-audit.md](./artifacts/day3-install-shape-package-metadata-audit.md)
- [day4-downstream-consumer-expectation-audit.md](./artifacts/day4-downstream-consumer-expectation-audit.md)
- [day5-package-abi-product-decision.md](./artifacts/day5-package-abi-product-decision.md)
- [day6-selected-static-contract-design.md](./artifacts/day6-selected-static-contract-design.md)
- [day7-build-install-contract-batch.md](./artifacts/day7-build-install-contract-batch.md)
- [day8-package-metadata-documentation-batch.md](./artifacts/day8-package-metadata-documentation-batch.md)
- [day9-static-deferral-proof-design.md](./artifacts/day9-static-deferral-proof-design.md)
- [day10-static-deferral-proof-implementation.md](./artifacts/day10-static-deferral-proof-implementation.md)
- [day11-downstream-cmake-consumer-proof.md](./artifacts/day11-downstream-cmake-consumer-proof.md)
- [day12-pkg-config-consumer-proof.md](./artifacts/day12-pkg-config-consumer-proof.md)
- [day13-integrated-package-validation.md](./artifacts/day13-integrated-package-validation.md)
- [day14-closeout-package-abi-handoff.md](./artifacts/day14-closeout-package-abi-handoff.md)
