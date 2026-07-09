# Sprint 115 Retrospective

**Sprint:** 115 - Residual Package Platform Parity & ABI Productization Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-115`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 115 started from Sprint 112 package/platform evidence and Sprint
      114 residual deferral boundaries.
- [x] Completed Sprint 112 package work was explicitly excluded from duplicate
      scope:
  - package surface audit;
  - static-first support-tier decision;
  - Make install and `pkg-config` proof;
  - CMake install/export proof;
  - downstream consumer proof;
  - platform-tier contract;
  - packaging docs alignment;
  - package/platform validation closeout.
- [x] Sprint 114 non-package residuals were explicitly kept outside Sprint 115:
  - eigensolver private-owner movement;
  - `s20_select_indices` movement;
  - `s20_lift_ritz_vectors` movement;
  - shift-invert setup/conversion movement;
  - `lanczos_iterate_op` movement;
  - broad direct/iterative generated-RHS oracle abstraction;
  - broad SVD proof-helper abstraction.
- [x] Published the Linux install proof CI no-promotion decision.
- [x] Published the macOS CMake install/export parity deferral.
- [x] Published the Windows install-validation lane deferral.
- [x] Audited Windows thread/fuzz/property staged exclusions and published the
      staged-exclusion follow-through.
- [x] Reviewed macOS backend/toolchain coverage, Homebrew GCC, OpenMP, and TSan
      assumptions and preserved reviewed/supplemental boundaries.
- [x] Published the shared-library/dynamic ABI future product contract.
- [x] Published the package-manager support future-work decision.
- [x] Consumed Sprint 114 package/platform-facing residuals without absorbing
      non-package source-boundary debt.
- [x] Captured final package/platform decision matrix, unsupported-claim
      checklist, Sprint 116 adoption QA handoff, and Sprint 117 closeout
      handoff.
- [x] Documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan across `docs/planning/EPIC_10/SPRINT_115`
- [x] Full C quality gate was not required because Sprint 115 changed only
      planning documentation and did not touch `.c` or `.h` files.

## What Went Well

1. **The sprint kept package/platform scope narrow.**
   Day 1 duplicate-fenced Sprint 112 completed package work and Sprint 114
   non-package residuals. That prevented the package sprint from silently
   becoming another source-boundary sprint.

2. **Install proof decisions stayed evidence-bounded.**
   Linux install scripts remained strong local Unix-side proof rather than a
   new reviewed CI lane. macOS CMake install/export parity and Windows
   install-validation were explicitly deferred instead of being implied by
   adjacent local or CMake-first evidence.

3. **Windows staged exclusions are now explicit through closeout.**
   The sprint audited `test_threads`, `test_sprint4_integration`, and
   `test_fuzz`, then preserved the Windows reviewed CTest count and the
   CMake-first consumer scope without claiming thread/fuzz/property parity.

4. **The macOS support model stayed coherent.**
   The reviewed Apple Clang lane, supplemental Homebrew GCC leg, supplemental
   Make install/`pkg-config` confidence path, local coverage targets, and TSan
   constraints are now recorded as separate evidence types.

5. **Product contracts were separated from install metadata.**
   Shared-library/dynamic ABI support and package-manager support are now
   future product contracts with concrete proof checklists. The sprint avoids
   treating `pkg-config`, `find_package(Sparse)`, or `DESTDIR` staged installs
   as package-manager or dynamic ABI support.

6. **Validation matched the touched surface.**
   Since the branch only added planning docs, Sprint 115 used documentation
   hygiene checks and did not run unrelated C quality gates.

## What Didn't Go Well

1. **Several package/platform decisions remain deferrals.**
   The sprint made support truth clearer, but it did not promote Linux install
   proof, macOS CMake install/export parity, Windows install-validation, or
   Windows thread/fuzz/property proof into reviewed lanes.

2. **Windows parity remains intentionally narrow.**
   Windows still lacks reviewed Makefile parity, separate install-validation,
   thread/fuzz/property proof, package-manager support, and dynamic/shared
   library evidence.

3. **macOS package confidence remains split.**
   macOS has useful reviewed and supplemental evidence, but local CMake
   install/export proof still has not become reviewed macOS install/export
   parity.

4. **Future productization work is substantial.**
   Shared-library/dynamic ABI and package-manager support both require real
   build, metadata, runtime, symbol, versioning, and downstream-consumer proof
   before public support wording can change.

## Final Metrics

### Validation

| Metric | Sprint 115 close state |
|---|---:|
| documentation hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed |
| full C quality gate | not required; no `.c` or `.h` changes |
| changed source/header files | 0 |
| changed workflow files | 0 |
| changed Make/CMake/package metadata files | 0 |
| reviewed CTest membership changes | 0 |
| public/install-header changes | 0 |
| package-manager recipes added | 0 |
| shared-library build rules added | 0 |
| dynamic ABI claims added | 0 |

### Package and Platform Decisions

| Surface | Sprint 115 close decision |
|---|---|
| Linux install proof | Local Unix-side proof only; no reviewed Linux install CI lane. |
| macOS CMake install/export | Deferred; no full reviewed macOS install/export parity claim. |
| Windows install-validation | Deferred; Windows remains reviewed CMake-first consumer subset only. |
| Windows thread/fuzz/property | Staged exclusions remain for `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |
| macOS backend/toolchain | Apple Clang reviewed; Homebrew GCC/install/coverage/TSan supplemental or local as documented. |
| Shared-library/dynamic ABI | Future product contract; static-first package story remains current truth. |
| Package managers | Future work; no Homebrew/vcpkg/distro/Windows package-manager recipe support. |
| Sprint 114 residuals | Package/platform claim fences consumed; non-package proof-owner residuals deferred. |

### Sprint 115 Artifact Package

| Metric | Sprint 115 close state |
|---|---:|
| artifact files under `SPRINT_115/artifacts/` | 14 |
| artifact lines before retrospective | 1655 |
| working notes lines before retrospective | 594 |
| plan lines | 462 |
| retrospective files | 1 |

Notes:

- intake, install, and platform artifacts:
  - `day1-residual-package-platform-intake.md`
  - `day2-linux-install-ci-promotion-design.md`
  - `day3-linux-install-ci-no-promotion-decision.md`
  - `day4-macos-cmake-install-export-design.md`
  - `day5-macos-cmake-install-export-deferral.md`
  - `day6-windows-install-validation-design.md`
  - `day7-windows-install-validation-deferral.md`
  - `day8-windows-thread-fuzz-portability-audit.md`
  - `day9-windows-thread-fuzz-staged-exclusion-follow-through.md`
- macOS, ABI, package-manager, residual, and closeout artifacts:
  - `day10-macos-backend-toolchain-follow-through.md`
  - `day11-shared-library-dynamic-abi-contract.md`
  - `day12-package-manager-support-decision.md`
  - `day13-sprint114-package-platform-residual-intake.md`
  - `day14-validation-package-platform-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Promote Linux install proof to reviewed CI only if a future sprint accepts
  the added runtime/dependency ownership and updates support wording.
- Promote macOS CMake install/export parity only with reviewed CI proof for
  `cmake --install`, installed CMake package files, downstream
  `find_package(Sparse)`, exact-version behavior, and static artifact shape.
- Add Windows install-validation only with MSVC `cmake --install`,
  downstream installed target lookup, compile/link/run proof, reviewed-count
  clarity, and non-claims.
- Port or split Windows thread/fuzz/property proof only with native Windows
  thread/temp-file behavior and explicit CTest count updates.
- Add shared-library/dynamic ABI support only with build rules, package
  metadata, runtime-loader proof, symbol policy, versioning policy, ABI tests,
  and platform ownership.
- Add package-manager support only with actual recipes and install/consumer
  proof for each claimed manager/platform.
- Carry Sprint 114 non-package residuals to Sprint 117 or post-Epic work
  unless explicitly promoted with source-list, CMake, focused consumer,
  reviewed CTest, and rollback evidence.

Still consciously constrained rather than silently solved:

- no reviewed Linux install CI lane;
- no full reviewed macOS CMake install/export parity;
- no Windows install-validation parity;
- no Windows thread/fuzz/property parity;
- no Windows Makefile parity;
- no macOS coverage reviewed-lane claim;
- no Homebrew GCC reviewed-lane promotion;
- no shared-library package support;
- no dynamic ABI compatibility guarantee;
- no package-manager support;
- no public API or install-header expansion;
- no source-list, helper-target, or reviewed CTest membership change.

Not carried forward as unresolved Sprint 115 debt:

- Sprint 115 residual package/platform intake and duplicate fence;
- Linux install proof CI promotion analysis and no-promotion decision;
- macOS CMake install/export design and deferral;
- Windows install-validation design and deferral;
- Windows thread/fuzz portability audit and staged-exclusion follow-through;
- macOS backend/toolchain follow-through;
- shared-library/dynamic ABI product-contract decision;
- package-manager support decision;
- Sprint 114 package/platform residual intake;
- Sprint 115 validation and package/platform handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-package-platform-intake.md](./artifacts/day1-residual-package-platform-intake.md)
- [day2-linux-install-ci-promotion-design.md](./artifacts/day2-linux-install-ci-promotion-design.md)
- [day3-linux-install-ci-no-promotion-decision.md](./artifacts/day3-linux-install-ci-no-promotion-decision.md)
- [day4-macos-cmake-install-export-design.md](./artifacts/day4-macos-cmake-install-export-design.md)
- [day5-macos-cmake-install-export-deferral.md](./artifacts/day5-macos-cmake-install-export-deferral.md)
- [day6-windows-install-validation-design.md](./artifacts/day6-windows-install-validation-design.md)
- [day7-windows-install-validation-deferral.md](./artifacts/day7-windows-install-validation-deferral.md)
- [day8-windows-thread-fuzz-portability-audit.md](./artifacts/day8-windows-thread-fuzz-portability-audit.md)
- [day9-windows-thread-fuzz-staged-exclusion-follow-through.md](./artifacts/day9-windows-thread-fuzz-staged-exclusion-follow-through.md)
- [day10-macos-backend-toolchain-follow-through.md](./artifacts/day10-macos-backend-toolchain-follow-through.md)
- [day11-shared-library-dynamic-abi-contract.md](./artifacts/day11-shared-library-dynamic-abi-contract.md)
- [day12-package-manager-support-decision.md](./artifacts/day12-package-manager-support-decision.md)
- [day13-sprint114-package-platform-residual-intake.md](./artifacts/day13-sprint114-package-platform-residual-intake.md)
- [day14-validation-package-platform-handoff.md](./artifacts/day14-validation-package-platform-handoff.md)
