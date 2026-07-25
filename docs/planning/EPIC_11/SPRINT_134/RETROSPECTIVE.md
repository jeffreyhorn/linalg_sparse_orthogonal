# Sprint 134 Retrospective

**Sprint:** 134 - Cross-Platform Install, Windows Staged Lanes & CI Tier Follow-Through
**Duration:** 14 days (Days 1-14 landed on branch `sprint-134`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 134 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read the Sprint 134 Epic 11 project-plan scope and Sprint 133
      static-first package/ABI handoff before implementation.
- [x] Audited Linux, macOS, and Windows platform support tiers before making
      workflow changes.
- [x] Promoted Linux package proof to a reviewed static-first package-contract
      CI lane.
- [x] Added supplemental macOS CMake install/export confidence while preserving
      the narrower reviewed Apple Clang lane.
- [x] Designed and implemented supplemental Windows CMake install/downstream
      confidence for the CMake-first consumer story.
- [x] Re-audited Windows staged CTest exclusions and preserved the reviewed
      Windows expected count at `54`.
- [x] Reinforced Windows staged blockers:
  - `test_threads` remains staged because it uses pthread APIs directly;
  - `test_sprint4_integration` remains staged because it uses pthread APIs
    directly;
  - `test_fuzz` remains staged because it uses POSIX temp-file APIs through
    `unistd.h`, `mkstemps`, `close`, and `unlink`.
- [x] Aligned README, INSTALL, maintainer guide, workflow comments, and Sprint
      artifacts with final reviewed/supplemental/staged/deferred support
      tiers.
- [x] Published final platform support truth, staged-exclusion register,
      residual queue, PR summary material, and Sprint 135 handoff notes.
- [x] Integrated validation passed:
  - workflow YAML parse for Linux, macOS, and Windows passed;
  - package proof script syntax checks passed;
  - `bash tests/test_install.sh` passed 22 checks, 0 failures;
  - `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips;
  - `bash scripts/static_package_deferral_check.sh` passed;
  - local CMake/CTest registration audit reported 57 non-Windows tests,
    reconciling to Windows 54 after three staged exclusions.
- [x] `git diff --check` and focused trailing-whitespace scans passed.
- [x] No `.c`, `.h`, or `CMakeLists.txt` changes were present, so the full
      `make format && make lint && make test` gate was not required by the
      Sprint 134 validation rule.
- [x] Post-PR CI follow-up fixed `tests/test_install.sh` so
      `pkg-config --cflags` validation parses shell tokens and tolerates
      harmless trailing whitespace while still requiring the installed include
      directory.
- [x] Post-PR CI follow-up validation passed:
  - `bash -n tests/test_install.sh`
  - `bash tests/test_install.sh` passed 22 checks, 0 failures
  - `bash scripts/static_package_deferral_check.sh`
- [x] Post-PR Windows CI follow-up fixed the supplemental
      install/downstream proof so captured example output is joined before
      matching `OK`.

## What Went Well

1. **Linux package proof moved into the reviewed tier.**
   Sprint 134 converted the static-first package proof stack from local-only
   evidence into a reviewed Linux CI lane. That lane now runs Make
   install/`pkg-config`, CMake install/export, and static deferral proof
   together.

2. **macOS confidence improved without overstating parity.**
   The sprint added supplemental macOS CMake install/export proof while keeping
   reviewed macOS support scoped to Apple Clang compile-quality, CMake parity,
   wall-check, and sanitizer coverage.

3. **Windows gained installed-consumer evidence without changing CTest scope.**
   The new Windows supplemental job proves the CMake-first install/downstream
   path separately from the reviewed CMake/CTest subset. `EXPECTED_WINDOWS_CTEST_COUNT`
   remains `54`, so CTest membership did not drift.

4. **Windows staged exclusions now have concrete blockers.**
   The sprint stopped treating the staged list as just a count adjustment.
   `test_threads`, `test_sprint4_integration`, and `test_fuzz` now have named
   source-level blockers and future promotion gates.

5. **Support wording stayed tiered across public and maintainer surfaces.**
   README, INSTALL, maintainer guide, and workflow comments now consistently
   separate reviewed, supplemental, local, staged, deferred, and unsupported
   evidence.

6. **Validation matched the real changed surface.**
   The branch changed workflows and documentation, not C implementation. The
   sprint still ran package proofs, static deferral proof, workflow syntax,
   CTest registration, and docs hygiene without inventing unrelated C-source
   churn.

## What Didn't Go Well

1. **Hosted-runner proof remains the final word for new supplemental lanes.**
   Local validation can prove package semantics and workflow syntax, but this
   host cannot execute macOS hosted-runner behavior or the Windows MSVC
   PowerShell install/downstream job.

2. **Windows staged tests still require source portability work.**
   The sprint clarified blockers but did not remove them. Promotion requires
   Windows-native equivalents or portability wrappers plus CTest count updates
   and hosted MSVC configure/build/execute proof.

3. **Working notes needed cleanup late in the sprint.**
   Day sections became out of chronological order during iterative updates.
   Day 14 fixed the ordering mechanically, but future sprints should keep the
   working-note structure clean as part of each day closeout.

4. **Historical artifacts can look stale without the Day 12/14 context.**
   Day 1 and Day 2 artifacts correctly describe pre-decision state. The sprint
   now has final truth artifacts, but readers need to use Day 12 or Day 14 for
   the current support model.

## Final Metrics

### Validation

| Metric | Sprint 134 close state |
|---|---:|
| tracked `.c`/`.h` changes | 0 |
| `CMakeLists.txt` changes | 0 |
| package proof script changes | 1 |
| workflow YAML parse | passed |
| Make install/pkg-config proof | 22 passed, 0 failed |
| CMake install/export proof | 21 passed, 0 failed, 0 skipped |
| static deferral proof | passed |
| local CTest registration count | 57 |
| Windows expected CTest count | 54 |
| staged Windows exclusions | 3 |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |
| full C quality gate | not required; no `.c`/`.h` changes |

### Sprint 134 Artifact Package

| Metric | Sprint 134 close state |
|---|---:|
| total artifact files under `SPRINT_134/artifacts/` | 14 |
| audit and intake artifacts | 4 |
| decision and design artifacts | 4 |
| implementation/follow-through artifacts | 4 |
| validation and closeout artifacts | 2 |

Notes:

- audit and intake artifacts:
  - `day1-platform-install-intake.md`
  - `day2-platform-gap-audit.md`
  - `day5-macos-install-export-parity-audit.md`
  - `day10-windows-staged-test-reaudit.md`
- decision and design artifacts:
  - `day3-linux-install-ci-decision.md`
  - `day6-macos-install-export-decision.md`
  - `day8-windows-install-validation-design.md`
  - `day12-support-tier-doc-alignment.md`
- implementation/follow-through artifacts:
  - `day4-linux-install-ci-implementation.md`
  - `day7-macos-install-export-implementation.md`
  - `day9-windows-install-validation-implementation.md`
  - `day11-windows-staged-lane-follow-through.md`
- validation and closeout artifacts:
  - `day13-integrated-platform-validation.md`
  - `day14-platform-tier-closeout-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- hosted-runner history for the new macOS supplemental CMake install/export
  lane;
- hosted-runner history for the new Windows supplemental CMake
  install/downstream lane;
- future decision on whether macOS supplemental package confidence should ever
  become reviewed install/export parity;
- future decision on whether Windows supplemental install/downstream confidence
  should ever become reviewed install-validation parity;
- Windows-native or portable replacements for pthread/POSIX-bound staged tests;
- exact CTest count updates if any Windows staged test is promoted;
- package-manager support design and proof;
- shared-library packaging, dynamic ABI policy, runtime-loader validation, and
  package metadata ownership.

Still consciously constrained rather than silently solved:

- no full reviewed macOS install/export parity claim;
- no separate reviewed Windows install-validation claim;
- no Windows Makefile parity claim;
- no Windows `pkg-config` support claim;
- no reviewed Windows thread/fuzz/property claim;
- no package-manager support claim;
- no shared-library packaging or dynamic ABI compatibility claim;
- no runtime-loader behavior claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-platform-install-intake.md](./artifacts/day1-platform-install-intake.md)
- [day2-platform-gap-audit.md](./artifacts/day2-platform-gap-audit.md)
- [day3-linux-install-ci-decision.md](./artifacts/day3-linux-install-ci-decision.md)
- [day4-linux-install-ci-implementation.md](./artifacts/day4-linux-install-ci-implementation.md)
- [day5-macos-install-export-parity-audit.md](./artifacts/day5-macos-install-export-parity-audit.md)
- [day6-macos-install-export-decision.md](./artifacts/day6-macos-install-export-decision.md)
- [day7-macos-install-export-implementation.md](./artifacts/day7-macos-install-export-implementation.md)
- [day8-windows-install-validation-design.md](./artifacts/day8-windows-install-validation-design.md)
- [day9-windows-install-validation-implementation.md](./artifacts/day9-windows-install-validation-implementation.md)
- [day10-windows-staged-test-reaudit.md](./artifacts/day10-windows-staged-test-reaudit.md)
- [day11-windows-staged-lane-follow-through.md](./artifacts/day11-windows-staged-lane-follow-through.md)
- [day12-support-tier-doc-alignment.md](./artifacts/day12-support-tier-doc-alignment.md)
- [day13-integrated-platform-validation.md](./artifacts/day13-integrated-platform-validation.md)
- [day14-platform-tier-closeout-handoff.md](./artifacts/day14-platform-tier-closeout-handoff.md)

## Sprint 135 Handoff

- Treat Linux as the only reviewed static-first package-contract CI owner.
- Treat macOS package install/export jobs as supplemental until hosted-runner
  history justifies a reviewed parity decision.
- Treat Windows install/downstream proof as supplemental until hosted-runner
  history justifies a reviewed install-validation decision.
- Do not promote Windows staged thread/fuzz/property tests without source
  portability changes, exact CTest count updates, and hosted MSVC proof.
- Preserve Sprint 133 static-first package/ABI non-claims unless a new product
  decision funds shared-library packaging, dynamic ABI compatibility,
  runtime-loader validation, and package metadata ownership.
