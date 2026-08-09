# Sprint 144 Day 4 Portability Design

## Purpose

Design the selected-lane source, script, path, shell, CMake, and workflow
changes before implementation. The selected lane remains macOS reviewed
static-first install/export parity.

## Design Decision

Day 4 does not identify a source portability defect. The selected macOS lane
already runs the maintained static-first package proof scripts, and those
scripts passed locally during Day 3.

The implementation path is therefore a support-tier promotion design:

- promote the two existing macOS static-first package jobs from supplemental
  confidence to reviewed macOS install/export proof;
- keep the commands unchanged unless a later CI failure exposes a concrete
  blocker;
- update wording and evidence ownership without changing package mechanics;
- preserve static-first non-claims from Sprint 143.

## Selected-Lane Scope

Promote only this macOS lane:

| Job | Command(s) | Promoted claim |
| --- | --- | --- |
| `install-and-pkgconfig` | `bash tests/test_install.sh` | Reviewed macOS static-first Make install/`pkg-config` proof. |
| `cmake-install-export` | `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | Reviewed macOS static-first CMake install/export proof plus static deferral guard. |

The promoted claim is limited to hosted `macos-latest` execution of the
maintained static-first package contract.

## Source/Script/Build Change Checklist

| Surface | Planned Day 5 action | Rationale |
| --- | --- | --- |
| `.github/workflows/macos-ci.yml` top comment | Update from supplemental package confidence to reviewed static-first package install/export proof. | This is the primary support-tier blocker. |
| macOS package job names | Rename away from `supplemental` and `confidence path`. | CI status should advertise reviewed proof accurately. |
| macOS package step names | Rename away from `maintained supplemental`. | Step output should identify the reviewed proof owner. |
| `tests/test_install.sh` | No planned change. | Day 3 proof passed and the script already owns static Make install/`pkg-config` validation. |
| `tests/test_cmake_install.sh` | No planned change. | Day 3 proof passed and the script already owns static CMake install/export validation. |
| `scripts/static_package_deferral_check.sh` | No planned change. | Day 3 proof passed and the script already guards static-first deferrals. |
| `CMakeLists.txt` and CMake package templates | No planned change. | No CMake package mechanics blocker exists for macOS promotion. |
| `Makefile` and `sparse.pc.in` | No planned change. | No install or `pkg-config` metadata blocker exists for macOS promotion. |
| Public headers and library source | No planned change. | The selected lane is package support-tier promotion, not API or ABI work. |

## Portability Rules

The selected lane must preserve these portability rules:

1. Use the same shell proof scripts on macOS that already define local
   Unix-side package proof.
2. Preserve shell quoting and temp-prefix behavior in
   `tests/test_install.sh` and `tests/test_cmake_install.sh`.
3. Preserve the static package deferral guard in the macOS CMake install/export
   job.
4. Do not add macOS-only script branches unless a hosted macOS failure requires
   one.
5. Do not add PowerShell or Windows-specific logic to close a macOS lane.
6. Do not change CTest counts; the selected lane does not promote or stage any
   compiled tests.
7. Keep report rows source-controlled and advisory unless a later report-index
   change explicitly updates source-owned semantics.

## Promoted, Added, Skipped, And Staged Test Decisions

| Test or proof surface | Decision | Reason |
| --- | --- | --- |
| `tests/test_install.sh` on macOS | Promoted from supplemental to reviewed macOS static-first install proof. | It validates the Make install/`pkg-config` static package path on hosted macOS. |
| `tests/test_cmake_install.sh` on macOS | Promoted from supplemental to reviewed macOS CMake install/export proof. | It validates installed CMake consumers, exact-version behavior, and static package metadata. |
| `scripts/static_package_deferral_check.sh` on macOS | Promoted as part of reviewed CMake install/export proof. | It keeps shared-library, ABI, and selector deferrals executable in the promoted lane. |
| macOS Homebrew GCC direct build/test | Kept supplemental. | Second-compiler coverage is useful but not part of package install/export parity. |
| macOS Apple Clang compile-quality/CMake/sanitize/wall checks | Kept reviewed. | Existing reviewed macOS path remains unchanged and separate from package promotion. |
| Windows CMake install/downstream proof | Kept supplemental. | Backup lane is not activated. |
| Windows staged `test_threads`, `test_sprint4_integration`, `test_fuzz` | Kept staged. | Known pthread/POSIX blockers are outside selected macOS lane. |
| Linux package contract proof | Kept reviewed source of truth. | macOS promotion complements Linux; it does not replace or weaken Linux ownership. |

## Implementation Order

Day 5 should implement in this order:

1. Update `.github/workflows/macos-ci.yml` top comment to describe reviewed
   macOS static-first install/export proof.
2. Rename `install-and-pkgconfig` job from supplemental confidence to reviewed
   static-first Make install/`pkg-config` proof.
3. Rename `cmake-install-export` job from supplemental confidence to reviewed
   static-first CMake install/export proof.
4. Rename package proof steps to remove `supplemental` wording.
5. Run workflow-focused text scans for stale macOS supplemental package
   wording.
6. Run `git diff --check` and any available YAML parse check.

README, INSTALL, maintainer guide, and report alignment should wait until Days
8-9 unless Day 5 needs a minimal cross-reference to keep workflow wording
coherent.

## Rollback And Stop Conditions

Rollback the Day 5 workflow promotion if:

- workflow syntax validation fails and the failure is not a trivial formatting
  issue;
- the macOS package jobs cannot be described as reviewed without implying
  shared-library, dynamic ABI, package-manager, or full platform parity claims;
- the implementation requires changing package scripts or build mechanics
  without a concrete failing proof;
- the promotion would require Windows or Linux support-tier changes beyond
  consistency wording;
- hosted macOS CI later fails because of a real script or package portability
  defect that cannot be fixed inside selected-lane scope.

Stop and ask before changing:

- public headers;
- `.c` files;
- shared/static package semantics;
- Make install mechanics;
- CMake target or install/export mechanics;
- Windows reviewed/supplemental boundaries;
- Linux reviewed package-contract ownership.

## Static-First Compatibility Review

The design preserves Sprint 143 package boundaries:

- no shared-library build/install/export support is added;
- no dynamic ABI compatibility claim is added;
- no runtime-loader proof is added;
- no package-manager support is added;
- no static/shared selector is added;
- no package metadata command surface changes;
- existing `pkg-config` and CMake consumer commands remain unchanged.

## Report Freshness Compatibility

Day 4 does not change report rows. Day 8 should decide whether
`tests/corpus/manifests/report_families.tsv` needs a source-owned semantics
update.

Current interpretation remains:

- package rows identify maintained proof-owner commands and templates;
- CI rows identify hosted checks whose logs live outside source control;
- report rows do not manufacture fresh hosted-run proof;
- macOS promotion must cite workflow jobs and CI results, not report freshness
  rows alone.

## Day 5 Implementation Checklist

- [ ] Update macOS workflow top comment.
- [ ] Rename macOS Make install/`pkg-config` package job.
- [ ] Rename macOS CMake install/export package job.
- [ ] Rename macOS package proof steps.
- [ ] Preserve commands exactly unless a validation failure requires otherwise.
- [ ] Scan workflow for stale selected-lane `supplemental` package wording.
- [ ] Run whitespace and workflow syntax checks.
- [ ] Update working notes with implementation evidence.

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Portability changes are scoped to the selected lane. | Complete | Source/script/build checklist changes only macOS workflow support-tier wording and job/step names. |
| Test promotion does not rely on stale expected counts or vague exclusions. | Complete | No CTest counts change; promoted proof surfaces are named scripts with existing commands. |
| Design preserves Sprint 143 static-first package boundaries. | Complete | Static-first compatibility review preserves no shared-library, ABI, loader, package-manager, or selector claims. |
