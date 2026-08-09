# Day 7 Cross-Platform Reconciliation

## Scope

Day 7 reconciles the platform support tiers against the available hosted CI
evidence from Day 6 and the passing local validation baseline from Day 5. It
does not promote any hosted support claim for `sprint-146` because the branch
has not produced hosted workflow runs yet. The latest hosted evidence remains
the green `master` baseline at commit `daac9a85d516f72100c34b90b92ec78941a72200`.

## Reconciled Platform Support-Tier Table

| Platform | Current Support Tier | Available Hosted Evidence | Day 5 Local Evidence | Reconciliation Result |
| --- | --- | --- | --- | --- |
| Linux | Strongest reviewed source of truth: reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and reviewed static-first package contract; supplemental runtime/sanitizer/bench-fast, TSan, and coverage. | `master` run `31335415785` succeeded for all Linux reviewed and supplemental jobs at `daac9a85`. | Local corpus/report/package/examples/QR/partial-SVD baseline passed on `sprint-146`. | Wording matches evidence. Branch-specific hosted Sprint 146 evidence remains missing until branch/PR CI runs. |
| macOS | Reviewed Apple Clang path plus reviewed static-first Make install/`pkg-config` and CMake install/export proof; Homebrew GCC remains supplemental. | `master` run `31335415782` succeeded for Apple Clang reviewed path, Homebrew GCC supplemental leg, reviewed install/`pkg-config`, and reviewed CMake install/export at `daac9a85`. | Local package scripts passed on macOS host, but local output is not hosted `macos-latest` proof. | Wording matches evidence. Reviewed macOS claims remain bounded to current workflow lanes and latest green master run. |
| Windows | Reviewed MSVC CMake-first consumer subset with `56` expected CTest registrations; supplemental CMake install/downstream confidence; staged pthread/POSIX and Make/`pkg-config` parity gaps. | `master` run `31335415791` succeeded for reviewed CMake subset and supplemental CMake install/downstream confidence at `daac9a85`. | No local Windows/MSVC execution available. | Wording matches evidence. No Windows Makefile, `pkg-config`, reviewed install-validation parity, or staged test closure claim is supported. |

## Linux Evidence Summary

The Linux support tier remains coherent:

- reviewed Makefile compile-quality path succeeded in hosted run `31335415785`
  job `93300350112`;
- reviewed CMake parity path succeeded in hosted run `31335415785` job
  `93300350118`;
- reviewed dead-code report and completeness path succeeded in hosted run
  `31335415785` job `93300350128`;
- reviewed static-first package contract succeeded in hosted run `31335415785`
  job `93300350105`;
- supplemental runtime/sanitizer/bench-fast path succeeded in hosted run
  `31335415785` job `93300350121`;
- supplemental TSan path succeeded in hosted run `31335415785` job
  `93300350072`;
- supplemental coverage path succeeded in hosted run `31335415785` job
  `93300350151`.

The Day 5 local baseline additionally passed corpus schema, report
normalization/freshness, static package deferral, Make install, CMake install,
examples build, QR corpus proof, partial-SVD corpus proof, and local
oracle/report refresh. That local evidence supports the Sprint 146 branch
content but does not replace hosted Linux CI for final PR status.

## macOS Evidence Summary

The macOS support tier remains coherent:

- Apple Clang reviewed path succeeded in hosted run `31335415782` job
  `93300350055`, including reviewed Makefile compile-quality, reviewed CMake
  parity, wall-check, and sanitizer steps;
- Homebrew GCC supplemental matrix leg succeeded in hosted run `31335415782`
  job `93300350045`;
- reviewed static-first Make install/`pkg-config` proof succeeded in hosted
  run `31335415782` job `93300350051`;
- reviewed static-first CMake install/export proof and static deferral guard
  succeeded in hosted run `31335415782` job `93300350039`.

The current docs correctly limit macOS package support to reviewed
static-first install/export proof for the maintained static archive package
contract. They do not imply shared-library packaging, dynamic ABI,
runtime-loader compatibility, package-manager support, static/shared
selectors, broad macOS platform parity, portable performance, or
state-of-the-art status.

## Windows Evidence And Staged-Blocker Summary

The Windows support tier remains coherent:

- reviewed MSVC CMake consumer subset succeeded in hosted run `31335415791`
  job `93300350063`;
- the workflow expects `EXPECTED_WINDOWS_CTEST_COUNT=56`;
- configure, Release build, `ctest -N`, and `ctest --output-on-failure`
  succeeded on `windows-2022`;
- supplemental CMake install/downstream confidence succeeded in hosted run
  `31335415791` job `93300350119`;
- the supplemental job checks static `.lib` install, no DLL artifacts,
  installed headers, CMake package metadata, installed `sparse.pc`, installed
  example output, exact-version consumer, and mismatched-version rejection.

The following remain staged or deferred:

| Surface | Status | Owner/Blocker | Promotion Gate |
| --- | --- | --- | --- |
| `test_threads` on Windows | staged | pthread API dependency | Source portability change or Windows-native equivalent plus intentional CTest count update. |
| `test_sprint4_integration` on Windows | staged | pthread API dependency | Source portability change or Windows-native equivalent plus intentional CTest count update. |
| `test_fuzz` on Windows | staged | POSIX temp-file API dependency | Windows-compatible temp-file implementation or Windows-native equivalent plus intentional CTest count update. |
| Windows Makefile parity | staged | no maintained Windows Makefile reviewed lane | Workflow, docs, and hosted proof for a supported Makefile route. |
| Windows `pkg-config` parity | staged | no Windows pkg-config support decision | Toolchain/support decision plus hosted proof and docs/report updates. |
| Windows reviewed install-validation parity | staged | current install/downstream job is explicitly supplemental | Product decision, workflow wording, report row update, hosted proof, and claim audit. |

## Support-Tier Fix Or Defer List

| Finding | Action |
| --- | --- |
| README support-tier summary matches current workflows. | No code/doc fix needed on Day 7. Recheck during Day 8 public claim audit. |
| INSTALL supported-platforms table matches current workflows. | No code/doc fix needed on Day 7. Recheck during Day 8 public claim audit. |
| Maintainer guide package/platform interpretation matches current workflows. | No code/doc fix needed on Day 7. Recheck during Day 9 support audit. |
| Report-family CI row correctly treats hosted CI logs as external proof. | No code/doc fix needed on Day 7. Recheck during Day 9 support audit. |
| `sprint-146` has no hosted branch runs. | Defer to final PR/branch CI. Do not claim Sprint 146 hosted pass yet. |
| Local Day 5 baseline is green. | Keep as local Sprint 146 proof; do not use as hosted platform proof. |
| Latest hosted master baseline is green at `daac9a85`. | Cite as latest inspected master baseline only. Do not overstate as branch-specific Sprint 146 CI. |

## Support-Tier Mismatch Decision

No current support-tier wording mismatch was found that required a Day 7 source
change. The remaining mismatch risk is temporal rather than textual: the
branch has local validation evidence but no hosted branch/PR CI evidence yet.
Final closeout must update this status after the branch is pushed and PR checks
run, or explicitly state that hosted Sprint 146 evidence was unavailable at the
time of closeout.

## Day 8 Handoff

Day 8 should audit public docs and selected headers for unsupported claims
using this reconciled platform table. The audit should preserve these support
tiers:

- Linux: strongest reviewed source of truth, plus explicitly supplemental
  lanes.
- macOS: reviewed Apple Clang path and reviewed static-first install/export
  proof only.
- Windows: reviewed CMake-first subset plus supplemental CMake
  install/downstream confidence; staged pthread/POSIX and Make/`pkg-config`
  gaps remain visible.
- All platforms: no shared-library ABI, package-manager, portable
  performance, broad parity, or state-of-the-art claim.
