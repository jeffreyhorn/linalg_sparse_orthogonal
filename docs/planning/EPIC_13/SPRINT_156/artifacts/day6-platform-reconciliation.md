# Sprint 156 Day 6: Platform Reconciliation

## Purpose

Reconcile the final Epic 13 platform and CI evidence across Linux, macOS,
Windows, reviewed lanes, supplemental lanes, staged/deferred surfaces, local
proof, and external-service failures. The goal is to keep every platform claim
mapped to a specific maintained lane and keep non-claims visible before the
Sprint 156 public claim audit.

## Inputs Reviewed

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_13/SPRINT_148/artifacts/day14-closeout-handoff.md`
- `docs/planning/EPIC_13/SPRINT_149/artifacts/day14-closeout-handoff.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day4-local-baseline.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day5-package-validation.md`

## Support Labels

| Label | Meaning |
| --- | --- |
| Reviewed hosted proof | Enforced workflow lane on GitHub-hosted CI. |
| Supplemental hosted proof | Hosted signal that improves confidence but is not the strongest claim owner. |
| Local-only proof | Command passed on this workstation and must not be treated as hosted platform evidence. |
| Source-controlled policy | Documentation, workflow, script, or report-index ownership row that defines the maintained contract. |
| Deferred non-claim | Explicitly unsupported or intentionally unpromoted surface. |
| PR-time pending | Current branch evidence that can only be established after PR CI runs. |
| External outage | Hosted-service setup/download failure outside repository command execution. |

## Final Platform Evidence Table

| Platform / Surface | Maintained Lane | Current Evidence | Support Label | Claim Boundary |
| --- | --- | --- | --- | --- |
| Linux reviewed Makefile compile-quality | `.github/workflows/ci.yml::lint` runs `make quality-review-compile` | Latest master merge baseline for PR #172 completed successfully: CI run `31723661978` on `c7981c6ef7fa887e87575279009113b1dcf3a630`. Local Day 4 also passed `make quality-review-full`. | Reviewed hosted proof plus local-only proof | Strongest reviewed source of truth for compile-quality, not a Windows/macOS substitute. |
| Linux reviewed CMake parity | `.github/workflows/ci.yml::cmake-build-and-test` runs `make quality-review-cmake` | Latest master merge baseline CI run `31723661978` completed successfully. Local Day 4 CMake parity registered and ran 59 tests with zero failures. | Reviewed hosted proof plus local-only proof | CMake parity proof does not imply package-manager or shared-library support. |
| Linux reviewed dead-code completeness | `.github/workflows/ci.yml::deadcode` runs `make deadcode-report` and `make deadcode-check` | Latest master merge baseline CI run `31723661978` completed successfully. Local Day 4 `make quality-review-full` included `deadcode-check`. | Reviewed hosted proof plus local-only proof | Dead-code report rows are reviewed maintenance evidence, not end-user capability claims. |
| Linux reviewed static-first package contract | `.github/workflows/ci.yml::package-contract` runs `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh` | Latest master merge baseline CI run `31723661978` completed successfully. Local Day 5 passed all three package commands. | Reviewed hosted proof plus local-only proof | Static archive install/export only; no shared-library, dynamic ABI, runtime-loader, or package-manager claim. |
| Linux supplemental direct runtime and benchmark signal | `.github/workflows/ci.yml::build-and-test` runs `make test`, sanitize paths, `make bench-build`, and `make bench-fast` | Latest master merge baseline CI run `31723661978` completed successfully. | Supplemental hosted proof | Runtime and `bench-fast` improve regression confidence but are not portable performance claims. |
| Linux supplemental ThreadSanitizer/OpenMP signal | `.github/workflows/ci.yml::tsan` | Latest master merge baseline CI run `31723661978` completed successfully. | Supplemental hosted proof | TSan/OpenMP lane is Linux-specific and carries known runtime-suppression boundaries. |
| Linux supplemental coverage report | `.github/workflows/ci.yml::coverage` runs `make coverage` and uploads HTML coverage | Latest master merge baseline CI run `31723661978` completed successfully. | Supplemental hosted proof | Coverage report is a maintained signal, not a quality or correctness guarantee by itself. |
| macOS reviewed Apple Clang path | `.github/workflows/macos-ci.yml::build-and-test` for `compiler=apple-clang` runs reviewed Makefile/CMake paths, `wall-check`, and sanitizer | Latest master merge baseline for PR #172 completed successfully: macOS CI run `31723661840` on `c7981c6ef7fa887e87575279009113b1dcf3a630`. | Reviewed hosted proof | Reviewed macOS path is Apple Clang scoped; Homebrew GCC remains supplemental. |
| macOS supplemental Homebrew GCC path | `.github/workflows/macos-ci.yml::build-and-test` for `compiler=homebrew-gcc` runs direct build/test/wall-check | Latest master merge baseline macOS CI run `31723661840` completed successfully. | Supplemental hosted proof | Second-compiler signal only; does not replace reviewed Apple Clang lane. |
| macOS reviewed static-first Make install/pkg-config | `.github/workflows/macos-ci.yml::install-and-pkgconfig` runs `tests/test_install.sh` | Latest master merge baseline macOS CI run `31723661840` completed successfully. Local Day 5 Unix-side install proof also passed. | Reviewed hosted proof plus local-only proof | Static-first package proof only; no shared-library, dynamic ABI, runtime-loader, package-manager, static/shared selector, or broad macOS parity claim. |
| macOS reviewed CMake install/export | `.github/workflows/macos-ci.yml::cmake-install-export` runs `tests/test_cmake_install.sh` and the static deferral guard | Latest master merge baseline macOS CI run `31723661840` completed successfully. Local Day 5 CMake install/export proof also passed. | Reviewed hosted proof plus local-only proof | Static CMake package contract only. |
| Windows reviewed CMake consumer subset | `.github/workflows/windows-ci.yml::build-and-test` configures/builds with MSVC 2022, checks `ctest -N`, and runs full CTest | Latest master merge baseline for PR #172 completed successfully: Windows CI run `31723661771` on `c7981c6ef7fa887e87575279009113b1dcf3a630`. Workflow expects 59 registered CTest tests and includes promoted `test_threads`, `test_sprint4_integration`, and `test_fuzz`. | Reviewed hosted proof | Windows support remains CMake-first; no Makefile parity, `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Windows reviewed CMake install/downstream validation | `.github/workflows/windows-ci.yml::install-and-downstream` validates installed static `.lib`, headers, CMake metadata, `sparse.pc` metadata, downstream CMake consumers, exact-version behavior, mismatch rejection, and no DLL/shared metadata | Latest master merge baseline Windows CI run `31723661771` completed successfully. | Reviewed hosted proof | Windows package confidence is CMake install/downstream scoped and does not include Windows Makefile install or Windows `pkg-config` execution. |
| Sprint 156 current branch hosted CI | Future PR workflows for `sprint-156` | No hosted PR run exists yet for Day 6 local documentation changes. | PR-time pending | Current branch hosted evidence must be assessed after PR creation; do not transfer master run status to unrun branch changes. |

## Windows Staged And Deferred Surface

Sprint 148 closed the prior Windows staged-test exclusions by promoting
`test_threads`, `test_sprint4_integration`, and `test_fuzz` into the reviewed
MSVC CMake CTest subset. The current workflow enforces this with
`EXPECTED_WINDOWS_CTEST_COUNT=59`, a `ctest -N` registration check, and full
hosted `ctest`.

The remaining Windows boundaries are deferred non-claims, not hidden staged
tests:

- Windows Makefile parity remains deferred.
- Windows `pkg-config` execution and downstream parity remain deferred.
- Windows package-manager support remains deferred.
- Windows shared-library support, DLL/runtime-loader behavior, dynamic ABI
  support, and broad Windows platform parity remain deferred.

## Linux And macOS Package Reconciliation

Linux and macOS both carry reviewed static-first package proof, but they do
not prove exactly the same surface as Windows:

- Linux reviews the full Unix-side package contract through Make install,
  `pkg-config`, CMake install/export, and static deferral guard.
- macOS reviews the Unix-side static-first Make install/`pkg-config` proof and
  CMake install/export proof on hosted macOS runners.
- Windows reviews CMake install/downstream validation for the static-first
  package surface but deliberately does not execute or claim Windows
  `pkg-config` parity.

Day 5 local package validation supports the same static-first interpretation:
local `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
`scripts/static_package_deferral_check.sh` all passed, but those are local
macOS proofs and cannot replace hosted Linux/macOS/Windows evidence.

## External-Service Failure Classification

External setup/download failures should be classified separately from
repository defects. A GitHub Actions setup failure such as inability to resolve
action download metadata, `Service Unavailable`, or transient network request
failure before repository commands run is an external outage unless a rerun
reaches the repository command phase and reproduces a command failure.

No current Day 6 hosted failure was observed in the latest `gh run list`
sample. The latest master merge baseline showed successful CI, macOS CI, and
Windows CI runs for PR #172's merge commit:

- CI: `31723661978`
- macOS CI: `31723661840`
- Windows CI: `31723661771`

## Claim Rules For Days 7-10

- Cite Linux as the strongest reviewed source of truth.
- Cite macOS as reviewed Apple Clang plus reviewed static-first install/export
  proof, with Homebrew GCC as supplemental.
- Cite Windows as reviewed MSVC CMake-first support with 59 registered CTest
  tests and reviewed CMake install/downstream validation.
- Do not describe local Day 4 or Day 5 commands as hosted platform evidence.
- Do not convert package metadata into shared-library, dynamic ABI,
  runtime-loader, package-manager, or static/shared selector support.
- Do not convert benchmark, coverage, or report rows into broad
  state-of-the-art or portable performance claims.
- Treat Sprint 156 branch CI as PR-time pending until the branch has hosted
  PR runs.

## Completion Criteria Check

- Each platform claim maps to a reviewed or supplemental CI lane.
- Windows promoted tests and remaining non-claims are explicit.
- Linux/macOS/Windows package surfaces are separated by maintained lane.
- External-service outages are separated from repository command failures.
- Local-only evidence from Days 4 and 5 is not widened into hosted proof.
