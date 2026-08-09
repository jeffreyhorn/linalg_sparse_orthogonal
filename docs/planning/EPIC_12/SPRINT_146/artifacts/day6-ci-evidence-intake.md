# Day 6 CI Evidence Intake

## Scope

Day 6 collects the current hosted CI and platform evidence available for Sprint
146 reconciliation. The `sprint-146` branch has not produced hosted runs yet,
so the latest available hosted baseline is the current `master` push created
by merging PR #161:

- branch: `master`
- commit: `daac9a85d516f72100c34b90b92ec78941a72200`
- short commit: `daac9a85`
- event: `push`
- created: `2026-08-09T20:53:41Z`

The Day 5 local baseline passed on `sprint-146`, but local results cannot be
used as hosted Linux, macOS, or Windows proof. Hosted branch/PR evidence for
Sprint 146 remains unavailable until the branch is pushed and workflows run.

## Hosted Run Status

| Workflow | Run ID | Branch | Commit | Status | Conclusion | URL |
| --- | ---: | --- | --- | --- | --- | --- |
| CI | `31335415785` | `master` | `daac9a85` | completed | success | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/31335415785` |
| macOS CI | `31335415782` | `master` | `daac9a85` | completed | success | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/31335415782` |
| Windows CI | `31335415791` | `master` | `daac9a85` | completed | success | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/31335415791` |
| Sprint 146 branch workflows | none found | `sprint-146` | `daac9a85` plus uncommitted planning docs | unavailable | unavailable | `gh run list --branch sprint-146` returned no runs |

## CI Lane Inventory

| Platform | Workflow Job | Classification | Commands Or Steps | Current Hosted Evidence | Claim Boundary |
| --- | --- | --- | --- | --- | --- |
| Linux | Linux enforced reviewed Makefile compile-quality path | reviewed | install `clang-format`, `clang-tidy`, `cppcheck`; `make quality-review-compile` | master run `31335415785` job `93300350112` succeeded | Reviewed compile-quality proof only; not full product parity or state-of-the-art proof. |
| Linux | Linux enforced reviewed CMake parity path | reviewed | `make quality-review-cmake` | master run `31335415785` job `93300350118` succeeded | Reviewed CMake parity proof only. |
| Linux | Linux enforced dead-code report and completeness path | reviewed | install dead-code tools; `make deadcode-report`; `make deadcode-check`; upload artifacts | master run `31335415785` job `93300350128` succeeded | Dead-code report/completeness path; no zero-dead-code or release quality claim beyond the maintained check. |
| Linux | Linux reviewed static-first package contract | reviewed | install CMake/pkg-config; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | master run `31335415785` job `93300350105` succeeded | Static archive package contract only; no shared-library, dynamic ABI, loader, package-manager, or broad platform claim. |
| Linux | Linux supplemental runtime and bench-fast path | supplemental | `make test`; `make sanitize`; `make asan`; `make bench-build`; `make bench-fast` | master run `31335415785` job `93300350121` succeeded | Supplemental runtime/sanitizer/fast-benchmark signal; no portable performance claim. |
| Linux | Linux supplemental ThreadSanitizer coverage | supplemental | thread tests under TSan; eigensolver tests under TSan + OpenMP | master run `31335415785` job `93300350072` succeeded | Supplemental TSan coverage; OpenMP runtime caveats remain documented. |
| Linux | Linux supplemental coverage report | supplemental | install lcov; `make coverage`; upload coverage report | master run `31335415785` job `93300350151` succeeded | Supplemental coverage report; no coverage completeness claim. |
| macOS | macOS enforced Apple Clang reviewed path | reviewed | install reviewed-path tools; `make quality-review-compile`; `make quality-review-cmake`; `make wall-check`; `make sanitize` | master run `31335415782` job `93300350055` succeeded | Reviewed Apple Clang path; no broader macOS parity or portable performance claim. |
| macOS | macOS Homebrew GCC matrix leg | supplemental | install GCC; `make CC=gcc-15`; `make CC=gcc-15 test`; `make CC=gcc-15 wall-check` | master run `31335415782` job `93300350045` succeeded | Supplemental second-compiler signal. |
| macOS | macOS reviewed static-first install and pkg-config proof | reviewed | `bash tests/test_install.sh` | master run `31335415782` job `93300350051` succeeded | Reviewed macOS static-first Make install/`pkg-config` proof only. |
| macOS | macOS reviewed static-first CMake install/export proof | reviewed | `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | master run `31335415782` job `93300350039` succeeded | Reviewed macOS static-first CMake install/export proof only. |
| Windows | Windows enforced reviewed CMake consumer subset (MSVC) | reviewed | configure with VS 2022 x64; build Release; `ctest -N`; `ctest --output-on-failure` | master run `31335415791` job `93300350063` succeeded | Reviewed Windows CMake-first subset; no Makefile, `pkg-config`, install-validation parity, or staged POSIX/pthread closure. |
| Windows | Windows supplemental CMake install/downstream confidence path | supplemental | CMake install; static `.lib` check; no DLLs; installed headers; CMake package files; `sparse.pc`; installed example; exact-version consumer; mismatched-version rejection | master run `31335415791` job `93300350119` succeeded | Supplemental CMake-first package confidence only; not a reviewed Windows install-validation lane. |

## Platform Assumptions And Expected Counts

| Platform | Assumption | Expected Count Or Guard |
| --- | --- | --- |
| Linux | Ubuntu runners provide GNU Make, CMake, compiler tooling, sanitizers, lcov, and dead-code dependencies installed by workflow steps. | No fixed total test-count assertion in CI workflow; reviewed Make/CMake/dead-code/package jobs must pass. |
| macOS | `macos-latest` provides Apple Clang; Homebrew installs GCC and reviewed-path tools. | Matrix has Apple Clang reviewed path and Homebrew GCC supplemental path; package proof jobs run separately. |
| Windows | `windows-2022` provides the Visual Studio 17 2022 generator and MSVC x64 toolchain. | `EXPECTED_WINDOWS_CTEST_COUNT` is `56`; staged exclusions remain `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |

## Reviewed, Supplemental, Staged, Local-Only, Hosted-Only, Deferred

| Classification | Lanes |
| --- | --- |
| Reviewed | Linux Makefile compile-quality, Linux CMake parity, Linux dead-code, Linux static-first package contract, macOS Apple Clang reviewed path, macOS static-first install/`pkg-config`, macOS static-first CMake install/export, Windows CMake consumer subset. |
| Supplemental | Linux direct runtime/sanitizers/bench-fast, Linux TSan, Linux coverage, macOS Homebrew GCC, Windows CMake install/downstream confidence. |
| Staged | Windows `test_threads`, `test_sprint4_integration`, `test_fuzz`; Windows Makefile parity; Windows `pkg-config` parity; Windows reviewed install-validation parity. |
| Local-only | Day 5 local corpus schema, report normalization/freshness, local install scripts, examples build, focused QR and partial-SVD corpus proof, generated oracle/report refresh. |
| Hosted-only | Current Linux/macOS/Windows platform support proof and final Sprint 146 PR status once a PR exists. |
| Deferred | Shared-library ABI, runtime-loader behavior, package-manager distribution, static/shared selector support, portable performance, broad platform parity, state-of-the-art status. |

## Platform Blocker Register

| Blocker | Platform | Current Evidence | Required Promotion Gate |
| --- | --- | --- | --- |
| `test_threads` uses pthread APIs | Windows | Workflow explicitly excludes staged test; Windows reviewed lane prints blocker text. | Source-level portability change or Windows-native equivalent plus intentional CTest count update. |
| `test_sprint4_integration` uses pthread APIs | Windows | Workflow explicitly excludes staged test; Windows reviewed lane prints blocker text. | Source-level portability change or Windows-native equivalent plus intentional CTest count update. |
| `test_fuzz` uses POSIX temp-file APIs | Windows | Workflow explicitly excludes staged test, including bounded lifecycle property lane. | Windows-compatible temp-file implementation or separate reviewed Windows equivalent plus CTest count update. |
| Windows Makefile parity absent | Windows | Workflow states Makefile reviewed wrappers remain staged. | Maintained Windows Makefile route with hosted proof and docs/report updates. |
| Windows `pkg-config` parity absent | Windows | Workflow limits package confidence to CMake-first install/downstream proof. | Hosted Windows pkg-config toolchain/support decision plus install proof. |
| Windows reviewed install-validation parity absent | Windows | Supplemental CMake install/downstream job succeeds but is not reviewed install-validation parity. | Explicit product decision, workflow wording update, report row update, and hosted proof. |
| Shared-library ABI absent | all | Static deferral guard and docs intentionally reject shared-library support. | Export/import macro, visibility, symbol allowlist, ABI versioning, loader proof, package metadata, and cross-platform tests. |

## Support-Tier Mismatch Intake

| Surface | Intake Result |
| --- | --- |
| README/INSTALL/maintainer platform wording | Matches workflow classification: Linux source-of-truth, macOS reviewed static-first install/export, Windows CMake-first with supplemental install/downstream confidence. |
| Report-family CI row | Matches hosted-external model; source-controlled CI rows identify lanes but do not replace hosted logs. |
| Day 5 local validation | Complements hosted lanes but does not prove Linux/macOS/Windows hosted status for Sprint 146. |
| Sprint 146 branch hosted evidence | Missing. No `sprint-146` GitHub runs exist yet because the branch has not been pushed. This must stay explicit in Day 7 and later closeout work. |
| Windows CTest count | Workflow expects `56`; latest master hosted Windows reviewed lane succeeded with that expectation. No local Windows proof exists. |

## Day 7 Handoff

Day 7 should reconcile the support-tier wording against this intake. The latest
hosted `master` baseline is green across Linux, macOS, and Windows at
`daac9a85`, and Day 5 local validation is green on `sprint-146`. The remaining
gap is branch-specific hosted evidence: no `sprint-146` hosted run exists yet,
so final Sprint 146 platform promotion must either wait for PR/branch CI or
state that only the latest master baseline has been inspected.
