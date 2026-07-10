# Sprint 118 Day 2 Reviewed Baseline Validation Inventory

## Purpose

Day 2 inventories the reviewed and supplemental validation surfaces before
Sprint 118 executes the baseline recheck on Day 3. The goal is to identify the
commands, expected counts, staged exclusions, and platform boundaries that
define current product truth without widening claims.

## Surfaces Inspected

| Surface | Files or targets inspected | Day 2 conclusion |
|---|---|---|
| Makefile reviewed quality | `quality-review-compile`, `quality-review`, `quality-review-full`, `format-check`, `source-list-check`, `lint`, `test`, `deadcode-check` | Local reviewed Makefile proof is explicit and operator-facing. |
| CMake parity | `quality-review-cmake-compile`, `quality-review-cmake`, `CMakeLists.txt` | CMake parity is reviewed through configure, clean build, `ctest -N`, Makefile/CMake count parity, and full CTest. |
| Source-list completeness | `make source-list-check`, `scripts/check_library_sources.py` | Source-list proof is part of `quality-review-compile`; source movement must keep it green. |
| Dead-code evidence | `deadcode-report`, `deadcode-check` | Dead-code is a report completeness/supporting-context gate, not a zero-finding claim. |
| Install/package | `make install`, `make uninstall`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in` | Static-first Make and CMake install/export proof exists as focused package evidence. |
| Benchmarks and performance | `bench-build`, `bench-fast`, `bench-canonical-report`, `performance-sentinels`, `large-matrix-guardrails`, `wall-check`, `benchmarks/README.md` | Benchmark lanes are local or supplemental unless explicitly promoted by reviewed validation. |
| Coverage | `make coverage`, `coverage-lcov`, `coverage-gcovr`, CI coverage job | Coverage is supplemental assurance with an 80% line threshold, not a product claim by itself. |
| Sanitizers and threading | `make sanitize`, `make asan`, `make sanitize-thread`, `make tsan`, `make omp`, Linux TSan workflow | Sanitizers are supplemental runtime evidence unless included in a platform-specific reviewed lane. |
| Linux CI | `.github/workflows/ci.yml` | Linux is the strongest reviewed source of truth for Makefile compile-quality, CMake parity, and dead-code completeness; it also carries supplemental runtime, benchmark, TSan, and coverage signals. |
| macOS CI | `.github/workflows/macos-ci.yml` | macOS enforces Apple Clang reviewed Makefile/CMake paths plus wall-check and sanitize; Homebrew GCC and static-first install/pkg-config are supplemental. |
| Windows CI | `.github/workflows/windows-ci.yml` | Windows enforces the MSVC CMake consumer subset only; expected CTest count is `51`. |

## Reviewed Baseline Command Matrix

| Command or lane | Classification | What it proves | When Sprint 118 should run it |
|---|---|---|---|
| `git diff --check` | Required docs hygiene for this sprint's current touched surface | No whitespace errors in diffs. | Every day with documentation changes. |
| `rg -n '[ \t]+$' docs/planning/EPIC_11/SPRINT_118` | Required focused docs hygiene | No trailing whitespace in Sprint 118 planning docs. | Every documentation-only day before closeout. |
| `make quality-review-compile` | Reviewed Makefile compile-quality path | `format-check`, source-list completeness, strict warning compile, `clang-tidy`, `cppcheck`, benchmark/example compile coverage through `lint`. | Day 3 if tools are available; required for source-list or compile-quality truth. |
| `make quality-review` | Reviewed local Makefile quality path | `format-check`, `lint`, full Makefile tests, and `deadcode-check`. | Day 3 if local runtime/tooling permit, or when code/test surfaces change. |
| `make quality-review-cmake-compile` | Reviewed CMake parity compile path | CMake configure, clean build, `ctest -N`, and Makefile/CMake test-count parity. | Day 3 if CMake/tooling permit, and whenever CMake/test registration changes. |
| `make quality-review-cmake` | Reviewed CMake parity execution path | `quality-review-cmake-compile` plus full CTest execution. | Day 3 if CMake/tooling permit, and whenever CMake/test registration or CMake consumers change. |
| `make quality-review-full` | Strongest local reviewed baseline | `quality-review` plus `quality-review-cmake`. | Preferred Day 3 baseline command if local tools and time permit. |
| Linux CI `quality-review-compile` job | Enforced reviewed CI lane | Makefile compile-quality path on Ubuntu. | CI-owned; use as reviewed platform evidence after push/PR. |
| Linux CI `quality-review-cmake` job | Enforced reviewed CI lane | CMake parity and full CTest on Ubuntu. | CI-owned; use as reviewed platform evidence after push/PR. |
| Linux CI `deadcode` job | Enforced reviewed CI lane | Dead-code report generation and completeness. | CI-owned; use as reviewed dead-code completeness evidence after push/PR. |
| macOS Apple Clang reviewed jobs | Enforced reviewed macOS lane | Makefile compile-quality, CMake parity, wall-check, and sanitize under Apple Clang. | CI-owned; use as macOS reviewed platform evidence after push/PR. |
| Windows MSVC CMake job | Enforced reviewed Windows subset | Configure, build, `ctest -N`, expected-count check, and full CTest for the CMake consumer subset. | CI-owned; use as Windows reviewed consumer evidence after push/PR. |

## Supplemental Validation Lane Inventory

| Command or lane | Classification | What it can support | Boundary |
|---|---|---|---|
| `make test` | Supplemental direct runtime path when used alone | Full Makefile test execution. | Not a substitute for format, lint, source-list, dead-code, or CMake parity. |
| `make sanitize` / `make asan` | Supplemental runtime safety | UBSan/ASan confidence on current tests. | Tree-mutating; run `make clean` before returning to reviewed path. |
| `make sanitize-thread` / `make tsan` | Supplemental thread-safety evidence | Focused TSan coverage, especially around thread/eigensolver paths. | Platform/runtime suppression boundaries apply; not broad data-race proof for every OpenMP path. |
| `make bench-build` | Supplemental benchmark compile coverage | Benchmark sources compile. | Does not execute benchmarks or prove performance. |
| `make bench-fast` | Supplemental PR-time benchmark signal | Fast local benchmark subset and `bench_reorder --skip-factor`. | Local timing context only; no portable performance claim. |
| `make bench-canonical-report` | Supplemental report artifact | Canonical benchmark report generation. | Report interpretation remains local unless reviewed gates say otherwise. |
| `make performance-sentinels` | Supplemental local regression evidence | Sentinel report generation for selected paths. | Does not imply portable speed or vendor/backend parity. |
| `make large-matrix-guardrails` | Supplemental guardrail evidence | Large-matrix graph/reorder/benchmark guardrails. | Local/named-fixture evidence, not universal scalability proof. |
| `make wall-check` | macOS enforced signal and local supplemental signal | Local wall-time regression tripwire against stored baseline. | Machine-class dependent; no cross-platform timing claim. |
| `make coverage` | Supplemental coverage architecture | Coverage report and 80% threshold on active test surface. | Coverage is assurance context, not correctness by itself. |
| `bash tests/test_install.sh` | Focused package/install proof | Unix-side Make install/uninstall plus `pkg-config` downstream consumer. | Static-first package proof only; no shared-library ABI or package-manager claim. |
| `bash tests/test_cmake_install.sh` | Focused package/install proof | CMake install/export, `find_package(Sparse)`, installed example, version checks. | Static-first installed CMake proof only; Windows install-validation remains unclaimed. |
| macOS Homebrew GCC leg | Supplemental second-compiler signal | Direct build/test/wall-check under GCC. | Not part of the reviewed Apple Clang lane. |
| macOS install/pkg-config job | Supplemental package confidence | Static-first Make install/pkg-config confidence on macOS. | Not full reviewed macOS install/export parity. |
| Linux coverage job | Supplemental coverage signal | Coverage report uploaded by CI. | Not part of reviewed source-of-truth baseline. |
| Linux benchmark/TSan jobs | Supplemental runtime signals | Fast benchmark, sanitizer, thread-safety evidence. | Do not widen product claims without separate proof. |

## Expected Counts and Staged Exclusions

| Surface | Expected current value | Source or basis | Notes |
|---|---:|---|---|
| Makefile test binaries | `54` | `Makefile` `TEST_SRCS` / Sprint 117 final validation | Used for CMake parity comparison. |
| Local CMake `ctest -N` count | `54` | Sprint 117 final validation and `quality-review-cmake-compile` parity check | Day 3 should investigate any mismatch before recording evidence. |
| Local CMake full CTest result | `54 / 54` passing expected | Sprint 117 final validation | Day 3 should stop on failure or record blocker. |
| Windows reviewed CTest count | `51` | `.github/workflows/windows-ci.yml` `EXPECTED_WINDOWS_CTEST_COUNT` | Windows intentionally excludes `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |
| Coverage threshold | `80%` line coverage | `Makefile` `COV_THRESHOLD` | Supplemental assurance threshold, not a public quality claim. |

Windows staged exclusions remain current product truth:

- `test_threads`;
- `test_sprint4_integration`;
- `test_fuzz`, including the bounded lifecycle property lane.

Windows reviewed scope remains CMake-first consumer proof only. It does not
claim Windows Makefile parity, Windows install-validation parity, or separate
Windows thread/fuzz/property parity.

## Platform Support Boundary Notes

| Platform | Reviewed evidence | Supplemental evidence | Boundary |
|---|---|---|---|
| Linux | Enforced Makefile compile-quality path, CMake parity path, and dead-code completeness path. | Direct `make test`, UBSan, ASan, benchmark compile/fast run, TSan, coverage. | Strongest reviewed source of truth, but install/package scripts remain focused proof surfaces rather than a separate reviewed install CI lane. |
| macOS | Apple Clang reviewed Makefile compile-quality, CMake parity, wall-check, and sanitizer path. | Homebrew GCC direct build/test/wall-check and static-first Make install/pkg-config confidence. | Do not claim full macOS install/export parity or symmetric Linux/macOS parity. |
| Windows | MSVC CMake configure/build, `ctest -N`, expected-count check, and full CTest for reviewed subset. | None promoted as reviewed by current workflow. | CMake-first consumer subset only; no Makefile, install-validation, thread/fuzz/property, or full parity claim. |

## Day 3 Execution Checklist

Day 3 should execute the strongest local reviewed baseline if local tooling and
runtime permit:

1. Record starting branch, HEAD commit, changed files, and touched surfaces.
2. Run documentation hygiene:
   - `git diff --check`;
   - `rg -n '[ \t]+$' docs/planning/EPIC_11/SPRINT_118`.
3. Run `make quality-review-full`.
4. Capture the reviewed baseline breakdown:
   - `format-check`;
   - `source-list-check`;
   - `lint`;
   - `test`;
   - `deadcode-check`;
   - CMake configure;
   - CMake clean build;
   - CMake `ctest -N`;
   - Makefile/CMake test-count parity;
   - CMake full CTest.
5. Confirm CMake and Makefile counts are `54` vs `54`.
6. Record whether package/install, benchmark, sanitizer, coverage, or platform
   workflow lanes were run or intentionally skipped.
7. Stop and investigate before proceeding if any required lane fails or if
   `ctest -N` count changes unexpectedly.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All baseline commands are known before execution. | Complete. |
| Reviewed and supplemental evidence are clearly separated. | Complete. |
| Expected CTest counts and staged exclusions are recorded. | Complete. |
| Platform boundaries are documented as current truth. | Complete. |
| Day 3 can execute without rediscovering validation scope. | Complete. |
