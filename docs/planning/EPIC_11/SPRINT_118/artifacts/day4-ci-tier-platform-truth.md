# Sprint 118 Day 4 CI Tier and Platform Truth Freeze

## Purpose

Day 4 freezes the current CI, platform, package, and install-support truth for
Sprint 118. It reconciles workflow definitions, public support wording, Day 3
local reviewed validation, and package/install documentation without promoting
new support claims.

## Inputs Reviewed

| Input | Day 4 role |
|---|---|
| `.github/workflows/ci.yml` | Linux reviewed and supplemental CI tier source. |
| `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang and supplemental GCC/install confidence source. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake consumer subset source and expected CTest count. |
| `README.md` | Public quick-reference support and install wording. |
| `INSTALL.md` | Maintained static-first install contract and platform support table. |
| `artifacts/day2-validation-inventory.md` | Reviewed/supplemental lane inventory and expected counts. |
| `artifacts/day3-baseline-quality-recheck.md` | Fresh local reviewed baseline evidence. |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprints 124-125 | Package/ABI and platform follow-through owner targets. |

## CI-Tier Support Map

| Tier | Platform or command | Current role | Evidence source |
|---|---|---|---|
| Strongest local reviewed baseline | `make quality-review-full` | Local reviewed baseline for Sprint 118 Day 3. | Day 3 artifact: passed. |
| Linux enforced reviewed Makefile compile-quality | `make quality-review-compile` in `.github/workflows/ci.yml` | Reviewed CI source for format, source-list, lint, strict compile, benchmark/example compile coverage. | Linux CI workflow. |
| Linux enforced reviewed CMake parity | `make quality-review-cmake` in `.github/workflows/ci.yml` | Reviewed CI source for CMake configure/build, `ctest -N`, test-count parity, and full CTest. | Linux CI workflow. |
| Linux enforced dead-code completeness | `make deadcode-report` and `make deadcode-check` | Reviewed CI source for dead-code report generation and completeness. | Linux CI workflow. |
| Linux supplemental runtime and benchmark | `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast` | Supplemental runtime, sanitizer, and benchmark signal. | Linux CI workflow. |
| Linux supplemental TSan | focused thread/eigensolver TSan jobs | Supplemental thread-safety and OpenMP signal with suppression/runtime caveats. | Linux CI workflow. |
| Linux supplemental coverage | `make coverage` | Supplemental coverage report and threshold signal. | Linux CI workflow. |
| macOS reviewed Apple Clang path | `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` | Reviewed macOS lane for Apple Clang. | macOS CI workflow. |
| macOS supplemental GCC path | Homebrew GCC direct build/test/wall-check | Supplemental second-compiler confidence. | macOS CI workflow. |
| macOS supplemental install/pkg-config | `bash tests/test_install.sh` | Supplemental static-first Make install and `pkg-config` confidence. | macOS CI workflow. |
| Windows reviewed CMake consumer subset | MSVC configure/build, `ctest -N`, expected count check, full `ctest` | Reviewed Windows CMake-first consumer proof. | Windows CI workflow. |

## Platform Validation Boundary Table

| Platform | Reviewed truth | Supplemental truth | Explicit boundary |
|---|---|---|---|
| Linux | Strongest reviewed source of truth for Makefile compile-quality, CMake parity, and dead-code completeness. | Direct runtime tests, UBSan, ASan, fast benchmarks, TSan, and coverage. | Focused install/package scripts remain proof surfaces, not a separate reviewed Linux install CI lane. |
| macOS | Apple Clang reviewed Makefile compile-quality, CMake parity, wall-check, and sanitizer. | Homebrew GCC direct build/test/wall-check; static-first Make install/`pkg-config` confidence. | Does not claim full macOS install/export parity or symmetric Linux/macOS support. |
| Windows | MSVC CMake configure/build, reviewed `ctest -N` count check, and full CTest for the reviewed subset. | No supplemental Windows lane is promoted as reviewed by current workflows. | Does not claim Makefile parity, install-validation parity, thread/fuzz/property parity, or full CTest parity. |

## Package and Install Claim Map

| Claim area | Current supported wording | Evidence | Boundary |
|---|---|---|---|
| Install model | Maintained install surface is static-first. | `README.md`, `INSTALL.md`, `make install`, CMake install/export rules, install scripts. | No shared-library or dynamic ABI promise. |
| Make install | Unix-like `make install` installs static archive, headers, generated version header, and `sparse.pc`. | Makefile install target and `tests/test_install.sh`. | Local/focused proof unless promoted into reviewed CI. |
| CMake install/export | `cmake --install` exports static `Sparse::sparse_lu_ortho` package. | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `tests/test_cmake_install.sh`. | Windows reviewed lane is CMake consumer proof, not separate install-validation proof. |
| `pkg-config` | `pkg-config --cflags --libs sparse` describes the installed static archive surface. | `sparse.pc.in`, Make install target, `tests/test_install.sh`. | No package-manager recipe claim. |
| `find_package(Sparse)` | Downstream CMake consumers can use the installed static package. | CMake install/export rules and `tests/test_cmake_install.sh`. | Exact-version package file; no dynamic ABI guarantee. |
| Version metadata | Version propagates from `VERSION` to generated header, CMake config version, and `sparse.pc`. | Makefile/CMake generation and install tests. | Version metadata is not an ABI compatibility policy. |
| Shared library | Deferred. | README/INSTALL wording and Epic 11 Sprint 124 plan. | Requires build rules, package metadata, installed-consumer proof, symbol/version policy, and loader coverage before support can be claimed. |
| Package managers | Deferred. | Epic 11 review/todo and Sprint 124-125 scope. | No Homebrew, apt, vcpkg, Conan, Spack, or similar recipe support is claimed. |

## Staged-Exclusion Register

| Area | Current staged or unclaimed state | Owner candidate |
|---|---|---|
| Windows test count | Reviewed Windows CTest count remains `51`, not local full `54`. | Sprint 125 platform staged-lane follow-through. |
| Windows excluded tests | `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain outside reviewed Windows subset. | Sprint 125 Windows staged test follow-through. |
| Windows install validation | No separate reviewed Windows install-validation lane. | Sprint 125 Windows install validation design. |
| Windows Makefile parity | Not claimed. | Sprint 125 platform gap audit or future epic if not feasible. |
| Windows thread/fuzz/property parity | Not claimed; fuzz/property lane remains outside reviewed Windows subset. | Sprint 125 staged test follow-through. |
| Linux install CI | Not promoted to a separate reviewed CI lane. | Sprint 125 Linux install CI decision. |
| macOS full install/export parity | Not claimed; Make install/`pkg-config` confidence is supplemental. | Sprint 125 macOS install/export follow-through. |
| Shared-library ABI | Deferred. | Sprint 124 package/ABI product decision. |
| Package-manager support | Deferred. | Sprint 124 package/ABI closeout or future-epic residual. |
| Portable performance/platform timing | Not claimed. | Sprint 123 performance governance and Sprint 127 claim recalibration. |

## Claims Needing Fence or Future Owners

No current public wording reviewed on Day 4 requires immediate downgrade for
the current touched surface. The existing docs already preserve the relevant
fences:

- static-first install support only;
- no shared-library/dynamic ABI promise;
- no package-manager support;
- Linux as strongest reviewed source of truth;
- macOS reviewed Apple Clang lane plus supplemental support;
- Windows reviewed CMake consumer subset only;
- local benchmark and wall-check timing as non-portable evidence.

Future-owner candidates should remain visible:

| Candidate | Sprint owner | Reason |
|---|---|---|
| Shared-library or explicit static-first continuation decision | Sprint 124 | Product/ABI decision requires explicit implementation or deferral proof. |
| ABI/symbol/version policy | Sprint 124 | Needed only if shared-library support is added; otherwise deferral checks should stay explicit. |
| Linux install CI promotion or deferral | Sprint 125 | Current install scripts are focused proof surfaces, not a reviewed Linux install CI lane. |
| macOS CMake install/export parity decision | Sprint 125 | Current macOS install confidence is supplemental. |
| Windows install/downstream consumer proof | Sprint 125 | Current Windows reviewed lane is CMake-first consumer subset only. |
| Windows staged test membership | Sprint 125 | Current expected count is `51` with explicit exclusions. |
| Package-manager support decision | Sprint 124 or post-Epic residual | No current recipe or package-manager validation exists. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Current platform truth is explicit and evidence-backed. | Complete. |
| Reviewed and supplemental platform lanes are separated. | Complete. |
| Package/install claims do not exceed validation. | Complete. |
| Staged exclusions are recorded as current truth. | Complete. |
| Sprint 124-125 handoff candidates are visible. | Complete. |
