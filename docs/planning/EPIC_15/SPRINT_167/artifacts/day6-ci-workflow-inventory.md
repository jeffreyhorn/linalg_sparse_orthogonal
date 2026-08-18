# Sprint 167 Day 6: CI And Workflow Inventory

## Purpose

Day 6 inventories hosted workflows, supplemental lanes, platform tiers, and CI
evidence boundaries. The goal is to map hosted CI evidence to specific claims,
preserve platform support tiers, and identify gaps ready for Epic 15 selection.

## Workflow Files Reviewed

| Workflow | File | Primary role |
| --- | --- | --- |
| CI | `.github/workflows/ci.yml` | Linux source-of-truth reviewed baseline plus Linux supplemental runtime, benchmark, sanitizer, TSan, coverage, dead-code, package, and selected report freshness lanes. |
| macOS CI | `.github/workflows/macos-ci.yml` | macOS reviewed Apple Clang build/test/package surface plus supplemental Homebrew GCC coverage. |
| Windows CI | `.github/workflows/windows-ci.yml` | Windows reviewed CMake-first MSVC test and CMake install/downstream package validation surface. |

## Linux Hosted Lane Map

| Job | Evidence level | Commands/actions | Claims supported | Claims not supported |
| --- | --- | --- | --- | --- |
| `build-and-test` | Supplemental Linux runtime and bench-fast path | `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast` | Linux supplemental runtime, sanitizer, benchmark compile coverage, fast benchmark smoke. | Full benchmark publication, portable performance, backend superiority, state-of-the-art performance. |
| `cmake-build-and-test` | Enforced reviewed CMake parity path | `make quality-review-cmake` | Linux reviewed CMake configure/build/test parity. | Windows/macOS parity, package-manager support, dynamic ABI. |
| `package-contract` | Reviewed static-first package contract | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Linux static archive install, CMake package, pkg-config, downstream consumers, shared deferral guard. | Shared-library support, dynamic ABI, runtime-loader behavior, package-manager distribution. |
| `generated-report-freshness` | Reviewed hosted selected oracle/comparison freshness | `make report-index-oracle-freshness`, `make report-index-comparison-freshness`, artifact uploads | Selected QR/partial-SVD oracle rows and selected QR/partial-SVD comparison rows after hosted pass. | Broad report-family freshness, unselected generated rows, generated API HTML, performance reports. |
| `tsan` | Supplemental Linux ThreadSanitizer coverage | TSan builds/runs for thread tests and selected OpenMP eigensolver tests | Supplemental race-detection signal for selected thread/OMP paths. | Complete thread-safety proof, Archer-instrumented OpenMP runtime proof, platform parity. |
| `lint` | Enforced reviewed Makefile compile-quality path | `make quality-review-compile` | Linux reviewed format/lint/static analysis/build surface. | Runtime correctness by itself, package install, hosted generated report proof. |
| `deadcode` | Enforced dead-code report and completeness path | `make deadcode-report`, `make deadcode-check`, xunused build | Dead-code report generation and completeness gate. | Semantic correctness, performance, package, platform parity. |
| `coverage` | Supplemental Linux coverage report | `make coverage`, artifact upload | Supplemental coverage report and threshold signal. | Coverage completeness, branch coverage parity, release-quality proof. |

## macOS Hosted Lane Map

| Job | Evidence level | Commands/actions | Claims supported | Claims not supported |
| --- | --- | --- | --- | --- |
| `build-and-test` Apple Clang matrix | Reviewed macOS path plus supplemental checks | `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` | macOS Apple Clang reviewed Make/CMake compile/test path and sanitizer/wall-check signal. | Full platform parity, package-manager support, dynamic ABI, performance superiority. |
| `build-and-test` Homebrew GCC matrix | Supplemental second-compiler coverage | `brew install gcc`, `make CC=gcc-15`, `make CC=gcc-15 test`, `make wall-check` | macOS second-compiler supplemental build/test evidence. | Reviewed Apple Clang parity, package proof, broad compiler support. |
| `install-and-pkgconfig` | Reviewed macOS static-first Make install/pkg-config proof | `bash tests/test_install.sh` | macOS static archive Make install and pkg-config proof. | Shared-library package, package-manager support, dynamic ABI, broad macOS parity. |
| `cmake-install-export` | Reviewed macOS static-first CMake install/export proof | `bash tests/test_cmake_install.sh`, `bash scripts/static_package_deferral_check.sh` | macOS static CMake install/export and shared deferral guard. | Runtime-loader compatibility, static/shared selectors, package-manager support. |

## Windows Hosted Lane Map

| Job | Evidence level | Commands/actions | Claims supported | Claims not supported |
| --- | --- | --- | --- | --- |
| `build-and-test` | Reviewed Windows CMake-first MSVC subset | CMake configure/build, `ctest -N`, full `ctest` | Windows MSVC CMake configure/build/test proof for expected CTest count `59`. | Windows Makefile parity, Windows pkg-config execution parity, package-manager support, broad Windows parity. |
| `install-and-downstream` | Reviewed Windows CMake install/downstream package validation | CMake install, installed static `.lib`, 19 headers plus version header, CMake metadata, `sparse.pc` metadata inspection, downstream CMake consumers, exact/mismatch version checks | Windows static-first CMake install/downstream package proof. | pkg-config command execution, shared library/DLL support, dynamic ABI, runtime-loader behavior, package-manager support. |

## Generated Report And Package Gates

| Gate | Hosted status | Owner | Scope |
| --- | --- | --- | --- |
| `make report-index-oracle-freshness` | Reviewed Linux hosted selected lane | Linux `generated-report-freshness` job | Selected QR and partial-SVD oracle rows only. |
| `make report-index-comparison-freshness` | Reviewed Linux hosted selected lane | Linux `generated-report-freshness` job | Selected `qr-minnorm`, `qr-compatible-ls`, and `partial-svd-diag6-k2` comparison families only. |
| `make bench-canonical-report` | Local-only | Makefile/report scripts | Canonical benchmark rows, not hosted publication or superiority proof. |
| `make performance-sentinels` | Local-only hard/advisory mix | Makefile/report scripts | Bounded local wall-checks and measurements, not portable performance proof. |
| `make coverage` | Supplemental Linux hosted | Linux `coverage` job | Supplemental coverage report, not completeness or platform parity. |
| `tests/test_install.sh` | Reviewed Linux and macOS hosted; Unix shell path | Linux package job, macOS install job | Static Make install and pkg-config proof on Unix-like hosted runners. |
| `tests/test_cmake_install.sh` | Reviewed Linux and macOS hosted; Windows has inline PowerShell equivalent | Linux/macOS package jobs, Windows install job | Static CMake install/export proof. |
| `scripts/static_package_deferral_check.sh` | Reviewed Linux/macOS hosted and package-scope local | package jobs | Static-first shared-library deferral guard. |

## Local-Only Or Advisory Evidence Without Hosted Proof

| Surface | Current status | Epic 15 implication |
| --- | --- | --- |
| Generated API HTML | Local-only by product decision | R167-08 should either reaffirm local-only status with checks or choose a publication path. |
| Canonical benchmark report | Local-only methodology-bound rows | R167-02 needs hosted performance publication or an explicit retained local-only decision. |
| Performance sentinels | Local-only generated hard/advisory rows | Useful for local regression signals, not portable performance claims. |
| Broad report-family freshness | Selected oracle/comparison families only are hosted | R167-07 can promote one additional family or retain explicit deferral. |
| Optional external data | Source-controlled skip/defer policy | Not pass evidence; should not become SuiteSparse/external corpus parity. |
| Package-manager distribution | Unsupported | R167-04 needs a selected provider proof or formal deferral. |
| Shared-library and dynamic ABI | Unsupported with static-first guards | R167-03 needs product decision work before any package-manager claim. |
| Windows Makefile and Windows `pkg-config` execution | Explicit non-claims | Do not infer from Windows CMake package proof. |

## CI Brittleness Notes

| Area | Observation | Risk | Future handling |
| --- | --- | --- | --- |
| Windows expected CTest count | `.github/workflows/windows-ci.yml` pins `EXPECTED_WINDOWS_CTEST_COUNT` to `59`. | Adding/removing CMake tests can break CI unless the expected count is updated with the test registration change. | Treat count changes as intentional evidence changes and document them in PRs. |
| Hosted action availability | Workflows depend on `actions/checkout@v4` and `actions/upload-artifact@v4`. | External service outages can fail setup before project commands run. | Classify as hosted infrastructure failure if no project command executes. |
| Runner images | Windows is pinned to `windows-2022`; Linux uses `ubuntu-latest` and `ubuntu-24.04`; macOS uses `macos-latest`. | Image toolchain changes can affect CMake generators, compilers, shell behavior, or package tools. | Keep platform claims tied to runner image and workflow status. |
| Homebrew GCC name | macOS supplemental GCC uses `gcc-15`. | Homebrew formula changes can break supplemental second-compiler coverage. | Keep this supplemental, not primary reviewed proof. |
| Generated artifact paths | Hosted selected comparison uploads hard-code three selected comparison directories. | Adding comparison families requires explicit workflow path updates or artifact coverage will lag. | Make selected-family changes update both freshness targets and upload paths. |
| Windows path/newline behavior | Windows PowerShell package proof checks regexes, paths, and generated output strings. | Cross-platform newline/path changes can fail package proof. | Keep Windows package proof CMake-first and metadata-specific. |
| TSan OpenMP suppressions | Linux TSan uses suppressions for uninstrumented OpenMP runtime behavior. | Suppressions can hide real OpenMP races when libomp/libgomp appears in stack traces. | Keep as supplemental signal, not complete race-freedom proof. |
| Slow benchmark exclusion | Full `make bench` is not hosted because it exceeded job limits. | Hosted benchmark evidence is limited to compile/fast paths today. | R167-02 should design a bounded hosted performance lane if selected. |

## Platform Tier Summary

| Platform | Strongest reviewed evidence | Supplemental evidence | Retained non-claims |
| --- | --- | --- | --- |
| Linux | Makefile compile-quality, CMake parity, dead-code, static package contract, selected oracle/comparison freshness | `make test`, sanitizers, bench-fast, TSan, coverage | Portable performance superiority, full benchmark publication, broad report freshness, shared/dynamic ABI, package-manager distribution. |
| macOS | Apple Clang reviewed Make/CMake path, static Make install/pkg-config, static CMake install/export | Homebrew GCC build/test, wall-check, sanitizer | Shared-library package, dynamic ABI, package-manager support, broad macOS parity. |
| Windows | MSVC CMake configure/build/CTest, CMake install/downstream validation | None treated as package-manager or Makefile parity | Makefile parity, pkg-config command execution parity, shared/DLL support, dynamic ABI, runtime-loader behavior, package-manager support, broad Windows parity. |

## CI Gaps Ready For Epic 15 Selection

| Residual ID | CI gap | Selection-ready closure |
| --- | --- | --- |
| R167-02 | Hosted performance publication is missing. | Add one bounded hosted performance report lane or explicitly retain local-only performance rows. |
| R167-07 | Broad generated-report platform parity is missing. | Promote one selected report family beyond Linux or formally defer broad parity. |
| R167-08 | Generated API HTML is not hosted. | Publish as hosted/artifact/committed output or reaffirm local-only policy with checks. |
| R167-04 | Package-manager support is missing from CI. | Add one provider proof or formal deferral, after package/ABI decision. |
| R167-03 | Shared-library ABI is intentionally unsupported. | Decide product direction; if unsupported, preserve fail-closed static-first guards. |
| R167-06 | Additional comparison families are not in hosted selected comparison upload paths. | If a new selected family is added, update freshness target and artifact upload path together. |

## Day 7 Handoff

Day 7 should inventory package and install evidence with attention to:

- Make static install and pkg-config proof;
- CMake install/export proof;
- Windows CMake package metadata and downstream validation;
- static-first shared-library deferral guard;
- package-manager non-claims and possible provider-selection inputs;
- ABI/version metadata boundaries.

## Validation Notes

Day 6 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Hosted CI evidence is mapped to specific claims. | Complete | Linux, macOS, and Windows lane maps identify supported and unsupported claims. |
| Platform tier boundaries are explicit. | Complete | Platform tier summary separates Linux, macOS, and Windows reviewed/supplemental/non-claim surfaces. |
| CI gaps are ready for Epic 15 selection. | Complete | CI gaps table maps R167-02, R167-03, R167-04, R167-06, R167-07, and R167-08 to selection-ready closure choices. |
