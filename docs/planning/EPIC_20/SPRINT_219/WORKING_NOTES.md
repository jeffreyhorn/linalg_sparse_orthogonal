# Sprint 219 Working Notes

**Plan:** [PLAN.md](PLAN.md)
**Current scope:** Day 14 local closeout complete; hosted acceptance pending
**Source revision:** `145b5a5a21dbb5191d5d1e1e2550b80c98819861`

These notes retain checks and unresolved risks as the index-width contract
is implemented. They do not promote an unrun or mismatched consumer to a
supported configuration. The detailed pre-change map is in the
[Day 1 ABI inventory](artifacts/day1-public-abi-width-inventory.md).
The [Day 2 contract design](artifacts/day2-width-contract-failure-design.md)
freezes the selection, package-identity, pre-use check, and negative-test
matrix; it is not implementation evidence.
The [Day 3 CMake selection evidence](artifacts/day3-cmake-width-selection.md)
records separate 32/64 build trees and configure-time negative checks.
The [Day 4 Make selection evidence](artifacts/day4-make-width-selection.md)
records Make's width stamp, clean-tree checks, and 32/64 focused tests.
The [Day 5 parity evidence](artifacts/day5-build-system-parity.md) records
matching generated headers, the shared width-identity test, and the remaining
installed-consumer boundary.
The [Day 6 CMake install evidence](artifacts/day6-cmake-installed-width.md)
records width-aware imported targets and relocated 32/64 consumers.

## Owner Map

| Owner | Current surface | Next evidence owner |
| --- | --- | --- |
| Core public ABI | `include/sparse_types.h`, `src/sparse_types.c`, generated `sparse_version.h` | Day 6 header rejection landed early; Days 7-8 own metadata cross-check and broader consumer negatives. |
| Public index-bearing consumers | Matrix/IO, CSR/CSC, analysis, reorder, direct/iterative/eigs/QR/SVD headers and callback structs | Days 8-11 installed probes and selected 64-bit functional lane. |
| Make package | `Makefile` compile rules and install recipe, `sparse.pc.in` | Day 4 selector and stamp landed; Days 7 and 10 own installed metadata and consumer tests. |
| CMake package | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, exported `SparseTargets.cmake` | Day 6 installed usage requirement and relocated 32/64 smoke landed; Day 7 owns validator checks and Day 9 the wider matrix. |
| Contract tests | `tests/test_idx_width.c`, `tests/test_sparse_matrix.c`, install scripts, static validator | Day 5 build-tree identity landed; Days 7-10 own installed positive/negative matrix. |
| Hosted scope | Linux/macOS Make and CMake install jobs; Windows CMake install job | Day 12 registration/status; Day 14 gate review. |

## Initial Risk Ledger (Day 6 Snapshot)

| ID | Risk and current evidence | Required closure |
| --- | --- | --- |
| R219-1 | Maintained CMake consumers now reject explicit header-width conflicts, but raw manual links can still omit the pre-use query. | Finish installed metadata/validator checks and broader consumer negatives. |
| R219-2 | Build-tree identity and the CMake example have safe pre-use checks; other installed fixtures and package routes still need the same coverage. | Reject every maintained installed mismatch before any index-bearing call. |
| R219-3 | CMake target now exports width, but `sparse.pc` does not and the static validator does not compare package/archive identities. | Day 7 pkg-config propagation and validator mutations. |
| R219-4 | Make build trees are stamped and Make/CMake generated headers match; CMake reconfigure and installed-package integrity still lack negative proof. | Exercise N8 and validate generated metadata against the archive. |
| R219-5 | A genuine 64-bit library/solver/IO/conversion run has not been performed in this sprint. | Selected local and hosted 64-bit lane with exact results. |

## Claims Boundary

| Claim | Day 1 status |
| --- | --- |
| Default 32-bit source/static build | Observed locally: fresh Make and CMake archives built; Make `test_sparse_matrix` passed 63/63. |
| Default 32-bit installed package file contract | Observed locally: both fresh prefixes passed `check_static_install_contract.py`. This checker does not validate index width. |
| 64-bit `idx_t` typedef compiles in a consumer | Observed only in a safe probe; does not prove 64-bit archive support. |
| Mismatched 64-bit consumer and 32-bit package is rejected | False today: Make/pkg-config probe linked and ran its width query; CMake consumer compiled and linked. Neither invoked unsafe APIs. |
| 64-bit library and installed consumer correctness | Untested; no support claim from Day 1. |
| Mixed-width, cross-version, or shared ABI | Unsupported; outside Sprint 219's intended claim. |

## Command And Evidence Log

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `make -j4 BUILDDIR=build/s219-day1-default build/s219-day1-default/libsparse_lu_ortho.a` | Exit 0, no `SPARSE_IDX_BITS` compile flag. | Fresh default Make archive on Apple Clang 17. |
| 2026-10-24 | `make BUILDDIR=build/s219-day1-default build/s219-day1-default/test_sparse_matrix`; run binary | Exit 0; 63/63 cases and 473 assertions passed, including `test_idx_width_contract`. | Same-build default 32-bit test, not a consumer-negative test. |
| 2026-10-24 | Install Make archive to `build/s219-day1-make-prefix`; `pkg-config --cflags sparse` | Exit 0; only installed include path emitted. | No width metadata. |
| 2026-10-24 | Default and forced-64 query probes against the 32-bit Make archive | Both exit 0; `consumer=32 library=32` and `consumer=64 library=32`. | Query only; mismatched data was not passed to the library. |
| 2026-10-24 | Configure/build/install CMake archive under `build/s219-day1-cmake*` | Exit 0; imported target has include/link interface but no width definition. | Default CMake archive only. |
| 2026-10-24 | Configure/build `examples/cmake_example` with `-DCMAKE_C_FLAGS=-DSPARSE_IDX_BITS=64` against 32-bit install | Both exit 0; mismatched executable intentionally not run. | Compile/link rejection absent, runtime solver behavior not tested. |
| 2026-10-24 | `check_static_install_contract.py` for Make and CMake prefixes | Both report `static-install-contract: PASS`. | Static artifact contract only; width not checked. |

## Day 2 Handoff

The Day 2 design is complete. Days 3-5 own build selectors and identity;
Days 6-8 own installed metadata and the pre-use check; Days 9-10 own the
isolated consumer matrix. A generated-header compile guard rejects an
effective consumer width that differs from package width. A safe
`sparse_idx_bits()` query catches swapped archives before maintained fixtures
call index-bearing APIs. Package metadata uses a distinct
`SPARSE_PACKAGE_IDX_BITS` macro so it cannot overwrite a consumer's explicit
`SPARSE_IDX_BITS`. Same-name duplicate `-D` options with a masked earlier
conflict cannot always be diagnosed, and raw manual links that skip the
pre-use check remain outside the maintained contract.

| Date | Day 2 decision / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | Reviewed Day 1 ABI evidence, `sparse_types.h`, generated header, Make/CMake install rules, and static validator | Frozen P1-P4/N1-N8 matrix and ownership in the Day 2 artifact. | Design only; no package rejection or 64-bit functional run yet. |

## Day 3 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | Configure independent CMake trees at `SPARSE_IDX_BITS=32` and `64`; build `test_sparse_matrix` in each | Both configure/build paths exited 0. | Build-tree evidence, not installed packages. |
| 2026-10-24 | Focused `ctest -R '^test_sparse_matrix$'` in each tree | 1/1 passed in each; same-build typedef and archive query agree. | Does not exercise all algorithms at 64 bits. |
| 2026-10-24 | Inspect both `compile_commands.json` files with `jq` | All 139 commands in each tree carry their selected `-DSPARSE_IDX_BITS=N`; representative benchmark and example built at 64 bits. | Compile propagation only; no benchmark/solver run. |
| 2026-10-24 | `CMAKE_BUILD_PARALLEL_LEVEL=4 bash tests/test_cmake_install.sh` | 38 passed, 0 failed, 0 skipped; invalid `0`, `16`, `128`, `text`, empty, and three ambient flag forms rejected at configure. | Default installed route still has no exported width identity. |

Day 4 owns Make selection and stale-tree refusal. Day 5 owns build-system
parity and any reconfigure/stale-identity cases not proven today. The
installed mismatch risks R219-1 through R219-3 remain open until the
generated header, CMake target, pkg-config metadata, validator, and pre-use
check are implemented.

## Day 4 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash -n tests/test_make_idx_width.sh`; `bash tests/test_make_idx_width.sh` | Syntax check passed; 23 width checks passed, 0 failed. | Local macOS GNU Make 3.81 only; hosted registration remains later work. |
| 2026-10-24 | Build library, `test_sparse_matrix`, benchmark, and example in separate 32/64 temp `BUILDDIR` values | Both builds passed; same-build typedef/archive query test passed at both widths. | Focused build-tree proof, not installed-consumer proof. |
| 2026-10-24 | Invalid selectors, CFLAGS/CC conflicts, opposite-width reuse, unstamped object/header | All rejected before unsafe reuse; archive checksum unchanged on opposite-width request. | Cannot recover earlier duplicate assignments to the same Make CLI variable. |
| 2026-10-24 | Dry-run `lint` at 64 bits | Strict compile and clang-tidy commands included the selected width. | Dry run only; lint analysis itself was not executed. |
| 2026-10-24 | `make format-check` | Passed with clang-format dry run. | No C or header source was changed on Day 4. |

Day 5 should compare Make/CMake defaults, generated headers, compile flags,
and archive queries in clean trees. The Make stamp closes the silent
opposite-width reuse path for maintained Make invocations, but R219-4
remains open for cross-build-system and installed-package identity. No
installed-width safety claim is made yet.

## Day 5 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | Build/run shared `test_idx_width` in fresh CMake default, 32, and 64 trees | Default and explicit 32 reported `typedef=32 header=32 archive=32`; 64 reported all 64. Focused CTest passed 2/2 in each explicit tree. | Same-build check, not automatic consumer enforcement. |
| 2026-10-24 | Generate Make 32/64 headers and compare with CMake using `cmp` | Both comparisons passed byte-for-byte. | Headers alone do not prove installed archive integrity. |
| 2026-10-24 | `bash tests/test_make_idx_width.sh` | 26 passed, 0 failed; fresh default/32/64 identity and safe mixed-archive negative included. | Local macOS Make lane only. |
| 2026-10-24 | `make format-check` | Passed. | Formatting only. |
| 2026-10-24 | `make BUILDDIR=build/s219-day5-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck. | Default 32-bit lint lane only. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day5-quality -j4 test` | All Make test binaries passed, including `test_idx_width`. | Default 32-bit full suite; 64-bit algorithm lane is Day 11. |
| 2026-10-24 | `make BUILDDIR=build/s219-day5-quality source-list-check` | Passed: 49 library sources. | Default unstamped `build/` tree was left untouched after its width guard rejected an initial run. |

The generated `SPARSE_BUILT_IDX_BITS` macro is ready for package use. Days
6-8 must still carry the width through exported CMake/pkg-config metadata,
validate the package, and reject consumer overrides before index-bearing
calls. A safe test executable failing on swapped files does not protect a
consumer that never invokes it. Item 219.2 is closed; R219-1 through R219-3
and the installed portion of R219-4 remain open.

## Day 6 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash tests/test_cmake_idx_width_install.sh` | 8/8 passed: relocated 32/64 imports, target width flags, LU solves, and compile-time conflicts. | Does not validate a deliberately swapped archive or pkg-config route. |
| 2026-10-24 | `CMAKE_BUILD_PARALLEL_LEVEL=4 bash tests/test_cmake_install.sh` | 38/38 passed with static package validator and existing CMake consumer. | Default 32-bit install lane. |
| 2026-10-24 | `bash tests/test_make_idx_width.sh` | 26/26 passed after public-header guard. | Build-tree Make width checks, not installed pkg-config consumers. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day6-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck after adding the generated-header include path and static typedef assertion. | Default 32-bit lint lane; 64-bit focused build passed separately. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day6-quality -j4 test` | Full Make suite passed on final source state. | Default 32-bit full suite; selected 64-bit algorithm lane remains Day 11. |
| 2026-10-24 | `make format-check`; direct clang-format check of `examples/cmake_example/main.c`; isolated `source-list-check` | All passed; 49 library sources. | Local quality checks, not hosted CI. |

Day 6 closes the imported-target propagation and clean-prefix CMake smoke
for both supported widths. The public-header conflict check was implemented
ahead of Day 8 to make the 64-bit installed target usable without a manual
selector and make explicit conflicts fail at compile time. Day 7 must publish
width in both `sparse.pc` paths and validate all installed identities against
the archive; Days 8-10 must complete the remaining installed consumer matrix.

## Day 7 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `python3 tests/test_static_install_contract.py` | 26 mutation/positive tests passed, including missing, duplicate, unsupported, misplaced, and mismatched package widths. | Synthetic fixtures mock the archive query; native links are covered below. |
| 2026-10-24 | `bash tests/test_pkgconfig_idx_width_install.sh` | 16/16 passed for Make and CMake at 32/64 bits: clean-prefix install, real archive query, exact pkg-config flags, and solver consumer. | Native Unix-like host only; Windows probe awaits CI. |
| 2026-10-24 | `bash tests/test_install.sh` | 26/26 passed with two-flag pkg-config output and a downstream consumer. | Default 32-bit Make lane. |
| 2026-10-24 | `bash tests/test_cmake_install.sh` | 38/38 passed with static validator and installed CMake example. | Default 32-bit CMake lane. |
| 2026-10-24 | `make format-check`; `bash -n tests/test_pkgconfig_idx_width_install.sh` | Both passed. | No C or header source changed on Day 7. |
| 2026-10-24 | `git diff --check` | Passed. | Patch hygiene only. |

Day 7 closes the pkg-config metadata and four-route native archive-width
agreement checks. The static validator fails closed if it cannot query the
archive. Day 8 still owns swapped-archive and pre-use consumer negatives;
Days 9-10 will broaden the isolated installed-consumer matrix.

## Day 8 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash tests/test_installed_width_check.sh` | 12/12 passed: clean-prefix 32/64 CMake and pkg-config consumers solved after the width check; opposite archives failed validator and both consumers before solver use. | Native macOS test; no automatic protection for manual consumers that omit the check. |
| 2026-10-24 | `bash -n tests/test_installed_width_check.sh`; direct `clang-format --dry-run --Werror` on example/helper; `make format-check` | All passed. | Formatting and shell syntax only. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day8-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck. | Default 32-bit library lint lane. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day8-quality -j4 test` | Full Make suite passed. | Default 32-bit test lane; selected 64-bit algorithm lane remains Day 11. |
| 2026-10-24 | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` | 26/26 and 38/38 passed, respectively, with the shared fixture check. | Default 32-bit existing install suites. |

The reusable fixture check is unconditional at the beginning of each
maintained consumer. Item 219.3's 26-hour propagation allocation is closed;
the first 4 hours of item 219.4 are complete. Day 9 owns the remaining
CMake-installed consumer matrix and stale/mixed-prefix negatives.

## Day 9 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash tests/test_cmake_idx_width_install.sh` | 18/18 passed. Both relocated packages auto-selected their width and solved; wrong explicit width, foreign header, foreign archive, exact version, and stale cache cases were rejected at their expected phases. | Native macOS static packages; no shared-library or cross-version claim. |
| 2026-10-24 | Static validator on both clean installed prefixes | Both header, target, and real archive identities agreed before relocation. | The override fixture deliberately bypasses the validator to exercise the executable pre-use check. |
| 2026-10-24 | `bash -n tests/test_cmake_idx_width_install.sh`; `make format-check`; `git diff --check` | All passed. | No C or header source changed on Day 9; Day 8 full lint/test remains the last C-source quality run. |

The CMake consumer matrix closes Day 9's 10-hour portion of item 219.4.
The fixture requests the exact current package version and confirms wrong
versions fail during configure. A stale `Sparse_DIR` binding is made visible
by an opposing explicit width, which fails at compile time. Day 10 should
complete the pkg-config and remaining mixed-prefix consumer cases.

## Day 10 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash tests/test_pkgconfig_idx_width_install.sh` | 40/40 passed across Make/CMake installs at 32/64 bits. Each prefix passed static identity validation, emitted exact pkg-config flags and version, and ran both a normal pkg-config link and an isolated absolute-archive link. | Native macOS static packages; no shared-library claim. |
| 2026-10-24 | Cross-route/opposite-width `PKG_CONFIG_PATH`, include order, absolute archive, `-L` order, and missing archive mutations | All 20 negatives rejected: discovery prefix, compile-time header, executable pre-use, or link phase as appropriate. | The pre-use guard is in the maintained consumer fixture, not automatic for manually linked clients. |
| 2026-10-24 | `bash -n tests/test_pkgconfig_idx_width_install.sh`; `make format-check`; `git diff --check` | All passed. | Day 10 changed no C/header source; Day 8 full lint/test remains the last C-source quality run. |

Item 219.4's full 24-hour consumer allocation is closed: Day 8 supplied the
initial pre-use checks, Day 9 the imported-target matrix, and Day 10 the
four-route pkg-config/mixed-prefix matrix. An additional positive link uses
the validated absolute archive for each route, preventing a host
installation with the same library name from satisfying the overall fixture.
Day 11 should now exercise selected 64-bit algorithms and index boundaries.

## Day 11 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `bash tests/test_cmake_idx_width_functional.sh` | Fresh 64-bit CMake cache, generated header, test/library compile flags, and runtime archive query agree; focused CTest 1/1 and all five named cases passed. | Local macOS selected lane only, not all algorithms or hosted platforms. |
| 2026-10-24 | `make format-check`; `bash -n tests/test_cmake_idx_width_functional.sh` | Both passed. | Formatting and shell syntax checks. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day11-quality -j4 lint` | Passed strict compile, clang-tidy, and cppcheck, including the new C test. | Default 32-bit Make lint lane. |
| 2026-10-24 | `make -s BUILDDIR=build/s219-day11-quality -j4 test` | Full Make suite passed, including the new test at 32 bits. | Default width; the dedicated script is the 64-bit functional proof. |

The selected fixture covers insertion, a deterministic LU solve, Matrix
Market IO, CSR/CSC conversion and reconstruction, and index text roundtrips
above `INT32_MAX` without large allocation. No selected-case defect required
repair. Day 12 should register the same focused 64-bit lane in CI, preserve
the default-width jobs, and record actual hosted results separately.

## Day 12 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | Register Linux/macOS selected 64-bit jobs and installed-consumer matrices | Workflow YAML parsed; existing default-width reviewed jobs retained. | No hosted execution on the unpublished branch. |
| 2026-10-24 | Extend Windows CMake install job and count guard | Default 32-bit installed header, target, metadata, and pre-use consumer checks registered; 62 CTests enumerated locally. | Windows PowerShell executable checks unavailable without `pwsh`; no Windows pkg-config execution or 64-bit claim. |
| 2026-10-24 | `bash tests/test_cmake_idx_width_functional.sh`; `python3 tests/test_validate_windows_powershell.py` | Local macOS selected CTest 1/1 and five cases passed; structural mutation tests passed. | Local, not hosted. |
| 2026-10-24 | `gh run list --branch sprint-219`; PR lookup | No branch runs or PR; current branch has no upstream. | Hosted status is unavailable, not passed. |

Detailed scope and provenance are in
[`artifacts/day12-hosted-64-bit-registration.md`](artifacts/day12-hosted-64-bit-registration.md).
**Day 12 completion and item 219.5 remain open** until actual hosted Linux,
macOS, and relevant package-job results are green and linked. The older
successful `master` runs are baseline only, not evidence for this change.

## Day 13 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | Public header, README, and `INSTALL.md` width contract updates | Documented explicit Make/CMake 32/64 builds, isolated pkg-config and imported-target consumption, pre-use archive check, and static-only non-claims. | Documentation does not create hosted evidence. |
| 2026-10-24 | `bash tests/test_cmake_idx_width_install.sh`; `bash tests/test_pkgconfig_idx_width_install.sh` | 18/18 and 40/40 passed locally; generated header, CMake target, pkg-config metadata, and archive widths reconciled. | Local macOS installed static packages. |
| 2026-10-24 | `make format-check`; isolated default-width `make lint`; isolated default-width `make test` | All passed. | Default-width full quality, not a universal 64-bit suite. |

The [Day 13 contract record](artifacts/day13-public-install-width-contract.md)
maps generated fields to the user commands and retained claims. Day 13's
documentation completion criteria are met locally. Day 14 still owns the
final sprint quality sweep, and Day 12's required hosted 64-bit results
remain unavailable on this unpublished branch; item 219.5 is not closed.

## Day 14 Evidence And Handoff

| Date | Command / check | Outcome | Limit |
| --- | --- | --- | --- |
| 2026-10-24 | `make format`; final `make format-check`, `make source-list-check`, `git diff --check` | Passed; source list has 49 library files. | Local source and diff checks. |
| 2026-10-24 | Isolated default-width `make lint` and `make test` after Day 14 C fixes | Both passed. | Default 32-bit Make path. |
| 2026-10-24 | `make quality-review-cmake` after Day 14 C fixes | Clean build and 62/62 CTests passed. | Default 32-bit local CMake parity. |
| 2026-10-24 | Both-width build/install/consumer scripts, package validator, selected 64-bit functional script | All passed locally, including negative width and mixed-prefix fixtures. | Static native macOS packages and selected 64-bit algorithms. |
| 2026-10-24 | 64-bit focused UBSan CMake/CTest | 3/3 selected tests passed with width-sensitive warnings as errors. | Full 64-bit sanitizer target was interrupted at long-running `test_reorder_nd`; no full-suite pass. |
| 2026-10-24 | Hosted branch runs and PR lookup | No runs, PR, or upstream branch. | Required Linux/macOS/Windows hosted acceptance remains unavailable. |

The [Day 14 closeout record](artifacts/day14-quality-and-contract-closeout.md)
contains exact commands, pass counts, width identities, and sanitizer limits.
Local Day 14 quality work is complete. **Sprint 219 remains open:** item
219.5's hosted gate and item 219.6's final acceptance gate require green
branch-specific hosted jobs before their allocations or the 140-hour sprint
total can be closed. No commit, push, or PR was requested for this day.
