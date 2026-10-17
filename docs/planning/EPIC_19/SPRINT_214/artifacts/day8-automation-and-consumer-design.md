# Sprint 214 Day 8: Automation Gate And Consumer Design

## Selected Boundary

Day 2 selected ABI/release design and stronger static-first deferral, not a
shared-library implementation. Day 8 connects the Day 6 configure guard and
Day 7 installed-prefix validator to existing package CI lanes. No dynamic
loader metadata is required for the selected static package; its presence is
an error. The Day 5 Linux, macOS, and Windows loader tests remain future
acceptance gates, not evidence earned here.

## CI Ownership

| Hosted job | Enforced selected-path checks | Deliberate limit |
| --- | --- | --- |
| Linux `package-contract` in `ci.yml` | Make and CMake temporary-prefix installs and static consumers; 14 installed-contract positive/mutation fixtures; configure/generated-metadata deferral and public docs non-claim guard; package-provider claim guard. | No shared artifact, loader, ABI compatibility, or release proof. |
| macOS `install-and-pkgconfig` in `macos-ci.yml` | Make static install, `pkg-config` resolution, out-of-tree compiled consumers, and installed-prefix validator. | No dylib or runtime-loader proof. |
| macOS `cmake-install-export` in `macos-ci.yml` | CMake static install, `find_package(Sparse)` consumers, 14 mutation fixtures, configure/generated-metadata deferral and public docs non-claim guard. | No shared import target or ABI epoch. |
| Windows `install-and-downstream` in `windows-ci.yml` | Hosted MSVC CMake static install/consumer checks plus the same installed-prefix validator with `--platform windows` and 14 mutation fixtures. Existing PowerShell checks still inspect headers, package files, static target, version behavior, and DLL absence. | `sparse.pc` remains metadata-only; no Windows Make/`pkg-config` execution, DLL search, or COFF archive-member proof. |

These workflow edits define required jobs on pushes and pull requests to the
repository's main branches. They do **not** establish that hosted jobs have
passed on this branch. The Windows Python validator invocation runs after
`cmake --install` and before the existing PowerShell artifact/consumer checks;
it compares installed file names and metadata, not `.lib` binary contents.
The static deferral script is the docs-claim check: it requires the README,
INSTALL, and maintainer non-claims and rejects its known positive shared/ABI
claim patterns. It also tests `BUILD_SHARED_LIBS=ON` and eleven unsupported
selector spellings with actionable configure diagnostics. No public support
wording changed on Day 8.

## Failure Diagnostics And Local Results

| Failure class | Required diagnostic / exercised fixture |
| --- | --- |
| Unsupported shared or link-mode selector | CMake configure fails with the unsupported selector and `static-only`; the deferral script verifies this for eleven aliases and `BUILD_SHARED_LIBS=ON`. |
| Missing static archive or package metadata | The installed validator reports `missing regular static archive`, `missing regular file`, or `missing CMake package files`; new fixtures remove each independently. |
| Accidental shared/import/runtime artifact | The validator reports `unexpected shared/import artifact`, `unexpected installed runtime artifact`, or an unexpected extra Windows `.lib`; existing fixtures cover `.so*`, `.dylib`, `.dll`, `.dll.a`, `.tbd`, and runtime placement. |
| Incorrect installed export or version | The validator rejects non-static imported targets, wrong or extra imported locations, loader/selector fields, and a changed CMake package version; fixtures exercise each class. |
| Missing shared-loader metadata | Not a selected-path failure: no shared artifact or loader contract exists. Any loader metadata in the static package is rejected instead. A future shared proof must use Day 5's per-platform positive metadata and runtime tests. |

Local macOS verification: `python3 tests/test_static_install_contract.py`
passed 14 tests; `bash tests/test_install.sh` passed 24/24; `bash
tests/test_cmake_install.sh` passed 28/28; and `bash
scripts/static_package_deferral_check.sh` passed. All three edited workflow
files parsed as YAML. `python3 scripts/validate_windows_powershell.py` passed
its structural checks but exited 2 because local `pwsh` is unavailable; it is
not a hosted Windows result. Its local unit suite,
`python3 tests/test_validate_windows_powershell.py`, passed. No C or header
file changed, so the C quality gate was not required for Day 8. The optional
mutex/OpenMP package variants remain outside the default installed-prefix
validator fixture scope.

## Item 214.4 Consumer Acceptance Specification

The selected path is **installed static consumption**. Day 9 must record the
actual out-of-tree run evidence without promoting a dynamic-ABI claim.

| Input or condition | Unix Make/`pkg-config` route | CMake route, including Windows MSVC |
| --- | --- | --- |
| Producer and prefix | Build default static archive, then install into a fresh `mktemp` prefix via `make install PREFIX=<prefix>`. | Configure default static target with `CMAKE_INSTALL_PREFIX=<prefix>` and `CMAKE_INSTALL_LIBDIR=lib`, build and install; use a distinct temporary prefix from the source/build tree. Windows uses `Release` and an x64 Visual Studio 2022 generator. |
| Consumer source and resolution | Compile a basic API/version consumer and maintained example from outside the installed tree; set `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig`, obtain `--cflags`/`--libs`, and verify they resolve only to the installed prefix. | Configure separate consumer build directories with `CMAKE_PREFIX_PATH=<prefix>`; use `find_package(Sparse REQUIRED)` and `Sparse::sparse_lu_ortho`. Also configure an `ExactVersion` consumer using the installed `VERSION`. |
| Link and runtime environment | Link the installed `.a` and required system libraries; execute with no `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` adjustment. | Target must be `STATIC IMPORTED` and resolve to the installed `.a` on Unix or installed static `.lib` on Windows. Execute without adding the install prefix to dynamic-library search paths. Windows `sparse.pc` is inspected, never executed via `pkg-config`. |
| Positive outcomes | Consumer compiles, links, prints installed version and expected matrix result, and exits zero; archive, 19 headers, `sparse.pc`, and no runtime artifact are observed. | Installed example/basic and exact-version consumers build/run; package metadata points to installed static archive, exact version succeeds, and no DLL/shared target or loader metadata appears. |
| Negative outcomes | Missing archive or redirected prefix/libdir fails validation or link; injected `.so*`/`.dylib`/`.dll` and unsupported `sparse.pc` fields fail the validator. | `BUILD_SHARED_LIBS=ON` and aliases fail configure; mismatched `find_package` version fails; missing archive/config, wrong target/location/version, extra import/shared artifact, or loader selector fails validator or consumer configuration. |

Existing install tests already exercise most positive and negative static
outcomes. Day 9 should retain their exact command/output evidence and specify
the future shared-consumer fixtures from Day 5: installed-only link, named
loader search policy, artifact ID and export inspection, missing/wrong-library
failure, and retained old-executable/new-library execution before any ABI
compatibility claim. No such dynamic fixture is executed by the selected
Day 8 automation.

## Handoff

If hosted CI rejects a legitimate default static install, diagnose the
platform-specific installed file or metadata before changing the contract;
do not accept a shared artifact to make the test pass. Day 9 owns consumer
evidence, Day 10 owns the negative matrix, and Days 13-14 must record hosted
job results or mark them unverified. The current public claim remains
static-first only.
