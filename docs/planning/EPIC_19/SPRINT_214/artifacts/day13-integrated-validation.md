# Sprint 214 Day 13: Integrated Validation

## Scope And Environment

Selected path: static-first package with shared-library, dynamic ABI,
runtime-loader, public provider, binary-release, and release-readiness
claims deferred. This is a local integrated validation of the uncommitted
`sprint-214` worktree on Darwin 24.6.0 x86_64 with Apple Clang and CMake;
`HEAD` was `4e896822` at inspection time. It is not a hosted Linux, macOS,
or Windows candidate result, and no release tag/archive was produced.

## Executed Checks

| Command | Result | Evidence and scope |
| --- | --- | --- |
| `bash tests/test_install.sh` | PASS: 25, failed 0 | Fresh temporary Make install, 19 headers, static archive/`sparse.pc`, version `2.2.0`, two installed `pkg-config` consumers, missing-archive link rejection, uninstall cleanup. |
| `bash tests/test_cmake_install.sh` | PASS: 30, failed 0, skipped 0 | Fresh CMake install/export, static imported target and prefix metadata, installed consumers, exact-version acceptance, wrong-version rejection, missing imported archive/validator diagnostics. |
| `python3 tests/test_static_install_contract.py` | PASS: 14 tests | Positive and mutation fixtures for missing/static/shared metadata, package version, and installed-prefix contract. Windows-style fixtures are not a hosted Windows build. |
| `bash scripts/static_package_deferral_check.sh` | PASS | `BUILD_SHARED_LIBS=ON` and unsupported link-mode selectors rejected; default generated package metadata remains static-only; no shared export/ABI metadata; public and Windows package non-claims preserved. |
| `make api-docs-local-only` | PASS | Doxygen coverage for 18 checked-in public headers, local-only workflow and ignore checks, plus the complete generated API local-only regression suite. Generated `docs/api/html/` remains ignored and local. |
| `make support-docs-guard` | PASS | Public support quick-reference documentation guard. |
| `make package-manager-deferral-guard` | PASS | Sprint 171/198/207 records, provider recipe absence, public provider non-claims, metadata neutrality, and developer-mode local Homebrew static-source proof boundary. The Homebrew proof bootstrapped CMake from source; it is not public provider or release evidence. |
| `make quality-review-compile` | PASS | Format-check, 49-source list, strict compile/lint with `clang-tidy` and `cppcheck` across 109 files, and LDLT CSC helper guard/regression. |
| `make quality-review-cmake` | PASS: 60/60 CTest | Clean CMake rebuild, 60 registered tests equal 59 Make tests plus one focused CTest-only selector, then full CTest execution. |
| `python3 tests/test_validate_windows_powershell.py` | PASS | Workflow wiring, nine selected PowerShell snippets, and Windows scope regressions. A printed `--require-pwsh` failure is an expected negative fixture; no local `pwsh` or hosted Windows run is claimed. |
| `bash -n scripts/static_package_deferral_check.sh tests/test_install.sh tests/test_cmake_install.sh` | PASS | Syntax validation of changed Bash surfaces. |

`make quality-review-cmake` emitted one non-fatal Apple Clang
`-Wdouble-promotion` warning at unchanged `tests/test_svd_partial_corpus.c:83`
for returning `INFINITY` as `double`. The reviewed target exited zero and all
tests passed. The warning is recorded here rather than treated as a new
Sprint 214 C/header change.

## Failures, Reruns And Skips

No required local selected-path command failed, and no failed command needed
a code fix or rerun. The Homebrew dependency bootstrap and the API-docs
fixture suite were long-running but finished successfully.

`make format && make lint && make test` was not separately required: no
`*.c` or `*.h` file changed in this sprint worktree. Its format/lint
components were exercised by `make quality-review-compile`, and the full
CMake/CTest suite was exercised by `make quality-review-cmake`; this is not
being relabeled as a `make test` run. A selected shared-library consumer or
loader test was not run because Day 2 selected exact static deferral rather
than a shared proof. Its future acceptance tests remain in the Day 3-5 and
Day 9-10 artifacts.

Hosted Linux/macOS/Windows branch jobs, native Windows MSVC package proof,
Windows `pkg-config` execution, source-archive reproducibility, binary
provenance, tag/signing, and release approval were not run or claimed by this
local Day 13 validation. They remain separate CI or future release gates.

## Changed Surface And Day 14 Handoff

Validated existing sprint edits include `CMakeLists.txt`, static package
guard, Make/CMake install tests, static install contract validator/fixtures,
Linux/macOS/Windows workflow registrations, and README/INSTALL/maintainer
non-claim wording. Day 13 adds this transcript and updates working notes only;
it does not alter code, build rules, workflow behavior, or public claims.

Day 14 should audit each item against these exact results, recheck affected
gates after any further edit, update Epic 19/residual status, and keep hosted
CI and release-readiness evidence absent until actually produced. Passing
local static proof does not supersede Sprint 170/207 deferrals.
