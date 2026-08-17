# Sprint 162 Day 13 Evidence And Claim Review

## Scope

Day 13 reviews the Sprint 162 package decision as one evidence surface: source
guards, workflow wording, public docs, maintainer docs, validation artifacts,
and retained non-claims.

The selected product decision remains:

- Windows package validation is CMake-first.
- Windows `sparse.pc` handling is metadata-only inspection.
- Windows Makefile install/uninstall parity is a retained non-claim.
- Windows `pkg-config` command execution parity is a retained non-claim.
- Linux/macOS remain the reviewed Make install and `pkg-config` execution
  proof surfaces.

## Positive Claim To Evidence Trace

| Positive Claim | Evidence Owner | Supporting Evidence | Boundary |
| --- | --- | --- | --- |
| Windows has reviewed CMake configure/build/CTest proof. | `.github/workflows/windows-ci.yml::build-and-test` | Visual Studio 17 2022 configure/build, `ctest -N`, `EXPECTED_WINDOWS_CTEST_COUNT=59`, full hosted `ctest`. | CMake/MSVC reviewed subset only. |
| Windows has reviewed CMake install/downstream package proof. | `.github/workflows/windows-ci.yml::install-and-downstream` | `cmake --install`, installed static `.lib`, headers, CMake package files, generated and maintained installed CMake consumers. | CMake install/downstream scoped. |
| Windows installs static package metadata. | `.github/workflows/windows-ci.yml`, `CMakeLists.txt`, `sparse.pc.in` | Installed `SparseConfig.cmake`, `SparseTargets.cmake`, `SparseConfigVersion.cmake`, and `lib/pkgconfig/sparse.pc`. | Metadata proof, not command proof for `pkg-config`. |
| Windows `sparse.pc` validation is metadata-only. | `.github/workflows/windows-ci.yml`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Workflow comments/logs and docs use metadata-only wording. | Does not run Windows `pkg-config`. |
| CMake package metadata remains static-first. | `.github/workflows/windows-ci.yml`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Static imported target checks, archive path checks, no DLL/shared imported metadata, no loader/static-shared selector metadata. | No shared-library or dynamic ABI support. |
| Linux/macOS Make install and `pkg-config` proof remains reviewed. | `tests/test_install.sh`, Linux/macOS CI lanes | Local Day 10 and Day 12 runs passed 23 checks; existing hosted lanes own reviewed execution. | Unix-like package execution proof, not Windows parity. |
| CMake install/export proof remains healthy locally. | `tests/test_cmake_install.sh` | Local Day 10 and Day 12 runs passed 27 checks. | Local CMake package proof; Windows analogue remains hosted-only. |

## Retained Non-Claim Trace

| Retained Non-Claim | Guard Or Documentation | Current Evidence |
| --- | --- | --- |
| Windows Makefile install/uninstall parity is not claimed. | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `.github/workflows/windows-ci.yml`, `scripts/static_package_deferral_check.sh` | Static guard checks public docs and workflow wording, and rejects Windows workflow `make install`/`make uninstall` execution. |
| Windows `pkg-config` execution parity is not claimed. | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `.github/workflows/windows-ci.yml`, `scripts/static_package_deferral_check.sh` | Static guard checks public docs and workflow wording, and rejects Windows workflow `pkg-config` execution. |
| Installed Windows `sparse.pc` is not command proof. | `.github/workflows/windows-ci.yml`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | CI wording and docs call this metadata-only inspection. |
| Shared-library support is deferred. | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | `BUILD_SHARED_LIBS=ON` is rejected and the static guard verifies rejection wording. |
| Dynamic ABI support is deferred. | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh`, docs | Static guard checks dynamic ABI deferral wording and absence of ABI/export/import metadata. |
| Runtime-loader behavior is unsupported. | `CMakeLists.txt`, `.github/workflows/windows-ci.yml`, `scripts/static_package_deferral_check.sh`, docs | Static guard checks runtime-loader deferral wording and workflow rejects loader metadata. |
| Package-manager support is unsupported. | `scripts/static_package_deferral_check.sh`, `tests/test_install.sh`, `.github/workflows/windows-ci.yml`, docs | Package metadata checks reject package-manager wording such as Homebrew, apt, dnf, pacman, vcpkg, and Conan. |
| Broad Windows parity is not claimed. | README, INSTALL, maintainer guide, Windows workflow | Windows wording remains scoped to CMake/MSVC tests plus CMake install/downstream validation. |

## CI And Docs Wording Review

Reviewed wording surfaces:

- `.github/workflows/windows-ci.yml`;
- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `docs/planning/EPIC_14/SPRINT_162/WORKING_NOTES.md`;
- Sprint 162 artifacts through Day 12.

Review result:

- positive Windows package claims are bounded by CMake install/downstream
  workflow evidence;
- `sparse.pc` is consistently treated as metadata-only on Windows;
- Makefile and `pkg-config` parity are treated independently;
- unselected Windows package execution is absent from workflow commands;
- package-manager, shared-library, dynamic ABI, runtime-loader, static/shared
  selector, and broad Windows parity claims remain unsupported;
- validation artifacts distinguish local Unix/CMake proof from hosted-only
  Windows proof.

## Diff Review Notes

Changed implementation surfaces remain narrow:

- `.github/workflows/windows-ci.yml`: wording and hosted-output diagnostics
  only; no new commands;
- `scripts/static_package_deferral_check.sh`: new retained non-claim wording
  and workflow command guards;
- `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`: support-tier wording
  alignment;
- `docs/planning/EPIC_14/SPRINT_162/*`: plan, working notes, and artifacts.

No C source, public header, CMake install rule, package template, Make install
script, or CMake install test was changed.

## Sprint 163 Performance Publication Handoff

Sprint 163 is `Methodology-Bound Performance Publication`. It must not reuse
Sprint 162 package evidence as performance evidence.

Handoff requirements:

1. Performance publication may cite Sprint 162 only for package support-tier
   boundaries, not speed, scalability, or superiority.
2. Windows package evidence remains CMake install/downstream scoped and
   metadata-only for `sparse.pc`.
3. Windows Makefile and `pkg-config` execution parity remain non-claims unless
   a future sprint selects and proves them.
4. Performance claims must be methodology-bound and tied to benchmark/report
   commands or generated report artifacts.
5. Package-manager, shared-library, dynamic ABI, runtime-loader, and broad
   Windows parity non-claims must not be softened by performance publication
   wording.

## Day 13 Conclusion

The Windows package decision is reviewable end to end. Positive package wording
is bounded by actual CMake install/downstream proof, and retained non-claims
are backed by explicit docs plus source-controlled guard checks. Sprint 163 is
ready to proceed with methodology-bound performance publication while keeping
package and performance evidence separate.
