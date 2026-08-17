# Sprint 162 Day 12 Cross-Platform Package Validation

## Scope

Day 12 re-runs the package confidence paths available from the current macOS
development environment and records the hosted-only Windows validation
expectations. The Sprint 162 package decision remains static-first and
CMake-first on Windows, with Windows Makefile and `pkg-config` execution parity
retained as non-claims.

## Available Local Validation Record

| Command | Result | Package Surface |
| --- | --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-first package contract, support wording, Windows package non-claim wording, and no unselected Windows package execution in CI. |
| `bash tests/test_install.sh` | Passed 23 checks | Make install/uninstall, installed static archive, headers, `sparse.pc`, `pkg-config` command proof, downstream `pkg-config` consumers, and unsupported package/ABI wording. |
| `bash tests/test_cmake_install.sh` | Passed 27 checks | CMake install/export, static imported target metadata, downstream CMake example, exact-version behavior, mismatch rejection, installed `sparse.pc` metadata, and unsupported shared-loader metadata. |

## Static Guard Output

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

## Make Install And pkg-config Output

```text
=== Install Validation Tests ===
  [PASS] static library installed
  [PASS] no shared-library artifacts installed
  [PASS] all 19 headers installed
  [PASS] pkg-config file installed
  [PASS] pkg-config can resolve sparse
  [PASS] pkg-config exact version constraint works
  [PASS] pkg-config prefix points at install prefix
  [PASS] pkg-config libdir points at installed libdir
  [PASS] pkg-config includedir points at installed includedir
  [PASS] pkg-config --cflags returns installed include path
  [PASS] pkg-config --libs returns installed static archive link flags
  [PASS] pkg-config --static libs match current self-contained link flags
  [PASS] pkg-config file has no private dependency stanza
  [PASS] pkg-config file describes static archive package metadata
  [PASS] pkg-config file has no unsupported packaging or ABI claims
  [PASS] pkg-config --modversion returns 2.2.0
  [PASS] basic pkg-config consumer compiles and links
  [PASS] basic pkg-config consumer runs correctly
  [PASS] maintained example source compiles with pkg-config
  [PASS] maintained example source runs with pkg-config install
  [PASS] library removed after uninstall
  [PASS] headers removed after uninstall
  [PASS] pkg-config file removed after uninstall
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

## CMake Install/Export Output

```text
=== CMake Install Validation Tests ===
  [PASS] cmake configure
  [PASS] cmake build
  [PASS] cmake install
  [PASS] static library installed
  [PASS] no shared-library artifacts installed
  [PASS] headers installed (19 files)
  [PASS] SparseConfig.cmake installed
  [PASS] SparseConfigVersion.cmake installed
  [PASS] SparseTargets.cmake installed
  [PASS] sparse.pc installed
  [PASS] CMake imported target is static
  [PASS] CMake package has no shared-library imported metadata
  [PASS] CMake package has no unsupported loader or shared-selector metadata
  [PASS] CMake imported target uses install include prefix
  [PASS] CMake imported archive uses install prefix
  [PASS] CMake package has no source-tree paths
  [PASS] CMake package has no build-tree paths
  [PASS] pkg-config metadata describes static archive package
  [PASS] pkg-config metadata has no unsupported package or ABI claims
  [PASS] cmake_example configure (find_package works)
  [PASS] cmake_example build
  [PASS] cmake_example runs correctly
  [PASS] find_package exact installed version works
  [PASS] find_package exact-version consumer builds
  [PASS] find_package exact-version consumer runs correctly
  [PASS] find_package mismatched version is rejected
  [PASS] pkg-config version = 2.2.0
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Hosted-Only Windows Verification Checklist

The following checks require hosted Windows CI because this local environment
does not provide PowerShell Core, MSVC, or the Visual Studio 17 2022 generator:

- `Windows enforced reviewed CMake consumer subset (MSVC)`;
- `EXPECTED_WINDOWS_CTEST_COUNT=59`;
- full hosted `ctest` execution on `windows-2022`;
- `Windows reviewed CMake install/downstream validation path`;
- installed static `lib/sparse_lu_ortho.lib`;
- no installed DLL artifacts;
- 19 installed headers plus `sparse_version.h`;
- CMake package files installed under `lib/cmake/Sparse`;
- installed `lib/pkgconfig/sparse.pc` metadata-only inspection;
- generated installed CMake downstream consumer configure/build/run;
- maintained `examples/cmake_example` configure/build/run;
- exact-version installed CMake consumer configure/build/run;
- mismatched-version CMake package rejection;
- no Windows `pkg-config`, `make install`, or `make uninstall` execution.

## Package Evidence Support-Tier Notes

| Platform Surface | Evidence Status |
| --- | --- |
| Linux/macOS Make install and `pkg-config` | Locally passed through `tests/test_install.sh`; reviewed hosted lanes remain Linux/macOS owned. |
| CMake install/export/downstream | Locally passed through `tests/test_cmake_install.sh`; Windows has a hosted CMake-first analogue. |
| Windows CMake install/downstream | Hosted-only verification; source-level workflow wording and static guard passed locally. |
| Windows `sparse.pc` | Metadata-only inspection on Windows; command execution proof remains Linux/macOS owned. |
| Windows Makefile install/uninstall | Retained non-claim. |
| Shared library, dynamic ABI, runtime-loader, package-manager support | Retained non-claims guarded by static package checks. |

## Local Tool Availability

- `actionlint`: not available locally.
- `pwsh`: not available locally.

Those limitations do not block the Sprint 162 local validation because the
changed implementation is source-controlled guard/docs/workflow wording. Hosted
Windows execution remains the required final proof for the Windows lane.

## Day 12 Conclusion

Available local package confidence paths passed. Hosted-only Windows
expectations are explicit, and the package evidence remains static-first,
CMake-first on Windows, and bounded away from Windows Makefile or `pkg-config`
execution parity.
