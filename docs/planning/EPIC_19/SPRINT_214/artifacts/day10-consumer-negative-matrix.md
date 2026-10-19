# Sprint 214 Day 10: Consumer Negative Matrix

## Selected Path And Scope

The Sprint 214 decision remains static-first with explicit shared-library and
dynamic-ABI deferral. Day 10 adds negative **installed static consumer**
checks on local macOS. It does not build a shared library or execute a loader,
cross-architecture, or cross-version ABI experiment. Day 9 records the
positive static consumer output and the future shared-consumer fixture design.

## Executed Local Matrix

| Case | Invocation and failure diagnostic | Result |
| --- | --- | --- |
| Make-installed archive absent at link | `tests/test_install.sh` installs to a fresh prefix, obtains `pkg-config --cflags/--libs`, and runs both positive consumers with those flags. After renaming `lib/libsparse_lu_ortho.a`, the review follow-up requires installed-prefix validation to report the missing archive before any new package-flag link. A separate link naming the missing archive by absolute path checks the linker diagnostic; it does not prove that `-lsparse_lu_ortho` cannot find another host installation. The archive is restored before uninstall. | Day 10 originally passed 25/25 with missing-archive link rejection. The review follow-up passes 26/26 locally, including the validator rejection and explicit-path diagnostic. |
| CMake-installed archive absent | `tests/test_cmake_install.sh` builds/runs normal and exact-version installed consumers first, then temporarily moves the installed `.a` outside the prefix. `check_static_install_contract.py` must report `missing regular static archive`, and a new out-of-tree `find_package(Sparse)` configure must reject its imported target with a `references the file` diagnostic naming `libsparse_lu_ortho.a`. The archive is restored. | Passed locally. Both negative checks reported the expected class; total CMake result: 30 passed, 0 failed, 0 skipped. |
| Package version mismatch | The installed `VERSION` is `2.2.0`; the CMake test requests a lower same-major version (`2.1.0`) from a separate consumer source/build directory. It now requires CMake's incompatible-package diagnostic and the considered `version: 2.2.0` line, not merely any configure failure. Exact-version `2.2.0` still configures, builds, and runs. | Passed locally within the 30 CMake checks. This proves exact package-version selection only, not binary compatibility. |
| Unsupported shared configuration | `scripts/static_package_deferral_check.sh` configures with `BUILD_SHARED_LIBS=ON` and eleven unsupported selector aliases, requiring nonzero results and static-only/unsupported-selector diagnostics. It also inspects generated static package metadata and public non-claims. | Passed locally. No shared target was enabled. |
| Corrupt or missing package metadata | `tests/test_static_install_contract.py` mutates a temporary fixture: absent archive/`sparse.pc`/CMake export, wrong package version or imported location, extra shared/import artifacts, loader fields, and selectors. Each must produce the relevant validator error. | 14 fixture tests passed locally. These are metadata diagnostics, not a linked consumer for every mutation. |
| Positive static preservation | Make's basic and maintained-example consumers link/run from an installed prefix; CMake's example and exact-version consumers resolve `Sparse::sparse_lu_ortho STATIC IMPORTED` and run. Installed metadata remains static-only. | Make 25/25; CMake 30/30. Output retains version `2.2.0`, `nnz: 1`, `Solution: [1.000000, 1.000000, 1.000000]`, and `OK`. |

The first CMake run exposed an overly literal **test assertion**, not a
product failure: Python normalized a doubled slash in the temporary prefix
before printing the missing-archive path. The assertion now matches the
diagnostic class and archive basename. The archive is moved outside the
prefix so the fixture isolates a missing artifact, and the full CMake test
passed on rerun.

## Future-Only Cases

| Case not executed | Why it is not applicable to the selected static path | Future acceptance gate |
| --- | --- | --- |
| Wrong architecture or platform shared binary | No shared artifact, selected ABI identity, or cross-platform loader exists. The local archive test runs only on macOS x86_64. | On each named platform, install an incompatible-architecture or width/epoch artifact under a controlled prefix; metadata validation or load must fail before selected API use, with no fallback binary. Inspect ELF/Mach-O/PE identity as specified on Day 5. |
| Absent dynamic runtime path | A statically linked consumer does not require an installed project DLL/dylib/`.so` search path. `DYLD_LIBRARY_PATH` and `LD_LIBRARY_PATH` were unset in the local Day 9 environment. | Run the future shared consumer with and without the approved RPATH/install-name/DLL directory. Only the documented loader route may succeed; missing route must name the expected artifact. |
| Missing dynamic export | The current static archive has symbols used by the installed consumers, but there is no curated dynamic export table. | Compare the installed shared export table to the exact approved Day 3/4 manifest, then remove one required export in a controlled fixture and require link/load failure. |
| Retained old binary against new shared library | `find_package(... EXACT)` is package version matching, not old-binary/new-library evidence. | Preserve an old executable and its ABI manifest; run that unchanged binary against a later same-epoch installed shared artifact before claiming compatibility. |
| Windows DLL/import-library failure | Windows CI currently owns CMake static install/downstream proof and `sparse.pc` metadata-only inspection. A local Windows-style `.lib` fixture does not inspect COFF members or DLL loading. | Hosted MSVC fixture must distinguish the static `.lib` from an import `.lib`, inspect DLL/import metadata, and fail predictably for missing/wrong DLL or import library. |

## CI Registration And Diagnostic Ownership

| CI job | Registered check | Diagnostic owned |
| --- | --- | --- |
| Linux `package-contract` | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, `python3 tests/test_static_install_contract.py`, `bash scripts/static_package_deferral_check.sh` | Missing archive at link/configure, incompatible package version, installed metadata mutations, and unsupported shared selectors. |
| macOS `install-and-pkgconfig` | `bash tests/test_install.sh` | Installed Make static consumer and missing archive linker diagnostic. |
| macOS `cmake-install-export` | `bash tests/test_cmake_install.sh`, mutation fixtures, deferral/docs guard | Installed CMake static consumer, missing imported archive, package version mismatch, and selector deferral. |
| Windows `install-and-downstream` | Python mutation fixtures, installed-prefix validator in Windows mode, and PowerShell CMake static consumer/version checks | Static artifact/metadata diagnostics and consumer execution. The hosted job has not run for this branch; no new Windows missing-archive consumer execution is claimed. |

The workflow files parsed as YAML, and a repository search confirmed the
above command registrations. The installed tests themselves require the
named failure text; a mere nonzero command status cannot pass the new Unix
consumer checks. CI definitions are ownership, not hosted execution evidence.

## Commands And Boundary

```sh
bash -n tests/test_install.sh tests/test_cmake_install.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 tests/test_static_install_contract.py
bash scripts/static_package_deferral_check.sh
```

All listed commands passed locally on macOS after the diagnostic assertion
fix. No C/header file changed, so the C format/lint/test gate was not required
for Day 10. The selected static package still does not advertise a runtime
loader, a stable dynamic ABI, or broad platform/release readiness. Day 11
should start the release checklist from those explicit non-claims.
