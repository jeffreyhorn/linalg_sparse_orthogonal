# Sprint 165 Day 10 Package Metadata Validation

## Purpose

Day 10 validates the generated installed package metadata and confirms the
static-first boundary is present in the installed CMake and pkg-config files,
not just in source templates.

## Temporary Install Metadata Inspection

A temporary CMake configure/build/install was run outside the repository with:

```text
cmake -S . -B <tmp>/build -DCMAKE_INSTALL_PREFIX=<tmp>/usr -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_STANDARD=11
cmake --build <tmp>/build
cmake --install <tmp>/build
```

The installed metadata inspection reported:

| Check | Result |
| --- | --- |
| Installed package version | `2.2.0` |
| Installed static archive | present |
| Installed shared artifacts | absent |
| Installed headers | `19` |
| CMake imported target type | `Sparse::sparse_lu_ortho` is `STATIC IMPORTED` |
| CMake include metadata | uses `${_IMPORT_PREFIX}/include` |
| CMake archive metadata | uses `${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a` |
| CMake unsupported shared/loader metadata | absent |
| CMake source/build path leaks | absent |
| `sparse.pc` name | `Name: sparse` |
| `sparse.pc` description | `Description: Static archive package metadata for sparse linear algebra` |
| `sparse.pc` version | `Version: 2.2.0` |
| `sparse.pc` cflags | `Cflags: -I${includedir}` |
| `sparse.pc` libs | `Libs: -L${libdir} -lsparse_lu_ortho -lm` |
| `sparse.pc` unsupported package/ABI metadata | absent |

## Static-First Metadata Boundary

Installed CMake metadata continues to prove only:

- the static imported target;
- installed-prefix include metadata;
- installed-prefix static archive metadata;
- exact package-version compatibility metadata;
- absence of source/build path leaks;
- absence of shared imported target and runtime-loader metadata.

Installed `sparse.pc` metadata continues to prove only:

- package name;
- static archive package description;
- package version;
- installed include flags;
- installed static archive link flags;
- no private dependency stanza for the current self-contained link surface;
- no unsupported shared-library, loader, package-manager, or dynamic ABI
  wording.

## Skipped And Deferred Platform Notes

Local Day 10 validation ran on macOS with Unix shell, CMake, and pkg-config
tools. It did not locally execute the hosted Windows lane.

Deferred/non-claim boundaries remain:

- Windows `.lib` CMake install/downstream metadata is covered by hosted Windows
  CI, not this local run;
- Windows `sparse.pc` validation remains metadata-only inspection in hosted CI;
- Windows Makefile install/uninstall parity remains unsupported;
- Windows `pkg-config` command execution parity remains unsupported;
- shared-library packaging remains unsupported;
- runtime-loader behavior remains unsupported;
- dynamic ABI compatibility remains unsupported;
- package-manager distribution remains unsupported.

## Validation Commands

Static package deferral proof:

```text
bash scripts/static_package_deferral_check.sh
```

Result: passed.

Make install and pkg-config package proof:

```text
bash tests/test_install.sh
```

Result:

```text
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

CMake install/export package proof:

```text
bash tests/test_cmake_install.sh
```

Result:

```text
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Completion Check

- Installed CMake metadata matches the static-first package contract.
- Installed `sparse.pc` metadata matches the static-first package contract.
- Unsupported shared-library, runtime-loader, package-manager, and dynamic ABI
  metadata terms are absent from installed package files.
- Local and deferred platform validation gaps are explicit.
