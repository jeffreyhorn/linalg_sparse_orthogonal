# Day 6 Make Install Proof

## Purpose

Day 6 runs the selected static-first Make install proof from the Day 5 design.
The proof validates the local Unix-side Make install, `pkg-config`, downstream
consumer, and uninstall path without broadening claims to CMake install/export,
shared-library packaging, dynamic ABI stability, or platform-wide reviewed
install parity.

## Command

```sh
bash tests/test_install.sh
```

## Environment

| field | value |
|---|---|
| repository root | `$REPO_ROOT` |
| staged prefix | temporary `mktemp` directory under `/var/folders/.../sparse.*` |
| version under test | `2.2.0` |
| package tier | static-first |
| cleanup rule | script `trap` removes temp directory; `make uninstall` validates installed artifact removal before exit |

## Results Summary

| metric | result |
|---|---:|
| checks passed | 14 |
| checks failed | 0 |
| final status | `ALL INSTALL TESTS PASSED` |

## Installed Artifact Proof

| artifact / behavior | result |
|---|---|
| static library installed | Passed: `libsparse_lu_ortho.a` was present in staged `lib`. |
| no shared-library artifacts installed | Passed: no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts were found. |
| public headers installed | Passed: all `19` expected headers were installed, including generated `sparse_version.h`. |
| pkg-config metadata installed | Passed: staged `sparse.pc` was present. |

## pkg-config Proof

| check | result |
|---|---|
| `pkg-config --cflags sparse` | Passed: returned an include path. |
| `pkg-config --libs sparse` | Passed: returned the expected library flag. |
| `pkg-config --modversion sparse` | Passed: returned `2.2.0`. |

## Downstream Consumer Proof

| consumer | result |
|---|---|
| generated pkg-config consumer | Passed: compiled and linked against the staged install. |
| generated pkg-config consumer runtime | Passed: ran successfully and printed `OK`. |
| `examples/cmake_example/main.c` via pkg-config | Passed: compiled and linked against the staged install. |
| maintained example runtime | Passed: ran successfully and printed `OK`. |

## Uninstall Proof

| artifact | result |
|---|---|
| installed static library | Passed: removed after `make uninstall`. |
| installed headers | Passed: removed after `make uninstall`. |
| installed `sparse.pc` | Passed: removed after `make uninstall`. |

## Supported Claim

Day 6 supports this bounded claim:

> The local Unix-side Make install path installs and uninstalls the maintained
> static package surface, and the installed `pkg-config` metadata supports
> downstream compile/link/run consumers using installed public headers.

## Non-Claims

Day 6 does not claim:

- CMake install/export behavior;
- CMake `find_package(Sparse)` consumer behavior;
- shared-library package support;
- dynamic ABI stability;
- platform-wide reviewed install parity;
- Windows install-validation support;
- macOS reviewed install/export parity.

## Follow-Up

- Day 7 should run the CMake install/export proof:
  `bash tests/test_cmake_install.sh`.
- Day 8 should compare the Make and CMake downstream consumer coverage and
  decide whether any additional static-first public consumer proof is needed
  before documentation alignment.

## Completion Criteria Status

- Make install truth is validated for the local Unix-side static install path.
- Consumer proof uses installed public headers and staged package metadata.
- No support claim exceeds the validated Make install behavior.
