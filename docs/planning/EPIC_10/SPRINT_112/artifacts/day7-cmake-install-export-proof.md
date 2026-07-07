# Day 7 CMake Install, Export, and pkg-config Proof

## Purpose

Day 7 runs the selected static-first CMake install/export proof from the Day 5
design. The proof validates the local CMake package path, installed CMake
consumer, exact-version package metadata, and staged pkg-config version output
without broadening claims to dynamic ABI compatibility, shared-library runtime
behavior, or Windows install-validation parity.

## Command

```sh
bash tests/test_cmake_install.sh
```

## Environment

| field | value |
|---|---|
| repository root | `/Users/jeff/experiments/linalg_sparse_orthogonal` |
| staged prefix | temporary `mktemp` directory under `/var/folders/.../sparse.*` |
| version under test | `2.2.0` |
| package tier | static-first |
| cleanup rule | script `trap` removes temp directory at exit |

## Results Summary

| metric | result |
|---|---:|
| checks passed | 16 |
| checks failed | 0 |
| checks skipped | 0 |
| final status | `ALL CMAKE INSTALL TESTS PASSED` |

## Configure, Build, and Install Proof

| phase | result |
|---|---|
| CMake configure | Passed |
| CMake build | Passed |
| CMake install | Passed |

## Installed Artifact Proof

| artifact / behavior | result |
|---|---|
| static library installed | Passed: `libsparse_lu_ortho.a` was present in staged `lib`. |
| no shared-library artifacts installed | Passed: no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts were found. |
| public headers installed | Passed: `19` headers were installed under `include/sparse`. |
| `SparseConfig.cmake` installed | Passed |
| `SparseConfigVersion.cmake` installed | Passed |
| `SparseTargets.cmake` installed | Passed |
| `sparse.pc` installed | Passed |

## Downstream CMake Consumer Proof

| check | result |
|---|---|
| `examples/cmake_example/` configure | Passed: `find_package(Sparse)` resolved from the staged prefix. |
| `examples/cmake_example/` build | Passed |
| `examples/cmake_example/` run | Passed |

## Version and Metadata Proof

| check | result |
|---|---|
| exact installed version | Passed: `find_package(Sparse 2.2.0 EXACT REQUIRED)` succeeded. |
| lower mismatched version | Passed: mismatched lower version request was rejected. |
| pkg-config version | Passed: `pkg-config --modversion sparse` returned `2.2.0`. |

## Supported Claim

Day 7 supports this bounded claim:

> The local CMake install/export path installs the maintained static package
> surface and supports an installed downstream `find_package(Sparse)` consumer
> with exact-version package metadata.

## Non-Claims

Day 7 does not claim:

- dynamic ABI compatibility across versions;
- shared-library package support;
- shared-library runtime-loader behavior;
- Makefile parity on every platform;
- Windows separate install-validation support;
- macOS reviewed install/export parity.

## Follow-Up

- Day 8 should compare Day 6 and Day 7 consumer coverage and decide whether
  any additional installed public-header consumer proof is needed before docs
  alignment.
- Day 9 should use the refreshed Make/CMake package evidence when defining
  platform tiers.

## Completion Criteria Status

- CMake and pkg-config package truth is validated for the selected static-first
  tier.
- Downstream CMake consumer proof does not depend on source-tree-only package
  paths.
- CMake support wording can be updated from refreshed evidence if Day 12 finds
  wording drift.
