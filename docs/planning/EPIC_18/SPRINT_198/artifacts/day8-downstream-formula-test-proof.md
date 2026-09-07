# Sprint 198 Day 8: Downstream Formula Test Proof

## Purpose

Review the local Homebrew formula `test do` downstream consumer proof and
record the execution boundary while approved license metadata is still absent.

## Day 8 Disposition

`brew test` is not executed on Day 8 because the formula cannot be rendered or
installed without approved standalone root license metadata and an exact
Homebrew formula license identifier. This is a blocked execution outcome, not
a failed downstream consumer proof.

Day 8 therefore validates the downstream proof contract from source-controlled
template and proof-script checks.

## Downstream Consumer Contract

| Contract element | Evidence | Day 8 status |
| --- | --- | --- |
| Exact package lookup | `find_package(Sparse #{expected_version} EXACT REQUIRED)` | Present. |
| Imported target link | `Sparse::sparse_lu_ortho` | Present. |
| Installed public headers | `#include <sparse/sparse_matrix.h>` and `#include <sparse/sparse_types.h>` | Present. |
| Minimal runtime exercise | Creates a 3x3 sparse matrix, inserts one value, checks `sparse_nnz(A)`, prints version and `OK`. | Present. |
| Installed package prefix | `cmake ... -DCMAKE_PREFIX_PATH=#{prefix}` | Present. |
| Static archive check | `lib/"libsparse_lu_ortho.a"` | Present. |
| CMake config check | `lib/"cmake/Sparse/SparseConfig.cmake"` | Present. |
| pkg-config metadata check | `lib/"pkgconfig/sparse.pc"` | Present. |
| Shared artifact rejection | `.dylib`, `.so`, `.so.*`, and `.dll` scans | Present. |

## Proof-Script Guard Coverage

`scripts/homebrew_local_formula_proof.sh` validates that the template retains:

- exact-version `find_package(Sparse ...)`;
- `Sparse::sparse_lu_ortho` imported target linkage;
- installed public header includes;
- successful downstream executable output assertion;
- installed static archive, CMake package config, and pkg-config metadata
  checks;
- shared-library artifact rejection.

These checks run before metadata detection reaches archive/render/install/test
work, so template drift is caught even while the license metadata blocker
remains active.

## Source-Tree Leakage Review

The downstream test writes its own `CMakeLists.txt` and `main.c` under
Homebrew `testpath`. Its CMake invocation supplies only
`-DCMAKE_PREFIX_PATH=#{prefix}`. It does not add repository-root include
paths, build-tree include paths, source-tree library paths, or direct archive
paths.

This keeps the downstream test scoped to the installed package surface once a
rendered and installed formula exists.

## Claim Boundary

No downstream formula test proof is earned on Day 8 because there is no
rendered formula or installed package. The current evidence proves only that
the source-controlled `test do` contract remains present and guarded.

Homebrew/core readiness, bottles, Linuxbrew support, public taps, binary
packages, package-manager distribution, shared-library support, dynamic ABI
support, runtime-loader behavior, and broad package-manager support remain
unclaimed.

## Day 9 Handoff

Day 9 should attempt the full proof sequence only if approved root license
metadata and an exact Homebrew formula license identifier are available. If
they remain unavailable, Day 9 should record an end-to-end proof blocker
rather than manufacturing archive, render, install, or `brew test` evidence.

## Validation

Day 8 changes planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
