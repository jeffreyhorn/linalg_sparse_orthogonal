# Day 3 ABI Support Options

## Purpose

Day 3 compares the current static-first package contract with the proof that
would be required to add shared-library or dynamic-ABI support. This artifact
does not make the Day 4 decision; it gives Day 4 enough evidence to decide
without overstating Sprint 110 public-header stability or existing package
metadata.

## Current ABI-Adjacent Surface Inventory

| Surface | Current state | ABI relevance |
|---|---|---|
| public headers | `include/*.h` install under `include/sparse` through Make and CMake. | Source and compile contract for consumers; not binary ABI proof by itself. |
| generated version header | `sparse_version.h` is generated from `VERSION` and installed with public headers. | Package/source version metadata; not ABI compatibility policy. |
| index width | `idx_t` is selected by `SPARSE_IDX_BITS` at compile time. | Consumer and library must agree on compile-time width; this is not a stable binary ABI surface. |
| scalar type | `sparse_scalar_t` is currently `double`. | Public source contract; future widening would be ABI-sensitive. |
| library artifact | `libsparse_lu_ortho.a` static archive. | Static link product, not shared runtime ABI. |
| CMake target | `Sparse::sparse_lu_ortho` installed target. | Package target for downstream consumers; currently points at static artifact. |
| pkg-config metadata | `-lsparse_lu_ortho -lm` plus optional flags. | Static package metadata and link flags; not ABI guarantee. |
| CMake package version | `SparseConfigVersion.cmake` uses `COMPATIBILITY ExactVersion`. | Avoids overclaiming compatibility across versions. |
| shared artifact | none maintained. | No shared ABI surface exists today. |
| symbol policy | no export map, visibility policy, or ABI symbol baseline. | Required before claiming ABI stability. |
| runtime loader proof | none. | Required before claiming shared-library package support. |

## Static-First Support Evidence

The static-first path is already supported by live package surfaces:

| Evidence | Owner | What it proves | What it does not prove |
|---|---|---|---|
| Make static install | `Makefile`, `tests/test_install.sh` | Static archive, installed public headers, generated version header, `sparse.pc`, and uninstall behavior. | CMake package behavior or platform-wide reviewed parity. |
| pkg-config consumers | `sparse.pc.in`, `tests/test_install.sh` | Downstream compile/link/run through installed pkg-config flags. | Shared-library loader behavior or ABI compatibility. |
| CMake static install/export | `CMakeLists.txt`, `tests/test_cmake_install.sh` | Installed static target, config/version files, targets file, and generated version header. | Makefile parity on every platform. |
| CMake `find_package` consumer | `examples/cmake_example/`, `tests/test_cmake_install.sh` | Installed consumer can configure, build, link, and run against `Sparse::sparse_lu_ortho`. | Windows install-validation lane or shared artifact support. |
| exact-version package metadata | `CMakeLists.txt`, `VERSION`, `sparse.pc.in` | Package version is single-sourced and exact-version CMake matching is enforced. | Dynamic ABI compatibility between versions. |
| documentation non-claims | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public docs describe static-first behavior and fence unsupported ABI/shared claims. | Product-grade ABI support. |

## Required Proof for Shared-Library or Dynamic-ABI Claims

Any Day 4 decision to add shared-library or ABI support would require new
evidence in several areas:

| Proof area | Required evidence |
|---|---|
| shared artifact generation | Build rules that intentionally produce `.so`, `.dylib`, `.dll`, and platform import libraries where supported. |
| install/export metadata | Make, CMake, pkg-config, and CMake package metadata that identify shared artifacts accurately. |
| runtime loader behavior | Linux RPATH/loader behavior, macOS install-name behavior, Windows DLL/import-library behavior, and downstream runtime execution. |
| symbol export policy | Exported symbol baseline, visibility rules, private symbol hiding, and platform-specific export handling. |
| ABI version policy | SONAME/SOVERSION or equivalent policy, compatibility rules, break criteria, and exact scope of any stable ABI promise. |
| downstream consumers | Compile/link/run consumers that use the shared artifacts, not only static archive consumers. |
| platform validation | Reviewed Linux, macOS, and Windows shared-library install and runtime lanes, or explicit platform exclusions. |
| optional feature flags | Thread/OpenMP package metadata and runtime behavior for shared artifacts. |
| uninstall and cleanup | Removal of shared artifacts, import libraries, symlinks, package metadata, and runtime files. |
| documentation | README, INSTALL, CMake/pkg-config docs, maintainer guide, and CI comments updated with earned and non-earned claims. |

## Sprint 110 Evidence-Use Boundaries

Sprint 110 reported no public/install-header drift. That evidence is useful,
but narrow:

| Possible use | Allowed? | Reason |
|---|---|---|
| Baseline for package/source compatibility discussion | Yes | It shows Sprint 110 source movement did not change public or installed headers. |
| Evidence that static package consumers should not need source changes because of Sprint 110 | Yes, bounded | It supports no public-header drift from that sprint only. |
| Evidence of dynamic ABI stability | No | Header stability does not prove symbol layout, binary compatibility, calling convention, or linker/runtime behavior. |
| Evidence of shared-library support | No | Sprint 110 did not create or validate shared artifacts. |
| Evidence of expanded Windows coverage | No | Sprint 110 did not change the Windows reviewed subset or add install-validation lanes. |

## Option Comparison

| Option | Benefits | Cost / risk | Validation burden | Public-doc impact |
|---|---|---|---|---|
| Preserve static-first support | Matches current implementation, install scripts, docs, and CI boundaries. Keeps Sprint 112 focused on validating and clarifying already maintained package truth. | Leaves shared-library and dynamic ABI support as explicit non-claims. | Make install, CMake install/export, pkg-config, downstream consumers, platform-tier docs, and final hygiene. | Mostly confirm or tighten existing wording. |
| Add shared-library package support | Could broaden package ergonomics for downstream users that prefer dynamic linking. | Requires new build rules, metadata, platform loader handling, symbol policy, CI expansion, and docs. High risk of under-proving platform behavior. | Shared artifact generation/install/runtime checks across Linux/macOS/Windows or explicit exclusions; symbol and version policy proof. | Significant README, INSTALL, maintainer, CMake, pkg-config, and CI wording changes. |
| Claim ABI stability without shared-library proof | None. | Unsound: no symbol/version/runtime evidence exists. | Not satisfiable from current evidence. | Would create an unsupported public claim. |

## Day 4 Decision Inputs

Day 4 should choose between:

1. Preserve static-first as the explicit Epic 10 package support tier, then use
   Days 5-8 to refresh Make/CMake/pkg-config/downstream proof and Days 9-12
   to align platform/docs wording.
2. Attempt a new shared-library support tier only if Sprint 112 is willing to
   add build-system, package metadata, runtime-loader, symbol-policy, CI, and
   documentation work before claiming it.

The current evidence strongly favors option 1. Option 2 is possible only as a
larger implementation and validation project. Option 3 is not acceptable.

## Completion Criteria Status

- Sprint 110 public-header stability is explicitly bounded and not overstated
  as ABI stability.
- Shared-library and dynamic-ABI proof requirements are explicit.
- Day 4 has enough evidence to make a support-tier decision.
