# Sprint 133 Day 5 - Package and ABI Product Decision

## Scope

| Field | Value |
| --- | --- |
| Sprint/day | Sprint 133 Day 5 |
| Artifact owner | Package/ABI product decision |
| Decision surface | Static-first package contract versus shared-library/dynamic ABI support |
| Touched surfaces | Sprint 133 planning artifacts only |
| Explicitly out of scope | Build rules, public headers, install scripts, package metadata, workflows, and support docs are not changed by this decision artifact. |

## Decision Summary

Sprint 133 will **preserve the static-first package contract** and strengthen
static-first deferral/enforcement proof. It will **not** implement
shared-library or dynamic ABI support in this sprint.

The decision is evidence-based:

- current install/export behavior intentionally emits a static archive and
  rejects shared artifacts in install validation;
- current public headers expose many layout-sensitive structs, callbacks,
  typedefs, enums, and function signatures that would require an explicit ABI
  policy before shared-library support;
- current CMake and `pkg-config` metadata describe the static archive surface
  but do not encode library type, ABI identity, index width, scalar width,
  soname, symbol export policy, or static/shared selection;
- current downstream consumer proof is functional but static-first;
- current public and maintainer documentation already says shared-library
  packaging and dynamic ABI compatibility are deferred.

## Baseline

| Baseline item | Current value |
| --- | --- |
| Current package contract | Static-first install/export surface. |
| Current install proof | `tests/test_install.sh` for Make install/uninstall and `pkg-config`; `tests/test_cmake_install.sh` for CMake install/export and `find_package(Sparse)`. |
| Current installed library | `libsparse_lu_ortho.a` or platform static equivalent. |
| Current installed headers | 18 source headers plus generated `sparse_version.h` for a 19-header contract. |
| Current CMake package | `Sparse::sparse_lu_ortho`, `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, exact-version package behavior. |
| Current `pkg-config` package | `sparse.pc` with `-I${includedir}`, `-L${libdir}`, `-lsparse_lu_ortho`, `-lm`, and optional build flags. |
| Current non-claims | Shared-library support, dynamic ABI compatibility, package-manager support, runtime-loader behavior, Windows Makefile parity, Windows install-validation parity, and macOS full install/export parity. |

## Decision Criteria

| Criterion | Required for shared-library support | Current evidence | Decision impact |
| --- | --- | --- | --- |
| ABI stability | Public layout, enum, callback, typedef, and symbol compatibility policy. | Day 2 found many high-risk public layouts and no ABI policy. | Blocks shared support. |
| Symbol visibility | Export/import annotations, hidden-private policy, and symbol inventory. | No `SPARSE_API`, export map, or symbol visibility policy exists. | Blocks shared support. |
| Versioning | ABI epoch or soname/install-name policy separate from package version. | `VERSION`, `sparse_version.h`, CMake exact-version, and `sparse.pc` version exist, but are package/source metadata only. | Blocks shared support. |
| Install behavior | Intentional shared artifacts and platform install names. | Install tests intentionally fail if shared artifacts appear. | Blocks shared support. |
| Downstream selection | CMake and `pkg-config` can select or identify static versus shared linkage. | Target and link flag expose static archive identity; no selection contract exists. | Blocks shared support. |
| Loader/runtime proof | Installed shared consumer runs from staged prefix without build-tree leakage. | No loader/runtime proof exists. | Blocks shared support. |
| Dependency metadata | Static/shared public/private dependency rules. | Static dependencies are exposed through CMake target and `sparse.pc` `Libs`; no `Libs.private` or shared dependency policy. | Blocks shared support. |
| Platform ownership | Linux, macOS, and Windows shared-loader status reviewed, supplemental, or deferred. | Platform install truth is tiered and static-first. | Blocks shared support. |
| Validation cost | Proof can fit Sprint 133 without destabilizing package support. | Required shared proof spans headers, build system, metadata, tests, docs, workflows, and platform policy. | Too broad for this sprint. |
| Static-first value | Existing package contract can be made more robust. | Day 3-4 identified concrete static-first proof gaps. | Supports static-first strengthening. |

## Decision

| Decision question | Answer | Evidence |
| --- | --- | --- |
| Preserve static-first only for Sprint 133? | Yes. | Existing install/export behavior, docs, and validation all point to a maintained static archive contract. |
| Add shared-library support in Sprint 133? | No. | Shared support lacks ABI, symbol, version, artifact, selection, loader, dependency, and platform proof. |
| Define dynamic ABI compatibility policy now? | No public ABI guarantee. | Version metadata exists, but no ABI epoch or compatibility policy exists. |
| Strengthen static-first deferral proof? | Yes. | Day 3-4 gap queues identify precise enforcement and consumer-proof improvements. |
| Add package-manager support? | No. | No package-manager recipes or manager-specific consumer proof exist. |
| Change platform tier? | No. | Current platform truth remains Linux strongest reviewed, macOS narrower/supplemental for install, Windows CMake-first subset. |

## Selected Contract

Sprint 133 Days 6-13 should execute this selected contract:

- static archive install remains the maintained package shape;
- Make and CMake install paths should continue to prove absence of shared
  artifacts unless a future product decision reverses this contract;
- package metadata should not imply shared-library support, ABI stability, or
  package-manager availability;
- generated version metadata remains package/source metadata, not ABI
  compatibility policy;
- downstream CMake and `pkg-config` proofs should be strengthened for the
  static contract;
- any shared-library, dynamic ABI, package-manager, or broader platform
  install work remains a residual queue item with explicit proof gates.

## Static-First Contract

| Contract item | Expected state |
| --- | --- |
| Static archive install | `libsparse_lu_ortho.a` remains present in Make and CMake installs. |
| Installed headers | Source headers plus generated `sparse_version.h`; current contract is 19 headers. |
| `pkg-config` metadata | Describes installed static archive link route and version metadata without shared ABI claims. |
| CMake package metadata | Exports `Sparse::sparse_lu_ortho` for installed static package consumers and keeps exact-version package behavior. |
| Explicitly absent artifacts | `.so`, `.so.*`, `.dylib`, `.dll`, and shared import-library artifacts remain absent and validated as absent. |
| `BUILD_SHARED_LIBS` behavior | Should stay explicit static-first enforcement; Day 6 should decide whether warning-only remains sufficient or should become a clearer deferral check. |

## Shared-Library and ABI Contract

| Contract item | Decision |
| --- | --- |
| Shared library artifact | Deferred; no shared artifact should be installed in Sprint 133. |
| Symbol/version policy | Deferred; no public ABI or soname/install-name policy is claimed. |
| Loader/runtime proof | Deferred; no runtime-loader behavior is claimed. |
| ABI compatibility test | Deferred; version metadata remains exact package metadata only. |
| Public claim wording | Preserve and, if needed, sharpen wording that shared-library packaging and dynamic ABI compatibility are not maintained support claims today. |

## Implementation Touch Points

| Touch point | Selected-contract action |
| --- | --- |
| `CMakeLists.txt` | Day 6 should decide whether `BUILD_SHARED_LIBS=ON` warning-only is enough or whether a stronger static-first deferral check is warranted. |
| `tests/test_cmake_install.sh` | Candidate Day 11/13 tightening: exact 19-header count and stronger installed-prefix checks. |
| `tests/test_install.sh` | Candidate Day 12 tightening: exact `pkg-config` include/lib path and link-flag checks. |
| `sparse.pc.in` | Day 6/12 should decide whether static-first comments or metadata checks are needed without inventing unsupported fields. |
| `cmake/SparseConfig.cmake.in` | Day 6/11 should decide whether package config needs static-first explanatory metadata or validation only. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Day 8 should align wording only if implementation changes or deferral proof needs clearer explanation. |
| `include/*.h` | No Day 5 header changes. Shared ABI remains deferred; Day 9 may design checks that no export macro/ABI claim is accidentally present. |

## Rejected Alternative: Implement Shared Support Now

Sprint 133 rejects implementing shared-library support now because it would
require all of these to be designed and validated before the claim is honest:

- public export/import macro or symbol visibility policy;
- exported symbol inventory and hidden-private policy;
- ABI epoch, soname/install-name, or equivalent compatibility policy;
- public struct-layout and callback payload compatibility rules;
- `idx_t` width and `sparse_scalar_t` binary identity policy;
- CMake and `pkg-config` static/shared selection semantics;
- static versus shared dependency visibility rules;
- installed loader/runtime proof on at least one platform;
- explicit platform support tiers for shared runtime behavior;
- replacement of current no-shared-artifact checks with shared-aware proof.

That work is larger than a safe Sprint 133 implementation batch and would
force support wording ahead of evidence.

## Validation Plan

| Change class | Required validation |
| --- | --- |
| Decision artifact only | `git diff --check` and focused Sprint 133 markdown whitespace scan. |
| Static-first CMake enforcement changes | `bash tests/test_cmake_install.sh` plus focused configure behavior proof for `BUILD_SHARED_LIBS=ON`. |
| Make install or `pkg-config` proof changes | `bash tests/test_install.sh` and exact staged `pkg-config` output inspection. |
| CMake consumer proof changes | `bash tests/test_cmake_install.sh` and installed-prefix path-origin checks if added. |
| Public header or C source changes | Focused package proof plus `make format && make lint && make test`. |
| Documentation wording changes | Claim-boundary scan plus docs hygiene. |

## Drift Check

| Public/support surface | Impact | Action |
| --- | --- | --- |
| `README.md` | No immediate Day 5 change. | Day 8 may sharpen wording only if selected-contract implementation changes require it. |
| `INSTALL.md` | No immediate Day 5 change. | Current static-first/shared-deferral wording matches the decision. |
| `docs/maintainer_guide.md` | No immediate Day 5 change. | Current package/ABI non-claims match the decision. |
| Workflows | No Day 5 change. | Platform tiers remain unchanged. |

## Package-Manager Disposition

| Manager | Status | Proof | Public claim |
| --- | --- | --- | --- |
| Homebrew | Deferred | No formula or manager-specific consumer proof. | No package-manager support claim. |
| apt/deb | Deferred | No package recipe or manager-specific consumer proof. | No package-manager support claim. |
| rpm/dnf | Deferred | No package recipe or manager-specific consumer proof. | No package-manager support claim. |
| vcpkg | Deferred | No portfile or Windows install consumer proof. | No package-manager support claim. |
| Conan | Deferred | No recipe or package consumer proof. | No package-manager support claim. |

## Residual Handoff

| Residual | Next owner | Evidence link |
| --- | --- | --- |
| Stronger `BUILD_SHARED_LIBS=ON` deferral behavior | Day 6 design, Day 7 implementation if selected | Day 3 metadata gap queue and this decision. |
| Exact CMake installed header count | Day 11 or Day 13 | Day 3 and Day 4 gap queues. |
| Installed CMake target path-origin checks | Day 11 | Day 4 consumer gap queue. |
| Exact `pkg-config` include/lib path checks | Day 12 | Day 4 consumer gap queue. |
| Static/shared selection design for future support | Future epic or post-Sprint 133 residual | Day 4 shared-library proof requirements. |
| Symbol export and ABI version policy | Future epic or post-Sprint 133 residual | Day 2 symbol visibility risk and this decision. |
| Package-manager recipes and consumer proof | Future epic or post-Sprint 133 residual | Day 5 package-manager disposition. |

## Completion Check

| Criterion | Status |
| --- | --- |
| The shared-library versus static-first decision is explicit. | Complete: Sprint 133 preserves static-first support and defers shared-library/dynamic ABI support. |
| Support wording follows evidence rather than aspiration. | Complete: the decision follows current static install/export proof and records shared-support blockers. |
| Implementation days have a single selected contract to execute. | Complete: Days 6-13 should strengthen static-first deferral/enforcement and downstream consumer proof. |
