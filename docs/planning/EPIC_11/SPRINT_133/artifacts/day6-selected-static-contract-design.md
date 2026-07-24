# Sprint 133 Day 6 - Selected Static Contract Design

## Purpose

Day 6 translates the Day 5 product decision into an implementation-ready
static-first package contract design. The selected contract does not add
shared-library or dynamic ABI support. It strengthens deferral and validation
around the maintained static archive install/export surface.

This is a design-only artifact. It does not change CMake behavior, install
scripts, package metadata, public headers, tests, workflows, or support docs.

## Selected Contract

Sprint 133 implements the static-first path:

- the maintained install artifact remains `libsparse_lu_ortho.a` or the
  platform static equivalent;
- CMake and Make install paths continue to validate absence of shared-library
  artifacts;
- CMake package consumers continue to link `Sparse::sparse_lu_ortho`;
- `pkg-config` consumers continue to link `-lsparse_lu_ortho`;
- generated version metadata remains package/source metadata, not ABI
  compatibility policy;
- shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, package-manager support, and broader platform install parity remain
  explicit non-claims.

## Design Decision

| Question | Decision | Rationale |
| --- | --- | --- |
| Should Sprint 133 implement shared-library support? | No. | Day 5 found missing ABI, symbol, version, loader, metadata, and platform proof. |
| Should `BUILD_SHARED_LIBS=ON` remain warning-only? | No. | Warning-only still permits a successful configure while ignoring a caller's shared-library request. A static-first deferral should fail clearly. |
| Should CMake package metadata advertise static/shared components? | No. | There is no supported shared component to select. Adding selectors would imply unsupported dual-mode packaging. |
| Should `sparse.pc` gain ABI or library-type fields? | No for Day 7. | Nonstandard fields would not give consumers real support. Day 12 can tighten observed output checks instead. |
| Should public headers gain export/import macros now? | No. | Export macros would imply shared-library preparation without a selected ABI contract. |
| Should package docs change before implementation? | No. | Existing docs already state static-first and shared deferral; Day 8 can sharpen wording if Day 7 changes require it. |

## Build and Install Requirements

| Requirement | Target behavior | Implementation owner |
| --- | --- | --- |
| Static CMake target | `sparse_lu_ortho` remains declared as `STATIC`. | `CMakeLists.txt` |
| Shared request deferral | `cmake -S . -B <dir> -DBUILD_SHARED_LIBS=ON` fails at configure time with explicit static-first/shared-deferral wording. | `CMakeLists.txt` Day 7 |
| Normal CMake configure | `cmake -S . -B <dir>` and install/export flows remain unchanged for default static builds. | `CMakeLists.txt`, `tests/test_cmake_install.sh` |
| Make install | `make install` continues to install static archive, headers, generated version header, and `sparse.pc`. | No Day 7 Makefile change planned |
| No shared artifacts | Install validation continues to fail if `.so`, `.so.*`, `.dylib`, or `.dll` artifacts appear. | Existing install tests |
| CMake package version | Exact-version package behavior remains unchanged. | `CMakeLists.txt`, `tests/test_cmake_install.sh` |
| Package-manager support | No package-manager files or claims are added. | Residual queue |

## Error Wording Design

The `BUILD_SHARED_LIBS=ON` failure should:

- name the rejected input: `BUILD_SHARED_LIBS=ON`;
- name the maintained contract: static archive package surface;
- name the deferred work: shared-library packaging and dynamic ABI support;
- point to the evidence requirement: future explicit build rules, package
  metadata, installed-consumer proof, and runtime-loader validation;
- avoid implying that shared support is partially available.

Candidate wording:

```text
BUILD_SHARED_LIBS=ON was requested, but sparse_lu_ortho currently maintains
only the static archive package surface. Shared-library packaging and dynamic
ABI support are deferred until explicit build rules, package metadata,
installed-consumer proof, and runtime-loader validation are added.
```

## File-Level Implementation Map

| File | Change type | Day | Validation |
| --- | --- | --- | --- |
| `CMakeLists.txt` | Replace warning-only `BUILD_SHARED_LIBS` handling with a configure-time static-first deferral failure. | Day 7 | Focused `cmake -S . -B <tmp> -DBUILD_SHARED_LIBS=ON` failure probe, `bash tests/test_cmake_install.sh`; full C quality not required unless C/header files change. |
| `tests/test_cmake_install.sh` | No Day 7 change planned; later tighten exact installed header count and optional installed-target path checks. | Day 11 or 13 | `bash tests/test_cmake_install.sh`. |
| `tests/test_install.sh` | No Day 7 change planned; later tighten exact `pkg-config` include/lib path checks. | Day 12 | `bash tests/test_install.sh`. |
| `sparse.pc.in` | No Day 7 change planned; keep standard static archive metadata. | Day 12 if touched | `bash tests/test_install.sh` and staged `pkg-config` output inspection. |
| `cmake/SparseConfig.cmake.in` | No Day 7 change planned; avoid unsupported component/static-shared selectors. | Day 11 if touched | `bash tests/test_cmake_install.sh`. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | No Day 7 change planned; Day 8 can update only if implementation wording needs clearer user guidance. | Day 8 | Docs hygiene and claim-boundary scan. |
| `include/*.h` | No Sprint 133 static-first implementation change planned. | Deferred | Full C quality if any header changes occur. |

## Deferral-Check Design

| Check | Current state | Selected design |
| --- | --- | --- |
| CMake shared request | `BUILD_SHARED_LIBS=ON` emits a status message and continues with static output. | Configure should fail with explicit deferral wording. |
| Install shared artifacts | Install scripts fail if shared artifacts are present. | Preserve existing no-shared-artifact checks. |
| Symbol export/ABI claim | No export macro, symbol map, soname, or ABI policy exists. | Preserve absence; Day 9 can add documentation/checks if needed. |
| Package metadata shared selector | No CMake component or `pkg-config` selector exists. | Preserve absence; do not invent unsupported selectors. |
| Version as ABI proof | Exact CMake package version and `sparse.pc` version exist. | Preserve exact package metadata; do not call it ABI compatibility. |

## Downstream Validation Plan

| Validation | Purpose | When |
| --- | --- | --- |
| Focused shared-request failure probe | Proves `BUILD_SHARED_LIBS=ON` fails with selected deferral wording. | Day 7 if `CMakeLists.txt` changes. |
| `bash tests/test_cmake_install.sh` | Proves default CMake install/export and installed CMake consumer remain static-first and functional. | Day 7 if `CMakeLists.txt` changes; Day 11/13 for CMake proof tightening. |
| `bash tests/test_install.sh` | Proves Make install/uninstall and `pkg-config` static consumer remain functional. | Day 12 if `tests/test_install.sh` or `sparse.pc.in` changes; Day 13 integrated validation. |
| `git diff --check` and Sprint 133 whitespace scan | Documentation hygiene. | Every documentation-only day and after docs updates. |
| `make format && make lint && make test` | Full C quality gate. | Only if `.c` or `.h` files change. |

## Rollback Plan

| Change | Rollback if it fails |
| --- | --- |
| `BUILD_SHARED_LIBS=ON` configure failure | Revert to the previous warning-only block only if the failure breaks an existing reviewed lane; keep Day 5 decision and record blocker. |
| CMake install proof tightening | Revert only the failing tightened assertion; preserve baseline install proof and record exact missing evidence. |
| pkg-config proof tightening | Revert only the failing assertion; preserve functional compile/link/run proof and record the metadata gap. |
| Documentation wording | Revert wording that implies unsupported shared ABI, package-manager, or platform parity support. |

## Blockers and Deferrals

| Blocker or deferred item | Disposition |
| --- | --- |
| Shared-library artifact support | Deferred beyond Sprint 133 unless a future product decision adds build rules and proof. |
| Dynamic ABI compatibility | Deferred; no ABI epoch, soname/install-name, layout policy, or compatibility test exists. |
| Symbol visibility/export policy | Deferred; no `SPARSE_API`, export map, or hidden-private policy is selected. |
| Static/shared package selection | Deferred; no CMake component or `pkg-config` selector is supported. |
| Package-manager recipes | Deferred; no manager-specific package recipes or consumer proof exist. |
| Cross-platform shared-loader behavior | Deferred; no Linux/macOS/Windows runtime-loader proof exists. |

## Day 7 Handoff

Day 7 should implement the narrow build/install contract batch:

1. Update `CMakeLists.txt` so `BUILD_SHARED_LIBS=ON` fails configure with the
   Day 6 static-first deferral wording.
2. Leave the static `add_library(... STATIC ...)` target and install/export
   layout otherwise unchanged.
3. Run a focused failing configure probe for `BUILD_SHARED_LIBS=ON`.
4. Run `bash tests/test_cmake_install.sh` to prove the normal static CMake
   install/export path still works.
5. Do not touch public headers, shared export macros, package-manager files,
   or static/shared selection metadata.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation work is sequenced and bounded. | Complete | File-level implementation map limits Day 7 to CMake shared-request deferral and leaves consumer proof tightening to Days 11-12. |
| Shared and static-first paths are not mixed after the decision. | Complete | Selected design rejects shared artifacts, export macros, ABI policy, and selectors while preserving static install/export behavior. |
| Validation expectations are known before code or script changes begin. | Complete | Downstream validation plan defines the focused configure probe, CMake install proof, pkg-config proof, docs hygiene, and full C trigger. |
