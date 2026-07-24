# Sprint 133 Day 9 - Static Deferral Proof Design

## Purpose

Day 9 designs the proof mechanism for the selected static-first contract. Since
Sprint 133 explicitly rejected shared-library and dynamic ABI support, the
proof should verify that shared support is not advertised, accidentally
enabled, or silently emitted.

This is a design-only artifact. It does not change scripts, CMake files,
package metadata, headers, docs, workflows, or source files.

## Selected Proof Type

| Product decision | Proof type |
| --- | --- |
| Static-first package contract remains selected. | Static-deferral proof. |
| Shared-library support remains deferred. | Negative proof for shared artifacts, shared selectors, export policy, and loader claims. |
| Dynamic ABI compatibility remains deferred. | Negative proof for ABI epoch, soname/install-name policy, and package-version-as-ABI wording. |

## Proof Goals

| Goal | Why it matters |
| --- | --- |
| Prove `BUILD_SHARED_LIBS=ON` fails clearly. | Prevents a caller from mistaking CMake acceptance for supported shared packaging. |
| Prove default CMake install/export remains static. | Keeps the selected package contract functional. |
| Prove Make and CMake installs do not emit shared artifacts. | Keeps shared-library support from appearing accidentally. |
| Prove no public export macro or symbol map is introduced. | Prevents partial shared-library signaling without ABI policy. |
| Prove no soname/install-name or ABI epoch policy appears. | Prevents package-version metadata from becoming an accidental ABI claim. |
| Prove CMake and `pkg-config` metadata do not expose static/shared selectors. | Prevents unsupported dual-mode packaging claims. |
| Prove docs keep shared-library, dynamic ABI, and package-manager support deferred. | Prevents support wording from outrunning evidence. |

## Proposed Day 10 Proof Surface

The Day 10 implementation should add a local package-deferral check script
rather than fold all checks into install scripts.

Candidate path:

```text
scripts/static_package_deferral_check.sh
```

Reasoning:

- the proof spans CMake behavior, package metadata, headers, and docs;
- it is about support-contract drift, not only install correctness;
- install scripts should continue to prove installed consumers;
- a separate script can be reused by Day 13 integrated validation and later
  maintainer workflows without overstating reviewed CI placement.

## Proposed Checks

| Check | Expected result | Failure message intent |
| --- | --- | --- |
| `cmake -S . -B <tmp> -DBUILD_SHARED_LIBS=ON` | Configure fails. | `BUILD_SHARED_LIBS=ON unexpectedly configured; shared-library support is still deferred.` |
| CMake shared-deferral wording | Error output mentions `BUILD_SHARED_LIBS=ON`, static archive package surface, shared-library packaging, and dynamic ABI deferral. | `BUILD_SHARED_LIBS deferral wording drifted.` |
| `CMakeLists.txt` target declaration | `add_library(sparse_lu_ortho STATIC` is present. | `sparse_lu_ortho is no longer declared as an explicit STATIC target.` |
| Shared artifact install probes | Existing install scripts continue to fail if `.so`, `.so.*`, `.dylib`, or `.dll` are emitted. | Leave to install scripts; the deferral script may cite them rather than duplicate install runs. |
| Export macro scan | No installed header declares `SPARSE_API`, `SPARSE_EXPORT`, `SPARSE_IMPORT`, or equivalent. | `Public export/import macro appeared without a shared ABI decision.` |
| CMake shared policy scan | No `SOVERSION`, `WINDOWS_EXPORT_ALL_SYMBOLS`, `C_VISIBILITY_PRESET`, or install-name/soname policy is present. | `Shared-library ABI metadata appeared without support decision.` |
| Package metadata selector scan | `cmake/SparseConfig.cmake.in` and `sparse.pc.in` do not expose static/shared components, `Libs.private`, or library-type selectors. | `Static/shared package selector appeared without support decision.` |
| ABI wording scan | Public package docs keep shared-library/dynamic ABI/package-manager support deferred or unclaimed. | `Package docs may imply unsupported shared ABI or package-manager support.` |
| Header ABI-break wording note | Existing `@warning **ABI break ...**` comments are treated as source/layout migration notes, not maintained dynamic ABI promises. | If checked, failure should require explicit maintainer review rather than automatic removal. |

## Expected Output

The script should print a compact pass/fail summary. Suggested successful
output:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: no shared export/ABI metadata found
static-package-deferral-check: package metadata has no static/shared selector
static-package-deferral-check: support wording remains deferred
static-package-deferral-check: passed
```

Suggested failure style:

```text
static-package-deferral-check: FAIL: <specific drift>
```

The script should exit non-zero on any drift.

## Support Tier and Placement

| Proof | Placement | Support tier |
| --- | --- | --- |
| Static-deferral script | Local maintainer/package proof in `scripts/`. | Local support-contract guard. |
| `tests/test_cmake_install.sh` | Local installed CMake consumer proof. | Local install/export proof. |
| `tests/test_install.sh` | Local Make install and `pkg-config` consumer proof. | Local install proof, with supplemental macOS usage. |
| CI integration | Not selected on Day 9. | Future owner decision only. |

Day 10 should not add reviewed CI placement unless the implementation remains
small and the maintainer can justify runtime, platform impact, and support
wording. The default Day 9 design is local-only.

## Implementation Touch Points

| File | Day 10 action |
| --- | --- |
| `scripts/static_package_deferral_check.sh` | Add the local static-deferral proof script. |
| `docs/maintainer_guide.md` | Document the new local proof only if the script is added. |
| `INSTALL.md` / `README.md` | No required Day 10 change unless user-facing validation guidance changes. |
| `Makefile` | Optional future target only; Day 9 does not require adding one. |
| `tests/test_cmake_install.sh` | No Day 10 change; remains install/export proof. |
| `tests/test_install.sh` | No Day 10 change; remains Make/pkg-config proof. |

## Validation Plan

If Day 10 adds the script:

| Validation | Required because |
| --- | --- |
| `bash -n scripts/static_package_deferral_check.sh` | Shell syntax check. |
| `bash scripts/static_package_deferral_check.sh` | Main static-deferral proof. |
| Focused negative fixture or temporary-copy probe if practical | Proves a selected failure path without mutating repo files. |
| `git diff --check` and whitespace scan | Shell/docs hygiene. |
| `bash tests/test_cmake_install.sh` | Required only if CMake/package behavior changes beyond the script. |
| `make format && make lint && make test` | Required only if `.c` or `.h` files change. |

## Residual Risks

| Risk | Mitigation |
| --- | --- |
| Negative text scans can be brittle. | Keep patterns narrow and tied to selected contract tokens, not broad natural language. |
| Existing header `ABI break` comments could be confused with dynamic ABI support. | Treat them as source/layout migration notes and document the distinction rather than failing automatically. |
| CMake output line wrapping can break naive wording checks. | Use token-level checks instead of one contiguous sentence. |
| A local script could be mistaken for reviewed CI support. | Label it local maintainer/package proof until a future artifact promotes it. |
| Package-manager claims can appear in prose without metadata changes. | Include docs wording scan for unsupported package-manager language. |

## Day 10 Handoff

Day 10 should implement the local static-deferral script with these priorities:

1. Verify `BUILD_SHARED_LIBS=ON` fails configure and emits token-level
   deferral wording.
2. Verify `sparse_lu_ortho` remains an explicit static CMake target.
3. Scan installed public headers for unsupported export/import macro names.
4. Scan CMake/package metadata for shared ABI metadata or selectors.
5. Scan package docs for deferred/non-claim wording around shared-library,
   dynamic ABI, and package-manager support.
6. Keep the proof local unless a later artifact explicitly promotes it.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Proof checks trace directly to the selected contract. | Complete | Proposed checks map to static target, shared request rejection, no export/ABI metadata, no selectors, and deferred support wording. |
| Failure output would prevent package-support drift. | Complete | Expected failure messages identify the exact drift category. |
| Local-only proofs are not mislabeled as reviewed CI support. | Complete | Placement table classifies the new proof as local maintainer/package proof. |
