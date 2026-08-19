# Sprint 170 Day 8: Product Decision Synthesis

## Purpose

Synthesize the Sprint 170 ABI/header, lifecycle, symbol, Makefile, CMake, and
claim-surface audits into one selected product posture for shared-library ABI
support.

## Evidence Inputs

| Input | Evidence summary | Decision impact |
| --- | --- | --- |
| Day 2 header ABI inventory | Installed surface includes 18 checked-in public headers plus generated `sparse_version.h`; only `SparseMatrix` is fully opaque, while many factor/result/option structs expose layout. | Dynamic ABI would freeze a large public layout surface before it is designed as ABI-stable. |
| Day 3 lifecycle audit | Object lifecycles are documented, but many library-owned allocations are reachable through caller-allocated concrete structs; callback, cancellation, and width behavior would become ABI commitments. | Shared-library support needs an allocator/runtime and lifecycle compatibility policy, especially for Windows. |
| Day 4 symbol visibility audit | Local static archive has 47 objects and 359 global defined symbols, including many internal-looking helpers, workspace internals, backend functions, and test override hooks. | A naive shared build would leak accidental ABI; a symbol allowlist and hidden-by-default policy are prerequisites. |
| Day 5 Make feasibility | Make builds and installs only `libsparse_lu_ortho.a`, headers, generated version header, and static archive `sparse.pc`. No PIC, shared target, loader metadata, or export policy exists. | Make is coherent for static-first support but not ready for shared-library promotion. |
| Day 6 CMake feasibility | CMake rejects `BUILD_SHARED_LIBS=ON`, uses an explicit `STATIC` target, installs archive-only metadata, exports a static imported target, and uses exact package-version compatibility. | CMake is intentionally static-first and should not silently enable shared builds. |
| Day 7 claim-surface audit | README, INSTALL, maintainer guide, API docs, workflows, package metadata, tests, and report manifests consistently preserve shared-library/dynamic-ABI non-claims. | Current public wording is already aligned with static-first continuation. |

## Option Comparison

| Decision factor | Static-first-only continuation | Staged shared-library path now |
| --- | --- | --- |
| User value | Preserves a real, tested install/export path for static archive consumers on Unix and CMake consumers across reviewed lanes. | Could eventually help dynamic-link consumers, but current users would not get a trustworthy ABI without substantial new work. |
| Maintenance cost | Low to moderate; existing guards and package tests already match the selected posture. | High; requires API/ABI design, symbol curation, loader policy, platform CI, documentation, and compatibility process. |
| Test burden | Existing Make install, CMake install/export, static deferral, and platform package lanes remain sufficient with targeted guard additions. | Requires new shared build, shared install, downstream dynamic consumer, export allowlist, runtime-loader, and upgrade/compatibility tests. |
| Symbol governance | Current static archive can tolerate internal global helpers because no dynamic export boundary is promised. | Must hide or remove accidental exports from 359 global definitions and keep only approved public symbols visible. |
| Platform burden | Matches current Linux, macOS, and Windows evidence tiers. | Needs Linux SONAME/version-script decisions, macOS install-name/RPATH decisions, Windows DLL/import-library policy, and platform-specific runtime proof. |
| Packaging burden | Current `pkg-config` and CMake package metadata are static archive scoped and validated. | Needs static/shared selector or package naming policy, dependency propagation split, install collision policy, and uninstall behavior. |
| Claim risk | Low if docs keep saying static-first and shared/dynamic ABI are non-claims. | High if enabled before proof; users could infer binary compatibility, runtime-loader support, or platform parity from incomplete metadata. |
| Sprint 170 fit | Fits the sprint: decide, record, guard, align docs, validate. | Does not fit safely in the remaining sprint without partially closing several high-risk gaps. |

## Selected Product Posture

Sprint 170 selects **static-first-only continuation** for the maintained
package and ABI product posture.

Current releases may claim:

- maintained static archive build and install support;
- Unix-side Make install/uninstall plus `pkg-config` proof;
- Unix-side CMake install/export plus `find_package(Sparse)` proof;
- reviewed Linux static-first package-contract CI;
- reviewed macOS static-first Make install/`pkg-config` and CMake
  install/export package lanes;
- reviewed Windows CMake install/downstream validation for the maintained
  static-first package surface;
- generated version metadata as source/package version identity;
- exact CMake package-version compatibility for the installed static package.

Current releases must not claim:

- shared-library support;
- dynamic ABI compatibility;
- stable exported dynamic symbol list;
- runtime-loader behavior;
- Linux SONAME support;
- macOS install-name/RPATH support;
- Windows DLL/import-library support;
- package-manager distribution;
- static/shared package selectors;
- Windows Makefile parity;
- Windows `pkg-config` command execution parity;
- broad platform parity;
- state-of-the-art status from package, install, or ABI evidence.

## Rationale

The evidence supports the current static archive product and does not support a
dynamic ABI claim. The main blockers are not superficial:

- many public structs expose layout and owned pointer fields;
- allocator ownership crosses public boundaries;
- `SPARSE_IDX_BITS` changes signatures and layouts;
- callback signatures and cancellation semantics would need a platform-aware
  ABI policy;
- the static archive exposes many global internal helpers and test hooks;
- there is no hidden-by-default visibility policy or symbol allowlist;
- there is no ABI epoch, SONAME, install-name, DLL/import-library, or
  runtime-loader policy;
- package metadata intentionally rejects shared/static selectors and dynamic
  ABI wording.

Retaining static-first support is not a retreat from the package work already
done. It is the posture that matches the validated product.

## Acceptance Evidence For Selected Decision

The selected static-first-only posture is accepted when Sprint 170 records and
keeps the following evidence:

| Evidence | Required owner |
| --- | --- |
| Product decision record says shared-library and dynamic ABI support remain deferred. | Day 9 decision record |
| README, INSTALL, maintainer guide, and API reference continue to point package readers to static-first support and non-claims. | Day 10 documentation alignment |
| `BUILD_SHARED_LIBS=ON` remains a configure-time rejection with blocker wording. | CMake and static deferral guard |
| `sparse_lu_ortho` remains an explicit CMake `STATIC` target. | CMake and static deferral guard |
| CMake install metadata remains archive-only and exports `Sparse::sparse_lu_ortho` as `STATIC IMPORTED`. | `tests/test_cmake_install.sh` and platform lanes |
| `sparse.pc.in` continues to describe static archive package metadata and avoids `Libs.private`, shared/static selectors, ABI, loader, or package-manager wording. | `tests/test_install.sh`, `tests/test_cmake_install.sh`, and static deferral guard |
| Install tests continue to reject installed shared artifacts. | Make and CMake install tests |
| Windows keeps `sparse.pc` inspection metadata-only and avoids Makefile/pkg-config execution parity claims. | Windows CI and static deferral guard |
| Any future export macro, `SOVERSION`, install-name, RPATH, DLL/import-library, or component selector fails the guard until a support decision changes. | Static deferral guard |

## Deferred Shared-Library Path

Shared-library support should be treated as a future product path with its own
acceptance gate. That future path should include:

1. ABI scope selection: decide whether all public headers are ABI-stable or
   whether only an opaque subset is supported dynamically.
2. Public export policy: add `SPARSE_API`, a `.def` file, or platform export
   lists with hidden-by-default behavior.
3. Symbol allowlist: generate or maintain approved exported symbols and test
   actual platform exports against it.
4. Layout policy: freeze, opaque-ify, or version concrete public structs.
5. Allocator policy: define allocation/free behavior across shared-library and
   Windows CRT boundaries.
6. ABI version policy: define ABI epoch, break criteria, package/source version
   relationship, and compatibility guarantees.
7. Loader policy: add Linux SONAME, macOS install-name/RPATH, and Windows
   DLL/import-library install behavior.
8. Package selector policy: decide whether static/shared use separate packages,
   components, target names, or pkg-config files.
9. Downstream proof: add installed shared consumer compile/link/run tests on
   every supported platform.
10. Documentation and guard update: broaden claims only after all selected
    proof exists.

## Consequences

Immediate consequences:

- Day 9 should create a formal decision record for static-first-only
  continuation.
- Day 10 should align docs to point to that decision if necessary.
- Day 11 should update guards to require the decision record and keep
  unsupported shared-library metadata rejected.
- Days 12-14 should validate that the static-first package proof stack and
  non-claim wording remain coherent.

Deferred consequences:

- Dynamic ABI support remains a future investment, not a Sprint 170 feature.
- The public header layout surface can continue evolving under source
  compatibility expectations without promising binary compatibility.
- Static archive consumers keep the current maintained path without being
  exposed to incomplete shared-library metadata.

## Day 8 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Option comparison | Complete | Compared static-first continuation against staged shared-library work across user value, cost, tests, symbols, platforms, packaging, and claim risk. |
| Selected product posture | Complete | Selected static-first-only continuation. |
| Acceptance evidence list | Complete | Defined decision-record, docs, CMake, pkg-config, install-test, Windows, and guard evidence. |
| Deferred surface and non-claim list | Complete | Listed current non-claims and future shared-library prerequisites. |
| Day 8 decision-synthesis artifact | Complete | This file. |

## Validation

Day 8 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| One product posture is selected. | Complete | Static-first-only continuation is selected. |
| Decision rationale is concrete and evidence-based. | Complete | Rationale ties directly to Days 2-7 evidence. |
| Unsupported support claims remain rejected. | Complete | Shared-library, dynamic ABI, runtime-loader, package-manager, and broad platform claims remain explicit non-claims. |
