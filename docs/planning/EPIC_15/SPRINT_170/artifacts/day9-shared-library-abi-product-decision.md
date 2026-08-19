# Sprint 170 Day 9: Shared-Library ABI Product Decision Record

## Status

Accepted for Sprint 170.

## Decision

Sprint 170 keeps `linalg_sparse_orthogonal` on a **static-first-only package
and ABI product posture**.

The maintained package surface is the static archive install/export surface.
Shared-library packaging and dynamic ABI compatibility remain explicitly
unsupported and deferred.

## Context

The project now has real static package evidence:

- Make builds and installs `libsparse_lu_ortho.a`, public headers, generated
  `sparse_version.h`, and static archive `sparse.pc` metadata.
- CMake builds and installs an explicit `STATIC` target and exports
  `Sparse::sparse_lu_ortho` as a static imported target.
- Linux CI runs the reviewed static-first package-contract lane.
- macOS CI runs reviewed static-first Make install/`pkg-config` and CMake
  install/export lanes.
- Windows CI runs reviewed CMake install/downstream validation for the
  maintained static-first package surface and treats `sparse.pc` inspection as
  metadata-only.

The project does not have enough evidence or infrastructure for a
shared-library ABI claim:

- many public structs expose concrete layout;
- library-owned allocations cross public boundaries;
- callback signatures and cancellation semantics would become ABI;
- `SPARSE_IDX_BITS` changes public signatures and layouts;
- local archive symbol inspection found a large uncurated global symbol
  surface, including internal-looking helpers and test override hooks;
- there is no hidden-by-default visibility policy, export macro, symbol
  allowlist, dynamic ABI epoch, SONAME, install-name/RPATH, Windows
  DLL/import-library policy, installed shared consumer proof, or runtime-loader
  validation.

## Evidence Links

| Evidence | Artifact |
| --- | --- |
| Sprint intake and prior evidence | `day1-abi-intake.md` |
| Public header and layout inventory | `day2-header-abi-inventory.md` |
| Lifecycle and ownership audit | `day3-lifecycle-audit.md` |
| Symbol and visibility feasibility | `day4-symbol-visibility.md` |
| Makefile package feasibility | `day5-make-feasibility.md` |
| CMake package feasibility | `day6-cmake-feasibility.md` |
| Package/ABI claim-surface audit | `day7-claim-surface-audit.md` |
| Decision synthesis | `day8-decision-synthesis.md` |

## Supported Claims After This Decision

Current releases may claim:

- maintained static archive build support;
- maintained static archive install support;
- Unix-side Make install/uninstall plus `pkg-config` proof;
- Unix-side CMake install/export plus `find_package(Sparse)` proof;
- reviewed Linux static-first package-contract CI;
- reviewed macOS static-first Make install/`pkg-config` proof;
- reviewed macOS static-first CMake install/export proof;
- reviewed Windows CMake install/downstream validation for the static-first
  package surface;
- generated `sparse_version.h`, `sparse.pc`, and CMake package-version metadata
  as source/package version identity;
- exact CMake package-version compatibility for installed static consumers.

These claims are package/install claims for the selected static-first surface.
They are not dynamic ABI claims.

## Deferred And Unsupported Claims

Current releases must not claim:

- shared-library builds or installs;
- dynamic ABI compatibility;
- stable exported dynamic symbol lists;
- runtime-loader behavior;
- Linux SONAME support;
- macOS install-name/RPATH support;
- Windows DLL/import-library support;
- installed shared-library downstream consumers;
- static/shared package selectors;
- package-manager distribution;
- Windows Makefile install parity;
- Windows `pkg-config` command execution parity;
- broad platform parity;
- source-compatible public headers as binary ABI compatibility;
- package-version exactness as binary ABI compatibility;
- state-of-the-art status from package, install, or ABI evidence.

## Alternatives Considered

### Alternative A: Enable Shared Builds Now

Rejected.

Enabling `BUILD_SHARED_LIBS` or adding a shared Make/CMake target now would
make accidental ABI easy to infer from an uncurated symbol table and
layout-exposed headers. It would also create platform-specific loader and
allocation responsibilities without corresponding tests.

### Alternative B: Add An Experimental Shared Target But Keep It Unsupported

Rejected for Sprint 170.

An experimental shared target could be useful in a future investigation, but
it would weaken the current guard model unless it had a separate name,
separate install behavior, explicit non-install/default-off semantics, and
tests proving consumers cannot mistake it for supported packaging.

### Alternative C: Retain Static-First Support And Defer Shared ABI

Accepted.

This matches the evidence already maintained by Make, CMake, install tests,
CI lanes, package metadata, and public docs. It also leaves room for a future
shared-library product path with a proper ABI scope and validation stack.

## Consequences

Immediate consequences:

- Keep `BUILD_SHARED_LIBS=ON` as a configure-time rejection.
- Keep `sparse_lu_ortho` as an explicit CMake `STATIC` target.
- Keep CMake install/export metadata archive-only.
- Keep `Sparse::sparse_lu_ortho` as a static imported target.
- Keep `sparse.pc` static archive scoped.
- Keep exact CMake package-version compatibility as package-version behavior,
  not ABI compatibility.
- Keep install tests rejecting shared artifacts.
- Keep Windows `sparse.pc` checks metadata-only.
- Update documentation and guards to cite this decision after Day 10+ work.

Deferred consequences:

- Shared-library support needs a future product plan rather than opportunistic
  build-system toggles.
- Dynamic ABI compatibility must be earned by ABI scope selection, export
  control, loader policy, allocation policy, and platform proof.
- Public concrete structs can continue under source-compatibility management
  without being treated as frozen binary layouts.

## Follow-Up Gates For Any Future Shared-Library Product

A future shared-library decision must provide all selected gates before support
is claimed:

1. ABI scope: define which headers, functions, structs, enums, typedefs, and
   callbacks are dynamic ABI.
2. Layout policy: freeze, version, or hide concrete public structs.
3. Export policy: add an explicit export/import mechanism and
   hidden-by-default visibility.
4. Symbol allowlist: validate exported symbols against an approved public list
   on each supported platform.
5. Allocation policy: document and test library-owned versus caller-owned
   allocation across dynamic-library boundaries.
6. Version policy: define ABI epoch, compatibility rules, and break criteria.
7. Linux loader policy: define SONAME/SOVERSION and any version-script rules.
8. macOS loader policy: define install-name and RPATH behavior.
9. Windows loader policy: define DLL, import-library, and CRT expectations.
10. Package metadata policy: define static/shared target names, components,
    selectors, and `pkg-config` semantics.
11. Consumer proof: compile/link/run installed shared consumers on every
    claimed platform.
12. Documentation/guard proof: update public docs and guard scripts only after
    implementation and validation exist.

## Day 9 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Shared-library ABI product decision record | Complete | This file records static-first-only continuation and shared ABI deferral. |
| Supported claim list | Complete | Static archive package/install claims are listed. |
| Deferred/non-claim list | Complete | Shared-library, dynamic ABI, loader, package-manager, and broad platform non-claims are listed. |
| Evidence links | Complete | Links to Day 1 through Day 8 sprint evidence are listed. |
| Day 9 decision-record artifact | Complete | This file. |

## Validation

Day 9 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| The shared-library ABI product question has a documented answer. | Complete | Static-first-only continuation is accepted; shared-library ABI remains deferred. |
| Consequences and future gates are explicit. | Complete | Immediate consequences and future shared-library gates are listed. |
| Documentation updates can follow the decision without ambiguity. | Complete | Day 10+ can cite this record as the canonical Sprint 170 decision. |
