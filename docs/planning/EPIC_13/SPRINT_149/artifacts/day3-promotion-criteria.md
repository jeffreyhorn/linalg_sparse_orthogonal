# Sprint 149 Day 3: Promotion Criteria

## Purpose

Define exact criteria for deciding whether the Windows install/downstream lane
can be promoted to a reviewed Windows install-validation support tier, should
remain supplemental, should be split, or should be explicitly rejected for this
sprint.

The criteria intentionally target a narrow Windows CMake install/downstream
claim. They do not define Windows Makefile parity, Windows `pkg-config`
execution parity, package-manager support, shared-library ABI support, dynamic
ABI support, or runtime-loader support.

## Candidate Outcomes

| Outcome | Meaning | Required Wording |
| --- | --- | --- |
| Promote reviewed Windows CMake install validation | The Windows hosted MSVC lane becomes a reviewed CMake install/downstream proof for the static-first installed package. | "reviewed Windows CMake install/downstream validation" or equivalent narrow wording. |
| Split reviewed and supplemental Windows package lanes | A reviewed subset covers CMake install/export and installed CMake consumers, while metadata-only or broader confidence remains supplemental. | Name reviewed and supplemental pieces separately. |
| Retain supplemental-only status | Current or strengthened Windows lane remains useful confidence but does not become reviewed install validation. | Keep "supplemental CMake install/downstream confidence" wording. |
| Explicitly reject promotion | Sprint 149 decides Windows install-validation promotion is not product-ready. | State blockers and retain Windows reviewed support as CMake configure/build/CTest only. |

## Promotable Claim Shape

If promotion is selected, the maximum supported claim is:

> Windows has reviewed CMake install/downstream validation for the maintained
> static-first package surface on hosted MSVC 2022.

That claim means:

- CMake configure/build/install works through the Visual Studio generator;
- the installed static `.lib`, headers, CMake package files, and `sparse.pc`
  metadata are present;
- installed CMake package metadata points at installed static artifacts and
  does not leak source/build paths;
- shared-library artifacts and shared imported metadata are absent;
- maintained and generated installed CMake consumers configure/build/run;
- exact-version `find_package(Sparse ... EXACT REQUIRED)` behavior works;
- lower same-major mismatch-version behavior fails closed.

That claim does not mean:

- Windows Makefile install or uninstall parity;
- Windows `pkg-config` execution parity;
- package-manager distribution;
- shared-library packaging;
- runtime-loader behavior;
- dynamic ABI support;
- broad Windows parity beyond the hosted MSVC CMake lanes.

## Must-Pass Evidence Checklist

| Criterion ID | Evidence | Required For Promotion | Current Day 2 Status | Owner |
| --- | --- | --- | --- | --- |
| WIV-01 | Hosted Windows CMake install configure succeeds with Visual Studio 17 2022 x64. | Yes | Present | Workflow |
| WIV-02 | Hosted Windows CMake install build succeeds in Release. | Yes | Present | Workflow |
| WIV-03 | Hosted Windows `cmake --install` succeeds into a temp prefix. | Yes | Present | Workflow |
| WIV-04 | Installed static library `lib/sparse_lu_ortho.lib` exists. | Yes | Present | Workflow |
| WIV-05 | Installed shared-library artifacts are absent, including DLLs. | Yes | Present | Workflow |
| WIV-06 | Installed header count matches the public header contract. | Yes | Present as fixed `19` count | Workflow |
| WIV-07 | `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `SparseTargets.cmake` are installed. | Yes | Present | Workflow |
| WIV-08 | Installed `sparse.pc` exists and is treated as metadata only. | Yes | Present | Workflow |
| WIV-09 | Installed CMake target is positively identified as `STATIC IMPORTED`. | Yes | Missing direct positive check | Workflow |
| WIV-10 | Installed CMake target include directories use the install prefix. | Yes | Missing | Workflow |
| WIV-11 | Installed CMake target imported location points at the installed `.lib`. | Yes | Missing | Workflow |
| WIV-12 | Installed CMake package files do not contain source-tree or build-tree paths. | Yes | Missing | Workflow |
| WIV-13 | Installed CMake package metadata contains no shared/module imported target or `.so`/`.dylib`/`.dll` imported location. | Yes | Present | Workflow |
| WIV-14 | `sparse.pc` describes static archive package metadata. | Yes | Present | Workflow |
| WIV-15 | `sparse.pc` contains no `Libs.private`, shared-library, ABI, or package-manager wording. | Yes | Present | Workflow |
| WIV-16 | Maintained installed CMake example configures, builds, runs, and emits expected output. | Yes | Present | Workflow |
| WIV-17 | Exact-version generated installed CMake consumer configures, builds, runs, and emits expected output. | Yes | Present | Workflow |
| WIV-18 | Lower same-major mismatch-version generated consumer fails configure. | Yes | Present | Workflow |
| WIV-19 | Static package deferral guard is either run on Windows or explicitly retained under Linux/macOS reviewed package-contract ownership with narrower Windows wording. | Conditional | Missing on Windows | Day 4 decision |
| WIV-20 | Workflow comments, docs, and report rows preserve Windows Makefile, Windows `pkg-config`, package-manager, shared-library, runtime-loader, dynamic ABI, and broad parity non-claims. | Yes | Present in current docs; must stay present after edits | Docs/CI |
| WIV-21 | Hosted Windows job passes after criteria-backed workflow edits. | Yes | Pending PR/CI | Hosted CI |

## Explicit Non-Goals

| Non-Goal | Reason |
| --- | --- |
| Windows Makefile install parity | The maintained Windows route is CMake-first; no Windows Makefile install evidence exists. |
| Windows Makefile uninstall cleanup proof | No CMake uninstall target is established, and Make uninstall is Unix-side proof. |
| Windows `pkg-config` execution parity | Current Windows lane checks `sparse.pc` as metadata only and does not execute `pkg-config`. |
| Windows `pkg-config` downstream compile/link/run | This belongs to Unix-side Make/`pkg-config` validation unless a separate Windows evidence lane is added. |
| Package-manager support | Installed CMake package files are not package-manager distribution evidence. |
| Shared-library packaging | The project is static-first and rejects shared-library support without a separate product decision. |
| Dynamic ABI compatibility | Static package metadata and CMake consumers do not establish ABI stability. |
| Runtime-loader behavior | No DLL/shared-library runtime loading surface is supported. |
| Broad Windows parity | Reviewed Windows proof remains scoped to hosted MSVC CMake lanes. |

## Failure Semantics

| Failure | Required Interpretation |
| --- | --- |
| CMake configure/build/install fails | Reviewed Windows CMake install validation cannot be claimed. |
| Static `.lib` missing | Installed static package shape is broken. |
| Header count mismatch | Installed public header contract drifted; fix source/install rules or update the checked contract intentionally. |
| Required CMake package file missing | Installed CMake package export is incomplete. |
| `sparse.pc` missing | Static metadata install is incomplete, even though Windows `pkg-config` execution remains out of scope. |
| DLL or shared imported metadata appears | Static-first/no-shared-artifact contract is broken; promotion is blocked. |
| Installed target is not positively `STATIC IMPORTED` | CMake package metadata does not prove the static package contract. |
| Installed target include path does not use install prefix | Installed package may not be relocatable or may point at source/build paths. |
| Installed target archive location does not point at installed `.lib` | Downstream consumers may link the wrong artifact. |
| Source/build path leak appears | Installed package metadata is not clean for downstream consumers. |
| Unsupported wording appears in `sparse.pc` | Package metadata widened claims beyond the supported static-first contract. |
| Maintained example fails configure/build/run | Installed CMake downstream path is broken. |
| Exact-version consumer fails configure/build/run | Installed package version compatibility proof is broken. |
| Mismatch-version consumer unexpectedly configures | Version fail-closed behavior is broken. |
| Hosted CI not available | Keep claim pending; do not mark reviewed Windows install validation as complete. |

## Workflow-To-Criterion Mapping

| Workflow Step / Artifact | Criteria Covered |
| --- | --- |
| `Run maintained supplemental CMake install/downstream proof`: configure block | WIV-01 |
| `Run maintained supplemental CMake install/downstream proof`: build block | WIV-02 |
| `Run maintained supplemental CMake install/downstream proof`: install block | WIV-03 |
| `$staticLib` assertion | WIV-04 |
| `$dlls` assertion | WIV-05 |
| `$headers.Count` assertion | WIV-06 |
| package-file loop | WIV-07, WIV-08 |
| CMake package metadata text checks | WIV-09, WIV-10, WIV-11, WIV-12, WIV-13 after Day 8 strengthening |
| `$pcText` checks | WIV-14, WIV-15 |
| installed example configure/build/run | WIV-16 |
| exact-version generated project configure/build/run | WIV-17 |
| mismatch-version generated project configure failure | WIV-18 |
| `scripts/static_package_deferral_check.sh` or explicit Day 4 decision artifact | WIV-19 |
| README, INSTALL, maintainer guide, workflow comments, report rows | WIV-20 |
| PR hosted Windows workflow result | WIV-21 |

## Support-Tier Wording Templates

### Reviewed Promotion Template

Use only if all required criteria pass:

> Windows carries reviewed MSVC CMake configure/build/CTest proof and reviewed
> CMake install/downstream validation for the maintained static-first package
> surface. This does not claim Windows Makefile parity, Windows `pkg-config`
> execution parity, package-manager support, shared-library support, dynamic
> ABI support, runtime-loader behavior, or broad Windows parity.

### Split-Lane Template

Use if only part of the Windows lane is promoted:

> Windows carries reviewed MSVC CMake configure/build/CTest proof and reviewed
> CMake installed-consumer validation for the static package export. Additional
> package metadata checks remain supplemental. This does not claim Windows
> Makefile or `pkg-config` parity.

### Supplemental-Only Template

Use if promotion is not selected:

> Windows carries reviewed MSVC CMake configure/build/CTest proof plus
> supplemental CMake install/downstream confidence. The supplemental lane does
> not establish a separate reviewed Windows install-validation tier and does
> not claim Windows Makefile or `pkg-config` parity.

### Explicit Rejection Template

Use if promotion is rejected:

> Windows install-validation promotion is explicitly deferred because the
> current evidence does not satisfy the reviewed Windows CMake package criteria.
> Windows reviewed support remains the hosted MSVC CMake configure/build/CTest
> subset.

## Day 4 Decision Input

Day 4 should apply this checklist to the current lane and decide whether Sprint
149 will:

1. promote after adding the missing CMake metadata checks;
2. split reviewed installed-consumer validation from supplemental package
   metadata checks;
3. retain the entire lane as supplemental; or
4. explicitly reject promotion and document blockers.

The default engineering reading after Day 3 is that promotion is plausible only
after closing WIV-09 through WIV-12 and resolving WIV-19.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Promotion cannot occur without a named evidence row for every required check. | Complete | Must-pass evidence checklist defines WIV-01 through WIV-21. |
| Rejected or supplemental-only outcomes have concrete wording requirements. | Complete | Support-tier wording templates cover promotion, split, supplemental-only, and rejection outcomes. |
| Support-tier language is ready for implementation without ambiguity. | Complete | Promotable claim shape and explicit non-goals define the allowed wording. |
