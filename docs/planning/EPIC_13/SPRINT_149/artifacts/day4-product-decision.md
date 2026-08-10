# Sprint 149 Day 4: Product Decision

## Purpose

Apply the Day 3 promotion criteria to the current Windows supplemental
CMake install/downstream lane and decide the Sprint 149 implementation path
before changing workflow files or public support wording.

## Decision

Sprint 149 will pursue **conditional promotion** of the Windows package lane:

> Promote `.github/workflows/windows-ci.yml::install-and-downstream` from
> supplemental confidence to a reviewed Windows CMake install/downstream
> validation lane only after the missing CMake package metadata checks are
> added and hosted Windows evidence passes.

The promoted claim remains narrow:

> Windows has reviewed CMake install/downstream validation for the maintained
> static-first package surface on hosted MSVC 2022.

This decision intentionally does not promote Windows Makefile install,
Windows `pkg-config` execution, package-manager support, shared-library
support, dynamic ABI support, runtime-loader behavior, or broad Windows parity.

## Criteria Application

| Criterion | Current Status | Decision |
| --- | --- | --- |
| WIV-01 through WIV-03: CMake configure/build/install | Present | Keep and make reviewed after promotion. |
| WIV-04: installed static `.lib` | Present | Keep and make reviewed after promotion. |
| WIV-05: no installed DLLs | Present | Keep and make reviewed after promotion. |
| WIV-06: installed header count | Present as fixed `19` | Keep fixed count for Sprint 149; future header additions must update the contract intentionally. |
| WIV-07 through WIV-08: CMake package files and `sparse.pc` metadata file | Present | Keep and make reviewed after promotion. |
| WIV-09: positive `STATIC IMPORTED` check | Missing | Required before promotion. |
| WIV-10: installed include-prefix metadata check | Missing | Required before promotion. |
| WIV-11: installed `.lib` imported-location metadata check | Missing | Required before promotion. |
| WIV-12: source/build path leak rejection | Missing | Required before promotion. |
| WIV-13: no shared/module imported metadata | Present | Keep and make reviewed after promotion. |
| WIV-14 through WIV-15: `sparse.pc` static metadata and unsupported wording checks | Present | Keep as metadata checks only; not `pkg-config` execution parity. |
| WIV-16: maintained installed CMake example | Present | Keep and make reviewed after promotion. |
| WIV-17: exact-version generated CMake consumer | Present | Keep and make reviewed after promotion. |
| WIV-18: mismatch-version rejection | Present | Keep and make reviewed after promotion. |
| WIV-19: static package deferral guard | Missing on Windows | Retain under Linux/macOS reviewed package-contract ownership; Windows promotion remains CMake install/downstream scoped and must preserve static-first text/shared-artifact checks. |
| WIV-20: non-claim wording | Present | Must update docs/workflow to the new narrow reviewed wording without widening claims. |
| WIV-21: hosted Windows pass | Pending | Required before closeout can treat promotion as earned. |

## Rejected Alternatives

| Alternative | Rejection Reason |
| --- | --- |
| Immediate promotion with current workflow | Rejected because WIV-09 through WIV-12 are missing. |
| Split reviewed consumer proof from supplemental metadata proof | Rejected for Sprint 149 because package metadata checks are central to CMake install/downstream validation; splitting would leave ambiguous support wording. |
| Retain supplemental-only status | Rejected as the primary path because current evidence is close enough to promotion after bounded workflow assertions are added. |
| Explicitly reject Windows install-validation promotion | Rejected because no product blocker was found beyond missing metadata assertions and hosted evidence. |
| Promote broad Windows package parity | Rejected because Windows Makefile and `pkg-config` execution remain unsupported and unproven. |

## Implementation Target List

Days 5-10 should implement the conditional promotion path by adding or
preparing:

1. workflow comment and job-name changes that say reviewed Windows CMake
   install/downstream validation, not broad package parity;
2. positive `STATIC IMPORTED` check in installed CMake package metadata;
3. installed include-prefix check for exported target metadata;
4. installed static `.lib` imported-location check in generated targets files;
5. source/build path leak rejection across installed CMake package files;
6. retained DLL/shared imported metadata rejection;
7. retained `sparse.pc` static metadata and unsupported-wording checks;
8. retained maintained example, exact-version, and mismatch-version installed
   CMake consumer proof;
9. documentation and report-row updates that preserve all non-claims;
10. hosted Windows evidence requirement before closeout.

## WIV-19 Resolution

`scripts/static_package_deferral_check.sh` remains a Linux/macOS reviewed
package-contract guard for Sprint 149. It is intentionally not made part of the
Windows reviewed claim because it is a Unix shell guard over source and support
wording, not a Windows CMake install/downstream execution path.

Windows promotion must still prove the static-first boundary through:

- installed static `.lib` presence;
- absence of DLLs;
- absence of shared/module imported metadata;
- positive `STATIC IMPORTED` target metadata;
- `sparse.pc` static archive description;
- unsupported wording rejection in `sparse.pc`;
- retained public non-claims for shared-library and dynamic ABI support.

## Rollback Rules

| Trigger | Rollback |
| --- | --- |
| Hosted Windows install/downstream job fails after workflow edits | Revert reviewed wording to supplemental or fix the failing criterion before closeout. |
| WIV-09 through WIV-12 cannot be implemented robustly in PowerShell | Retain supplemental-only status and document exact blockers. |
| Docs imply Windows Makefile or `pkg-config` execution parity | Revert docs wording before merge. |
| Workflow comments imply shared-library, dynamic ABI, package-manager, or broad Windows parity | Revert workflow wording before merge. |
| CMake reviewed CTest lane is weakened while editing package lane | Stop and fix CTest lane separately; package promotion cannot compensate for CTest regression. |
| Header-count drift appears | Update the checked public header count only if the install rules and docs intentionally changed. |

## Docs And Report Update Map

| File / Surface | Required Update |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Rename/comment install job to reviewed Windows CMake install/downstream validation after criteria-backed checks land. |
| `README.md` | Update Cross-Platform CI Contract wording to mention reviewed Windows CMake install/downstream validation while preserving non-claims. |
| `INSTALL.md` | Update supported-platform and verification sections to describe the reviewed Windows CMake install/downstream lane. |
| `docs/maintainer_guide.md` | Update platform/package ownership guidance from supplemental Windows confidence to reviewed CMake install/downstream validation. |
| `tests/corpus/manifests/report_families.tsv` | Check whether package or CI report-family rows need wording updates for the promoted Windows lane. |
| Sprint 149 artifacts | Keep decision, implementation, validation, and closeout rows aligned with the narrow claim. |

## Remaining Unsupported Windows Package Surfaces

- Windows Makefile install.
- Windows Makefile uninstall.
- Windows execution of `tests/test_install.sh`.
- Windows `pkg-config --exists`, `--cflags`, `--libs`, `--static`,
  `--modversion`, or variable parity.
- Windows `pkg-config` downstream compile/link/run.
- Package-manager installation or resolver behavior.
- Shared-library packaging.
- Dynamic ABI compatibility.
- Runtime-loader behavior.
- Broad Windows parity beyond hosted MSVC CMake lanes.

## Day 5 Handoff

Day 5 should design the workflow edits for conditional promotion:

1. decide exact job and step names;
2. define where WIV-09 through WIV-12 assertions live in PowerShell;
3. preserve current CMake configure/build/install and consumer checks;
4. preserve WIV-19 as Linux/macOS reviewed package-contract ownership in
   wording;
5. define local review commands for workflow syntax and unsupported-claim
   searches.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| The selected outcome is traceable to Day 3 criteria. | Complete | Criteria application table maps each WIV row to the selected conditional promotion path. |
| No CI or documentation change claims more than the selected outcome supports. | Complete | Decision text, WIV-19 resolution, and docs/report map preserve narrow CMake install/downstream wording. |
| Remaining unsupported Windows package surfaces are named. | Complete | Unsupported Windows package surfaces section lists each non-claim. |
