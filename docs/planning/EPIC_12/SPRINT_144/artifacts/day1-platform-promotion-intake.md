# Sprint 144 Day 1 Platform Promotion Intake

## Purpose

Establish Sprint 144 scope, inherited package/ABI evidence, candidate platform
lanes, initial promotion criteria, and stop conditions before selecting one
lane for complete closure.

## Inputs Reviewed

| Input | Relevant Sprint 144 signal |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` | Sprint 144 must close or explicitly reject one platform promotion lane. |
| `docs/planning/EPIC_12/SPRINT_143/PLAN.md` | Sprint 143 deliberately separated package/ABI work from platform promotion. |
| `docs/planning/EPIC_12/SPRINT_143/RETROSPECTIVE.md` | macOS and Windows install/downstream lanes remain supplemental pending Sprint 144 promotion. |
| `docs/planning/EPIC_12/SPRINT_143/artifacts/day14-closeout-validation-summary.md` | Static-first package behavior is implemented; remaining platform claims are non-claims. |
| `.github/workflows/ci.yml` | Linux is the reviewed static-first package-contract source of truth. |
| `.github/workflows/macos-ci.yml` | macOS has reviewed Apple Clang paths and supplemental static-first install/export jobs. |
| `.github/workflows/windows-ci.yml` | Windows has a reviewed MSVC CMake subset and supplemental CMake install/downstream proof. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Public and maintainer docs encode reviewed, supplemental, and staged support tiers. |

## Sprint 143 Handoff Summary

Sprint 143 selected and implemented the stricter static-first package contract:

- CMake rejects `BUILD_SHARED_LIBS=ON`;
- the installed package surface is a static archive package surface;
- Make install/`pkg-config` proof validates install, metadata, downstream
  compile/link/run, exact version, no shared artifacts, and uninstall cleanup;
- CMake install/export proof validates static imported metadata, downstream
  `find_package(Sparse)` consumers, exact-version behavior, mismatched-version
  rejection, no source/build path leaks, and no shared metadata;
- `scripts/static_package_deferral_check.sh` guards static-first deferrals;
- Linux CI runs the reviewed package-contract lane;
- macOS and Windows package jobs are supplemental confidence paths;
- shared-library packaging, dynamic ABI, runtime-loader proof, package-manager
  support, static/shared selectors, Windows Makefile parity, Windows
  `pkg-config` parity, Windows reviewed install-validation parity, and macOS
  reviewed install/export parity remain explicit non-claims.

## Candidate Lane Inventory

| Candidate lane | Current evidence | Primary blocker or uncertainty | Candidate closure shape |
| --- | --- | --- | --- |
| macOS reviewed install/export parity | Apple Clang reviewed path already runs `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, and `make sanitize`; separate macOS jobs run `tests/test_install.sh`, `tests/test_cmake_install.sh`, and static package deferral proof. | Current workflow comments mark package jobs as supplemental, not reviewed parity; promotion needs hosted-runner evidence, failure ownership, docs/report alignment, and exact static-first scope. | Promote macOS static-first Make install/`pkg-config` and CMake install/export jobs to reviewed install/export parity if evidence and support-tier wording can be made consistent. |
| Windows reviewed install/downstream parity | Windows reviewed subset configures, builds, lists 56 CTest tests, and runs CTest under MSVC; supplemental job installs the static library, validates CMake package metadata, checks no DLLs, builds/runs installed example and exact-version consumer, and rejects mismatched versions. | Current claim is CMake-first supplemental confidence, not separate reviewed install-validation; no Windows Makefile or `pkg-config` parity; local validation cannot fully reproduce MSVC hosted behavior. | Promote only the exact Windows CMake static install/downstream lane, while preserving Windows Makefile and `pkg-config` non-claims. |
| Windows staged test portability | Workflow output names staged exclusions: `test_threads`, `test_sprint4_integration`, and `test_fuzz`; docs identify pthread APIs and POSIX temp-file APIs as source-level blockers. | Closing this lane likely requires source/test portability changes and intentional CTest count changes; risk is broader than package promotion. | Select one staged test family and either port it to Windows or explicitly reject promotion with source-level blocker evidence. |
| Linux source-of-truth strengthening | Linux already owns reviewed Makefile compile-quality, reviewed CMake parity, dead-code, reviewed static-first package contract, and supplemental runtime/benchmark/TSan/coverage signals. | Lower platform-promotion value because Linux is already the strongest reviewed source of truth; remaining opportunities may be polish rather than lane closure. | Strengthen Linux package/report/freshness evidence only if macOS and Windows promotion lanes are not viable. |

## Initial Evidence Requirements

| Evidence area | Requirement |
| --- | --- |
| Scope | Select exactly one platform lane for complete closure. |
| CI | Workflow commands, expected counts, support-tier comments, and failure messages must match the selected lane. |
| Source/script portability | Any portability fix must have a direct blocker and focused validation. |
| Package contract | Static-first metadata and unsupported shared-artifact guards must remain intact. |
| Reports | Package/report/freshness rows must identify proof owners without implying stale run evidence. |
| Documentation | README, INSTALL, and maintainer guide must use the same reviewed/supplemental/staged interpretation. |
| Validation | Locally feasible checks must pass; hosted-only evidence must be explicitly identified. |

## Item-To-Day Owner Map

| Sprint 144 item | Project-plan estimate | Day-level owner |
| --- | ---: | --- |
| Item 1: Platform Lane Selection | 20 hours | Days 1-2 |
| Item 2: Source/Script Portability Fixes | 36 hours | Days 3-5 |
| Item 3: CI Promotion Implementation | 30 hours | Days 6-7 |
| Item 4: Package/Report Integration | 20 hours | Day 8 |
| Item 5: Documentation Alignment | 18 hours | Day 9 |
| Item 6: Validation | 24 hours | Days 10-12 |
| Item 7: Closeout | 20 hours | Days 13-14 |

## Initial Promotion Criteria

A lane is promotable only when all of the following are true:

1. The support-tier wording names the exact lane being promoted.
2. Existing static-first package semantics remain unchanged unless a direct
   selected-lane requirement justifies a change.
3. The lane has a repeatable validation path in CI or an explicit hosted-only
   proof requirement.
4. Expected test counts or staged exclusions have a named owner.
5. Public documentation and maintainer guidance agree with workflow comments.
6. The lane does not imply shared-library, dynamic ABI, runtime-loader,
   package-manager, Windows Makefile, Windows `pkg-config`, or broader
   platform parity claims unless those claims are directly tested.

## Initial Non-Claim Register

Sprint 144 does not start with claims for:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager distribution;
- static/shared package selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- portable performance parity;
- state-of-the-art sparse linear algebra status from platform work alone;
- macOS or Windows reviewed install/export parity before selected-lane proof
  is earned.

## Stop Conditions

Stop and ask for direction if:

- Day 2 scoring cannot identify a single lane that can close fully;
- selected-lane closure requires public C/header API changes outside Sprint 144
  scope;
- locally required checks fail after focused fixes;
- hosted CI is the only evidence source and workflow changes cannot express a
  clear failure owner;
- a proposed change would promote multiple lanes at once;
- a docs/report update would turn supplemental confidence into a reviewed claim
  without matching CI proof.

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 144 project-plan item has a day-level owner. | Complete | Item-to-day owner map above and `WORKING_NOTES.md`. |
| Sprint 143 package/ABI proof is treated as prerequisite evidence, not a new platform claim by itself. | Complete | Handoff summary preserves static-first package scope and routes promotion to Sprint 144 selection. |
| Platform promotion requires explicit proof, not CI wording alone. | Complete | Initial promotion criteria require repeatable validation, support-tier alignment, and claim-boundary checks. |

## Day 2 Handoff

Day 2 should score the four candidate lanes and select exactly one primary lane.
The likely highest-value candidates are macOS reviewed install/export parity and
Windows reviewed CMake static install/downstream parity. Windows staged test
portability is higher source risk, and Linux source-of-truth strengthening is
lower promotion value because Linux is already reviewed for the package
contract.
