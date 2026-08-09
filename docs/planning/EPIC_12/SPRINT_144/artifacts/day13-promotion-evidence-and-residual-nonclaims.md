# Sprint 144 Day 13 Promotion Evidence And Residual Non-Claims

## Purpose

Finalize the selected-lane status, evidence index, residual blocker ledger,
support-tier consistency check, and Sprint 145 adoption handoff before Sprint
144 closeout.

## Final Selected-Lane Decision

Sprint 144 promotes **macOS reviewed static-first install/export proof**.

The promoted lane is scoped to hosted `macos-latest` execution of:

- `bash tests/test_install.sh`;
- `bash tests/test_cmake_install.sh`;
- `bash scripts/static_package_deferral_check.sh`.

This closes the selected Sprint 144 platform promotion lane for the maintained
static archive package contract. It does not promote shared-library packaging,
dynamic ABI compatibility, runtime-loader compatibility, package-manager
support, static/shared selectors, or broader macOS platform parity.

## Evidence Index

| Evidence area | Source |
| --- | --- |
| Lane selection | `docs/planning/EPIC_12/SPRINT_144/artifacts/day2-platform-lane-selection.md` |
| Before-change blocker baseline | `docs/planning/EPIC_12/SPRINT_144/artifacts/day3-blocker-reproduction-evidence-baseline.md` |
| Portability/support-tier design | `docs/planning/EPIC_12/SPRINT_144/artifacts/day4-portability-design.md` |
| Workflow promotion implementation | `.github/workflows/macos-ci.yml` and `docs/planning/EPIC_12/SPRINT_144/artifacts/day5-source-script-fix-batch.md` |
| CI promotion design | `docs/planning/EPIC_12/SPRINT_144/artifacts/day6-ci-promotion-design.md` |
| CI implementation review | `docs/planning/EPIC_12/SPRINT_144/artifacts/day7-ci-promotion-implementation.md` |
| Report integration | `tests/corpus/manifests/report_families.tsv` and `docs/planning/EPIC_12/SPRINT_144/artifacts/day8-package-report-integration.md` |
| Documentation alignment | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and `docs/planning/EPIC_12/SPRINT_144/artifacts/day9-documentation-support-tier-alignment.md` |
| Focused validation | `docs/planning/EPIC_12/SPRINT_144/artifacts/day10-selected-lane-validation.md` |
| Cross-platform non-regression | `docs/planning/EPIC_12/SPRINT_144/artifacts/day11-cross-platform-non-regression-review.md` |
| Formal quality gate | `docs/planning/EPIC_12/SPRINT_144/artifacts/day12-quality-gate-execution.md` |

## Support-Tier Consistency

| Lane | Current support-tier status |
| --- | --- |
| Linux | Strongest reviewed source of truth, including reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and reviewed static-first package contract. |
| macOS Apple Clang | Reviewed compile-quality, CMake parity, wall-check, and sanitizer path. |
| macOS static-first package proof | Reviewed Make install/`pkg-config` and CMake install/export proof for the maintained static archive package contract. |
| macOS Homebrew GCC | Supplemental second-compiler coverage. |
| Windows CMake subset | Reviewed MSVC CMake subset with explicit CTest count ownership. |
| Windows CMake install/downstream | Supplemental CMake-first confidence only. |
| Windows staged tests | `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain staged due source-level pthread/POSIX blockers. |

## Residual Blocker Ledger

| Residual | Owner | Current disposition |
| --- | --- | --- |
| Hosted macOS proof | PR CI | Local checks passed; hosted `macos-latest` execution remains final external proof. |
| Windows Makefile parity | Future Windows/platform owner | Non-claim; not selected in Sprint 144. |
| Windows `pkg-config` parity | Future Windows/package owner | Non-claim; not selected in Sprint 144. |
| Windows reviewed install-validation parity | Future Windows/package owner | Non-claim; Windows install/downstream remains supplemental CMake-first confidence. |
| Windows `test_threads` and `test_sprint4_integration` | Future portability owner | Staged due pthread APIs. |
| Windows `test_fuzz` | Future portability owner | Staged due POSIX temp-file APIs. |
| Shared-library packaging | Future package/ABI owner | Deferred by Sprint 143 static-first contract. |
| Dynamic ABI compatibility | Future ABI owner | Deferred; no ABI compatibility policy is claimed. |
| Runtime-loader compatibility | Future package/platform owner | Deferred; no loader proof was added. |
| Package-manager distribution | Future release/package owner | Deferred; no Homebrew/apt/dnf/pacman/vcpkg/conan support is claimed. |
| Static/shared selectors | Future package owner | Deferred; package metadata remains static-first without selectors. |
| Portable performance parity | Future performance owner | Non-claim; platform promotion does not prove performance portability. |

## Residual Non-Claims

Sprint 144 still does not claim:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager availability;
- static/shared package selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- Windows staged test closure;
- broader macOS platform parity beyond reviewed static-first install/export
  proof;
- portable performance parity;
- state-of-the-art sparse linear algebra status from platform support work.

## Validation Summary

| Check | Result |
| --- | --- |
| Day 12 formal quality gate | Passed |
| package report normalization/freshness | Passed |
| CI report normalization/freshness | Passed |
| static package deferral guard | Passed |
| local Make install/`pkg-config` proof | Passed: 23 passed, 0 failed |
| local CMake install/export proof | Passed: 26 passed, 0 failed, 0 skipped |
| workflow YAML parse | Passed |
| support-tier and stale wording scans | Passed |
| unsupported-claim scan | Passed; matches are explicit non-claims |
| `git diff --check` | Passed |

## Sprint 145 Adoption Handoff Draft

Sprint 145 can present adoption guidance with this platform contract:

- Linux remains the strongest reviewed source-of-truth baseline for broad local
  quality and package-contract confidence.
- macOS now has reviewed static-first install/export proof for the maintained
  static archive package contract, including Make install/`pkg-config`, CMake
  install/export, downstream consumers, and static deferral checks.
- Windows remains the maintained CMake-first consumer path with reviewed CMake
  subset proof and supplemental CMake install/downstream confidence.
- Adoption docs should prefer static-first install wording and avoid
  package-manager, shared-library, ABI, loader, Windows Makefile, Windows
  `pkg-config`, and broad platform parity claims.

## Day 14 Handoff

Day 14 should produce the final closeout validation summary, confirm all Sprint
144 deliverables are present, and prepare retrospective inputs. It should not
expand the platform scope unless hosted CI reveals a blocker.

## Day 13 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected-lane status is backed by concrete evidence. | Complete | Evidence index ties the macOS promotion to workflow, report, docs, and Day 10-12 validation artifacts. |
| Residual platform non-claims are explicit and source-owned. | Complete | Residual blocker ledger names owners and preserves all non-selected lane boundaries. |
| Sprint 145 handoff identifies adoption-facing platform constraints. | Complete | Handoff draft names Linux, macOS, Windows, and unsupported adoption boundaries. |
