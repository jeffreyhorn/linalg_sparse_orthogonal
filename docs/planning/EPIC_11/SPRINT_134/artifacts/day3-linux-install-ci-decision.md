# Sprint 134 Day 3 - Linux Install CI Decision

## Purpose

Day 3 decides whether Linux install/package proof should remain local or move
into CI. The decision is intentionally narrower than the existing Linux
reviewed compile-quality, CMake parity, and dead-code lanes: it covers the
Sprint 133 static-first package contract only.

## Audited Evidence

| Evidence | Current state |
| --- | --- |
| Linux workflow | `.github/workflows/ci.yml` already carries reviewed Makefile compile-quality, reviewed CMake parity, reviewed dead-code, and supplemental runtime/benchmark/TSan/coverage jobs. |
| Local Make install proof | `bash tests/test_install.sh` passed 22 checks in Sprint 133. |
| Local CMake install/export proof | `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips in Sprint 133. |
| Static deferral proof | `bash scripts/static_package_deferral_check.sh` passed in Sprint 133. |
| Package contract | Static-first package support only; shared-library packaging, dynamic ABI compatibility, package-manager support, and runtime-loader behavior remain deferred. |
| Current workflow comments | Linux comments say focused install/package scripts remain developer-side proof surfaces rather than a separate reviewed CI lane. |

## Decision

Promote a bounded Linux package-contract lane in CI on Day 4.

The selected lane should run the local static-first package proof stack on
Ubuntu:

1. `bash tests/test_install.sh`
2. `bash tests/test_cmake_install.sh`
3. `bash scripts/static_package_deferral_check.sh`

This promotion makes Linux install/package proof reviewed for the selected
static-first package contract. It does not promote shared-library packaging,
dynamic ABI compatibility, package-manager support, runtime-loader behavior,
macOS install/export parity, Windows install validation, or Windows Makefile
parity.

## Decision Rationale

| Criterion | Assessment |
| --- | --- |
| Contract value | High. Sprint 133 added real package proof gates; running them in Linux CI keeps the maintained static-first contract from drifting. |
| Runtime cost | Acceptable for a separate bounded package job. The scripts run focused install/export consumers, not the full test suite. |
| Tool availability | Acceptable on Ubuntu. The existing CI image already uses Make, CMake, compiler tools, and package tooling; `pkg-config` is expected for package proof. |
| Failure specificity | Good. The scripts fail with package-specific messages around installed files, CMake metadata, `pkg-config`, downstream consumers, and static deferral wording. |
| Flake risk | Moderate but bounded. Temporary install prefixes and CMake builds are local to the job; no network dependency is introduced by the package proof stack. |
| Support clarity | Good if workflow comments and docs call this a reviewed Linux static-first package-contract lane, not broad platform/package parity. |

## Selected Implementation Plan

Day 4 should update `.github/workflows/ci.yml` by adding a separate Linux job
for static-first package contract validation.

Recommended job shape:

- job name: `Linux reviewed static-first package contract`
- runner: `ubuntu-latest`
- steps:
  - checkout
  - install required package tools if absent, including `pkg-config` and CMake
    if needed by the runner image
  - run `bash tests/test_install.sh`
  - run `bash tests/test_cmake_install.sh`
  - run `bash scripts/static_package_deferral_check.sh`

Keep the job separate from the existing reviewed compile-quality, reviewed
CMake parity, dead-code, and supplemental runtime lanes so failures are easy
to classify.

## Workflow Comment Update Queue

| Location | Required update |
| --- | --- |
| `.github/workflows/ci.yml` top comment | Replace the statement that focused install/package scripts remain developer-side only with wording that Linux now has a reviewed static-first package-contract lane once Day 4 implements it. |
| `README.md` CI summary | Update only if Day 4 implements the workflow lane; mention reviewed Linux static-first package contract without implying shared-library/package-manager support. |
| `INSTALL.md` platform interpretation | Update only if Day 4 implements the workflow lane; distinguish Linux reviewed package-contract lane from local Unix-side scripts and non-Linux tiers. |
| `docs/maintainer_guide.md` package/platform section | Update only if Day 4 implements the workflow lane; keep macOS and Windows narrower. |

## Validation Plan

If Day 4 implements the selected lane, run locally:

1. `bash -n scripts/static_package_deferral_check.sh`
2. `bash tests/test_install.sh`
3. `bash tests/test_cmake_install.sh`
4. `bash scripts/static_package_deferral_check.sh`
5. `git diff --check`
6. focused whitespace scan over touched workflow, docs, and Sprint 134 paths

If Day 4 edits only workflow/docs/planning files and no `.c` or `.h` files,
`make format && make lint && make test` is not required by the sprint rule.

## Risk Controls

| Risk | Control |
| --- | --- |
| Package proof is misread as shared-library support. | Keep static-first and non-claim wording in workflow comments and support docs. |
| Linux promotion is misread as macOS/Windows install parity. | Keep platform tier tables asymmetric and explicit. |
| Job duration grows unexpectedly. | Keep package proof in a separate job; Day 13 can decide whether runtime cost is acceptable after CI evidence. |
| Tool availability differs on runner image. | Install `pkg-config` explicitly in the package job if Day 4 finds the runner image does not guarantee it. |
| Existing reviewed baseline meaning is diluted. | Keep compile-quality/CMake/dead-code reviewed lanes named separately from the package-contract lane. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Linux install proof has an explicit support tier. | Complete | Decision promotes a bounded reviewed Linux static-first package-contract lane. |
| Reviewed CI promotion is scoped or deferred with evidence. | Complete | Selected implementation scope is limited to three package proof scripts and static-first support. |
| No local package proof is silently described as reviewed CI. | Complete | Day 4 must update workflow/docs wording when the lane is added; until then this artifact records the selected transition. |
