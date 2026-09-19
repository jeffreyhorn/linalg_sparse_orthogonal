# Day 5: Public Claim Recalibration Batch One

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Update user-facing documentation so public claims match earned Epic
18 evidence.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 5 applies the public-documentation portion of the Day 3 claim audit. The
audit found the main public support docs were already claim-safe, so the Day 5
implementation is deliberately narrow: fix stale public package-manager
evidence wording without promoting broad package support.

## Changed Public Surfaces

| File | Change | Claim rationale |
| --- | --- | --- |
| `README.md` | The installation section now names the retained proof as the Sprint 198 Homebrew proof and states that it is not a user-facing Homebrew install path or a change to the supported install commands. | Makes the front-door package wording match the current Epic 18 selected proof while preserving source install via Make/CMake as the user path. |
| `INSTALL.md` | The package-manager distribution row now names Sprint 198 Homebrew proof material and the package-manager deferral guard as the evidence owner, replacing the stale Sprint 188 artifact reference. | Keeps `INSTALL.md#support-readiness-matrix` aligned with the merged Sprint 198 closure without claiming package-manager distribution. |

## Public Claim Classification After Edits

| Claim area | Day 5 public status | Evidence boundary |
| --- | --- | --- |
| Local source build and first solve | Supported public path. | README, examples, Makefile, and tests; not package-manager or performance proof. |
| Static source install via Make/CMake | Supported/validated public path. | `INSTALL.md`, `tests/test_install.sh`, `tests/test_cmake_install.sh`; static-first only. |
| Developer-mode local Homebrew proof | Selected proof material only. | Sprint 198 local static source formula proof; not a user-facing Homebrew install path. |
| Package-manager distribution | Not claimed. | No Homebrew/core, bottles, Linuxbrew, public tap, vcpkg, Conan, pkgsrc, distro/system package, provider registry, recipe, or binary package support. |
| Shared-library/dynamic ABI support | Deferred. | Static install remains the supported package shape. |
| Windows support | Bounded to documented MSVC CMake install/downstream validation and guarded/re-deferred selected workflow evidence. | No broad Windows parity, Windows Makefile parity, Windows `pkg-config` execution parity, Windows QR selected freshness, or Windows selected benchmark freshness. |
| Generated API HTML | Local-only. | `make api-docs-freshness` and `docs/api_reference.md`; no hosted, retained artifact, committed HTML, or release evidence. |
| Selected benchmark freshness | Bounded hosted selected evidence only. | Linux/macOS selected lane metadata; no portable performance, timing threshold, platform parity, release benchmark, or state-of-the-art claim. |
| Release/state-of-the-art | Not claimed. | No Day 5 edit changes release readiness or state-of-the-art status. |

## Public Surfaces Reviewed But Not Edited

| Surface | Day 5 reason no edit was needed |
| --- | --- |
| `docs/api_reference.md` | Already states generated API HTML is local-only ignored output and not hosted, retained, source-controlled, package, ABI, broad Windows, release, or state-of-the-art evidence. |
| `docs/cookbook.md` | Sprint 205 quick reference already routes problem shapes without promoting package, ABI, Windows, generated API, performance, or state-of-the-art claims. |
| `docs/solver_selection.md` | Already distinguishes selected fixture/local evidence from broad parity and support claims. |
| `docs/tutorial.md` | No Day 3 public-claim contradiction identified. |
| `examples/README.md` | Already routes support/readiness interpretation to `INSTALL.md` and does not widen package-manager, Windows, hosted API, release, or state-of-the-art claims. |
| `benchmarks/README.md` | Benchmark claim recalibration is maintainer/report adjacent and remains Day 6 scope; Day 3 found the public benchmark boundaries claim-safe. |

## Explicit Non-Promotions

Day 5 does not promote or claim:

- Homebrew/core readiness;
- Homebrew bottles;
- Linuxbrew support;
- public tap maintenance;
- vcpkg, Conan, pkgsrc, distro/system packages, provider registries, recipes,
  or binary packages;
- shared-library packaging or dynamic ABI compatibility;
- broad Windows parity or Windows selected freshness;
- hosted generated API publication, retained generated-doc artifacts, or
  committed generated HTML;
- portable performance, release readiness, or state-of-the-art status.

## Validation And Hygiene

| Check | Day 5 result |
| --- | --- |
| `git diff --check` | Passed after Day 5 edits. |
| C/header quality gate | Not required for Day 5; no `.c` or `.h` edits. |
| Generated-output status | No generated output intentionally created. |
| Guard scripts/workflows/manifests | Not edited on Day 5. |
| `make support-docs-guard` | Passed after README/INSTALL support wording changed. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Public documentation no longer contradicts Sprint 197-205 evidence. | Met for Day 5 public surfaces. The stale public Sprint 188 package evidence reference was replaced with Sprint 198 proof material. |
| Package, platform, benchmark, performance, API, ABI, and release boundaries are clear to users. | Met. README and INSTALL keep static source install as the user path and preserve package-manager, ABI, benchmark, generated API, Windows, release, and state-of-the-art non-claims. |
| Item 206.2 has user-facing implementation progress with recorded evidence. | Met. Public claim recalibration is recorded here and in `WORKING_NOTES.md`. |

## Day 5 Disposition

Day 5 is complete. Day 6 should handle maintainer, benchmark, corpus,
generated API, selected-report, and planning-adjacent claim-surface alignment.
