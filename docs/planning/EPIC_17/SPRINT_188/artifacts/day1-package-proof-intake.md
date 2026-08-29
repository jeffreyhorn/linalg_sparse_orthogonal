# Sprint 188 Day 1: Package Proof Intake

## Purpose

Establish the Sprint 188 package proof baseline before license metadata,
formula proof, package guard, or documentation changes begin.

## Scope Source

| Source | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Defines Sprint 188 items 188.1 through 188.6 and the 168-hour sprint budget. |
| `docs/planning/EPIC_17/SPRINT_188/PLAN.md` | Defines Day 1 tasks, deliverables, and completion criteria. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day7-package-acceptance-gates.md` | Defines package proof gates, proof-script outcomes, package guards, claim promotion rules, and retained non-claims. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day12-quality-surface-map.md` | Defines changed-surface validation, including Homebrew proof, package/install, docs, and mandatory C/header gates. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day13-implementation-handoffs.md` | Defines the Sprint 188 implementation handoff from Sprint 187. |

## Owner Surface Inventory

| Surface | Owner files | Baseline state |
| --- | --- | --- |
| Standalone license metadata | `LICENSE`, `COPYING`, `NOTICE` | Missing. No root standalone license metadata file exists. |
| Version metadata | `VERSION` | Present. |
| Formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` | Present. The template is local-proof-only, uses required placeholders, builds with CMake, installs the static archive, and runs a downstream CMake `test do`. |
| Proof script | `scripts/homebrew_local_formula_proof.sh` | Present and executable. The script checks tools/placeholders, creates a temporary archive, requires standalone license metadata and `SPARSE_HOMEBREW_LICENSE`, renders a temporary formula, installs from source, checks static artifacts, runs `brew test`, uninstalls, and cleans temporary output. |
| Package-manager guard | `scripts/package_manager_deferral_check.sh` | Present and passing. Guards package-manager non-claims and the selected local Homebrew proof boundary. |
| Static package guard | `scripts/static_package_deferral_check.sh` | Present and passing. Guards static-first package support and shared-library/dynamic ABI deferrals. |
| User docs | `README.md`, `INSTALL.md` | Present. Current wording states Homebrew local proof artifacts exist but the proof remains blocked and unclaimed. |
| Package docs | `packaging/homebrew/README.md` | Present. Documents local proof-only scope, temporary outputs, license metadata requirement, and retained provider non-claims. |
| Maintainer docs | `docs/maintainer_guide.md` | Present. Documents package-manager deferral guard, Homebrew local proof script, missing license metadata blocker, and non-claims. |

## Local Tool Snapshot

| Tool | Path | Baseline interpretation |
| --- | --- | --- |
| `brew` | `/usr/local/bin/brew` | Available. |
| `cmake` | `/usr/local/bin/cmake` | Available. |
| `ruby` | `/usr/bin/ruby` | Available. |
| `tar` | `/usr/bin/tar` | Available. |
| `shasum` | `/usr/bin/shasum` | Available. |
| `cc` | `/usr/bin/cc` | Available. |

The Day 1 proof blocker is not local tool availability. Required local
prerequisites are present.

## Baseline Command Results

| Command | Exit | Key result | Day 1 disposition |
| --- | ---: | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | 2 | The script reports that formula rendering is blocked because no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists for provider metadata. | Expected unavailable blocker. Homebrew support remains unclaimed. |
| `scripts/package_manager_deferral_check.sh` | 0 | Deferral record, provider recipe absence, selected Homebrew boundary, metadata neutrality, and public non-claims pass. | Guard baseline is clean. |
| `scripts/static_package_deferral_check.sh` | 0 | Static-first package contract, `BUILD_SHARED_LIBS=ON` rejection, static target metadata, install metadata, and shared ABI non-claims pass. | Static package baseline is clean. |

## Active Blocker

Sprint 188 starts with one active proof blocker:

- no standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists for provider
  metadata.

The proof script also requires `SPARSE_HOMEBREW_LICENSE` to be set to accurate
local-proof license metadata, but Day 1 cannot select that value until the Day
2 license strategy decision is made.

## Day 2 License Strategy Checklist

Day 2 should resolve these questions before metadata implementation begins:

1. What license governs the repository, and what source already establishes
   that license?
2. Should Sprint 188 add `LICENSE`, `COPYING`, or `NOTICE` at the repository
   root?
3. What exact text should be placed in the selected standalone metadata file?
4. What exact Homebrew license identifier should be used for
   `SPARSE_HOMEBREW_LICENSE`?
5. Should the proof script enforce the expected identifier or only require a
   non-empty value?
6. How should docs explain the difference between license metadata required
   for local proof and broader package-manager/provider support?
7. What validation should run after metadata implementation if no source or
   public header files change?

## Retained Non-Goals

Sprint 188 must not claim:

- Homebrew/core submission or acceptance;
- bottles or hosted binary artifacts;
- Linuxbrew support;
- public tap maintenance;
- vcpkg, Conan, pkgsrc, apt, dnf, pacman, or distro packaging;
- provider registry readiness;
- binary package install/update/uninstall support;
- shared-library package support;
- dynamic ABI compatibility;
- static/shared package selector support;
- broad package-manager support.

## Day 1 Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
