# Sprint 198 Day 1: Package Metadata Intake

## Purpose

Establish the Sprint 198 package metadata and Homebrew proof baseline before
license metadata, formula proof, guard promotion, or documentation promotion
changes begin.

## Scope Source

| Source | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Defines Sprint 198 items 198.1 through 198.6 and the 168-hour sprint budget. |
| `docs/planning/EPIC_18/SPRINT_198/PLAN.md` | Defines Day 1 tasks, deliverables, and completion criteria. |
| `docs/planning/EPIC_17/SPRINT_188/WORKING_NOTES.md` | Provides prior Homebrew proof state, blocker history, validation commands, and retained non-claims. |
| `docs/planning/EPIC_17/SPRINT_188/artifacts/day1-package-proof-intake.md` | Provides prior package owner inventory and proof-script baseline. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Identifies Homebrew/package-manager support blocker as priority residual `E18-RQ-001`. |

## Item-to-Artifact Traceability

| Item | Owner artifacts |
| --- | --- |
| 198.1 | Day 2 license metadata decision record; root `LICENSE`, `COPYING`, or `NOTICE`; Homebrew license identifier notes. |
| 198.2 | Root license metadata file; `packaging/homebrew/sparse-lu-ortho.rb.in`; proof-script metadata/render logic. |
| 198.3 | `scripts/homebrew_local_formula_proof.sh`; formula template; proof execution logs; Day 5 through Day 9 artifacts. |
| 198.4 | `scripts/package_manager_deferral_check.sh`; `scripts/static_package_deferral_check.sh`; focused guard validation logs. |
| 198.5 | `README.md`; `INSTALL.md`; `packaging/homebrew/README.md`; `docs/maintainer_guide.md`; claim-boundary notes. |
| 198.6 | Homebrew proof result; package/static guards; install checks; docs checks; full C gate if source/header files change. |

## Owner Surface Inventory

| Surface | Owner files | Day 1 baseline |
| --- | --- | --- |
| Standalone root license metadata | `LICENSE`, `COPYING`, `NOTICE` | Missing. No standalone root license metadata file exists. |
| Version metadata | `VERSION` | Present. |
| Formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` | Present. It is scoped as a temporary local formula and uses placeholder replacement for homepage, archive URL, checksum, version, and license metadata. |
| Proof script | `scripts/homebrew_local_formula_proof.sh` | Present and executable. It validates tools and placeholders, requires standalone root license metadata and `SPARSE_HOMEBREW_LICENSE`, creates a source archive, renders a temporary formula, installs, checks static package artifacts, runs `brew test`, uninstalls, and cleans up. |
| Package-manager guard | `scripts/package_manager_deferral_check.sh` | Present. It permits only the selected local Homebrew proof material and preserves package-manager non-claims. |
| Static package guard | `scripts/static_package_deferral_check.sh` | Present. It preserves static-first package support and shared-library/dynamic ABI deferrals. |
| Public docs | `README.md`, `INSTALL.md`, `packaging/homebrew/README.md` | Present. They keep package-manager distribution unclaimed and describe Homebrew proof material as blocker/provenance evidence. |
| Maintainer docs | `docs/maintainer_guide.md` | Present. They document package guard ownership, local proof script expectations, missing-license blocker behavior, and retained non-claims. |

## Local Tool Snapshot

| Tool | Path |
| --- | --- |
| `brew` | `/usr/local/bin/brew` |
| `cmake` | `/usr/local/bin/cmake` |
| `ruby` | `/usr/bin/ruby` |
| `tar` | `/usr/bin/tar` |
| `shasum` | `/usr/bin/shasum` |
| `cc` | `/usr/bin/cc` |

The current proof blocker is metadata, not missing local tooling.

## Baseline Command Results

| Command | Exit | Key result | Day 1 disposition |
| --- | ---: | --- | --- |
| `bash scripts/homebrew_local_formula_proof.sh` | 2 | Reports that formula rendering is blocked because no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists for provider metadata. | Expected unavailable blocker; local Homebrew proof remains unclaimed. |
| `bash scripts/package_manager_deferral_check.sh` | 0 | Deferral record, provider recipe absence, selected Homebrew boundary, package metadata neutrality, and public non-claims pass. | Package-manager guard baseline is clean. |
| `bash scripts/static_package_deferral_check.sh` | 0 | Static-first package contract, `BUILD_SHARED_LIBS=ON` rejection, static target metadata, install metadata, and shared ABI non-claims pass. | Static package guard baseline is clean. |

## Active Blocker

Sprint 198 starts with one active closure blocker:

- no standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists for provider
  metadata.

The proof script also requires `SPARSE_HOMEBREW_LICENSE` to be set to accurate
local-proof license metadata. Day 1 records this as pending Day 2 decision
work, not as an implementation choice.

## Day 2 Decision Inputs

Day 2 must resolve:

1. the approved standalone root metadata file path;
2. the exact license text or reference that belongs in that file;
3. the exact Homebrew formula license identifier;
4. whether guards should enforce a single expected identifier or reject only
   missing and placeholder metadata;
5. how docs should distinguish local formula proof from package-manager
   distribution support.

## Retained Non-Goals

Sprint 198 must not claim:

- Homebrew/core submission or readiness;
- bottle or hosted binary support;
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

Day 1 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required.
