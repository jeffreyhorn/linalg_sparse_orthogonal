# Sprint 100 Day 11 Platform and Packaging Evidence Templates

## Purpose

Day 11 creates reusable templates for package proof, platform tiers, ABI
decisions, and installed-consumer validation. These templates are designed for
Sprint 108, where Epic 10 must decide whether to preserve the current
static-first support contract or earn wider package/platform claims with new
proof.

## Files

| file | role |
|---|---|
| `templates/package-proof-template.md` | reusable blank template for install/export and downstream consumer proof |
| `templates/platform-tier-template.md` | reusable blank template for Linux, macOS, Windows, and staged platform tiers |
| `templates/abi-decision-template.md` | reusable blank template for shared-library and ABI support decisions |
| `templates/consumer-validation-checklist.md` | reusable checklist for install/export, package metadata, and platform wording changes |

## Current Contract Encoded

| area | current truth |
|---|---|
| package shape | static-first |
| Make install proof | `tests/test_install.sh`: Make install/uninstall plus `pkg-config` consumer proof |
| CMake install proof | `tests/test_cmake_install.sh`: install/export plus `find_package(Sparse)` consumer proof |
| version source | repo `VERSION` file |
| CMake package version | exact-version only |
| Linux tier | broadest reviewed source of truth |
| macOS tier | reviewed Apple Clang path plus supplemental GCC and install/pkg-config confidence |
| Windows tier | reviewed CMake-first MSVC consumer subset with expected CTest count `51` |
| shared-library status | not claimed; unexpected shared artifacts are rejected in install proof |

## Template Design Requirements

Future platform and packaging artifacts should include:

- exact command and proof owner;
- platform and compiler/generator;
- installed artifact list;
- downstream consumer compile, link, and run behavior;
- package metadata and version behavior;
- reviewed versus supplemental status;
- expected test counts where enforced;
- staged exclusions;
- static/shared/ABI decision state;
- explicit non-claims.

## Required Separations

| evidence type | must stay separate because |
|---|---|
| Make install proof | `pkg-config` consumer proof does not prove CMake package behavior |
| CMake install proof | `find_package(Sparse)` proof does not prove Makefile parity on every platform |
| platform reviewed lane | reviewed commands are narrower than all supplemental CI confidence |
| package version behavior | exact-version package metadata is not a broad dynamic ABI guarantee |
| shared-library decision | static archive proof does not imply runtime loader or ABI support |
| Windows CMake subset | CMake-first consumer proof is not Windows Makefile or install parity |

## Existing Surface Patterns Used

| pattern | current owner | template effect |
|---|---|---|
| static-first install contract | `Makefile`, `CMakeLists.txt`, install tests | package template requires static/shared artifact expectations |
| pkg-config consumer proof | `tests/test_install.sh` | package template requires downstream compile/link/run fields |
| CMake package proof | `tests/test_cmake_install.sh` | package template requires config, version, target, and exact-version fields |
| reviewed platform tiers | `.github/workflows/*.yml`, `docs/maintainer_guide.md` | platform template requires reviewed and supplemental lanes separately |
| Windows expected count | `.github/workflows/windows-ci.yml` | platform template requires expected count and staged exclusions |
| ABI non-claim | `CMakeLists.txt`, `docs/maintainer_guide.md` | ABI template requires a decision before widening claims |

## Usage Notes

1. Use the package proof template whenever install/export outputs,
   `sparse.pc`, CMake package files, installed headers, or consumer examples
   change.
2. Use the platform tier template whenever CI workflow scope, expected CTest
   counts, platform support wording, or staged exclusions change.
3. Use the ABI decision template before changing static-first wording or
   introducing shared-library support.
4. Use the consumer validation checklist as the minimum package/platform audit
   before Sprint 108 marks a support claim earned.
5. Keep package claims and platform claims separate unless one artifact
   actually proves both.

## Completion Rule

A future platform or package claim is not earned unless the filled artifact
names:

- the proof command;
- the platform and compiler/generator;
- installed artifacts and explicitly absent artifacts;
- downstream consumer behavior;
- version metadata behavior;
- reviewed or supplemental status;
- staged exclusions;
- remaining non-claims.
