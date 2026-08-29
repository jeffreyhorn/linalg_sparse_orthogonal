# Sprint 188 Day 11: Homebrew and Maintainer Documentation

## Purpose

Update Homebrew-specific and maintainer documentation so maintainers can rerun
the proof, interpret pass/block/fail states, understand generated-output
cleanup, and know which validation commands are required when package surfaces
change.

## Changed Surfaces

| Surface | Change |
| --- | --- |
| `packaging/homebrew/README.md` | Added the local proof command, exit-code interpretation, support wording rules, required package guards, and follow-up validation guidance. |
| `docs/maintainer_guide.md` | Updated proof command wording to match the Day 9 blocker state: the proof exits before archive, render, install, or `brew test` work while root metadata is absent. |
| `docs/maintainer_guide.md` | Added explicit validation ownership for package-manager guard, static-package guard, install checks, CMake install checks, and package report normalization checks. |

## Proof Command

Maintainers should run the local proof from the repository root:

```sh
SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh
```

The value must match approved standalone root license metadata. Placeholder
values remain blocker evidence, not proof metadata.

## Exit-Code Interpretation

| Exit | Meaning | Maintainer action |
| ---: | --- | --- |
| `0` | Local static source formula proof passed through render, archive, checksum, install, installed-surface validation, `brew test`, uninstall, and cleanup. | Package wording may mention only the exact local proof scope after guards pass. |
| `2` | A required local prerequisite or approved license metadata is unavailable. | Keep Homebrew support unclaimed and document the blocker. |
| Any other nonzero exit | The proof failed. | Stop and fix the failure before changing support wording. |

## Required Validation By Change Type

| Changed surface | Required validation |
| --- | --- |
| Homebrew template, proof script, package-manager wording, provider recipe inventory, or provider support claims | `scripts/package_manager_deferral_check.sh` |
| Static/shared package wording, install metadata, CMake package metadata, Makefile install behavior, shared-library wording, or dynamic ABI wording | `scripts/static_package_deferral_check.sh` |
| Install behavior, installed consumer documentation, CMake package files, `sparse.pc`, or downstream compile/link/run examples | `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh` |
| Package report metadata or package report artifacts | `python3 scripts/normalize_report_index.py --family package --check` and `python3 scripts/normalize_report_index.py --family package --check-freshness` |
| Any `.c` or `.h` file | `make format && make lint && make test` |

## Generated Output Policy

Generated formula files, local taps, source archives, logs, Homebrew caches,
build trees, install prefixes, and bottle outputs are temporary proof outputs.
They may be kept only for diagnostics with `--keep-temp`; they must not be
committed or treated as support evidence by themselves.

## Retained Non-Claims

Day 11 documentation retains these non-claims:

- Homebrew/core readiness;
- bottle or hosted binary support;
- Linuxbrew support;
- public tap maintenance;
- vcpkg, Conan, pkgsrc, distro/system package support;
- provider registry readiness;
- binary package install/update/uninstall support;
- shared-library package support;
- dynamic ABI compatibility;
- static/shared package selector support; and
- broad package-manager distribution.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and stops before archive/render/install/test work because root license metadata is absent. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and Homebrew blocker wording remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain enforced. |

## Day 12 Handoff

Day 12 should run the integrated package validation gate for all changed
surfaces:

1. expected-unavailable Homebrew proof;
2. package-manager deferral guard;
3. static-package deferral guard;
4. script syntax and documentation hygiene;
5. generated-output scan; and
6. no C/header change confirmation.

## Validation Scope

Day 11 changed documentation and planning documentation but no `.c` or `.h`
files, so the full C quality gate is not required.
