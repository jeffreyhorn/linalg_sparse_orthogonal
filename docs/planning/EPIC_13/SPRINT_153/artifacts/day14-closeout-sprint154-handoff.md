# Sprint 153 Day 14 Closeout And Sprint 154 Handoff

## Purpose

Day 14 closes Sprint 153 by consolidating the shared-library ABI product
decision, implementation evidence, validation status, residual debt, and the
handoff constraints for Sprint 154 external-comparison work.

## Sprint 153 Product Decision

Sprint 153 selected stronger static-first deferral. It did not implement
shared-library support.

The maintained product contract remains:

- static archive install/export only;
- `BUILD_SHARED_LIBS=ON` rejected at CMake configure time;
- installed CMake package metadata exposes `Sparse::sparse_lu_ortho` as a
  static imported target;
- installed `sparse.pc` describes static archive package metadata;
- Unix Make install and CMake install proofs compile, link, and run installed
  static consumers;
- Windows CI mirrors the CMake-first installed static consumer proof;
- no platform claims shared-library packaging, dynamic ABI compatibility, or
  runtime-loader support.

## Implementation Summary

Sprint 153 implemented the selected decision by strengthening the static-first
deferral rather than widening the supported package surface:

- `CMakeLists.txt` now rejects `BUILD_SHARED_LIBS=ON` with exact blockers for
  export/import policy, symbol visibility policy, dynamic ABI policy, Linux
  SONAME metadata, macOS install-name/RPATH metadata, Windows
  DLL/import-library behavior, installed shared consumer proof, and
  runtime-loader validation.
- `scripts/static_package_deferral_check.sh` verifies that the rejection path
  still names those exact blocker categories and that unsupported shared
  metadata does not appear in the CMake source surface.
- `tests/test_cmake_install.sh` rejects unsupported installed CMake package
  metadata for loader behavior and static/shared selectors before such support
  is selected.
- `.github/workflows/windows-ci.yml` mirrors the unsupported-loader and
  static/shared selector metadata check in the Windows CMake install/downstream
  confidence path.
- `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` now describe the
  exact static-first package boundary and shared-library deferral blockers.

## Artifact Index

| Day | Artifact | Evidence Role |
| --- | --- | --- |
| Day 1 | `day1-abi-intake-baseline.md` | Static-first package baseline, artifact structure, and stop conditions. |
| Day 2 | `day2-public-abi-surface-audit.md` | Public ABI-relevant header, type, ownership, callback, and export-risk inventory. |
| Day 3 | `day3-platform-loader-audit.md` | Linux, macOS, and Windows shared-loader requirement audit. |
| Day 4 | `day4-product-decision-criteria.md` | Criteria for implementing shared support versus strengthening static deferral. |
| Day 5 | `day5-shared-library-abi-product-decision.md` | Product decision record selecting stronger static-first deferral. |
| Day 6 | `day6-build-install-design.md` | Build/install design for the selected static-first decision. |
| Day 7 | `day7-build-install-implementation.md` | CMake diagnostic and static-deferral guard implementation record. |
| Day 8 | `day8-downstream-proof-design.md` | Downstream proof design and metadata-check scope. |
| Day 9 | `day9-downstream-proof-implementation.md` | Installed CMake package metadata guard implementation record. |
| Day 10 | `day10-platform-ci-policy.md` | Linux, macOS, and Windows CI policy review. |
| Day 11 | `day11-ci-docs-implementation.md` | Windows CI mirror and documentation alignment record. |
| Day 12 | `day12-integrated-package-abi-validation.md` | Integrated package, ABI-boundary, and report-index validation record. |
| Day 13 | `day13-quality-gate-residual-review.md` | Quality-gate decision and residual debt register. |
| Day 14 | `day14-closeout-sprint154-handoff.md` | Closeout summary and Sprint 154 comparison handoff. |

## Sprint 154 External-Comparison Handoff

Sprint 154 may compare the project against external libraries and package
ecosystems only within the supported evidence boundary recorded here.

Comparison work may cite:

- static archive package install/export proof;
- installed Unix `pkg-config` consumer proof;
- installed Unix and Windows CMake consumer proof;
- exact CMake rejection diagnostics for unsupported shared-library requests;
- documented non-claims for dynamic ABI, runtime-loader, shared-library,
  package-manager, and Windows parity surfaces.

Comparison work must not infer:

- shared-library support from static archive package proof;
- dynamic ABI compatibility from public header availability;
- Linux `.so`, macOS `.dylib`, or Windows DLL/import-library support from
  CMake package installation;
- SONAME, install-name, RPATH, DLL lookup, or runtime-loader support from
  installed static consumer proof;
- package-manager distribution from generated package metadata;
- Windows Makefile parity or Windows `pkg-config` execution parity from the
  Windows CMake-first lane.

If external comparison mentions SuiteSparse, Eigen, CHOLMOD, distro package
managers, Homebrew, vcpkg, Conan, or dynamic linking, the comparison must keep
those capabilities separate from the current supported static-first package
surface unless new implementation and validation evidence is added.

## Residuals Carried Forward

The shared-library path remains deferred until these blockers are closed:

- public export/import macro policy;
- symbol visibility or export allowlist;
- dynamic ABI compatibility policy for public structs, callbacks, enum values,
  allocator/lifetime boundaries, error state, and version metadata;
- Linux SONAME and `.so` loader proof;
- macOS install-name/RPATH and `.dylib` loader proof;
- Windows DLL/import-library behavior and runtime lookup proof;
- installed shared CMake consumer proof;
- shared/static `pkg-config` and CMake package selector semantics;
- package-manager distribution proof;
- Windows Makefile parity;
- Windows `pkg-config` execution parity.

## Final Validation Results

Day 14 reran the final focused package, report, documentation, and whitespace
validation set.

| Validation | Result | Evidence |
| --- | --- | --- |
| Static deferral guard | Pass | `bash scripts/static_package_deferral_check.sh` passed. |
| Make install/package proof | Pass | `bash tests/test_install.sh` passed with `23` checks and `0` failures. |
| CMake install/export proof | Pass | `bash tests/test_cmake_install.sh` passed with `27` checks, `0` failures, and `0` skips. |
| Package report-index structure | Pass | `python3 scripts/normalize_report_index.py --family package --check` reported `6` rows ok. |
| Package report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family package --check-freshness` reported freshness ok for `6` source-controlled package rows. |
| Runtime-backend report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` reported freshness ok for `1` source-controlled row. |
| Focused stale wording scan | Pass | Remaining hits are expected non-claims, static deferral diagnostics, and guard/test patterns. |
| Whitespace | Pass | `git diff --check` passed. |
| Final status | Pass | `git status --short` shows only intended Sprint 153 branch changes. |

## Closeout Status

Sprint 153 is ready for retrospective preparation. The sprint closed the
selected shared-library ABI product decision by strengthening the static-first
deferral and preserving exact unsupported-claim boundaries for Sprint 154.
