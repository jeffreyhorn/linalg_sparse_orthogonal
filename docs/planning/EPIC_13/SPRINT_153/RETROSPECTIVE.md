# Sprint 153 Retrospective

**Sprint:** 153 - Shared-Library ABI Product Decision
**Duration:** 14 days (Days 1-14 landed on branch `sprint-153`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 153 day-by-day plan, working notes, artifact directory,
      closeout artifact, Sprint 154 handoff, and retrospective.
- [x] Audited the current static-first package/install baseline across Make,
      CMake, `pkg-config`, installed headers, generated version metadata, and
      CI package lanes.
- [x] Audited public ABI-relevant headers, concrete structs, callback
      contracts, allocator/lifetime boundaries, process/static state, and
      accidental export risks.
- [x] Audited Linux, macOS, and Windows shared-loader requirements, including
      SONAME, install-name/RPATH, DLL/import-library, symbol visibility, and
      downstream installed consumer proof obligations.
- [x] Selected the Sprint 153 product path: keep static-first packaging and
      strengthen the shared-library deferral with exact, test-backed blockers.
- [x] Preserved the static target, static install/export layout,
      `Sparse::sparse_lu_ortho` static imported target behavior, and static
      archive `sparse.pc` metadata.
- [x] Strengthened `BUILD_SHARED_LIBS=ON` rejection diagnostics so they name
      export/import policy, symbol visibility policy, dynamic ABI policy,
      Linux SONAME metadata, macOS install-name/RPATH metadata, Windows
      DLL/import-library behavior, installed shared consumer proof, and
      runtime-loader validation.
- [x] Strengthened the static package deferral guard to verify exact blocker
      wording and reject unsupported shared metadata in the CMake/package
      surface.
- [x] Added installed CMake package metadata proof that rejects unsupported
      loader and static/shared selector metadata before such support is
      selected.
- [x] Mirrored the installed CMake metadata guard in the Windows CMake
      install/downstream confidence path.
- [x] Updated README, INSTALL, and maintainer guidance so package and ABI
      wording matches the static-first decision and exact shared-library
      blockers.
- [x] Ran final focused package, report-index, documentation, stale wording,
      and whitespace validation.
- [x] Prepared the Sprint 154 external-comparison handoff.

## What Went Well

1. **The shared-library decision became explicit.** Sprint 153 avoided the
   common trap of treating `BUILD_SHARED_LIBS` as a product decision. The
   sprint records why shared-library support is deferred and what must be true
   before that claim can be made.

2. **Static-first support became more defensible.** The maintained package
   surface remains static archive install/export only, but the rejection path
   now names exact blockers rather than leaving shared support vaguely
   unsupported.

3. **ABI risk was inspected before build changes.** The sprint audited public
   structs, callbacks, ownership, allocator/lifetime behavior, error state,
   version metadata, and accidental internal-symbol exposure before deciding
   whether to implement shared support.

4. **Platform loader requirements were separated from package metadata.**
   Linux `.so`, macOS `.dylib`, and Windows DLL/import-library work now have
   concrete proof requirements instead of being bundled into generic install
   wording.

5. **The downstream proof stayed aligned with the selected claim.** Unix Make
   and CMake install proofs still compile, link, and run installed static
   consumers, while CMake package metadata checks now reject unsupported loader
   and static/shared selector leakage.

6. **Windows CI stayed scoped.** The Windows lane mirrors the installed CMake
   metadata checks without claiming Windows Makefile parity, Windows
   `pkg-config` execution parity, DLL support, dynamic ABI support, or
   runtime-loader behavior.

## What Didn't Go Well

1. **No shared-library implementation landed.** That is the correct product
   decision for this sprint, but it means the project still lacks an actual
   supported dynamic-linking surface.

2. **The public ABI surface remains too concrete for casual binary promises.**
   Public structs, callbacks, enum-like macros, allocator/lifetime boundaries,
   and process/static state still need a deliberate compatibility policy before
   shared-library support can be considered mature.

3. **Symbol visibility is unresolved.** The source tree still has internal
   helper symbols that would need hiding, export lists, linker scripts, `.def`
   files, attributes, or refactoring before a shared artifact could be
   responsibly shipped.

4. **The package proof is intentionally absence/rejection proof.** The new
   checks are valuable, but they prove static-first boundaries and unsupported
   metadata absence. They do not prove runtime-loader behavior.

5. **External comparison remains constrained.** Sprint 154 can compare static
   package evidence, but it cannot honestly compare this project as a
   shared-library or package-manager product without new evidence.

## Final Metrics

### Validation

| Metric | Sprint 153 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no; focused package/report/docs gate used |
| static deferral guard | passed: `bash scripts/static_package_deferral_check.sh` |
| Make install/package proof | passed: `23` checks, `0` failures |
| CMake install/export proof | passed: `27` checks, `0` failures, `0` skips |
| package report-index structure | passed: `6` rows ok |
| package report-index freshness meaning | passed: freshness ok for `6` source-controlled rows |
| runtime-backend report-index freshness meaning | passed: freshness ok for `1` source-controlled row |
| focused stale wording search | passed; remaining hits are expected non-claims, diagnostics, and guard/test patterns |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 153 close state |
| --- | ---: |
| daily artifacts under `SPRINT_153/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| C source files changed | 0 |
| public header files changed | 0 |
| build-system files changed | 1 |
| workflow files changed | 1 |
| shell/package proof files changed | 2 |
| public/support docs changed | 3 |
| selected shared-library product decision records | 1 |
| exact shared-library blocker categories recorded | 8 |

## Closed Claim

Sprint 153 closes this package/ABI product claim:

The project intentionally maintains a static-first package surface and rejects
shared-library requests until explicit export/import policy, symbol visibility
policy, dynamic ABI policy, Linux SONAME metadata, macOS install-name/RPATH
metadata, Windows DLL/import-library behavior, installed shared consumer proof,
and runtime-loader validation are designed, implemented, documented, and
tested.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-abi-intake-baseline.md](./artifacts/day1-abi-intake-baseline.md);
- [day2-public-abi-surface-audit.md](./artifacts/day2-public-abi-surface-audit.md);
- [day3-platform-loader-audit.md](./artifacts/day3-platform-loader-audit.md);
- [day4-product-decision-criteria.md](./artifacts/day4-product-decision-criteria.md);
- [day5-shared-library-abi-product-decision.md](./artifacts/day5-shared-library-abi-product-decision.md);
- [day6-build-install-design.md](./artifacts/day6-build-install-design.md);
- [day7-build-install-implementation.md](./artifacts/day7-build-install-implementation.md);
- [day8-downstream-proof-design.md](./artifacts/day8-downstream-proof-design.md);
- [day9-downstream-proof-implementation.md](./artifacts/day9-downstream-proof-implementation.md);
- [day10-platform-ci-policy.md](./artifacts/day10-platform-ci-policy.md);
- [day11-ci-docs-implementation.md](./artifacts/day11-ci-docs-implementation.md);
- [day12-integrated-package-abi-validation.md](./artifacts/day12-integrated-package-abi-validation.md);
- [day13-quality-gate-residual-review.md](./artifacts/day13-quality-gate-residual-review.md);
- [day14-closeout-sprint154-handoff.md](./artifacts/day14-closeout-sprint154-handoff.md).

## Next-Sprint Readiness

Sprint 154 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| External comparisons | Compare only against the maintained static-first package evidence unless new implementation proof is added. |
| Shared-library claims | Treat shared-library packaging, dynamic ABI compatibility, and runtime-loader behavior as unsupported. |
| Linux/macOS dynamic linking | Do not infer `.so`, `.dylib`, SONAME, install-name, RPATH, or loader support from static package proof. |
| Windows package behavior | Treat the reviewed Windows lane as CMake-first static install/downstream proof only. |
| Package metadata | Keep `sparse.pc` and CMake package comparisons scoped to static archive metadata. |
| ABI claims | Require separate policy and proof for public structs, callbacks, allocator/lifetime boundaries, error state, version metadata, and symbol exports. |
| Package managers | Do not cite distro packages, Homebrew, vcpkg, Conan, or other manager support without actual packaging evidence. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 153 close:

- public export/import macro policy;
- symbol visibility and export allowlist;
- dynamic ABI compatibility policy for public structs, callbacks, enum values,
  allocator/lifetime boundaries, error state, and version metadata;
- Linux `.so` support and SONAME/loader proof;
- macOS `.dylib` support and install-name/RPATH/loader proof;
- Windows DLL/import-library support, runtime lookup, and C runtime allocator
  boundary proof;
- installed shared CMake consumer proof;
- shared/static CMake and `pkg-config` selector semantics;
- package-manager distribution proof;
- Windows Makefile parity;
- Windows `pkg-config` execution parity.

Still consciously constrained rather than silently solved:

- static archive install proof is not shared-library proof;
- public header availability is not dynamic ABI compatibility proof;
- CMake package installation is not runtime-loader proof;
- source-controlled package rows are governed by Git review and schema checks,
  not generated freshness proof;
- hosted workflow logs are external evidence and not local report freshness
  artifacts;
- external library comparisons must preserve the static-first evidence
  boundary.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-abi-intake-baseline.md](./artifacts/day1-abi-intake-baseline.md)
- [day2-public-abi-surface-audit.md](./artifacts/day2-public-abi-surface-audit.md)
- [day3-platform-loader-audit.md](./artifacts/day3-platform-loader-audit.md)
- [day4-product-decision-criteria.md](./artifacts/day4-product-decision-criteria.md)
- [day5-shared-library-abi-product-decision.md](./artifacts/day5-shared-library-abi-product-decision.md)
- [day6-build-install-design.md](./artifacts/day6-build-install-design.md)
- [day7-build-install-implementation.md](./artifacts/day7-build-install-implementation.md)
- [day8-downstream-proof-design.md](./artifacts/day8-downstream-proof-design.md)
- [day9-downstream-proof-implementation.md](./artifacts/day9-downstream-proof-implementation.md)
- [day10-platform-ci-policy.md](./artifacts/day10-platform-ci-policy.md)
- [day11-ci-docs-implementation.md](./artifacts/day11-ci-docs-implementation.md)
- [day12-integrated-package-abi-validation.md](./artifacts/day12-integrated-package-abi-validation.md)
- [day13-quality-gate-residual-review.md](./artifacts/day13-quality-gate-residual-review.md)
- [day14-closeout-sprint154-handoff.md](./artifacts/day14-closeout-sprint154-handoff.md)
