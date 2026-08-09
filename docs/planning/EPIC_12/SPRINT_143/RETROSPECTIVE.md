# Sprint 143 Retrospective

**Sprint:** 143 - Shared-Library ABI Decision & Static-First Contract Follow-Through
**Duration:** 14 days (Days 1-14 landed on branch `sprint-143`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 143 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Consumed Sprint 142 package/ABI handoff while keeping runtime/backend
      sentinel evidence separate from package proof.
- [x] Audited the installed public header surface, public declarations,
      ABI-sensitive type/layout/callback surfaces, and static archive symbol
      visibility risk.
- [x] Audited Make install/uninstall, CMake install/export,
      `SparseConfigVersion.cmake`, `sparse.pc.in`, install proof scripts,
      static package deferral guard, and package proof-owner rows.
- [x] Audited Linux, macOS, and Windows CI package lanes, platform support
      tiers, loader risks, Windows/MSVC DLL/import-library risks, macOS
      install-name risks, Linux soname/RPATH risks, and Sprint 144 platform
      promotion boundaries.
- [x] Made the explicit package/ABI product decision:
      Sprint 143 implements stricter static-first-only support and defers
      shared-library ABI support.
- [x] Implemented the selected static-first package path:
  - removed the unused CMake `LIBRARY DESTINATION` from the explicit static
    target install rule;
  - clarified CMake `pkg-config` ownership comments;
  - changed `sparse.pc.in` to describe static archive package metadata;
  - strengthened the static package deferral guard for explicit static target,
    archive install metadata, no runtime/shared install destinations, and the
    static `.pc` description.
- [x] Strengthened Make install and CMake install/export proof:
  - installed `.pc` metadata must describe the static archive package
    contract;
  - `pkg-config` metadata must avoid unsupported package/ABI wording;
  - CMake package metadata must avoid shared/module imported targets and shared
    imported locations;
  - downstream runtime output checks became deterministic.
- [x] Strengthened downstream consumer proof:
  - Make/`pkg-config` basic consumer and maintained example compile, link, and
    run;
  - installed CMake example configure/build/run remains proved;
  - exact-version CMake consumer now configures, builds, and runs;
  - mismatched-version CMake lookup remains rejected.
- [x] Aligned Linux, macOS, and Windows CI comments and supplemental Windows
      CMake install/downstream proof with the selected static-first package
      contract.
- [x] Preserved package report rows as source-controlled proof-owner metadata,
      not fresh install-run evidence.
- [x] Updated README, INSTALL, and maintainer guide package wording for static
      `.pc` metadata, downstream proof, exact-version proof, support-tier
      boundaries, and non-claims.
- [x] Published earned static-first package claims, preserved non-claims, and
      future-owner handoff for shared ABI, loader, package-manager, and
      platform promotion work.
- [x] Ran Sprint 143 validation:
  - `bash -n tests/test_install.sh tests/test_cmake_install.sh
    scripts/static_package_deferral_check.sh`;
  - `bash scripts/static_package_deferral_check.sh`;
  - `bash tests/test_install.sh`;
  - `bash tests/test_cmake_install.sh`;
  - `python3 scripts/normalize_report_index.py --family package --check`;
  - `python3 scripts/normalize_report_index.py --family package
    --check-freshness`;
  - Linux/macOS/Windows workflow YAML parse;
  - package/docs/workflow claim-boundary scans;
  - generated-output hygiene checks;
  - `git diff --check`;
  - trailing-whitespace scans.
- [x] No C or header files changed, so the sprint did not require
      `make format && make lint && make test`.

## What Went Well

1. **The sprint made one product decision instead of partial support for two
   paths.** The Day 5 decision selected stricter static-first-only support and
   explicitly rejected shared-library ABI implementation for Sprint 143.

2. **The static-first contract became executable rather than descriptive.**
   The guard script now checks explicit static target shape, archive-only
   install metadata, absence of runtime/shared destinations, no shared ABI
   metadata, no package selectors, and the static `.pc` description.

3. **Downstream proof became stronger without changing consumer commands.**
   Existing users still use `pkg-config --cflags --libs sparse` and
   `find_package(Sparse REQUIRED)` with `Sparse::sparse_lu_ortho`, while tests
   now prove runtime output and exact-version build/run behavior more tightly.

4. **Platform support-tier wording stayed honest.** Linux remains the reviewed
   static-first package-contract source of truth, while macOS and Windows
   install/downstream lanes remain supplemental confidence paths pending
   Sprint 144 promotion decisions.

5. **The docs now reflect the actual package mechanics.** README, INSTALL, and
   the maintainer guide describe static `.pc` metadata, no selectors,
   downstream proof, exact-version proof, and the non-claim boundary without
   widening package-manager, loader, ABI, or platform claims.

## What Didn't Go Well

1. **The shared-library gap is still large.** Day 2-4 audits made clear that
   real shared support requires export/import macros, hidden visibility,
   symbol allowlists, ABI versioning, loader proof, platform-specific
   metadata, and selector semantics. Sprint 143 rightly deferred it, but the
   residual remains substantial.

2. **Package proof spans many surfaces.** A narrow static-first decision still
   touched CMake, `pkg-config`, shell scripts, CI, README, INSTALL, maintainer
   docs, report rows, and planning artifacts. The scope stayed coherent, but
   future package work should keep the Day 6 ownership map close.

3. **Windows proof can only be partially validated locally.** The workflow YAML
   parsed and the PowerShell block was reviewed structurally, but local `pwsh`
   was unavailable. The strengthened Windows supplemental proof will receive
   hosted validation in CI.

4. **The package report rows are easy to overread.** The normalized package
   rows are useful source-controlled proof-owner metadata, but they still must
   not be cited as evidence that an install command just ran.

## Final Metrics

### Validation

| Metric | Sprint 143 close state |
| --- | --- |
| tracked `.c`/`.h` changes | no |
| full C quality gate required | no |
| shell syntax checks | passed |
| `scripts/static_package_deferral_check.sh` | passed |
| `tests/test_install.sh` | passed: 23 passed, 0 failed |
| `tests/test_cmake_install.sh` | passed: 26 passed, 0 failed, 0 skipped |
| package report index check | passed: 6 rows |
| package report freshness check | passed: 6 source-controlled advisory rows |
| workflow YAML parse | passed for Linux, macOS, and Windows |
| package/docs/workflow claim-boundary scan | passed |
| generated-output hygiene | passed |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

### Artifact Package

| Metric | Sprint 143 close state |
| --- | ---: |
| daily artifacts under `SPRINT_143/artifacts/` | 14 |
| final retrospective files | 1 |
| package/build metadata files changed | 2 |
| shell package proof scripts changed | 3 |
| workflow files changed | 3 |
| public/maintainer documentation surfaces changed | 3 |
| C/header files changed | 0 |
| source-controlled generated report files | 0 |

## Closed Claim

Sprint 143 closes this claim:

The project now has an explicit maintained static-first package contract with
stronger install/export metadata, downstream consumer proof, exact-version
proof, unsupported shared-artifact checks, CI/support-tier alignment, and
public/maintainer documentation. Shared-library packaging, dynamic ABI,
runtime-loader behavior, package-manager support, static/shared selectors, and
macOS/Windows install parity remain explicit non-claims.

This claim is supported by:

- `CMakeLists.txt`;
- `sparse.pc.in`;
- `scripts/static_package_deferral_check.sh`;
- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `.github/workflows/ci.yml`;
- `.github/workflows/macos-ci.yml`;
- `.github/workflows/windows-ci.yml`;
- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- Day 12 focused validation evidence;
- Day 13 quality gate and claim closure;
- Day 14 closeout validation summary.

## Sprint 144 Readiness

Sprint 144 should consume Sprint 143's static-first package semantics as its
platform-promotion input:

| Handoff field | Sprint 144 requirement |
| --- | --- |
| macOS package promotion | Decide whether supplemental static-first Make install/`pkg-config` and CMake install/export confidence should become reviewed macOS install/export parity, and define hosted-runner repetition and failure ownership. |
| Windows package promotion | Decide whether supplemental CMake install/downstream confidence should become reviewed Windows install-validation parity for the exact static-first scope. |
| Windows parity boundaries | Keep Windows Makefile parity, Windows `pkg-config` parity, and POSIX-backed staged tests separate from static CMake package proof unless explicitly promoted. |
| hosted proof | Use hosted CI results, not local planning rows, before promoting a supplemental platform lane. |
| package non-claims | Preserve no shared-library, dynamic ABI, runtime-loader, package-manager, static/shared selector, portable performance, or state-of-the-art claims. |

## Residual Deferred Debt

Most important carry-forward work:

- shared-library build/install/export support;
- dynamic ABI compatibility and ABI version policy;
- public export/import macro and visibility policy;
- exported symbol allowlist and internal symbol hiding;
- Linux RPATH/RUNPATH, macOS install-name, and Windows DLL/import-library
  loader proof;
- static/shared package selector semantics if dual artifacts are ever
  supported;
- package-manager distribution;
- Sprint 144 macOS/Windows reviewed install/export parity decisions.

Still consciously constrained rather than silently solved:

- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager availability;
- no Windows Makefile or `pkg-config` parity;
- no macOS/Windows reviewed install/export parity;
- no portable performance claim;
- no state-of-the-art claim;
- no hosted platform promotion from local source-controlled rows.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-package-abi-intake.md](./artifacts/day1-package-abi-intake.md)
- [day2-public-header-symbol-audit.md](./artifacts/day2-public-header-symbol-audit.md)
- [day3-install-export-metadata-audit.md](./artifacts/day3-install-export-metadata-audit.md)
- [day4-platform-loader-risk-audit.md](./artifacts/day4-platform-loader-risk-audit.md)
- [day5-package-abi-product-decision.md](./artifacts/day5-package-abi-product-decision.md)
- [day6-static-first-implementation-design.md](./artifacts/day6-static-first-implementation-design.md)
- [day7-package-batch1-static-first-metadata.md](./artifacts/day7-package-batch1-static-first-metadata.md)
- [day8-package-batch2-install-proof-diagnostics.md](./artifacts/day8-package-batch2-install-proof-diagnostics.md)
- [day9-downstream-consumer-proof.md](./artifacts/day9-downstream-consumer-proof.md)
- [day10-ci-package-report-alignment.md](./artifacts/day10-ci-package-report-alignment.md)
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md)
- [day12-focused-package-validation.md](./artifacts/day12-focused-package-validation.md)
- [day13-quality-gate-claim-closure.md](./artifacts/day13-quality-gate-claim-closure.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 143 is complete. It closes the package/ABI decision sprint by choosing
and implementing the stricter static-first path, strengthening install/export
and downstream consumer proof, aligning CI and documentation, publishing
validation evidence, and handing macOS/Windows platform-promotion decisions to
Sprint 144. It does not implement or imply shared-library ABI support.
