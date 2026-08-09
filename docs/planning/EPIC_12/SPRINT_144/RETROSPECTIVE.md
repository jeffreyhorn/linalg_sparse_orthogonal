# Sprint 144 Retrospective

**Sprint:** 144 - Platform Promotion Lane Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-144`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 144 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Consumed Sprint 143's static-first package/ABI decision as prerequisite
      evidence without widening shared-library, ABI, loader, or
      package-manager claims.
- [x] Inventoried candidate platform lanes:
  - macOS reviewed static-first install/export proof;
  - Windows reviewed install/downstream parity;
  - Windows staged test portability;
  - Linux source-of-truth strengthening.
- [x] Selected exactly one lane for complete closure: macOS reviewed
      static-first Make install/`pkg-config` and CMake install/export proof
      for the maintained static archive package contract.
- [x] Captured the selected-lane before-change evidence baseline, including
      local package proof results and the support-tier wording drift that kept
      macOS package jobs supplemental.
- [x] Designed the portability and CI promotion path while keeping the sprint
      source-neutral unless hosted macOS CI exposed a real defect.
- [x] Promoted the macOS workflow package proof wording:
  - Make install/`pkg-config` proof is now reviewed macOS static-first proof;
  - CMake install/export proof is now reviewed macOS static-first proof;
  - proof commands remained unchanged.
- [x] Preserved Linux and Windows boundaries:
  - Linux remains the strongest reviewed source-of-truth baseline;
  - Windows remains reviewed CMake subset plus supplemental CMake
    install/downstream confidence;
  - Windows staged `test_threads`, `test_sprint4_integration`, and
    `test_fuzz` remain source-level pthread/POSIX blocker work.
- [x] Integrated the selected macOS lane into source-controlled CI report
      semantics while keeping hosted CI logs external proof.
- [x] Updated README, INSTALL, and maintainer guidance so support-tier wording
      matches the earned macOS reviewed static-first package proof.
- [x] Published promotion evidence, residual blocker ledger, preserved
      non-claims, Sprint 145 adoption handoff, and Day 14 closeout summary.
- [x] Ran Sprint 144 validation:
  - Ruby workflow YAML parse for Linux, macOS, and Windows workflows;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/normalize_report_index.py --family package --check`;
  - `python3 scripts/normalize_report_index.py --family ci --check`;
  - `python3 scripts/normalize_report_index.py --family package
    --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family ci
    --check-freshness`;
  - `bash scripts/static_package_deferral_check.sh`;
  - `bash tests/test_install.sh`;
  - `bash tests/test_cmake_install.sh`;
  - Linux/macOS/Windows support-tier preservation scans;
  - stale macOS supplemental install/export wording scans;
  - unsupported package/platform claim-boundary scans;
  - `git diff --check`.
- [x] No C or header files changed, so the sprint did not require
      `make format && make lint && make test`.

## What Went Well

1. **The sprint closed one platform lane instead of partially touching all of
   them.** Day 2 selected macOS reviewed static-first install/export proof and
   deferred Windows parity and staged-test portability with explicit reasons.

2. **The selected lane needed support-tier promotion, not package mechanics
   churn.** Day 3 and Day 4 showed that Make install/`pkg-config`, CMake
   install/export, downstream consumers, and static deferral checks already had
   local proof. The implementation could therefore promote proof ownership in
   workflow, report, and documentation surfaces without changing C sources,
   install scripts, or package templates.

3. **The macOS claim is now precise.** The reviewed macOS claim is limited to
   static-first install/export proof for the maintained static archive package
   contract. It does not imply shared libraries, ABI stability, loader
   behavior, package-manager support, static/shared selectors, or broad macOS
   parity.

4. **Cross-platform boundaries stayed coherent.** Linux remained the strongest
   reviewed source of truth, Windows stayed CMake-first with supplemental
   install/downstream confidence, and staged Windows tests remained explicitly
   blocked by pthread/POSIX source dependencies.

5. **The final handoff is adoption-ready.** Sprint 145 can now describe a
   simple platform contract: Linux is the strongest source-of-truth baseline,
   macOS has reviewed static-first install/export proof, and Windows remains
   CMake-first with narrower reviewed support.

## What Didn't Go Well

1. **The strongest proof still depends on hosted macOS CI.** Local validation
   proves commands, scripts, workflow syntax, report rows, and support wording,
   but the reviewed macOS support-tier claim receives final external proof only
   from GitHub-hosted `macos-latest` execution.

2. **Historical artifacts make stale-wording scans noisy.** Earlier Day 1-4
   artifacts correctly describe the before-change supplemental state, so final
   scans must distinguish current support surfaces from historical planning
   evidence.

3. **Windows parity remains unresolved.** Sprint 144 deliberately avoided
   partial Windows promotion. Windows still lacks reviewed Makefile parity,
   `pkg-config` parity, reviewed install-validation parity, and staged
   pthread/POSIX test closure.

4. **Static-first packaging remains a constrained product posture.** The sprint
   promoted one platform lane for static package proof, but it did not reduce
   the larger shared-library ABI, loader, symbol, and package-manager gaps.

## Final Metrics

### Validation

| Metric | Sprint 144 close state |
| --- | --- |
| tracked `.c`/`.h` changes | no |
| full C quality gate required | no |
| workflow YAML parse | passed for Linux, macOS, and Windows |
| corpus schema validation | passed |
| package report index check | passed: 6 rows |
| CI report index check | passed: 1 row |
| package report freshness check | passed: 6 source-controlled advisory rows |
| CI report freshness check | passed: 1 source-controlled advisory row |
| `scripts/static_package_deferral_check.sh` | passed |
| `tests/test_install.sh` | passed: 23 passed, 0 failed |
| `tests/test_cmake_install.sh` | passed: 26 passed, 0 failed, 0 skipped |
| support-tier preservation scans | passed |
| stale macOS supplemental install/export wording scan | passed for current support surfaces |
| unsupported package/platform claim-boundary scan | passed |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 144 close state |
| --- | ---: |
| daily artifacts under `SPRINT_144/artifacts/` | 14 |
| final retrospective files | 1 |
| workflow files changed | 1 |
| report manifest files changed | 1 |
| public/maintainer documentation surfaces changed | 3 |
| C/header files changed | 0 |
| package scripts/templates changed | 0 |
| source-controlled generated report files | 0 |

## Closed Claim

Sprint 144 closes this claim:

The project now has reviewed macOS static-first Make install/`pkg-config` and
CMake install/export proof for the maintained static archive package contract,
with workflow, report, README, INSTALL, maintainer guidance, sprint evidence,
and closeout notes aligned on the same support-tier boundary.

This claim is supported by:

- `.github/workflows/macos-ci.yml`;
- `tests/corpus/manifests/report_families.tsv`;
- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `scripts/static_package_deferral_check.sh`;
- Day 10 selected-lane validation;
- Day 11 cross-platform non-regression review;
- Day 12 quality gate execution;
- Day 13 promotion evidence and residual non-claims;
- Day 14 closeout validation summary.

## Sprint 145 Readiness

Sprint 145 should consume Sprint 144's platform support-tier contract as its
adoption-facing baseline:

| Handoff field | Sprint 145 requirement |
| --- | --- |
| Linux | Present Linux as the strongest reviewed source-of-truth baseline for Make, CMake, dead-code, package-contract, and broad local quality proof. |
| macOS | Present reviewed static-first Make install/`pkg-config` and CMake install/export proof for the maintained static archive package contract. |
| Windows | Present Windows as CMake-first: reviewed MSVC CMake subset proof plus supplemental CMake install/downstream confidence. |
| staged Windows tests | Keep `test_threads`, `test_sprint4_integration`, and `test_fuzz` outside reviewed Windows evidence until source-level pthread/POSIX blockers are closed. |
| package/adoption wording | Prefer static-first install language and avoid shared-library, ABI, loader, package-manager, Windows Makefile, Windows `pkg-config`, Windows install-validation parity, and broad platform parity claims. |

## Residual Deferred Debt

Most important carry-forward work:

- Windows reviewed install-validation parity;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows staged `test_threads` and `test_sprint4_integration` portability;
- Windows staged `test_fuzz` portability;
- shared-library build/install/export support;
- dynamic ABI compatibility and ABI version policy;
- public export/import macro and visibility policy;
- exported symbol allowlist and internal symbol hiding;
- Linux RPATH/RUNPATH, macOS install-name, and Windows DLL/import-library
  loader proof;
- package-manager distribution;
- portable performance parity.

Still consciously constrained rather than silently solved:

- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager availability;
- no static/shared package selector support;
- no Windows Makefile or `pkg-config` parity;
- no Windows reviewed install-validation parity;
- no Windows staged test closure;
- no broad macOS platform parity beyond reviewed static-first install/export
  proof;
- no portable performance claim;
- no state-of-the-art claim;
- no hosted proof inferred from source-controlled report rows alone.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-platform-promotion-intake.md](./artifacts/day1-platform-promotion-intake.md)
- [day2-platform-lane-selection.md](./artifacts/day2-platform-lane-selection.md)
- [day3-blocker-reproduction-evidence-baseline.md](./artifacts/day3-blocker-reproduction-evidence-baseline.md)
- [day4-portability-design.md](./artifacts/day4-portability-design.md)
- [day5-source-script-fix-batch.md](./artifacts/day5-source-script-fix-batch.md)
- [day6-ci-promotion-design.md](./artifacts/day6-ci-promotion-design.md)
- [day7-ci-promotion-implementation.md](./artifacts/day7-ci-promotion-implementation.md)
- [day8-package-report-integration.md](./artifacts/day8-package-report-integration.md)
- [day9-documentation-support-tier-alignment.md](./artifacts/day9-documentation-support-tier-alignment.md)
- [day10-selected-lane-validation.md](./artifacts/day10-selected-lane-validation.md)
- [day11-cross-platform-non-regression-review.md](./artifacts/day11-cross-platform-non-regression-review.md)
- [day12-quality-gate-execution.md](./artifacts/day12-quality-gate-execution.md)
- [day13-promotion-evidence-and-residual-nonclaims.md](./artifacts/day13-promotion-evidence-and-residual-nonclaims.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 144 is complete. It closes the platform promotion sprint by promoting
the macOS static-first install/export lane to reviewed proof for the maintained
static archive package contract, aligning workflow/report/docs/planning
evidence, preserving non-selected platform boundaries, and handing an
adoption-ready platform contract to Sprint 145. It does not implement or imply
shared-library ABI support, package-manager support, Windows parity, or broad
platform parity.
