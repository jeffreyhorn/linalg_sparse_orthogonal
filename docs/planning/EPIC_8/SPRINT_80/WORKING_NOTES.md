# Sprint 80 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 80 baseline for Epic 8 by grounding the sprint in
the validated Epic 7 end state, the live Epic 8 review/todo/project-plan
package, and the current permanent validation, package, benchmark, and policy
surfaces rather than another generic planning reset.

### Actions
- Re-read the Sprint 80 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 80 day-by-day plan in `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-read the Epic 7 close state in
  `docs/planning/EPIC_7/EPIC_7_RETROSPECTIVE.md`.
- Re-read the strongest direct closeout handoff artifact from Epic 7:
  `docs/planning/EPIC_7/SPRINT_79/artifacts/day14-closeout-and-handoff.md`.
- Re-read the Epic 8 opening review package:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`
- Rechecked the maintained reviewed wrapper surface with
  `make -n quality-review-full`.
- Reconfirmed the reviewed CMake parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 80
  touch surfaces across support/policy, package/install/export, workflow, and
  review-package surfaces.
- Opened Sprint 80 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 80 starts from the same strongest local reviewed baseline Epic 7
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 80 work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Sprint 80 is not another broad architecture or feature sprint. Its highest
  value is to make Epic 8 executable from one fresh, truthful, measurable
  starting point.
- The highest-value Sprint 80 workstreams are now fixed explicitly:
  - baseline recheck
  - competitive gap inventory
  - external oracle contract
  - performance / benchmark contract
  - non-goal and risk fence
  - review-package and closeout documentation
- The strongest likely Sprint 80 support, policy, and product-story surfaces
  are explicit from the live tree:
  - `README.md` = `1050`
  - `INSTALL.md` = `265`
  - `docs/maintainer_guide.md` = `698`
  - `benchmarks/README.md` = `393`
  - `Makefile` = `899`
  - `CMakeLists.txt` = `413`
  - `scripts/bench_canonical_report.sh` = `101`
  - `tests/test_install.sh` = `172`
  - `tests/test_cmake_install.sh` = `146`
  - `.github/workflows/ci.yml` = `223`
  - `.github/workflows/macos-ci.yml` = `117`
  - `.github/workflows/windows-ci.yml` = `63`
- The strongest likely Sprint 80 review-package surfaces are also explicit:
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md` = `464`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md` = `339`
  - `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`
- The strongest Day 1 clarification is now fixed:
  - Sprint 80 should not start implementation work.
  - It should first lock one current baseline, one contract for external
    comparison, one benchmark/performance reading, and one risk fence for the
    rest of Epic 8.
- The preserved Epic 8 non-goal pressure is explicit before Day 2:
  - no fake state-of-the-art claim inflation
  - no broad subsystem redesign hidden inside baseline work
  - no external dependency sprawl without an explicit contract
  - no benchmark-threshold or platform-parity claim widening detached from
    maintained proof

### Validation
- Rechecked `make -n quality-review-full`.
- Reconfirmed the reviewed parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live support/policy, package/install/export, workflow, and
  review-package hotspot maps from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 80 no longer starts from generic Epic 8 planning prose.
- The baseline, gap-inventory, external-oracle, benchmark-contract, and
  non-goal/risk-fence workstreams are fixed in writing.
- The strongest likely Sprint 80 touch surfaces are explicit before the
  validation-baseline recheck begins.

## Day 2 - Validation Baseline

### Goal
Reconfirm the Sprint 80 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, canonical benchmark/report command surfaces, and install/export proof
owners before any Epic 8 baseline or contract batch lands.

### Actions
- Re-read the Sprint 80 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_80/PLAN.md`.
- Re-materialized the reviewed CMake parity tree with
  `make quality-review-cmake-compile`.
- Reconfirmed the reviewed CMake parity anchor with
  `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner tests and representative examples
  most likely to matter early in Epic 8:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical report command surface with
  `make -n bench-canonical-report`.
- Reconfirmed the root `build/` canonical benchmark emitters consumed by that
  report path:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`

### Findings
- Sprint 80 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 80 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial baseline, comparison-contract, or integration batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Epic-8 proof
  surfaces:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- The canonical benchmark/reporting lane remains command- and script-owned
  rather than reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest early-Epic-8 proof and truth-surface ownership split is
  already stable:
  - reviewed CMake proof owners and representative examples remain the main
    executable truth surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 80 rerun set is now fixed around the likely touched
  baseline and contract seams:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Ran `make quality-review-cmake-compile`.
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Epic 8.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 80 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the competitive gap inventory
  and external-oracle design work begins.
