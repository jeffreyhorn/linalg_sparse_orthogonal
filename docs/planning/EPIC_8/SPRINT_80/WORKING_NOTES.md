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
