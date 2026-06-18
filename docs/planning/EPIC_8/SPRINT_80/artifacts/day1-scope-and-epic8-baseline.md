# Sprint 80 Day 1 - Scope and Epic 8 Baseline

Date: 2026-06-18  
Branch: sprint-80

## Purpose
Turn the Epic 8 project-plan section, the Epic 7 closeout state, and the live
Epic 8 review package into one bounded Sprint 80 execution baseline rather than
another generic planning reset.

## Main Result
Sprint 80 now starts from one explicit, measured baseline and one explicit set
of workstreams:

- baseline recheck
- competitive gap inventory
- external oracle contract
- performance / benchmark contract
- non-goal and risk fence
- review-package and closeout documentation

The strongest local reviewed baseline remains:
- `make quality-review-full`

Reviewed CMake parity is explicit before any Sprint 80 work:
- `ctest -N --test-dir build/quality-review-cmake` = `53`

## Starting Reading Fixed in Writing
Sprint 80 is not an implementation sprint yet.

It exists to make Epic 8 executable from:
- one fresh post-Epic-7 baseline
- one live gap inventory grounded in the tree
- one bounded external-comparison contract
- one bounded benchmark/performance reading
- one preserved non-goal and risk fence

## Strongest Likely Sprint 80 Touch Surfaces
Support, policy, and product-story surfaces:
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

Review-package surfaces:
- `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md` = `464`
- `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md` = `339`
- `docs/planning/EPIC_8/PROJECT_PLAN.md` = `351`

## Interpretation
The useful Day 1 clarification is now explicit:

- Sprint 80 should not start coding architectural fixes.
- It should first lock the baseline, comparison, benchmark, and non-goal
  contract for the rest of Epic 8.
- The main failure mode to avoid is fake forward motion that widens claims or
  dependencies before the evidence and execution fence are fixed.

## Preserved Epic 8 Non-goal Pressure
- no fake state-of-the-art claim inflation
- no broad subsystem redesign hidden inside baseline work
- no external dependency sprawl without an explicit contract
- no benchmark-threshold or platform-parity claim widening detached from
  maintained proof

## Exit State
- Sprint 80 now starts from an explicit baseline/setup package rather than from
  generic Epic 8 review prose.
- The workstreams, likely touch surfaces, and preserved non-goal pressure are
  fixed in writing.
- Day 2 can proceed from this bounded starting state.
