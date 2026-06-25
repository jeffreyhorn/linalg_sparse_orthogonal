# Sprint 89 Day 1: Scope and Final-Integration Baseline

## Purpose

Turn the Sprint 89 project-plan section and the Sprint 88 validated closeout
into one bounded final-integration, external-comparison, and Epic 8 closeout
execution package before any last-mile fix, validation, or summary-writing
change lands.

## Starting Truth

Sprint 89 begins from a validated Sprint 88 close state, not from another
generic Epic 8 reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Sprint 88 already moved the strongest prior contradiction:

- one bounded README/front-door simplification package landed
- one bounded examples/workflow simplification package landed
- one bounded support-surface consolidation package landed

That means Sprint 89 can start from the next real Epic 8 contradiction center:

- the current final evidence ceiling on the live post-Sprint-88 tree across
  end-state review, external comparison, final validation/reporting, and
  truthful project closeout

## Sprint 89 Workstreams

The highest-value Sprint 89 package is now fixed explicitly around:

- end-state re-audit
- external comparison sweep
- final cross-surface fix batch
- full validation and reporting sweep
- residual queue finalization
- retrospective, handoff, and final project-summary closeout

## Strongest End-State Starting Point

The live maintained project state is already sharper and more truthful than
earlier Epic 8 phases:

- the front door is cleaner
- the package/install/export contract is better bounded
- maintained local install/export proof is real and explicit
- canonical benchmark reporting remains single-owned and repeatable
- the strongest reviewed baseline still exists as one retained source of truth

Sprint 89 therefore does not begin from "write the final summary." It begins
from one explicit evidence question:

- what still fails the original Epic 8 review when measured against the live
  post-Sprint-88 tree, and what needs to be fixed, calibrated, or explicitly
  carried forward before Epic 8 can close truthfully

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 89 surfaces:

- planning and closeout owners:
  - `docs/planning/EPIC_8/PROJECT_PLAN.md`
  - `docs/planning/EPIC_8/SPRINT_88/RETROSPECTIVE.md`
  - `docs/planning/EPIC_8/SPRINT_88/artifacts/day14-closeout-and-handoff.md`
- maintained proof and reporting owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/README.md`
- support and package-truth surfaces:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt`
  - `Makefile`
- representative reviewed runtime and graph/reorder surfaces that may still
  matter if the re-audit forces a final fix batch:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
  - `tests/test_reorder_amd_qg.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`

## Preserved Fence

Sprint 89 is explicitly bounded against:

- reopening earlier sprint scope without a fresh end-state contradiction
- widening capability, runtime, packaging, or usability claims without a
  maintained proof owner
- treating comparison or benchmark evidence as stronger than the retained
  reviewed and install/export proof surfaces
- writing Epic 8 closeout prose before the final validation/reporting baseline
  exists
- drifting into generic cleanup detached from the final evidence package

## Day 1 Result

Sprint 89 now starts from one precise final-integration and Epic 8 closeout
execution package rather than from a generic "wrap things up" bucket. The
strongest likely touch surfaces, preserved non-goals, and maintained
end-state baseline are fixed in writing before the validation and
cross-surface recheck begins.
