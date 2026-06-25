# Sprint 90 Day 1: Scope and Epic 9 Baseline

## Purpose

Turn the Sprint 90 project-plan section and the Epic 8 validated close state
into one bounded Epic 9 baseline, target-freeze, and planning execution
package before any deeper review, todo, or project-plan widening lands.

## Starting Truth

Sprint 90 begins from a validated Epic 8 close state, not from another
generic planning reset:

- strongest local reviewed baseline remains `make quality-review-full`
- reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`

Epic 8 already moved the strongest prior contradiction classes:

- one bounded external SPD comparison lane exists and agrees
- one bounded package/install/export contract exists and passes
- one materially cleaner front-door/support split exists
- one smaller and better-calibrated residual queue exists than before Sprint
  80

That means Sprint 90 can start from the next real contradiction center:

- the post-Epic-8 gap between a highly disciplined sparse library and a truly
  state-of-the-art sparse numerical product

## Sprint 90 Workstreams

The highest-value Sprint 90 package is now fixed explicitly around:

- baseline recheck
- target-state freeze
- product-model audit
- comparison/measurement contract
- non-goal and risk fence
- review/todo/project-plan package

## Strongest Baseline Starting Point

The live maintained project state is sharper and more truthful than the tree
Epic 8 began from:

- the strongest reviewed baseline still exists as one retained source of truth
- local install/export proof is real and explicit
- canonical benchmark reporting remains single-owned and repeatable
- the package and front-door story are materially cleaner than at Epic 8
  start
- the epic-close residual queue already distinguishes carry-forward work from
  bounded non-claims

Sprint 90 therefore does not begin from "start planning the next epic." It
begins from one explicit baseline question:

- what still keeps the live post-Epic-8 tree from reading as a state-of-the-art
  sparse numerical product, and how should Epic 9 rank, bound, and sequence
  those contradictions before implementation starts

## Strongest Likely Touch Surfaces

The live tree currently points most strongly at these Sprint 90 surfaces:

- Epic 9 planning and prior-epic interpretation owners:
  - `docs/planning/EPIC_9/PROJECT_PLAN.md`
  - `docs/planning/EPIC_8/EPIC_8_RETROSPECTIVE.md`
  - `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
  - `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`
- strongest support, package, build, and workflow owners:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`
- strongest residual code and proof hotspots likely to matter in the review:
  - `src/sparse_matrix.c`
  - `src/sparse_dense.c`
  - `src/sparse_iterative.c`
  - `src/sparse_ldlt_csc.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`

## Preserved Fence

Sprint 90 is explicitly bounded against:

- writing a generic repo review detached from the live tree
- widening into implementation work before the contradiction map and target
  state are fixed
- inflating package, capability, runtime, or cross-platform claims beyond the
  maintained proof-owner surfaces
- treating benchmark/runtime evidence as stronger than the reviewed and
  install/export proof surfaces
- drafting the Epic 9 execution order before the anti-sprawl fence is written

## Day 1 Result

Sprint 90 now starts from one precise Epic 9 baseline package rather than
from a generic "plan the next epic" bucket. The strongest likely touch
surfaces, preserved non-goals, and maintained starting truth are fixed in
writing before the validation and maintained-surface recheck begins.
