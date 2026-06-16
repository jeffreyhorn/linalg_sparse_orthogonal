# Sprint 73 Day 12: Proof Alignment and Build Map

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Confirm that the landed Sprint 73 configuration boundary already has the right
focused proof owners and align the maintainer policy wording with that live
proof map.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day11-docs-maintainer-header-follow-through-batch.md`
- `docs/maintainer_guide.md`
- `include/sparse_analysis.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

## Day 12 Review Result

### 1. No new regression code was needed

The live proof already sits in the right owners:

- `tests/test_graph.c`
  - FM-family compatibility env behavior
  - `SPARSE_HCC_DEBUG` internal override precedence
- `tests/test_reorder_nd.c`
  - typed analysis ND controls overriding compatibility env vars
  - internal/default-policy ND fallback behavior
  - `SPARSE_ND_PROFILE` internal override precedence

The focused precedence regressions already landed in the correct proof owners:

- `test_hcc_debug_override_precedence`
- `test_nd_profile_override_precedence`

The broader ND typed/default/env contract was already covered by the retained
typed override and default/fallthrough regressions in `tests/test_reorder_nd.c`.

### 2. The real Day 12 gap was proof-owner wording

Touched maintained surface:

- `docs/maintainer_guide.md`

The maintainer guide now states directly that:

- `tests/test_graph.c` is the maintained proof owner for graph/FM
  compatibility behavior and `SPARSE_HCC_DEBUG` override precedence
- `tests/test_reorder_nd.c` is the maintained proof owner for ND
  typed/default/env behavior and `SPARSE_ND_PROFILE` override precedence
- `src/sparse_reorder_amd_qg.c` and `SPARSE_QG_PROFILE` remain explicitly
  deferred support-only context
- examples and reorder benchmarks remain support/reporting surfaces, not proof
  owners, on this lane

### 3. The support-only surfaces stayed bounded

The batch did not widen into:

- `include/sparse_analysis.h`
- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

## Build / Validation Map for Day 13

The final Sprint 73 validation queue is now explicit:

- standard gate:
  - `make format`
  - `make lint`
  - `make test`
- strongest reviewed baseline:
  - `make quality-review-full`
- focused follow-ons:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

## Sanity Checks

This was a docs-only batch, so I did not run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I used the targeted docs-only sanity set instead:

- touched-surface diff review
- proof-owner alignment reread
- touched-surface `wc -l`
- branch-status recheck

Raw `wc -l` counts after the landing:

- `docs/maintainer_guide.md` = `621`
- `tests/test_graph.c` = `2925`
- `tests/test_reorder_nd.c` = `2287`

## Exit State

Sprint 73 Day 12 closes with:

1. no new regression code added
2. proof ownership stated directly in `docs/maintainer_guide.md`
3. the Day 13 validation queue fixed explicitly from the live touched surfaces
4. the `SPARSE_QG_PROFILE` lane still deferred as support-only context
