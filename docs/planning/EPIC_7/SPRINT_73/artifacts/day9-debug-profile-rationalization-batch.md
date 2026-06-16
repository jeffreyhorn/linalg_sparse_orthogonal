# Sprint 73 Day 9: Debug/Profile Rationalization Batch

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Land the bounded second Sprint 73 implementation batch around the strongest
remaining developer-only/profile spill after the Day 8 design pass.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day8-debug-profile-rationalization-design.md`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_graph_internal.h`
- `src/sparse_reorder_nd_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

## Day 9 Implementation Results

### 1. `SPARSE_HCC_DEBUG` now resolves through one internal owner

Touched implementation surfaces:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_internal.h`

Landed contract:

- `sparse_graph_hcc_debug_current()` is now the one internal owner of HCC
  debug activation
- current-thread override begin/end helpers now exist:
  - `sparse_graph_hcc_debug_override_begin(...)`
  - `sparse_graph_hcc_debug_override_end()`
- legacy `SPARSE_HCC_DEBUG` env behavior is preserved when no override is
  active
- HCC debug print sites now consume the internal owner instead of open-coding
  repeated `getenv(...)` checks

### 2. `SPARSE_ND_PROFILE` now resolves through one internal owner

Touched implementation surfaces:

- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`

Landed contract:

- `sparse_reorder_nd_profile_current()` is now the one internal owner of ND
  profile activation
- current-thread override begin/end helpers now exist:
  - `sparse_reorder_nd_profile_override_begin(...)`
  - `sparse_reorder_nd_profile_override_end()`
- legacy `SPARSE_ND_PROFILE` env behavior is preserved when no override is
  active
- the top-level ND entry path now consumes the internal owner instead of
  open-coding another direct env read

### 3. Focused proof landed in the correct owners

Touched proof surfaces:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

New regressions:

- `test_hcc_debug_override_precedence`
- `test_nd_profile_override_precedence`

Those tests pin the exact Day 9 precedence rule:

- env-set default is visible when no override is active
- explicit override wins while active
- clearing the override restores env-driven behavior
- explicit internal enable/disable also works with the env unset

### 4. Support-only QG profile follow-through stayed deferred

The batch did not widen into:

- `src/sparse_reorder_amd_qg.c`

That preserves the Day 8 fence:

- `SPARSE_QG_PROFILE` remains support-only follow-through
- no new public typed debug/profile option family was added
- no docs/header/platform/SVD spill landed

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 296.46 sec`

## Exit State

Sprint 73 Day 9 closes with:

1. one explicit internal precedence seam for `SPARSE_HCC_DEBUG`
2. one explicit internal precedence seam for `SPARSE_ND_PROFILE`
3. two focused precedence regressions in `test_graph` and `test_reorder_nd`
4. the `SPARSE_QG_PROFILE` lane still deferred as support-only follow-through
