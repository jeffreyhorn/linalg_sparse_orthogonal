# Sprint 61 Day 6: Typed Analysis/Reorder Option Batch 1

Date: 2026-06-09
Branch: `sprint-61`


## Purpose

Land the first bounded Epic 6 configuration-surface modernization batch by
adding the first caller-facing typed analysis/reorder controls, translating
them through a resolved internal ND policy seam, preserving legacy env-var
compatibility for unspecified fields, and proving typed-over-env precedence on
live ordering behavior.

## Scope

### Touched files

- `include/sparse_analysis.h`
- `src/sparse_graph_internal.h`
- `src/sparse_reorder_nd_internal.h`
- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`

### Selected controls in this batch

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`

### Explicit non-goals

- public FM tuning controls
- debug/profile option migration
- `include/sparse_reorder.h` widening
- generic repo-wide configuration helpers
- backend/AUTO policy work
- packaging/platform work

## Landed Public Surface

`include/sparse_analysis.h` now exposes one bounded nested reorder-options
surface on `sparse_analysis_opts_t`:

- `sparse_analysis_reorder_opts_t`
  - `supernodal_postorder`
  - `nd_root_bisect`
  - `nd_root_bisect_max_n`

The public typed enums are:

- `sparse_analysis_supernodal_postorder_t`
  - `SPARSE_ANALYSIS_SUPERNODAL_POSTORDER_DEFAULT`
  - `SPARSE_ANALYSIS_SUPERNODAL_POSTORDER_OFF`
  - `SPARSE_ANALYSIS_SUPERNODAL_POSTORDER_ON`
- `sparse_analysis_nd_root_bisect_t`
  - `SPARSE_ANALYSIS_ND_ROOT_BISECT_DEFAULT`
  - `SPARSE_ANALYSIS_ND_ROOT_BISECT_MULTILEVEL`
  - `SPARSE_ANALYSIS_ND_ROOT_BISECT_SPECTRAL`

The widened API remains zero-init safe:

- `DEFAULT` / `0` leaves the field unspecified
- unspecified fields continue to resolve through compatibility overrides and
  then internal defaults

## Internal Policy Bridge

The first resolved internal ND policy seam is now explicit:

- `src/sparse_graph_internal.h`
  - `sparse_graph_supernodal_postorder_mode_t`
  - `sparse_graph_nd_root_bisect_mode_t`
  - `sparse_graph_nd_policy_t`
- `src/sparse_reorder_nd_internal.h`
  - `sparse_reorder_nd_with_policy(...)`

`src/sparse_analysis.c` now resolves the selected controls through one explicit
precedence chain:

1. explicit typed option
2. legacy compatibility override, only when the typed field is left
   unspecified/default
3. internal typed policy default

Current internal defaults remain:

- supernodal postorder: `OFF`
- ND root bisect: `MULTILEVEL`
- ND root bisect max N: `50000`

Validation in the resolver is explicit:

- invalid public enum values return `SPARSE_ERR_BADARG`
- negative `nd_root_bisect_max_n` returns `SPARSE_ERR_BADARG`

## ND Consumer Integration

`src/sparse_reorder_nd.c` now supports both lanes:

- public compatibility wrapper:
  - `sparse_reorder_nd(...)`
- internal policy-aware entry:
  - `sparse_reorder_nd_with_policy(...)`

The policy now threads through the root spectral-bisect decision and ND
recursion instead of re-reading only process-global env vars at the
`sparse_analyze(...)` call site.

This keeps the public reorder API stable while allowing `sparse_analyze(...)`
to honor typed controls without a reorder-API redesign.

## Proof Surface

`tests/test_reorder_nd.c` now carries bounded typed-over-env precedence tests
for the first selected controls:

- typed `MULTILEVEL` overriding env `SPARSE_ND_ROOT_BISECT=spectral`
- typed `nd_root_bisect_max_n = 50000` overriding env
  `SPARSE_ND_ROOT_BISECT_MAX_N=1`
- typed `supernodal_postorder = OFF` overriding env
  `SPARSE_SUPERNODAL_POSTORDER=on`

These tests are behavior-level proofs:

- they drive `sparse_analyze(...)`
- they compare actual resulting permutations
- they only skip when the chosen matrix path cannot distinguish the compared
  strategies meaningfully

## Validation

Required gate after a clean rebuild:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 280.04 sec`

Representative retained proof points:

- `test_reorder_nd` passed with the new typed-over-env precedence tests
- reviewed CMake parity rebuilt and passed the graph/reorder proof surface:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

## Clean-Rebuild Note

The first post-edit `make test` run reported false integration failures after
the `sparse_analysis_opts_t` layout change because some dependent objects were
still built against the old header layout.

This was resolved by:

- `make clean`
- rerunning the full required gate

The final validated Day 6 baseline is the clean-rebuild result above.

## Exit State

Sprint 61 Day 6 now has the first live Phase 1 configuration-modernization
slice:

- the first public typed analysis/reorder options are shipped
- the precedence bridge from typed option to compatibility override to internal
  default is shipped
- the ND consumer path honors the resolved policy without a public reorder-API
  redesign
- legacy env-var behavior remains intact for unspecified typed fields
- the Day 7 queue is now narrowed to the deeper selected ND controls rather
  than the initial public bridge
