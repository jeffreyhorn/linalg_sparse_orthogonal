# Sprint 119 Day 2 Eigensolver Movement Candidate Inventory

## Purpose

Day 2 inventories the eigensolver residual movement candidates before any
ranking or code movement. The goal is to make every candidate's consumers,
source/build touch points, and public/private API impact visible enough for the
Day 3 feasibility ranking.

This artifact does not recommend movement order. It records audit inputs only.

## Input Evidence

| Input | Day 2 use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 119 Item 1 | Authoritative candidate list: private owner movement, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup, and `lanczos_iterate_op`. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day6-residual-owner-map.md` | Owner, dependency, and proof-gate expectations for Sprint 119 eigensolver residuals. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day10-hotspot-owner-handoff.md` | Hotspot interpretation, Sprint 119 source-boundary handoff, and required source-movement prerequisites. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day14-sprint-closeout-handoff.md` | Sprint 119 first proof gate, non-claim boundary, and residual deferred debt handoff. |
| `src/sparse_eigs.c` | Current owner of the public eigensolver front door, grow-m driver, shared selection/lift helpers, shift-invert setup, and `lanczos_iterate_op`. |
| `src/sparse_eigs_internal.h` | Current private declaration surface for candidate helpers and backend consumers. |
| `src/sparse_eigs_thick_restart.c` | Thick-restart consumer for shared Lanczos, selection, and lift helpers. |
| `src/sparse_eigs_lobpcg.c` | LOBPCG consumer for shared selection helper and shift-invert operator flow. |
| `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Focused consumer proof surfaces for grow-m, thick-restart, LOBPCG, shift-invert, repeated-handle, and public-result behavior. |
| `Makefile`, `CMakeLists.txt` | Source/test membership touch points for any future movement. |

## Current Eigensolver Source Boundary

| Surface | Current role | Movement relevance |
|---|---|---|
| `src/sparse_eigs.c` | Public eigensolver front door plus shared Lanczos-family helpers. Owns `lanczos_iterate_op`, grow-m driver, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup/conversion, backend dispatch, refinement, and one-shot/workspace public entry paths. | Primary source-boundary pressure point. Any extraction must avoid changing public entry behavior. |
| `src/sparse_eigs_internal.h` | Private shared eigensolver declarations for tests and backend source files. | Any movement that creates a new private owner likely changes this internal contract or adds a narrower internal header. |
| `src/sparse_eigs_workspace_internal.c` / `.h` | Reusable grow-m, thick-restart, and LOBPCG workspace allocation/view owner. | Candidate consumers depend on workspace views but Day 2 found no requirement to move workspace code with the Item 1 candidates. |
| `src/sparse_eigs_dense_internal.c` | Dense/tridiagonal helper owner used by eigensolver internals. | Touch point for selection/lift proof only through dense Ritz-pair outputs; no Day 2 candidate owns this file directly. |
| `src/sparse_eigs_thick_restart.c` | Thick-restart backend source owner. Consumes shared Lanczos helpers and internal declarations. | Primary consumer for `lanczos_iterate_op`, `s20_select_indices`, and `s20_lift_ritz_vectors`. |
| `src/sparse_eigs_lobpcg.c` | LOBPCG backend source owner. Consumes shared selection helper and operator flow. | Primary consumer for `s20_select_indices`; shift-invert behavior also flows through the public front door before dispatch. |

## Movement Candidate Inventory

| Candidate | Current owner | Private/public surface | Known consumers | Build touch points if moved |
|---|---|---|---|---|
| `s20_select_indices` | `src/sparse_eigs.c`; declared in `src/sparse_eigs_internal.h` | Private internal helper, not in public headers. | Grow-m backend in `src/sparse_eigs.c`; thick-restart outer loop in `src/sparse_eigs_thick_restart.c`; LOBPCG RR step in `src/sparse_eigs_lobpcg.c`; tests in `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, and `tests/test_eigs_lobpcg.c` indirectly through public/backend behavior. | New source file would need `Makefile` `LIB_SRCS`, `CMakeLists.txt` library source list, and private include handling. If left in existing file, no build metadata change. |
| `s20_lift_ritz_vectors` | `src/sparse_eigs.c`; declared in `src/sparse_eigs_internal.h` | Private internal helper, not in public headers. | Grow-m backend in `src/sparse_eigs.c`; thick-restart restart/vector publication in `src/sparse_eigs_thick_restart.c`; public vector publication tests in `tests/test_eigs.c` and `tests/test_eigs_thick_restart.c`. | New source file would need `Makefile`/CMake library source membership and private header contract; no public header change expected. |
| Shift-invert setup/conversion | Setup and conversion currently in `s46_sparse_eigs_sym_impl` and grow-m result publication in `src/sparse_eigs.c`; operator helper `s20_op_shift_invert` is file-local static. | Public behavior through `sparse_eigs_sym`/workspace entry for `SPARSE_EIGS_NEAREST_SIGMA`; private implementation uses LDLT factorization and `lanczos_op_fn`. | Grow-m, thick-restart, and LOBPCG backends receive the shift-invert operator after front-door setup; tests cover diagonal, indefinite, singular, eigenvector, wide-spectrum, CSC/linked-list dispatch, thick-restart KKT parity, and LOBPCG nearest-sigma parity. | Movement likely needs a new private owner for setup/cleanup and possibly a private struct/contract. It would touch source lists and may require broader include dependencies on `sparse_ldlt.h`, `sparse_matrix.h`, and `sparse_types.h`. |
| `lanczos_iterate_op` | `src/sparse_eigs.c`; declared in `src/sparse_eigs_internal.h` | Private internal recurrence helper used by multiple backends and targeted tests. | `lanczos_iterate` wrapper, grow-m backend, thick-restart empty-state delegation, thick-restart equivalent recurrence path, tests for grow-m public behavior and thick-restart empty-state matching. | Strong candidate for private recurrence owner, but movement would affect all Lanczos-family compile units. New file needs source-list/CMake membership and internal-header review. |
| Eigensolver private owner movement | Currently broad owner is `src/sparse_eigs.c`. | Private owner split only; public `include/sparse_eigs.h` should remain unchanged unless later evidence proves a bounded docs/API note is required. | All public eigensolver entry points and tests route through this file; candidate sub-owners above define the possible split axes. | Any actual split requires Makefile/CMake updates, CTest count proof, and focused tests. |

## Consumer Map

| Candidate | Grow-m Lanczos | Thick restart | LOBPCG | Shift-invert | Repeated handle/workspace | Focused tests |
|---|---|---|---|---|---|---|
| `s20_select_indices` | Selects converged and partial grow-m Ritz values in `s46_run_growm_backend`. | Selects locked/published Ritz pairs in `s21_thick_restart_outer_loop`. | Selects Ritz values in `s21_lobpcg_rr_step`. | Drives transformed `NEAREST_SIGMA` ordering because shift-invert uses largest `|theta|` selection before conversion. | Indirectly affects repeated handle outputs when grow-m, thick-restart, or LOBPCG use a reusable workspace. | `test_shift_invert_diagonal_k3`, `test_s114_shift_invert_vector_publication_boundary`, `test_s114_shift_invert_growm_conversion_nearest_sigma`, `test_thick_restart_*`, `test_lobpcg_nearest_sigma_*`, `test_lobpcg_adjacent_lanczos_public_result_parity`. |
| `s20_lift_ritz_vectors` | Publishes grow-m eigenvectors for converged and partial results. | Publishes thick-restart eigenvectors from the arrowhead/dense Jacobi basis. | Not a direct source consumer on Day 2; LOBPCG owns its own vector publication path. | Publishes original-space vectors for shift-invert because eigenspaces are shared. | Indirectly affects repeated handle vector-publication behavior. | `test_s114_growm_vector_lift_public_boundary`, `test_s114_growm_partial_vector_publication_sentinel_boundary`, `test_s114_shift_invert_vector_publication_boundary`, `test_s114_thick_restart_vector_publication_boundary`. |
| Shift-invert setup/conversion | Grow-m uses `s20_op_shift_invert` and converts `theta` to `sigma + 1/theta` in result publication. | Thick-restart receives the same operator and validates nearest-sigma parity. | LOBPCG receives the same operator flow through backend dispatch and validates nearest-sigma parity. | Primary owner: copy `A`, subtract sigma from diagonal, factor with LDLT, set `used_csc_path_ldlt`, route operator, clean up factor/matrix, convert transformed Ritz values. | Public handle/workspace path reuses `s46_sparse_eigs_sym_impl`, so setup/cleanup must remain compatible with one-shot and workspace-backed calls. | `test_shift_invert_diagonal_k3`, `test_shift_invert_indefinite_small`, `test_shift_invert_singular_sigma`, `test_shift_invert_eigenvectors`, `test_shift_invert_wide_spectrum_middle`, `test_indefinite_shift_invert_uses_csc_above_threshold`, `test_indefinite_shift_invert_uses_linked_list_below_threshold`, `test_thick_restart_kkt_nearest_sigma_parity`, `test_lobpcg_nearest_sigma_diagonal`, `test_lobpcg_nearest_sigma_kkt`. |
| `lanczos_iterate_op` | Core recurrence for grow-m backend and `lanczos_iterate` wrapper. | Empty-state thick-restart delegates directly; later thick-restart phase mirrors the recurrence and shares MGS semantics. | Not a direct source consumer; LOBPCG uses the same `lanczos_op_fn` abstraction but not this recurrence. | Core recurrence applies either default matvec or LDLT shift-invert operator. | Indirectly affects grow-m repeated handle behavior because workspace-backed grow-m uses the same recurrence. | `test_growm_lanczos_iterate_op_public_behavior`, `test_thick_restart_iterate_empty_state_matches_lanczos`, `test_thick_restart_iterate_tridiag_empty_state_matches_lanczos`, grow-m public handle tests, shift-invert tests. |

## Source-List And CMake Touch Points

| Surface | Current state | Day 2 movement implication |
|---|---|---|
| `Makefile` `LIB_SRCS` | Includes `src/sparse_eigs_workspace_internal.c`, `src/sparse_eigs_dense_internal.c`, `src/sparse_eigs_lobpcg.c`, `src/sparse_eigs_thick_restart.c`, and `src/sparse_eigs.c`. | Any new eigensolver helper source file must be added near the existing eigensolver cluster. |
| `Makefile` `TEST_SRCS` | Includes `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, and `tests/test_eigs_lobpcg.c`. | Day 2 expects CTest membership to remain unchanged unless a later movement deliberately adds a focused test owner. |
| `CMakeLists.txt` library sources | Includes the same eigensolver source cluster. | Any new source file must be added to the `sparse_lu_ortho` source list. |
| `CMakeLists.txt` tests | Registers `test_eigs`, `test_eigs_thick_restart`, and `test_eigs_lobpcg`. | Any test membership change needs expected-count evidence. Candidate movement alone should not change CTest count. |
| `src/sparse_eigs_internal.h` | Declares candidate helpers used across eigensolver compile units and tests. | Movement may keep declarations here or introduce a narrower private header; public headers should not change for Day 2 candidates. |
| `include/sparse_eigs.h` | Public API and option/result contract. | No Day 2 candidate requires public API movement or claim expansion. |

## Public/Private API Impact Notes

| Candidate | Public API impact | Private API impact |
|---|---|---|
| `s20_select_indices` | None expected; public behavior impact only through result ordering and nearest-sigma conversion. | Declaration could stay in `sparse_eigs_internal.h` or move to a narrower private selection header if extraction proceeds. |
| `s20_lift_ritz_vectors` | None expected; public behavior impact through eigenvector publication shape, order, and partial-result behavior. | Declaration could stay in `sparse_eigs_internal.h` or move with selection/vector-publication helpers. |
| Shift-invert setup/conversion | Public behavior must remain exactly `SPARSE_EIGS_NEAREST_SIGMA` setup, error propagation, `used_csc_path_ldlt`, backend dispatch, eigenvalue conversion, and cleanup. | Likely requires a private setup context/cleanup helper if moved; the operator callback must retain `lanczos_op_fn` compatibility. |
| `lanczos_iterate_op` | None expected; public behavior impact through all Lanczos-family backends and internal tests. | Private recurrence owner would need a stable internal declaration and include boundary for MGS, timer, and operator callback dependencies. |

## Day 3 Ranking Checklist

Day 3 should not rank a candidate until it answers:

1. Which exact behavior boundary would move?
2. Which source file currently owns the behavior?
3. Which new source/private-header owner would receive it?
4. Which grow-m, thick-restart, LOBPCG, shift-invert, repeated-handle, and
   test consumers must pass?
5. Does movement require Makefile source-list changes?
6. Does movement require CMake source-list or CTest membership changes?
7. Does movement preserve `include/sparse_eigs.h` and public claim wording?
8. What rollback is possible if compile, CTest count, or focused behavior
   proof fails?

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 1 audit inputs are complete for Day 2. | Complete. |
| Every named movement candidate has consumers recorded. | Complete. |
| Source-list and CMake touch points are visible. | Complete. |
| Public/private API impact notes are recorded. | Complete. |
| Day 3 ranking checklist exists. | Complete. |
| No candidate is ranked before its consumer and build impacts are visible. | Complete. |
