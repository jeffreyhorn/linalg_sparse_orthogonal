# Sprint 119 Day 3 Movement Feasibility Ranking

## Purpose

Day 3 ranks the eigensolver movement candidates from Day 2 by behavior-boundary
clarity, consumer breadth, proof cost, build-system impact, and rollback cost.
The output is a first movement recommendation for Days 4-7 plus explicit
move/defer conditions for the higher-risk candidates.

This artifact completes Sprint 119 Item 1. It does not move code.

## Baseline Inputs

| Evidence | Value |
|---|---|
| Primary source owner | `src/sparse_eigs.c` |
| Primary private contract | `src/sparse_eigs_internal.h` |
| Backend consumers | `src/sparse_eigs_thick_restart.c`, `src/sparse_eigs_lobpcg.c` |
| Focused test consumers | `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` |
| Build metadata touch points | `Makefile` `LIB_SRCS`, `CMakeLists.txt` library source list |
| CTest membership expectation | unchanged for movement-only candidates |
| Public API expectation | `include/sparse_eigs.h` unchanged |
| Public claim expectation | no broadened eigensolver, ARPACK, SciPy, LAPACK, or state-of-the-art claim |

## Starting Owner Metrics

| Owner | Lines | Feasibility relevance |
|---|---:|---|
| `src/sparse_eigs.c` | 1412 | Broad public front door plus shared helpers; source-boundary pressure is real. |
| `src/sparse_eigs_internal.h` | 631 | Large private declaration/comment surface; any split needs private-contract discipline. |
| `src/sparse_eigs_thick_restart.c` | 915 | Primary consumer for recurrence, selection, and lift helpers. |
| `src/sparse_eigs_lobpcg.c` | 401 | Primary LOBPCG consumer for selection and shift-invert-dispatched behavior. |
| `tests/test_eigs.c` | 2155 | Grow-m, shift-invert, repeated-handle, and public-result proof surface. |
| `tests/test_eigs_thick_restart.c` | 1377 | Thick-restart consumer and vector-publication proof surface. |
| `tests/test_eigs_lobpcg.c` | 1417 | LOBPCG behavior, dispatch, nearest-sigma, and parity proof surface. |

## Ranking Criteria

| Criterion | Lower-risk signal | Higher-risk signal |
|---|---|---|
| Behavior-boundary clarity | Pure helper with narrow inputs/outputs. | Setup/cleanup lifecycle or broad recurrence semantics. |
| Consumer breadth | One or two backend families and focused tests. | All public entry paths, all backends, and repeated workspace behavior. |
| Build impact | Single new private source plus stable internal declaration. | New context types, wider include fanout, or source/test membership changes. |
| Public behavior risk | Ordering/vector publication invariants already focused in tests. | Error propagation, cleanup, dispatch, or iteration semantics can drift. |
| Rollback cost | Remove one source from source lists and restore helper body. | Revert multi-file lifecycle extraction and context ownership. |
| Future value | Creates a clean owner for later selection/lift proof. | Large split that delays lower-risk source-boundary progress. |

## Ranked Candidate Table

| Rank | Candidate | Recommendation | Risk | Rationale |
|---:|---|---|---|---|
| 1 | `s20_select_indices` + `s20_lift_ritz_vectors` as one selection/publication helper owner | First movement batch candidate for Days 4-7. | Low-medium | Both helpers are private, already declared in `sparse_eigs_internal.h`, have explicit value/vector publication boundaries, and have focused grow-m/thick-restart/LOBPCG tests. Moving them together avoids splitting ordering from vector-publication proof. |
| 2 | `lanczos_iterate_op` recurrence owner | Design after selection/lift or defer if proof cost crowds Sprint 119. | Medium-high | Boundary is clear but behavior is central: grow-m, shift-invert, thick-restart empty-state delegation, and public repeated-handle behavior all depend on it. Movement likely needs broader recurrence proof and careful MGS/private-header ownership. |
| 3 | Shift-invert setup/conversion owner | Explicitly defer until Day 11 decision unless Day 8-10 proof reduces risk. | High | Public setup, LDLT lifecycle, `used_csc_path_ldlt`, operator selection, transformed eigenvalue conversion, error propagation, and cleanup all live in the front-door flow. A move can easily change public failure or cleanup behavior. |
| 4 | Broad eigensolver private-owner movement bucket | Do not treat as a single move. Use the ranked sub-candidates above. | High | The bucket is too broad to move safely without first carving specific behavior owners. Moving the broad owner would mix backend dispatch, workspace, recurrence, selection/lift, shift-invert, and refinement concerns. |

## First Movement Batch Recommendation

The recommended first movement batch is a bounded extraction of:

- `s20_select_indices`
- `s20_lift_ritz_vectors`

The likely Day 4 design should evaluate a new private source owner such as
`src/sparse_eigs_selection_internal.c` with declarations either remaining in
`src/sparse_eigs_internal.h` or moving to a narrower private header only if the
include impact stays small.

Why this batch first:

- the helpers are private and do not require a public header change;
- behavior is narrow enough to define as ordering plus vector-publication;
- the same build metadata pattern already supports eigensolver helper files;
- focused tests already cover grow-m vector lift, partial publication,
  shift-invert publication, thick-restart vector publication, LOBPCG selection,
  nearest-sigma parity, and backend dispatch adjacency;
- rollback is simple: restore helper definitions to `src/sparse_eigs.c`, remove
  the new source from `Makefile`/CMake, and rerun focused tests.

## Candidate-Specific Move/Defer Conditions

### `s20_select_indices`

Move only if Day 4-5 records:

- exact old/new file plan;
- no public API change;
- same private signature and sorted-ascending `theta` precondition;
- focused proof for `LARGEST`, `SMALLEST`, and `NEAREST_SIGMA` ordering;
- LOBPCG, grow-m, and thick-restart consumers still compile.

Defer if:

- selection requires new public options or output ordering changes;
- LOBPCG and thick-restart need divergent behavior;
- CTest membership or public result ordering changes unexpectedly.

### `s20_lift_ritz_vectors`

Move with `s20_select_indices` only if Day 4-5 records:

- unchanged column-major input/output contract;
- unchanged selected-index interpretation;
- grow-m and thick-restart vector-publication proof;
- partial-result vector-publication proof;
- no LOBPCG vector-publication coupling hidden behind the helper.

Defer if:

- lift proof requires changing result vector layout;
- grow-m and thick-restart need different lifting semantics;
- movement would force public result or docs changes.

### `lanczos_iterate_op`

Do not include in the first movement batch. Reconsider after the selection/lift
move only if there is enough sprint capacity for:

- recurrence-only source owner design;
- MGS helper/private-header ownership plan;
- compile proof for grow-m and thick-restart;
- focused proof for shift-invert recurrence, invariant-subspace behavior,
  `m_max <= n` validation, and grow-m repeated-handle behavior;
- rollback that does not disturb selection/lift work.

Defer if any recurrence semantics need changing or if proof would overlap Day
8-12 selection/lift and shift-invert work.

### Shift-Invert Setup/Conversion

Defer to the Day 11 boundary decision unless the intervening selection/lift
proof finds a much smaller split. Any movement must first prove:

- LDLT factor lifecycle and cleanup on all success/failure exits;
- diagonal shift construction and error propagation;
- `used_csc_path_ldlt` publication;
- operator callback selection;
- transformed eigenvalue conversion for converged and partial results;
- grow-m, thick-restart, and LOBPCG nearest-sigma parity.

Defer if ownership requires a new public context, changes `SPARSE_EIGS_NEAREST_SIGMA`
errors, or broadens claims.

## Proof-Risk Notes

| Risk | Applies to | Mitigation |
|---|---|---|
| Selection ordering drift | `s20_select_indices` | Focus tests on largest, smallest, nearest-sigma, LOBPCG adjacency, and shift-invert conversion. |
| Vector publication layout drift | `s20_lift_ritz_vectors` | Re-run grow-m, partial, shift-invert, and thick-restart vector-publication tests. |
| Backend compile-unit fanout | selection/lift and recurrence | Keep private signatures stable; update Makefile/CMake in the same commit as movement. |
| Recurrence semantic drift | `lanczos_iterate_op` | Do not move until a recurrence-specific proof plan exists. |
| Cleanup/error propagation drift | shift-invert | Defer until Day 11 lifecycle decision and failure-path proof. |
| Public claim drift | all candidates | Preserve Sprint 118 eigensolver non-claims in every movement artifact. |

## Rollback-Risk Notes

| Candidate | Rollback cost | Rollback path |
|---|---|---|
| `s20_select_indices` + `s20_lift_ritz_vectors` | Low-medium | Restore definitions to `src/sparse_eigs.c`, remove new source from Makefile/CMake, keep tests unchanged. |
| `lanczos_iterate_op` | Medium-high | Revert recurrence source/header split and restore body near `lanczos_iterate`; rerun all grow-m/thick-restart/shift-invert tests. |
| Shift-invert setup/conversion | High | Revert lifecycle extraction and restore setup/cleanup in `s46_sparse_eigs_sym_impl`; rerun all nearest-sigma and backend parity tests. |
| Broad private-owner bucket | High | Avoid broad movement; rollback would be too large and hard to prove within Sprint 119. |

## Day 4 Design Gate

Day 4 should design the first movement batch around the selection/lift helper
owner. The design should stop before implementation unless it records:

1. exact old/new file names;
2. unchanged helper signatures or a justified private-header plan;
3. Makefile and CMake source-list changes;
4. expected CTest count unchanged;
5. focused command list for `test_eigs`, `test_eigs_thick_restart`, and
   `test_eigs_lobpcg`;
6. rollback plan;
7. public API, package, ABI, and claim impact as none.

## Item 1 Completion Check

| Criterion | Status |
|---|---|
| Private-owner movement candidates are ranked. | Complete. |
| `s20_select_indices` and `s20_lift_ritz_vectors` are ranked by grow-m, thick-restart, and LOBPCG dependency risk. | Complete. |
| Shift-invert setup/conversion is ranked by LDLT lifecycle, operator selection, error propagation, and cleanup risk. | Complete. |
| `lanczos_iterate_op` is ranked by consumer breadth and compile-unit risk. | Complete. |
| Lowest-risk first movement batch is identified. | Complete. |
| Every higher-risk candidate has a defer condition or proof prerequisite. | Complete. |
