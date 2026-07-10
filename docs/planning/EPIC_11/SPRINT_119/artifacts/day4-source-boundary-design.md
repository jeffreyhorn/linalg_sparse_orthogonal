# Sprint 119 Day 4 Source Boundary Design

## Purpose

Day 4 turns the Day 3 first-movement recommendation into an exact
source-boundary design before implementation. The planned movement is the
paired private selection/publication helper extraction:

- `s20_select_indices`
- `s20_lift_ritz_vectors`

No code is moved by this artifact. It defines the gate that Day 5-7 must use
before and after implementation.

## Scope

| Field | Decision |
|---|---|
| Movement batch | Selection and Ritz-vector publication helper owner. |
| Current owner | `src/sparse_eigs.c` |
| Proposed new source owner | `src/sparse_eigs_selection_internal.c` |
| Existing private declaration owner | `src/sparse_eigs_internal.h` |
| Public headers | No change to `include/sparse_eigs.h`. |
| Tests | No new test file planned for the first movement batch. |
| CTest membership | Expected unchanged. |
| Public claims | No eigensolver claim expansion. |

## Behavior Boundary

### Boundary Being Moved

The first movement batch moves only the following private helper behavior:

1. selection of Ritz indices from sorted ascending `theta[0..m)`;
2. Ritz-vector lift from the Lanczos basis and small projected eigenvector
   matrix into public-result eigenvectors.

### Boundary Not Being Moved

The first movement batch does not move:

- `lanczos_iterate_op`;
- `lanczos_iterate`;
- `s20_op_shift_invert`;
- shift-invert setup, LDLT lifecycle, cleanup, or eigenvalue conversion;
- grow-m retry logic;
- thick-restart state management;
- LOBPCG RR step implementation;
- public option/result validation;
- refinement logic;
- workspace allocation.

## Old/New File Plan

| Current file | Proposed file | Ownership after change | Notes |
|---|---|---|---|
| `src/sparse_eigs.c` | `src/sparse_eigs_selection_internal.c` | Selection ordering and Ritz-vector publication helper implementation. | Move the existing `s20_select_indices` and `s20_lift_ritz_vectors` function bodies without signature or semantic changes. |
| `src/sparse_eigs_internal.h` | unchanged | Private declarations remain available to grow-m, thick-restart, LOBPCG, and focused tests. | Do not add a narrower header in this batch; it would add include churn without reducing current risk. |
| `include/sparse_eigs.h` | unchanged | Public eigensolver API remains stable. | No public API, ABI, or docs contract change. |

## Internal Header And Private API Contract

| Contract item | Decision |
|---|---|
| Helper signatures | Preserve exactly: `s20_select_indices(...)` and `s20_lift_ritz_vectors(...)`. |
| Header ownership | Keep declarations in `src/sparse_eigs_internal.h` for this batch. |
| Required includes for new source | `sparse_eigs_internal.h`, plus standard headers needed by the moved bodies, expected `<math.h>` and `<stddef.h>` only if compiler requires them. |
| Public API impact | None. |
| ABI/package impact | None; helpers are private and not exported as public headers. |
| Naming | Keep existing `s20_` names for rollback simplicity and to avoid widening the change. |
| Behavior comments | Move or preserve the existing behavior comments with the function bodies so future ownership remains readable. |

## Source-List, Makefile, And CMake Impact

| Surface | Expected update | Validation expectation |
|---|---|---|
| `Makefile` `LIB_SRCS` | Add `$(SRCDIR)/sparse_eigs_selection_internal.c` near the existing eigensolver cluster before `sparse_eigs.c`. | Source-list/build proof on implementation day; full C quality if `.c`/`.h` changes. |
| `CMakeLists.txt` library sources | Add `src/sparse_eigs_selection_internal.c` near `src/sparse_eigs_dense_internal.c`, `src/sparse_eigs_lobpcg.c`, `src/sparse_eigs_thick_restart.c`, and `src/sparse_eigs.c`. | CMake configure/build or reviewed CMake parity lane as affected. |
| `CMakeLists.txt` tests | No change expected. | `ctest -N` count should remain unchanged. |
| `Makefile` `TEST_SRCS` | No change expected. | Makefile test membership should remain unchanged. |
| Public install/export metadata | No change expected. | No package/install validation required unless implementation unexpectedly touches those files. |

## Consumer Impact Plan

| Consumer path | Expected impact | Proof need |
|---|---|---|
| Grow-m backend in `src/sparse_eigs.c` | Calls unchanged private functions from a different source owner. | Compile/link plus focused grow-m value/vector publication tests. |
| Thick-restart backend in `src/sparse_eigs_thick_restart.c` | Calls unchanged private functions from a different source owner. | Compile/link plus thick-restart vector-publication and parity tests. |
| LOBPCG backend in `src/sparse_eigs_lobpcg.c` | Calls unchanged selection helper from a different source owner. | Compile/link plus LOBPCG selection/nearest-sigma/parity tests. |
| Shift-invert flow | Selection and lift behavior still used after shift-invert operator and eigenvalue conversion. | Focused nearest-sigma and shift-invert vector-publication tests. |
| Repeated-handle/workspace paths | No workspace contract change expected. | Focused public handle grow-m/thick-restart/LOBPCG reuse tests if implementation touches anything beyond helper bodies. |

## Public API And Claim Impact

| Surface | Impact |
|---|---|
| `include/sparse_eigs.h` | None. |
| README/support docs | None. |
| Package/ABI docs | None. |
| Benchmark/performance docs | None. |
| Public eigensolver claims | None; this is an internal maintainability move only. |

Preserved non-claims:

- no ARPACK parity claim;
- no SciPy/LAPACK eigensolver parity claim;
- no broad nonsymmetric eigensolver support claim;
- no state-of-the-art eigensolver replacement claim;
- no portable performance claim from this movement.

## Rollback And Partial-Move Handling

| Scenario | Rollback or stop action |
|---|---|
| New source file fails to compile | Move helper bodies back to `src/sparse_eigs.c`, remove new source from `Makefile` and CMake, keep declarations unchanged. |
| CMake or Makefile source membership fails | Revert source-list edits and helper movement together; do not leave duplicate or missing definitions. |
| Focused grow-m or thick-restart vector publication fails | Revert the movement; behavior drift is not acceptable for this batch. |
| LOBPCG selection/nearest-sigma parity fails | Revert the movement; do not fork selection behavior by backend. |
| CTest count changes unexpectedly | Stop and investigate before continuing; movement should not change test registration. |
| Public API or docs change appears necessary | Stop and defer; Day 4 design requires no public API/docs change. |

Partial movement is not allowed for this batch. `s20_select_indices` and
`s20_lift_ritz_vectors` should move together or both stay in `src/sparse_eigs.c`.

## Day 5 Proof Design Requirements

Before Day 6 implementation, Day 5 should record:

1. focused tests covering grow-m value selection;
2. focused tests covering grow-m vector lift and partial vector publication;
3. focused tests covering thick-restart vector publication;
4. focused tests covering LOBPCG selection and backend parity;
5. focused tests covering shift-invert nearest-sigma selection and vector
   publication;
6. expected CTest count before implementation;
7. exact quality commands required when `.c`/`.h` and build metadata change.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 2 design is complete for the first movement batch. | Complete. |
| Exact old/new file plan is recorded. | Complete. |
| Internal header contract is recorded. | Complete. |
| Source-list and CMake impacts are explicit. | Complete. |
| Public API and public-claim impact is recorded. | Complete. |
| Rollback and partial-move handling is recorded. | Complete. |
| Movement can proceed or stop from a documented design gate. | Complete. |
