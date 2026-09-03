# Sprint 195 Day 3: Selected Owner Invariant Record

## Purpose

Define the failure-path contract for the selected Sprint 195 owner,
`sparse_symbolic_cholesky()`, before harness or regression-test changes.

## Selected Owner

| Field | Value |
| --- | --- |
| Function | `sparse_symbolic_cholesky()` |
| Implementation | `src/sparse_etree.c` |
| Declaration | `src/sparse_analysis_internal.h` |
| Proof-owner test binary | `tests/test_etree.c` |
| Output object | Caller-provided `sparse_symbolic_t *sym` |
| Cleanup function | `sparse_symbolic_free(sym)` |

This is a symbolic Cholesky out-struct proof only. It is not broad etree,
analysis, LU symbolic, direct-solver, matrix-construction, or all-library
allocation-failure proof.

## Allocation Path Map

| Class | Current allocation site | Hook status | Required proof |
| --- | --- | --- | --- |
| Empty matrix `col_ptr` | `sparse_calloc_array(1, sizeof(idx_t), ...)` | Covered by existing hook. | Failure returns `SPARSE_ERR_ALLOC` with zeroed `sym`. |
| Non-empty `col_ptr` | direct `malloc(col_ptr_bytes)` | Not covered today. | Day 4 should convert to `sparse_malloc_array` or add an equivalent local checkpoint. |
| `row_idx` | `sparse_malloc_idx_array(sym->nnz, ...)` | Covered. | Failure frees `col_ptr` and clears `sym`. |
| Child and marker workspace | `child_head`, `child_next`, `marker`, `tmp` via wrappers | Covered. | Failure frees any earlier workspace plus symbolic output. |
| Column-row workspace | `col_rows`, `col_nrows` via wrappers | Covered. | Failure frees child/marker workspace plus symbolic output. |
| Propagated row sets | per-column `sparse_malloc_array(count, ...)` | Covered. | Failure frees all propagated row sets, workspace, and symbolic output. |

## Publication Contract

`sparse_symbolic_cholesky()` may publish successful output only by returning
`SPARSE_OK`. On failure after argument validation enters the function, the
selected Sprint 195 contract is:

- `sym->col_ptr == NULL`;
- `sym->row_idx == NULL`;
- `sym->n == 0`;
- `sym->nnz == 0`;
- caller-owned `A`, `parent`, `postorder`, and `colcount` remain caller-owned;
- `sparse_symbolic_free(sym)` remains safe after the failed call.

The function currently clears `sym` at entry with `memset` and uses
`sparse_symbolic_free(sym)` on partial-output failures. Day 4/Day 5 work must
preserve that shape.

## Retry Contract

A failed selected-owner call must not poison a later successful call. The
planned retry proof should:

1. force a deterministic allocation failure;
2. reset the private allocation hook before assertions can early-return;
3. verify the failed output is zeroed and free-safe;
4. rerun the same fixture without injection; and
5. compare the successful symbolic structure against existing expected
   `test_etree` fixture values.

The known 5x5 symbolic fixture is the preferred retry fixture because it
exercises nontrivial row propagation and already has explicit expected rows.

## Assertion Map

| Invariant | Planned assertion |
| --- | --- |
| Hooked allocation failures return `SPARSE_ERR_ALLOC`. | Fail-at-count loop or named cases around each allocation class. |
| Stale caller output is cleared on entered failure paths. | Initialize `sym` with stale allocated state, force failure, then check all fields are zero/NULL. |
| Partial `col_ptr` / `row_idx` publication is cleared. | Force failures after each output-publication milestone and assert `sparse_symbolic_free`-safe state. |
| Temporary row-set allocations do not leak into caller state. | Force per-column propagated-row allocation failures and assert caller-visible `sym` is cleared. |
| Retry succeeds after reset. | Rerun selected fixture without injection and compare expected symbolic structure. |
| Focused gate cannot drift. | Add a guard that names selected symbolic allocation-failure tests. |

## Unsupported Breadth

The Sprint 195 selected proof will not cover:

- `sparse_symbolic_lu()` or its optional L/U publication behavior;
- `sparse_analyze()` and `sparse_analysis_t` publication cleanup;
- standalone `sparse_etree_compute()`, `sparse_etree_postorder()`, or
  `sparse_colcount()` failure paths;
- numeric Cholesky, LU, LDLT, QR, SVD, eigensolver, graph, or matrix
  construction allocation paths;
- direct `malloc`/`calloc` allocations outside this selected owner;
- real operating-system OOM behavior;
- concurrent allocation-hook usage.

## Day 4 Handoff

Day 4 should design the harness around the existing private allocation hook.
The main implementation decision is how to make the direct `sym->col_ptr`
allocation deterministic. The preferred path is to replace that direct
`malloc` with `sparse_malloc_array(col_ptr_len, sizeof(idx_t), ...)`, because
that keeps the proof inside the existing private hook model and avoids a new
owner-local global.

## Day 8 Cleanup Proof Update

Day 8 tightened the cleanup invariant from "failed outputs are empty" to
"failed outputs are empty and remain safe after repeated cleanup." The focused
test helper now asserts this sequence for the empty-column allocation failure,
the non-empty `sym->col_ptr` allocation failure, and each selected known-5x5
partial-state allocation class:

1. `sparse_symbolic_cholesky()` returns `SPARSE_ERR_ALLOC`.
2. `sym->col_ptr == NULL`, `sym->row_idx == NULL`, `sym->n == 0`, and
   `sym->nnz == 0`.
3. `sparse_symbolic_free(&sym)` leaves the same empty state.
4. A second `sparse_symbolic_free(&sym)` still leaves the same empty state.

This covers stale-output suppression, free-after-failure safety, and
double-cleanup safety for the selected owner. It does not add a leak counter or
claim OS-level leak detection; the repository exposes deterministic allocation
failure injection for this lane, not per-test allocation accounting.

## Day 9 Retry Proof Update

Day 9 completed the retry contract by adding a known-5x5 symbolic output
oracle and a table-driven retry test. For each selected fail-after checkpoint,
including `sym->col_ptr`, `sym->row_idx`, child workspace, marker/temp
workspace, column-row workspace, and propagated row-set allocation, the test
now proves this sequence:

1. force the selected allocation failure;
2. reset the allocation hook before assertions;
3. verify `SPARSE_ERR_ALLOC`, caller-owned matrix preservation, and
   free-after-failure safety;
4. rerun `sparse_symbolic_cholesky()` with the same matrix, parent, postorder,
   and column-count arrays after reset; and
5. compare the retry output against the same known-5x5 symbolic oracle used by
   the baseline success fixture.

This closes the planned retry invariant for the selected owner without
expanding claims to other etree, analysis, symbolic LU, or direct-solver paths.

## Validation

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

`git diff --check` passes.
