# Day 5 Matmul Enforcement and Sorted-Add Builder Design

## Outcome

`sparse_matmul` now enforces the same frozen original-state contract as `sparse_add` and `sparse_add_inplace`. After output-pointer, input-pointer, and shape validation, both operands must be unfactored and have identity row, inverse-row, column, and inverse-column maps. Rejection returns `SPARSE_ERR_BADARG` before result or workspace allocation and leaves both inputs unchanged.

The public comments for all three APIs now state this precondition and error result. They also state the supported out-of-place alias behavior and the rejected in-place alias behavior. No public function, type, struct field, exported symbol, or index-width behavior changed.

## Validation and regression closure

The formerly experimental nine-case matmul matrix is now always on. It covers left, right, and both operands across row-only, column-only, and combined valid permutations. Additional regressions cover:

- inverse-map-only violations on the left and right;
- factored state on the left and right;
- shape precedence over state validation;
- output clearing before NULL-input rejection;
- state rejection before injectable result allocation;
- unchanged input values, maps, metadata, links, and counts;
- successful retry and original-state output structure after allocation injection.

Existing identity-state tests continue to cover rectangular multiplication, empty patterns, cancellation, thresholds, nonfinite behavior, deterministic ordering, `A*A`, allocation rollback, and dense-oracle agreement. Two Cholesky reconstruction tests previously passed a factor-state matrix directly. They now deliberately copy the physical lower factor into fresh original-state matrices before multiplying, while a dedicated regression verifies direct factor-state rejection.

## Sorted-add builder design for Day 6

### Scope

Introduce one file-local builder in `src/sparse_matrix.c` for monotonically ordered output. Day 6 uses it only in `sparse_add`; matmul's existing append helper remains unchanged until later consolidation is separately justified. The builder does not alter arithmetic, thresholds, public ownership, or the original-state contract.

### Proposed state

The builder owns:

- one partially constructed `SparseMatrix *out`;
- one zero-initialized `Node **col_tails` array of length `out->cols`;
- one `Node *row_tail` for the current row;
- `next_row`, `current_row`, and `last_col` ordering sentinels;
- an active/finished state preventing double transfer or cleanup.

No per-node index, entry staging array, or per-row tail array is required. The add merge already visits rows in ascending order and emits at most one value per column in strictly ascending order.

### Operations

1. **Initialize:** create an injectable original-state output shell and allocate zeroed column tails. Until finish, the builder exclusively owns both.
2. **Begin row:** require the next consecutive row, reset `row_tail`, and reset `last_col`. Empty rows are represented by beginning a row and appending nothing.
3. **Append:** require a valid column strictly greater than `last_col`, reject `nnz == IDX_MAX`, allocate one injectable node, initialize all node fields, append through `row_tail`, append through `col_tails[col]`, then increment `nnz` and advance the ordering sentinel.
4. **Finish:** require every output row to have begun, free `col_tails`, transfer the matrix to the caller, and clear builder ownership.
5. **Abort:** free `col_tails`, call `sparse_free(out)` to release every slab and shell array, and clear ownership. Abort is safe after any successful append and is a no-op after finish.

### Allocation boundaries

The complete injectable sequence is:

1. result matrix shell;
2. row-header array;
3. column-header array;
4. row permutation;
5. inverse-row permutation;
6. column permutation;
7. inverse-column permutation;
8. column-tail workspace;
9. first node slab;
10. each later node slab at the fixed slab boundary.

Every failure before finish returns `SPARSE_ERR_ALLOC`, leaves the caller's output NULL, preserves both inputs, and runs abort. Day 7 must sweep shell/workspace failures, the first node slab, at least one later slab boundary, successful retry, and destroyability.

### Structural invariants

After each successful append:

- the new node owns the exact physical `(row, col, value)` emitted by the merge;
- the current row chain is strictly increasing by column and terminates at `row_tail`;
- each touched column chain is strictly increasing by row and terminates at `col_tails[col]`;
- no coordinate is duplicated because the two-pointer merge emits each selected column once;
- `out->nnz` equals the number of nodes linked from both row and column headers;
- all four output maps remain identity, cache/factor metadata remain in fresh-matrix state, and no input field changes;
- values with magnitude below `1e-15` are never appended, while magnitude exactly `1e-15` is retained.

The append operation initializes the node before publishing either link and increments `nnz` only after both links are installed. Because node allocation is the only failing append step, no rollback is needed after link publication.

### Day 6 acceptance boundary

Day 6 may replace `sparse_insert` only after focused tests inspect row and column ordering, duplicate freedom, counts, maps, metadata, and dense-oracle values for overlapping, disjoint, cancellation, empty-row, rectangular, and same-input merges. Allocation hardening beyond initial representative failures remains Day 7 work. No performance claim follows from the design alone.

## Evidence

| Command or inspection | Result |
| --- | --- |
| Focused CMake build of `test_sparse_arith` and `test_matmul` | Pass |
| Focused CTest selection | 2/2 pass |
| `make format` | Pass |
| `make lint` | Pass: strict compile, clang-tidy, and cppcheck (112/112) |
| `make test` | Pass: full runtime suite, including both migrated Cholesky reconstruction paths |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| Strict standalone compile of all changed test files with `-Werror` | Pass |
| `git diff --check` | Pass |
| Arithmetic contract suite | Complete matmul permutation/state matrix always on |
| Production call-site audit | Only benchmark use; identity-state construction remains compatible |
| Public ABI inspection | Comments only; declarations and layouts unchanged |

