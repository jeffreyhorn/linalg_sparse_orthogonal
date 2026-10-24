# Day 1: Matmul Contract And Cost Inventory

**Date:** 2026-10-23
**Scope:** Sprint 218 item 218.1, first 10 of 20 planned hours
**Source revision:** `271bd3b704326296332cb397385ed458436ca0b7`

This is a pre-change inventory. It records current code behavior and existing
test evidence; it is not an approval of undocumented numerical behavior or a
performance measurement. Day 2 owns pinned fixtures and timing/memory data.

## Output Construction And Cost

`sparse_matmul` in `src/sparse_matrix.c:794-877` creates a result matrix, then
allocates three `B->cols`-length workspaces: `acc`, `nz_flag`, and `touched`.
For each physical row `i` of `A`, it walks its linked entries in column order.
For each `A(i,j)`, it walks physical row `j` of `B` in column order and adds
`A(i,j) * B(j,col)` into `acc[col]`. `nz_flag` adds each first-seen column to
`touched` once, even when multiple paths contribute to that coordinate.
After accumulation, `qsort` orders touched columns; the flush calls
`sparse_insert(out, i, col, acc[col])` for each retained value. It then resets
the accumulator and flag for every touched column. The result is published
through `*C` only after all rows succeed.

| Stage | Current location | Work/ownership consequence |
| --- | --- | --- |
| Input row walks | `src/sparse_matrix.c:829-845` | Traverses `A` and selected `B` row chains; no input mutation. |
| Touched-column sort | `src/sparse_matrix.c:847-852` | `O(t_i log t_i)` for `t_i` touched columns in row `i`. |
| Flush loop | `src/sparse_matrix.c:854-869` | Up to `r_i` retained values per row, each passed to `sparse_insert`. |
| Row head scan | `src/sparse_matrix.c:362-368` | Every insert begins at `out->row_headers[i]`; ascending insertion still scans all earlier row nodes, giving `Theta(r_i^2)` comparisons for a wide retained row. |
| Column head scan | `src/sparse_matrix.c:394-405` | Every insert begins at `out->col_headers[col]`; earlier result rows are scanned again for a populated column. |
| Node allocation and links | `src/sparse_matrix.c:97-106,378-407` | `pool_alloc` owns the node slab; each node is linked into both sorted chains and increments `out->nnz`. |

`Node` has physical row/column, value, `right` (ascending column within a row),
and `down` (ascending row within a column) in
`src/sparse_matrix_internal.h:128-145`. The output shell owns both header arrays,
four permutation arrays, and the node pool. `sparse_free` frees all pool slabs,
arrays, factor metadata, and the shell (`src/sparse_matrix.c:231-242`). Current
output has identity permutations, `nnz` updated per inserted node, an invalid
norm cache (`-1.0`), and no factorization state after `sparse_create`.

There is already a tail-based bulk constructor in
`src/sparse_matrix_build_internal.c:24-110`: it links sorted entries through
`row_tails` and `col_tails` without per-node head scans. Reuse is a Day 3
design question, not a Day 1 decision. It requires a full entry array,
coalesces duplicate coordinates by taking the last entry, and skips only
exact zero, while matmul accumulates contributions and drops values below
`1e-15`. Passing matmul contributions directly to it would change semantics.

## Observed Contract

The public declaration at `include/sparse_matrix.h:446-466` documents
physical-index multiplication and excludes non-identity input permutations.
The table distinguishes tested behavior from behavior inferred by reading the
current implementation.

| Topic | Current behavior | Evidence / remaining check |
| --- | --- | --- |
| Shape and errors | `C == NULL` returns `SPARSE_ERR_NULL` before input inspection; with non-NULL `C`, `*C` is cleared first, then NULL inputs return `SPARSE_ERR_NULL`, dimension mismatch returns `SPARSE_ERR_SHAPE`, and allocation failure returns `SPARSE_ERR_ALLOC`. | `src/sparse_matrix.c:794-822`; error-precedence/stale-output tests at `tests/test_matmul.c:437-470`. |
| Output ownership | Success returns a fresh caller-owned matrix. Failure leaves `*C == NULL`; an old pointer formerly held by `*C` is not freed. | Header contract and `tests/test_matmul.c:388-415`. |
| Input space | Direct physical row headers and physical columns are used. `sparse_get`/matvec apply permutations, but matmul does not. | Header note; source at `src/sparse_matrix.c:826-845`. Permuted-input rejection is documented usage, not validated at runtime. |
| Output order | Each touched column is sorted before flush; `sparse_insert` keeps rows sorted by column and columns sorted by row. Result permutations are identity. | Source at `src/sparse_matrix.c:847-875`, `347-408`, `178-228`; current matmul tests mostly use point lookups, not full structural traversal. |
| Duplicate contributions | Multiple products for one `(i,col)` sum in `acc[col]` in `A`-then-`B` traversal order; one touched index is flushed. Existing input coordinates are unique/update-on-insert. | Source at `src/sparse_matrix.c:829-845`; rectangular and cancellation cases exercise repeated contributions, but no independent dense oracle. |
| Zero dropping | A touched result is stored only if `fabs(acc[col]) >= 1e-15`; equality is retained. Exact/near cancellation below threshold is omitted. | Source at `src/sparse_matrix.c:854-868`; exact cancellation has a regression, boundary values do not. |
| Non-finite arithmetic | No explicit finite guard or numeric error exists in matmul. `fabs(NAN) >= 1e-15` is false, so a NaN accumulator is dropped; infinity passes the flush. Intermediate overflow can therefore yield stored infinity or later NaN. | Code inference only; existing matmul tests do not pin NaN, infinity, overflow, or signed-zero behavior. Freeze with fixtures before changing this behavior. |
| Empty result | Positive-dimension matrices with zero stored entries return a fresh result with `nnz == 0`. `sparse_create` rejects nonpositive dimensions, so zero-dimension products are not represented by this API. | `tests/test_matmul.c:239-253`; `src/sparse_matrix.c:178-180`. |

## Allocation And Rollback Map

| Site | Allocation route | Current owner / failure cleanup | Injection coverage |
| --- | --- | --- | --- |
| Output shell | Raw `malloc` in `sparse_create` (`src/sparse_matrix.c:182`) | `sparse_create` returns NULL; matmul reports `SPARSE_ERR_ALLOC`. | Not covered by `sparse_alloc_test_fail_after`. |
| Output row headers, column headers, row/inverse-row and column/inverse-column permutations (six arrays) | Checked array helpers in `sparse_matrix_alloc_shell_buffers` (`src/sparse_matrix.c:145-153`) | Shell constructor frees partially allocated buffers, then shell. | Counted by fault injector; no matmul test for each shell position. |
| `acc` (`B->cols` doubles) | `sparse_calloc_idx_array` (`src/sparse_matrix.c:816`) | Matmul frees workspaces already allocated and `out`. | `fail_after(6)` in `tests/test_matmul.c:302-384`. |
| `nz_flag` (`B->cols` ints) | `sparse_calloc_idx_array` (`src/sparse_matrix.c:817`) | Same cleanup path. | `fail_after(7)` in the same suite. |
| `touched` (`B->cols` indices) | `sparse_malloc_idx_array` (`src/sparse_matrix.c:818`) | Same cleanup path. | `fail_after(8)` in the same suite. |
| Output node slabs | Raw `malloc` in `pool_alloc` (`src/sparse_matrix.c:29-44`) via `sparse_matrix_make_node` | Failed insert returns `SPARSE_ERR_ALLOC`; matmul frees workspaces and the partially built matrix; `sparse_free` releases all slabs. | Not covered by this fault injector or current matmul failure cases. |

`sparse_malloc_idx_array` and `sparse_calloc_idx_array` check index-to-size
conversion and multiplication overflow before allocation
(`src/sparse_alloc_internal.c:21-81`). The test injector intercepts these
helper calls but not raw shell/slab `malloc`. Existing `fail_after(6..8)`
assumes six successful shell-buffer helper allocations, so builder changes
will require remapping the failure positions rather than keeping magic counts.
There is no new partial output publication path today: `out` is freed on
workspace or insert failure, and `*C` remains NULL.

## Test And Registration Inventory

`tests/test_matmul.c:586-616` registers 18 cases: five small arithmetic/
identity/diagonal controls, two argument-error controls, rectangular and
cancellation cases, three empty/row-column/single-entry cases, four
allocation/error-precedence cases, a SuiteSparse `nos4` associativity check,
and a Cholesky reconstruction check. The SuiteSparse check compares
`(A*A)*x` with `A*(A*x)`; it is not an independent dense product oracle.
No matmul test currently pins a wide result row, randomized seeded shapes,
column traversal, CSR/CSC conversion of a product, threshold boundaries, or
non-finite arithmetic. `tests/test_csr.c` exercises standalone conversion,
and `src/sparse_csr.c:31-69` traverses row chains, making conversion useful
for later structural validation.

- Make includes `tests/test_matmul.c` in `TEST_SRCS` (`Makefile:128`).
- `make matmul-allocation-failure-gate` runs a registration script and the
  entire `build/test_matmul` binary (`Makefile:303-308`).
- CMake registers `test_matmul` with `matmul;allocation_failure` labels
  (`CMakeLists.txt:227-228`). CTest execution was not run on Day 1.
- The registration script checks literal target/test names, not whether every
  allocation site is exercised
  (`tests/test_matmul_allocation_failure_gate_registration.py`).

## Day 1 Evidence And Day 2 Handoff

On Darwin 24.6.0 x86_64 with Apple clang 17.0.0
(`clang-1700.6.4.2`), `make matmul-allocation-failure-gate` exited 0 at the
source revision above. The registration check passed, and `test_matmul`
reported **18 run, 0 failed, 0 skipped, 185 assertions**. This is a focused
correctness/failure baseline, not a measured matmul runtime: the suite's
`Time: 0.003 s` is framework time, not a controlled kernel measurement, and
excludes the build.
No full `make format && make lint && make test`, CMake/CTest, sanitizer, or
peak-memory/performance measurement was run on Day 1; those have separate
sprint gates.

Day 2 should pin the narrow, wide-row, and rectangular/dense-result fixtures;
record their shapes, result support, exact command/build flags, raw timings,
and peak-memory method before any kernel edit. It should also decide how to
measure the row and column head-scan cost separately, and carry forward the
uncovered node-slab failure and non-finite behavior risks.
