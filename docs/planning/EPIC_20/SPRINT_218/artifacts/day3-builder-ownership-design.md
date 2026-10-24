# Day 3: Matmul Output Builder Design

**Status:** Design only. No matmul kernel or test source changed on Day 3.
**Inputs:** [Day 1 contract/cost inventory](day1-matmul-contract-cost-inventory.md)
and [Day 2 pinned baseline](day2-matmul-baseline.md).

## Decision

Use **direct ordered append** to the private `SparseMatrix *out`, with one
`Node *row_tail` for the current row and a checked, zero-initialized
`Node **col_tails` array of length `B->cols`. Keep the existing dense
accumulator, touched-column flags/list, traversal order, column sort, and
`fabs(acc[col]) >= 1e-15` predicate. Allocate exactly one pool node per
retained result entry using `sparse_matrix_make_node`.

| Candidate | Advantages | Costs / decision |
| --- | --- | --- |
| Direct row/column tails | Reuses `SparseMatrix` pool and shell; links each node once; only an O(n) tail array is added. | Requires a private append path and disciplined partial-build cleanup. **Chosen.** |
| Temporary CSR/coordinate output then convert | Separates numerical accumulation from final linking; can count exact nnz before allocating nodes. | Adds O(z) temporary columns/values (and possibly row offsets), another pass and conversion failure phase; the current `sparse_matrix_build_from_entries` requires materializing `SparseBuildEntry[z]`, and its last-write duplicate/exact-zero rules cannot receive raw matmul contributions unchanged. Not needed for sorted touched columns. |

The internal sorted-entry builder in `src/sparse_matrix_build_internal.c`
already uses row/column tails correctly, but it owns a sorted entry array and
different duplicate/drop semantics. Reuse its **link invariant**, not its
entry-materialization API. No public header or ABI change is proposed.

## Ordered-Link Invariants

Let `m=A->rows`, `n=B->cols`, and process physical output rows `i=0..m-1`.
The current `nz_flag` makes `touched[0..ntouched)` unique within each row.
Sorting it ascending means retained columns are strictly increasing. Reset
`row_tail=NULL` at the start of every row; leave the row header NULL when
there are no retained entries. `col_tails[j]` persists across rows and points
to the last node appended in physical column `j`, if any.

For each retained `(i,j,value)`, after validating capacity, create a node
with `right=down=NULL`. Append through `row_tail` (or set
`out->row_headers[i]`), then append through `col_tails[j]` (or set
`out->col_headers[j]`), update both tails, and increment `out->nnz` once.
There are no allocations or error-producing calls between the two links.
Only after both links exist is that node part of a complete output entry.
Since each row visits columns in strictly increasing order, each row chain
is sorted. Since rows are visited in increasing order, each column chain is
sorted. The last node of every chain remains NULL-terminated; no head scan
or duplicate search is needed for a result entry.

`out` is a newly created matrix with identity row/column permutations,
unfactored state, and invalid cached norm. The append path must not alter
those fields except `nnz`; no result permutation is inferred from either
input. `A` and `B` are read by physical row/column index exactly as today.
They may be the same matrix object, but `out` and its pool are distinct, and
neither input is mutated. The API excludes non-identity input permutations;
this optimization does not add a new check or reinterpret them. The caller's
`*C` is cleared at entry as before and receives `out` only on full success.
The caller must still retain its own reference to any old `*C` allocation;
matmul has never freed it.

The numerical path remains unchanged: contributions are accumulated in the
same A-row/B-row order, a touched column is emitted only if the existing
predicate is true, and each touched accumulator/flag is reset even when the
value is dropped. Thus exact cancellation and sub-threshold values drop;
NaN fails the predicate and infinity passes it, matching current code rather
than introducing a new numeric policy.

## Ownership And Rollback

```text
caller: *C = NULL
  |
  +-- sparse_create(m,n) -> out owns shell buffers, identity perms, pool
  |
  +-- allocate acc[n], nz_flag[n], touched[n], col_tails[n]
  |      each temporary has local function ownership
  |
  +-- row loop -> sparse_matrix_make_node(out, ...)
  |      successful nodes immediately belong to out->pool
  |      complete row+column links are installed before nnz increments
  |
  +-- success: free four temporaries, publish *C = out
  |
  +-- any failure: free non-NULL temporaries, sparse_free(out), return error
                   *C stays NULL; inputs are untouched
```

The tail array stores **non-owning** pointers into `out->pool`; freeing it
never frees nodes. Likewise, the row tail is one non-owning local pointer.
`sparse_free(out)` releases every slab, including nodes linked before an
allocation failure, so rollback needs no per-node unlink or `pool_release`.
If a new slab allocation fails, `sparse_matrix_make_node` returns NULL
before a node is linked or `nnz` changes. `sparse_create` failure is handled
before temporaries exist. Any checked workspace/tail allocation failure
frees all prior temporaries and `out`. Empty results use the same path; no
special partial publication occurs.

| Exit point | Owned live resources at that point | Required action |
| --- | --- | --- |
| NULL `C`, NULL input, or shape mismatch | None created by matmul | Preserve existing error precedence; for non-NULL `C`, leave `*C=NULL`. |
| `sparse_create` fails | Its constructor cleans partial shell storage | Return allocation error; no temporary cleanup. |
| Any of four checked array allocations fails | `out` and any earlier arrays | Free non-NULL arrays, `sparse_free(out)`, return allocation error. |
| `nnz` capacity guard or pool-slab allocation fails during flush | `out` owns every previously emitted node; four arrays remain local | Free arrays, `sparse_free(out)`, return allocation error. |
| All rows complete | `out` owns complete row/column links; four arrays remain local | Free arrays, publish `*C=out`, return success. |

Use `sparse_calloc_idx_array(n, sizeof(*col_tails), ...)` so dimensions and
byte multiplication are checked and the existing allocation-failure hook can
exercise the new site. Retain the existing checked `acc`, `nz_flag`, and
`touched` allocations. Before incrementing `out->nnz`, guard its `idx_t`
limit (`IDX_MAX`) and return an allocation/capacity error rather than wrap;
the output must still roll back. This guard is a defensive large-shape
requirement for Days 4-5, not a measured-path claim. Review failure-position
tests after adding the tail allocation because the checked allocation order
changes. Raw shell and pool-slab `malloc` failures remain separately owned
by `sparse_create`/`pool_alloc` and are not injected by the checked helper.

**Staging:** Day 4 can introduce ordered row append while preserving correct
column links through the old column search; Day 5 replaces that search with
the persistent column tails. No intermediate commit should publish a matrix
with row-only links. The final builder has no per-entry head scan in either
direction.

## Complexity And Memory Model

For each output row `i`, let `T_i` be the number of A-by-B contributions,
`t_i` the number of distinct touched columns, and `z_i` the number retained
after the threshold. Let `T=sum T_i`, `z=sum z_i`, `S` be
`SPARSE_NODES_PER_SLAB`, and `n=B->cols`.

| Phase | Final builder time | Extra live memory (excluding A/B) |
| --- | --- | --- |
| Create output shell | O(m+n) | O(m+n) headers/permutations. |
| Accumulate/touch | O(T) expected direct-index work | `n` doubles + `n` ints + `n` indices. |
| Sort touched columns | Inherited libc `qsort`; typically O(sum `t_i log t_i`), with implementation-specific worst case | `Q(max_i t_i)` implementation-dependent auxiliary memory; no builder-owned O(z) copy. |
| Drop/reset and ordered linking | O(sum `t_i` + z) | `n` non-owning column-tail pointers + one row-tail pointer. |
| Output storage | O(z) node operations | `ceil(z/S)` slabs of `S` nodes (zero slabs when z=0), plus shell. |

The builder-controlled live memory is therefore O(m+n+S ceil(z/S)) for
output plus O(n) workspace, excluding inputs and allocator overhead. Total
process peak also includes `Q(max_i t_i)` from libc `qsort`; the C API does
not specify that implementation's auxiliary-memory bound, and
`max_i t_i <= n`. Concretely,
the four length-`n` arrays cost
`n*(sizeof(sparse_scalar_t)+sizeof(int)+sizeof(idx_t)+sizeof(Node *))`
bytes before allocator overhead. The new cost over today's workspace is
`n*sizeof(Node *)`; no O(z) staging buffer is introduced. Current
`sparse_insert` adds head-scan work proportional to
`sum_i z_i(z_i-1)/2 + sum_j c_j(c_j-1)/2` in sorted output, where `c_j`
is retained column length. The final append path removes both sums.

| Pinned fixture | T | z | Current sorted head-scan comparisons | Final link work | Added tail bytes on 64-bit host |
| --- | ---: | ---: | ---: | ---: | ---: |
| `banded` (`m=n=256`) | 766 | 766 | 1,528 (764 row + 764 column) | 766 appends | 2,048 |
| `wide-row` (`m=1,n=1024`) | 1,024 | 1,024 | 523,776 row + 0 column | 1,024 appends | 8,192 |
| `rectangular` (`m=64,n=96`) | 196,608 | 6,144 | 485,376 (291,840 row + 193,536 column) | 6,144 appends | 768 |

These counts isolate **link searching**, not total wall time. The 4096-node
default slab means one slab for banded/wide-row and two for rectangular;
slab slack and input setup are visible in process RSS. Day 2's RSS is a
whole-process peak, not a direct measurement of this workspace formula.
Days 11-12 must compare the same fixture/build configuration and retain raw
sample variation rather than turning the model into an asserted speedup.
