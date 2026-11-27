# Sprint 228 Day 9: Allocation and Numeric Failure Evidence

## Outcome

Every allocation reached by `sparse_svd_lowrank_sparse()` in economy mode is
now routed through the internal checked allocation hook. Exhaustive public-call
sweeps cover the maintained square route, the wide transpose prelude, the
all-zero shortcut, and the retained exact-`off` compatibility route. Every
injected failure returns `SPARSE_ERR_ALLOC`, leaves the output pointer NULL,
preserves the input matrix exactly, destroys partial factors and sparse nodes,
and permits an identical clean retry.

No public declaration or ABI layout changed. The new entry points are private
implementation surfaces used to make solver-owned transpose, matrix-shell,
and node construction observable to the existing test allocator.

## Injectable ownership chain

The complete economy low-rank call now uses checked allocation for:

1. wide-input transpose entries, shell, six header/permutation arrays, row and
   column tails, and node slabs;
2. bidiagonal dense workspace, diagonal and superdiagonal storage, reflector
   pointer/beta arrays, Householder scratch, and every left/right reflector;
3. bidiagonal QR copies, economy U/V work factors, wide extraction scratch,
   and published Vt;
4. maintained output shell, six header/permutation arrays, and node slabs;
5. all-zero output shell and its six arrays; and
6. exact-`off` output shell/arrays, dense compatibility accumulator, and node
   slabs.

Full-mode U/V padding and unrelated dense SVD applications are not reachable:
the sparse low-rank API fixes `compute_uv=1` and `economy=1`. They are therefore
outside this public-call sweep rather than silently counted as covered.

## Exhaustive allocation matrix

`test_s228_public_lowrank_exhaustive_allocation_sweeps` increments the failure
index from zero until the first successful call. Thus every lower index names
one reached allocation and is required to fail. It then runs both one index
beyond the final allocation and a clean post-sweep retry.

| Public path | Shape/rank | Failing indices | First success | Additional proofs |
| --- | --- | ---: | ---: | --- |
| Maintained default | 3x3, rank 2 | 0–25 | 26 | baseline equality, sorted output, unchanged input, clean retry |
| Maintained wide | 2x3, rank 2 | 0–36 | 37 | includes transpose entries/shell/tails/slab and wide factor extraction |
| All-zero shortcut | 2x2, rank 1 | 0–22 | 23 | empty valid output and no node ownership |
| Exact-`off` compatibility | 3x3, rank 2 | 0–26 | 27 | includes dense accumulator and output slab |

For every row, setting the hook to `first_success + 1` also succeeds and
matches the uninjected baseline. This proves that the sweep crossed the last
reachable allocation rather than stopping at an intermediate success.

The smaller Day 8 direct-factor sweep remains useful ownership localization:
its indices 0–7 are the output shell, six arrays, and first slab, with success
at index 8.

## Argument and numeric failure matrix

| Condition | Expected status/output | Evidence |
| --- | --- | --- |
| rank zero or greater than `min(m,n)` | `SPARSE_ERR_BADARG`, NULL | public Day 9 corpus |
| finite negative tolerance | success using default threshold | public Day 9 corpus and frozen contract |
| NaN, positive infinity, or negative infinity tolerance | `SPARSE_ERR_BADARG`, NULL before decomposition | public Day 9 corpus |
| non-finite matrix value with valid selector | `SPARSE_ERR_NUMERIC`, NULL | public Day 9 corpus |
| impossible factor byte extent/dimension product | `SPARSE_ERR_ALLOC`, NULL before construction | retained checked-dimension regression |
| non-finite sigma or factor | `SPARSE_ERR_NUMERIC`, NULL | retained direct-factor regression |
| scaled product, term, or running-sum overflow | `SPARSE_ERR_NUMERIC`, NULL | retained product/sum overflow regressions |
| numeric failure after a node was inserted | `SPARSE_ERR_NUMERIC`, partial output destroyed | retained partial-output regression and successful retry |

The argument corpus also rechecks the original matrix byte-for-value view after
all failures. The allocation corpus rechecks dimensions, `nnz`, and every cell
after every injected index.

## Sanitizer evidence

The focused SVD binary was rebuilt in the isolated
`build/sprint228-day9-sanitize` tree with Homebrew Clang 22.1.5,
AddressSanitizer, UndefinedBehaviorSanitizer, frame pointers, and leak
detection enabled. All 133 tests and 19,219 assertions passed with no ASan,
LeakSanitizer, or UBSan report. This includes every allocation failure, partial
transpose, partial factor, partial output, one-beyond, and retry case above.

## Day 10 handoff

Day 10 should retain these exhaustive allocation tests and concentrate on
mixed successful/numeric/allocation/default/compatibility call sequences,
repeated destruction and reuse, zero/high-threshold outputs, and a final
contract-to-test ownership map.
