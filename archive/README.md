# Archive: Original Prototype Files

These files are the original monolithic prototypes developed iteratively before
the code was restructured into a proper library. They are preserved here for
reference.

## File Evolution (chronological order)

1. `sparse_lu_orthogonal.c` — Baseline: Node struct, orthogonal linked-list
   sparse matrix, complete-pivoting LU, forward/backward substitution, solveLU.
   Individual malloc/free per node.

2. `sparse_lu_with_pool_and_nnz.c` — Added slab pool allocator
   (NodePool/NodeSlab) and NNZ counting.

3. `sparse_lu_orthogonal_with_pool_nnz_mem_io.c` — Added memory estimator and
   Matrix Market file I/O (save/load).

4. `sparse_lu_orthogonal_with_pool_nnz_mem_io_mm.c` — Cleaner Matrix Market
   I/O, consolidated code.

5. `sparse_lu_orthogonal_complete_pivoring_mm_with_errors.c` — Added error
   codes (ERR_OK, ERR_NULL_PTR, etc.) and return-code error handling.
   Note: filename contains typo "pivoring" (should be "pivoting").

6. `sparse_lu_orthogonal_complete_pivoting_mm_with_unit_tests.c` — Added
   forward declarations, prototype block, macro-based unit test framework, and
   error-path tests. WARNING: computeLU body is stubbed out in this file.

7. `sparse_lu_orthogonal_complete_pivoting_mm_full.c` — Full computeLU
   restored with error codes, unit tests + working example. This was the most
   complete version before restructuring.

`sparse_lu` — Compiled binary from file 7.

## Known Issues in These Prototypes

See `planning/reviews/initial-review.md` for the full review. Key issues:
- Unsafe linked-list traversal during modification in computeLU
- Forward substitution early-break assumption is wrong after pivoting
- No proper library separation (everything in one file with main())
