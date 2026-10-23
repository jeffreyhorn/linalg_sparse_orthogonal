# Day 9: Allocation-Site Failure Map

**State:** Matmul's output shell and node slabs now participate in the
existing checked-allocation fault hook. Other `sparse_create` and pool
callers retain their prior allocation sequence. The focused gate exercises
every distinct output allocation site and a later slab failure after
partially linked output.

## Matmul Allocation Sequence

The counter below is the `sparse_alloc_test_fail_after` value set immediately
before `sparse_matmul` on already-built inputs. Each site is reached in this
order for a nonempty product; failed calls return `SPARSE_ERR_ALLOC` and leave
`*C == NULL`. The result is private until the success assignment.

| Counter | Allocation site | Coverage |
| --- | --- | --- |
| 0 | Result `SparseMatrix` shell | Stale-output fixture. |
| 1 | Result row headers | Stale-output fixture. |
| 2 | Result column headers | Stale-output fixture. |
| 3 | Result row permutation | Stale-output fixture. |
| 4 | Result inverse row permutation | Stale-output fixture. |
| 5 | Result column permutation | Stale-output fixture. |
| 6 | Result inverse column permutation | Stale-output fixture. |
| 7 | Dense row accumulator | Existing retry and stale-output fixtures, updated index. |
| 8 | Nonzero flags | Existing retry and stale-output fixtures, updated index. |
| 9 | Touched-column list | Existing retry and stale-output fixtures, updated index. |
| 10 | Column-tail workspace | Existing retry and stale-output fixtures, updated index. |
| 11 | First output-node slab | Stale-output fixture. |
| 12 on the default 4096-node slab build | Second output-node slab | Two-row partial-link fixture. |

Node emission itself takes a node from the current slab and performs no
additional heap allocation until the slab fills. The append helper checks
`nnz` before requesting a node; row and column pointers are linked only
after a node is acquired. On failure, the private result and its linked
slabs are freed by the single matmul cleanup path. Checked workspace arrays
are freed on the same path.

The later-slab fixture uses two rows and `NODES_PER_SLAB / 2 + 2` columns on
the default build. The first slab fills after all 2050 nodes of row 0 and
2046 nodes of row 1. Failure at row 1, column 2046 therefore occurs after
earlier nodes in the active row and column have linked successfully. The
fixture derives the next slab boundary from the configured slab size; a
separate three-node-slab build also passes. Each failure case checks the
error, null output, original stale matrix ownership, unchanged input values
and `nnz`, followed by a successful uninjected retry. The large fixture
also checks the rebuilt output's column links.

## Verification

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; gate registration passed; 39 tests, 0 failed/skipped, 48453 assertions. |
| Default CMake `test_matmul` build and focused CTest | Both exit 0; 1/1 passed. |
| CMake configure/build with `-DSPARSE_NODES_PER_SLAB=3`, then focused CTest | All exit 0; 1/1 passed. |
| `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. |
| `make lint` | Exit 0; strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; final output `All tests passed.` |

Day 10 owns the separate rollback/retry and instrumentation audit. This
record closes Day 9's allocation-site map; it does not claim that a failed
call frees caller-owned stale output storage.
