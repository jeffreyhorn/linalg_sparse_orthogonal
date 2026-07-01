# Sprint 101 Day 6 Constructor and Import Batch 1

## Purpose

Day 6 lands the first bounded compressed-first constructor/import batch from
the Day 5 implementation boundary. The batch clarifies the existing CSR/CSC
constructor contract and strengthens focused tests. It does not add new
solver APIs, adopt/no-copy constructors, Matrix Market compressed-object
publication, or broad solver-parity claims.

## Changed Files

| file | change |
|---|---|
| `include/sparse_csr.h` | clarified simple compressed-first constructor comments and diagnostic constructor comments for CSR/CSC |
| `tests/test_csr.c` | added CSR/CSC bad-input diagnostics, caller-owned array copy checks, and one bounded LU solver-entry smoke test from a CSR-built matrix |
| `docs/planning/EPIC_10/SPRINT_101/WORKING_NOTES.md` | recorded Day 6 actions, validation, and exit state |
| `docs/planning/EPIC_10/SPRINT_101/artifacts/day6-constructor-import-batch1.md` | recorded implementation evidence |

## Implemented Behavior

### Header Contract

The public header now distinguishes:

- `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` as simple
  compressed-first constructors that return a new caller-owned
  `SparseMatrix *` or `NULL`;
- `sparse_from_csr(...)` and `sparse_from_csc(...)` as diagnostic
  compressed-first constructors that return `sparse_err_t`;
- caller-owned CSR/CSC arrays are validated and copied, not adopted or
  modified;
- the returned matrix is an independent public `SparseMatrix` shell.

### Focused Test Coverage

Added focused coverage for:

- null compressed input;
- null output pointer on diagnostic constructors;
- negative dimensions and `nnz`;
- missing pointer arrays;
- nonzero pointer-array start;
- non-monotonic pointer arrays;
- pointer-array end not equal to `nnz`;
- out-of-range row/column indices;
- unsorted and duplicate structural entries;
- simple constructor `NULL` behavior on representative invalid input;
- copy ownership after successful CSR/CSC construction;
- a CSR-built matrix entering a one-shot LU factor/solve path.

## Compatibility Evidence

| compatibility surface | evidence |
|---|---|
| public ABI | no function signatures changed |
| mutable matrix shell | no `sparse_create`, `sparse_insert`, `sparse_remove`, or `sparse_set` behavior changed |
| CSR/CSC export ownership | no `sparse_to_csr/csc` implementation changed |
| simple constructors | retain `NULL` on invalid input or allocation failure |
| diagnostic constructors | retain `sparse_err_t` return semantics |
| solver APIs | no new solver entry points added |

## Focused Validation

```bash
make format
make build/test_csr
./build/test_csr
```

Result:

- `test_csr`: 18 tests, 0 failures, 580 assertions.

## Full Required Quality Gate

Because Day 6 modified `.h` and `.c` test files, the required quality gate was
run:

```bash
make format
make lint
make test
```

Results:

- `make format`: passed.
- `make lint`: first run failed on two cppcheck `intToPointerCast` warnings
  from test sentinel pointer casts; the test was changed to avoid non-portable
  sentinel pointers.
- `make lint`: passed on rerun.
- `make test`: passed.

## Deferred Scope

Still deferred:

- direct CSR/CSC solver entry APIs;
- adopt/no-copy CSR/CSC constructors;
- internal CSR/CSC build-path optimization;
- Matrix Market compressed-object publication;
- promotion of `lu_csr_factor_solve` as the default front door;
- broad solver-family compressed parity claims.

## Day 6 Conclusion

Batch 1 makes the existing CSR/CSC constructors clearer and better proven
without changing ABI or solver entry shape. The compressed-first front-door
claim is now backed by header-local contract wording, diagnostic tests,
ownership tests, and one bounded solver-entry smoke proof.
