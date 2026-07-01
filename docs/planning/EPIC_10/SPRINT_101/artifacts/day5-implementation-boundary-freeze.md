# Sprint 101 Day 5 Implementation Boundary Freeze

## Purpose

Day 5 freezes the implementation boundary for the first compressed-first
constructor/import batch. It converts the Day 4 design into a file-level plan,
focused validation matrix, rollback strategy, and quality gate list before any
source, header, test, example, or public documentation edits are made.

## Frozen Implementation Goal

Make the current CSR/CSC constructor surface visibly and testably serve as the
compressed-first front door:

1. clarify simple versus diagnostic constructor roles;
2. strengthen diagnostic and ownership coverage for CSR/CSC imports;
3. prove CSR/CSC-built matrices can enter representative solver workflows;
4. preserve all existing ABI and mutable-shell compatibility;
5. avoid broad direct CSR/CSC solver APIs.

## File-Level Implementation Plan

| phase | file | planned change | owner rationale |
|---|---|---|---|
| Batch 1 header contract | `include/sparse_csr.h` | tighten comments for `sparse_create_from_csr/csc` and `sparse_from_csr/csc` so simple and diagnostic front-door roles are explicit | public API contract belongs next to declarations |
| Batch 1 diagnostic tests | `tests/test_csr.c` | add focused bad-input tests for CSR and CSC diagnostic constructors | existing suite already owns CSR/CSC conversion and constructor coverage |
| Batch 1 ownership tests | `tests/test_csr.c` | prove constructor success does not mutate caller-owned CSR/CSC arrays and returned matrix owns independent state | validates Day 4 copy/build ownership contract |
| Batch 1 solver smoke proof | `tests/test_csr.c` or a small existing solver suite | add bounded proof that CSR/CSC-built matrices enter one or two representative solver paths | supports compressed-input workflow without claiming broad solver parity |
| Implementation evidence | `docs/planning/EPIC_10/SPRINT_101/artifacts/day6-constructor-import-batch1.md` | record changed files, behavior, focused checks, full checks, and deferred scope | keeps Sprint 101 evidence trail current |
| Later docs/example work | `README.md`, `docs/tutorial.md`, `examples/README.md`, optional focused example | wait until implementation proof exists | prevents public narrative from outrunning validated behavior |

No Day 6 change should touch build-system, packaging, CMake, benchmarks,
workflow files, or broad solver-family APIs.

## Selected Day 6 Test Cases

| case | expected behavior |
|---|---|
| CSR diagnostic constructor null input | `sparse_from_csr(NULL, &mat)` returns `SPARSE_ERR_NULL` and leaves output unusable/null |
| CSR diagnostic constructor null output pointer | `sparse_from_csr(&csr, NULL)` returns `SPARSE_ERR_NULL` |
| CSR invalid shape or `nnz` | negative rows, cols, or `nnz` returns `SPARSE_ERR_BADARG` |
| CSR invalid pointer arrays | missing `row_ptr`, or missing `col_idx`/`values` with nonzero `nnz`, returns `SPARSE_ERR_BADARG` |
| CSR pointer monotonicity/end | row pointers must start at zero, be monotonic, stay in range, and end at `nnz` |
| CSR invalid index | out-of-range column returns `SPARSE_ERR_BADARG` |
| CSR unsorted or duplicate entry | non-strict per-row column order returns `SPARSE_ERR_BADARG` |
| CSC diagnostic constructor mirrors CSR cases | same coverage for `SparseCsc` using column pointers and row indices |
| simple constructors mirror diagnostic failures | `sparse_create_from_csr/csc` returns `NULL` for representative invalid structures |
| ownership independence | caller mutating CSR/CSC arrays after successful construction does not alter returned `SparseMatrix` |
| solver smoke | a CSR/CSC-built matrix can enter a representative one-shot direct or iterative solve path with expected result |

The solver smoke should be intentionally narrow. A single small direct solve
and, if cheap, one iterative matvec/solve path is enough. It must not be named
or documented as full compressed solver parity.

## Compatibility Contract

| behavior | required outcome |
|---|---|
| existing `sparse_create`, `sparse_insert`, `sparse_remove`, `sparse_set` users | unchanged |
| existing `sparse_to_csr/csc` users | unchanged export ownership and free rules |
| existing `sparse_create_from_csr/csc` callers | same signatures and simple `NULL` failure behavior |
| existing `sparse_from_csr/csc` callers | same signatures and `sparse_err_t` status behavior |
| existing solver callers | unchanged `SparseMatrix`-first public solver entry model |
| lower-level `sparse_lu_csr.h` callers | unchanged expert working-format surface |

## Rollback Notes

| change type | rollback action |
|---|---|
| header wording only | revert comment block without ABI impact |
| added tests reveal existing behavior mismatch | stop and ask; do not weaken the Day 4 contract silently |
| solver smoke flakes or becomes too broad | remove smoke from Batch 1 and record as Day 12 regression candidate |
| internal build-path optimization becomes tempting | defer unless it is isolated, measurable, and covered by the full quality chain |
| formatting/lint fallout from touched C/H files | fix locally, rerun full required chain |

## Focused Validation Plan

Run focused checks before the full required chain when Day 6 touches code,
headers, or tests:

```bash
make build/test_csr
./build/test_csr
```

If a solver smoke lands outside `tests/test_csr.c`, also run the touched
suite's focused executable.

If Day 6 only edits planning documentation, run:

```bash
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_101
```

## Required Quality Gate

If Day 6 modifies any `.c` or `.h` file, including tests or public headers,
the required gate is:

```bash
make format && make lint && make test
```

All three commands must pass before proceeding.

If Day 6 changes only markdown/planning documentation, the full quality chain
is not required, but documentation hygiene still must pass.

## Explicit Non-Goals for Batch 1

- no new direct CSR/CSC solver API;
- no ABI rename or replacement of `sparse_from_csr/csc`;
- no adopt/no-copy constructor;
- no Matrix Market compressed-object publication;
- no default promotion of `lu_csr_factor_solve`;
- no broad solver-family parity claim;
- no performance claim from constructor wording or smoke tests alone.

## Day 6 Entry Criteria

Day 6 can start implementation when:

- edits are limited to the frozen file set unless a blocker is found;
- the selected tests can be expressed as focused additions, not broad suite
  rewrites;
- any `.c`/`.h` edit is paired with the full required quality chain;
- any unclear behavior or quality failure stops the sprint for user input.

## Day 5 Conclusion

The implementation boundary is intentionally small. Batch 1 should refine the
existing compressed constructor contract and prove it with targeted tests. The
front-door claim remains "CSR/CSC data can enter the public matrix-shell
workflow cleanly," not "every solver accepts CSR/CSC directly."
