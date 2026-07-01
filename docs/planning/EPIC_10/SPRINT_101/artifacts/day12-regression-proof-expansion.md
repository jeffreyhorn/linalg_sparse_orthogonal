# Sprint 101 Day 12 Regression Proof Expansion

## Purpose

Day 12 completes focused regression proof for compressed-first construction,
ownership, error handling, and solver entry behavior. It reviews the Day 6
constructor tests, the Day 9 public wording batch, and the Day 11 executable
example, then fills the one remaining narrow test gap: CSC-built matrices
entering a representative solver path.

## Coverage Review

| behavior | existing proof before Day 12 | assessment |
|---|---|---|
| CSR simple constructor success | `test_create_from_csr_entry_path` | covered |
| CSC simple constructor success | `test_create_from_csc_entry_path` | covered |
| CSR diagnostic invalid input | `test_csr_diagnostic_constructor_rejects_bad_inputs` | covered |
| CSC diagnostic invalid input | `test_csc_diagnostic_constructor_rejects_bad_inputs` | covered |
| CSR copy ownership | `test_csr_constructor_copies_caller_owned_arrays` | covered |
| CSC copy ownership | `test_csc_constructor_copies_caller_owned_arrays` | covered |
| CSR-built matrix enters solver workflow | `test_compressed_constructed_matrix_enters_lu_solve` | covered |
| CSC-built matrix enters solver workflow | not covered before Day 12 | added |
| public executable compressed-input workflow | `example_compressed_input.c` plus Day 11 validation | covered |

## Added Regression

Day 12 adds `test_csc_constructed_matrix_enters_cholesky_solve` to
`tests/test_csr.c`.

The test:

- builds a symmetric positive definite 2x2 matrix from caller-owned CSC
  arrays with `sparse_create_from_csc(...)`;
- enters the normal public Cholesky factor/solve workflow;
- verifies the expected solution to `[[4, 1], [1, 3]] x = [1, 1]`;
- frees the constructed matrix with `sparse_free(...)`.

This is intentionally narrow. It proves CSC construction can enter a
representative solver workflow without claiming broad direct CSR/CSC solver
entry APIs or compressed parity across every solver family.

## Registration Notes

| surface | implication |
|---|---|
| Make test registration | no change; the new test is inside existing `tests/test_csr.c` |
| CMake test registration | no change; the existing `test_csr` binary remains the registered CTest surface |
| Windows reviewed CTest count | no change expected because no new test executable was added |
| example registration | unchanged from Day 11 |

## Validation Requirements

Day 12 modified a `.c` test file, so validation must include:

```bash
make format
make build/test_csr
./build/test_csr
make lint
make test
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_101 tests/test_csr.c
```

## Validation Results

| command | result |
|---|---|
| `make format` | passed |
| `make build/test_csr` | passed |
| `./build/test_csr` | passed; `test_csr` now runs 19 tests, 0 failures, 585 assertions |
| `make lint` | passed |
| `make test` | passed |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

## Remaining Test Gaps

| gap | status |
|---|---|
| direct CSR/CSC solver APIs | non-goal for Sprint 101 |
| adopt/no-copy constructor behavior | non-goal for Sprint 101 |
| broad compressed solver parity across all solver families | non-goal for Sprint 101 |
| Matrix Market compressed-object publication | deferred beyond Sprint 101 |
| internal CSR/CSC build-path performance optimization | deferred beyond Sprint 101 |

## Day 12 Conclusion

Sprint 101 now has focused regression proof for simple and diagnostic CSR/CSC
construction, invalid input, copy ownership, CSR-to-LU entry, CSC-to-Cholesky
entry, and public executable compressed-input adoption. The proof remains
bounded to the earned compressed-first product model: caller-owned compressed
arrays are copied into a normal caller-owned `SparseMatrix`, then existing
solver APIs apply.
