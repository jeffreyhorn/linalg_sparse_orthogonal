# Sprint 106 Day 12: Source-List and CMake Parity Reconciliation

## Scope

Day 12 audited the files added or split during Sprint 106 and reconciled their
ownership across Make, CMake, source-list checking, test registration, and
reviewed CI assumptions.

## Extracted File Inventory

| file | role | intended owner |
|---|---|---|
| `src/sparse_ldlt_csc_rowadj.c` | LDLT CSC row-adjacency helpers | library source |
| `src/sparse_qr_householder.c` | QR Householder and sparse-column helpers | library source |
| `src/sparse_qr_internal.h` | private QR internal declarations | private source header |
| `src/sparse_lu_csr_struct.c` | LU CSR growth and structural helpers | library source |
| `src/sparse_lu_csr_internal.h` | private LU CSR internal declarations | private source header |
| `tests/test_graph_fixtures.h` | graph/reorder shared fixtures | test-only helper |
| `tests/test_direct_solver_helpers.h` | direct solver assertion/residual helpers | test-only helper |
| `tests/test_integration_fixtures.h` | integration progress/oracle fixtures | test-only helper |

## Make and CMake Library Ownership

The three new library `.c` owners are present in all required library source
surfaces:

| library source | Makefile `LIB_SRCS` | CMake `add_library` | `library_sources.txt` |
|---|---|---|---|
| `src/sparse_ldlt_csc_rowadj.c` | present | present | present |
| `src/sparse_qr_householder.c` | present | present | present |
| `src/sparse_lu_csr_struct.c` | present | present | present |

The private headers are included only by their owning implementation families:

- `src/sparse_qr_internal.h` is included by `src/sparse_qr.c` and
  `src/sparse_qr_householder.c`.
- `src/sparse_lu_csr_internal.h` is included by `src/sparse_lu_csr.c` and
  `src/sparse_lu_csr_struct.c`.
- `src/sparse_ldlt_csc_internal.h` remains the private contract for
  `src/sparse_ldlt_csc.c`, `src/sparse_ldlt_csc_rowadj.c`, and the existing
  LDLT CSC owners.

No public install/export header surface changed.

## Test Helper Ownership

The new test helper headers are intentionally header-only and are not Make or
CMake target source-list entries:

- `tests/test_graph_fixtures.h` is included by `tests/test_graph.c` and
  `tests/test_reorder_nd.c`.
- `tests/test_direct_solver_helpers.h` is included by `tests/test_lu_csr.c`.
- `tests/test_integration_fixtures.h` is included by
  `tests/test_integration.c`.

No new test executables were introduced, and no existing test registration was
renamed or removed.

## Reviewed Test Surface

The reviewed CMake test surface remains unchanged:

- CMake registered tests: 54
- Makefile test binaries: 54
- test-count parity: passed

The fixture/helper extraction did not introduce accidental unreviewed CTest
lanes.

## Validation

Source-list validation passed:

```sh
python3 scripts/check_library_sources.py
```

Result:

- `source-list-check: PASS (45 library sources)`

Reviewed CMake compile/parity validation passed:

```sh
make quality-review-cmake-compile
```

Result:

- CMake configure passed
- clean CMake rebuild passed
- `ctest -N --test-dir build/quality-review-cmake` reported 54 tests
- Make/CMake test-count parity passed with 54 tests on each surface

Final hygiene validation passed:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_106
```

## Day 12 Conclusion

Every Sprint 106 extracted library source is owned by Make, CMake, and the
source-list checker. Every extracted test helper remains test-only and included
through its owning proof translation unit. Reviewed CTest shape remains exact,
with 54 registered CMake tests matching the Makefile test surface.
