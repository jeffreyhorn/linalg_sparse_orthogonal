# Day 13 Full Quality Gate & Metrics

## Purpose

Day 13 runs the required full quality gate for Sprint 109 code, test, and
build-system changes, then captures maintainability metrics for the changed
owners before closeout.

## Required Quality Gate

Command:

```sh
make format && make lint && make test
```

Result:

```text
All tests passed.
```

Quality-gate details:

| Step | Result | Evidence |
|---|---|---|
| `make format` | Passed | `clang-format` completed across source, header, test, benchmark, example, and public header surfaces. |
| tooling build | Passed | Bench and example binaries built for lint/tooling without execution. |
| strict syntax check | Passed | `cc -fsyntax-only` with `-Werror` completed for all 46 library sources. |
| `clang-tidy` | Passed | Completed for all 46 library sources. |
| `cppcheck` | Passed | Completed for 101 source/test files with `--error-exitcode=1`. |
| `make test` | Passed | Full test suite completed; final output reported `All tests passed.` |

No quality failure was deferred.

## Changed Owner Metrics

| Owner | Current Lines | Sprint 109 Role |
|---|---:|---|
| `src/sparse_eigs.c` | 1412 | Public eigensolver orchestration after dense Jacobi extraction. |
| `src/sparse_eigs_dense_internal.c` | 129 | Private dense Jacobi implementation owner. |
| `tests/test_qr.c` | 3194 | QR proof-owner test after exact-RHS setup cleanup. |
| `CMakeLists.txt` | 435 | CMake library source registration and reviewed CTest registration owner. |
| `Makefile` | 972 | Make library source registration and quality/test target owner. |
| `build-metadata/library_sources.txt` | 50 | Reviewed library source manifest. |

## Helper and Source-List Metrics

| Metric | Value |
|---|---:|
| Library sources | 46 |
| `s21_dense_sym_jacobi` implementation owners | 1 |
| `s21_dense_sym_jacobi` public/header declarations | 0 |
| `make_qr_exact_rhs` helper definitions | 1 |
| `make_qr_exact_rhs` call sites | 7 |
| New helper targets | 0 |
| CTest registrations from Day 12 | 54 |
| Public/header diffs | 0 |

Source-list registration remains aligned across:

- `Makefile`;
- `CMakeLists.txt`;
- `build-metadata/library_sources.txt`.

The new private source appears in the eigensolver cluster:

```text
src/sparse_eigs_workspace_internal.c
src/sparse_eigs_dense_internal.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_thick_restart.c
src/sparse_eigs.c
```

## Changed-File Summary

Implementation and build diff summary:

```text
CMakeLists.txt                     |   1 +
Makefile                           |   1 +
build-metadata/library_sources.txt |   1 +
src/sparse_eigs.c                  | 126 -------------------------------------
tests/test_qr.c                    | 110 ++++++++++++++------------------
```

Additional untracked Sprint 109 artifacts and the new private source are part
of the branch working tree but are not represented in `git diff --stat` until
tracked by commit.

## Public and Install Surface

Header drift check:

```sh
git diff --name-only -- include src/*.h tests/*.h
```

Result:

```text
<no output>
```

This confirms:

- no installed public header changed;
- no private source header changed;
- no shared test helper header changed;
- no public support surface was broadened by Sprint 109.

## Validation Gaps

No Day 13 validation gap remains.

Deferred work remains intentionally outside Sprint 109 implementation scope:

- matrix-shell Matrix Market source split;
- grow-m/refinement/dispatch/handle eigensolver movement;
- QR sequential RHS helper follow-through;
- LDLT CSC external dense-reference oracle cleanup;
- per-solver iterative exact-RHS helper cleanup;
- SVD storage-layout proof-loop cleanup.

These are residual planning items, not failed Day 13 checks.

## Completion Criteria Status

- Required full quality gate passed before closeout.
- Metrics cover every changed owner.
- Public/header drift evidence is explicit.
- Source-list and helper metrics are explicit.
- No quality failure is deferred silently.
- Validation evidence is ready for the Sprint 109 retrospective.
