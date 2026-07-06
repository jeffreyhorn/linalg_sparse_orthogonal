# Day 4 Dense Jacobi Extraction

## Purpose

Day 4 implements the approved Sprint 109 dense Jacobi source-boundary move.
The goal is to move only `s21_dense_sym_jacobi` into a private source owner
while preserving behavior, private declaration ownership, source-list parity,
and reviewed public/test surfaces.

## Decision

The Day 2 and Day 3 go criteria held, so Day 4 extracted the helper instead of
publishing a no-split deferral.

Only the dense Jacobi section comment and `s21_dense_sym_jacobi` implementation
moved.

## Changed Files

| File | Change |
|---|---|
| `src/sparse_eigs_dense_internal.c` | New private source containing `s21_dense_sym_jacobi`. |
| `src/sparse_eigs.c` | Removed only the moved dense Jacobi section. |
| `Makefile` | Added `$(SRCDIR)/sparse_eigs_dense_internal.c` to `LIB_SRCS`. |
| `CMakeLists.txt` | Added `src/sparse_eigs_dense_internal.c` to the static library source list. |
| `build-metadata/library_sources.txt` | Added `src/sparse_eigs_dense_internal.c` in matching order. |

## Unchanged Boundaries

- `src/sparse_eigs_internal.h` remains the private declaration owner.
- `src/sparse_eigs_thick_restart.c` remains a direct caller.
- `src/sparse_eigs_lobpcg.c` remains a direct caller.
- No public header under `include/` changed.
- No install/export or pkg-config surface changed.
- No helper target changed.
- No test source changed.
- No CTest registration changed.

## Source-List Ordering

All source-list owners now use this eigensolver ordering:

```text
src/sparse_eigs_workspace_internal.c
src/sparse_eigs_dense_internal.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_thick_restart.c
src/sparse_eigs.c
```

`make source-list-check` passed with 46 library sources.

## Symbol Location After Extraction

| Symbol | Owner | Declaration | Direct Callers |
|---|---|---|---|
| `s21_dense_sym_jacobi` | `src/sparse_eigs_dense_internal.c` | `src/sparse_eigs_internal.h` | `src/sparse_eigs_thick_restart.c`, `src/sparse_eigs_lobpcg.c` |

Inspection command:

```sh
rg -n "s21_dense_sym_jacobi\\(" src tests include
```

Observed matches:

```text
src/sparse_eigs_thick_restart.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_internal.h
src/sparse_eigs_dense_internal.c
```

## Source-Size Metrics

| File | Before | After |
|---|---:|---:|
| `src/sparse_eigs.c` | 1538 lines | 1412 lines |
| `src/sparse_eigs_dense_internal.c` | 0 lines | 129 lines |

The library source count increased from 45 to 46 because the moved helper now
has its own private compilation unit.

## Focused Validation

Focused build and eigensolver tests passed:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_sprint29_integration
```

Results:

| Test | Result |
|---|---|
| `test_eigs` | Passed, 31 tests |
| `test_eigs_thick_restart` | Passed, 21 tests |
| `test_eigs_lobpcg` | Passed, 27 tests |
| `test_sprint29_integration` | Passed, 3 tests |

## Full Validation

Required code-change quality gate passed:

```sh
make format && make lint && make test
```

Additional hygiene checks passed:

```sh
make source-list-check
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_109
```

`rg` returned no trailing-whitespace matches.

## Completion Criteria Status

- Private dense spectral helper extraction completed.
- Source-list parity updated across Makefile, CMake, and manifest ownership.
- Behavior-sensitive eigensolver callers remained unchanged.
- Focused and full validation passed.
- No public API, install-header, helper-target, test-source, or CTest drift was
  introduced.
