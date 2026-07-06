# Day 3 Source-List Parity & Validation Harness Prep

## Purpose

Day 3 prepares the private dense Jacobi extraction path without changing
behavior. The goal is to make Day 4 executable from a checklist: every source
membership owner, validation command, no-drift check, and stop condition must
be known before `s21_dense_sym_jacobi` moves.

This artifact changes no code, build file, manifest, or source-list owner.

## Source Membership Baseline

Current eigensolver library source ordering is consistent across all three
source-list owners:

| Owner | Current Eigensolver Order |
|---|---|
| `Makefile` `LIB_SRCS` | `sparse_eigs_workspace_internal.c`, `sparse_eigs_lobpcg.c`, `sparse_eigs_thick_restart.c`, `sparse_eigs.c` |
| `CMakeLists.txt` `add_library` | `sparse_eigs_workspace_internal.c`, `sparse_eigs_lobpcg.c`, `sparse_eigs_thick_restart.c`, `sparse_eigs.c` |
| `build-metadata/library_sources.txt` | `sparse_eigs_workspace_internal.c`, `sparse_eigs_lobpcg.c`, `sparse_eigs_thick_restart.c`, `sparse_eigs.c` |

Day 4 should add the new private source in the same relative position in all
three places. Recommended ordering:

```text
src/sparse_eigs_workspace_internal.c
src/sparse_eigs_dense_internal.c
src/sparse_eigs_lobpcg.c
src/sparse_eigs_thick_restart.c
src/sparse_eigs.c
```

Rationale:

- workspace storage remains the lowest-level eigensolver support owner;
- dense spectral helper comes before backend owners that consume it;
- backend owners remain before the public/orchestration owner
  `src/sparse_eigs.c`.

## Source-List Parity Gate

The manifest parity checker is:

```sh
make source-list-check
```

It runs:

```sh
python3 scripts/check_library_sources.py
```

Day 4 must run this after adding the new source to all membership owners and
before relying on focused tests.

## Proposed Day 4 File-Level Change Set

If Day 4 extraction proceeds, the intended file set is:

| File | Expected Change |
|---|---|
| `src/sparse_eigs_dense_internal.c` | New private source containing only `s21_dense_sym_jacobi` and its local explanatory comment. |
| `src/sparse_eigs.c` | Remove only the `s21_dense_sym_jacobi` implementation and the local Jacobi section comment. |
| `src/sparse_eigs_internal.h` | No change expected; keep the existing private declaration. |
| `Makefile` | Add the new source to `LIB_SRCS`. |
| `CMakeLists.txt` | Add the new source to the static library source list. |
| `build-metadata/library_sources.txt` | Add the new source in matching order. |

Day 4 should not touch:

- public headers under `include/`;
- install/export configuration;
- pkg-config templates;
- helper targets;
- test registration;
- eigensolver tests except for validation output artifacts, if any.

## Include and Linkage Plan

Recommended new source skeleton:

```c
#include "sparse_eigs_internal.h"

#include <math.h>
```

The implementation should retain the existing signature:

```c
sparse_err_t s21_dense_sym_jacobi(double *A_scratch, idx_t K,
                                  double *theta_out, double *Q_out)
```

Do not make the function `static`; it is intentionally shared by
`src/sparse_eigs_thick_restart.c` and `src/sparse_eigs_lobpcg.c` through the
private internal declaration.

## Focused Test Targets

The focused eigensolver tests exist in both Make and CMake registration
surfaces:

| Test | Makefile Source List | CMake `add_sparse_test` | Purpose |
|---|---|---|---|
| `test_eigs` | yes | yes | Public/default eigensolver and grow-m behavior. |
| `test_eigs_thick_restart` | yes | yes | Thick-restart caller of dense Jacobi. |
| `test_eigs_lobpcg` | yes | yes | LOBPCG caller of dense Jacobi. |
| `test_sprint29_integration` | yes | yes | Cross-feature eigensolver/refinement/progress integration. |

Focused build and run commands:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg build/test_sprint29_integration
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
./build/test_sprint29_integration
```

## Current Reviewed CTest Surface

The relevant CMake test registrations are stable:

```text
add_sparse_test(test_sprint29_integration)
add_sparse_test(test_eigs)
add_sparse_test(test_eigs_thick_restart)
add_sparse_test(test_eigs_lobpcg)
```

Day 4 extraction should not add, remove, rename, or reorder tests for the
dense Jacobi move. Any CTest registration delta is a stop condition unless it
is intentionally reviewed separately.

## No-Drift Checklist

After Day 4 edits, verify:

- no public header under `include/` changed;
- `src/sparse_eigs_internal.h` still owns the private declaration;
- no install/export or pkg-config file changed;
- no new helper target was added;
- no test source or CTest registration changed;
- Makefile, CMake, and `build-metadata/library_sources.txt` agree;
- direct callers remain in `src/sparse_eigs_thick_restart.c` and
  `src/sparse_eigs_lobpcg.c`;
- `src/sparse_eigs.c` no longer contains the implementation but still builds
  and links through the library archive.

Suggested inspection commands:

```sh
git diff --name-only
git diff -- include CMakeLists.txt Makefile build-metadata/library_sources.txt src/sparse_eigs.c src/sparse_eigs_dense_internal.c src/sparse_eigs_internal.h
rg -n "s21_dense_sym_jacobi\\(" src tests include
```

## Required Broad Quality Gate for Extraction

Because Day 4 extraction would touch `.c` files and build/source-list
membership, run:

```sh
make format && make lint && make test
git diff --check
```

If the branch remains documentation-only on Day 4 because extraction is
deferred, use docs-only validation:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_109
```

## Atomic Extraction Checklist

Day 4 can proceed in this order:

1. Create `src/sparse_eigs_dense_internal.c`.
2. Move only the dense Jacobi section comment and
   `s21_dense_sym_jacobi` function body from `src/sparse_eigs.c`.
3. Include `sparse_eigs_internal.h` and `<math.h>` in the new source.
4. Leave `src/sparse_eigs_internal.h` unchanged unless compilation proves a
   private include dependency is missing.
5. Add the new source to `Makefile`, `CMakeLists.txt`, and
   `build-metadata/library_sources.txt` in matching order.
6. Run `make source-list-check`.
7. Run focused eigensolver builds and tests.
8. Run `make format && make lint && make test`.
9. Run `git diff --check`.
10. Record validation and drift evidence before closeout.

## Stop Conditions

Stop and defer the extraction if:

- moving the helper requires moving any other eigensolver function;
- the new source requires a public header or install-header change;
- source-list parity cannot be made exact;
- focused thick-restart or LOBPCG validation fails;
- reviewed CTest registration changes unexpectedly;
- the broad quality gate fails.

## Completion Criteria Status

- Every build-system/source-list touch point is identified.
- Focused eigensolver validation commands are known and bounded.
- Public header, install-header, helper-target, and CTest no-drift checks are
  explicit before implementation.
- Day 4 can proceed from an atomic extraction checklist without discovering new
  source-list owners.
