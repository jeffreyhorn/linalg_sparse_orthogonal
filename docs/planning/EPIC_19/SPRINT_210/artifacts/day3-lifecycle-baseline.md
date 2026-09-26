# Sprint 210 Day 3: Lifecycle Baseline

## Purpose

Document the selected linked-list LDLT allocation owner lifecycle before test,
harness, or source edits. This baseline freezes the selected owner boundary,
owned outputs, cleanup expectations, caller-input preservation expectations,
and retry behavior for Days 4 through 9.

## Selected Owner Boundary

Sprint 210 selected owner:

- public entry point: `sparse_ldlt_factor_opts(const SparseMatrix *A,
  const sparse_ldlt_opts_t *opts, sparse_ldlt_t *ldlt)`;
- required option boundary: `opts->backend = SPARSE_LDLT_BACKEND_LINKED_LIST`;
- internal owner path: `ldlt_factor_internal(...)` in `src/sparse_ldlt.c`;
- caller-owned inputs: `A` and optional `opts`;
- caller-visible output: caller-provided `sparse_ldlt_t`;
- cleanup API: `sparse_ldlt_free(&ldlt)`;
- primary proof binary: `tests/test_ldlt.c`;
- out of scope: CSC LDLT, reordered LDLT proof, solve/refine/condest
  workspace failures, all-direct-solver reliability, OS OOM, concurrent hooks,
  hosted proof, package/install reliability, platform parity, performance,
  release, external-library parity, and state-of-the-art reliability.

## Lifecycle Map

| Phase | Behavior | Allocation-failure invariant |
| --- | --- | --- |
| Output reset | `sparse_ldlt_factor_opts()` clears every `ldlt` owned field before validation. `ldlt_factor_internal()` also clears fields on entry. | Failure must leave `ldlt` free-safe. |
| Validation | Null, shape, original-state, and symmetry checks happen before selected owner allocations. | Validation failures are not allocation proof but must not publish success state. |
| Metadata publication | `ldlt->n`, `factor_norm`, and `tol` are assigned before some allocations. | Allocation failure after metadata publication must clear metadata through cleanup. |
| Factor arrays | `D`, `D_offdiag`, and `pivot_size` are allocated for length `n`. | Any failure leaves all factor arrays null or free-safe after cleanup. |
| Working copy | `W = sparse_copy(A)` creates a mutable working matrix. | Failure releases previously allocated factor arrays and leaves `A` unchanged. |
| Factor matrix | `ldlt->L = sparse_create(n, n)` and identity diagonal insertion build the unit lower factor. | Failure releases `W`, arrays, and partial `L`. |
| Pivot permutation | `ldlt->perm` is allocated and initialized to identity. | Failure releases `W` and every owned output field. |
| Dense workspaces | `col_acc`, `nz_flag`, `nz_list`, `col_acc_r`, `nz_flag_r`, and `nz_list_r` are temporary elimination workspaces. | Failure releases all temporaries, `W`, and selected output fields. |
| Elimination | Bunch-Kaufman pivoting mutates `W`, `ldlt->L`, `D`, `D_offdiag`, `pivot_size`, and `perm`. | Propagated allocation/insertion failure goes through `err_cleanup` and clears selected output. |
| Success | `ldlt` owns `L`, `D`, `D_offdiag`, `pivot_size`, `perm`, `n`, `factor_norm`, and `tol`. | Retry after an injected failure must produce this fresh success state. |

## Allocation And Publication Points

| Point | Current allocation form | Owner | Day 4 decision |
| --- | --- | --- | --- |
| `D`, `D_offdiag`, `pivot_size` | direct `calloc` | `ldlt` output | Convert or otherwise make injectable. |
| `W = sparse_copy(A)` | matrix copy path | temporary | Include as propagated allocation failure, not broad matrix-copy proof. |
| `L = sparse_create(n, n)` | matrix constructor path | `ldlt` output | Include as selected-output allocation failure, not broad matrix-constructor proof. |
| identity `sparse_insert()` | matrix insertion path | partial `ldlt->L` | Include as selected-output initialization failure. |
| `perm` | direct `malloc` | `ldlt` output | Convert or otherwise make injectable. |
| dense accumulator arrays | direct `calloc`/`malloc` | temporaries | Convert or otherwise make injectable. |
| elimination `sparse_insert()` | matrix insertion path | `ldlt->L` entries | Include as selected-owner insertion propagation. |
| swap helper temporaries | direct `malloc` in helper functions | temporaries | Decide on Day 4 whether to include or defer as helper-level swap allocation proof. |

## Caller-Owned Input Preservation

| Input | Required invariant |
| --- | --- |
| `A` | Linked-list LDLT must not mutate the caller-owned matrix. Tests should compare structure and values before and after injected allocation failure. |
| `opts` | Options are caller-owned and should remain unchanged. |
| `opts->used_csc_path` output pointer | May be written with `0` because the API documents backend telemetry. This is allowed telemetry, not corruption. |
| progress callback state | Out of allocation-failure scope unless later fixture explicitly selects cancellation behavior. |

## Stale-Output Invariants

After selected-owner allocation failure, `sparse_ldlt_free(&ldlt)` must be safe
to call repeatedly and the observed output should be empty:

| Field | Expected failed state |
| --- | --- |
| `L` | `NULL` |
| `D` | `NULL` |
| `D_offdiag` | `NULL` |
| `pivot_size` | `NULL` |
| `perm` | `NULL` |
| `n` | `0` |
| `factor_norm` | `0.0` |
| `tol` | `0.0` |

## Retry Expectations

For each selected injected allocation-failure point:

1. configure deterministic allocation failure;
2. call the selected linked-list LDLT factorization fixture;
3. reset allocation hooks before assertions that can allocate;
4. assert error status and empty/free-safe output;
5. assert caller-owned input preservation;
6. rerun the same fixture without injection;
7. assert success, non-null owned fields, usable solve or reconstruction, and
   successful cleanup.

## Day 4 Harness Questions

1. Which minimal fixture exercises factor arrays, `L`, `perm`, dense
   workspaces, and elimination insertion?
2. Which direct `malloc`/`calloc` sites in `ldlt_factor_internal()` should be
   converted to private allocation wrappers?
3. Are swap-helper temporary allocations included in this selected proof or
   deferred as helper-level swap proof?
4. Should the focused gate be named `ldlt-allocation-failure-gate` or
   `ldlt-linked-list-allocation-failure-gate` to avoid CSC overclaims?
5. What registration guard should ensure the selected LDLT allocation tests stay
   in the intended test binary and gate?

## Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 3 changes planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

