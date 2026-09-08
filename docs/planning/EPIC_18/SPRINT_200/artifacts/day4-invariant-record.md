# Sprint 200 Day 4: Invariant Record

## Purpose

Convert the Day 3 `sparse_symbolic_lu()` lifecycle trace into the pre-edit
contract that later harness, regression-test, focused-gate, and documentation
work must satisfy.

## Selected Owner

| Field | Value |
| --- | --- |
| Function | `sparse_symbolic_lu()` |
| Implementation | `src/sparse_etree.c:412-665` |
| Declaration | `src/sparse_analysis_internal.h:140-141` |
| Proof-owner test binary | `tests/test_etree.c` |
| Outputs | Caller-provided `sparse_symbolic_t *sym_L` and `sparse_symbolic_t *sym_U` |
| Cleanup function | `sparse_symbolic_free()` |

This is a selected symbolic LU allocation-failure proof only. It is not a
broad etree, analysis, direct-solver, matrix-construction, OS OOM, concurrent
allocation-hook, hosted CI, or state-of-the-art reliability proof.

## Invariants

| ID | Invariant |
| --- | --- |
| S200-LU-CLEAN-01 | Every selected deterministic allocation failure releases temporary owner resources and leaves requested outputs free-safe. |
| S200-LU-CLEAN-02 | Propagated failure from the local symbolic Cholesky intermediate does not transfer `sym_full` to `sym_L` and does not retain partial U output. |
| S200-LU-PUB-01 | `sym_L` is published only on `SPARSE_OK`; allocation failure leaves it empty: null arrays, `n == 0`, and `nnz == 0`. |
| S200-LU-PUB-02 | `sym_U` is empty/free-safe on allocation failure, including after early zeroing, `n` assignment, partial `col_ptr`, or partial `row_idx` work. |
| S200-LU-STALE-01 | No failed selected-owner call returns a success-looking symbolic object. |
| S200-LU-RETRY-01 | After `sparse_alloc_test_reset()`, the same fixture can succeed and produce fresh L+U, L-only, or U-only output. |
| S200-LU-INPUT-01 | Caller-owned matrix `A` remains unchanged across selected allocation failures. |
| S200-LU-INPUT-02 | Caller-owned valid permutation `perm` remains unchanged across selected allocation failures. |
| S200-LU-HOOK-01 | Tests reset the process-global allocation hook before assertion macros can early-return. |
| S200-LU-HOOK-02 | Required direct allocation sites are either converted to existing private wrappers or excluded from the earned proof. |
| S200-LU-SCOPE-01 | Gates and docs name selected symbolic LU behavior and retain broad non-claims. |

## Output-State Decisions

The proof will use zeroed or previously freed `sparse_symbolic_t` outputs,
matching the existing internal precondition. It will not pass live allocated
stale outputs, because the current contract says doing so leaks memory.

| Output mode | Required failure state |
| --- | --- |
| L+U | `sym_L` and `sym_U` both have `col_ptr == NULL`, `row_idx == NULL`, `n == 0`, and `nnz == 0`. |
| L-only | `sym_L` has null arrays and zero metadata. |
| U-only | `sym_U` has null arrays and zero metadata; local `sym_full` is freed internally. |
| Valid permutation | Outputs are empty/free-safe and the caller-owned permutation array is unchanged. |

## Cleanup Checklist

| Resource | Required cleanup behavior |
| --- | --- |
| Temporary interaction matrix `B` | Freed after any failure following creation. |
| `seen` and `inv_perm` | Freed on allocation failure, invalid permutation, and later selected failures. |
| `row_cols` | Freed after use or on failure before graph construction completes. |
| `parent`, `postorder`, `cc` | Freed on direct allocation failure and through the common cleanup label. |
| Local `sym_full` | Freed on every failure after creation unless transferred to `sym_L` on success. |
| `u_cnt` | Freed on every U-building failure and after successful U row fill. |
| Partial `sym_U` arrays | Cleared with `sparse_symbolic_free(sym_U)` before returning allocation failure. |
| Allocation hook | Reset before assertions, fixture cleanup, and retries. |

## Planned Test Mapping

| Test family | Invariants covered |
| --- | --- |
| Symbolic LU hook reachability | S200-LU-HOOK-01, S200-LU-HOOK-02. |
| L+U allocation-failure table | S200-LU-CLEAN-01, S200-LU-CLEAN-02, S200-LU-PUB-01, S200-LU-PUB-02, S200-LU-STALE-01. |
| L-only failure and retry | S200-LU-PUB-01, S200-LU-RETRY-01, S200-LU-INPUT-01. |
| U-only failure and retry | S200-LU-PUB-02, S200-LU-RETRY-01, S200-LU-INPUT-01. |
| Valid-permutation failure and retry | S200-LU-INPUT-02, S200-LU-RETRY-01. |
| Gate registration guard | S200-LU-SCOPE-01. |

## Harness Decisions Deferred To Day 5

The current implementation has three direct selected-owner allocation sites
that are not controlled by `sparse_alloc_test_fail_after(...)`:

1. `seen = calloc(1, seen_bytes)`;
2. `inv_perm = malloc(inv_perm_bytes)`;
3. `sym_U->col_ptr = malloc(u_col_ptr_bytes)`.

Day 5 should decide whether to convert these to `sparse_calloc_array()` and
`sparse_malloc_array()` so they participate in the existing private hook. It
should also decide whether `sparse_insert(B, ...)` allocation failures are
inside the selected symbolic LU proof or retained as matrix-construction
breadth.

## Claim Boundary

Once implementation and tests exist, the earned claim should be no broader
than:

`sparse_symbolic_lu()` has focused local deterministic allocation-failure
proof for selected fixtures covering failure status, requested-output cleanup,
stale-output suppression, caller-owned matrix/permutation preservation, and
retry-after-reset behavior.

Retained non-claims:

- broad allocation-failure coverage;
- `sparse_analyze()` lifecycle cleanup;
- standalone etree, postorder, or colcount helper allocation failures;
- direct solvers, eigensolvers, graph routines, SVD, sparse matrix
  construction, conversion, or IO;
- operating-system OOM behavior;
- concurrent allocation-hook behavior;
- hosted CI, package, platform, ABI, performance, release, or
  state-of-the-art reliability support.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 4 validation command.
