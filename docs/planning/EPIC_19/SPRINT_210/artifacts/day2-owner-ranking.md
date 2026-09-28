# Sprint 210 Day 2: Owner Ranking

## Purpose

Rank Sprint 210 allocation-failure owner candidates, exclude existing proof
lanes, select exactly one primary owner, name a fallback owner, and freeze the
initial proof boundary before lifecycle tracing or code edits.

## Scoring Method

Day 2 uses a 1-5 score where 5 is strongest for Sprint 210. The sprint values
complete selected-owner closure over broad partial coverage.

| Criterion | Meaning |
| --- | --- |
| User impact | The owner is important to a public workflow. |
| Allocation density | The owner has enough allocation behavior to make deterministic failure proof meaningful. |
| Ownership clarity | The owner has a narrow lifecycle and output boundary. |
| Cleanup risk | Failed allocation could leave partial state or unclear ownership. |
| Stale-output exposure | Caller-visible outputs need explicit failure behavior. |
| Retry clarity | A failed fixture can reset and then succeed with clear expected output. |
| Fixture/review cost | Lower implementation and review surface earns a higher score. |

## Selected Owner

Sprint 210 selects the linked-list LDLT numeric factorization owner in
`src/sparse_ldlt.c`.

Selected boundary:

- public entry point: `sparse_ldlt_factor_opts(const SparseMatrix *A,
  const sparse_ldlt_opts_t *opts, sparse_ldlt_t *ldlt)` with
  `opts->backend = SPARSE_LDLT_BACKEND_LINKED_LIST`;
- default `sparse_ldlt_factor()` is in scope only when the selected fixture
  routes to the same linked-list numeric path;
- internal owner path: `ldlt_factor_internal(...)`;
- caller-owned inputs: `A` and optional `opts`;
- caller-visible output: caller-provided `sparse_ldlt_t` released by
  `sparse_ldlt_free()`;
- selected proof binary: `tests/test_ldlt.c`;
- likely focused gate: selected LDLT allocation-failure Make target plus a
  registration guard if the tests are a focused subset.

## Ranking

| Rank | Candidate owner | Total | Disposition |
| ---: | --- | ---: | --- |
| 1 | Selected linked-list LDLT numeric factorization owner | 33 | Selected. High-value direct-solver owner with separate `sparse_ldlt_t` output and clear free/retry semantics. |
| 2 | Selected QR factorization workspace owner | 30 | Fallback. Strong value and hook reachability, but larger review surface and recent QR churn. |
| 3 | Selected Cholesky CSC numeric owner | 27 | Deferred. In-place matrix publication makes stale-output semantics more delicate. |
| 4 | Matrix construction/conversion owner | 25 | Deferred. High impact but too likely to imply broad sparse-matrix allocation reliability unless split further. |
| 5 | Selected SVD workspace owner | 24 | Deferred. Multi-output numerical proof cost is high. |
| 6 | Selected eigensolver workspace owner | 23 | Deferred. Handle/caller-buffer lifecycle needs a separate lifecycle-owner sprint. |
| 7 | Matrix import/export owner | 22 | Deferred. Lower allocation density and IO/parse semantics could distract from allocation proof. |

## Why Linked-List LDLT

Linked-list LDLT factorization is the best owner for Sprint 210 because it
closes a direct-solver allocation-failure gap left after the selected symbolic
LU proof while keeping the scope narrow. The owner publishes into a separate
`sparse_ldlt_t` rather than replacing the input matrix, which gives the sprint
clear cleanup, stale-output suppression, repeated cleanup, and
retry-after-reset checks. Existing `tests/test_ldlt.c` fixtures already cover
basic factor/solve behavior, and the public header documents that the output
object is reset on entry and freed with `sparse_ldlt_free()`.

## Fallback Owner

If Day 3 tracing shows selected linked-list LDLT cannot be reached with bounded
deterministic allocation hooks without broad source churn, Sprint 210 will
fall back to selected QR factorization workspace proof.

Fallback boundary:

- public entry point: `sparse_qr_factor_opts(...)` or `sparse_qr_factor(...)`;
- output owner: caller-provided `sparse_qr_t` released by `sparse_qr_free()`;
- proof binary: `tests/test_qr.c`;
- fallback reason: `src/sparse_qr.c` already uses private allocation wrappers
  for several workspace allocations and has clear cleanup semantics;
- fallback caution: QR has a larger review surface and recent evidence churn.

## Existing Owners Excluded

| Existing proof lane | Reason excluded |
| --- | --- |
| Iterative repeated-run handles | Already covered by `make iterative-allocation-failure-gate`. |
| `sparse_matmul()` workspace | Already covered by `make matmul-allocation-failure-gate`. |
| Selected symbolic Cholesky | Already covered by `make symbolic-allocation-failure-gate`. |
| Selected `sparse_symbolic_lu()` | Closed by Sprint 200 through `make symbolic-lu-allocation-failure-gate`. |

## Proof Checklist

| Proof area | Required evidence |
| --- | --- |
| Failed allocation | Deterministic fail-after cases return `SPARSE_ERR_ALLOC` or a documented propagated allocation error. |
| Cleanup | Partial `sparse_ldlt_t` state is free-safe and releases owned `L`, `D`, `D_offdiag`, `pivot_size`, and `perm` fields. |
| Stale-output suppression | Failed calls do not leave `ldlt` success-looking or solve-ready. |
| Caller-owned input | Input matrix and options remain valid and unchanged after injected allocation failure. |
| Retry | The same fixture succeeds after allocation hook reset and produces a usable LDLT factor. |
| Unsupported breadth | No all-LDLT, CSC LDLT, Cholesky, all-direct-solver, QR/SVD/eigs, matrix-construction, OS OOM, concurrent-hook, hosted, package/install, platform, performance, release, external-library parity, or state-of-the-art reliability claim is added. |

## Day 3 Handoff

Day 3 should trace selected linked-list LDLT allocation and publication points
in detail:

1. `sparse_ldlt_t` reset and `sparse_ldlt_free()` safety on entry and error
   paths.
2. `D`, `D_offdiag`, `pivot_size`, `L`, and `perm` allocation and ownership.
3. Direct `calloc`/`malloc` sites that may need wrapper conversion for
   deterministic hook reachability.
4. `sparse_create()` and `sparse_insert()` behavior while building `ldlt->L`.
5. Pivot callback cancellation boundaries versus allocation-failure
   boundaries.
6. Small fixture success criteria and retry-after-reset behavior.
7. Explicit exclusion of CSC LDLT and broader direct-solver families.

## Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 2 changes planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

