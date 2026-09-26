# Sprint 210 Working Notes: Additional Allocation-Failure Owner Proof

## Sprint Goal

Add deterministic allocation-failure proof for one new high-value owner outside
the already closed selected symbolic LU path.

## Scope Boundary

Sprint 210 is a selected-owner reliability sprint. It may close one
allocation-failure owner with deterministic failure, cleanup, stale-output,
caller-input preservation, and retry evidence. It must not claim broad
allocation-failure coverage, package-manager support, ABI/shared-library
support, broad platform parity, portable performance, release readiness,
external-library parity, or state-of-the-art reliability.

## Day 1: Allocation Proof Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial artifact |
| --- | --- | --- |
| 210.1 Owner Selection | Rank candidate owners from the Epic 19 candidate families and select exactly one on Day 2. | Candidate ledger and Day 2 scoring inputs. |
| 210.2 Lifecycle Invariant Record | Prepare status, cleanup, stale-output, caller-input, retry, and partial-publication invariant categories before code edits. | Invariant checklist placeholder. |
| 210.3 Harness Extension | Reuse existing deterministic allocation hooks and focused fixture patterns where possible; extend only for the selected owner. | Harness reachability notes. |
| 210.4 Regression Tests | Plan failed-allocation, cleanup, stale-output, preservation, and retry tests for the selected owner. | Test requirement matrix. |
| 210.5 Gate And Documentation | Add a focused Make/CTest gate and claim-safe docs only after the owner is chosen. | Gate inventory and documentation boundary notes. |
| 210.6 Validation And Closeout | Run focused, family, source-list, docs, and full C quality checks once implementation begins. | Validation matrix and closeout checklist. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 210 is a 168-hour sprint to prove one new high-value allocation-failure owner outside selected symbolic LU. |
| `docs/planning/EPIC_19/reviews/review-codex-2026-09-20.md` | Epic 19 still treats allocation-failure coverage as selected rather than broad; remaining gaps include direct solver internals, matrix construction/import/export, eigensolver/SVD/QR workspace, and package/install tooling. |
| `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md` | Closure Track 4 names candidate families: matrix construction/import/export, QR factorization workspace, LDLT/Cholesky numeric factorization, eigensolver workspace, and SVD workspace. |
| `docs/planning/EPIC_18/SPRINT_200/RETROSPECTIVE.md` | Sprint 200 closed selected `sparse_symbolic_lu()` allocation-failure proof and explicitly preserved broad allocation-failure, direct solver, eigensolver, SVD, matrix construction/conversion/IO, package/install, hosted, platform, OS OOM, concurrent hook, release, and state-of-the-art residuals. |
| `Makefile` | Existing allocation-failure gates include `iterative-allocation-failure-gate`, `matmul-allocation-failure-gate`, `symbolic-allocation-failure-gate`, and `symbolic-lu-allocation-failure-gate`. Sprint 210 must not re-close those owners. |
| `tests/test_matmul_allocation_failure_gate_registration.py`, `tests/test_symbolic_allocation_failure_gate_registration.py`, and `tests/test_symbolic_lu_allocation_failure_gate_registration.py` | Registration-guard patterns exist for focused allocation-failure gates and should be reused if Sprint 210 adds a new focused gate or selected test subset. |
| `src/sparse_alloc_internal.c` and `src/sparse_alloc_internal.h` | Private deterministic allocation hooks remain the preferred proof mechanism for wrapped allocation sites. |

### Existing Allocation-Failure Gates

| Gate | Owner surface | Sprint 210 interpretation |
| --- | --- | --- |
| `make iterative-allocation-failure-gate` | Iterative repeated-run handle allocation behavior. | Existing selected proof; exclude as Sprint 210 owner. |
| `make matmul-allocation-failure-gate` | `sparse_matmul()` workspace allocation behavior. | Existing selected proof; reuse stale-output and retry proof style only. |
| `make symbolic-allocation-failure-gate` | Selected symbolic Cholesky allocation behavior. | Existing selected proof; reuse selected-owner gate pattern only. |
| `make symbolic-lu-allocation-failure-gate` | Selected `sparse_symbolic_lu()` allocation behavior. | Sprint 200 closed this owner; exclude as Sprint 210 owner. |

### Current Allocation And Cleanup Surface Scan

The Day 1 source scan counted source lines matching allocation wrappers or
direct `malloc`/`calloc`/`realloc`, then cleanup/failure indicators such as
`cleanup:`, `goto cleanup`, `SPARSE_ERR_ALLOC`, `return NULL`, and `free(`.
Counts are ranking signals only, not proof of owner boundaries.

| Candidate source | Allocation signal | Cleanup/failure signal | Day 1 interpretation |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 58 | 118 | Highest allocation signal; strong LDLT/Cholesky numeric candidate if narrowed to one owner. |
| `src/sparse_qr.c` | 33 | 149 | Strong QR workspace candidate with high cleanup density; review cost and recent QR churn require careful scope. |
| `src/sparse_ldlt.c` | 29 | 135 | Public LDLT owner candidate with visible user impact and meaningful temporary workspace semantics. |
| `src/sparse_chol_csc.c` | 28 | 71 | Direct CSC Cholesky candidate distinct from symbolic Cholesky and symbolic LU proofs. |
| `src/sparse_svd_partial.c` | 14 | 97 | SVD workspace candidate with multi-output publication risk and higher fixture complexity. |
| `src/sparse_matrix.c` | 14 | 56 | Matrix construction/conversion candidate with high user impact but broad API boundary risk. |
| `src/sparse_svd.c` | 13 | 104 | Dense/SVD candidate with meaningful output cleanup and numerical retry concerns. |
| `src/sparse_eigs_thick_restart.c` | 10 | 39 | Eigensolver workspace candidate with selected algorithm boundary potential. |
| `src/sparse_eigs.c` | 6 | 37 | Public eigensolver owner candidate; likely composed with workspace helpers. |
| `src/sparse_cholesky.c` | 6 | 60 | Public Cholesky path candidate, but may be too thin or delegated depending on selected entry point. |
| `src/sparse_matrix_io.c` | 1 | 14 | Matrix import/export candidate with user-facing failure semantics but lower allocation signal in this scan. |

### Candidate Owner Ledger

| Candidate owner | User impact | Hook fit | Current gap shape | Day 1 disposition |
| --- | --- | --- | --- | --- |
| Selected LDLT CSC numeric factorization owner | High for direct-solver users. | Likely mixed; dense allocation and cleanup paths need tracing. | Allocation-heavy and cleanup-heavy, but the owner must be narrowed to avoid all-LDLT claims. | Leading Day 2 candidate. |
| Selected QR workspace owner | High for least-squares and QR users. | Likely mixed; `src/sparse_qr.c` has many cleanup paths. | High value but recent QR review surface increases risk of broad or brittle proof. | Leading Day 2 candidate with review-surface caution. |
| Selected Cholesky CSC numeric owner | High for SPD direct-solver users. | Likely reachable through existing test fixtures. | Distinct from symbolic Cholesky, but must avoid implying broad Cholesky correctness or symbolic coverage. | Day 2 candidate. |
| Selected SVD workspace owner | Medium-high for advanced users. | Likely reachable in selected paths. | Multi-output publication and retry checks may increase complexity. | Day 2 candidate if a narrow output owner is found. |
| Selected eigensolver workspace owner | Medium-high for eigensolver users. | Unclear until workspace lifecycle is traced. | Handles and caller buffers have rich failure semantics; composed algorithms may be broad. | Day 2 candidate with lifecycle caution. |
| Matrix construction/conversion owner | Very high for all users. | Existing matrix tests help, but broad API boundary risk is high. | Could close an important reliability gap if narrowed to exactly one constructor or conversion path. | Candidate only with a precise owner boundary. |
| Matrix import/export owner | Medium-high user-facing IO value. | Lower allocation signal; parse and IO errors may dominate allocation behavior. | Valuable failure semantics, but allocation owner may be less dense than numeric workspaces. | Candidate if Day 2 values user-facing IO over allocation density. |

### Initial Invariant Categories

| Category | Day 1 placeholder |
| --- | --- |
| Return status | Failed selected-owner allocation must return a documented allocation/error status without reporting success. |
| Cleanup | Failed allocation must release selected-owner temporary and partially owned resources. |
| Stale-output suppression | Caller-visible outputs must remain unchanged, null-cleared, or explicitly status-gated according to the selected owner contract. |
| Partial publication | Failed allocation must not leave a success-looking factor, workspace, handle, matrix, or result structure. |
| Caller-owned input | Input matrices, options, vectors, permutation arrays, and caller buffers must remain valid and unchanged unless the selected owner contract explicitly permits mutation. |
| Retry | After resetting deterministic failure injection, the same fixture should succeed and produce fresh selected-owner output. |
| Unsupported breadth | The proof will not imply broad allocation-failure, OS OOM, concurrent allocation-hook, hosted CI, package/install, platform, performance, release, external-library, or state-of-the-art reliability coverage. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only Day 1 changes. |
| Focused selected-owner gate | Not yet applicable. | Owner selection is scheduled for Day 2. |
| Relevant family tests | Not yet applicable. | Family depends on selected owner. |
| Source-list check | Not yet applicable. | No source/test registration changes on Day 1. |
| Documentation checks | Not yet applicable. | Day 1 creates planning evidence only. |
| `make format && make lint && make test` | Not required for Day 1. | No `.c` or `.h` files modified. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad an owner | Sprint 210 should completely close one selected gap rather than partially touch many. | Day 2 must freeze exactly one owner and reject broad family claims. |
| Re-proving an already closed owner | Sprint 210 must be outside selected symbolic LU and existing allocation gates. | Exclude iterative, matmul, symbolic Cholesky, and symbolic LU gates from owner selection. |
| Direct allocations bypass hooks | Deterministic failure tests prove only allocation paths reachable through the hook or controlled wrapper. | Prefer wrapper-reachable owner paths or explicitly budget a narrow wrapper conversion. |
| Ambiguous stale-output semantics | Tests can overclaim if output publication is not documented before implementation. | Day 3 must record output, cleanup, partial-publication, and retry invariants before edits. |
| Multi-output numerical owners increase review surface | SVD/eigensolver/QR owners may require complex numerical fixtures and broad docs. | Prefer one narrow workspace or output owner with small deterministic fixtures. |
| Process-global hook contamination | Early returns can leave allocation hooks active and poison unrelated tests. | Follow Sprint 200 reset discipline and add focused cleanup paths in tests. |
| Claim overreach | One selected proof is not broad reliability. | Keep all public and maintainer wording selected-owner-only. |

### Open Questions For Day 2

1. Which candidate has the clearest selected-owner boundary outside symbolic LU?
2. Which candidate has the strongest combination of user impact, allocation
   density, cleanup risk, and retry clarity?
3. Which candidate is reachable with existing deterministic allocation hooks
   without broad source churn?
4. Which candidate has an output publication contract that can be asserted
   without numerical ambiguity?
5. Which candidate can receive a focused gate and registration guard with low
   review surface?

### Day 1 Validation

Commands run:

```sh
git diff --check
git status --short
find docs/planning/EPIC_19/SPRINT_210 -type f | sort
git diff --name-only -- '*.c' '*.h'
```

Day 1 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 2: Owner Ranking

### Scoring Method

Day 2 scores use a 1-5 scale where 5 is strongest for Sprint 210. The scoring
prioritizes complete selected-owner closure over broad family coverage.

| Criterion | Meaning |
| --- | --- |
| User impact | The owner sits on a meaningful public workflow. |
| Allocation density | The implementation has enough allocation behavior to make the proof valuable. |
| Ownership clarity | The owner has a narrow output/lifecycle boundary that can be documented. |
| Cleanup risk | Failed allocation could leave partial state, leaked ownership, or unclear teardown. |
| Stale-output exposure | Caller-visible outputs need explicit failure behavior. |
| Retry clarity | The same fixture can fail under injection, reset, and then succeed with clear expected output. |
| Fixture and review cost | Lower implementation/review surface earns a higher score. |

### Ranked Candidate Table

| Rank | Candidate owner | User impact | Allocation density | Ownership clarity | Cleanup risk | Stale-output exposure | Retry clarity | Fixture/review cost | Total | Disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Selected linked-list LDLT numeric factorization owner | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 33 | Selected. High-value direct-solver owner with separate caller-visible `sparse_ldlt_t` output and clear free/retry semantics. |
| 2 | Selected QR factorization workspace owner | 5 | 4 | 4 | 5 | 5 | 4 | 3 | 30 | Fallback. Strong user value and hook reachability, but larger review surface and more recent QR churn. |
| 3 | Selected Cholesky CSC numeric owner | 5 | 4 | 3 | 4 | 4 | 4 | 3 | 27 | Deferred. Valuable but in-place matrix publication makes stale-output semantics more delicate. |
| 4 | Selected SVD workspace owner | 4 | 3 | 3 | 4 | 5 | 3 | 2 | 24 | Deferred. Multi-output numerical proof is higher cost. |
| 5 | Selected eigensolver workspace owner | 4 | 3 | 3 | 4 | 4 | 3 | 2 | 23 | Deferred. Handle/caller-buffer lifecycle needs a separate lifecycle sprint. |
| 6 | Matrix construction/conversion owner | 5 | 3 | 2 | 4 | 4 | 5 | 2 | 25 | Deferred despite high impact. Too easy to imply broad sparse-matrix allocation reliability unless one constructor is isolated. |
| 7 | Matrix import/export owner | 4 | 2 | 3 | 3 | 3 | 3 | 4 | 22 | Deferred. Lower allocation density and IO/parse errors would compete with allocation proof scope. |

### Selected Owner

Sprint 210 selects the linked-list LDLT numeric factorization owner in
`src/sparse_ldlt.c`.

Selected boundary:

- public entry point: `sparse_ldlt_factor_opts(const SparseMatrix *A,
  const sparse_ldlt_opts_t *opts, sparse_ldlt_t *ldlt)` with
  `opts->backend = SPARSE_LDLT_BACKEND_LINKED_LIST`;
- default convenience entry point in scope only when it routes to the same
  linked-list numeric path for the selected fixture;
- internal owner path: `ldlt_factor_internal(...)` in `src/sparse_ldlt.c`;
- caller-owned inputs: `A` and optional `opts`;
- caller-visible output: caller-provided `sparse_ldlt_t` object whose owned
  fields are released by `sparse_ldlt_free()`;
- selected success fixtures: small symmetric indefinite/SPD fixtures that force
  the linked-list backend and have existing LDLT tests as correctness
  references;
- primary proof binary: `tests/test_ldlt.c`;
- expected focused gate: a new selected LDLT allocation-failure gate or a
  focused `test_ldlt` subset with registration guard.

### Selection Rationale

Linked-list LDLT factorization is the strongest Day 2 selection because it
closes a direct-solver allocation-failure gap left after the symbolic LU proof
without selecting all direct solvers. The owner publishes into a separate
`sparse_ldlt_t` object instead of mutating the input matrix as the final
factor, which makes cleanup, stale-output suppression, repeated cleanup, and
retry-after-reset assertions more direct than in-place Cholesky. The source has
high allocation and cleanup density, existing public API documentation already
states that the output object is reset on entry, and `tests/test_ldlt.c`
provides a natural proof binary.

### Fallback Owner

If Day 3 lifecycle tracing shows that linked-list LDLT cannot be reached with
bounded deterministic allocation hooks without broad source churn, Sprint 210
falls back to the selected QR factorization workspace owner:

- public entry point: `sparse_qr_factor_opts(...)` or `sparse_qr_factor(...)`;
- output owner: caller-provided `sparse_qr_t` released by `sparse_qr_free()`;
- primary proof binary: `tests/test_qr.c`;
- reason fallback remains viable: `src/sparse_qr.c` already uses the private
  allocation wrappers for several workspace allocations and has clear
  `sparse_qr_free()` cleanup semantics;
- reason it is not primary: broader review surface, multiple QR modes, and
  recent QR evidence churn increase risk.

### Rejected And Deferred Candidates

| Candidate | Disposition | Reason | Future follow-up |
| --- | --- | --- | --- |
| Iterative repeated-run handles | Excluded | Already covered by `make iterative-allocation-failure-gate`. | Maintain existing gate. |
| `sparse_matmul()` workspace | Excluded | Already covered by `make matmul-allocation-failure-gate`. | Maintain existing registration guard. |
| Selected symbolic Cholesky | Excluded | Already covered by `make symbolic-allocation-failure-gate`. | Use selected-owner gate pattern only. |
| Selected symbolic LU | Excluded | Closed by Sprint 200 through `make symbolic-lu-allocation-failure-gate`. | Use reset and retry proof pattern only. |
| Selected Cholesky CSC numeric owner | Deferred | In-place matrix publication complicates stale-output semantics for this sprint. | Revisit with a Cholesky-specific in-place publication invariant record. |
| Selected SVD workspace owner | Deferred | Multi-output numerical owner has higher fixture and claim-boundary cost. | Split one output/workspace owner first. |
| Selected eigensolver workspace owner | Deferred | Handle and caller-buffer lifecycle needs more predesign. | Treat as a future lifecycle-owner sprint. |
| Matrix construction/conversion owner | Deferred | High impact but broad sparse-matrix allocation wording risk. | Isolate exactly one constructor/conversion path later. |
| Matrix import/export owner | Deferred | Allocation behavior is less dense and IO/parse semantics could distract from allocation proof. | Revisit as an IO reliability sprint. |

### Initial Proof Checklist

| Proof area | Required evidence for selected linked-list LDLT owner |
| --- | --- |
| Failed allocation | Deterministic fail-after cases return `SPARSE_ERR_ALLOC` or a documented propagated allocation error. |
| Cleanup | Partial `sparse_ldlt_t` state is free-safe and releases `L`, `D`, `D_offdiag`, `pivot_size`, and `perm` ownership. |
| Stale-output suppression | Failed calls must not leave `ldlt` looking factored or success-ready. |
| Caller-owned input | Input matrix and options remain valid and unchanged after injected allocation failure. |
| Retry | The same fixture succeeds after allocation hook reset and produces a usable LDLT factor. |
| Unsupported breadth | The proof does not claim all LDLT backends, CSC LDLT, Cholesky, all direct solvers, QR/SVD/eigs, matrix construction, OS OOM, concurrent hooks, hosted proof, package/install, platform, performance, release, external-library parity, or state-of-the-art reliability. |

### Day 3 Handoff

Day 3 should trace selected linked-list LDLT allocation and publication points,
especially:

1. output reset and `sparse_ldlt_free()` safety on entry and error paths;
2. `D`, `D_offdiag`, `pivot_size`, `L`, and `perm` ownership;
3. direct `calloc`/`malloc` sites that may need wrapper conversion for
   deterministic hook reachability;
4. sparse matrix insertion failures while building `ldlt->L`;
5. pivot callback cancellation versus allocation failure boundaries;
6. final success criteria for a small fixture and retry-after-reset behavior;
7. exact out-of-scope boundary for CSC LDLT and broader direct solvers.

### Day 2 Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 2 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 3: Lifecycle Baseline

### Selected Owner Lifecycle Map

| Phase | Selected linked-list LDLT behavior | Allocation-failure expectation |
| --- | --- | --- |
| Public entry | `sparse_ldlt_factor_opts(A, opts, ldlt)` resets every `ldlt` field to empty before validating `A`. | A failed call starts from an empty output object and must remain free-safe. |
| Backend selection | Day 3 scope requires `opts->backend = SPARSE_LDLT_BACKEND_LINKED_LIST`; `used_csc_path`, when provided, records attempted backend before factorization. | The Sprint 210 proof must not claim CSC LDLT allocation-failure coverage. |
| Reorder handling | Reorder paths allocate `perm`, may allocate a permuted matrix `PA`, then delegate to the selected backend. | Initial proof should use `SPARSE_REORDER_NONE`; reordered LDLT allocation proof is out of scope unless later explicitly selected. |
| Internal owner entry | `ldlt_factor_internal(A, ldlt, tol, progress_cb, progress_user)` zeroes output fields again, checks null/shape/original-state/symmetry, records `n`, `factor_norm`, and `tol`. | Validation failures before allocation must return documented status and leave output free-safe. |
| Owned factor arrays | Internal owner allocates `D`, `D_offdiag`, and `pivot_size`, then creates `L` and initializes identity diagonal, then allocates `perm`. | Failure after any partial allocation must call `sparse_ldlt_free(ldlt)` or equivalent cleanup. |
| Working copy and workspaces | Internal owner creates working copy `W` and dense accumulators `col_acc`, `nz_flag`, `nz_list`, `col_acc_r`, `nz_flag_r`, and `nz_list_r`. | Temporary workspaces and `W` must be released on every failed allocation or propagated failure. |
| Elimination loop | Bunch-Kaufman pivoting writes `D`, `D_offdiag`, `pivot_size`, `perm`, and `L`; insert/swap helper failures go through `err_cleanup`. | Partial factor state must not remain solve-ready on allocation or insertion failure. |
| Callback cancellation | Progress callback cancellation frees temporaries and `ldlt`, then returns `SPARSE_ERR_CANCELLED`. | Cancellation is a neighboring lifecycle path, not allocation-failure proof, but cleanup expectations should remain compatible. |
| Success publication | On success, `ldlt` owns `L`, `D`, `D_offdiag`, `pivot_size`, `perm`, `n`, `factor_norm`, and `tol`; caller releases them with `sparse_ldlt_free()`. | Retry-after-reset must produce a usable success object after prior injected allocation failure. |

### Selected Allocation And Publication Points

| Point | Current allocation form | Owned object | Day 3 proof expectation |
| --- | --- | --- | --- |
| Overflow guard for `n` | no allocation | status only | Return `SPARSE_ERR_ALLOC` and clear `ldlt`. |
| `D` | direct `calloc` | `ldlt->D` | Convert or otherwise make deterministically injectable; failure leaves all owned fields free-safe. |
| `D_offdiag` | direct `calloc` | `ldlt->D_offdiag` | Same as `D`. |
| `pivot_size` | direct `calloc` | `ldlt->pivot_size` | Same as `D`. |
| `W = sparse_copy(A)` | matrix constructor/copy path | temporary `W` | Failure returns `SPARSE_ERR_ALLOC` and clears `ldlt`; matrix-copy internals are not broadly claimed. |
| `L = sparse_create(n, n)` | matrix constructor path | `ldlt->L` | Failure releases factor arrays and clears `ldlt`. |
| Identity `sparse_insert(ldlt->L, i, i, 1.0)` | matrix insertion path | partial `ldlt->L` | Failure releases `W`, factor arrays, and partial `L`. |
| `perm` | direct `malloc` | `ldlt->perm` | Convert or otherwise make deterministically injectable; failure releases `W` and `ldlt` fields. |
| Dense accumulators | direct `calloc`/`malloc` | temporary workspaces | Convert or otherwise make deterministically injectable; failure releases all temporaries, `W`, and `ldlt`. |
| Elimination `sparse_insert(ldlt->L, ...)` | matrix insertion path | `ldlt->L` entries | Propagated insertion allocation failure clears accumulators, frees `W`, and clears `ldlt`. |
| Swap helper temporary arrays | direct `malloc` in helper paths | temporary swap arrays | Day 4 must decide whether these helper allocations are inside the selected proof sweep or excluded as non-primary swap-helper paths. |

### Caller-Owned Inputs

| Input | Preservation expectation |
| --- | --- |
| `A` | Must remain valid and numerically unchanged after injected allocation failure; selected linked-list LDLT writes to `W` and `ldlt`, not `A`. |
| `opts` | Must remain unchanged; Day 3 selected fixture uses local options with linked-list backend and no reorder. |
| `opts->used_csc_path` target | If provided, may be set to `0` to report attempted linked-list backend before failure; this telemetry mutation is documented and not considered input corruption. |
| Progress callback state | Out of allocation-failure proof unless Day 4 chooses a callback fixture; cancellation remains separate from allocation failure. |

### Stale-Output And Cleanup Invariants

| Output field | Failure expectation |
| --- | --- |
| `ldlt->L` | `NULL` after allocation failure or safe to free repeatedly before observation. |
| `ldlt->D` | `NULL` after allocation failure or safe to free repeatedly before observation. |
| `ldlt->D_offdiag` | `NULL` after allocation failure or safe to free repeatedly before observation. |
| `ldlt->pivot_size` | `NULL` after allocation failure or safe to free repeatedly before observation. |
| `ldlt->perm` | `NULL` after allocation failure or safe to free repeatedly before observation. |
| `ldlt->n` | `0` after cleanup on allocation failure; early validation may set and then clear through cleanup for allocated paths. |
| `ldlt->factor_norm` | `0.0` after cleanup on allocation failure. |
| `ldlt->tol` | `0.0` after cleanup on allocation failure. |

### Retry Expectations

| Retry scenario | Expected result |
| --- | --- |
| Fail before factor arrays | Reset hook, rerun same fixture, and obtain `SPARSE_OK` with non-NULL `L`, `D`, `D_offdiag`, `pivot_size`, and `perm`. |
| Fail after partial factor arrays | Reset hook, rerun same fixture, and prove previous partial state did not poison success. |
| Fail after `L` creation or identity insertion | Reset hook, rerun same fixture, and verify solve or reconstruction on fresh factor succeeds. |
| Fail in dense workspace allocation | Reset hook, rerun same fixture, and verify fresh factor metadata and solve behavior. |
| Fail in elimination insertion | Reset hook, rerun same fixture, and verify fresh factor is usable. |

### Day 4 Handoff

Day 4 should design harness and wrapper reachability for the selected owner:

1. choose the smallest linked-list LDLT fixture that exercises factor arrays,
   `L`, `perm`, dense workspaces, and at least one elimination insertion;
2. decide which direct `malloc`/`calloc` sites in `ldlt_factor_internal()` must
   be converted to `sparse_malloc*_array`/`sparse_calloc*_array` for
   deterministic failure injection;
3. decide whether swap-helper direct allocations are in scope or deferred;
4. define output-empty assertions and repeated `sparse_ldlt_free()` checks;
5. define caller-input preservation checks for `A` and linked-list options;
6. define retry-after-reset success checks using `sparse_ldlt_solve()` or
   reconstruction;
7. name the focused gate and registration guard.

### Day 3 Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 3 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 4: Harness Design

### Harness Decision

Sprint 210 will reuse the existing private deterministic allocation hook:

- `sparse_alloc_test_fail_after(...)`;
- `sparse_alloc_test_reset()`;
- `sparse_malloc_array(...)`;
- `sparse_calloc_array(...)`;
- `sparse_malloc_idx_array(...)`;
- `sparse_calloc_idx_array(...)`.

No public allocator API, environment variable, or new test framework is needed.
The selected linked-list LDLT owner needs a narrow wrapper-conversion pass
inside `ldlt_factor_internal()` so direct allocation sites become reachable by
the existing hook.

### Selected Fixture Design

| Fixture | Purpose | Expected success baseline |
| --- | --- | --- |
| 3x3 SPD tridiagonal LDLT | Exercises factor arrays, `L`, `perm`, dense workspaces, and solve verification with 1x1 pivots. | `sparse_ldlt_factor_opts(... LINKED_LIST ...)` succeeds, `ldlt.L`, `D`, `D_offdiag`, `pivot_size`, and `perm` are non-null, solve residual is near zero. |
| 2x2 indefinite LDLT | Exercises 2x2 pivot metadata and `D_offdiag` publication. | Factor succeeds with `pivot_size[0] == 2`, `pivot_size[1] == 2`, and solve succeeds. |
| 3x3/4x4 mixed pivot fixture | Optional Day 6 expansion if the 3x3 fixture does not reach elimination insertion or swap-adjacent behavior. | Factor succeeds and reconstruction or solve check passes. |

Day 4 primary fixture is the 3x3 SPD tridiagonal case because it has stable
numerical expectations and low review cost. The 2x2 indefinite fixture is the
metadata fallback for 2x2 pivot publication coverage.

### Wrapper Reachability Plan

| Selected owner site | Current form | Day 5 action | Claim boundary |
| --- | --- | --- | --- |
| `ldlt->D` | direct `calloc` | Convert to `sparse_calloc_idx_array(n, sizeof(double), ...)`. | Selected linked-list LDLT output allocation only. |
| `ldlt->D_offdiag` | direct `calloc` | Convert to `sparse_calloc_idx_array(n, sizeof(double), ...)`. | Selected linked-list LDLT output allocation only. |
| `ldlt->pivot_size` | direct `calloc` | Convert to `sparse_calloc_idx_array(n, sizeof(int), ...)`. | Selected linked-list LDLT output allocation only. |
| `ldlt->perm` | direct `malloc` | Convert to `sparse_malloc_idx_array(n, sizeof(idx_t), ...)`. | Selected linked-list LDLT output allocation only. |
| `col_acc`, `col_acc_r` | direct `calloc` | Convert to `sparse_calloc_idx_array(n, sizeof(double), ...)`. | Temporary workspace proof, not broad dense workspace coverage. |
| `nz_flag`, `nz_flag_r` | direct `calloc` | Convert to `sparse_calloc_idx_array(n, sizeof(int), ...)`. | Temporary workspace proof. |
| `nz_list`, `nz_list_r` | direct `malloc` | Convert to `sparse_malloc_idx_array(n, sizeof(idx_t), ...)`. | Temporary workspace proof. |
| `sparse_copy(A)` | existing matrix copy path | Keep as propagated failure source only. | Not broad sparse matrix copy proof. |
| `sparse_create(n, n)` and `sparse_insert(ldlt->L, ...)` | existing matrix path | Keep as propagated selected-output construction failure source. | Not broad matrix construction/insertion proof. |
| swap-helper temporary arrays | direct helper `malloc` | Defer from initial sweep unless selected fixture reaches them naturally after wrapper conversion. | Helper-level swap allocation proof remains separate if not reached. |

### Failure Case Plan

Named failure cases should be established after wrapper conversion and verified
by observed fail-after indices rather than guessed permanently on Day 4.

| Planned case family | Intended failure points |
| --- | --- |
| Output arrays | `D`, `D_offdiag`, `pivot_size`. |
| Matrix setup propagation | `sparse_copy(A)`, `sparse_create(n, n)`, identity `sparse_insert`. |
| Output permutation | `perm`. |
| Dense workspace | `col_acc`, `nz_flag`, `nz_list`, `col_acc_r`, `nz_flag_r`, `nz_list_r`. |
| Elimination insertion | selected `sparse_insert(ldlt->L, i, k, ...)` propagated allocation failures, if reached by fixture. |

### Test Helper Plan

| Helper | Purpose |
| --- | --- |
| `make_ldlt_allocation_failure_matrix()` | Build the selected fixture with deterministic values and no allocation hook active. |
| `assert_ldlt_failure_output_empty(const sparse_ldlt_t *ldlt)` | Verify failed outputs are not success-looking: null owned pointers and zero metadata. |
| `assert_ldlt_failure_output_free_safe(sparse_ldlt_t *ldlt)` | Call `sparse_ldlt_free()` repeatedly and verify the output remains empty. |
| `assert_ldlt_input_matrix_intact(const SparseMatrix *A)` | Verify caller-owned matrix structure and values after injected failure. |
| `assert_ldlt_success_baseline(const SparseMatrix *A, const sparse_ldlt_t *ldlt)` | Verify success metadata and solve or reconstruction behavior. |
| `expect_ldlt_allocation_failure(const LdltAllocationFailureCase *case)` | Apply fail-after hook, call owner, reset hook before assertions, then verify status, cleanup, stale-output, and input preservation. |
| `expect_ldlt_allocation_failure_recovers(const LdltAllocationFailureCase *case)` | Prove retry success after hook reset. |
| `assert_allocation_hook_probe_after_reset()` | Reuse Sprint 200 hook-reset probe style to detect leaked fail-after state. |

### Focused Gate And Registration Plan

| Surface | Day 4 plan |
| --- | --- |
| Make target | Add `ldlt-allocation-failure-gate` or `ldlt-linked-list-allocation-failure-gate`; prefer the longer name if review feedback indicates broad-LDLT ambiguity. |
| Test binary | Use existing `tests/test_ldlt.c`; avoid a new binary unless source-list churn becomes smaller than conditional/focused test routing. |
| CTest label | Add or verify `ldlt;allocation_failure` label only after tests exist. |
| Registration guard | Add `tests/test_ldlt_allocation_failure_gate_registration.py` requiring the Make target, CTest label if used, and the selected `RUN_TEST(...)` entries. |
| Source-list impact | No new `.c` file expected; Python registration guard may need Makefile or docs-check registration only if project patterns require it. |

### Hook Cleanup Rules

1. Call `sparse_alloc_test_reset()` before every fail-after setup.
2. Store the LDLT status in a local variable.
3. Call `sparse_alloc_test_reset()` before any assertion macro can early return.
4. Reset again before retry calls and after fixture cleanup.
5. Free matrices and `sparse_ldlt_t` only after the hook is reset.
6. Keep fixture creation outside the injected failure window unless the test is
   explicitly proving fixture setup behavior.

### Day 5 Implementation Handoff

Day 5 should:

1. convert selected direct allocations in `ldlt_factor_internal()` to private
   allocation wrappers;
2. add the LDLT fixture and helper skeleton to `tests/test_ldlt.c`;
3. establish observed fail-after indices for named cases;
4. add initial cleanup/stale-output and retry tests;
5. wire only the minimum focused gate pieces needed for local execution;
6. defer docs claim wording until Day 11.

### Day 4 Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 4 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 5: Harness Implementation

### Changed Surfaces

| File | Day 5 change |
| --- | --- |
| `src/sparse_ldlt.c` | Included `sparse_alloc_internal.h` and converted selected linked-list LDLT factor arrays, permutation output, and dense elimination workspaces from direct allocation to private allocation wrappers. |
| `tests/test_ldlt.c` | Included the private allocation hook header and added selected linked-list LDLT allocation-failure fixture helpers, cleanup/stale-output assertions, hook-reset probe, success-baseline assertions, and retry tests. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 5 implementation details and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day5-harness-implementation.md` | Added Day 5 implementation artifact. |

### Wrapper Conversion Implemented

| Selected owner allocation | Day 5 implementation |
| --- | --- |
| `ldlt->D` | `sparse_calloc_idx_array(n, sizeof(double), ...)` |
| `ldlt->D_offdiag` | `sparse_calloc_idx_array(n, sizeof(double), ...)` |
| `ldlt->pivot_size` | `sparse_calloc_idx_array(n, sizeof(int), ...)` |
| `ldlt->perm` | `sparse_malloc_idx_array(n, sizeof(idx_t), ...)` |
| `col_acc` and `col_acc_r` | `sparse_calloc_idx_array(n, sizeof(double), ...)` |
| `nz_flag` and `nz_flag_r` | `sparse_calloc_idx_array(n, sizeof(int), ...)` |
| `nz_list` and `nz_list_r` | `sparse_malloc_idx_array(n, sizeof(idx_t), ...)` |

These conversions are private implementation changes for deterministic test
reachability. They do not add public allocator API and do not claim broad LDLT,
CSC LDLT, or all-direct-solver allocation-failure coverage.

### Implemented Test Helpers

| Helper | Purpose |
| --- | --- |
| `make_ldlt_allocation_failure_matrix()` | Builds the 3x3 SPD tridiagonal selected fixture outside the injected failure window. |
| `ldlt_linked_list_allocation_opts()` | Forces `SPARSE_LDLT_BACKEND_LINKED_LIST` and disables reordering. |
| `assert_ldlt_allocation_failure_input_intact(...)` | Verifies caller-owned matrix structure and values after injected failure. |
| `assert_ldlt_failure_output_empty(...)` | Verifies no success-looking LDLT output remains after failure. |
| `assert_ldlt_failure_output_free_safe(...)` | Verifies repeated `sparse_ldlt_free()` is safe after failure. |
| `assert_ldlt_allocation_hook_probe_after_reset()` | Verifies the process-global allocation hook is reset after each case. |
| `assert_ldlt_allocation_success_baseline(...)` | Verifies fresh retry success with non-null owned fields and solve residual. |
| `expect_ldlt_allocation_failure(...)` | Executes one deterministic fail-after case and checks status, cleanup, stale-output, and caller-input preservation. |
| `expect_ldlt_retry_after_allocation_failure(...)` | Proves retry success after reset for one failure case. |

### Initial Failure Cases

| Case | `fail_after` | Coverage |
| --- | ---: | --- |
| `D output array` | 0 | First selected output allocation. |
| `D_offdiag output array` | 1 | Partial output cleanup after `D` succeeds. |
| `pivot_size output array` | 2 | Partial output cleanup after two output arrays succeed. |

Day 6 should extend the observed failure sweep to `sparse_copy`, `L`
construction, `perm`, dense workspaces, and selected insertion propagation.

### Added Tests

| Test | Evidence |
| --- | --- |
| `test_ldlt_linked_list_allocation_failures_clear_outputs` | Deterministic output-array allocation failures return `SPARSE_ERR_ALLOC`, preserve the input matrix, leave output empty/free-safe, and reset hooks. |
| `test_ldlt_linked_list_allocation_failures_recover_on_retry` | Each initial failure case can be retried after hook reset and produces a usable linked-list LDLT factor. |

### Day 5 Validation

Commands run:

```sh
clang-format -i src/sparse_ldlt.c tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `91` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `1233` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed on rerun after stopping an older duplicate local
  `make test` process that had been running concurrently and caused a
  transient temp-file collision in `test_sparse_matrix`.
- `git diff --check` passed.

## Day 6: Failure Sweep Tests

### Changed Surfaces

| File | Day 6 change |
| --- | --- |
| `tests/test_ldlt.c` | Expanded the selected linked-list LDLT allocation-failure case table from three initial output-array failures to 25 deterministic fail-after sites. Added an explicit case-count assertion so future edits cannot silently shrink the sweep. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded the Day 6 failure-sweep map and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day6-failure-sweep.md` | Added the Day 6 failure-sweep artifact. |

### Deterministic Sweep Map

| `fail_after` | Named site | Outcome asserted |
| ---: | --- | --- |
| 0 | `D output array` | `SPARSE_ERR_ALLOC`; output empty/free-safe; input matrix intact; hook reset. |
| 1 | `D_offdiag output array` | `SPARSE_ERR_ALLOC`; output empty/free-safe; input matrix intact; hook reset. |
| 2 | `pivot_size output array` | `SPARSE_ERR_ALLOC`; output empty/free-safe; input matrix intact; hook reset. |
| 3 | `working copy entry buffer` | Propagated `sparse_copy(A)` setup allocation failure. |
| 4 | `working copy row headers` | Propagated working-copy `sparse_create()` shell allocation failure. |
| 5 | `working copy column headers` | Propagated working-copy `sparse_create()` shell allocation failure. |
| 6 | `working copy row permutation` | Propagated working-copy permutation allocation failure. |
| 7 | `working copy inverse row permutation` | Propagated working-copy permutation allocation failure. |
| 8 | `working copy column permutation` | Propagated working-copy permutation allocation failure. |
| 9 | `working copy inverse column permutation` | Propagated working-copy permutation allocation failure. |
| 10 | `working copy row-tail scratch` | Propagated working-copy build scratch allocation failure. |
| 11 | `working copy column-tail scratch` | Propagated working-copy build scratch allocation failure. |
| 12 | `L row headers` | Propagated output-`L` `sparse_create()` shell allocation failure. |
| 13 | `L column headers` | Propagated output-`L` `sparse_create()` shell allocation failure. |
| 14 | `L row permutation` | Propagated output-`L` permutation allocation failure. |
| 15 | `L inverse row permutation` | Propagated output-`L` permutation allocation failure. |
| 16 | `L column permutation` | Propagated output-`L` permutation allocation failure. |
| 17 | `L inverse column permutation` | Propagated output-`L` permutation allocation failure. |
| 18 | `permutation output array` | Selected `ldlt->perm` output allocation failure. |
| 19 | `column accumulator workspace` | Selected dense workspace allocation failure. |
| 20 | `nonzero flag workspace` | Selected dense workspace allocation failure. |
| 21 | `nonzero list workspace` | Selected dense workspace allocation failure. |
| 22 | `pivot-candidate accumulator workspace` | Selected dense workspace allocation failure. |
| 23 | `pivot-candidate nonzero flag workspace` | Selected dense workspace allocation failure. |
| 24 | `pivot-candidate nonzero list workspace` | Selected dense workspace allocation failure. |

The Day 6 sweep intentionally remains limited to allocations visible through
the existing private allocation hook. Identity and elimination `sparse_insert`
node-slab allocations use the shared matrix pool's direct `malloc` path and are
not claimed as closed by this selected linked-list LDLT proof. Converting the
pool allocator would broaden the sprint beyond the selected owner boundary.

### Assertion Coverage

The two Day 5 tests now cover all 25 Day 6 cases:

- `test_ldlt_linked_list_allocation_failures_clear_outputs` asserts
  `SPARSE_ERR_ALLOC`, caller-input preservation, empty/free-safe output, hook
  reset, and exact 25-case registration.
- `test_ldlt_linked_list_allocation_failures_recover_on_retry` verifies every
  failed site can be retried after hook reset and produces a usable linked-list
  LDLT factor with solve residual matching the fixture RHS.

### Day 6 Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `91` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `3587` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.

## Day 10: Focused Gate Wiring

### Changed Surfaces

| File | Day 10 change |
| --- | --- |
| `Makefile` | Added `ldlt-linked-list-allocation-failure-gate`, which builds `build/test_ldlt`, runs the LDLT allocation-failure registration guard, runs `build/test_ldlt`, and emits a selected-gate pass banner. |
| `CMakeLists.txt` | Added CTest labels `ldlt;linked_list;allocation_failure` to `test_ldlt`. |
| `tests/test_ldlt_allocation_failure_gate_registration.py` | Added registration guard for the selected linked-list LDLT gate, CMake label, required `RUN_TEST(...)` entries, representative fail-after cases, and key cleanup/stale-output/retry assertions. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 10 focused-gate wiring and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day10-focused-gate.md` | Added the Day 10 gate-wiring artifact. |

### Focused Gate

```sh
make ldlt-linked-list-allocation-failure-gate
```

The gate intentionally names the selected owner (`linked-list LDLT`) instead
of broad LDLT, CSC LDLT, or direct-solver allocation reliability. It reuses the
existing `test_ldlt` binary rather than adding a new C source file.

### Registration Guard Coverage

`tests/test_ldlt_allocation_failure_gate_registration.py` verifies:

- the Make target is present and depends on `$(BUILDDIR)/test_ldlt`;
- the Make target invokes the registration guard;
- CMake registers `test_ldlt`;
- CMake labels `test_ldlt` with `ldlt;linked_list;allocation_failure`;
- the selected allocation-failure, cleanup, stale-output, and retry tests are
  registered in `tests/test_ldlt.c`;
- representative fail-after cases and key proof assertions remain present.

### Day 10 Validation

Commands run:

```sh
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make ldlt-linked-list-allocation-failure-gate
```

Focused gate result:

- registration guard passed;
- `95` LDLT tests passed;
- `0` tests failed;
- `0` tests skipped;
- `7781` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.

## Day 11: Documentation Calibration

### Implementation Summary

Day 11 updated user-facing, install/readiness, and maintainer claim surfaces so
Sprint 210 documentation names the selected linked-list LDLT allocation-failure
proof added during Days 5 through 10 without promoting broad allocation,
direct-solver, package, platform, performance, release, or state-of-the-art
claims.

| File | Day 11 change |
| --- | --- |
| `README.md` | Adds `make ldlt-linked-list-allocation-failure-gate` to selected allocation-failure proof wording, the command list, and the repeated-run reliability boundary notes. |
| `INSTALL.md` | Extends the support-readiness matrix row for local selected allocation-failure proof with the linked-list LDLT gate, owner, and retained non-claims. |
| `docs/maintainer_guide.md` | Extends the reliability proof-owner ledger with Sprint 210 LDLT tests, gate, guard, artifacts, and selected-only non-claims. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Records Day 11 documentation changes, earned claim, retained non-claims, and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day11-documentation-calibration.md` | Adds the Day 11 documentation-calibration artifact. |

### Earned Claim

The current earned claim is selected-owner only:

Selected no-reorder linked-list LDLT numeric factorization has focused local
deterministic allocation-failure proof for bounded known fixtures covering 25
injected allocation-failure sites, cleanup, stale-output suppression,
caller-input preservation, free-safe output state, repeated cleanup after
failure, and retry-after-reset behavior.

### Retained Non-Claims

The Day 11 docs continue to reject broad claims for:

- broad allocation-failure coverage across the library;
- CSC LDLT allocation-failure proof;
- reordered LDLT allocation-failure proof;
- Cholesky, broad direct solvers, QR, SVD, eigensolver, sparse matrix
  construction, conversion, IO, package/install, or generated-tooling
  allocation-failure proof;
- operating-system OOM behavior;
- platform parity, hosted CI proof, package-manager proof, shared-library ABI
  proof, performance proof, release readiness, or state-of-the-art reliability
  support;
- concurrent allocation-hook behavior.

### Day 11 Validation

Commands run:

```sh
make ldlt-linked-list-allocation-failure-gate
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make docs-check
make support-docs-guard
git diff --check
```

Results:

- `make ldlt-linked-list-allocation-failure-gate`: PASS; `95` LDLT tests,
  `0` failures, `0` skips, and `7781` assertions.
- `python3 tests/test_ldlt_allocation_failure_gate_registration.py`: PASS;
  `ldlt-allocation-failure-gate-registration: passed`.
- `make docs-check`: PASS; Doxygen generation and API docs coverage completed
  with 18 checked-in public headers, 18 generated reference pages, 18
  generated source pages, and `sparse_version.h` kept under its separate
  installed-header policy.
- `make support-docs-guard`: PASS; support quick-reference docs guard passed.
- `git diff --check`: PASS.

## Day 12: Integrated Validation

### Implementation Summary

Day 12 reran the focused selected-owner gate, registration guard,
source-list/docs checks, and the full required C validation chain for the
Sprint 210 linked-list LDLT allocation-failure proof. No code changes were
made for Day 12; the day records integrated pass evidence across the code,
test, gate, and documentation surfaces touched by Days 5 through 11.

| Validation surface | Command | Result |
| --- | --- | --- |
| Focused selected-owner gate | `make ldlt-linked-list-allocation-failure-gate` | PASS |
| Registration guard | `python3 tests/test_ldlt_allocation_failure_gate_registration.py` | PASS |
| Source-list guard | `make source-list-check` | PASS |
| API docs check | `make docs-check` | PASS |
| Support docs guard | `make support-docs-guard` | PASS |
| Formatting | `make format` | PASS |
| Lint/static analysis | `make lint` | PASS |
| Full test suite | `make test` | PASS |
| Whitespace check | `git diff --check` | PASS |

### Focused Gate Evidence

`make ldlt-linked-list-allocation-failure-gate` reran the selected linked-list
LDLT proof:

- `95` LDLT tests passed;
- `0` tests failed;
- `0` tests skipped;
- `7781` assertions passed.

The standalone registration guard printed
`ldlt-allocation-failure-gate-registration: passed`.

### Full Validation Evidence

- `make source-list-check`: `PASS (49 library sources)`.
- `make docs-check`: Doxygen generation and API docs coverage passed with 18
  checked-in public headers, 18 generated reference pages, 18 generated source
  pages, and `sparse_version.h` kept under its separate installed-header
  policy.
- `make support-docs-guard`: `test-support-quick-reference-docs: ok`.
- `make format`: passed after applying repository clang-format rules.
- `make lint`: passed clang-tidy and cppcheck across the library and test
  surfaces.
- `make test`: passed all test binaries; the final line was `All tests
  passed.`
- `git diff --check`: passed after the Day 12 notes and artifact update.

### Day 12 Boundary

The integrated validation supports only the selected no-reorder linked-list
LDLT allocation-failure owner. It does not add CSC LDLT, reordered LDLT,
Cholesky, broad direct-solver, package/install, platform, performance, release,
external-library parity, or state-of-the-art reliability claims.

## Day 13: Review Hardening

### Implementation Summary

Day 13 reviewed the selected linked-list LDLT proof against the Day 3 lifecycle
invariants, Day 10 gate wiring, Day 11 claim wording, and Day 12 integrated
validation ledger. The review found one guard-hardening issue: the Day 10
registration guard searched for raw `RUN_TEST(...)` substrings, so a
commented-out registration could still satisfy the guard.

| Surface | Day 13 action |
| --- | --- |
| `tests/test_ldlt_allocation_failure_gate_registration.py` | Hardened required proof-owner test registration checks to require active `RUN_TEST(...)` lines exactly once. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 13 review-hardening findings, boundary audit, and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day13-review-hardening.md` | Added the Day 13 review-hardening artifact. |

### Reviewed Invariants

| Invariant area | Day 13 disposition |
| --- | --- |
| Failed allocation status | Preserved by all 25 selected fail-after cases returning the expected allocation failure path. |
| Cleanup and free safety | Preserved by failure-output and success-output repeated-free helpers. |
| Stale-output suppression | Preserved by pre-seeded sentinel output tests across all 25 fail-after cases. |
| Caller-input preservation | Preserved by fixture shape, nnz, and value assertions after failure and success cleanup. |
| Retry-after-reset | Preserved by all-case retry and representative baseline-match retry tests. |
| Gate registration | Hardened to reject missing/commented-out proof-owner `RUN_TEST(...)` registrations. |
| Claim boundary | README, INSTALL, maintainer guide, and planning artifacts retain selected no-reorder linked-list LDLT wording and broad non-claims. |

### Retained Residuals

Day 13 keeps these residuals explicit:

- CSC LDLT allocation-failure proof;
- reordered LDLT allocation-failure proof;
- Cholesky, broad direct solvers, QR, SVD, eigensolver, sparse matrix
  construction, conversion, IO, package/install, or generated-tooling
  allocation-failure proof;
- operating-system OOM behavior;
- platform parity, hosted CI proof, package-manager proof, shared-library ABI
  proof, performance proof, release readiness, external-library parity, or
  state-of-the-art reliability support;
- concurrent allocation-hook behavior.

### Day 13 Validation

Commands run:

```sh
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make ldlt-linked-list-allocation-failure-gate
make docs-check
make support-docs-guard
git diff --check
```

Results:

- `python3 tests/test_ldlt_allocation_failure_gate_registration.py`: PASS;
  `ldlt-allocation-failure-gate-registration: passed`.
- `make ldlt-linked-list-allocation-failure-gate`: PASS; `95` LDLT tests,
  `0` failures, `0` skips, and `7781` assertions.
- `make docs-check`: PASS; Doxygen generation and API docs coverage completed
  with 18 checked-in public headers, 18 generated reference pages, 18
  generated source pages, and `sparse_version.h` kept under its separate
  installed-header policy.
- `make support-docs-guard`: PASS; support quick-reference docs guard passed.
- `git diff --check`: PASS.

## Day 14: Closeout Review

### Implementation Summary

Day 14 reconciled Sprint 210 item status, project-plan status, artifact
coverage, retained residuals, and final focused validation for retrospective
and PR review. Sprint 210 closes exactly one new selected allocation-failure
owner: no-reorder linked-list LDLT numeric factorization.

| Item | Final disposition | Evidence |
| --- | --- | --- |
| 210.1 Owner Selection | Complete | Day 2 selected linked-list LDLT numeric factorization after candidate ranking. |
| 210.2 Lifecycle Invariant Record | Complete | Day 3 recorded status, cleanup, stale-output, caller-input, retry, and boundary invariants. |
| 210.3 Harness Extension | Complete | Days 4-5 converted selected allocations to private allocation wrappers and added deterministic hook-based harness coverage. |
| 210.4 Regression Tests | Complete | Days 6-9 added 25-site failure sweep, cleanup proof, stale-output/caller-input preservation, and retry baseline-match tests. |
| 210.5 Gate And Documentation | Complete | Days 10-11 added focused Make/CTest gate, active registration guard, README/INSTALL/maintainer docs, and selected-only non-claims. |
| 210.6 Validation And Closeout | Complete | Days 12-14 recorded integrated validation, review hardening, final project-plan status, residuals, and focused closeout validation. |

### Final Changed Surfaces

| Surface | Closeout interpretation |
| --- | --- |
| `src/sparse_ldlt.c` | Selected linked-list LDLT output/workspace allocations now route through private allocation wrappers for deterministic failure injection. |
| `tests/test_ldlt.c` | Owns selected linked-list LDLT failure sweep, cleanup, stale-output, caller-input, retry, and success-cleanup tests. |
| `tests/test_ldlt_allocation_failure_gate_registration.py` | Guards focused gate wiring, CMake label, active proof-owner `RUN_TEST(...)` registrations, representative fail-after cases, and key assertions. |
| `Makefile` | Adds `ldlt-linked-list-allocation-failure-gate`. |
| `CMakeLists.txt` | Labels `test_ldlt` with `ldlt;linked_list;allocation_failure`. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Document the selected proof and retain broad non-claims. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Marks Sprint 210 closed with selected linked-list LDLT allocation-failure proof. |
| Sprint 210 planning artifacts | Provide day-by-day evidence for selection, invariants, implementation, tests, gate, docs, validation, hardening, and closeout. |

### Final Residuals

The following remain deferred after Sprint 210:

- CSC LDLT allocation-failure proof;
- reordered LDLT allocation-failure proof;
- Cholesky allocation-failure proof beyond existing selected symbolic lanes;
- broad direct-solver allocation-failure coverage;
- QR, SVD, eigensolver, sparse matrix construction, conversion, IO,
  package/install, or generated-tooling allocation-failure proof;
- operating-system OOM behavior;
- platform parity, hosted CI proof, package-manager proof, shared-library ABI
  proof, performance proof, release readiness, external-library parity, or
  state-of-the-art reliability support;
- concurrent allocation-hook behavior.

### Day 14 Validation

Commands run:

```sh
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make ldlt-linked-list-allocation-failure-gate
make docs-check
make support-docs-guard
git diff --check
```

Results:

- `python3 tests/test_ldlt_allocation_failure_gate_registration.py`: PASS;
  `ldlt-allocation-failure-gate-registration: passed`.
- `make ldlt-linked-list-allocation-failure-gate`: PASS; `95` LDLT tests,
  `0` failures, `0` skips, and `7781` assertions.
- `make docs-check`: PASS; Doxygen generation and API docs coverage completed
  with 18 checked-in public headers, 18 generated reference pages, 18
  generated source pages, and `sparse_version.h` kept under its separate
  installed-header policy.
- `make support-docs-guard`: PASS; support quick-reference docs guard passed.
- `git diff --check`: PASS.

## Day 9: Retry Proof

### Changed Surfaces

| File | Day 9 change |
| --- | --- |
| `tests/test_ldlt.c` | Added success-output comparison helper and representative/boundary retry baseline-match regression. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 9 retry proof and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day9-retry-proof.md` | Added the Day 9 retry-proof artifact. |

### Retry Proof Additions

| Addition | Evidence |
| --- | --- |
| Success-output comparison | `assert_ldlt_success_outputs_match(...)` compares retry output against an independently produced success baseline for `n`, `factor_norm`, `tol`, `L` dimensions/nnz/entries, `D`, `D_offdiag`, `pivot_size`, and `perm`. |
| Representative and boundary retry set | `test_ldlt_linked_list_retry_matches_success_baseline` covers output-array failures, working-copy setup, final working-copy scratch, output-`L` shell setup, permutation output, first workspace, and final workspace. |
| Hook and cleanup interaction | Each case fails once under the allocation hook, resets the hook, checks the failed output is free-safe, probes hook reset, retries the same fixture, and compares the successful retry against the baseline factor. |
| Caller input preservation | The retry test verifies the caller-owned fixture remains intact before and after retry success. |

### Retry Cases

| `fail_after` | Named site | Retry role |
| ---: | --- | --- |
| 0 | `D output array` | First selected output allocation boundary. |
| 1 | `D_offdiag output array` | Partial output allocation boundary. |
| 2 | `pivot_size output array` | Last initial output array before setup propagation. |
| 3 | `working copy entry buffer` | First propagated `sparse_copy(A)` setup allocation. |
| 11 | `working copy column-tail scratch` | Last working-copy scratch allocation before `L` construction. |
| 12 | `L row headers` | First output-`L` shell allocation. |
| 18 | `permutation output array` | Selected `ldlt->perm` output allocation. |
| 19 | `column accumulator workspace` | First selected dense workspace allocation. |
| 24 | `pivot-candidate nonzero list workspace` | Last selected dense workspace allocation. |

Day 9 keeps the broader all-25 retry test from Day 5/6 and adds stronger
baseline equality for representative and boundary points. It does not claim
retry behavior for CSC LDLT, reorder allocation failures, solve/refine/condest,
or broad direct-solver owners.

### Day 9 Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `95` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `7781` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.

## Day 8: Stale Output And Preservation

### Changed Surfaces

| File | Day 8 change |
| --- | --- |
| `tests/test_ldlt.c` | Added stale-output sentinel storage, seeding/assertion helpers, linked-list LDLT stale-output failure regression, and `used_csc_path` telemetry sentinel coverage. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 8 stale-output and preservation proof with validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day8-stale-output-preservation.md` | Added the Day 8 stale-output artifact. |

### Stale-Output Proof Additions

| Addition | Evidence |
| --- | --- |
| Pre-seeded output handle | `seed_ldlt_stale_output_sentinel(...)` fills `L`, `D`, `D_offdiag`, `pivot_size`, `perm`, `n`, `factor_norm`, and `tol` with non-success sentinel values before injected failure. |
| Sentinel assertion | `assert_ldlt_stale_output_sentinel_seeded(...)` verifies the test actually starts from stale-looking output slots. |
| Failure clears stale outputs | `expect_ldlt_stale_output_cleared_after_failure(...)` verifies every Day 6 allocation-failure site returns `SPARSE_ERR_ALLOC` and leaves the factor output empty after the public entrypoint resets it. |
| Telemetry sentinel | The stale-output test seeds `used_csc_path = -77`; forced linked-list failure must publish `0`, proving the selected telemetry slot is status-gated and not stale. |
| Caller-input preservation | Each stale-output failure case reuses `assert_ldlt_allocation_failure_input_intact(...)`, including dimensions, `nnz == 7`, and all fixture values. |

### Added Test

| Test | Evidence |
| --- | --- |
| `test_ldlt_linked_list_allocation_failures_clear_stale_outputs` | Runs all 25 fail-after sites with pre-seeded stale outputs, verifies stale fields are cleared, verifies linked-list telemetry, proves caller-owned `A` remains unchanged, and probes hook reset. |

### Boundary

The Day 8 proof follows the documented LDLT one-shot contract: factor
functions overwrite the output struct without freeing existing contents, and
callers must call `sparse_ldlt_free()` before reusing a populated object. The
sentinel test therefore uses static sentinel slots and proves they are cleared,
not freed or reused. This does not claim support for passing a live populated
factor object without first freeing it.

### Day 8 Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `94` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `6800` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.

## Day 7: Cleanup Proof

### Changed Surfaces

| File | Day 7 change |
| --- | --- |
| `tests/test_ldlt.c` | Added cleanup-focused assertions for caller-owned fixture shape/nnz, success-output double-free safety, repeated failure cleanup, and success teardown. Registered two cleanup tests. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Recorded Day 7 cleanup proof and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_210/artifacts/day7-cleanup-proof.md` | Added the Day 7 cleanup-proof artifact. |

### Cleanup Proof Additions

| Addition | Evidence |
| --- | --- |
| Caller-owned matrix `nnz` assertion | `assert_ldlt_allocation_failure_input_intact(...)` now verifies the fixture still has `7` nonzeros after every injected failure and after success cleanup. |
| Success-output teardown helper | `assert_ldlt_success_output_free_safe(...)` verifies successful outputs own `L`, `D`, `D_offdiag`, `pivot_size`, and `perm`, then `sparse_ldlt_free()` clears the handle and remains safe when repeated. |
| Repeated failure cleanup | `test_ldlt_linked_list_allocation_failure_cleanup_repeatable` runs every Day 6 failure site twice in reverse order, proving partial cleanup does not depend on test order or a previous clean state. |
| Success cleanup | `test_ldlt_linked_list_success_cleanup_free_safe` proves success-path teardown clears the handle without freeing or mutating caller-owned `A`. |

### Ownership Boundary

The selected linked-list LDLT owner owns only the `sparse_ldlt_t` outputs it
publishes and the temporary working objects it creates internally. The caller
continues to own the input `SparseMatrix *A`; all Day 7 failure and success
cleanup assertions verify `A` remains valid and unchanged. The cleanup proof
does not claim broad matrix-pool allocation ownership, CSC LDLT teardown,
reorder workspace cleanup, or solve/refine/condest cleanup.

### Day 7 Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `93` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `5925` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.
