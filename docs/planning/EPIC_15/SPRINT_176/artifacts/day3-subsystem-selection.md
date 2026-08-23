# Day 3: Subsystem Selection Decision

## Purpose

Choose the single subsystem for Sprint 176 deterministic allocation-failure
proof and define the exact API, setup path, allocation points, cleanup paths,
expected errors, out-of-scope surfaces, and pass/fail criteria.

## Decision

Sprint 176 selects the **iterative repeated-run workspace owner** as the single
allocation-failure proof target.

The selected proof scope is:

- public handle lifecycle APIs in `include/sparse_iterative.h`;
- handle setup/free implementation in `src/sparse_iterative.c`;
- internal workspace owner and typed view helpers in
  `src/sparse_iterative_workspace_internal.c` and
  `src/sparse_iterative_workspace_internal.h`;
- focused tests that exercise deterministic allocation failure during
  repeated-run handle preparation.

The proof is intentionally a shared-subsystem proof, not a broad iterative
solver correctness proof.

## Selection Rationale

| Criterion | Assessment |
| --- | --- |
| User value | Repeated-run iterative handles are public adoption-facing APIs for callers that want reusable solver workspace. |
| Allocation relevance | The subsystem owns reusable double and int buffers and grows capacity through allocation helpers. |
| Cleanup relevance | `sparse_iter_handle_free()` and `sparse_iter_workspace_free()` own the cleanup path and reset state after free. |
| Testability | The prepare path is narrow, deterministic, and wrapper-backed, making allocation-failure injection feasible without broad solver execution. |
| Blast radius | Lower than LU CSR, LDLT CSC, QR, matrix core, or partial SVD because the proof can target workspace preparation rather than numerical kernels. |
| Claim safety | The resulting claim can stay narrow: one repeated-run workspace subsystem has deterministic failure-path cleanup evidence. |

## Selected API And Setup Scope

| API or helper | File | In-scope behavior |
| --- | --- | --- |
| `sparse_iter_handle_init()` | `src/sparse_iterative.c` | Zero-initializes the public repeated-run handle. |
| `sparse_iter_handle_free()` | `src/sparse_iterative.c` | Frees internal workspace state and resets the handle to zero. |
| `sparse_iter_handle_prepare_cg()` | `src/sparse_iterative.c` | Allocates or reuses a repeated-run CG workspace for dimension `n`. |
| `sparse_iter_handle_prepare_gmres()` | `src/sparse_iterative.c` | Allocates or reuses a repeated-run GMRES workspace for dimension `n` and `restart`. |
| `sparse_iter_handle_prepare_minres()` | `src/sparse_iterative.c` | Allocates or reuses a repeated-run MINRES workspace for dimension `n`. |
| `s49_iter_handle_ensure()` | `src/sparse_iterative.c` | Allocates the internal workspace owner when the public handle is empty. |
| `sparse_iter_workspace_prepare_*()` | `src/sparse_iterative_workspace_internal.c` | Reserves typed workspace slices and reports allocation/overflow failures. |
| `sparse_iter_workspace_free()` | `src/sparse_iterative_workspace_internal.c` | Releases owned buffers and resets the internal workspace struct. |

## Deterministic Failure-Point Targets

Day 4 should design a harness that can prove these failure points:

| Target | Allocation point | Expected error | Required postcondition |
| --- | --- | --- | --- |
| Empty handle owner allocation | `s49_iter_handle_ensure()` allocating `sparse_iter_workspace_t` | `SPARSE_ERR_ALLOC` | Public handle remains zeroed; `sparse_iter_handle_free()` remains safe. |
| CG double workspace growth | `sparse_iter_workspace_prepare_cg()` reserving `4*n` doubles | `SPARSE_ERR_ALLOC` | Existing handle owner remains valid; no partially assigned CG view is treated as success. |
| GMRES double workspace growth | `sparse_iter_workspace_prepare_gmres()` reserving combined `v/h/cs/sn/g/y/w` storage | `SPARSE_ERR_ALLOC` | Existing capacity is not discarded on failed growth; handle can still be freed or reused at old capacity. |
| Block-CG int workspace growth, if chosen later | `sparse_iter_workspace_prepare_block_cg()` reserving convergence flags | `SPARSE_ERR_ALLOC` | Existing double workspace remains owned and cleanup-safe if int growth fails. |
| MINRES double workspace growth | `sparse_iter_workspace_prepare_minres()` reserving six or eight work vectors | `SPARSE_ERR_ALLOC` | Handle state remains cleanup-safe and no unsupported partial state is exposed. |

The first Sprint 176 proof should prioritize owner allocation plus one or two
public prepare paths. Block-CG may remain a stretch target because its public
repeated-run handle path is not the primary Day 3 scope.

## Cleanup And Ownership Invariants In Scope

The selected proof should establish these invariants:

- a failed prepare call returns `SPARSE_ERR_ALLOC` when allocation is injected
  to fail;
- an empty public handle remains empty when owner allocation fails;
- an already-prepared handle keeps its old workspace owner and capacity when a
  later growth allocation fails;
- `sparse_iter_handle_free()` is safe after failed prepare;
- repeated `sparse_iter_handle_free()` calls leave the public handle zeroed;
- successful prepare after clearing injection still works after a prior
  injected failure.

## Out Of Scope

Sprint 176 does not select these surfaces for implementation:

- LU CSR factorization or solve allocation failure;
- linked-list LU allocation failure;
- LDLT or LDLT CSC allocation failure;
- QR factorization or solve allocation failure;
- partial-SVD allocation failure;
- matrix pool or slab allocator failure;
- eigensolver workspace failure;
- broad solver cancellation or callback cleanup;
- external-library allocation behavior;
- package/install allocation behavior.

These remain future candidates or retained non-claims unless later Sprint 176
days explicitly narrow them as documentation-only residuals.

## Claim Boundary

The earned Sprint 176 claim, if implementation and validation pass, should be
no broader than:

> The iterative repeated-run workspace handle has deterministic
> allocation-failure cleanup evidence for selected prepare paths.

It must not imply:

- all iterative solvers are allocation-failure hardened;
- one-shot iterative solver calls have deterministic OOM proof;
- direct solvers, matrix core, QR, SVD, LDLT, LU CSR, or graph paths have OOM
  cleanup proof;
- broad memory-exhaustion behavior is guaranteed across platforms or
  allocators;
- state-of-the-art memory reliability.

## Pass/Fail Criteria For Later Implementation

| Criterion | Pass condition | Fail condition |
| --- | --- | --- |
| Determinism | Tests can force the selected allocation point to fail predictably. | Failure depends on real system memory pressure. |
| Error code | Public prepare returns `SPARSE_ERR_ALLOC` for injected allocation failure. | Failure returns `SPARSE_OK`, `SPARSE_ERR_BADARG`, or an unrelated code. |
| Cleanup | `sparse_iter_handle_free()` is safe after each injected failure. | Free crashes, leaks obvious ownership state, or leaves stale public handle data. |
| Reuse | Previously prepared capacity remains usable after failed growth, if that path is tested. | Failed growth destroys old capacity or corrupts later successful preparation. |
| Visibility | Injection remains internal/test-only. | Public headers expose unsupported allocator-control API. |
| Scope | Tests name selected repeated-run workspace paths. | Tests imply broad allocation-failure proof across all solvers. |

## Rejected Alternatives

| Alternative | Rejection reason |
| --- | --- |
| Partial SVD | Strong user-facing value, but more numerical and output-state complexity than needed for the first deterministic allocation-failure proof. Retain as a future candidate. |
| LU CSR | Highest allocation density, but many direct allocator calls make deterministic proof more invasive and higher risk. |
| LDLT CSC | High value but broad factorization cleanup surface with larger implementation risk. |
| QR | Active comparison family, but multiple factorization modes and ordering paths would make the first proof less focused. |
| Matrix core | Too central; broad matrix allocation failure would overrun the one-subsystem Sprint 176 scope. |
| Shared allocation wrappers only | Wrapper tests alone would not prove cleanup behavior in a real subsystem. |

## Day 3 Completion Record

- One subsystem is selected: iterative repeated-run workspace owner.
- Public APIs, internal setup paths, allocation points, and cleanup paths are
  listed.
- Expected allocation-failure behavior is testable and deterministic.
- Out-of-scope surfaces and retained non-claims prevent broad
  allocation-failure overclaiming.
