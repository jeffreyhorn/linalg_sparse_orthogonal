# Sprint 210 Day 4: Harness Design

## Purpose

Define the deterministic allocation-failure harness for the selected
linked-list LDLT owner before source or test implementation. This design
identifies fixture data, allocation-hook reachability work, helper APIs,
failure sweep boundaries, focused gate wiring, and registration guard needs.

## Harness Decision

Reuse the existing private allocation hook:

- `sparse_alloc_test_fail_after(...)`;
- `sparse_alloc_test_reset()`;
- `sparse_malloc_array(...)`;
- `sparse_calloc_array(...)`;
- `sparse_malloc_idx_array(...)`;
- `sparse_calloc_idx_array(...)`.

No public allocator API, environment variable, or new test framework is needed.
The selected linked-list LDLT proof does require a narrow wrapper-conversion
pass for direct allocations inside `ldlt_factor_internal()`.

## Selected Fixtures

| Fixture | Purpose | Success baseline |
| --- | --- | --- |
| 3x3 SPD tridiagonal LDLT | Primary low-noise fixture for factor arrays, `L`, `perm`, dense workspaces, and solve verification. | Linked-list factor succeeds, owned fields are non-null, solve residual matches `b`. |
| 2x2 indefinite LDLT | Metadata fixture for 2x2 pivot publication and `D_offdiag`. | Linked-list factor succeeds with 2x2 pivot metadata and solve success. |
| 3x3/4x4 mixed pivot LDLT | Optional expansion if primary fixtures do not reach elimination insertion or swap behavior. | Linked-list factor succeeds and reconstruction or solve verification passes. |

The Day 5 implementation should start with the 3x3 SPD tridiagonal fixture and
add the 2x2 indefinite fixture only when needed for output metadata coverage.

## Wrapper Reachability

| Site | Current form | Day 5 action | Boundary |
| --- | --- | --- | --- |
| `ldlt->D` | direct `calloc` | Use `sparse_calloc_idx_array`. | Selected LDLT output allocation. |
| `ldlt->D_offdiag` | direct `calloc` | Use `sparse_calloc_idx_array`. | Selected LDLT output allocation. |
| `ldlt->pivot_size` | direct `calloc` | Use `sparse_calloc_idx_array`. | Selected LDLT output allocation. |
| `ldlt->perm` | direct `malloc` | Use `sparse_malloc_idx_array`. | Selected LDLT output allocation. |
| `col_acc`, `col_acc_r` | direct `calloc` | Use `sparse_calloc_idx_array`. | Selected LDLT temporary workspace. |
| `nz_flag`, `nz_flag_r` | direct `calloc` | Use `sparse_calloc_idx_array`. | Selected LDLT temporary workspace. |
| `nz_list`, `nz_list_r` | direct `malloc` | Use `sparse_malloc_idx_array`. | Selected LDLT temporary workspace. |
| `sparse_copy(A)` | existing matrix path | Keep as propagated setup failure if observed. | Not broad matrix-copy proof. |
| `sparse_create()` and `sparse_insert()` for `ldlt->L` | existing matrix path | Keep as propagated output-construction failure if observed. | Not broad matrix-construction proof. |
| swap-helper temporaries | direct helper `malloc` | Defer unless naturally reached after wrapper conversion. | Not helper-level swap proof unless explicitly added later. |

## Failure Sweep Boundary

Named failure cases should be confirmed after wrapper conversion by observing
fail-after indices. The intended families are:

| Family | Intended coverage |
| --- | --- |
| Output arrays | `D`, `D_offdiag`, `pivot_size`. |
| Matrix setup propagation | `sparse_copy(A)`, `sparse_create(n, n)`, identity `sparse_insert`. |
| Output permutation | `perm`. |
| Dense workspaces | `col_acc`, `nz_flag`, `nz_list`, `col_acc_r`, `nz_flag_r`, `nz_list_r`. |
| Elimination insertion | selected `sparse_insert(ldlt->L, i, k, ...)` failures if reached. |

The initial selected proof should not count reorder allocation, CSC LDLT
allocation, solve/refine/condest workspaces, QR/SVD/eigs workspaces, or matrix
construction generally as closed.

## Test Helper Plan

| Helper | Responsibility |
| --- | --- |
| `make_ldlt_allocation_failure_matrix()` | Build the selected matrix fixture outside the injected failure window. |
| `assert_ldlt_failure_output_empty(...)` | Assert no success-looking LDLT output remains after failure. |
| `assert_ldlt_failure_output_free_safe(...)` | Prove repeated `sparse_ldlt_free()` is safe after failure. |
| `assert_ldlt_input_matrix_intact(...)` | Verify caller-owned matrix preservation. |
| `assert_ldlt_success_baseline(...)` | Verify fresh success output and solve/reconstruction behavior. |
| `expect_ldlt_allocation_failure(...)` | Execute one fail-after case with reset-before-assertion discipline. |
| `expect_ldlt_allocation_failure_recovers(...)` | Prove retry success after hook reset. |
| `assert_allocation_hook_probe_after_reset()` | Detect leaked allocation-hook state after each case. |

## Focused Gate Plan

| Surface | Decision |
| --- | --- |
| Make target | Prefer `ldlt-linked-list-allocation-failure-gate` to avoid broad LDLT/CSC claims. |
| Test binary | Use existing `tests/test_ldlt.c`. |
| CTest label | Add or verify `ldlt;allocation_failure` after tests exist. |
| Registration guard | Add `tests/test_ldlt_allocation_failure_gate_registration.py` to require the Make target, label if present, and selected `RUN_TEST(...)` entries. |
| Source-list impact | No new C source is expected; only existing test binary and optional Python guard wiring should change. |

## Hook Cleanup Rules

1. Reset hooks before each fail-after setup.
2. Store status locally.
3. Reset hooks before any assertion macro can early return.
4. Reset again before retry calls.
5. Free LDLT outputs and matrices after hook reset.
6. Keep fixture creation outside the injected failure window unless explicitly
   proving setup behavior.

## Day 5 Handoff

Day 5 should implement the narrow wrapper conversions in `ldlt_factor_internal`
and add the initial LDLT allocation-failure fixture/helpers to
`tests/test_ldlt.c`. It should establish observed fail-after indices before
claiming the exact failure case list and avoid documentation claim updates
until Day 11.

## Validation

Commands run:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 4 changes planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

