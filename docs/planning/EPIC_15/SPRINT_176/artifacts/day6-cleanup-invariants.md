# Day 6: Cleanup Invariants

## Purpose

Strengthen and verify ownership cleanup behavior exposed by the Day 5
allocation-failure harness for the selected subsystem: iterative repeated-run
handle workspace ownership.

## Defect Found

Day 6 found one cleanup-surface defect in the selected subsystem:

| Path | Previous behavior | Fix |
| --- | --- | --- |
| `sparse_iter_handle_prepare_gmres(&handle, n, restart <= 0)` | Validated `restart` after ensuring the private workspace owner, so an invalid prepare call could publish a non-NULL `handle.internal_state`. | Validate `restart <= 0` before `s49_iter_handle_ensure()` allocates the owner. |

This was not a leak, because `sparse_iter_handle_free()` still cleaned the
published owner, but it violated the stronger invariant that invalid prepare
arguments should not publish partial handle state.

## Invariants Enforced

Day 6 adds explicit assertions for these selected-subsystem invariants:

- `sparse_iter_handle_free(NULL)` is safe.
- CG prepare with invalid `n` does not allocate or publish handle state.
- MINRES prepare with invalid `n` does not allocate or publish handle state.
- GMRES prepare with invalid `restart` does not allocate or publish handle
  state.
- The invalid-argument checks still hold while allocation failure injection is
  armed.
- A handle that remains empty after invalid calls can later prepare GMRES
  successfully after the allocation hook is reset.
- Repeated cleanup after recovery leaves `handle.internal_state == NULL`.

## Test Change

Added `test_iter_handle_invalid_prepare_calls_do_not_publish_state` to
`tests/test_iterative_handle_helpers.h` and registered it in the existing
`test_iterative` executable.

The test deliberately arms the allocation-failure hook before invalid prepares.
That confirms the invalid-argument paths return before attempting allocation.

## Success-Path Regression Notes

The existing public repeated-run handle tests continue to cover successful
prepare, reuse, on-demand growth, and solve behavior for CG, GMRES, and MINRES.
The Day 6 change only moves GMRES restart validation earlier; valid GMRES
prepare and solve paths remain unchanged.

## Focused Validation

Command:

```sh
make build/test_iterative && build/test_iterative
```

Result:

- `Tests run: 85`
- `Tests failed: 0`
- `Tests skipped: 0`
- `Assertions: 743`

## Required Full Gate

Day 6 modifies `.c` and `.h` files, so the required quality gate is:

```sh
make format && make lint && make test
```

Result: passed.
