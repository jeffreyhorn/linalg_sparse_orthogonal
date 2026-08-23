# Day 8: Cleanup Invariant Documentation

## Purpose

Day 8 documents the selected Sprint 176 ownership and cleanup invariants
without widening the allocation-failure claim beyond the implemented evidence.

The selected subsystem remains:

- iterative repeated-run workspace handles;
- public handle type: `sparse_iter_handle_t`;
- public prepare paths: CG, GMRES, and MINRES;
- maintained proof owner: `tests/test_iterative.c`;
- focused Make gate: `make iterative-allocation-failure-gate`;
- focused CTest selector: `ctest -L allocation_failure`.

## Documentation Placement

The invariant belongs in three places:

| Surface | Update | Rationale |
| --- | --- | --- |
| `include/sparse_iterative.h` | Public lifecycle comment now states cleanup, invalid-argument, and bounded allocation-failure behavior. | Callers see the handle contract at the API boundary. |
| `README.md` | Repeated-run lifecycle section now states the bounded cleanup behavior and scope limit. | Adoption docs explain what users can rely on without reading tests. |
| `docs/maintainer_guide.md` | Proof-ownership list now names the exact tests and focused gates. | Maintainers can locate and preserve the evidence boundary. |

## Supported Behavior

For public iterative repeated-run handles:

- `sparse_iter_handle_free(NULL)` is safe.
- `sparse_iter_handle_free()` is safe on zero-initialized handles.
- repeated `sparse_iter_handle_free()` leaves the handle empty.
- invalid prepare arguments return `SPARSE_ERR_BADARG` without publishing
  internal handle state.
- selected allocation failures during owner allocation or workspace growth
  return `SPARSE_ERR_ALLOC`.
- selected allocation failures leave either an empty handle or the previously
  usable handle capacity intact.
- after the private test hook is reset, the selected failed prepare/growth
  paths can recover through a later successful prepare.

## Non-Claims

The documentation intentionally does not claim:

- broad allocation-failure cleanup coverage across all solvers;
- broad allocation-failure coverage for matrix construction, direct solvers,
  eigensolvers, graph/reorder paths, package/install flows, or generated-report
  tooling;
- public support for allocator fault injection;
- allocation-failure coverage for BiCGSTAB or block iterative repeated-run
  handles;
- proof that every `SPARSE_ERR_ALLOC` path preserves prior state in every
  subsystem.

## Regression Proof

The maintained proof lives in `tests/test_iterative.c` through
`tests/test_iterative_handle_helpers.h`:

- `test_iter_handle_owner_allocation_failure_leaves_handle_empty`
- `test_cg_handle_workspace_allocation_failure_recovers`
- `test_iter_handle_invalid_prepare_calls_do_not_publish_state`
- `test_gmres_handle_growth_allocation_failure_preserves_existing_workspace`
- `test_minres_handle_growth_allocation_failure_preserves_existing_workspace`

Focused validation:

```sh
make iterative-allocation-failure-gate
```

Result: passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.
