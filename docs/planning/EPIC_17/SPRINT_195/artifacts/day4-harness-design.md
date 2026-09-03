# Sprint 195 Day 4: Harness Design

## Purpose

Design the deterministic failure-injection harness for the selected
`sparse_symbolic_cholesky()` reliability proof before code or test changes.

## Decision

Use the existing private allocation-failure hook:

- `sparse_alloc_test_fail_after(...)`;
- `sparse_alloc_test_reset()`;
- `sparse_malloc_array`;
- `sparse_calloc_array`;
- `sparse_malloc_idx_array`;
- `sparse_calloc_idx_array`.

Do not add a new owner-local fail-at-count global. The selected direct
`sym->col_ptr` allocation should instead be converted to
`sparse_malloc_array(col_ptr_len, sizeof(idx_t), ...)` so all claimed
allocation classes are driven by one private hook model.

## Harness Scope

| Included | Excluded |
| --- | --- |
| `sparse_symbolic_cholesky()` selected allocation classes. | `sparse_symbolic_lu()`, `sparse_analyze()`, and broader etree failure paths. |
| Existing `test_etree` proof-owner binary. | New public API, new allocator framework, or new solver proof-owner binary. |
| Focused Make gate and registration guard. | Broad allocation-failure gate claiming all symbolic or all direct-solver paths. |
| Optional CTest label for `test_etree` if Day 10 chooses CMake selector parity. | Hosted CI promotion unless a later day explicitly scopes it. |

## Required Implementation Shape

1. Replace the direct `malloc(col_ptr_bytes)` in
   `sparse_symbolic_cholesky()` with a wrapper-controlled allocation.
2. Preserve the current overflow checks, `sym` zeroing, cleanup calls, status
   values, and success output layout.
3. Add tests to `tests/test_etree.c`; do not create a new C test binary unless
   implementation proves the existing binary cannot own the focused proof.
4. Add `symbolic-allocation-failure-gate` to the Makefile.
5. Add `tests/test_symbolic_allocation_failure_gate_registration.py` to check
   the Make target and required `RUN_TEST(...)` entries.
6. Decide on CMake `test_etree` labels during Day 10 gate definition; no
   source-list change is expected.

## Reset Contract

Any helper that arms the hook must reset it before assertions:

```c
sparse_alloc_test_reset();
sparse_alloc_test_fail_after(fail_after);
sparse_err_t err = sparse_symbolic_cholesky(A, parent, postorder, cc, &sym);
sparse_alloc_test_reset();
ASSERT_ERR(err, SPARSE_ERR_ALLOC);
```

The reset must happen even when the selected call unexpectedly succeeds or
returns a different error. Tests should use local status variables and cleanup
labels where needed so assertion macros cannot leave the process-global hook
armed.

## Planned Test Cases

| Planned test | Purpose |
| --- | --- |
| `test_symbolic_cholesky_allocation_failure_clears_stale_output` | Force early selected allocation failure after seeding `sym` with stale owned fields; assert failure status and cleared output. |
| `test_symbolic_cholesky_allocation_failures_clear_partial_state` | Force non-empty `col_ptr`, `row_idx`, workspace, column-row workspace, and propagated row-set allocation failures; assert zeroed output after each. |
| `test_symbolic_cholesky_allocation_failure_recovers` | Force one representative failure, reset the hook, rerun the known 5x5 fixture, and compare successful symbolic output with existing expected rows. |

## Build and Validation Ownership

| File or target | Expected change |
| --- | --- |
| `src/sparse_etree.c` | Bounded wrapper conversion only. |
| `tests/test_etree.c` | Add selected allocation-failure tests and `RUN_TEST(...)` entries. |
| `Makefile` | Add `symbolic-allocation-failure-gate`. |
| `CMakeLists.txt` | Optional `test_etree` label update; no new binary expected. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | New registration guard modeled on the matmul guard. |
| `build-metadata/library_sources.txt` | No change expected. |
| `docs/maintainer_guide.md` | Later documentation update for focused gate and non-claims. |

## Review Risks

| Risk | Control |
| --- | --- |
| Wrapper conversion alters behavior. | Keep the same byte-count validation and allocate the same number of `idx_t` elements. |
| Hook state leaks after early assertion return. | Store status, reset, then assert. Use cleanup labels for multi-step helpers. |
| Countdown values become fragile. | Name fail cases by allocation class and keep tests close to the selected owner. |
| Focused gate overclaims. | Name and document it as symbolic Cholesky allocation-failure proof only. |
| CTest label semantics broaden silently. | If labels change, update maintainer docs and registration guard together. |

## Day 5 Handoff

Day 5 should implement the minimal scaffold:

1. convert `sym->col_ptr` to wrapper allocation;
2. add small helper fixtures and hook-reset helpers in `tests/test_etree.c`;
3. add smoke-level tests proving the hook reaches the selected owner;
4. add initial Make/guard wiring only if the scaffold compiles cleanly;
5. run the smallest focused build/test path.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

`git diff --check` passes.
