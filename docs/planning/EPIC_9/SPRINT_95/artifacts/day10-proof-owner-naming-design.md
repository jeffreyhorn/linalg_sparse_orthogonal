# Sprint 95 Day 10: Proof-Owner Naming Design

## Purpose

Day 10 decides which sprint-named proof owners should move toward
product-oriented names before any files move. The goal is not to erase
historical regression context; it is to make the maintained proof surface easier
to find by capability.

## Audit Summary

The current sprint-named integration owners are:

| Owner | Current capability coverage | Day 10 classification |
|---|---|---|
| `tests/test_sprint4_integration.c` | Cholesky, CSR round-trip, solve integration, thread-gated by build files. | Historical regression owner; defer. |
| `tests/test_sprint5_integration.c` | Iterative/preconditioner and direct-solver cross checks. | Historical regression owner; defer. |
| `tests/test_sprint6_integration.c` | QR plus cross-solver SuiteSparse checks. | Historical regression owner; defer. |
| `tests/test_sprint8_integration.c` | SVD and matrix-free / low-rank cross checks. | Historical regression owner; defer. |
| `tests/test_sprint10_integration.c` | CSR LU, block solvers, packaging/version compatibility. | Candidate later split; defer because it mixes unrelated lanes. |
| `tests/test_sprint11_integration.c` | Tolerance, factored-state, thread-safe norm, version consistency. | Candidate later regrouping; defer because `test_edge_cases.c` already owns much of the product concept. |
| `tests/test_sprint12_integration.c` | LDL^T factorization integration with tolerance, reordering, inertia, refinement, condition estimation, SuiteSparse. | Historical LDL^T integration owner; defer behind direct-family naming. |
| `tests/test_sprint13_integration.c` | IC(0), MINRES, block MINRES, KKT and SuiteSparse validation. | Candidate later iterative/preconditioner regrouping; defer. |
| `tests/test_sprint18_integration.c` | Cholesky CSC dispatch threshold, forced backend parity, native LDL^T CSC smoke. | Rename/regroup candidate. |
| `tests/test_sprint19_integration.c` | Analyze/refactor smoke, CSC threshold lock, Kuu scalar CSC regression, row-adjacency, supernodal LDL^T parity. | Rename/regroup candidate, but only as part of direct CSC cluster. |
| `tests/test_sprint20_integration.c` | LDL^T backend selector, AUTO dispatch, forced backend routing, eigensolver early integration. | Rename/regroup candidate, but keep eigensolver helper scope explicit. |
| `tests/test_sprint29_integration.c` | Cross-feature SVD, eigs refinement, progress/cancel interactions. | Historical cross-feature owner; defer. |

## Public And Maintainer References

| Surface | Current reference pattern | Rename impact |
|---|---|---|
| `Makefile` | Lists every `test_sprint*_integration.c` in `TEST_SRCS`; has a special pthread rule for `test_sprint4_integration`. | Any file rename must update `TEST_SRCS`; `test_sprint4_integration` should not move in the Day 11 batch. |
| `CMakeLists.txt` | Uses `add_sparse_test(test_sprint*_integration)` for CTest target names. | File renames change CTest target names unless aliases are added; Day 11 should keep the batch small. |
| `.github/workflows/windows-ci.yml` | Mentions `test_sprint4_integration` in staged Windows exclusions. | Out of Day 11 scope because Sprint 4 is not selected. |
| `tests/test_reorder_nd.c` | Mentions helper provenance from `tests/test_sprint10_integration.c`. | Out of Day 11 scope; preserve until Sprint 10 owner is renamed or split. |
| `docs/maintainer_guide.md` | Names product-oriented owners such as `tests/test_chol_csc.c`, `tests/test_integration.c`, and benchmark owners; still carries historical sprint context where policy needs it. | Day 11 should update maintainer wording only where selected names change. |
| README, tutorial, examples, install docs | Do not directly reference selected `test_sprint18/19/20` filenames. | No user-facing adoption-doc update expected for the selected batch. |

## Rename And Regrouping Rules

1. Rename only when the new name maps to a stable capability, not a transient
   sprint deliverable.
2. Preserve build behavior:
   - `make test` must still compile and run the same coverage.
   - CMake/CTest target names must remain coherent with the new file names.
   - special platform gates, such as pthread or Windows exclusions, must remain
     attached to the same tests.
3. Do not rename a file that mixes several unrelated capability families unless
   the Day 11 batch also splits or clearly documents the retained mixed owner.
4. Keep historical sprint names in planning artifacts and old captured logs.
   Do not rewrite historical evidence.
5. Update active references in `Makefile`, `CMakeLists.txt`, test comments,
   maintainer docs, and any workflow scripts in the same batch.
6. Prefer one capability cluster per batch. If a rename touches direct solvers,
   do not also rename iterative, eigensolver, or benchmark owners in that batch.
7. Keep public benchmark targets and CLI options stable unless the sprint has a
   separate compatibility decision.

## Selected Day 11 Batch

The smallest high-value batch is the direct CSC dispatch proof-owner cluster:

| Current owner | Proposed product-oriented owner | Reason |
|---|---|---|
| `tests/test_sprint18_integration.c` | `tests/test_direct_csc_dispatch.c` | The file primarily proves Cholesky CSC threshold dispatch and forced backend parity through public options. |
| `tests/test_sprint19_integration.c` | `tests/test_direct_csc_regression.c` | The file is a retained direct-family CSC regression bundle: threshold lock, Kuu scalar path, row-adjacency, and supernodal LDL^T parity. |
| `tests/test_sprint20_integration.c` | `tests/test_ldlt_backend_dispatch.c` | The file primarily proves the public LDL^T backend selector and AUTO/forced dispatch behavior; eigensolver helper tests remain a documented residual inside the renamed owner unless split later. |

Day 11 should update:

- `Makefile` `TEST_SRCS`
- `CMakeLists.txt` `add_sparse_test(...)` entries
- `TEST_SUITE_BEGIN(...)` names in the renamed files
- file-header comments in the renamed files
- active documentation references if any new names are cited

Day 11 should run at least:

```bash
make format
make lint
make test
```

Because the selected batch renames `.c` files and changes build hooks, the full
quality chain is required.

## Deferred Naming Work

| Deferred owner | Why deferred |
|---|---|
| `test_sprint4_integration` | Thread-gated and Windows-exclusion referenced; rename would add platform-policy churn. |
| `test_sprint5_integration` / `test_sprint6_integration` / `test_sprint8_integration` | Older mixed cross-feature owners; product splits need a broader design than Day 11. |
| `test_sprint10_integration` | Mixes CSR LU, block solver, package/version, and compatibility checks; should be split before or during rename. |
| `test_sprint11_integration` / `test_sprint12_integration` / `test_sprint13_integration` | Have product themes but overlap existing product-oriented suites; defer until direct-family cluster lands. |
| `test_sprint29_integration` | Cross-feature interaction owner; useful historical bundle until progress/cancel and SVD/eigs interaction owners are redesigned. |
| Benchmark target names such as `bench-reorder-sprint86` | Public or maintainer workflow compatibility risk; not proof-owner test cleanup. |
| Planning artifacts and old captured logs | Historical evidence should remain chronological. |

## Completion Criteria Check

- Bounded proof naming plan exists before files move.
- Public and maintainer references are included in scope.
- Churn-only renames are rejected explicitly.
- The selected batch is small enough for Day 11 validation and review.
