# Sprint 201 Day 3: Selected Cluster Boundary

## Summary

Day 3 selects exactly one Sprint 201 review-surface reduction target and freezes
the no-behavior-change boundary before extraction begins.

Selected target:

- `tests/test_svd.c` rank, pseudoinverse, and low-rank helper cluster.

The selected cluster keeps `tests/test_svd.c` as the proof-owner binary and uses
the existing SVD helper surface as the preferred extraction destination. The
intended implementation shape is header-only helper extraction or tightly
coupled helper/test relocation that reduces the large `tests/test_svd.c` review
surface without changing solver behavior, public API, ABI, test registrations,
or expected output.

## Selection Decision

| Decision Field | Selected Value |
| --- | --- |
| Selected source surface | `tests/test_svd.c` |
| Selected helper surface | `tests/test_svd_selected_helpers.h`; shared fixtures remain in `tests/test_svd_helpers.h`. |
| Selected proof owner | `test_svd` executable remains the proof owner |
| Selected cluster | rank, pseudoinverse, and low-rank SVD tests and their local helper logic |
| Implementation preference | header-only helper extraction using existing family-local test helper patterns |
| Build registration preference | no Makefile or CMake registration changes |
| Public behavior posture | no behavior change |

## Why This Cluster

The selected SVD cluster is the highest-value Sprint 201 target because:

- `tests/test_svd.c` remains one of the largest current C test review surfaces;
- the rank, pseudoinverse, and low-rank tests form a cohesive non-partial SVD
  family with shared fixture and numerical helper needs;
- `tests/test_svd_helpers.h` already exists as the family-local helper surface;
- the change can preserve `RUN_TEST(...)` ownership inside `tests/test_svd.c`;
- production solver code and public headers can remain untouched;
- focused `test_svd` validation can prove behavior preservation before broader
  project gates.

## In-Scope Map

| Area | In Scope |
| --- | --- |
| Source proof owner | `tests/test_svd.c` remains the registered test binary. |
| Helper destination | `tests/test_svd_selected_helpers.h` may receive selected helper definitions for the chosen cluster while reusing shared fixtures from `tests/test_svd_helpers.h`. |
| Rank tests | `test_svd_rank_full`, `test_svd_rank_deficient`, `test_svd_rank_nearly_singular`, `test_svd_rank_diagonal_threshold_fixture`, `test_svd_qr_rank_dependent_row_fixture`, and `test_svd_rank_null`. |
| Pseudoinverse tests | `test_pinv_diagonal`, `test_pinv_moore_penrose`, `test_pinv_null`, `test_pinv_rectangular`, and `test_pinv_underdetermined_minnorm_solution`. |
| Low-rank dense tests | `test_lowrank_diagonal`, `test_lowrank_error_bound`, and `test_lowrank_errors`. |
| Helper logic | Cluster-local fixture builders, reconstruction checks, Moore-Penrose checks, dense low-rank error helpers, and assertion wrappers used only by the selected tests. |
| Test registrations | Existing `RUN_TEST(...)` entries stay in `tests/test_svd.c` and retain their current order. |
| Documentation artifacts | Day 4+ invariants, implementation notes, guard notes, and final validation evidence for Sprint 201. |

## Out-Of-Scope Map

| Area | Out Of Scope |
| --- | --- |
| Production implementation | `src/sparse_svd.c`, `src/sparse_svd_partial.c`, and other production solver sources. |
| Public API and ABI | `include/sparse_svd.h` and all other public headers, exported names, struct layouts, enum values, status codes, and caller contracts. |
| Partial SVD surfaces | `tests/test_svd_partial_helpers.h`, `tests/test_svd_partial_shared_helpers.h`, partial SVD vector tests, partial residual tests, and partial low-rank optimality tests. |
| External reference surfaces | External dense-reference fixture readers, corpus metadata, and generated reference comparison semantics. |
| Golub-Kahan and bidiagonal tests | GK extraction, GK orthogonality, bidiagonal SVD, and related early-file helper families. |
| Condition-number tests | `test_cond_*` tests and condition-number helper behavior. |
| Sparse low-rank extended tests | Later sparse low-rank outer-product and corpus-safety proof blocks are protected unless Day 4 finds they are inseparable from the dense cluster. |
| Build registrations | Makefile, CMake, source-list, and CI registration files unless an ownership guard later proves a registration edit is required. |
| Claims and release posture | Package-manager, platform, performance, release-readiness, and state-of-the-art claims. |

## Frozen Behavior Boundary

The selected extraction must preserve:

- public API and ABI exactly;
- production source behavior exactly;
- test names and `RUN_TEST(...)` order exactly;
- fixture dimensions, input values, deterministic construction, and matrix
  sparsity patterns exactly;
- numerical tolerances and threshold comparisons exactly;
- expected status codes, assertion behavior, and emitted diagnostic text;
- skip behavior and unsupported-environment behavior exactly;
- cleanup order, allocation ownership, and failure-path behavior exactly;
- `test_svd` as the single proof-owner executable for the selected tests.

Any change that requires a tolerance update, fixture data update, expected-output
update, new public symbol, production solver edit, or proof-owner split is
outside the Day 3 boundary and must be treated as a separate decision.

## Focused Validation Checklist

After implementation, Sprint 201 should validate the selected cluster with:

| Check | Purpose |
| --- | --- |
| `git diff --check` | Catch whitespace and conflict-marker issues in every Day 201 artifact or code edit. |
| `make build/test_svd` or equivalent rebuild path | Confirm the selected proof-owner binary still compiles after helper movement. |
| `./build/test_svd` | Prove the SVD rank, pseudoinverse, and low-rank tests still pass in their original proof owner. |
| `make source-list-check` | Confirm no unintended source-registration drift if any registration file changes. |
| `make format && make lint && make test` | Required once `.c` or `.h` files are modified. |

If the implementation remains documentation-only on a given day, the full C gate
is not required for that day. Once helper extraction modifies `.h` files, the
full C quality gate becomes mandatory before the sprint branch is finalized.

## Completion Evidence

Item 201.2 is complete for Day 3: one cluster has been selected, the
in-scope/out-of-scope map is frozen, and the no-behavior-change boundary is
explicit. Day 4 should turn this boundary into extraction invariants and guard
requirements before code movement begins.
