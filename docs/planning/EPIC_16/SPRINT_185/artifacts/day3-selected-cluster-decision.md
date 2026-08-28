# Sprint 185 Day 3: Selected Cluster Decision

## Decision

Sprint 185 selects exactly one active review-surface reduction cluster:

`tests/test_ldlt_csc.c`

The selected cluster is large enough to matter, has a clear family-local proof
owner, and can likely be reduced by extracting helpers while preserving the
existing `test_ldlt_csc` binary and its registration.

## Selection Rationale

| Criterion | Evidence |
| --- | --- |
| Review cost | `tests/test_ldlt_csc.c` is 3915 lines and has 130 static/function entries. |
| Extraction feasibility | Several helper families are already separable: KKT builders, dense/symmetric fixture builders, supernode snapshot/assertion helpers, dense symmetric-swap oracles, residual helpers, and native-wrapper comparison helpers. |
| Behavior-preservation confidence | The first-choice extraction can stay header-only and family-local, keeping all `RUN_TEST(...)` calls in the original binary. |
| Registration risk | Existing Make/CMake registration can remain unchanged if extraction uses an included test helper header. |
| Sprint fit | The cluster is a large direct-solver test surface, matching Sprint 185's test/source review-surface reduction goal. |

## Baseline Inventory

| Field | Baseline |
| --- | --- |
| Selected file | `tests/test_ldlt_csc.c` |
| Current line count | 3915 |
| Static/function count | 130 |
| Test binary | `test_ldlt_csc` |
| Make registration | `Makefile` `TEST_SRCS` includes `$(TESTDIR)/test_ldlt_csc.c` |
| CMake registration | `CMakeLists.txt` includes `add_sparse_test(test_ldlt_csc)` |
| Related library sources | `src/sparse_ldlt_csc.c`, `src/sparse_ldlt_csc_rowadj.c`, `src/sparse_ldlt_csc_supernodal.c` |
| Library source manifest | `build-metadata/library_sources.txt` already includes the LDLT CSC library sources |

## Selected Responsibilities

The selected file currently owns these proof surfaces:

- LDLT CSC allocation/free behavior.
- Row-adjacency append, growth, argument checks, and swap-state behavior.
- 2x2-aware supernode detection.
- Supernode extract/writeback round trips and argument checks.
- Supernodal LDLT factor cross-checks against scalar/native behavior.
- `ldlt_csc_from_sparse` and `ldlt_csc_from_sparse_with_analysis` coverage.
- KKT and random-indefinite fixtures for analysis-backed supernodal tests.
- External dense-reference checks and Windows skip behavior.
- Permutation, validation, elimination, native kernel, symmetric swap, solve,
  and inertia coverage.

## Candidate Helper Seams For Day 4

| Candidate seam | Why it fits | Day 4 question |
| --- | --- | --- |
| KKT and dense fixture builders | Fixture construction is repeated and independent from `RUN_TEST(...)` ownership. | Should these move into a single `tests/test_ldlt_csc_fixtures.h` or a broader `tests/test_ldlt_csc_helpers.h`? |
| Supernode state helpers | `build_dense_ldlt_with_pivots`, `cm_idx`, and `snapshot_supernode_state` support one local proof family. | Should they stay near supernode tests for locality or move as the first low-risk extraction? |
| External dense-reference state helpers | The state struct, allocation/free helpers, Python-reference reader, and assertion wrapper are cohesive. | Is the `_POSIX_C_SOURCE` and Windows skip dependency too sensitive for the first extraction pass? |
| Symmetric-swap dense oracle helpers | Dense lower-copy, symmetric dense swap, dense compare, and triple-builder helpers form a distinct oracle group. | Should this wait until after KKT/supernode helper movement proves the pattern? |
| Native wrapper comparison helpers | `ldlt_column_nonzeros_match`, `ldlt_factorizations_match`, and `check_native_matches_wrapper` are cohesive. | Are these better left local because many native tests read naturally with them nearby? |

## Deferred Alternatives

| Candidate | Decision | Reason |
| --- | --- | --- |
| `tests/test_svd.c` | Deferred alternate | Good helper-header precedent, but lower current review cost and broader full/partial SVD ownership. |
| `tests/test_graph.c` | Deferred fallback | Existing fixtures help, but graph/FM environment behavior is less direct-solver-local. |
| `tests/test_qr.c` | Deferred | Largest file, but recent QR work and existing QR proof-owner extraction increase review sensitivity. |
| `tests/test_integration.c` | Deferred | Cross-solver integration scope is too broad for the first Sprint 185 extraction. |
| `tests/test_iterative.c` | Deferred | Allocation-failure proof ownership increases behavior-preservation risk. |
| `src/sparse_ldlt_csc.c` | Deferred | Implementation extraction requires source-list registration and carries higher behavior risk than test helper extraction. |
| Python/report/documentation surfaces | Deferred | Lower fit for the selected solver/test review-surface reduction objective. |

## No-Behavior-Change Contract

- Preserve every existing test name and `RUN_TEST(...)` entry.
- Preserve fixture matrices, random seeds, assertion thresholds, skip behavior,
  diagnostic messages, and external reference command behavior.
- Preserve `_POSIX_C_SOURCE` placement and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` usage.
- Preserve process-global LDLT CSC kernel override reset behavior.
- Do not change public APIs, internal solver APIs, production `.c` files, or
  library source-list registration as part of helper extraction.
- If a new proof-owner binary becomes necessary, record the rationale first
  and update Make/CMake registration together.

## Baseline Validation Commands

Before and after mechanical extraction, use:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

If any `.c` or `.h` file changes, also run:

```sh
make format && make lint && make test
```

If a new test binary is added, also run:

```sh
make quality-review-cmake-compile
```

If a library source file is added, also run:

```sh
make source-list-check
```

## Days 4-8 Extraction Checklist

1. Day 4: inspect exact helper blocks and choose the helper-header boundary.
2. Day 5: confirm registration impact and existing guard coverage before code
   movement.
3. Day 6: move the first low-risk helper group and run focused validation.
4. Day 7: move the next approved fixture/setup group only if Day 6 is clean.
5. Day 8: clean call sites, declarations, and helper ordering while preserving
   the existing proof-owner binary.

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.
