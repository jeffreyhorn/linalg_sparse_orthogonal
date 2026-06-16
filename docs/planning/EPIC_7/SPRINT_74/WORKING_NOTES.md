# Sprint 74 Working Notes

## Day 1 - Scope Audit and Baseline Setup

### Goal

Turn the Sprint 74 project-plan scope plus the Sprint 70, Sprint 72, and
Sprint 73 handoff into one bounded capability-modernization sprint, with the
strongest live capability ceilings and non-goal fence fixed before deeper
audit begins.

### Actions

1. Re-read the Sprint 74 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Sprint 70 capability
   modernization fence, the Sprint 72 retrospective, and the Sprint 73
   closeout artifact.
2. Reconfirm the preserved Sprint 74 constraints:
   - no one-shot full type-generic conversion
   - no fake complex-readiness story without end-to-end proof
   - no broad capability widening hidden inside local helper work
   - no widened reviewed/platform/install claims
3. Reconfirm the strongest local reviewed baseline shape from:
   - `make -n quality-review-full`
   - `make quality-review-cmake-compile`
4. Capture the live Day 1 hotspot map across the strongest likely Sprint 74
   capability surfaces.
5. Record the intended Sprint 74 workstreams, touch surfaces, and proof-risk
   surfaces before Day 2 validation work begins.

### Findings

#### 1. Sprint 74 now starts from a precise capability queue

Sprint 74 does not need another broad Epic 7 planning reset, and it does not
need another public-surface or product-model cleanup wave.

The strongest next queue is explicitly:

- capability ceiling rerank
- bounded index/scalar architecture design
- first end-to-end index-width modernization seam
- scalar-surface preparation only where later widening truly needs it
- docs, packaging, and proof follow-through only where landed capability work
  moves the maintained contract
- focused overflow, correctness, and compatibility validation

#### 2. The Sprint 70 capability fence remains the right constraint set

The live repo state still supports the same capability fence:

- no one-shot full type-generic conversion
- no fake complex-readiness story without end-to-end proof
- no broad capability widening disguised as local cleanup
- no widened reviewed/platform/install claims detached from shipped evidence

That means Sprint 74 should stay bounded to the first real index-width seam
and only the minimum scalar-surface preparation needed to keep the broader
path coherent.

#### 3. The strongest live capability pressure is concentrated in width,
scalar, and algorithm-breadth seams

The current capability ceiling remains concentrated in:

- public width contract in `include/sparse_types.h`
  - documented migration path still says recompile after changing `idx_t`
  - `typedef int32_t idx_t;`
  - `IDX_MAX` remains `INT32_MAX`
- mutable matrix and product-shell breadth in:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- real-only callback and operator contracts in:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  where callback signatures still carry `const double *` and `double *`
- dense-real algorithm kernels and result ownership in:
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
  - `src/sparse_svd.c`

This is the right Day 1 narrowing: Sprint 74 should start from the index-width
path as the strongest first modernization lane, with scalar-surface
preparation as the strongest second lane and broader algorithm-family
expansion still explicitly later.

#### 4. The strongest likely Sprint 74 touch surfaces are now explicit

Raw Day 1 `wc -l` counts from the live tree:

##### Maintained public and policy surfaces

- `README.md` = `1037`
- `INSTALL.md` = `237`
- `docs/maintainer_guide.md` = `621`
- `include/sparse_types.h` = `233`
- `include/sparse_matrix.h` = `604`
- `include/sparse_analysis.h` = `499`
- `include/sparse_iterative.h` = `765`
- `include/sparse_eigs.h` = `650`

##### Capability-modernization implementation seams

- `src/sparse_types.c` = `50`
- `src/sparse_matrix.c` = `1073`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_eigs.c` = `1534`
- `src/sparse_svd.c` = `1319`

##### Strongest proof and adoption/reporting surfaces

- `tests/test_sparse_matrix.c` = `1054`
- `tests/test_integration.c` = `2448`
- `tests/test_iterative.c` = `2802`
- `tests/test_eigs.c` = `1522`
- `examples/example_analysis.c` = `210`
- `examples/example_basic_solve.c` = `110`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_chol_csc.c` = `407`

#### 5. The strongest reviewed baseline remains intact

The local reviewed baseline remains unchanged:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity was re-materialized live:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps Sprint 74 aligned with the Sprint 70 truthfulness fence before any
index-width or scalar-surface seam lands.

### Validation

This was a docs-only Day 1 baseline/setup pass, so I did not run
`make format`, `make lint`, or `make test`.

I did recheck the reviewed baseline shape and parity anchors with:

- `make -n quality-review-full`
- `make quality-review-cmake-compile`
- `ctest -N --test-dir build/quality-review-cmake`

I also captured the live Day 1 raw `wc -l` hotspot measurements and the
current width/scalar capability map across the strongest likely Sprint 74
surfaces.

### Day 1 Exit State

Sprint 74 Day 1 closes with:

1. one capability-modernization starting queue
2. one preserved Sprint 70 non-goal fence
3. one live reviewed baseline anchor
4. one ranked live capability hotspot map
