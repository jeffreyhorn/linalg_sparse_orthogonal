# Sprint 75 Working Notes

## Day 1 - Scope Audit and Baseline Setup

### Goal

Turn the Sprint 75 project-plan scope plus the Sprint 70, Sprint 64, and
Sprint 74 handoff into one bounded backend/performance sprint, with the
strongest live hotspot families and non-goal fence fixed before deeper audit
begins.

### Actions

1. Re-read the Sprint 75 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Sprint 70 performance target,
   the Sprint 64 backend closeout, and the Sprint 74 capability closeout.
2. Reconfirm the preserved Sprint 75 constraints:
   - no fake external-backend maturity story
   - no shared-library maturity claim
   - no backend widening that weakens the self-contained default build
   - no benchmark timing thresholds masquerading as portable proof
   - no widened reviewed-platform claim detached from maintained evidence
3. Reconfirm the strongest local reviewed baseline shape from:
   - `make -n quality-review-full`
   - `make quality-review-cmake-compile`
   - `ctest -N --test-dir build/quality-review-cmake`
4. Capture the live Day 1 hotspot map across the strongest likely Sprint 75
   backend/performance surfaces.
5. Record the intended Sprint 75 workstreams, touch surfaces, and proof-risk
   surfaces before Day 2 validation work begins.

### Findings

#### 1. Sprint 75 now starts from a precise backend/performance queue

Sprint 75 does not need another broad Epic 7 planning reset, and it does not
need another capability or public-surface cleanup wave.

The strongest next queue is explicitly:

- hotspot re-audit
- backend/policy design
- kernel integration batch
- callback/runtime follow-through
- benchmark proof refresh
- regression/fallback proof
- validation and closeout

#### 2. The Sprint 70 performance and truthfulness fence remains the right constraint set

The live repo state still supports the same backend/performance fence:

- no fake external-backend or shared-library maturity story
- no backend widening that weakens the self-contained default build
- no benchmark timing thresholds masquerading as portable proof
- no widened reviewed-platform claim detached from maintained evidence

That means Sprint 75 should stay bounded to one real second backend-aware
landing plus the minimum callback/runtime and benchmark-proof follow-through
needed to keep that path coherent.

#### 3. The strongest live backend/performance pressure is concentrated in kernel ownership, callback parity, and benchmark proof

The current backend/performance queue remains concentrated in:

- dense-kernel and backend-owned helper seams in:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
- strongest solver-family backend surfaces in:
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_eigs.c`
- runtime and cancellation truth surfaces in:
  - `include/sparse_types.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `docs/maintainer_guide.md`
- canonical backend-proof/reporting surfaces in:
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `benchmarks/bench_svd.c`
  - `README.md`
  - `benchmarks/README.md`

This is the right Day 1 narrowing: Sprint 75 should start from the strongest
current backend-aware hotspot seam and preserve the Sprint 64 truthfulness
contract instead of widening into a general backend framework or a broad
performance-governance rewrite.

#### 4. The strongest likely Sprint 75 touch surfaces are now explicit

Raw Day 1 `wc -l` counts from the live tree:

##### Maintained public and policy surfaces

- `README.md` = `1044`
- `docs/maintainer_guide.md` = `670`
- `benchmarks/README.md` = `370`
- `include/sparse_cholesky.h` = `220`
- `include/sparse_qr.h` = `385`
- `include/sparse_svd.h` = `257`

##### Backend/performance implementation seams

- `src/sparse_dense.c` = `597`
- `src/sparse_qr.c` = `1563`
- `src/sparse_svd.c` = `1319`
- `src/sparse_chol_csc_supernodal.c` = `507`
- `src/sparse_chol_csc.c` = `1564`
- `src/sparse_eigs.c` = `1534`

##### Strongest proof and reporting surfaces

- `tests/test_chol_csc.c` = `4664`
- `tests/test_qr.c` = `3197`
- `tests/test_svd.c` = `2766`
- `tests/test_integration.c` = `2448`
- `tests/test_eigs.c` = `1560`
- `benchmarks/bench_chol_csc.c` = `407`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_eigs_reuse.c` = `278`
- `benchmarks/bench_svd.c` = `180`
- `examples/example_analysis.c` = `210`

#### 5. The strongest reviewed baseline remains intact

The local reviewed baseline remains unchanged:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity was re-materialized live:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps Sprint 75 aligned with the Sprint 70 and Sprint 64 truthfulness
fence before any hotspot rerank or backend-aware implementation work lands.

### Validation

This was a docs-only Day 1 baseline/setup pass, so I did not run
`make format`, `make lint`, or `make test`.

I did recheck the reviewed baseline shape and parity anchors with:

- `make -n quality-review-full`
- `make quality-review-cmake-compile`
- `ctest -N --test-dir build/quality-review-cmake`

I also captured the live Day 1 raw `wc -l` hotspot measurements and the
current backend/performance hotspot map across the strongest likely Sprint 75
surfaces.

### Day 1 Exit State

Sprint 75 Day 1 closes with:

1. one backend/performance starting queue
2. one preserved Sprint 70 / Sprint 64 non-goal fence
3. one live reviewed baseline anchor
4. one ranked live backend/performance hotspot map
