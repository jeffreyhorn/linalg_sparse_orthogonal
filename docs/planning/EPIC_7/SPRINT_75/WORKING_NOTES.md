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

## Day 2 - Validation Baseline and Truth-Surface Recheck

### Goal

Reconfirm the Sprint 75 implementation-day validation contract and fix the
highest-signal rerun set before any backend-aware batch lands.

### Actions

1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 75 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial backend or architecture batches default to
     `make quality-review-full`
   - docs-only audit/design/review days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 75 is most likely to stress:
   - backend-aware solver proof owners
   - callback and cancellation parity tests
   - representative examples
   - maintained benchmark/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 75.
5. Record the authoritative validation split in the working notes.

### Findings

#### 1. The strongest reviewed baseline remains unchanged

Sprint 75 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps the sprint aligned with the Sprint 70 and Sprint 64 truthfulness
fence before any backend-aware hotspot work lands.

#### 2. The Sprint 75 authority split is now explicit before code work

The Day 2 recheck fixes the same three-part validation split Sprint 72-74
used:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial backend or architecture batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

That is the right split for Sprint 75 because the likely work crosses backend
selection, dense-kernel behavior, runtime/callback parity, and canonical
benchmark proof boundaries rather than one tiny helper seam.

#### 3. The live proof-surface split is now fixed for Sprint 75

The Day 2 recheck shows this live local split:

- the reviewed CMake tree currently owns the key backend-aware solver tests,
  representative examples, and maintained benchmark binaries most relevant to
  Sprint 75:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_eigs_reuse`
  - `./build/quality-review-cmake/bench_svd`
- maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

That truth matters: Sprint 75 should anchor its rerun set to the live
reviewed CMake tree plus the maintained proof scripts, rather than assuming a
different benchmark split than the current local build actually carries.

#### 4. The high-signal Sprint 75 rerun set is now explicit

The strongest likely rerun set for Sprint 75 is:

- backend-aware solver proof owners:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_eigs`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_eigs_reuse`
  - `./build/quality-review-cmake/bench_svd`
- maintained install/package proof:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

#### 5. The preserved truth-surface reading is now fixed before backend work

Sprint 75 Day 2 also fixes the maintained truth-surface interpretation before
implementation begins:

- `make quality-review-full` remains the strongest local reviewed baseline
- touched benchmark binaries remain proof/reporting context, not portable
  pass/fail timing gates
- install/package regressions remain real maintained proof surfaces, but they
  do not by themselves widen the reviewed platform contract
- callback/runtime follow-through must preserve the existing family-local
  cancellation truth rather than collapsing it into a generic backend claim

### Validation

This was a docs-only Day 2 pass, so I did not run `make format`, `make lint`,
or `make test`.

I did recheck the reviewed baseline wording, rerun the reviewed CMake parity
anchor, confirm the live Sprint 75 proof/test/example/benchmark binaries in
`build/quality-review-cmake`, confirm the maintained install/package proof
scripts, and re-read the current `make -n quality-review-full` wrapper shape.

### Day 2 Exit State

Sprint 75 Day 2 closes with:

1. one explicit implementation-day validation contract
2. one fixed live proof-surface split across reviewed tests, examples,
   benchmarks, and install scripts
3. one high-signal Sprint 75 rerun set for later backend landings
