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

## Day 2 - Validation Baseline and Truth-Surface Recheck

### Goal

Reconfirm the Sprint 74 implementation-day validation contract and fix the
highest-signal rerun set before any capability-modernization batch lands.

### Actions

1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 74 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or capability-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design/review days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 74 is most likely to stress:
   - matrix and integration proof owners
   - iterative and eigensolver proof owners
   - representative examples
   - maintained capability benchmark/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 74.
5. Record the authoritative validation split in the working notes.

### Findings

#### 1. The strongest reviewed baseline remains unchanged

Sprint 74 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps the sprint aligned with the Sprint 70 truthfulness fence before any
index-width or scalar-surface boundary work lands.

#### 2. The Sprint 74 authority split is now explicit before code work

The Day 2 recheck fixes the same three-part validation split Sprint 72 and
Sprint 73 used:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial architecture or capability-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

That is the right split for Sprint 74 because the likely work crosses public
typedef, overflow, ownership, callback-signature, and compatibility
boundaries rather than one tiny helper seam.

#### 3. The live proof-surface split is now fixed for Sprint 74

The Day 2 recheck shows this live local split:

- the reviewed CMake tree currently owns the key proof-owner tests,
  representative examples, and maintained capability benchmark binaries most
  relevant to Sprint 74:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_iterative_reuse`
  - `./build/quality-review-cmake/bench_eigs_reuse`
- maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- the root `build/` tree is not currently carrying the usual maintained
  capability benchmark binaries:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

That truth matters: Sprint 74 should anchor its rerun set to the live reviewed
CMake tree plus the maintained proof scripts, rather than assuming the root
benchmark binaries are materialized right now.

#### 4. The high-signal Sprint 74 rerun set is now explicit

The strongest likely rerun set for Sprint 74 is:

- matrix and direct-workflow proof owners:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
- scalar/callback and algorithm-breadth proof owners:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained capability benchmark/reporting surfaces currently materialized in
  the reviewed tree:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_iterative_reuse`
  - `./build/quality-review-cmake/bench_eigs_reuse`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

This is the right Day 2 fix: the rerun contract is now tied to the live
capability-risk surface and the current local binary split, not to a stale
assumption about which maintained benchmark binaries happen to exist in the
root build tree.

### Validation

This was a docs-only Day 2 pass, so I did not run `make format`, `make lint`,
or `make test`.

I did recheck the reviewed baseline and proof-surface split with:

- `ctest -N --test-dir build/quality-review-cmake`
- direct existence checks on the reviewed CMake proof/test/example/benchmark
  binaries
- direct existence checks on the root `build/` benchmark binaries
- direct existence checks on the install/package regression scripts

### Day 2 Exit State

Sprint 74 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one truthful live proof-surface map
4. one exact rerun set for the strongest likely Sprint 74 capability lanes
