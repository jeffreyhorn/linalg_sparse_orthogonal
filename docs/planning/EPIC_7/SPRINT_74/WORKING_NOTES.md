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

## Day 3 - Capability Ceiling Audit

### Goal

Re-rank the current capability ceilings against the live tree so Sprint 74
starts from one concrete contradiction map rather than one generic
"64-bit + scalar genericity + more algorithms" wishlist.

### Actions

1. Re-read the strongest current capability surfaces directly in:
   - `include/sparse_types.h`
   - `README.md`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - `include/sparse_svd.h`
   - `include/sparse_analysis.h`
2. Re-read the strongest implementation seams that actually carry those
   ceilings:
   - `src/sparse_types.c`
   - `src/sparse_matrix.c`
   - `src/sparse_iterative.c`
   - `src/sparse_eigs.c`
   - `src/sparse_svd.c`
3. Re-read the Epic 7 review baseline so the Day 3 ranking reflects the
   broader state-of-the-art gap, not only the latest Sprint 74 hotspot sizes.
4. Separate the remaining burdens into:
   - public product caveats
   - implementation-local assumptions
   - compatibility/package implications
5. Rank the strongest contradiction centers by:
   - user-value ceiling
   - cross-repo implementation cost
   - proof and migration burden

### Findings

#### 1. The strongest capability ranking remains stable

The live tree still supports the same capability order Sprint 70 identified:

- strongest first target:
  - 32-bit index width
- strongest second target:
  - real-only scalar support
- strongest later target:
  - symmetric-only sparse eigensolver breadth

That ranking is not stale planning inertia. The current public headers,
README caveats, and implementation seams still line up behind that exact
order.

#### 2. The width ceiling is still the strongest first modernization center

The first capability ceiling remains the global `idx_t` model:

- `typedef int32_t idx_t;`
- `IDX_MAX = INT32_MAX`
- README still documents the practical dimension/nnz cap and the "change the
  typedef and recompile" migration story

Why it remains first:

- it is the broadest current product ceiling
- it affects dimensions, nnz, permutations, and many workspace and allocation
  calculations across the whole repo
- it is still easier to isolate into one bounded typedef/overflow/build
  contract than scalar-type generalization

The strongest Day 3 clarification is now explicit:

- the real first modernization center is not "make everything 64-bit now"
- it is "make the 32-bit ceiling non-permanent through one real bounded
  width-modernization seam"

#### 3. The scalar ceiling is still second, but the live pressure center is
clearer now

The second capability ceiling is still the repo-wide real-only `double`
contract.

That pressure shows up directly in the live public and implementation seams:

- iterative callback signatures in `include/sparse_iterative.h` still expose
  `const double *` and `double *`
- eigensolver outputs and kernels in `include/sparse_eigs.h` and
  `src/sparse_eigs.c` remain `double`-typed throughout
- SVD options, results, and dense internal accumulators in
  `include/sparse_svd.h` and `src/sparse_svd.c` remain real-only

The useful narrowing is that the strongest scalar-preparation center is not
the entire library at once. It is the public callback/result and dense-kernel
surfaces where the real-only contract is most explicit and most reused.

Why it remains second instead of first:

- it is broader and more invasive than width modernization
- it touches nearly every public numerical contract simultaneously
- the proof, packaging, and migration burden is therefore still higher

#### 4. The eigensolver-family ceiling is still real, but it remains later

The current public eigensolver story is still explicitly symmetric:

- `sparse_eigs_sym(...)`
- symmetric-only backend documentation
- repeated-run handle semantics tied to symmetric eigensolves

This remains a real state-of-the-art positioning limit, but it still ranks
third because:

- it is narrower than the global width and scalar ceilings
- it affects one major algorithm family instead of the entire product model
- the current symmetric eigensolver lane is already comparatively credible

That makes it a real later Sprint 74 / Epic 7 concern, not the right first
bounded modernization lane.

#### 5. The public-caveat vs implementation-assumption split is now explicit

Public product caveats:

- 32-bit matrix dimensions and nnz
- real-only double-precision numerics
- symmetric-only sparse eigensolver contract

Implementation-local assumptions:

- pervasive `idx_t` use in dimensions, permutations, and workspace sizing
- pervasive `double`-typed callbacks, vectors, result arrays, and dense
  kernels in iterative/eigs/SVD lanes
- eigensolver naming and result contracts specialized to symmetric problems

Compatibility/package implications:

- index-width widening is a public typedef and downstream rebuild event
- scalar-surface widening is a larger API/ABI/product-line event
- eigensolver-family widening expands the public supported capability promise
  without solving the broader width or scalar ceilings

This distinction matters because Sprint 74 should not treat width, scalar
breadth, and algorithm-family expansion as if they were one interchangeable
capability batch.

### Validation

This was a docs-only Day 3 audit pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the audit in direct rereads of the current public capability
surfaces, the strongest implementation seams, and the Epic 7 review baseline.

### Day 3 Exit State

Sprint 74 Day 3 closes with:

1. one ranked live capability contradiction map
2. one narrower first-lane reading of the width-modernization problem
3. one clearer scalar-preparation center for later design work
4. one explicit separation between public caveats, implementation assumptions,
   and compatibility/package events

## Day 4 - First Capability Boundary

### Goal

Refine the Day 3 capability ranking and freeze the first bounded Sprint 74
modernization fence before implementation design begins.

### Actions

1. Re-read the Day 3 capability ranking and the Sprint 70 capability fence.
2. Reconfirm which surfaces actually own the width contract today:
   - public typedef and width caveat surfaces
   - highest-value mutable matrix shell and size-checking seams
   - proof-owner tests most sensitive to width and overflow boundaries
3. Separate the first bounded index-width lane from:
   - later scalar-surface preparation
   - later algorithm-family breadth work
   - docs/package follow-through that should remain support-only
4. Fix the required first landing surfaces, likely support surfaces, and
   explicit deferral set in writing.

### Findings

#### 1. The strongest first Sprint 74 fence is the width contract, not the
broader scalar or algorithm-family ceiling

The Day 4 rerank confirms the best first bounded lane is:

- index-width modernization centered on the public `idx_t` contract and the
  highest-value matrix/product shell size boundary

That lane has the strongest combination of:

- broad user-facing capability payoff
- bounded first-pass implementation scope
- real compatibility-path value
- acceptable first-pass proof and migration risk

The scalar and eigensolver-family ceilings remain real, but they are not the
right first landing because they widen more quickly into public API families,
result structs, callbacks, and broader proof cost.

#### 2. The scalar-preparation seam is support context for later work, not the
first landing

The real-only `double` ceiling remains the strongest second contradiction.

But the rerank shows it should be treated as:

- the strongest second batch
- not the first landing

because:

- the width contract is narrower and easier to make real end-to-end first
- scalar-surface work widens immediately into iterative, eigensolver, and SVD
  public contracts
- the migration and proof burden is much larger than the first width lane

That means the Day 4 fence should keep the scalar-preparation center explicit
but deferred:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`

#### 3. The first-batch landing surfaces are now explicit

Required first landing:

- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Likely support only if the first landing forces it:

- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

Deferred or explicitly later:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- examples and benchmark binaries beyond support-only wording follow-through
- package/install workflow changes beyond truthful width-contract wording

#### 4. The strongest non-goal fence is now explicit

Sprint 74 Day 4 fixes the first-lane non-goals:

- no repo-wide `int64_t` conversion in one batch
- no scalar-type genericity campaign hidden inside width cleanup
- no fake complex-readiness or broader precision-product claims
- no unsymmetric eigensolver expansion as part of the first lane
- no widened packaging/platform/install claims beyond the actual landed width
  seam

### Validation

This was a docs-only Day 4 boundary pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded it in the Day 3 ranked contradiction map, the Sprint 70 capability
fence, and direct rereads of the current width and scalar public contracts.

### Day 4 Exit State

Sprint 74 Day 4 closes with:

1. one explicit first modernization boundary around the width contract
2. one fixed support-only map for proof and maintained-surface follow-through
3. one explicit deferred map for scalar-surface and later algorithm-family
   work
4. one clear starting point for Day 5 implementation design
