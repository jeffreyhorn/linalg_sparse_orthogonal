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

## Day 5 - Index / Scalar Architecture Design

### Goal

Define the bounded implementation contract for Sprint 74's first
capability-modernization batch so code work can make the 32-bit ceiling
non-permanent without widening into a broad type-generic conversion campaign.

### Actions

1. Re-read the Day 4 width-first boundary.
2. Re-read the current public width contract and the matrix-shell seams that
   actually carry the first practical index-width burden:
   - `include/sparse_types.h`
   - `src/sparse_types.c`
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - `src/sparse_alloc_internal.h`
   - `src/sparse_alloc_internal.c`
3. Separate the first width lane into:
   - public contract and migration-path wording
   - internal size/overflow bridge ownership
   - matrix-shell allocation and dimension/bounds seam follow-through
4. Fix the preserved compatibility rules, required touch set, and explicit
   non-touch set before Day 6 implementation begins.

### Findings

#### 1. The first Sprint 74 batch is width-contract-first, not
full-conversion-first

The first bounded Sprint 74 landing should not attempt to ship full
repo-wide `int64_t` mode.

It should instead converge the width lane behind one clearer contract:

- the public width surface should read as one deliberate bounded
  modernization path
- the internal allocation and overflow helpers should be the width bridge
  between `idx_t`-counted public dimensions and `size_t`-based byte math
- the matrix shell should consume that bridge more consistently on the
  highest-value touched seams

That is the right first move because the strongest current pain is:

- the 32-bit ceiling still reading like a static caveat
- plus inconsistent width-bridge ownership across the public shell and its
  highest-value helpers
- not the absence of a broad immediately shippable 64-bit product line

#### 2. The ownership split is now explicit

Public width contract owner in the first batch:

- `include/sparse_types.h`

Internal width-bridge owner in the first batch:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- any existing `idx_t` <-> `size_t` checked helper path already used by the
  matrix shell

Highest-value matrix-shell follow-through in the first batch:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Support-only proof and wording surfaces:

- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

#### 3. The first-batch compatibility rules are fixed

The first batch must preserve:

- current shipped behavior with `idx_t == int32_t`
- current `IDX_MAX`-based width reading for downstream callers
- current allocation and overflow failure behavior on impossible or
  out-of-range counts
- current one-shot and repeated-run user-facing matrix-shell semantics

The implementation goal is therefore:

- make the width contract more explicit and less "edit typedef by hand and
  hope the rest follows"
- tighten the internal checked bridge between public `idx_t` counts and
  `size_t` allocation math
- improve the highest-value matrix-shell seams without promising a full
  alternate-width build matrix yet

#### 4. The first-batch touch and non-touch sets are now explicit

Required first implementation center:

- `include/sparse_types.h`
- `src/sparse_types.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

Support only if the implementation truly forces it:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `tests/test_sparse_matrix.c`
- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`

Explicit non-touch set:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- package/install workflow files beyond truthful width-contract wording
- benchmark-governance or backend/performance files
- broad product-model or configuration follow-through

### Validation

This was a docs-only Day 5 design pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the design in the Day 4 width-first boundary plus direct rereads of
the current width contract and the existing checked allocation/overflow
helpers.

### Day 5 Exit State

Sprint 74 Day 5 closes with:

1. one explicit width-contract-first design for the first capability lane
2. one preserved compatibility checklist
3. one exact first-batch touch set
4. one explicit non-touch set before Day 6 implementation begins

## Day 6 - Index-Width Integration Batch 1

### Goal

Land the first bounded Sprint 74 capability-modernization batch inside the
Day 5 fence so the width contract stops reading like a fixed hand-edited
typedef and the matrix shell consumes the checked width bridge more
consistently.

### Actions

1. Re-read the Day 5 contract and the touched seams:
   - `include/sparse_types.h`
   - `src/sparse_types.c`
   - `src/sparse_alloc_internal.h`
   - `src/sparse_alloc_internal.c`
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - `tests/test_sparse_matrix.c`
2. Replace the fixed-width typedef/readme path with one explicit compile-time
   width contract in `include/sparse_types.h`.
3. Tighten the internal `idx_t` <-> `size_t` bridge and route the highest-value
   matrix-shell allocations and byte-count math through it.
4. Add a focused proof that the public width contract is internally coherent in
   the maintained reviewed build.
5. Run the full required quality gate because `*.c` and `*.h` changed.

### Findings

#### 1. The width contract now reads as one bounded compile-time surface

The first batch did not attempt a repo-wide 64-bit conversion.

It instead landed one explicit compile-time width contract in
`include/sparse_types.h`:

- `SPARSE_IDX_BITS` now selects `32` or `64`
- `idx_t`, `IDX_MAX`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` now come from that
  one contract
- `_Static_assert` now ties the selected macro width back to `sizeof(idx_t)`
- `sparse_idx_bits()` now reports the selected width at runtime

That keeps the shipped default exactly where it was (`32`-bit) while making the
32-bit ceiling read as a deliberate bounded contract rather than an implicit
permanent typedef.

#### 2. The checked width bridge is now reused more consistently

The first batch stayed inside the Day 5 internal-bridge fence:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

The landed follow-through is:

- null-output hardening in the checked conversion helpers
- `IDX_MAX` comparison widened through `uintmax_t` rather than `size_t`
- `sparse_malloc_idx_array(...)` and `sparse_calloc_idx_array(...)` now reuse
  the checked helper path directly

This keeps the existing failure behavior but makes the width bridge more
clearly central instead of partially duplicated.

#### 3. The matrix shell now consumes that bridge on the highest-value touched seam

The matrix-shell follow-through stayed bounded to:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

The useful Day 6 convergence is:

- `sparse_create(...)` now allocates the row/column headers and permutation
  buffers through one checked shell-allocation helper
- `sparse_free(...)` now tears that shell state down through the paired helper
- `sparse_copy(...)` now computes permutation byte counts through checked
  conversions instead of raw casts
- `sparse_memory_usage(...)` now uses checked accumulation and returns
  `SIZE_MAX` on overflow
- `sparse_matmul(...)` now allocates its `idx_t` support buffers through the
  checked width bridge
- Matrix Market and matrix-print formatting/scanning now use
  `SPARSE_PRIDX` / `SPARSE_SCNIDX` instead of hard-coded 32-bit format
  specifiers

This is the right first seam: the matrix shell remains the same public product
surface, but its highest-value width-sensitive allocations and I/O formatting
now read more coherently.

#### 4. The proof stayed narrow and width-contract-local

The only required proof expansion was in `tests/test_sparse_matrix.c`.

The new `test_idx_width_contract(...)` proves:

- `sparse_idx_bits()` matches `SPARSE_IDX_BITS`
- `sizeof(idx_t)` matches that selected width
- the maintained reviewed build still maps the default width contract to the
  expected `idx_t` and `IDX_MAX` values

No wider Sprint 74 support surfaces were forced:

- `tests/test_integration.c` did not need edits
- `README.md` did not need edits
- `docs/maintainer_guide.md` did not need edits
- `INSTALL.md` did not need edits

#### 5. The landing stayed inside the Day 5 fence

Touched implementation and proof surfaces:

- `include/sparse_types.h`
- `src/sparse_types.c`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `tests/test_sparse_matrix.c`

Explicitly not touched:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_svd.h`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_svd.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- package/install workflow files
- broader docs/product/configuration surfaces

### Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 477.93 sec`

Raw touched-surface `wc -l` counts after the landing:

- `include/sparse_types.h` = `278`
- `src/sparse_types.c` = `52`
- `include/sparse_matrix.h` = `610`
- `src/sparse_matrix.c` = `1125`
- `src/sparse_alloc_internal.h` = `63`
- `src/sparse_alloc_internal.c` = `60`
- `tests/test_sparse_matrix.c` = `1071`

### Day 6 Exit State

Sprint 74 Day 6 closes with:

1. one explicit compile-time width contract instead of one fixed hand-edited
   width typedef
2. one clearer checked bridge between public `idx_t` counts and internal byte
   math
3. one bounded matrix-shell follow-through on the highest-value width-sensitive
   seams
4. one focused width-contract proof in the maintained matrix-shell test owner
5. one fully validated first capability landing inside the Sprint 74 fence

## Day 7 - Post-Landing Audit and Rerank

### Goal

Audit the Day 6 width-contract landing against the remaining Sprint 74
capability ceilings so the next implementation lane is chosen from the live
post-landing seam map rather than from the original broader backlog.

### Actions

1. Re-read the Day 5 design contract and the Day 6 landing artifact.
2. Re-audit the post-Day-6 state of:
   - `include/sparse_types.h`
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - `include/sparse_svd.h`
   - `src/sparse_iterative.c`
   - `src/sparse_eigs.c`
   - `src/sparse_svd.c`
3. Re-rank the remaining capability seams by:
   - what contradiction the Day 6 batch actually closed
   - which real-only public contracts still remain densest
   - which later lanes are still valid but not yet the strongest move
4. Fix the exact Day 8 design center and explicit non-centers in writing.

### Findings

#### 1. The Day 6 batch closed the strongest width-first contradiction

The Day 6 landing materially changed the capability queue:

- the width contract no longer reads like a fixed hand-edited typedef
- the checked `idx_t` <-> `size_t` bridge now has a clearer ownership center
- the matrix shell no longer reads like the strongest remaining capability
  contradiction

That means a second same-lane width batch is no longer the highest-value next
move.

#### 2. The strongest remaining seam has shifted to the real-only scalar surface

The strongest remaining capability contradiction is now the public real-only
numerics surface, centered on:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

with implementation support most likely in:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

The Day 7 useful clarification is:

- the next best lane is not "finish width everywhere"
- it is "prepare the strongest real-only callback/result surface so later
  scalar widening has a cleaner bounded contract"

That is where the live public ceiling still reads densest:

- `sparse_precond_fn` still hard-codes `const double *` / `double *`
- `sparse_matvec_fn` still hard-codes `const double *` / `double *`
- iterative one-shot and block solves still expose dense `double` RHS / result
  contracts directly
- `sparse_eigs_t` still exposes `double *eigenvalues` and
  `double *eigenvectors` as the main public result carrier

#### 3. The SVD and algorithm-breadth lanes are still real, but not next

The later capability lanes remain valid:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- later eigensolver-family breadth beyond `sparse_eigs_sym(...)`

But they are now more clearly later work because:

- SVD remains a dense-real result surface, but it is narrower and more
  family-local than the iterative/eigs callback and result contracts
- unsymmetric eigensolver breadth is still a product-expansion question, not
  the strongest current contract contradiction
- the width-first lane already moved enough that reopening it before the
  scalar surface would widen Sprint 74 for less value

#### 4. The Day 8 target set is now explicit

Required next design center:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Likely implementation center if the design proves it:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Likely proof homes:

- `tests/test_iterative.c`
- `tests/test_eigs.c`

Support only if wording truly forces it:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `README.md`
- `docs/maintainer_guide.md`

Explicitly not next:

- another broad width-contract batch in `include/sparse_types.h`
- another broad matrix-shell batch in `include/sparse_matrix.h` /
  `src/sparse_matrix.c`
- unsymmetric eigensolver expansion
- fake complex-readiness or broad scalar-generic implementation claims

### Validation

This was a docs-only Day 7 audit pass, so I did not rerun `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the rerank in direct rereads of the landed width-contract surfaces,
the deferred scalar and eigensolver headers, and the strongest real-only
callback/result seams in the live implementation.

### Day 7 Exit State

Sprint 74 Day 7 closes with:

1. the Day 6 width-first lane explicitly closed as the strongest first
   contradiction
2. one new strongest remaining seam fixed to the real-only scalar surface
3. one exact Day 8 design center around iterative and eigensolver public
   contracts
4. one explicit non-center list keeping later width and algorithm-breadth work
   deferred

## Day 8 - Scalar Surface Preparation Design

### Goal

Turn the Day 7 rerank into one explicit Day 9 implementation fence for the
strongest remaining real-only scalar contract, without widening Sprint 74 into
fake scalar genericity or broader algorithm-surface work.

### Actions

1. Re-read the Day 7 rerank artifact and the Sprint 74 plan around the scalar
   preparation lane.
2. Re-audit the strongest live public scalar-contract surfaces in:
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
3. Re-check the later or support-only scalar surfaces in:
   - `include/sparse_svd.h`
   - `src/sparse_iterative.c`
   - `src/sparse_eigs.c`
   - `src/sparse_svd.c`
4. Classify the strongest remaining scalar contradiction by:
   - public callback shape
   - public dense buffer/result carrier shape
   - current user-facing real-only wording
   - bounded Sprint 74 payoff without false widening claims
5. Fix the exact Day 9 implementation center, likely proof homes, and explicit
   non-goal fence in writing.

### Findings

#### 1. The strongest scalar contradiction is concentrated in iterative and eigs public contracts

The strongest remaining capability seam is still not the whole repo's use of
`double`.

It is the denser public callback and result contracts centered on:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

The strongest live contradiction is that these two headers still carry the most
reused and caller-facing real-only contract points through:

- `sparse_precond_fn`
- `sparse_matvec_fn`
- iterative one-shot and block solve RHS/result signatures
- iterative residual-history and progress fields
- `sparse_eigs_opts_t`
- `sparse_eigs_t`

That makes them the best bounded Sprint 74 scalar-preparation center.

#### 2. The right next move is contract preparation, not broad scalar genericity

The useful Day 8 clarification is:

- the next lane is not "make iterative and eigs type generic now"
- it is "prepare the strongest public real-only callback/result seam so later
  scalar widening has a cleaner bounded ownership center"

That means the next batch should favor:

- clearer real-only contract wording where the current public shape is densest
- separation between today's shipped real-only promise and later widening
  intent
- bounded implementation and proof only where public contract cleanup truly
  forces it

And it should explicitly avoid:

- repo-wide scalar abstraction
- fake complex-readiness language
- broad implementation churn across unrelated solver families

#### 3. SVD remains real but support-only for this batch

`include/sparse_svd.h` and `src/sparse_svd.c` remain real-only surfaces, but
they are not the strongest next center because:

- the SVD surface is narrower and more family-local
- its result carriers matter less to the broad public callback contract than
  iterative and eigs
- touching it now would widen Sprint 74 for less value than the denser
  iterative/eigs public seam

So SVD remains support-only if wording truly forces it, not a required Day 9
center.

#### 4. The Day 9 target set is now explicit

Required Day 9 design and implementation center:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

Likely implementation center if the public-contract cleanup proves it is
needed:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Likely proof homes:

- `tests/test_iterative.c`
- `tests/test_eigs.c`

Support only if wording truly forces it:

- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `README.md`
- `docs/maintainer_guide.md`

Explicitly not next:

- another broad width-contract batch
- repo-wide scalar-generic conversion
- fake complex-readiness or broader precision-product claims
- unsymmetric eigensolver expansion
- reopening broad matrix-shell or configuration lanes

### Validation

This was a docs-only Day 8 design pass, so I did not rerun `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the design in direct rereads of the iterative, eigensolver, and SVD
public headers plus the strongest real-only callback/result signatures in the
live implementation.

### Day 8 Exit State

Sprint 74 Day 8 closes with:

1. one exact scalar-preparation center fixed to iterative and eigensolver
   public contracts
2. one bounded Day 9 implementation lane that stays narrower than full scalar
   genericity
3. one support-only classification for SVD and broader public follow-through
4. one explicit non-goal fence keeping fake capability expansion out of Sprint
   74

## Day 9 - Scalar Surface Preparation Batch

### Goal

Land the first bounded scalar-surface preparation seam for later capability
widening, keeping the work inside the Day 8 fence and out of any fake
scalar-generic or broader algorithm-expansion story.

### Actions

1. Add one explicit public scalar owner in:
   - `include/sparse_types.h`
   - `src/sparse_types.c`
2. Move the strongest iterative/eigs public callback and result contracts onto
   that owner in:
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
3. Keep the landing bounded:
   - no implementation-wide scalar rewrite
   - no SVD widening
   - no unsymmetric eigensolver expansion
4. Add the minimum focused proof in:
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
5. Run the required capability-boundary validation set.

### Findings

#### 1. The landed scalar owner is explicit and still truthful

The batch added one deliberate public scalar owner:

- `sparse_scalar_t`
- `SPARSE_SCALAR_BITS`
- `sparse_scalar_bits()`

in `include/sparse_types.h` / `src/sparse_types.c`.

That changed the public capability reading in one bounded way:

- iterative and eigensolver dense callback/buffer/result contracts now read
  through one named scalar owner
- the shipped contract is still explicitly real-only and `double`-backed
- later widening now has a cleaner public seam than raw repeated `double`
  spelling in every touched callback and result surface

#### 2. The iterative and eigs public seams now use the same scalar owner

The strongest Day 8 target set landed directly:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

The touched iterative public seam now uses `sparse_scalar_t` for:

- progress and result residual fields
- tolerance fields
- residual-history buffers
- preconditioner callback vectors
- matrix-free matvec callback vectors
- one-shot, block, handle, and matrix-free RHS/result vectors

The touched eigs public seam now uses `sparse_scalar_t` for:

- usage examples
- `sigma`
- `tol`
- caller-owned eigenvalue/eigenvector buffers
- reported residual norm
- peak-basis byte interpretation comments

#### 3. The proof stayed narrow and public-contract-local

The focused proof stayed inside the touched scalar seam:

- `tests/test_iterative.c` now proves a matrix-free CG callback and
  caller-owned vectors can use `sparse_scalar_t` directly through the public
  iterative contract
- `tests/test_eigs.c` now proves caller-owned eigensolver result buffers and
  option fields can use `sparse_scalar_t` directly through the public eigs
  contract

That is enough proof for this batch because the landing is contract-owner
preparation, not a numeric-behavior redesign.

#### 4. The preserved fence stayed intact

The batch did not widen into:

- repo-wide scalar-generic conversion
- fake complex-readiness or broader precision-product claims
- `include/sparse_svd.h` / `src/sparse_svd.c`
- unsymmetric eigensolver expansion
- another width-contract or matrix-shell batch

### Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 438.00 sec`

### Day 9 Exit State

Sprint 74 Day 9 closes with:

1. one explicit public scalar owner added without widening the shipped numeric
   claim
2. one bounded iterative/eigs callback and result seam converged onto that
   owner
3. one focused proof pair confirming the touched public scalar contracts
4. one fully validated capability-boundary landing inside the Sprint 74 fence

## Day 10 - Docs / Packaging / Test Alignment Design

### Goal

Decide the smallest maintained-surface follow-through actually required by the
landed Day 6 width-contract batch and Day 9 scalar-surface batch, without
widening Sprint 74 into a generic docs or packaging cleanup pass.

### Actions

1. Re-read the Day 6 and Day 9 artifacts plus the current Sprint 74 plan for
   the follow-through lane.
2. Re-audit the strongest maintained public and policy surfaces against the
   landed capability contract:
   - `README.md`
   - `INSTALL.md`
   - `docs/maintainer_guide.md`
3. Re-check the now-touched public capability headers and support surfaces:
   - `include/sparse_types.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
4. Separate required capability wording follow-through from optional churn
   around install, examples, or support headers.
5. Fix the exact Day 11 touch set and preserved truthfulness checklist in
   writing.

### Findings

#### 1. The strongest required follow-through is public capability wording, not install/package cleanup

The Day 6 and Day 9 landings moved the capability contract in two exact ways:

- width selection is now a named compile-time contract, not an implicit
  hand-edited typedef
- the strongest public real-only scalar seam now routes through
  `sparse_scalar_t` instead of repeated raw `double` spelling

That means the strongest required maintained-surface follow-through is in the
public capability wording, not in the install/package story.

#### 2. `README.md` is the strongest maintained public follow-through surface

`README.md` still carries the strongest caller-facing capability summary in the
Known Limitations section.

The live contradiction is narrow and exact:

- the current 32-bit note still tells callers to manually change the typedef in
  `sparse_types.h`
- the current scalar note still states only the old real-only `double` reading,
  without acknowledging the new public scalar owner

That makes `README.md` a required Day 11 surface.

#### 3. `docs/maintainer_guide.md` is the strongest policy follow-through surface

The maintainer guide is now the right place to state the bounded Sprint 74
capability interpretation directly:

- current reviewed builds still ship the 32-bit `idx_t` lane by default
- the landed width modernization seam is compile-time-selectable and bounded
- the landed scalar modernization seam is public-contract preparation only,
  not broad numeric genericity
- later scalar breadth and later algorithm-family widening remain deferred

That makes `docs/maintainer_guide.md` the second required Day 11 surface.

#### 4. `INSTALL.md` and the touched public headers do not need Day 11 edits

The install/package surface did not materially move:

- no reviewed-platform claim changed
- no install/export contract changed
- no ABI/product claim widened

The touched public headers already read truthfully after the Day 6 and Day 9
landings:

- `include/sparse_types.h` now states the compile-time width contract and the
  bounded scalar owner directly
- `include/sparse_iterative.h` and `include/sparse_eigs.h` already express the
  touched scalar seam through `sparse_scalar_t`

That means these are support-only surfaces for Day 11:

- `INSTALL.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

#### 5. The proof-owner lane is not the Day 11 center

The touched proof owners are already clear from the landed code:

- `tests/test_sparse_matrix.c` owns the width-contract proof
- `tests/test_iterative.c` owns the iterative public scalar seam
- `tests/test_eigs.c` owns the eigs public scalar seam

That proof-owner interpretation may need explicit alignment on Day 12, but it
does not force a Day 11 wording batch by itself.

### Day 11 Touch Set

Required Day 11 surfaces:

- `README.md`
- `docs/maintainer_guide.md`

Support only if wording truly forces it:

- `INSTALL.md`
- `include/sparse_types.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

Explicit non-touch set:

- implementation `src/` files
- proof-owner test files
- benchmark binaries and benchmark docs
- platform/install workflow files
- broader solver-family headers such as `include/sparse_svd.h`

### Preserved Truthfulness Checklist

Day 11 must preserve:

- current shipped default width remains the reviewed 32-bit `idx_t` lane
- the new width seam is compile-time-selectable, not a broad repo-wide 64-bit
  completion claim
- current shipped scalar support remains real-only
- `sparse_scalar_t` is a bounded public preparation seam, not a claim of
  complex support or broad scalar genericity
- no reviewed-platform, install/export, or ABI claim widens
- touched proof stays owned by the focused test surfaces, not by docs/examples

### Day 10 Exit State

Sprint 74 Day 10 closes with:

1. one exact maintained-surface follow-through center fixed to capability
   wording rather than install/package cleanup
2. one bounded Day 11 touch set fixed to `README.md` plus
   `docs/maintainer_guide.md`
3. one explicit support-only map for install, touched headers, examples, and
   proof owners
4. one preserved truthfulness checklist for the post-landing capability story

## Day 11 - Docs / Packaging / Test Alignment Batch

### Goal

Land the smallest maintained-surface follow-through actually required by the
Day 6 width-contract seam and the Day 9 scalar-surface seam, while preserving
the Sprint 70 truthfulness fence and the Day 10 non-touch set.

### Actions

1. Update the exact Day 10 required surfaces:
   - `README.md`
   - `docs/maintainer_guide.md`
2. State the landed Sprint 74 capability split directly:
   - current shipped default width support
   - bounded compile-time width modernization seam
   - current shipped real-only scalar support
   - bounded public scalar preparation seam
3. Recheck whether support-only surfaces truly needed edits:
   - `INSTALL.md`
   - `include/sparse_types.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
4. Re-run the docs-only sanity set:
   - diff review
   - terminology/alignment checks
   - touched-surface `wc -l`
   - branch-state verification

### Findings

#### 1. `README.md` now states the landed width/scalar split directly

The Known Limitations surface now says plainly that:

- the reviewed build still ships the 32-bit `idx_t` lane by default
- wider indices now route through the bounded compile-time
  `SPARSE_IDX_BITS=64` seam
- downstream callers must rebuild against that same width contract
- scalar support is still real-only
- `sparse_scalar_t` is bounded public preparation, not a broader generic or
  complex-support claim

That is the exact caller-facing follow-through the Day 10 design required.

#### 2. `docs/maintainer_guide.md` now owns the narrower Sprint 74 capability interpretation

The maintainer guide now states directly that:

- Sprint 74 moved bounded width and scalar seams only
- reviewed builds still default to the 32-bit `idx_t` lane
- `SPARSE_IDX_BITS` is the compile-time width contract
- `sparse_scalar_t` is the touched dense-scalar owner on the iterative/eigs
  seam while shipped scalar support remains real-only
- broader scalar breadth and later algorithm-family widening remain deferred

It also names the focused proof owners directly:

- `tests/test_sparse_matrix.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

#### 3. No support-only surface actually needed follow-through

The Day 10 support-only map held:

- `INSTALL.md` did not need edits because the install/export and
  reviewed-platform contract did not move
- the touched headers already remained truthful after the landed code
- example surfaces did not need wording changes

That means the Day 11 batch stayed bounded to the exact planned touch set.

### Sanity Checks

This was a docs-only batch, so I did not run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I used the targeted docs-only sanity set instead:

- diff review
- terminology/alignment checks
- touched-surface `wc -l`
- branch-state verification

Touched-surface raw `wc -l` counts:

- `README.md` = `1044`
- `docs/maintainer_guide.md` = `670`

### Day 11 Exit State

Sprint 74 Day 11 closes with:

1. one bounded caller-facing capability follow-through batch in `README.md`
2. one bounded policy/proof-owner follow-through batch in
   `docs/maintainer_guide.md`
3. one confirmed support-only map for install, touched headers, and examples
4. one preserved truthfulness fence for the landed Sprint 74 capability story

## Day 12 - Regression Coverage & Safety Alignment

### Goal

Confirm that the landed Sprint 74 width and scalar seams already have the
right focused proof owners, add only the minimum regression follow-through if a
real gap remains, and fix the exact Day 13 validation queue from the
post-Day-11 state.

### Actions

1. Re-read the touched proof owners and their sustained contract points:
   - `tests/test_sparse_matrix.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `include/sparse_types.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
2. Re-read the maintained public/policy follow-through surfaces from Day 11:
   - `README.md`
   - `docs/maintainer_guide.md`
3. Decide whether any regression gap still remains for:
   - width-contract safety
   - touched scalar compatibility boundaries
   - narrowed capability-claim truthfulness
4. Fix the exact Day 13 validation queue in writing around the touched proof
   owners, representative examples, maintained capability benchmarks, and
   install/package scripts.

### Findings

#### 1. No new regression code is actually needed

The touched capability seams already sit in the right focused proof owners:

- `tests/test_sparse_matrix.c` owns the width-contract lane through:
  - `SPARSE_IDX_BITS`
  - `IDX_MAX`
  - `sparse_idx_bits()`
- `tests/test_iterative.c` owns the iterative public scalar seam through:
  - `sparse_scalar_t` matrix-free callback vectors
  - `sparse_scalar_bits()` on the public iterative contract
- `tests/test_eigs.c` owns the eigensolver public scalar seam through:
  - `sparse_scalar_t` caller-owned result buffers and option fields
  - `sparse_scalar_bits()` on the public eigs contract

Those are exactly the narrow proof owners Sprint 74 needed. Adding broader or
duplicated regression would weaken ownership clarity rather than improve it.

#### 2. The maintained proof-owner wording is already aligned after Day 11

The Day 11 maintainer-guide batch already names the touched Sprint 74 proof
owners directly and keeps the narrower capability interpretation explicit.

That means no extra policy or header wording is required on Day 12:

- `README.md` remains the caller-facing capability summary
- `docs/maintainer_guide.md` remains the policy and proof-owner authority
- touched public headers already express the width/scalar seams truthfully

#### 3. The real Day 12 output is the explicit Day 13 validation queue

The exact Day 13 validation queue is now fixed around the touched Sprint 74
surfaces:

- standard code-day gate:
  - `make format`
  - `make lint`
  - `make test`
- strongest reviewed baseline:
  - `make quality-review-full`
- reviewed proof-owner follow-ons:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_integration`
- representative examples:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained capability benchmark/reporting surfaces:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_iterative_reuse`
  - `./build/quality-review-cmake/bench_eigs_reuse`
- maintained install/package proof:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Build / Reference Alignment

The sustained Sprint 74 build/reference ownership is now explicit:

- `tests/test_sparse_matrix.c` is the maintained width-contract proof owner
- `tests/test_iterative.c` is the maintained iterative public scalar-seam owner
- `tests/test_eigs.c` is the maintained eigensolver public scalar-seam owner
- examples remain adoption/context surfaces, not proof owners
- benchmark binaries remain capability/reporting context, not truth owners for
  the width/scalar contract

### Day 12 Exit State

Sprint 74 Day 12 closes with:

1. one confirmed focused proof-owner map for the landed width and scalar seams
2. no redundant regression expansion added where no real gap remained
3. one explicit Day 13 validation queue for the touched capability package
