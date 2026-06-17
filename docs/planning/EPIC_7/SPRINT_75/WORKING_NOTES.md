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

## Day 3 - Backend Hotspot Re-audit

### Goal

Reduce Sprint 75's broad backend/performance question to one ranked live seam
map, so the sprint starts from the strongest bounded second landing rather
than from a generic "more backend architecture" pressure.

### Actions

1. Re-read the strongest current backend/performance authority surfaces:
   - `README.md`
   - `docs/maintainer_guide.md`
   - `benchmarks/README.md`
2. Re-read the strongest likely backend and runtime seams:
   - `src/sparse_dense.c`
   - `src/sparse_chol_csc_supernodal.c`
   - `src/sparse_chol_csc.c`
   - `include/sparse_eigs.h`
   - `src/sparse_eigs.c`
   - `include/sparse_qr.h`
   - `src/sparse_qr.c`
   - `include/sparse_svd.h`
   - `src/sparse_svd.c`
3. Recheck the maintained benchmark and proof surfaces most relevant to the
   live backend story:
   - `benchmarks/bench_chol_csc.c`
   - `benchmarks/bench_eigs_reuse.c`
   - `benchmarks/bench_svd.c`
   - `tests/test_chol_csc.c`
   - `tests/test_eigs.c`
   - `tests/test_qr.c`
   - `tests/test_svd.c`
4. Re-rank the strongest current contradiction centers by actual user value,
   proof cost, and implementation leverage.
5. Record the fixed Day 3 seam map for the Day 4 landing-boundary pass.

### Findings

#### 1. The strongest current backend seam is still the CSC supernodal dense-kernel lane

The densest real backend-aware implementation seam is still concentrated in:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `tests/test_chol_csc.c`

This remains the strongest first center because it already has all of the
pieces a bounded backend-aware architecture lane needs:

- one concrete dense-kernel owner
- one shipped runtime descriptor (`builtin`) rather than a hypothetical plugin
  layer
- one maintained benchmark-side proof surface
- one large family-local regression owner
- one explicit maintained truthfulness contract in the docs

That makes the Cholesky CSC supernodal path the strongest current
implementation-leverage center, not merely the largest file cluster.

#### 2. The strongest second seam is eigensolver backend/runtime parity

The second-ranked backend/performance seam is now the symmetric eigensolver
lane across:

- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `tests/test_eigs.c`
- `benchmarks/bench_eigs_reuse.c`

The key Day 3 clarification is that this lane already owns a real backend
selector and shared orchestration surface, not just future-looking comments.
That makes it the strongest second backend lane.

What keeps it second rather than first is that the bounded dense-kernel story
is still stronger and more coherent in CSC Cholesky:

- the Cholesky lane already exposes dense-kernel descriptor identity and
  benchmark proof directly
- the eigs lane is more of a runtime/callback parity seam than a dense-kernel
  or backend-descriptor seam
- `include/sparse_eigs.h` still carries explicit backend-family asymmetry in
  progress and cancellation behavior, especially around the grow-m path versus
  the other eigensolver families

So Sprint 75 should treat eigs as the strongest second landing, not as the
first one.

#### 3. QR and SVD remain real hotspots, but they are later lanes rather than the first landing

The Day 3 reread of:

- `include/sparse_qr.h`
- `src/sparse_qr.c`
- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `benchmarks/bench_svd.c`
- `tests/test_qr.c`
- `tests/test_svd.c`

shows that QR and SVD still matter, but not as the first backend-architecture
batch center.

The strongest reasons are:

- neither public header currently exposes a backend selector or backend-aware
  product contract comparable to the eigs lane
- `bench_svd.c` still reads as exploratory/profiling support rather than the
  most canonical maintained benchmark-proof owner
- the QR and SVD implementation seams are still more algorithmic and
  workspace-oriented than backend-descriptor-oriented

That means QR and SVD should remain later Sprint 75 targets unless the Day 4
boundary pass shows that the strongest bounded second landing is actually
callback/runtime parity rather than another kernel-owned lane.

#### 4. The strongest cross-cutting secondary seam is callback/runtime truth, not benchmark governance

The Day 3 reread also fixes one useful separation:

- the strongest cross-cutting secondary seam is callback/runtime parity
- it is not a broad benchmark-governance rewrite

This is clearest in:

- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `include/sparse_qr.h`
- `README.md`
- `docs/maintainer_guide.md`

The benchmark/report surfaces are already reasonably well bounded:

- `bench_chol_csc` is the maintained backend-side proof surface for the first
  dense-kernel-aware Cholesky lane
- `bench_eigs_reuse` is already fairly structured and retained
- `bench_svd` is weaker, but that is more a later proof-surface issue than the
  main architecture center

By contrast, callback and runtime truth still have enough family-local
asymmetry that any backend-aware Sprint 75 landing must preserve them
carefully rather than smoothing them into one generic story.

#### 5. The Sprint 75 contradiction map is now ranked explicitly

The Day 3 backend/performance rerank is now:

- strongest first target:
  - CSC supernodal Cholesky dense-kernel/runtime ownership
- strongest second target:
  - eigensolver backend/runtime parity
- strongest later target:
  - QR and SVD backend-aware follow-through
- strongest cross-cutting support seam:
  - callback/runtime truth across maintained families
- support-only, not first-batch centers:
  - broad benchmark governance rewrite
  - broad public-surface rewrite
  - fake pluggable-backend or shared-library maturity work

This is the right narrowing for Day 3: Sprint 75 should begin from the most
coherent shipped backend-aware lane and only widen where the benchmark proof,
callback truth, and family-local regression surfaces already justify it.

### Validation

This was a docs-only Day 3 audit pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the rerank in direct rereads of the current backend/performance
authority surfaces, the dense-kernel owner, the CSC supernodal Cholesky lane,
the eigs public/runtime seam, the QR and SVD public/implementation seams, and
the maintained benchmark and regression owners that currently define the live
backend truth surface.

### Day 3 Exit State

Sprint 75 Day 3 closes with:

1. one explicit ranked backend/performance contradiction map
2. one fixed strongest first lane in CSC supernodal Cholesky
3. one fixed strongest second lane in eigensolver backend/runtime parity
4. one explicit separation between real architecture pressure and later
   benchmark or public-surface follow-through

## Day 4 - First Backend Boundary

### Goal

Freeze one exact first Sprint 75 backend/policy fence so the next design pass
starts from a bounded implementation lane rather than a generic performance
architecture backlog.

### Actions

1. Re-rank the Day 3 hotspots against:
   - runtime leverage
   - proof cost
   - compatibility risk
   - bounded Sprint 75 payoff
2. Separate the likely Sprint 75 surfaces into:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces them
   - later or explicitly deferred backend work
3. Recheck the strongest live first-lane proof and observability surfaces:
   - `include/sparse_cholesky.h`
   - `src/sparse_dense.c`
   - `src/sparse_chol_csc_supernodal.c`
   - `src/sparse_chol_csc.c`
   - `benchmarks/bench_chol_csc.c`
   - `tests/test_chol_csc.c`
4. Reconfirm the strongest deferred second lane:
   - `include/sparse_eigs.h`
   - `src/sparse_eigs.c`
   - `benchmarks/bench_eigs_reuse.c`
   - `tests/test_eigs.c`
5. Fix the Day 4 boundary and non-goal fence in writing.

### Findings

#### 1. The first Sprint 75 landing should stay on the shipped CSC supernodal Cholesky lane

The strongest first landing remains the bounded CSC supernodal Cholesky
dense-kernel/runtime lane.

Required first-batch implementation center:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`

This is still the right first boundary because it combines:

- the clearest current backend-owner seam
- the strongest runtime leverage
- the most compact existing benchmark-side proof surface
- one already-maintained family-local regression owner
- the lowest risk of widening the product claim beyond the shipped
  self-contained default build

#### 2. The strongest support surfaces are bounded, not assumed

The first batch does not need to assume a broad support-wave.

Support only if the first batch truly forces it:

- `include/sparse_cholesky.h`
- `benchmarks/bench_chol_csc.c`
- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

This is the useful Day 4 narrowing:

- header wording should move only if the first batch changes local callback,
  dense-kernel-descriptor, or publish-back truth
- benchmark proof should move only if the first batch changes what must be
  made measurable
- regression owners should move only if the first batch changes correctness or
  fallback behavior enough to require new proof
- maintainer-policy wording should move only if the batch genuinely changes
  the bounded backend contract

#### 3. Eigs is fixed as the strongest second lane, not the first batch center

The eigensolver backend/runtime seam is now explicitly deferred behind the
first batch:

- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `benchmarks/bench_eigs_reuse.c`
- `tests/test_eigs.c`

That does not make eigs unimportant. It fixes the execution order:

- CSC supernodal Cholesky remains the best first landing
- eigs remains the best second landing
- callback/runtime parity should first be tightened where the CSC lane needs
  it, not reopened repo-wide through the eigs lane before the first batch is
  shipped

#### 4. QR, SVD, and broad benchmark refresh are now explicit later work

The Day 4 deferred set is now explicit:

- `include/sparse_qr.h`
- `src/sparse_qr.c`
- `tests/test_qr.c`
- `include/sparse_svd.h`
- `src/sparse_svd.c`
- `tests/test_svd.c`
- `benchmarks/bench_svd.c`
- broad `README.md` and `benchmarks/README.md` cleanup

The useful separation is:

- QR and SVD are later backend/performance lanes
- broad benchmark-proof refresh is later follow-through
- broad public-surface cleanup is later follow-through

None of those should widen the first Sprint 75 implementation fence.

#### 5. The non-goal fence is now fixed explicitly

The Day 4 non-goal fence is:

- no broad backend abstraction-layer rewrite
- no fake optional-backend or shared-library maturity claim
- no benchmark-threshold portability story
- no repo-wide callback/cancellation uniformity claim
- no broad governance or docs-cleanup spill before the first kernel lane lands

This keeps Sprint 75 aligned with the Sprint 70 truthfulness contract and the
Sprint 64 self-contained default-build reading.

### Validation

This was a docs-only Day 4 boundary pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the boundary in the Day 3 rerank plus direct rereads of the live
CSC supernodal Cholesky runtime, proof, benchmark, and header surfaces, and
the strongest deferred eigs lane that now sits behind the first batch.

### Day 4 Exit State

Sprint 75 Day 4 closes with:

1. one explicit first Sprint 75 backend landing fence
2. one bounded support-only map for proof, benchmark, header, and policy
   surfaces
3. one fixed deferred second lane in eigensolver backend/runtime parity
4. one explicit non-goal fence against broad backend or benchmark-governance
   widening

## Day 5 - Backend / Policy Design

### Goal

Define the bounded implementation contract for the first Sprint 75
backend-aware landing before code edits begin, so the batch lands as backend
ownership clarity rather than as generic refactoring.

### Actions

1. Re-read the Sprint 70 performance target and non-goal fence against the
   exact Day 4 first-batch surfaces.
2. Re-read the local Cholesky backend surfaces that define the first batch:
   - `include/sparse_cholesky.h`
   - `src/sparse_dense.c`
   - `src/sparse_chol_csc_supernodal.c`
   - `src/sparse_chol_csc.c`
   - `benchmarks/bench_chol_csc.c`
3. Decide the ownership split for:
   - touched solver/kernel owner
   - runtime/backend observability owner
   - fallback and self-contained default-build owner
4. Fix the guarantees the first batch must preserve.
5. Record the exact first-batch non-touch set and support-only map.

### Findings

#### 1. The first Sprint 75 batch should be dense-kernel-owner-first

The strongest Day 5 clarification is that the first batch should be
dense-kernel-owner-first, not benchmark-first and not callback-first.

Required implementation center:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`

The design reading is:

- `src/sparse_dense.c` owns the concrete dense-kernel descriptor and the
  self-contained default implementation
- `src/sparse_chol_csc_supernodal.c` owns supernodal batch-time consumption of
  that descriptor and the local backend-contract failure boundary
- `src/sparse_chol_csc.c` owns CSC-lane orchestration, dispatch into the
  supernodal path, and compatibility-shell publication back to the caller

That split is narrower and cleaner than treating all three files as one
undifferentiated hotspot cluster.

#### 2. Runtime and observability ownership should stay local to the Cholesky lane

The touched runtime/backend observability owner for the first batch is still
the Cholesky-local surface, not a repo-wide callback policy pass.

Runtime/backend observability owner in the first batch:

- `src/sparse_chol_csc.c`

Support only if wording truly moves:

- `include/sparse_cholesky.h`
- `benchmarks/bench_chol_csc.c`

The useful Day 5 conclusion is:

- `used_csc_path` publication remains a Cholesky-local runtime truth surface
- dense-kernel descriptor identity remains a Cholesky-local benchmark truth
  surface
- any callback or cancellation follow-through should stay local to what the
  first kernel batch actually changes, rather than reopening the broader eigs
  or QR callback story early

#### 3. Fallback and default-build ownership should stay in the dense-kernel seam itself

The fallback and self-contained default-build owner for the first batch should
remain the dense-kernel seam itself, not a wider packaging or plugin surface.

Fallback/default-build owner in the first batch:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`

This preserves the current truthful reading:

- the default shipped dense-kernel descriptor remains `builtin`
- test overrides remain proof-only and local
- `SPARSE_ERR_BACKEND_CONTRACT` remains a narrow public error for unresolved
  required internal dense-kernel callbacks or descriptors
- the batch must not turn that local seam into a fake optional-backend
  maturity story

#### 4. The first-batch guarantees are now fixed explicitly

The first Sprint 75 batch must preserve:

- the self-contained default build remains the main product path
- the default shipped dense-kernel descriptor remains explicit and measurable
- linked-list, CSC scalar, and CSC supernodal truth stay like-for-like on the
  maintained benchmark surface
- touched runtime/backend observability becomes clearer, not broader
- benchmark surfaces remain reporting/proof surfaces, not timing gates
- the one-shot Cholesky compatibility shell and publish-back story remain
  truthful and bounded

This is the right compatibility checklist because it keeps Sprint 75 inside
the Sprint 64 and Sprint 70 truthfulness fence while still allowing a real
backend-aware second landing.

#### 5. The exact non-touch set is now fixed

The Day 5 non-touch set is:

- eigensolver backend/runtime lane:
  - `include/sparse_eigs.h`
  - `src/sparse_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `tests/test_eigs.c`
- QR lane:
  - `include/sparse_qr.h`
  - `src/sparse_qr.c`
  - `tests/test_qr.c`
- SVD lane:
  - `include/sparse_svd.h`
  - `src/sparse_svd.c`
  - `tests/test_svd.c`
  - `benchmarks/bench_svd.c`
- broad docs/governance/platform spill:
  - `README.md`
  - `benchmarks/README.md`
  - `INSTALL.md`
  - packaging or reviewed-platform workflow files
- capability-surface reopening:
  - width/scalar modernization surfaces from Sprint 74

Support-only if the batch truly forces them:

- `include/sparse_cholesky.h`
- `benchmarks/bench_chol_csc.c`
- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

### Validation

This was a docs-only Day 5 design pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the design in the Day 4 boundary plus direct rereads of the local
Cholesky backend header, dense-kernel owner, supernodal batch owner,
publish-back/runtime owner, and maintained benchmark-side proof surface.

### Day 5 Exit State

Sprint 75 Day 5 closes with:

1. one explicit first-batch ownership split across dense-kernel, supernodal,
   and CSC-lane owners
2. one fixed fallback/default-build and observability reading
3. one preserved compatibility checklist for the first backend-aware landing
4. one exact non-touch set before code work begins

## Day 6 - Design Freeze & Proof Map

### Goal

Finalize the exact code/proof ownership split for the first Sprint 75 landing
before backend edits begin, so Day 7 can land without reopening design scope.

### Actions

1. Re-read the Day 5 design against the strongest live proof-owner tests and
   benchmark/runtime surfaces.
2. Map the first backend batch to:
   - implementation owners
   - callback/runtime follow-through owners
   - benchmark proof owners
   - regression/fallback proof owners
3. Confirm which headers and docs stay support-only for the first landing.
4. Fix the exact Day 7 implementation fence and the Day 8 post-landing audit
   criteria.
5. Record the finalized ownership and proof map.

### Findings

#### 1. The Day 7 implementation fence is now fixed explicitly

The exact Day 7 implementation center is:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc.c`

This remains the right code batch because those three files align directly with
the Day 5 ownership split:

- dense-kernel descriptor and shipped default implementation
- supernodal batch-time consumption and narrow backend-contract failure
  boundary
- CSC-lane orchestration and compatibility-shell publication

That means the Day 7 batch should not need to widen into a second solver
family or into broad benchmark/doc churn just to make its contract real.

#### 2. The first regression/fallback proof owners are now fixed

The strongest first regression/fallback proof owner is:

- `tests/test_chol_csc.c`

Likely support proof only if the Day 7 batch truly changes a caller-facing
public-path guarantee:

- `tests/test_integration.c`

The useful Day 6 clarification is:

- family-local backend/fallback correctness belongs in `tests/test_chol_csc.c`
- public-path lifecycle or publish-back contract proof belongs in
  `tests/test_integration.c` only if the Day 7 landing really changes that
  outer guarantee
- the Day 7 batch should prefer extending the family-local owner first, not
  scattering proof across multiple test binaries without a real contract need

#### 3. The benchmark proof owner is fixed and bounded

The benchmark proof owner for the first landing is:

- `benchmarks/bench_chol_csc.c`

That ownership is narrow:

- it owns benchmark-visible path identity
- it owns benchmark-visible dense-kernel descriptor identity
- it owns like-for-like timing proof across linked-list, CSC scalar, and CSC
  supernodal paths

It does not own:

- broad regression or oracle guarantees
- platform/performance threshold policy
- a generalized backend-governance rewrite

So if Day 7 changes what must be measurable, the first benchmark follow-through
belongs here and nowhere broader.

#### 4. Callback/runtime and policy follow-through stay support-only unless forced

The support-only follow-through list for the first landing is now fixed:

- `include/sparse_cholesky.h`
- `docs/maintainer_guide.md`

The runtime/observability interpretation is:

- `include/sparse_cholesky.h` moves only if the Day 7 batch changes local
  public truth around dense-kernel descriptors, `used_csc_path`, or
  callback/runtime interpretation
- `docs/maintainer_guide.md` moves only if the bounded backend contract itself
  becomes clearer in a way the policy surface should capture

That keeps the docs and header lane support-only instead of turning them into
default participants in the Day 7 implementation batch.

#### 5. The Day 8 audit criteria are now explicit

After the Day 7 landing, the Day 8 audit should rerank the remaining Sprint 75
queue against these exact questions:

- did the Day 7 batch close the strongest dense-kernel/backend-owner seam
- is the strongest remaining contradiction now:
  - callback/runtime parity
  - benchmark proof refresh
  - residual support-surface drift
- did any support-only surface actually need to move
- does the eigs lane remain the strongest second landing, or did the Day 7
  batch change that ordering

That gives the post-landing audit a concrete success rubric instead of a vague
“what feels left” pass.

### Validation

This was a docs-only Day 6 proof-map pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the proof map in the Day 5 design plus direct rereads of the local
Cholesky header/runtime surfaces, the maintained benchmark-side proof surface,
the strongest family-local regression owner, the likely public-path support
owner, and the maintainer-policy truth surface.

### Day 6 Exit State

Sprint 75 Day 6 closes with:

1. one exact Day 7 implementation fence
2. one fixed benchmark and regression/fallback proof-owner map
3. one support-only follow-through list for header and policy surfaces
4. one explicit Day 8 post-landing audit rubric

## Day 7 - Kernel Integration Batch 2

### Goal

Land the first real backend-aware kernel follow-through inside the bounded CSC
supernodal Cholesky lane without widening into eigs, QR, SVD, or broad
docs/benchmark churn.

### Actions

1. Re-read the dense-kernel descriptor, the supernodal panel-elimination path,
   and the CSC orchestration seam against the Day 6 proof map.
2. Replace the repeated single-RHS supernodal panel solve loop with one dense
   batched panel-solve callback in the internal kernel descriptor.
3. Extend family-local proof in `tests/test_chol_csc.c` around:
   - the builtin dense-kernel descriptor contract
   - direct batched panel-solve correctness
   - missing-callback backend-contract failure
4. Run the full required validation set for a substantial backend code batch.
5. Record the landed kernel-owner seam and the untouched support surfaces.

### Findings

#### 1. The dense-kernel descriptor now owns one explicit batched panel-solve seam

The landed Day 7 kernel extension is:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`

The dense-kernel descriptor now owns a real multi-RHS panel-solve callback:

- `chol_dense_solve_panel(...)`
- `chol_dense_kernels_t.solve_panel`

That is the right owner split because it keeps the shipped builtin dense
implementation in the dense-kernel seam itself instead of leaving supernodal
code to synthesize batching from repeated lower-solve calls.

#### 2. The supernodal CSC lane now consumes the batched panel-solve seam directly

The landed supernodal consumer change is:

- `src/sparse_chol_csc_supernodal.c`

`chol_csc_supernode_eliminate_panel(...)` now reads as one bounded publish-free
backend step:

- require `kernels->solve_panel`
- reject a missing callback with `SPARSE_ERR_BACKEND_CONTRACT`
- dispatch one dense batched panel solve over the whole below-diagonal panel

This closed the strongest local backend-owner contradiction from the Day 3-6
audit:

- dense batching no longer hides in a repeated single-RHS loop in the
  supernodal consumer
- the narrow backend-contract failure now points at the actual missing required
  dense callback

#### 3. Family-local proof now covers the new panel-solve and fallback contract

The first proof owner stayed exactly where the Day 6 map said it should:

- `tests/test_chol_csc.c`

The landed proof additions are:

- direct correctness coverage for a 2x2 lower-triangular batched two-RHS panel
  solve
- explicit default dense-kernel descriptor coverage for `solve_panel`
- explicit backend-contract coverage when `solve_panel` is missing

That means the new backend-aware seam is proven locally without widening into
`tests/test_integration.c`, because the outer caller-facing public-path
contract did not actually change.

#### 4. Support-only follow-through was not forced

The Day 6 support-only surfaces stayed untouched:

- `include/sparse_cholesky.h`
- `benchmarks/bench_chol_csc.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

That is the correct bounded result because the Day 7 landing changed:

- internal dense-kernel descriptor truth
- supernodal panel-consumption mechanics
- family-local fallback proof

It did not change:

- the public Cholesky header contract
- benchmark-visible path identity wording
- the outer integration/public lifecycle contract
- the maintainer policy split

#### 5. The reviewed validation anchors stayed exact after the backend batch

This was a substantial backend code batch, so I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 334.93 sec`

### Validation

Full required validation for the Day 7 backend batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The family-local proof owner retained the new focused passes:

- `test_chol_dense_solve_panel_2x2_two_rhs`
- `test_supernodal_dense_backend_default_contract`
- `test_supernode_eliminate_panel_missing_solve_panel_is_backend_contract_error`

### Day 7 Exit State

Sprint 75 Day 7 closes with:

1. one explicit dense-kernel batched panel-solve callback seam
2. one supernodal CSC consumer path that uses that seam directly
3. one local backend-contract failure boundary aligned to the actual required
   callback
4. one family-local proof expansion in `tests/test_chol_csc.c`
5. one validated reviewed close without widening into support-only surfaces

## Day 8 - Post-Landing Audit & Rerank

### Goal

Re-rank the remaining Sprint 75 queue after the Day 7 kernel landing and fix
the exact Day 9 design center instead of assuming a second generic backend
batch.

### Actions

1. Audit the Day 7 landing against the Day 3 hotspot ranking and Day 6 audit
   rubric.
2. Re-read the live Cholesky runtime/observability contract across:
   - `include/sparse_cholesky.h`
   - `src/sparse_chol_csc.c`
   - `src/sparse_chol_csc_supernodal.c`
3. Re-read the maintained benchmark and maintainer-policy surfaces to
   distinguish real next-batch seams from support-only drift.
4. Fix the exact Day 9 design center and support-only follow-through list.
5. Record what no longer needs to move in Sprint 75.

### Findings

#### 1. Day 7 closed the strongest dense-kernel/backend-owner seam

The Day 7 landing no longer leaves the strongest backend contradiction in:

- `src/sparse_dense.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`

The batched panel-solve seam is now explicit and locally proven:

- the dense-kernel descriptor owns the callback
- the supernodal consumer uses that callback directly
- the backend-contract failure boundary points at the actual missing required
  callback

That means a second same-family dense-kernel integration batch is no longer
the highest-value next move.

#### 2. The strongest remaining seam is now CSC callback/runtime parity

The strongest remaining contradiction is now the Cholesky CSC
callback/runtime truth seam across:

- `include/sparse_cholesky.h`
- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

The useful Day 8 clarification is:

- the public header still says only the linked-list backend emits progress
- the CSC one-shot dispatch and publish-back lane already owns `used_csc_path`
  and the bounded `SPARSE_ERR_BACKEND_CONTRACT` truth
- after the Day 7 kernel landing, the next high-value question is not another
  dense-kernel callback, but whether CSC runtime observability and
  progress/cancellation semantics should move toward bounded parity

That makes callback/runtime parity the strongest real next batch center.

#### 3. Benchmark proof refresh is real support drift, but not the next batch center

One real support drift now exists in:

- `benchmarks/bench_chol_csc.c`

Its maintained benchmark comment still describes the older row-by-row panel
solve reading even though Day 7 landed a batched panel-solve seam.

That matters, but it remains weaker than the runtime seam because:

- it is benchmark-surface commentary, not the backend-owner code path
- it does not block truthful family-local proof or reviewed validation
- it can move as support-only follow-through once the runtime batch is fixed

So benchmark proof refresh is no longer the strongest remaining contradiction.

#### 4. Residual support-surface drift stayed bounded

The maintained support surfaces remain coherent enough to stay out of the next
design center:

- `docs/maintainer_guide.md`
- `tests/test_integration.c`

Why they are support-only rather than the next batch center:

- `docs/maintainer_guide.md` still states the local backend/performance truth
  correctly at the policy layer
- `tests/test_integration.c` already owns linked-list Cholesky progress/cancel
  coverage and should only move if the CSC public-path runtime contract itself
  widens
- the Day 7 landing did not force a public lifecycle change

#### 5. The exact Day 9 design center is now fixed

Required Day 9 design center:

- `include/sparse_cholesky.h`
- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

Likely support only if the Day 9 design truly forces them:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `docs/maintainer_guide.md`

Explicitly not the next batch center:

- eigensolver backend/runtime parity
- QR backend-aware follow-through
- SVD backend-aware follow-through
- another dense-kernel descriptor expansion in `src/sparse_dense.c`

That fixes the next batch to one real follow-through center instead of
assuming a second generic backend code wave.

### Validation

This was a docs-only Day 8 audit pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the rerank in the Day 7 landed state plus direct rereads of the
Cholesky public runtime contract, CSC dispatch/orchestration seam, supernodal
consumer seam, maintained benchmark proof surface, integration proof owner,
and maintainer-policy truth surface.

### Day 8 Exit State

Sprint 75 Day 8 closes with:

1. one explicit confirmation that Day 7 closed the strongest dense-kernel seam
2. one rerank that moves the next batch to CSC callback/runtime parity
3. one bounded support-only list for benchmark, proof, and policy follow-through
4. one exact Day 9 design center across the Cholesky runtime contract surfaces

## Day 9 - Callback / Runtime Policy Design

### Goal

Define the bounded Cholesky CSC callback/runtime parity batch needed after the
Day 7 kernel landing without turning it into a fake full runtime-policy
rewrite.

### Actions

1. Re-read the Day 7 landing and Day 8 rerank against the live Cholesky
   callback owner in `src/sparse_cholesky.c`.
2. Re-map ownership for:
   - progress callback parity
   - cancellation semantics
   - `used_csc_path` and backend/runtime observability
   - support-only benchmark and maintainer follow-through
3. Decide whether the next batch should add:
   - full CSC kernel-level emission parity
   - bounded CSC orchestration-level emission/cancel checkpoints
   - observability-only docs/header cleanup
4. Fix the exact Day 10 touch set and non-touch list.
5. Record the preserved parity checklist.

### Findings

#### 1. The real callback/runtime owner is the public Cholesky wrapper, not the CSC storage file alone

The Day 8 rerank correctly identified the Cholesky CSC runtime seam, but the
owner split needed one correction after rereading the live code:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`

are the true first owners of the public callback/runtime contract, because:

- `sparse_cholesky_factor_opts(...)` owns backend selection
- it publishes `used_csc_path` before later errors
- it owns reordered-working-copy versus no-reorder input-mutation semantics
- it already owns the linked-list progress/cancel callback entry point

`src/sparse_chol_csc.c` and `src/sparse_chol_csc_supernodal.c` remain
important support seams, but they are not the first public contract owner by
themselves.

#### 2. The truthful next batch is bounded CSC orchestration-level parity, not fake per-column CSC parity

The strongest Day 9 conclusion is:

- do not claim linked-list and CSC callback parity at the same granularity

Why:

- the linked-list lane currently emits one callback per column-elimination
  iteration
- the CSC supernodal lane works through analysis, CSC conversion, supernodal
  factorization, and writeback as a different orchestration shape
- pretending the CSC lane already has equivalent per-column or per-supernode
  public progress semantics would overclaim what the current product surface
  can prove

So the right Day 10 design target is:

- bounded CSC orchestration-level progress/cancel checkpoints
- truthful phase-level runtime observability
- no fake claim of exact per-column parity with the linked-list path

#### 3. Cancellation semantics should stay narrower and more truthful on the CSC lane

The preserved cancellation contract should be:

- linked-list path keeps its current top-of-column callback contract unchanged
- CSC path, if widened, should cancel only at explicit orchestration
  checkpoints before publish-back commits the factor shell into the caller
  matrix

That is the right boundary because:

- it preserves the current truthful one-shot reordered-working-copy story
- it avoids implying mid-supernode rollback that the current backend does not
  actually implement
- it keeps `SPARSE_ERR_CANCELLED` meaningful without conflating it with
  `SPARSE_ERR_BACKEND_CONTRACT`

#### 4. `used_csc_path` and backend observability should remain explicit and unchanged

The Day 10 batch should preserve:

- `used_csc_path` publication after backend selection and before later error
  exit
- `SPARSE_ERR_BACKEND_CONTRACT` as a narrow internal-backend helper/callback
  failure
- the maintained benchmark-side path identity fields in
  `bench_chol_csc.c` as support-only proof, not as the primary callback owner

That means the next batch should tighten the runtime story around the already
real Cholesky wrapper truth instead of redefining the backend observability
contract.

#### 5. The exact Day 10 touch set is now fixed

Required Day 10 touch set:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

Likely support only if the implementation truly forces them:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`
- `tests/test_chol_csc.c`
- `benchmarks/bench_chol_csc.c`
- `docs/maintainer_guide.md`

Explicit non-touch set:

- `src/sparse_dense.c`
- `src/sparse_eigs.c`
- `include/sparse_eigs.h`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `README.md`
- broader platform/install/release workflow surfaces

That keeps Day 10 bounded to one real public callback/runtime follow-through
center.

#### 6. The preserved parity checklist is now explicit

Day 10 must preserve:

- linked-list backend callback semantics unchanged
- CSC `used_csc_path` reporting unchanged
- reordered CSC one-shot failure/cancel paths leave the caller matrix in the
  original coordinate space until successful publish-back
- no claim of mid-kernel or per-column CSC callback parity unless the code
  really lands and proves it
- no reinterpretation of benchmark proof as runtime-policy proof

### Validation

This was a docs-only Day 9 design pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the design in the Day 7 landed state, the Day 8 rerank, the live
public Cholesky wrapper/runtime owner in `src/sparse_cholesky.c`, the current
header truth in `include/sparse_cholesky.h`, the maintained integration proof
owner, and the benchmark/maintainer support surfaces.

### Day 9 Exit State

Sprint 75 Day 9 closes with:

1. one corrected runtime owner split centered on the public Cholesky wrapper
2. one bounded design for CSC orchestration-level callback/cancel parity
3. one exact Day 10 touch set and support-only list
4. one explicit preserved parity checklist
