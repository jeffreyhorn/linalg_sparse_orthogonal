# Sprint 64 Working Notes

## Day 1

**Objective:** Turn the Sprint 64 project-plan scope plus the Sprint 63
validated close into a concrete backend-architecture starting point by
confirming the preserved reviewed baseline, fixing the strongest live dense
kernel and supernodal hotspots, and making the sprint workstreams explicit
before design or code work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 64 project-plan source and the new sprint plan:
   - `sed -n '153,183p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
3. Re-read the strongest inherited Sprint 63 closeout source:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_63/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day14-closeout-and-handoff.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the main Sprint 64 docs/build/kernel/benchmark/proof hotspots:
   - `wc -l README.md docs/tutorial.md docs/maintainer_guide.md CMakeLists.txt Makefile benchmarks/README.md include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_analysis.h src/sparse_dense.c src/sparse_chol_csc_supernodal.c src/sparse_ldlt_csc_supernodal.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c src/sparse_qr.c src/sparse_svd.c benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_ldlt_csc.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt_csc.c examples/example_analysis.c`
7. Scan the live backend, supernodal, and build-option seams:
   - `rg -n "supernod|backend|BLAS|dense kernel|gemm|gemv|dispatch|used_csc_path|SPARSE_OPENMP|backend" src include benchmarks CMakeLists.txt Makefile docs/maintainer_guide.md README.md`

### Day 1 Findings

#### 1. Sprint 64 starts from the Sprint 63 validated close, not from renewed lifecycle or configuration work

Sprint 63 already closed the bounded direct-lifecycle uniformity package:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains the canonical reuse path
- large-`n` CSC-backed repeated-run direct failure-preserve semantics are now
  proved on the public direct path
- the strongest remaining Epic 6 queue is no longer direct-lifecycle
  coherence or Phase 1 typed configuration

Interpretation:

- Sprint 64 is not reopening the Sprint 61 configuration-first debate
- Sprint 64 is not another direct-usability sprint
- Sprint 64 is the first bounded Epic 6 sprint centered on backend
  architecture, selected hot kernels, build/options wiring, fallback
  preservation, and benchmark proof

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the entire backend-architecture sprint

The maintained local truth surfaces remain:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 64 should inherit the exact Sprint 63 truthfulness wording
- later `*.c` / `*.h` landing days should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial backend, build-option, or kernel-sensitive days should still
  treat `make quality-review-full` as the stronger default

#### 3. The broad Epic 6 backend-architecture claim is already concentrated in dense-kernel, supernodal, build-option, and benchmark-proof seams

The live repo now shows the strongest current Sprint 64 pressure clustered in:

- caller-facing and maintained truth surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- build/config surfaces:
  - `CMakeLists.txt`
  - `Makefile`
- highest-value selected implementation seams:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
- proof and workflow surfaces that already carry the backend story:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_ldlt_csc.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `examples/example_analysis.c`

Interpretation:

- Sprint 64 should not pretend every performance-sensitive file is equally
  important
- the highest-value first cut is dense-kernel plus CSC/supernodal follow-through
- the strongest proof burden is already split between correctness/fallback
  regression and benchmark measurement surfaces

#### 4. Sprint 64 reduces cleanly to seven bounded implementation workstreams

The project-plan scope collapses to:

1. hotspot audit
2. backend abstraction design
3. kernel integration batch 1
4. build and option surface
5. benchmark proof refresh
6. regression and safety checks
7. validation and closeout

Interpretation:

- the Sprint 64 implementation order is already smaller and clearer than a
  generic “improve backend architecture” description suggests
- the right Day 1 deliverable is a bounded implementation map with a fixed
  safety and non-goal fence

#### 5. The strongest likely Sprint 64 touch surfaces are now explicit from the live tree

The highest-value current Sprint 64 surfaces are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `988`
  - `docs/tutorial.md` = `469`
  - `docs/maintainer_guide.md` = `398`
  - `benchmarks/README.md` = `249`
- build and option surfaces:
  - `CMakeLists.txt` = `397`
  - `Makefile` = `881`
- public lifecycle/backend-adjacent headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `226`
  - `include/sparse_ldlt.h` = `334`
- strongest implementation/kernel seams:
  - `src/sparse_dense.c` = `506`
  - `src/sparse_chol_csc_supernodal.c` = `556`
  - `src/sparse_ldlt_csc_supernodal.c` = `392`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
- strongest benchmark/example/proof surfaces likely to matter in Phase 1:
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `393`
  - `benchmarks/bench_ldlt_csc.c` = `516`
  - `tests/test_integration.c` = `2367`
  - `tests/test_chol_csc.c` = `4617`
  - `tests/test_ldlt_csc.c` = `3680`
  - `examples/example_analysis.c` = `210`

Interpretation:

- the early code pressure is concentrated enough to support a bounded first
  landing
- the strongest proof pressure already sits in the direct CSC and supernodal
  benchmark/test surfaces rather than in a broad repo-wide performance sweep

#### 6. Sprint 64 needs an explicit Day 1 non-goal fence before any backend design begins

The preserved non-goal fence for Sprint 64 is:

- no broad framework rewrite
- no fake platform closure beyond reviewed evidence
- no backend widening that weakens the self-contained default build
- no benchmark-governance sprawl disguised as kernel work
- no packaging/platform expansion unless a selected kernel landing proves it is
  actually blocking the bounded backend path

Interpretation:

- Sprint 64 should modernize selected hot paths without turning the project
  into a new build or packaging effort
- success is one bounded backend-aware integration package with explicit
  fallback truthfulness, not “pluggable everything”

### Day 1 Close

Sprint 64 now starts from one explicit backend-architecture implementation
baseline:

- the Sprint 63 direct-lifecycle close remains frozen and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 backend claim has already narrowed to dense-kernel,
  supernodal, build-option, and benchmark-proof seams
- the docs/build/kernel/proof hotspots for the first follow-through batch are
  explicit
- the next step is to rank those live hotspots precisely before writing the
  bounded backend abstraction design

## Day 2

**Objective:** Freeze the validation and truthfulness baseline that Sprint 64
backend, build-option, and kernel implementation work must preserve before the
sprint moves into the deeper hotspot audit.

### Commands Run

1. Confirm branch cleanliness before the Day 2 pass:
   - `git status --short --branch`
2. Re-read the current Sprint 64 notes plus the Day 2 plan slice:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/WORKING_NOTES.md`
   - `sed -n '80,160p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
3. Re-read the strongest inherited Day 2 shape from Sprint 63:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day2-validation-baseline-and-touched-surface-recheck.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Re-read the current quality/truthfulness wording:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode|Windows|macOS|Linux|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows`
7. Confirm the Sprint 64 targeted rerun-set presence in the live build tree:
   - `for f in ./build/test_integration ./build/test_chol_csc ./build/test_ldlt_csc ./build/test_cholesky ./build/test_ldlt ./build/test_sparse_lu ./build/test_qr ./build/test_svd ./build/example_analysis ./build/example_basic_solve ./build/example_ldlt ./build/example_svd_lowrank ./build/bench_refactor ./build/bench_refactor_csc ./build/bench_chol_csc ./build/bench_ldlt_csc ./build/bench_eigs_reuse ./build/bench_iterative_reuse; do [ -x "$f" ] && echo "present $f" || echo "missing $f"; done`

### Day 2 Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 64 inherits the same authoritative local validation command as the
Sprint 63 close state:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

#### 2. The reviewed CMake parity count is still the main numerical truthfulness anchor

The current reviewed CMake inventory remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That count still matters because it is the simplest exact proof that:

- the reviewed CMake path still sees the maintained local full test surface
- Makefile/CMake parity has not drifted silently

#### 3. The current code-day gate versus stronger reviewed baseline split is stable

The maintained split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial backend, build-option, or
  performance-sensitive work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

That remains consistent with the repo’s current Sprint 63 close discipline and
does not need reinterpretation on Sprint 64 Day 2.

#### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while the broader Makefile
  reviewed wrappers stay staged
- coverage remains a supplemental signal, not an active reviewed-baseline
  residual
- dead-code remains serialized and separate from the core format/lint/test
  gate

That means Sprint 64 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start backend or build
implementation work.

#### 5. The targeted Sprint 64 rerun set is present and aligned to the actual backend-risk surface

The confirmed rerun set is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
- adjacent dense-kernel and factor/spectral proof sentinels:
  - `./build/test_qr`
  - `./build/test_svd`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
- representative workflow benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_ldlt_csc`
  - `./build/bench_eigs_reuse`
  - `./build/bench_iterative_reuse`

Interpretation:

- Sprint 64 already has a concrete validation surface that matches the actual
  backend-risk concentration
- CSC and supernodal work already has natural family-local proof homes
- the benchmark rerun set is strong enough to support bounded backend proof
  without widening into broad benchmark-governance work

### Day 2 Close

Sprint 64 now has one explicit validation contract before backend code changes
begin:

- `make quality-review-full` remains the strongest local reviewed baseline
- the reviewed CMake parity anchor remains exact at `53`
- the maintained quality/platform story is coherent across the live repo
  surfaces
- the targeted Sprint 64 rerun set is fixed and present in `build/`
- the next step is the deeper hotspot audit that ranks dense-kernel,
  supernodal, and related build/proof seams before design or code work lands

## Day 3

**Objective:** Reduce the broad Sprint 64 “performance backend architecture”
claim to a ranked live seam map by auditing the current dense-kernel,
supernodal, benchmark, and build-sensitive paths before choosing the first
bounded implementation target.

### Commands Run

1. Confirm branch cleanliness before the Day 3 audit:
   - `git status --short --branch`
2. Re-read the Day 3 sprint-plan slice plus current Sprint 64 notes:
   - `sed -n '120,230p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,320p' docs/planning/EPIC_6/SPRINT_64/WORKING_NOTES.md`
3. Re-read the strongest inherited Day 3 audit shape from Sprint 63:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day3-internal-path-audit.md`
4. Map the main backend, dense-kernel, supernodal, and proof-sensitive seams:
   - `rg -n "supernod|dense|gemm|gemv|daxpy|matmul|dispatch|used_csc_path|backend|SPARSE_OPENMP|parallel|thread|auto" include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_analysis.h src/sparse_dense.c src/sparse_chol_csc_supernodal.c src/sparse_ldlt_csc_supernodal.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c src/sparse_qr.c src/sparse_svd.c benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_ldlt_csc.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt_csc.c examples/example_analysis.c`
5. Re-read the strongest current implementation seams:
   - `sed -n '1,260p' src/sparse_dense.c`
   - `sed -n '1,260p' src/sparse_chol_csc_supernodal.c`
   - `sed -n '1,260p' src/sparse_ldlt_csc_supernodal.c`
6. Reconfirm how the dense helpers are exercised and proved:
   - `rg -n "dense_gemm|dense_gemv|chol_dense_factor|chol_dense_solve_lower|ldlt_dense_factor|supernode_eliminate_panel|supernode_eliminate_diag" src tests benchmarks`
   - `sed -n '1,260p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,260p' benchmarks/bench_ldlt_csc.c`

### Day 3 Findings

#### 1. The strongest first Sprint 64 target is the Cholesky CSC supernodal dense-kernel lane

The live Cholesky CSC supernodal path now carries the strongest first-phase
backend leverage:

- `src/sparse_chol_csc_supernodal.c` owns the full batched supernodal flow:
  - extract
  - diagonal-block factor
  - panel solve
  - writeback
- the dense diagonal/panel helpers are compact and self-contained:
  - `chol_dense_factor`
  - `chol_dense_solve_lower`
- the path already has strong public and internal proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`

Interpretation:

- this is the cleanest first backend-architecture landing because it combines
  real runtime leverage with a bounded touched surface
- the benchmark story is already established enough to measure a selected
  backend-aware acceleration without creating a new framework
- the fallback shape is already explicit because the scalar CSC and
  linked-list lanes remain nearby and heavily exercised

#### 2. LDL^T supernodal follow-through is the strongest second target, not the best first one

The LDL^T CSC supernodal path is also backend-worthy, but it is more complex
and more correctness-sensitive as a first landing:

- `src/sparse_ldlt_csc_supernodal.c` mirrors the same extracted dense-panel
  strategy
- but it couples the dense block path to:
  - Bunch-Kaufman pivot structure
  - `D` / `D_offdiag` / `pivot_size` ownership
  - stricter writeback and threshold semantics
- it already has strong family-local proof and benchmark support:
  - `tests/test_ldlt_csc.c`
  - `benchmarks/bench_ldlt_csc.c`

Interpretation:

- this is the best second target after the abstraction seam is proved on the
  Cholesky lane
- it should not be the first landing because its proof burden and pivot-state
  complexity are both higher
- Sprint 64 should avoid pretending “shared dense kernels” means Cholesky and
  LDL^T are equally cheap to modernize on the first pass

#### 3. The generic dense helper layer is important, but only as a bounded internal seam

`src/sparse_dense.c` is a real architecture seam, but it is not yet the whole
story:

- it currently owns only simple column-major helpers:
  - `dense_gemm`
  - `dense_gemv`
- those helpers are well-covered in `tests/test_dense.c`
- the hot CSC supernodal kernels still own their most performance-sensitive
  dense logic locally rather than routing through a broader backend layer

Interpretation:

- the right Sprint 64 move is not “make sparse_dense.c the universal backend
  hub”
- the right move is a bounded internal dense-kernel abstraction used by the
  selected supernodal lane first
- broad QR/SVD dense unification would be a later-phase expansion, not a Day 5
  landing target

#### 4. Build and threading seams are real, but they should remain subordinate to the selected kernel path

The live repo shows build and threading sensitivity mainly through:

- `CMakeLists.txt`
- `Makefile`
- existing `SPARSE_OPENMP` build-time switches
- benchmark and README wording around backend and dispatch behavior

Interpretation:

- build/options work is definitely part of Sprint 64
- but the first abstraction choice should drive the build/option shape, not the
  other way around
- starting from OpenMP or a generic “parallel backend” layer would widen the
  sprint too early and blur the self-contained default-build contract

#### 5. QR and SVD remain later backend candidates, not the best first phase

The live QR and SVD sources still expose dense-kernel opportunities:

- `src/sparse_qr.c`
- `src/sparse_svd.c`

But they rank lower for Sprint 64 Phase 1 because:

- the first benchmark/proof home is less tightly focused than the Cholesky CSC
  lane
- broad dense-kernel unification there would immediately widen the abstraction
  surface
- the fallback and public-story consequences are broader than the first CSC
  supernodal landing

Interpretation:

- QR/SVD should stay in the later lane unless the Day 4 rerank exposes a much
  lower-risk seam than the CSC supernodal path
- Sprint 64 should resist turning “performance backend architecture” into a
  repository-wide dense rewrite

#### 6. The strongest current proof burden already has a natural home

The existing proof burden is already split cleanly enough to support a bounded
Phase 1 landing:

1. `tests/test_chol_csc.c`
2. `benchmarks/bench_chol_csc.c`
3. `tests/test_ldlt_csc.c`
4. `benchmarks/bench_ldlt_csc.c`
5. `tests/test_dense.c`
6. `tests/test_integration.c`

Interpretation:

- Sprint 64 does not need a new benchmark harness or a new backend test
  framework just to start
- the first kernel landing can be proved through the existing CSC family-local
  tests plus the current benchmark surfaces
- `test_dense.c` is the natural lower-level proof surface if the bounded
  abstraction touches generic dense helpers

### Day 3 Close

Sprint 64 now has a ranked live hotspot map instead of a generic backend
architecture backlog:

- the Cholesky CSC supernodal dense-kernel lane is the strongest first target
- LDL^T supernodal follow-through is the strongest second target
- `src/sparse_dense.c` is an important internal seam, but not a universal
  first-class backend hub yet
- build/threading work is real but should follow the selected kernel path
- QR/SVD remain later backend candidates rather than first-phase targets

## Day 4

**Objective:** Re-rank the Day 3 hotspot map against the explicit Epic 6
state-of-the-art target and reduce Sprint 64 to one exact first landing
boundary instead of a broad performance shortlist.

### Commands Run

1. Confirm branch cleanliness before the Day 4 rerank:
   - `git status --short --branch`
2. Re-read the Day 4 sprint-plan slice plus the Day 3 audit:
   - `sed -n '160,260p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day3-performance-hotspot-audit-part1.md`
3. Re-read the explicit Epic 6 state-of-the-art target definition:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_60/artifacts/day5-state-of-the-art-target-definition.md`
4. Re-read the Epic 6 remediation plan section for backend architecture:
   - `sed -n '1,220p' docs/planning/EPIC_6/reviews/todo-codex-2026-06-08.md`

### Day 4 Findings

#### 1. The first Phase 1 Sprint 64 landing should stay anchored to the Cholesky CSC supernodal lane

Against the Epic 6 target definition, the Cholesky CSC supernodal path is
still the best first landing because it satisfies the right combination of:

- bounded touched surface
- real runtime relevance
- existing family-local proof
- explicit fallback neighbors
- low risk of widening the public product story

Interpretation:

- this is the strongest must-touch Phase 1 seam
- it supports a real backend/performance architecture claim without pretending
  the whole repository is already backend-pluggable
- it fits the Epic 6 requirement for a bounded modern backend seam on selected
  hot paths

#### 2. LDL^T supernodal follow-through remains important, but should stay in the second slot

The Day 3 ranking still holds after the target-definition rerank:

- LDL^T supernodal work is valuable
- but it is still more correctness-sensitive and pivot-state-heavy
- it therefore belongs in the next backend follow-through lane, not in the
  first abstraction-defining landing

Interpretation:

- this is an important later Sprint 64 or Sprint 65 seam
- it should benefit from the first bounded kernel abstraction rather than force
  that abstraction to absorb Bunch-Kaufman-specific complexity on day one

#### 3. The generic dense helper layer belongs inside the first landing, but only as an internal dependency seam

The rerank tightens the role of `src/sparse_dense.c`:

- it is now confirmed as part of the first landing boundary
- but only as an internal helper seam in service of the selected Cholesky CSC
  path
- it still should not become a broad “all dense math routes here” rewrite in
  Sprint 64

Interpretation:

- Day 5 should treat `src/sparse_dense.c` as a likely touched implementation
  seam
- Day 5 should not treat QR/SVD-wide dense unification as part of the first
  landing

#### 4. Build/options work is required for Sprint 64, but only as support for the selected kernel path

The rerank against the Epic 6 target confirms that build and option wiring is
real work, but not the thing that defines the first landing:

- it should follow the selected kernel abstraction
- it should preserve the default self-contained build
- it should avoid forcing a public API widening unless the first landing truly
  needs it

Interpretation:

- build/options wiring is in-bounds for Sprint 64
- broad platform or packaging work is not
- Day 5 should design the kernel abstraction first and then derive the minimum
  necessary build/option surface from it

#### 5. Benchmark-governance and broad packaging work remain explicitly out of the first landing boundary

The target-definition rerank makes the non-goal fence sharper:

- benchmark proof refresh is still in scope
- broad benchmark-governance redesign is not
- platform/packaging maturity remains an Epic 6 band, but not part of the
  first Sprint 64 landing

Interpretation:

- the first landing should only require:
  - bounded benchmark proof
  - bounded docs/maintainer truthfulness updates
- it should not absorb:
  - packaging strategy work
  - release-shape work
  - Windows/macOS parity expansion

### First Selected Sprint 64 Landing Surface

The exact first selected Sprint 64 landing surface is now:

- required first kernel lane:
  - Cholesky CSC supernodal dense-kernel path
- required nearby internal seam:
  - bounded dense-helper abstraction support
- required proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
- likely supporting truth surfaces later:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - build wiring only if the selected abstraction actually needs it

### Explicit Deferred / Later Queue

The Day 4 rerank fixes the later queue explicitly:

- second backend target:
  - LDL^T supernodal follow-through
- later dense-kernel/backend candidates:
  - QR
  - SVD
- later support bands:
  - broader benchmark-governance work
  - packaging/platform maturity work
  - broader threading-policy generalization

### Day 4 Close

Sprint 64 now has one exact first landing boundary instead of a generic
backend shortlist:

- the Cholesky CSC supernodal dense-kernel lane is fixed as the first landing
- LDL^T supernodal follow-through remains the strongest second target
- `src/sparse_dense.c` is part of the first landing only as a bounded internal
  seam
- build/options work is confirmed as support work, not the first design center
- packaging/platform and broad benchmark-governance work remain explicitly out
  of the first landing

## Day 5

**Objective:** Define the bounded backend abstraction contract for the selected
Sprint 64 hot path, including exact ownership across the local kernel seam,
fallback behavior, proof surfaces, and the Day 6-10 touched-file fence.

### Commands Run

1. Re-read the Day 5-6 sprint-plan slice plus the landed Day 4 rerank:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day4-performance-hotspot-rerank-and-first-landing-boundary.md`
2. Inspect the live selected-kernel and helper seams directly:
   - `sed -n '1,260p' src/sparse_chol_csc_supernodal.c`
   - `sed -n '1,260p' src/sparse_dense.c`
3. Reconfirm the actual helper/proof footprint around the selected lane:
   - `rg -n "chol_dense_|dense_gemm|dense_gemv|supernodal" src/sparse_chol_csc_supernodal.c src/sparse_dense.c src/sparse_chol_csc.c src/sparse_ldlt_csc_supernodal.c tests/test_chol_csc.c benchmarks/bench_chol_csc.c`

### Day 5 Findings

#### 1. The first Sprint 64 backend abstraction should stay local to the Cholesky CSC supernodal lane

The selected first landing does not justify a generic repository-wide backend
layer.

The live code already shows a tighter and safer boundary:

- `src/sparse_chol_csc_supernodal.c` owns the selected hot path directly
- the densest local operations are currently:
  - `chol_dense_factor(...)`
  - `chol_dense_solve_lower(...)`
- those kernels are tightly coupled to:
  - supernode extract
  - diagonal-block factor
  - panel solve
  - writeback

Interpretation:

- the first abstraction should be lane-local
- it should not try to turn all dense math in the repository into one new
  universal backend interface

#### 2. The generic dense helper seam belongs in-bounds, but only for bounded support

The live helper layer in `src/sparse_dense.c` matters, but its role in Sprint
64 is narrower than “new dense backend hub”:

- existing generic helpers:
  - `dense_gemm(...)`
  - `dense_gemv(...)`
- selected Cholesky CSC kernels still live outside that file
- the first landing therefore needs a bounded helper/support seam, not a
  repo-wide relocation of every dense operation

Interpretation:

- Day 6 should treat `src/sparse_dense.c` as optional support plumbing only if
  the selected Cholesky CSC kernel path actually benefits from shared helper
  entry points
- Day 6 should not widen into QR/SVD dense unification

#### 3. The default-path and fallback contract is now explicit

The first backend-aware landing must preserve the current authoritative path:

- the self-contained default build remains the truth surface
- any backend-aware fast path must be optional and bounded
- scalar CSC and current supernodal semantics remain the fallback/correctness
  anchors
- proof must demonstrate equivalence or preserved contract, not merely the
  presence of a new dispatch seam

Interpretation:

- Sprint 64 should prefer internal dispatch and build-time enablement first
- public option widening is only justified if the selected kernel path truly
  needs a caller-visible control

#### 4. The strongest proof home is already exact enough for the first landing

The first implementation batch does not need a new proof framework.

The natural proof split is already present:

- family-local correctness:
  - `tests/test_chol_csc.c`
- public lifecycle/non-regression:
  - `tests/test_integration.c`
- throughput/proof benchmark:
  - `benchmarks/bench_chol_csc.c`

Interpretation:

- Day 6-10 should keep the first proof burden centered there
- broader benchmark README or maintainer-guide truth surfaces should only move
  after the implementation shape is real

#### 5. The Day 6-10 touched-file fence is now exact

The first implementation fence is now:

- required implementation seam:
  - `src/sparse_chol_csc_supernodal.c`
- likely bounded support seam:
  - `src/sparse_dense.c`
- likely CSC wrapper/dispatch seam only if required by the landed contract:
  - `src/sparse_chol_csc.c`
- required proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
- likely build/options surfaces only if the selected abstraction truly needs
  them:
  - `CMakeLists.txt`
  - `Makefile`

Explicit non-goals for the first landing:

- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- public API/header widening by default
- packaging/platform work
- broad benchmark-governance work
- threading-policy generalization beyond the selected kernel path

### Day 5 Close

Sprint 64 now has an exact backend abstraction contract for the first landing:

- the abstraction stays local to the Cholesky CSC supernodal lane
- `src/sparse_dense.c` is only a bounded support seam
- default-build and fallback correctness stay authoritative
- the proof home is fixed to `test_chol_csc`, `test_integration`, and
  `bench_chol_csc`
- the Day 6-10 touched-file fence is explicit before implementation begins

## Day 6

**Objective:** Convert the Day 5 backend contract into an exact build/options
wiring plan without widening the public surface or weakening the self-contained
default build.

### Commands Run

1. Re-read the Day 6 sprint-plan slice plus the landed Day 5 contract:
   - `sed -n '120,260p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day5-backend-abstraction-contract-design.md`
2. Re-read the live build surfaces directly:
   - `sed -n '1,260p' CMakeLists.txt`
   - `sed -n '1,260p' Makefile`
3. Re-read the current internal CSC/Cholesky seam and existing build-option vocabulary:
   - `sed -n '1,260p' src/sparse_chol_csc_internal.h`
   - `rg -n "SPARSE_OPENMP|SPARSE_MUTEX|option\\(|find_package\\(|compile_definitions|backend" src include CMakeLists.txt Makefile docs/maintainer_guide.md benchmarks/README.md README.md`
4. Re-read the Day 7-12 sprint-plan slice to make the implementation fence line up with later batches:
   - `sed -n '260,420p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`

### Day 6 Findings

#### 1. Sprint 64 does not justify a new public runtime/backend option

The live repository already uses public-facing backend selectors where the
product contract truly needs them:

- `sparse_cholesky_opts_t::backend`
- `sparse_ldlt_opts_t::backend`
- `sparse_eigs_opts_t::backend`

The first Sprint 64 landing does not meet that bar.

Interpretation:

- no new public header field should be added for the first backend-aware kernel
  slice
- no new README/tutorial/runtime option should be introduced in Day 7-10
- no env-var control should be added

#### 2. If a toggle is needed, it should be build-time, target-private, and default-safe

The live build surface already distinguishes optional implementation features
through compile-time switches:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`

That pattern is the closest existing fit for Sprint 64, but the first landing
should be narrower:

- any Sprint 64 backend-selection toggle should be `PRIVATE` to the library
  target
- it should not become a documented general product feature during the first
  landing
- the default setting should preserve the current authoritative self-contained
  path

Interpretation:

- the preferred Day 10 wiring shape is a bounded compile-time switch only if
  the implementation truly needs a selectable local-versus-helper-backed path
- otherwise the first landing should avoid adding any new build option at all

#### 3. The natural internal policy home is the Cholesky CSC internal seam, not a new public config layer

The live code already has the right internal seam:

- `src/sparse_chol_csc_internal.h`

Interpretation:

- any first-landing policy enum, macro, or helper declaration should live in
  the Cholesky CSC internal seam
- Sprint 64 should not add a new repository-wide backend-config header
- Sprint 64 should not widen `include/` just to expose an implementation-local
  toggle

#### 4. The minimum viable build-surface plan is now explicit

The Day 6 wiring plan is:

- first preference:
  - no new build toggle if the landed kernel modernization can stay on the
    existing authoritative path directly
- second preference only if the implementation actually needs a selectable
  branch:
  - `CMakeLists.txt` gains one bounded cache/string or ON/OFF option for the
    selected Cholesky CSC supernodal kernel lane
  - `Makefile` mirrors that switch with the same semantics
  - both surfaces emit only a target-private compile definition
  - default remains the current self-contained path

The selected path should therefore be:

1. default authoritative local/self-contained path
2. optional bounded compile-time-selected Sprint 64 kernel path, only if
   needed
3. no public runtime/backend forcing in this phase

#### 5. The Day 7-10 touched-file fence is now exact

Required first-landing implementation surface:

- `src/sparse_chol_csc_supernodal.c`

Likely internal support surface:

- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`

Likely proof surfaces:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `benchmarks/bench_chol_csc.c`

Conditional wiring surfaces only if the implementation proves they are needed:

- `CMakeLists.txt`
- `Makefile`
- later truth surfaces:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`

Explicit non-goals:

- no new public option in `include/sparse_cholesky.h`
- no new repo-wide backend-config layer
- no widening into `src/sparse_ldlt_csc_supernodal.c`
- no widening into QR/SVD
- no benchmark-governance redesign
- no packaging/platform spillover

### Day 6 Close

Sprint 64 now has an exact build/options wiring plan before implementation:

- no new public runtime/backend option is justified for the first landing
- any needed toggle should be build-time, target-private, and default-safe
- the natural internal policy home is `src/sparse_chol_csc_internal.h`
- `CMakeLists.txt` and `Makefile` are conditional support surfaces, not
  mandatory first-batch edits
- the Day 7-10 implementation fence is explicit before code moves
