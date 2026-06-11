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

## Day 7

**Objective:** Convert the abstraction and option design into the exact
touched-file and proof plan for the first Sprint 64 code batch.

### Commands Run

1. Re-read the Day 7-9 sprint-plan slice plus the landed Day 5-6 contracts:
   - `sed -n '260,360p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day5-backend-abstraction-contract-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day6-build-option-surface-design.md`
2. Re-read the live family-local and public proof homes around the selected lane:
   - `sed -n '3750,3895p' tests/test_chol_csc.c`
   - `sed -n '1760,1945p' tests/test_integration.c`
3. Re-read the maintained benchmark proof surface:
   - `sed -n '1,280p' benchmarks/bench_chol_csc.c`

### Day 7 Findings

#### 1. The Day 8 implementation batch should stay centered on the supernodal Cholesky kernel file

The selected first code batch should keep one clear implementation center:

- `src/sparse_chol_csc_supernodal.c`

Interpretation:

- Day 8 should land the first backend-aware kernel slice there first
- `src/sparse_dense.c` should only move if the selected kernel change actually
  needs bounded helper support
- `src/sparse_chol_csc.c` should only move if the implementation truly needs a
  small dispatch or contract bridge

#### 2. Regression proof and benchmark proof are now explicitly separated

The live proof surfaces already split cleanly enough for the first landing:

- family-local correctness:
  - `tests/test_chol_csc.c`
- public lifecycle/non-regression:
  - `tests/test_integration.c`
- benchmark proof:
  - `benchmarks/bench_chol_csc.c`

Interpretation:

- Day 8 should use `tests/test_chol_csc.c` for kernel-local equivalence,
  fallback, and error-path checks
- Day 8 should use `tests/test_integration.c` only for the smallest public
  contract proof if the landed semantics cross the family boundary
- Day 8 should not widen benchmark proof into benchmark-governance work

#### 3. The minimum viable fallback-preserve contract is now fixed

The first backend-aware landing does not need to solve every backend/fallback
question in Sprint 64.

The minimum viable fallback-preserve rule is:

- the default self-contained path remains authoritative
- any new helper-backed or backend-aware fast path must preserve scalar CSC and
  current supernodal correctness where the same matrix lands on the same
  public workflow
- failure or unsupported-path behavior must remain explicit and non-silent

Interpretation:

- the first landing should prefer bounded equivalence/fallback proof over broad
  new dispatch policy
- Day 9 should audit any remaining missing truthfulness around selection or
  fallback after the code lands

#### 4. The minimum viable benchmark signal is now explicit

The maintained benchmark surface already provides the right first proof shape:

- linked-list baseline
- CSC scalar comparison lane
- CSC supernodal comparison lane
- comparable factor/solve timing columns
- residual checks

Interpretation:

- Day 8 does not need a new benchmark format
- Day 11 can refresh the benchmark proof surface only if the landed kernel path
  needs one or two extra signal fields to show:
  - selected path used
  - fallback path preserved
  - performance comparison still interpretable

#### 5. The Day 8 and Day 9-12 fence is now exact

Day 8 first code-batch fence:

- required implementation seam:
  - `src/sparse_chol_csc_supernodal.c`
- optional bounded support seam only if the landed code proves it necessary:
  - `src/sparse_dense.c`
- optional dispatch/bridge seam only if required:
  - `src/sparse_chol_csc.c`
- required proof surfaces:
  - `tests/test_chol_csc.c`
  - optional bounded `tests/test_integration.c`

Day 9-12 follow-through queue:

- Day 9:
  - post-landing safety audit
  - remaining selection/fallback/error-path proof rerank
- Day 10:
  - smallest required build/dispatch follow-through only if Day 8 proves it
    necessary
- Day 11:
  - bounded benchmark proof refresh in `benchmarks/bench_chol_csc.c`
- Day 12:
  - docs/maintainer truth follow-through only after landed semantics are real

Explicit non-goals:

- no LDL^T batch in the first landing
- no QR/SVD widening
- no public header widening
- no broad benchmark README rewrite in Day 8
- no packaging/platform or threading-policy widening

### Day 7 Close

Sprint 64 now has an exact first code-batch and proof plan:

- Day 8 should center on `src/sparse_chol_csc_supernodal.c`
- `src/sparse_dense.c` and `src/sparse_chol_csc.c` are conditional only
- regression proof and benchmark proof are explicitly separated
- the minimum viable fallback-preserve behavior is fixed
- the Day 9-12 follow-through queue is bounded before implementation begins

## Day 8

**Objective:** Land the first bounded backend-aware kernel integration slice on
the Cholesky CSC supernodal path without widening the public surface, the
build/option surface, or the sprint fence.

### Commands Run

1. Re-read the Day 7 landing fence and the selected implementation/proof
   surfaces:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_64/artifacts/day7-kernel-integration-landing-design.md`
   - `sed -n '1,220p' src/sparse_chol_csc_internal.h`
   - `sed -n '1,260p' src/sparse_dense.c`
   - `sed -n '1,320p' src/sparse_chol_csc_supernodal.c`
   - `sed -n '4300,4475p' tests/test_chol_csc.c`
2. Inspect the exact landed code diff:
   - `git diff -- src/sparse_chol_csc_internal.h src/sparse_dense.c src/sparse_chol_csc_supernodal.c tests/test_chol_csc.c`
3. Run the required formatting and compile-quality gates:
   - `make format`
   - `make lint`
4. Run the local test gate and confirm the final result:
   - `make test >/tmp/s64d8_make_test.log 2>&1`
   - `tail -n 20 /tmp/s64d8_make_test.log`
5. Run the strongest reviewed baseline and capture the final anchors:
   - `make quality-review-full >/tmp/s64d8_quality_review_full.log 2>&1`
   - `grep -n 'quality-review-cmake-compile: CMake tests:\\|Total Test time (real)\\|quality-review-cmake: passed\\|quality-review-full: passed' /tmp/s64d8_quality_review_full.log`
   - `tail -n 80 /tmp/s64d8_quality_review_full.log`

### Day 8 Findings

#### 1. Sprint 64 now has its first bounded backend-aware integration seam

The landed batch introduces one internal dense-kernel descriptor for the
selected Cholesky CSC supernodal lane:

- `chol_dense_kernels_t`
- `chol_csc_supernodal_dense_kernels()`

The ownership split stays inside the Day 7 fence:

- declaration and contract:
  - `src/sparse_chol_csc_internal.h`
- default builtin kernel implementation and descriptor:
  - `src/sparse_dense.c`
- selected hot-path consumption:
  - `src/sparse_chol_csc_supernodal.c`

Interpretation:

- Sprint 64 now has a real backend-aware integration seam
- the seam is local to the selected hot path
- the self-contained builtin path remains authoritative
- no public API or build-surface widening was required for the first landing

#### 2. The supernodal Cholesky lane no longer hardwires its dense helpers locally

Before the Day 8 batch, the selected hot path owned local dense helper
implementations directly inside `src/sparse_chol_csc_supernodal.c`.

After the landing:

- the dense factor helper and lower-triangular solve helper live in
  `src/sparse_dense.c`
- the supernodal path resolves them through the bounded internal descriptor
- the call sites now defend against a missing descriptor or missing function
  pointers with an explicit `SPARSE_ERR_BACKEND_CONTRACT`

Interpretation:

- the selected lane now reads as architecture-aware instead of file-local and
  closed-over
- the fallback/self-contained path is still the default shipped behavior
- the first abstraction stayed narrow enough to avoid fake repository-wide
  backend generalization

#### 3. The first proof burden stayed family-local and bounded

The Day 8 proof stayed in the natural family-local surface:

- `tests/test_chol_csc.c`

New proof added:

- `test_supernodal_dense_backend_default_contract`

That proof pins the minimum viable contract for the first landing:

- `chol_csc_supernodal_dense_kernels()` is present
- the builtin backend name is present
- the factor and solve function pointers are present
- the default builtin kernel pair still produces a correct small dense
  factor/solve result

Interpretation:

- the first backend-aware proof did not need public lifecycle widening
- `tests/test_integration.c` stayed untouched
- benchmark proof can remain a later follow-through concern instead of a Day 8
  blocker

#### 4. The batch stayed completely inside the Day 7 non-goal fence

The Day 8 landing did not widen into:

- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `CMakeLists.txt`
- `Makefile`
- `include/` public headers
- benchmark governance or platform work

Interpretation:

- the first Sprint 64 code batch is still one bounded Cholesky CSC supernodal
  architecture slice
- later build-option or benchmark follow-through can stay conditional on the
  live branch state rather than being forced into the first landing

#### 5. Validation completed cleanly from the strongest reviewed baseline

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 355.19 sec`

Non-blocking observation:

- the reviewed CMake pass took longer than the ordinary local `make test` path
  because `test_reorder_nd` ran to `229.54 sec` in the reviewed build tree,
  but the full reviewed path still completed cleanly and passed all parity
  gates

### Day 8 Close

Sprint 64 now has its first landed backend-aware integration batch:

- the Cholesky CSC supernodal hot path resolves dense kernels through a bounded
  internal descriptor
- the builtin self-contained dense implementation remains the shipped default
- family-local proof now pins the default backend contract explicitly
- the batch stayed inside the planned touched-file fence and passed the full
  reviewed validation baseline

## Day 9

**Objective:** Re-rank the remaining Sprint 64 backend queue from the live
Day 8 branch state, fix the highest-value missing fallback/truthfulness proof,
and bound the Day 10-12 follow-through to real residual seams instead of the
pre-landing design assumptions.

### Commands Run

1. Re-read the Day 8-10 sprint-plan slice plus the landed Day 8 artifact:
   - `sed -n '300,380p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/artifacts/day8-kernel-integration-batch1.md`
2. Re-read the live Day 8 implementation seams:
   - `sed -n '620,710p' src/sparse_chol_csc_internal.h`
   - `sed -n '200,260p' src/sparse_dense.c`
   - `sed -n '380,470p' src/sparse_chol_csc_supernodal.c`
3. Re-read the current family-local and benchmark proof surfaces:
   - `sed -n '2560,2615p' tests/test_chol_csc.c`
   - `sed -n '1,260p' benchmarks/bench_chol_csc.c`
4. Re-scan the current backend- and telemetry-sensitive wording:
   - `rg -n "backend|used_csc_path|supernodal|chol_dense|name|SPARSE_ERR_BACKEND_CONTRACT|SPARSE_ERR_BADARG" src tests benchmarks README.md docs/maintainer_guide.md include/sparse_cholesky.h`
5. Reconfirm branch cleanliness before the docs-only Day 9 writeup:
   - `git status --short --branch`

### Day 9 Findings

#### 1. The Day 8 landing closed the broad “first backend abstraction” problem

After the landed Day 8 batch, Sprint 64 no longer has a generic “introduce a
backend-aware seam” backlog item.

Already true on the live branch:

- the Cholesky CSC supernodal lane resolves dense helpers through one bounded
  internal descriptor
- the builtin self-contained kernel set remains the authoritative default
- family-local proof now pins the default descriptor contract directly
- no public runtime/backend surface or build-surface widening was needed for
  the first landing

Interpretation:

- the remaining Sprint 64 queue is no longer abstraction-first
- the remaining queue is now about bounded fallback/error-path truthfulness,
  proof placement, and later benchmark/docs follow-through only where the
  landed semantics actually require it

#### 2. The strongest remaining seam is now the internal fallback/error-path contract, not selection policy

The live Day 8 branch exposes one clear residual seam:

- `src/sparse_chol_csc_supernodal.c` now treats a missing dense-kernel
  descriptor or missing function pointer as an explicit error-path case
- the landed Day 8 notes/artifact already treat that path as a distinct
  backend-contract failure lane
- the live code currently returns `SPARSE_ERR_BADARG`

Interpretation:

- the strongest remaining Day 10 target is no longer broad build-option wiring
- it is one small fallback-truthfulness follow-through on the exact internal
  dense-kernel seam that Day 8 introduced
- the right next move is to align the shipped error classification and then
  prove it in the family-local test surface

#### 3. Benchmark proof is no longer the next implementation blocker

The current benchmark surface in `benchmarks/bench_chol_csc.c` still measures:

- linked-list baseline
- CSC scalar path
- CSC supernodal path
- factor/solve timings
- residual checks

What it does not yet surface is the internal dense-kernel descriptor name.

Interpretation:

- that is real later observability work
- it is not yet the strongest Day 10 target because the first landing still
  has only one authoritative builtin descriptor
- Day 11 should only widen benchmark output if the Day 10 follow-through
  introduces an observable internal-selection or fallback distinction worth
  printing

#### 4. Public docs/header follow-through is now secondary to the internal proof seam

No public header or README contradiction appeared after the Day 8 landing:

- no public backend selector was widened
- no public lifecycle rule moved
- no benchmark/docs claim currently depends on the dense-kernel descriptor name

Interpretation:

- Day 12 should stay conditional
- maintainer/docs truth surfaces should only move after the Day 10 internal
  seam lands, if the final error/fallback contract becomes sharper in a way
  worth recording

#### 5. The Day 10-12 queue is now exact and smaller than the Day 7 design implied

Updated rank order:

1. strongest next target:
   - internal fallback/error-path truthfulness on the Day 8 dense-kernel seam
2. secondary target:
   - family-local proof expansion for that seam
3. conditional later target:
   - benchmark observability follow-through only if the landed Day 10 shape
     creates a real output distinction
4. conditional later target:
   - maintainer/docs wording only if the landed Day 10 contract materially
     sharpens the current story

Exact Day 10 touched-file fence:

- required:
  - `src/sparse_chol_csc_supernodal.c`
  - `tests/test_chol_csc.c`
- likely support:
  - `src/sparse_chol_csc_internal.h`
  - `src/sparse_dense.c`
- explicitly not required unless the implementation proves otherwise:
  - `CMakeLists.txt`
  - `Makefile`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
  - public headers or top-level docs

The intended Day 10 proof shape is now explicit:

- keep the proof family-local in `tests/test_chol_csc.c`
- add a bounded override or test seam only if needed to simulate a missing
  descriptor or missing function pointer
- prove the supernodal lane returns the final intended internal error code
  explicitly rather than relying on the implicit builtin-always-present
  assumption

### Day 9 Close

Sprint 64 Day 9 closes with a materially smaller remaining queue:

- the first backend-aware abstraction problem is already solved
- the strongest remaining seam is the internal fallback/error-path contract on
  the new dense-kernel descriptor lane
- benchmark and docs follow-through are now conditional, not automatic
- Day 10 can proceed from an exact touched-file fence and a consciously
  smaller proof queue

## Day 10

**Objective:** Land the bounded fallback/error-path truthfulness slice on the
Day 8 dense-kernel seam by introducing a real public backend-contract error
code, wiring the supernodal Cholesky lane to use it, and proving the new
contract in the family-local test surface.

### Commands Run

1. Re-read the Day 9 design plus the touched implementation/proof surfaces:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day9-post-landing-safety-audit-and-proof-rerank.md`
   - `sed -n '390,455p' src/sparse_chol_csc_supernodal.c`
   - `sed -n '55,95p' include/sparse_types.h`
   - `sed -n '1,120p' src/sparse_types.c`
   - `sed -n '3838,3875p' tests/test_chol_csc.c`
2. Inspect the exact landed code diff:
   - `git diff -- include/sparse_types.h src/sparse_types.c src/sparse_chol_csc_internal.h src/sparse_dense.c src/sparse_chol_csc_supernodal.c tests/test_chol_csc.c`
3. Run the required code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
4. Run the stronger reviewed baseline:
   - `make quality-review-full >/tmp/s64d10_quality_review_full.log 2>&1`
   - `grep -n 'quality-review-cmake-compile: CMake tests:\\|Total Test time \\(real\\)\\|quality-review-cmake: passed\\|quality-review-full: passed' /tmp/s64d10_quality_review_full.log`
   - `tail -n 80 /tmp/s64d10_quality_review_full.log`

### Day 10 Findings

#### 1. Sprint 64 now has a real public error-taxonomy answer for backend-contract failure

The blocked Day 10 question was whether the new supernodal dense-kernel seam
should keep using `SPARSE_ERR_BADARG` or grow a dedicated error code.

The landed answer is explicit now:

- new public error enum value:
  - `SPARSE_ERR_BACKEND_CONTRACT`
- public declaration:
  - `include/sparse_types.h`
- stringification:
  - `src/sparse_types.c`

Interpretation:

- the backend-aware seam no longer overloads `BADARG` for an internal
  implementation-contract violation
- the repo now has a stable public error code for “the caller contract was
  valid, but the selected backend path could not resolve a required internal
  helper/callback”

#### 2. The supernodal Cholesky dense-kernel seam now uses the final shipped classification

The selected hot path now returns `SPARSE_ERR_BACKEND_CONTRACT` when:

- the dense-kernel descriptor is missing
- the descriptor is present but `factor` is missing
- the descriptor is present but `solve_lower` is missing

Touched implementation surfaces:

- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc_internal.h`
- `src/sparse_dense.c`

Interpretation:

- Sprint 64 no longer has a truthfulness mismatch between the Day 8/9 design
  and the live code
- the fallback/error-path contract on the first backend-aware seam is now
  explicit and stable

#### 3. The family-local proof now exercises the seam directly instead of assuming the builtin descriptor always exists

`tests/test_chol_csc.c` now includes a bounded test-only override seam for the
active dense-kernel descriptor:

- `chol_csc_supernodal_set_dense_kernels_override_for_test(...)`
- `chol_csc_supernodal_clear_dense_kernels_override_for_test(...)`

New proofs now pass:

- `test_supernode_eliminate_diag_missing_dense_kernel_descriptor_is_backend_contract_error`
- `test_supernode_eliminate_diag_missing_factor_kernel_is_backend_contract_error`
- `test_supernode_eliminate_panel_missing_solve_kernel_is_backend_contract_error`

Interpretation:

- the missing-descriptor and missing-function-pointer paths are now proved
  explicitly
- the proof stayed inside the Day 9 fence:
  - no `tests/test_integration.c`
  - no benchmark widening
  - no public docs/header follow-through yet

#### 4. The Day 10 batch stayed tightly bounded

The landed batch did not widen into:

- `CMakeLists.txt`
- `Makefile`
- `tests/test_integration.c`
- `benchmarks/bench_chol_csc.c`
- public Cholesky headers or top-level docs
- LDL^T / QR / SVD follow-through

Interpretation:

- Day 10 is one bounded contract-tightening slice, not a second architecture
  redesign
- Sprint 64 can still decide later whether benchmark observability or docs
  follow-through are justified from the actual landed semantics

#### 5. Validation completed cleanly from the reviewed baseline

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 365.95 sec`

Non-blocking note:

- `make test` continued to show the usual long `test_reorder_nd` tail
- the reviewed CMake path again re-emitted the ordinary
  `bench_eigs_reuse.c` double-promotion warnings while rebuilding
  `bench_eigs_reuse`, but still completed cleanly and passed all parity gates

### Day 10 Close

Sprint 64 Day 10 now hands off a much smaller queue:

- the first backend-aware Cholesky CSC seam has a real public error-taxonomy
  answer
- the supernodal path now reports internal backend-contract failure explicitly
- family-local proof now exercises missing descriptor and missing function
  pointer paths directly
- the next work can stay focused on whether benchmark or maintainer/docs
  follow-through is actually justified from the landed semantics

## Day 11

**Objective:** Refresh the maintained benchmark proof surface for the landed
Sprint 64 Cholesky CSC backend-aware path so the benchmark output identifies
the active dense-kernel descriptor directly without widening into broad
benchmark-governance work.

### Commands Run

1. Re-read the Day 11 sprint fence and the landed Day 10 backend seam:
   - `sed -n '300,390p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '650,730p' src/sparse_chol_csc_internal.h`
   - `sed -n '390,460p' src/sparse_chol_csc_supernodal.c`
   - `sed -n '3840,3965p' tests/test_chol_csc.c`
   - `sed -n '1,260p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,240p' benchmarks/README.md`
2. Land the bounded benchmark refresh:
   - `benchmarks/bench_chol_csc.c`
   - `benchmarks/README.md`
3. Run the required code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
4. Re-run the selected benchmark proof set:
   - `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/bench_chol_csc tests/data/suitesparse/bcsstk04.mtx --repeat 1`
   - `./build/bench_chol_csc --small-corpus --repeat 1 | head -n 6`

### Day 11 Findings

#### 1. The strongest remaining benchmark gap was path measurability, not another timing column

Before the Day 11 batch, `bench_chol_csc` already exposed:

- linked-list timing
- CSC scalar timing
- CSC supernodal timing
- residuals for all three

What it did not expose was the active dense-kernel descriptor behind the new
Sprint 64 backend-aware supernodal lane.

Interpretation:

- the maintained benchmark surface could show supernodal timing
- but it could not prove which dense-kernel descriptor backed that lane on the
  actual run
- the right Day 11 refresh was a narrow output-surface addition, not new
  benchmark workflows

#### 2. `bench_chol_csc` now identifies the maintained scalar lane, supernodal lane, and active dense-kernel descriptor explicitly

The benchmark CSV now adds three path-identification fields:

- `csc_scalar_path`
- `csc_supernodal_path`
- `csc_supernodal_dense_kernel`

The shipped default values are now explicit in the maintained proof surface:

- `csc_scalar_path = scalar`
- `csc_supernodal_path = supernodal`
- `csc_supernodal_dense_kernel = builtin`

Interpretation:

- the benchmark can now show that the maintained fallback lane is still the
  scalar CSC path
- the benchmark can now show that the accelerated lane is the supernodal path
- the benchmark can now show which dense-kernel descriptor backed the
  supernodal lane for the reported numbers

#### 3. The benchmark refresh stayed inside the bounded Sprint 64 fence

The landed Day 11 batch did not widen into:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_dense.c`
- `CMakeLists.txt`
- `Makefile`
- benchmark-governance policy rewrites

Touched surfaces stayed bounded to:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/README.md`

Interpretation:

- Day 11 is a benchmark proof refresh, not another backend-implementation
  sprint inside the sprint
- the measured path now reads truthfully without widening the kernel or build
  contract

#### 4. The refreshed benchmark output is now concrete and representative

Representative retained outputs:

- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `nos4.mtx,100,594,scalar,supernodal,builtin,0.800,1.024,0.715,0.010,0.010,0.005,0.78,1.12,7.06e-16,5.89e-16,5.89e-16`
- `./build/bench_chol_csc tests/data/suitesparse/bcsstk04.mtx --repeat 1`
  - `bcsstk04.mtx,132,3648,scalar,supernodal,builtin,4.375,4.144,4.347,0.047,0.023,0.018,1.06,1.01,6.05e-16,1.06e-15,9.08e-16`
- `./build/bench_chol_csc --small-corpus --repeat 1 | head -n 6`
  - header plus small-corpus rows now all carry `scalar,supernodal,builtin`

Interpretation:

- the benchmark proof surface now demonstrates the active dense-kernel
  descriptor directly on both SuiteSparse and threshold-retrospective rows
- the timing columns remain comparable because the refreshed fields only add
  identification, not a new measurement mode

#### 5. Validation completed cleanly for the bounded benchmark-surface change

Ran:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

Day 11 note:

- this was a bounded benchmark-surface refresh on a benchmark-only `*.c`
  binary plus benchmark docs
- I did not rerun `make quality-review-full` because the Day 11 batch did not
  change library implementation, public headers, build wiring, or fallback
  semantics; the required code-day gate plus the refreshed benchmark proof set
  were sufficient for this slice

### Day 11 Close

Sprint 64 Day 11 now hands off a smaller final queue:

- the maintained benchmark proof surface for the first backend-aware Cholesky
  CSC lane now identifies the active dense-kernel descriptor directly
- fallback and accelerated CSC lanes remain measurable and comparable in one
  CSV surface
- Day 12 can stay focused on any remaining bounded regression/docs/maintainer
  follow-through instead of reopening benchmark proof questions

## Day 12

**Objective:** Close the remaining bounded public-header, README, and
maintainer-story gaps around the Sprint 64 backend-aware Cholesky CSC lane so
the new benchmark/output truth and the new public error taxonomy read
coherently across maintained surfaces.

### Commands Run

1. Re-read the Day 12 sprint fence and the current touched surfaces:
   - `sed -n '420,520p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '150,215p' include/sparse_cholesky.h`
   - `sed -n '280,335p' docs/maintainer_guide.md`
   - `sed -n '520,590p' README.md`
   - `sed -n '80,155p' benchmarks/README.md`
2. Land the bounded follow-through batch:
   - `include/sparse_cholesky.h`
   - `docs/maintainer_guide.md`
   - `README.md`
3. Run the required code-day validation gate because a public header changed:
   - `make format`
   - `make lint`
   - `make test`
4. Recheck touched-surface alignment:
   - `rg -n "SPARSE_ERR_BACKEND_CONTRACT|csc_supernodal_dense_kernel|bench_chol_csc|builtin|supernodal" README.md docs/maintainer_guide.md benchmarks/README.md include/sparse_cholesky.h include/sparse_types.h`
   - `git diff -- README.md docs/maintainer_guide.md benchmarks/README.md include/sparse_cholesky.h`

### Day 12 Findings

#### 1. The remaining contradiction was public/header interpretation, not more kernel or benchmark work

After Day 11, the repo already had:

- the backend-aware dense-kernel seam
- family-local error-path proof
- benchmark-side path measurability

The strongest remaining gap was narrower:

- `SPARSE_ERR_BACKEND_CONTRACT` existed publicly
- `bench_chol_csc` exposed the active dense-kernel descriptor publicly
- but the affected public/header and maintainer-facing interpretation surfaces
  did not yet explain how those fit together

Interpretation:

- Day 12 did not need more implementation work
- it needed one bounded cross-surface truthfulness pass

#### 2. The public Cholesky header now states the shipped backend-contract lane directly

`include/sparse_cholesky.h` now makes two things explicit for
`sparse_cholesky_factor_opts(...)`:

- the CSC supernodal lane can surface `SPARSE_ERR_BACKEND_CONTRACT`
- that code is reserved for the bounded Sprint 64 backend-aware dense-kernel
  seam, not for ordinary caller misuse

Interpretation:

- callers now have the real API-local contract at the call site
- the error taxonomy no longer depends on maintainers remembering a sprint note

#### 3. The README and maintainer guide now align on what Sprint 64 actually landed

`README.md` now teaches the bounded backend-aware Cholesky CSC story at the
existing transparent-dispatch adoption point:

- `bench_chol_csc` reports:
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
- the default build reports:
  - `scalar`
  - `supernodal`
  - `builtin`
- `SPARSE_ERR_BACKEND_CONTRACT` is the public error code if that internal
  supernodal dense-kernel seam cannot resolve its required descriptor/callback

`docs/maintainer_guide.md` now owns the narrower policy interpretation:

- Sprint 64 is still a bounded CSC supernodal Cholesky lane, not a general
  backend framework
- `bench_chol_csc` is the maintained benchmark-side proof surface for this
  lane
- `SPARSE_ERR_BACKEND_CONTRACT` should stay narrow and should not be collapsed
  back into `SPARSE_ERR_BADARG`
- the deferred backend queue remains future-facing instead of implied solved

#### 4. The Day 12 batch stayed tightly bounded

The landed Day 12 batch did not widen into:

- `src/`
- benchmark binaries
- tests beyond the required validation gate
- `CMakeLists.txt`
- `Makefile`
- LDL^T / QR / SVD follow-through

Touched surfaces stayed bounded to:

- `include/sparse_cholesky.h`
- `README.md`
- `docs/maintainer_guide.md`

Interpretation:

- Sprint 64 now has one coherent public/maintainer narrative for the backend
  lane without inflating the sprint into broader architecture marketing

#### 5. Validation completed cleanly for the bounded header/docs follow-through

Ran:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

### Day 12 Close

Sprint 64 Day 12 now leaves a smaller final queue:

- the backend-aware Cholesky CSC lane is described coherently across the
  public header, README, benchmark docs, and maintainer policy surface
- the benchmark proof and public error taxonomy no longer rely on sprint-local
  interpretation
- Day 13 can proceed from a cleaner validated-surface story instead of
  lingering docs/header drift

## Day 13

**Objective:** Run the full reviewed validation set from the landed Sprint 64
state, then rerun the targeted backend-aware proof surfaces and capture the
retained benchmark/example signals for closeout.

### Commands Run

1. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm the reviewed CMake parity anchor from the reviewed wrapper path:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Re-run the targeted Sprint 64 proof surface:
   - `./build/test_integration | tail -n 8`
   - `./build/test_chol_csc | tail -n 8`
   - `./build/test_ldlt_csc | tail -n 8`
   - `./build/test_cholesky | tail -n 8`
   - `./build/test_ldlt | tail -n 8`
   - `./build/test_sparse_lu | tail -n 8`
   - `./build/test_qr | tail -n 8`
   - `./build/test_svd | tail -n 14`
   - `./build/example_analysis | tail -n 12`
   - `./build/example_basic_solve | tail -n 12`
   - `./build/example_ldlt | tail -n 12`
   - `./build/example_svd_lowrank | tail -n 12`
   - `./build/bench_refactor | tail -n 12`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1 | tail -n 12`
   - `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1 | tail -n 6`
   - `./build/bench_chol_csc tests/data/suitesparse/bcsstk04.mtx --repeat 1 | tail -n 6`
   - `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1 | tail -n 12`
   - `./build/bench_iterative_reuse | rg "speedup=|CG repeated|GMRES repeated|MINRES repeated" -n`
   - `./build/bench_eigs_reuse | rg "speedup=|Grow-m|Thick-restart|LOBPCG|parity:" -n`

### Day 13 Findings

#### 1. The full Sprint 64 validation baseline passed from the landed state

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

The reviewed anchors stayed exact:

- reviewed CMake parity count: `53`
- Makefile/CMake test-count parity: `53 vs 53`
- full reviewed CMake `ctest`: `53 / 53`
- reviewed CMake total test time: `574.42 sec`

Interpretation:

- Sprint 64 still closes from the strongest maintained local baseline
- the bounded backend-aware Cholesky CSC landing did not drift the reviewed
  parity contract

#### 2. The targeted CSC and direct proof binaries all still pass cleanly

The targeted Sprint 64 rerun set passed:

- `test_integration`: `47 / 47`
- `test_chol_csc`: `144 / 144`
- `test_ldlt_csc`: `96 / 96`
- `test_cholesky`: `21 / 21`
- `test_ldlt`: `84 / 84`
- `test_sparse_lu`: `37 / 37`
- `test_qr`: `72 / 72`
- `test_svd`: `97 / 97`

Interpretation:

- the backend-aware supernodal Cholesky seam did not break the family-local CSC
  proof surface
- adjacent direct/CSC/dense-kernel sentinels stayed stable

#### 3. The retained example signals still match the shipped repeated-run and solver story

Representative retained example outputs:

- `example_analysis`: repeated direct lifecycle residual stayed `4.44e-16`
- `example_basic_solve`: residual stayed `0.00e+00`
- `example_ldlt`: refinement residual stayed `0.000e+00`; `cond_1(K) ~ 26.89`
- `example_svd_lowrank`: sparse low-rank `k=2` kept `22 -> 6` nnz for `3.7x`
  compression

Interpretation:

- Sprint 64 did not disturb the user-facing direct lifecycle examples
- dense/SVD-adjacent example behavior still matches the pre-Day-8 story

#### 4. The retained benchmark proof surface reflects the new backend-aware lane truthfully

Representative retained benchmark outputs:

- `bench_refactor`:
  - `tridiag-200 1.78x`
  - `tridiag-500 1.34x`
  - `bcsstk04 1.34x`
  - `nos4 1.66x`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.63x`
  - residuals `8.24e-16` / `7.06e-16`
- `bench_chol_csc nos4`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
  - `speedup_csc_sn = 0.70x`
  - residuals `7.06e-16`, `5.89e-16`, `5.89e-16`
- `bench_chol_csc bcsstk04`:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
  - `speedup_csc = 1.20x`
  - `speedup_csc_sn = 1.17x`
- `bench_ldlt_csc nos4`:
  - `speedup_csc_native = 1.60x`
  - residuals `5.89e-16` / `5.89e-16`
- `bench_iterative_reuse`:
  - `cg-tridiag-300 1.07x`
  - `gmres-unsym-220 1.03x`
  - `minres-kkt-42 1.00x`
- `bench_eigs_reuse`:
  - `growm-nos4-k5 1.05x`
  - `thick-bcsstk14-k5 1.00x`
  - `lobpcg-diag40-k3 1.04x`
  - `|lambda|max diff = 0.000e+00`

Interpretation:

- the new Sprint 64 benchmark-side path-identification fields stayed stable
- the current default backend-aware lane still reads truthfully as
  `scalar/supernodal/builtin`

#### 5. The reviewed CMake rebuild still carries the existing non-blocking warning note

The reviewed CMake rebuild again emitted the existing `bench_eigs_reuse.c`
double-promotion warnings while rebuilding `bench_eigs_reuse`.

Interpretation:

- this remains a known non-blocking warning seam
- it does not change the Day 13 pass/fail baseline because the full reviewed
  path still completed cleanly and passed all parity gates

### Day 13 Close

Sprint 64 Day 13 now leaves a validated closeout baseline:

- the full required validation gate passed
- the reviewed parity anchors remained exact at `53`
- the targeted CSC/direct/example/benchmark proof surfaces all still pass from
  the landed Sprint 64 state
- Day 14 can now close from a fully validated backend-aware baseline rather
  than from partial benchmark or docs evidence

## Day 14

**Objective:** Convert the validated Sprint 64 branch into one coherent
backend/performance Phase 1 handoff package for Sprint 65 and the remaining
Epic 6 backend work.

### Commands Run

1. Re-read the Day 14 closeout fence and Sprint 64 project-plan scope:
   - `sed -n '484,560p' docs/planning/EPIC_6/SPRINT_64/PLAN.md`
   - `sed -n '153,183p' docs/planning/EPIC_6/PROJECT_PLAN.md`
2. Re-read the validated Day 13 baseline and the bounded Day 8-12 landing
   artifacts:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_64/artifacts/day13-full-validation-sweep.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day8-kernel-integration-batch1.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day10-backend-contract-error-and-fallback-truthfulness-batch.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day11-benchmark-proof-refresh.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day12-docs-and-maintainer-follow-through.md`
3. Write the Day 14 closeout artifact and final working-notes synthesis:
   - `docs/planning/EPIC_6/SPRINT_64/artifacts/day14-closeout-and-handoff.md`
   - `docs/planning/EPIC_6/SPRINT_64/WORKING_NOTES.md`
4. Recheck project-plan consistency:
   - `sed -n '153,183p' docs/planning/EPIC_6/PROJECT_PLAN.md`

### Day 14 Findings

#### 1. Sprint 64 closes with one coherent bounded backend-aware package, not a generic framework rewrite

The landed Sprint 64 package now reduces cleanly to five concrete outcomes:

- ranked hotspot selection fixed to the Cholesky CSC supernodal dense-kernel
  lane first
- internal dense-kernel abstraction landed through a bounded descriptor seam
- narrow backend-contract error taxonomy landed as
  `SPARSE_ERR_BACKEND_CONTRACT`
- benchmark-side proof refreshed with explicit path-identification fields
- public/header/maintainer interpretation aligned to the actual landed lane

Interpretation:

- Sprint 64 delivered the Phase 1 backend abstraction promised in the project
  plan
- it did so without widening into a repo-wide pluggable-backend story

#### 2. The preserved truthfulness fence is now explicit in one place

The Sprint 64 close state keeps the following contract intact:

- the self-contained default build remains authoritative
- the backend-aware path is bounded and optional, not a new global framework
- the first backend-aware lane is local to CSC supernodal Cholesky
- the default dense-kernel descriptor on that lane remains `builtin`
- fallback correctness stays explicit and proved
- `SPARSE_ERR_BACKEND_CONTRACT` remains narrow:
  - caller contract valid
  - internal backend-owned helper or callback contract failed

Interpretation:

- Sprint 65 inherits a real compatibility fence instead of a vague “keep it
  safe” instruction

#### 3. The Day 13 validated baseline is strong enough to hand off directly

Sprint 64 now closes from the Day 13 validated baseline:

- `make format`, `make lint`, `make test`, and `make quality-review-full`
  passed
- reviewed CMake parity stayed exact at `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 574.42 sec`

The strongest retained backend-aware proof signals stayed explicit:

- `bench_chol_csc` reports:
  - `csc_scalar_path=scalar`
  - `csc_supernodal_path=supernodal`
  - `csc_supernodal_dense_kernel=builtin`
- `bench_refactor_csc nos4`: `speedup_refactor = 1.63x`
- `bench_ldlt_csc nos4`: `speedup_csc_native = 1.60x`
- `test_chol_csc`: `144 / 144`
- `test_integration`: `47 / 47`

Interpretation:

- the closeout package rests on measured proof, not just on implementation
  narrative

#### 4. The remaining queue is now ranked for Sprint 65 instead of left as a generic backend backlog

Highest-value carry-forward queue after Sprint 64:

1. LDL^T CSC supernodal backend-aware follow-through
2. bounded shared dense-kernel seam reuse only where it reduces real duplicate
   risk
3. optional build-option or pluggable-kernel widening only if the default-path
   truth surface stays explicit
4. later QR / SVD backend layering only if a later sprint justifies the proof
   cost
5. broader benchmark-governance consolidation and packaging/platform work stay
   outside this immediate lane

Interpretation:

- Sprint 65 can start from a concrete, ranked queue
- Sprint 64 does not hand off an inflated “backend architecture everywhere”
  backlog

#### 5. No Sprint 64 project-plan correction is needed

Rechecked `docs/planning/EPIC_6/PROJECT_PLAN.md` against the landed sprint
state.

Result:

- no correction needed

Interpretation:

- the delivered Sprint 64 package still matches the intended Phase 1 backend
  scope

### Day 14 Close

Sprint 64 Day 14 now closes with:

- one coherent backend-aware Phase 1 package
- one explicit default-path / fallback-path / truthfulness fence
- one validated Day 13 baseline with retained benchmark and regression proof
- one ranked Sprint 65 carry-forward queue instead of a generic backend
  backlog
