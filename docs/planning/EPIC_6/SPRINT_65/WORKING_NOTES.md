# Sprint 65 Working Notes

## Day 1

**Objective:** Turn the Sprint 65 project-plan scope plus the Sprint 64
validated close into a concrete performance-governance starting point by
confirming the preserved reviewed baseline, fixing the strongest live
benchmark/truth/solver hotspot map, and making the sprint workstreams explicit
before design or code work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 65 project-plan source and the new sprint plan:
   - `sed -n '187,216p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the strongest inherited Sprint 64 closeout source:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_64/RETROSPECTIVE.md`
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_64/artifacts/day14-closeout-and-handoff.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the main Sprint 65 docs/build/benchmark/solver/proof hotspots:
   - `wc -l README.md docs/tutorial.md docs/maintainer_guide.md benchmarks/README.md Makefile CMakeLists.txt benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_ldlt_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c src/sparse_dense.c src/sparse_chol_csc_supernodal.c src/sparse_ldlt_csc_supernodal.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c src/sparse_iterative.c src/sparse_eigs.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt_csc.c examples/example_analysis.c`
7. Scan the live performance-governance, benchmark, and backend truth seams:
   - `rg -n "bench_|speedup_|csc_supernodal_dense_kernel|quality-review-full|SPARSE_ERR_BACKEND_CONTRACT" README.md docs/tutorial.md docs/maintainer_guide.md benchmarks/README.md benchmarks src tests include CMakeLists.txt Makefile`

### Day 1 Findings

#### 1. Sprint 65 starts from the Sprint 64 validated close, not from renewed backend-abstraction-first work

Sprint 64 already closed the first bounded Epic 6 backend-aware package:

- the first backend-aware landing target is fixed to the CSC supernodal
  Cholesky dense-kernel lane
- the self-contained default build remains authoritative
- the public error taxonomy now includes `SPARSE_ERR_BACKEND_CONTRACT` for the
  narrow backend-contract failure lane
- `bench_chol_csc` now exposes explicit path-identification fields for the
  shipped backend-aware Cholesky lane

Interpretation:

- Sprint 65 is not reopening the Day 3-10 Sprint 64 backend-lane selection
- Sprint 65 is not another “performance architecture phase 1” sprint
- Sprint 65 should convert the current broad benchmark surface into a smaller,
  more explicit performance-governance surface and then use that sharper map
  to drive targeted solver-efficiency follow-through

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the entire performance-governance sprint

The maintained local truth surfaces remain:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The live reviewed wrapper surface also still shows that the reviewed path
builds:

- `16` benchmark binaries
- `12` example binaries

Interpretation:

- Sprint 65 should inherit the exact Sprint 64 truthfulness wording
- later `*.c` / `*.h` landing days should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial benchmark-governance, regression-reporting, or solver-efficiency
  days should still treat `make quality-review-full` as the stronger default

#### 3. The broad Epic 6 performance-governance claim is already concentrated in benchmark-role, output, canonical-baseline, and solver-follow-through seams

The live repo now shows the strongest current Sprint 65 pressure clustered in:

- maintained truth surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `Makefile`
- build and benchmark-surface inventory:
  - `CMakeLists.txt`
  - benchmark binaries enumerated by the reviewed wrapper surface
- highest-value current benchmark proof surfaces:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_ldlt_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- highest-value likely solver/hotspot follow-through seams:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- strongest proof and adoption surfaces already carrying the story:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `examples/example_analysis.c`

Interpretation:

- Sprint 65 should not pretend every benchmark driver is equally authoritative
- the strongest first cut is benchmark-role audit plus output/taxonomy
  normalization on the maintained proof surfaces
- solver-efficiency work should be selected from what that audit exposes, not
  assumed in advance from generic “optimize hotspots” language

#### 4. Sprint 65 reduces cleanly to seven bounded implementation workstreams

The project-plan scope collapses to:

1. benchmark-role audit
2. output and taxonomy normalization
3. canonical performance surface selection
4. solver-efficiency follow-through
5. local and CI-friendly regression/reporting checks
6. docs and example alignment
7. validation and closeout

Interpretation:

- the Sprint 65 implementation order is already smaller and clearer than a
  generic “performance governance and efficiency” description suggests
- the right Day 1 deliverable is a bounded benchmark/performance
  implementation map with a fixed safety and non-goal fence

#### 5. The strongest likely Sprint 65 touch surfaces are now explicit from the live tree

The highest-value current Sprint 65 surfaces are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `997`
  - `docs/tutorial.md` = `469`
  - `docs/maintainer_guide.md` = `442`
  - `benchmarks/README.md` = `268`
- build and validation truth surfaces:
  - `Makefile` = `881`
  - `CMakeLists.txt` = `397`
- strongest benchmark binaries likely to matter in the first governance pass:
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `406`
  - `benchmarks/bench_ldlt_csc.c` = `516`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- strongest implementation/hotspot seams likely to be influenced by the audit:
  - `src/sparse_dense.c` = `597`
  - `src/sparse_chol_csc_supernodal.c` = `500`
  - `src/sparse_ldlt_csc_supernodal.c` = `392`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
- strongest proof/adoption surfaces likely to matter in Sprint 65:
  - `tests/test_integration.c` = `2367`
  - `tests/test_chol_csc.c` = `4716`
  - `tests/test_ldlt_csc.c` = `3680`
  - `examples/example_analysis.c` = `210`

Interpretation:

- the early Sprint 65 pressure is concentrated enough to support a bounded
  benchmark-governance first landing
- the highest-value later code follow-through is most likely to sit on the
  Cholesky/LDL^T CSC and repeated-run benchmark story, but Day 1 should not
  pre-choose the exact solver batch before the benchmark-role audit

#### 6. Sprint 65 needs an explicit Day 1 non-goal fence before any taxonomy or efficiency design begins

The preserved non-goal fence for Sprint 65 is:

- no fake performance claims beyond reviewed evidence
- no benchmark-governance sprawl disconnected from real proof surfaces
- no broad backend/platform rewrite disguised as efficiency work
- no widening that weakens the self-contained default build or truthfulness
  contract
- no fragile pseudo-regression gates that pretend noisy local timings are
  stable authoritative signals

Interpretation:

- Sprint 65 should improve how the repo explains and uses benchmark evidence
  without turning benchmarks into misleading product claims
- success is a smaller authoritative performance surface plus one bounded
  efficiency follow-through package, not “benchmark everything harder”

### Day 1 Close

Sprint 65 now starts from one explicit performance-governance implementation
baseline:

- the Sprint 64 backend-aware close remains frozen and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 performance-governance claim has already narrowed to
  benchmark-role, output, canonical-baseline, solver-follow-through, and
  regression-reporting seams
- the benchmark/truth/solver hotspots for the first follow-through batch are
  explicit
- the next step is to rank the live benchmark surface precisely before
  designing the normalization and canonical-baseline contract

## Day 2

**Objective:** Freeze the validation and truthfulness baseline that Sprint 65
benchmark-governance, solver-efficiency, and regression-reporting work must
preserve before the sprint moves into the deeper benchmark-role audit.

### Commands Run

1. Confirm branch cleanliness before the Day 2 pass:
   - `git status --short --branch`
2. Re-read the current Sprint 65 notes plus the Day 2 plan slice:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_65/WORKING_NOTES.md`
   - `sed -n '80,170p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the strongest inherited Day 2 shape from Sprint 64:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_64/artifacts/day2-validation-baseline-and-touched-surface-recheck.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Re-read the current quality/truthfulness wording:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode|Windows|macOS|Linux|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows`
7. Reconfirm the targeted Sprint 65 rerun-set presence from the live build tree:
   - `for p in ./build/test_integration ./build/test_chol_csc ./build/test_ldlt_csc ./build/test_cholesky ./build/test_ldlt ./build/test_sparse_lu ./build/test_qr ./build/test_svd ./build/example_analysis ./build/example_basic_solve ./build/example_ldlt ./build/example_svd_lowrank ./build/bench_refactor ./build/bench_refactor_csc ./build/bench_chol_csc ./build/bench_ldlt_csc ./build/bench_eigs_reuse ./build/bench_iterative_reuse; do if [ -e "$p" ]; then echo "present $p"; else echo "missing $p"; fi; done`

### Day 2 Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 65 inherits the same authoritative local validation command as the
Sprint 64 close state:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

This should remain the top-level local trust anchor unless a later Epic 6
implementation sprint proves that the contract itself must change.

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
- stronger default for substantial benchmark-governance, solver-efficiency, or
  regression-reporting work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This remains consistent with the repo’s current Sprint 64 close discipline and
does not need reinterpretation on Sprint 65 Day 2.

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

That means Sprint 65 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start benchmark taxonomy
or solver-efficiency work.

#### 5. The targeted Sprint 65 rerun set is present and aligned to the actual benchmark and solver-risk surface

The confirmed rerun set is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
- adjacent dense-kernel and spectral sentinels:
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

That is already strong enough to support:

- benchmark-role and output/taxonomy normalization on the maintained benchmark
  proof surface
- representative direct/CSC proof after behavior-affecting efficiency edits
- repeated-run benchmark follow-through without pretending every exploratory
  bench binary must become authoritative
- adjacent regression verification so Sprint 65 does not widen unrelated
  backend or platform claims by accident

### Day 2 Close

Sprint 65 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree around maintained benchmark
  proof, direct/CSC proof, and adjacent solver sentinels
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces

## Day 3

**Objective:** Reduce the broad Sprint 65 “performance governance” problem to
one explicit benchmark-role map by classifying the live benchmark binaries into
regression-sensitive, proof, and exploratory lanes from the current repo
state.

### Commands Run

1. Confirm branch cleanliness before the Day 3 audit:
   - `git status --short --branch`
2. Re-read the current Sprint 65 notes plus the Day 3-5 plan slice:
   - `sed -n '1,320p' docs/planning/EPIC_6/SPRINT_65/WORKING_NOTES.md`
   - `sed -n '110,240p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the current benchmark-local docs and inventory the benchmark set:
   - `sed -n '1,260p' benchmarks/README.md`
   - `ls benchmarks/*.c | sed 's#benchmarks/##' | sort`
   - `wc -l benchmarks/*.c`
4. Re-read the current CI/runtime subset and smoke-target positioning:
   - `sed -n '260,330p' Makefile`
   - `rg -n "bench-fast|bench-suitesparse|bench-eigs|Workflow groups|Compile-only gate|csc_supernodal_dense_kernel|speedup_refactor|speedup_csc_native|speedup_auto_vs_ll" benchmarks/README.md Makefile README.md docs/maintainer_guide.md`
5. Re-read the maintainer-facing benchmark policy and representative benchmark headers:
   - `sed -n '320,390p' docs/maintainer_guide.md`
   - `sed -n '560,620p' README.md`
   - `for f in benchmarks/*.c; do echo "=== ${f} ==="; sed -n '1,28p' "$f"; echo; done`
   - `sed -n '1,120p' benchmarks/bench_reorder.c`
   - `sed -n '1,120p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,120p' benchmarks/bench_refactor_csc.c`
   - `sed -n '1,120p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '1,120p' benchmarks/bench_eigs_reuse.c`
   - `sed -n '1,90p' benchmarks/bench_convergence.c`

### Day 3 Findings

#### 1. The live benchmark surface already separates into regression-sensitive, proof, and exploratory lanes, but that split is not written down cleanly in one place

The current benchmark inventory is `16` binaries:

- `bench_amd_qg`
- `bench_bicgstab`
- `bench_chol_csc`
- `bench_colamd`
- `bench_convergence`
- `bench_eigs`
- `bench_eigs_reuse`
- `bench_fillin`
- `bench_iterative_reuse`
- `bench_ldlt_csc`
- `bench_main`
- `bench_refactor`
- `bench_refactor_csc`
- `bench_reorder`
- `bench_scaling`
- `bench_svd`

The live repo already implies three real role classes:

- regression-sensitive runtime sentinels:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_amd_qg`
  - `bench_reorder --skip-factor`
- maintained proof surfaces:
  - `bench_refactor`
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
  - likely `bench_ldlt_csc` as a strong backend-comparison candidate
- exploratory or broad comparison surfaces:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - parts of `bench_reorder`

Interpretation:

- Sprint 65 does not need to invent new benchmark categories
- it needs to formalize the role split the repo is already using implicitly

#### 2. The current strongest maintained proof surfaces are narrower and better defined than the README table alone suggests

The current docs and headers already elevate a bounded set of benchmark-side
proof surfaces:

- repeated-run direct lifecycle:
  - `bench_refactor`
  - `bench_refactor_csc`
- bounded backend-aware Cholesky CSC proof:
  - `bench_chol_csc`
- iterative public-handle reuse:
  - `bench_iterative_reuse`
- eigensolver public-handle reuse:
  - `bench_eigs_reuse`

Those surfaces have the strongest current claim-bearing properties:

- they are tied directly to shipped public workflows or bounded backend lanes
- they have explicit output fields the docs already interpret
- they have adjacent correctness proof homes in tests and examples
- they are already treated more like product-facing proof than like open-ended
  sweep harnesses

Interpretation:

- the canonical Sprint 65 performance surface is more likely to come from this
  smaller set than from the entire benchmark catalog
- Day 4 should start by preserving these proof surfaces and shrinking the
  authoritative set around them

#### 3. The current regression-sensitive runtime subset is real, but it is CI-pragmatic rather than product-claim canonical

`make bench-fast` currently runs:

- `bench_scaling`
- `bench_fillin`
- `bench_colamd`
- `bench_amd_qg`
- `bench_reorder --skip-factor`

That subset exists because it stays within CI runtime limits and still gives a
bounded runtime regression signal. But it is not the same thing as the
strongest benchmark-side product proof surface.

Interpretation:

- Sprint 65 should not collapse “runs in CI quickly” into “canonical
  performance benchmark”
- the regression-sensitive lane and the claim-bearing proof lane need distinct
  names and ownership

#### 4. The strongest current category mismatches are now explicit

The current live mismatches are:

- `bench_reorder` has mixed roles:
  - CI runtime sentinel through `--skip-factor`
  - exploratory threshold and cross-ordering sweep harness otherwise
- `bench_amd_qg` remains in `bench-fast`, but its header is explicitly a
  historical A/B harness for a deleted bitset implementation rather than an
  enduring product benchmark
- `bench_main` still reads like a broad benchmark entry point, but it is too
  multi-mode and wide to be a clean canonical benchmark surface
- `bench_convergence`, `bench_bicgstab`, and `bench_eigs` are valuable
  exploratory comparison tools, but their wider sweep nature makes them poor
  first candidates for normalized regression-sensitive reporting
- `bench_ldlt_csc` is high-value and close to the proof lane, but the docs do
  not yet interpret it as clearly as `bench_chol_csc` or the repeated-run
  proof surfaces

Interpretation:

- Sprint 65 should normalize around stable role ownership first
- only after that should it decide which outputs need machine-readable
  normalization and which should remain broader exploratory tools

#### 5. The first canonical-surface candidate set is already smaller than the full benchmark catalog

The strongest current candidates for the canonical Sprint 65 surface are:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`
- likely `bench_ldlt_csc`

The strongest current candidates for the regression-sensitive runtime lane are:

- `bench_scaling`
- `bench_fillin`
- `bench_colamd`
- `bench_reorder --skip-factor`
- possibly `bench_amd_qg`, but with a weaker long-term claim than the others

The strongest current exploratory/developer-investigation lane is:

- `bench_main`
- `bench_convergence`
- `bench_svd`
- `bench_bicgstab`
- `bench_eigs`
- the non-`--skip-factor` wider modes of `bench_reorder`

Interpretation:

- Day 4 can now rerank a smaller real candidate set instead of treating all
  `16` benchmark binaries as equally authoritative

### Day 3 Close

Sprint 65’s broad benchmark/performance-governance claim has now reduced to a
concrete role map:

- one CI-pragmatic regression-sensitive runtime lane
- one smaller proof-oriented candidate canonical surface
- one exploratory comparison lane
- one explicit mismatch queue where current binaries still mix roles

The next step is to rerank those lanes against the Epic 6 target and separate
must-keep canonical surfaces from later exploratory or maintenance-only
drivers.

## Day 4

**Objective:** Re-rank the Day 3 benchmark-role map into a smaller canonical
candidate set, a first normalization target set, and an explicit deferred
queue before output/taxonomy design begins.

### Commands Run

1. Confirm branch cleanliness before the Day 4 rerank:
   - `git status --short --branch`
2. Re-read the current Sprint 65 notes plus the Day 4-6 plan slice:
   - `sed -n '320,520p' docs/planning/EPIC_6/SPRINT_65/WORKING_NOTES.md`
   - `sed -n '140,260p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the Day 3 benchmark-role artifact:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_65/artifacts/day3-benchmark-role-audit.md`
4. Re-read the direct and backend proof surfaces most likely to define the canonical set:
   - `sed -n '1,220p' benchmarks/bench_refactor.c`
   - `sed -n '1,180p' benchmarks/bench_ldlt_csc.c`
   - `rg -n "bench_refactor|bench_refactor_csc|bench_chol_csc|bench_ldlt_csc|bench_iterative_reuse|bench_eigs_reuse|bench-fast|canonical|proof surface|workflow-specific proof surfaces" README.md benchmarks/README.md docs/maintainer_guide.md Makefile`

### Day 4 Findings

#### 1. The strongest canonical-candidate set is smaller than the Day 3 proof lane

After reranking against the Epic 6 target, the strongest canonical maintained
performance-surface candidates are:

- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

Why these four move to the top:

- each maps to a narrow shipped workflow or bounded backend lane
- each already has structured or naturally normalizable output
- each carries a clearer stable story than the broader exploratory harnesses
- together they cover:
  - repeated-run direct throughput and CSC follow-through
  - bounded backend/path identity
  - iterative repeated-run efficiency
  - eigensolver repeated-run efficiency

Interpretation:

- Sprint 65’s first normalization and canonical-surface work should start
  from these four surfaces, not from the entire proof lane and not from the
  full benchmark catalog

#### 2. `bench_refactor` and `bench_ldlt_csc` should stay in the proof lane, but not in the first canonical normalization batch

Both remain high-value surfaces, but they are weaker first-batch canonical
candidates for different reasons:

- `bench_refactor`
  - high-signal repeated-run direct workflow proof
  - still outputs a human-readable summary rather than a stable CSV schema
  - overlaps materially with the more structured `bench_refactor_csc` direct
    repeated-run story
- `bench_ldlt_csc`
  - strong backend-comparison and dispatch surface
  - still mixes one-shot native/wrapper comparison and analyze-once supernodal
    interpretation
  - current docs do not yet explain its maintained role as tightly as the
    Cholesky CSC and handle-reuse surfaces

Interpretation:

- these should remain benchmark-side proof surfaces
- they should not define Day 5’s first normalization contract
- they are better treated as second-wave or supporting normalization targets

#### 3. The regression-sensitive runtime lane should stay distinct from the canonical maintained performance surface

The Day 3 runtime subset remains:

- `bench_scaling`
- `bench_fillin`
- `bench_colamd`
- `bench_reorder --skip-factor`
- possibly `bench_amd_qg`

This lane still matters, but its ownership is different:

- it supports bounded CI/local drift detection
- it is runtime-pragmatic rather than product-claim canonical
- it should not absorb the same output or interpretation burden as the
  smaller canonical performance set

Interpretation:

- Day 5 should define separate normalized roles for:
  - regression-sensitive runtime sentinels
  - benchmark-side proof surfaces
  - canonical maintained performance surfaces

#### 4. The first Sprint 65 normalization target set is now explicit

The first normalization target set should be:

- binary output:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- documentation and maintainer explanation:
  - `benchmarks/README.md`
  - `README.md`
  - `docs/maintainer_guide.md`

That target set is small enough to support:

- stable category vocabulary
- stable machine-readable output expectations
- explicit path/backend identifiers where relevant
- a believable maintained canonical story

Interpretation:

- Day 5 now has an exact first design surface instead of another generic
  “normalize benchmarks” prompt

#### 5. The deferred benchmark queue is now explicit

Sprint 65 should consciously defer or de-emphasize:

- `bench_main`
- `bench_convergence`
- `bench_svd`
- `bench_bicgstab`
- `bench_eigs`
- broader `bench_reorder` sweep behavior
- `bench_amd_qg` as a long-term canonical signal unless a later pass justifies it

It should also postpone first-batch canonical treatment for:

- `bench_refactor`
- `bench_ldlt_csc`

Interpretation:

- the Sprint 65 surface is now smaller and sharper than the broad Epic 6
  review suggested
- Day 5 can proceed without absorbing every valuable but lower-priority or
  mixed-role benchmark

### Day 4 Close

Sprint 65 now has one explicit first target set before output design starts:

- a four-surface canonical candidate set
- a bounded first normalization batch
- a still-real but explicitly secondary proof queue
- a separate regression-sensitive runtime lane
- a deferred exploratory benchmark queue that Sprint 65 should not absorb

## Day 5

**Objective:** Define the exact benchmark taxonomy vocabulary, normalized
output contract, and ownership split for the first Sprint 65 normalization
batch before any benchmark binaries or docs move.

### Commands Run

1. Confirm branch cleanliness before the Day 5 design pass:
   - `git status --short --branch`
2. Re-read the current Sprint 65 notes plus the Day 5-6 plan slice:
   - `sed -n '520,760p' docs/planning/EPIC_6/SPRINT_65/WORKING_NOTES.md`
   - `sed -n '170,310p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the Day 4 rerank artifact:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_65/artifacts/day4-benchmark-role-rerank-and-canonical-surface-candidates.md`
4. Re-read the exact live output/contract shape for the four selected canonical candidates:
   - `rg -n "Output|CSV|schema|speedup|csc_scalar_path|csc_supernodal_path|csc_supernodal_dense_kernel|workflow|median|repeated-call|wall time|wall_ms" benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c`
   - `sed -n '1,140p' benchmarks/bench_refactor_csc.c`
   - `sed -n '1,120p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,120p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '1,120p' benchmarks/bench_eigs_reuse.c`
5. Re-read the current benchmark-side interpretation surfaces:
   - `rg -n "regression-sensitive|proof|exploratory|canonical|performance surface|benchmark-side proof" benchmarks/README.md README.md docs/maintainer_guide.md`

### Day 5 Findings

#### 1. Sprint 65 needs a three-class maintained vocabulary, not one overloaded “benchmark” label

The right maintained taxonomy is:

1. `regression-sensitive`
   - bounded local/CI runtime sentinel
   - noise tolerance must be high enough for repeatable drift detection
   - not automatically a product-claim benchmark
2. `proof`
   - benchmark-side evidence for a bounded shipped workflow or backend lane
   - may be machine-readable or human-readable
   - should stay narrower than broad comparison or sweep harnesses
3. `exploratory`
   - broader developer comparison, corpus sweep, or historical A/B surface
   - useful, but outside the first authoritative regression/canonical lane

Interpretation:

- Day 6 should select canonical surfaces from within these classes
- Sprint 65 should not overload “proof” and “regression-sensitive” into the
  same meaning

#### 2. The first normalization batch already splits into two output families

The selected target set is not output-uniform today:

- already structured CSV proof surfaces:
  - `bench_refactor_csc`
  - `bench_chol_csc`
- currently human-readable repeated-run proof summaries:
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`

The two CSV surfaces already expose strong normalization anchors:

- `bench_refactor_csc`
  - `matrix`
  - `workflow`
  - `analyze_ms`
  - `refactor_public_ms`
  - `refactor_csc_ms`
  - `solve_public_ms`
  - `solve_csc_ms`
  - `speedup_refactor`
  - `res_public`
  - `res_csc`
- `bench_chol_csc`
  - `matrix`
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
  - factor/solve timing columns
  - speedup columns
  - residual columns

The two handle-reuse surfaces already expose stable conceptual summaries, but
not yet a machine-readable schema:

- one-shot wall time
- reuse wall time
- speedup
- last-run solver summary

Interpretation:

- Day 7-8 should not force a fake identical schema across all four binaries
- the first implementation batch should normalize around a shared contract
  shape while respecting that direct/backend proof and handle-reuse proof are
  different families

#### 3. The normalized output contract should preserve family-local meaning while adding one shared top-level structure

The first-batch normalized output contract should require:

- stable benchmark identity field:
  - benchmark or case label
- stable category field:
  - `proof`
  - later `regression-sensitive`
  - later `exploratory` only if output is intentionally surfaced
- stable workflow or scenario field where applicable:
  - repeated-run direct workflow
  - Cholesky backend path
  - iterative handle case
  - eigensolver handle case
- stable timing fields with `_ms` suffix for machine-readable timing output
- explicit path/backend identity fields where relevant:
  - already mandatory for `bench_chol_csc`
- speedup fields only where the comparison semantics are honest and stable
- residual or result-agreement fields where correctness signal is part of the
  benchmark’s maintained story

Interpretation:

- Day 7 does not need to invent a universal giant schema
- it needs a compact shared contract with family-local extensions

#### 4. Ownership should be split cleanly across binaries, benchmark docs, maintainer policy, and CI/reporting

The correct ownership split is:

- benchmark binary output owns:
  - stable emitted fields
  - stable field names
  - family-local scenario labels
- `benchmarks/README.md` owns:
  - category and usage explanation
  - per-benchmark schema description
  - interpretation notes for path/speedup/residual fields
- `README.md` owns:
  - compact top-level performance-governance story
  - where to look for the maintained proof surfaces
- `docs/maintainer_guide.md` owns:
  - the authoritative category policy
  - which surfaces are canonical candidates versus proof-only versus runtime
    sentinels
- CI/reporting owns:
  - only bounded runtime-sentinel use
  - no broad claim-bearing benchmark governance rewrite unless the local proof
    surface stays maintainable

Interpretation:

- this prevents taxonomy drift between binary output, benchmark docs, repo
  docs, and maintainer policy

#### 5. The preserved compatibility fence is now explicit for the first implementation batch

Sprint 65 should preserve the following rules when normalization begins:

- no misleading benchmark claims
- no unstable pseudo-regression gates
- no output churn without category or interpretive clarity as the reason
- no fake claim that all benchmark binaries are equal members of one canonical
  performance set
- no widening that turns human-readable exploratory tools into CI-authoritative
  signals by accident

Interpretation:

- the first implementation batch should normalize only the bounded selected
  surfaces and their explanation layers

### Day 5 Close

Sprint 65 now has an explicit normalization design before output edits start:

- one three-class benchmark taxonomy
- one shared normalized output contract with family-local extensions
- one clean ownership split across binaries/docs/policy/CI
- one preserved compatibility fence for the first implementation batch

The next step is to convert this design into an exact canonical-surface plan
and touched-file fence before the first edits land.

## Day 6

**Objective:** Convert the Day 5 taxonomy and normalization design into one
exact canonical maintained performance surface, one explicit non-canonical
queue, and one Day 7-10 touched-file fence before implementation starts.

### Commands Run

1. Confirm branch cleanliness before the Day 6 design pass:
   - `git status --short --branch`
2. Re-read the current Sprint 65 notes plus the Day 6-8 plan slice:
   - `sed -n '760,1040p' docs/planning/EPIC_6/SPRINT_65/WORKING_NOTES.md`
   - `sed -n '220,360p' docs/planning/EPIC_6/SPRINT_65/PLAN.md`
3. Re-read the Day 5 normalization-design artifact:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_65/artifacts/day5-output-and-taxonomy-normalization-design.md`
4. Re-measure the selected canonical candidates and their nearest solver/proof seams:
   - `wc -l benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c src/sparse_chol_csc.c src/sparse_chol_csc_supernodal.c src/sparse_ldlt_csc.c src/sparse_ldlt_csc_supernodal.c src/sparse_iterative.c src/sparse_iterative_workspace_internal.c src/sparse_eigs.c src/sparse_eigs_workspace_internal.c tests/test_chol_csc.c tests/test_integration.c tests/test_iterative.c tests/test_eigs.c`
5. Re-read the live output/identity signals and benchmark-policy wording:
   - `rg -n "one-shot=|reuse=|speedup=|speedup_refactor|csc_supernodal_dense_kernel|workflow,|matrix,workflow|category|canonical" benchmarks/bench_refactor_csc.c benchmarks/bench_chol_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c README.md benchmarks/README.md docs/maintainer_guide.md`
   - `sed -n '150,340p' benchmarks/bench_iterative_reuse.c`
   - `sed -n '120,220p' benchmarks/bench_eigs_reuse.c`

### Day 6 Findings

#### 1. The canonical maintained Sprint 65 performance surface should be four benchmark binaries, not the whole proof lane

The exact maintained canonical surface should be:

- `bench_refactor_csc`
- `bench_chol_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

What each canonical surface proves:

- `bench_refactor_csc`
  - repeated-run direct throughput and CSC follow-through on the maintained
    analyze-once / factor-many lane
- `bench_chol_csc`
  - backend/path identity and bounded Cholesky CSC throughput signal on the
    first backend-aware lane
- `bench_iterative_reuse`
  - repeated-run iterative public-handle efficiency signal
- `bench_eigs_reuse`
  - repeated-run eigensolver public-handle efficiency signal

Interpretation:

- Sprint 65’s canonical maintained performance story is now smaller than both
  the full benchmark catalog and the broader proof lane

#### 2. Important proof surfaces remain real, but intentionally non-canonical

The following should remain proof surfaces without becoming first-tier
canonical maintained baselines:

- `bench_refactor`
- `bench_ldlt_csc`

Why:

- `bench_refactor` overlaps substantially with `bench_refactor_csc` while
  still using a human-readable summary contract
- `bench_ldlt_csc` remains valuable, but its one-shot native/wrapper versus
  analyze-once supernodal interpretation is still more complex than the first
  canonical Cholesky CSC lane

Interpretation:

- Day 7-10 should not widen into these surfaces unless the first selected
  efficiency target truly forces it

#### 3. The regression-sensitive runtime lane and exploratory queue are now explicitly non-canonical

The explicit non-canonical sets are:

- regression-sensitive runtime:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - maybe `bench_amd_qg`
- exploratory or later:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

Interpretation:

- Sprint 65 should normalize and document these roles, but it should not make
  them the first maintained performance-baseline batch

#### 4. The Day 7-10 touched-file fence is now explicit

Required first-batch benchmark/doc surfaces:

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`
- `README.md`
- `docs/maintainer_guide.md`

Likely proof surfaces if output or efficiency follow-through requires them:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`

Likely solver/hotspot surfaces if the first efficiency target lands on the
direct repeated-run or backend-aware path:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`

Conditional only if the chosen efficiency target forces them:

- `src/sparse_iterative.c`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_workspace_internal.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`

Interpretation:

- the first implementation batch is now bounded enough to land without
  widening into the full benchmark catalog or every large solver file

#### 5. The first solver-efficiency shortlist now emerges from the benchmark audit instead of assumption

The ranked shortlist is:

1. direct repeated-run CSC/Cholesky follow-through
   - strongest evidence surfaces:
     - `bench_refactor_csc`
     - `bench_chol_csc`
   - likely touched solver seams:
     - `src/sparse_chol_csc.c`
     - `src/sparse_chol_csc_supernodal.c`
   - strongest proof homes:
     - `tests/test_integration.c`
     - `tests/test_chol_csc.c`
2. iterative public-handle reuse follow-through
   - strongest evidence surface:
     - `bench_iterative_reuse`
   - likely touched solver seams:
     - `src/sparse_iterative.c`
     - `src/sparse_iterative_workspace_internal.c`
   - proof home:
     - `tests/test_iterative.c`
3. eigensolver public-handle reuse follow-through
   - strongest evidence surface:
     - `bench_eigs_reuse`
   - likely touched solver seams:
     - `src/sparse_eigs.c`
     - `src/sparse_eigs_workspace_internal.c`
   - proof home:
     - `tests/test_eigs.c`

Interpretation:

- the direct repeated-run CSC/Cholesky lane is the strongest first candidate
  because two canonical benchmark surfaces already point at it and the solver
  proof burden is narrower than the iterative/eigensolver workspace stories

### Day 6 Close

Sprint 65 now has an exact canonical-surface plan before implementation
begins:

- one four-binary maintained canonical performance surface
- one explicit non-canonical proof/runtime/exploratory split
- one exact Day 7-10 touched-file fence
- one ranked solver-efficiency shortlist led by the direct repeated-run
  CSC/Cholesky lane

## Day 7 - Efficiency Design

### Goal

Take the Day 3-6 benchmark-role, normalization, and canonical-surface work and
turn it into one exact first solver-efficiency landing plan with a bounded
implementation fence.

### Actions

1. Re-read the live canonical benchmark outputs against the solver seams they
   actually exercise.
2. Re-rank the first efficiency target by measured maintained-surface evidence,
   touched-surface size, proof burden, and fallback risk.
3. Freeze the first code-batch fence for the benchmark/doc normalization lane
   versus the later solver-efficiency lane.
4. Record the exact required, likely, conditional, and deferred files for the
   first efficiency landing.

### Findings

#### 1. The strongest first efficiency target remains the direct repeated-run CSC/Cholesky lane

The live benchmark evidence still ranks the first efficiency candidate as:

1. direct repeated-run CSC/Cholesky follow-through
2. iterative public-handle reuse follow-through
3. eigensolver public-handle reuse follow-through

The main reason is unchanged but now more concrete:

- `bench_refactor_csc` already measures the public repeated-run direct path
  against a more direct CSC path using stable CSV output
- `bench_chol_csc` already reports linked-list versus CSC scalar versus CSC
  supernodal path identity using stable CSV output
- both surfaces point at the same direct CSC/Cholesky implementation family
- the proof burden remains narrower than the iterative/eigensolver handle
  workspace stories

Interpretation:

- Sprint 65 should not split the first efficiency landing across all four
  canonical benchmark binaries just because all four are canonical

#### 2. The first efficiency batch should stay on the Cholesky CSC side, not broaden to LDL^T or generic dense-kernel work

The first efficiency landing should stay centered on:

- `src/sparse_chol_csc_supernodal.c`

Likely support only if the landed change truly needs it:

- `src/sparse_chol_csc.c`
- `src/sparse_dense.c`

Not first-batch targets:

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_iterative.c`
- `src/sparse_iterative_workspace_internal.c`
- `src/sparse_eigs.c`
- `src/sparse_eigs_workspace_internal.c`

Interpretation:

- the first efficiency landing should reduce duplicate or avoidable overhead on
  the supernodal Cholesky CSC repeated-run lane, not reopen the broader backend
  architecture question

#### 3. The benchmark normalization batch and the solver-efficiency batch are related but not the same deliverable

The first implementation sequence should now be:

1. normalize the canonical maintained benchmark surface
2. land one bounded direct repeated-run CSC/Cholesky efficiency follow-through
3. document the maintained interpretation after the code path is stable

Required benchmark/doc normalization surfaces:

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/README.md`
- `README.md`
- `docs/maintainer_guide.md`

Required first efficiency surfaces:

- `src/sparse_chol_csc_supernodal.c`

Likely proof surfaces for the efficiency landing:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

Conditional only if the implementation proves it necessary:

- `src/sparse_chol_csc.c`
- `src/sparse_dense.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_refactor_csc.c`

Interpretation:

- benchmark-output normalization should not be used as an excuse to widen the
  first solver-efficiency batch
- solver-efficiency work should not rewrite all benchmark outputs at the same
  time

#### 4. The first efficiency proof burden is now bounded enough to land without widening into public API or build-policy work

The strongest proof home remains:

- `tests/test_chol_csc.c`

The bounded public non-regression home remains:

- `tests/test_integration.c`

The maintained runtime evidence remains:

- `bench_refactor_csc`
- `bench_chol_csc`

Explicit non-goals for the first efficiency landing:

- no public API or header widening
- no build-option or CMake/Makefile changes unless the implementation is truly
  blocked
- no LDL^T symmetry batch
- no iterative-handle or eigensolver-handle efficiency batch
- no broad benchmark catalog rewrite
- no CI runtime-lane expansion

### Day 7 Close

Sprint 65 now has:

- one exact first solver-efficiency target chosen from maintained benchmark
  evidence
- one narrow implementation fence centered on the Cholesky CSC supernodal lane
- one clear split between benchmark normalization work and later efficiency work
- one bounded proof and non-goal set for the first code landing

## Day 8 - Output Batch I

### Goal

Land the first bounded taxonomy/output normalization slice on the direct
canonical benchmark surfaces without widening into the later iterative,
eigensolver, or solver-efficiency batches.

### Actions

1. Normalize the direct canonical CSV surfaces first:
   - `bench_refactor_csc`
   - `bench_chol_csc`
2. Keep the first batch limited to stable per-row identity/category/scenario
   fields rather than broad schema churn.
3. Update benchmark-local docs only where the new fields need explanation.
4. Run the required benchmark-governance validation gate and inspect retained
   benchmark output.

### Findings

#### 1. The first normalization slice can stay narrow because the direct canonical surfaces were already structurally close

Before Day 8:

- `bench_refactor_csc` already emitted stable CSV timing and residual fields
- `bench_chol_csc` already emitted stable CSV timing, residual, and path fields
- the main missing governance fields were:
  - stable benchmark identity
  - stable category
  - stable scenario/workflow naming

Interpretation:

- the first batch did not need a broad format rewrite
- it only needed to add the missing governance fields on the direct canonical
  CSV surfaces

#### 2. The landed normalized direct schema now has stable identity, category, and scenario fields

The Day 8 batch now makes both direct canonical CSV surfaces start with:

- `benchmark`
- `category`
- `matrix`
- `scenario`

Landed values are intentionally explicit and stable:

- `bench_refactor_csc`
  - `benchmark = bench_refactor_csc`
  - `category = proof`
  - `scenario = chol_spd | ldlt_kkt`
- `bench_chol_csc`
  - `benchmark = bench_chol_csc`
  - `category = proof`
  - `scenario = chol_backend_compare`

Interpretation:

- the maintained direct benchmark outputs can now be identified and grouped
  without relying on binary name or external context
- the batch stayed compatible with the Day 5 normalization contract:
  - timing fields stayed `_ms`
  - path/backend identity fields stayed explicit where relevant
  - speedup and residual fields stayed unchanged where their meaning was
    already stable

#### 3. The batch stayed inside the Day 7 fence

Touched implementation surfaces:

- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`

Touched benchmark-local docs:

- `benchmarks/README.md`

Not touched:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `README.md`
- `docs/maintainer_guide.md`
- solver implementation files
- proof tests
- CI/build wiring

Interpretation:

- the first output batch stayed on the direct canonical lane only
- the later canonical-surface consolidation and solver-efficiency batches
  remain clearly separated

#### 4. Validation and retained output checks stayed clean

Because benchmark `*.c` files changed, the Day 8 validation gate was:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 686.65 sec`

The retained benchmark-proof spot checks were:

- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained normalized rows are now:

- `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,0.622,0.340,0.205,0.014,0.009,1.66,8.24e-16,7.06e-16`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,100,594,scalar,supernodal,builtin,3.252,1.076,0.597,0.017,0.007,0.006,3.02,5.45,7.06e-16,5.89e-16,5.89e-16`

### Day 8 Close

Sprint 65 now has:

- the first normalized canonical output slice landed on the direct benchmark
  lane
- stable `benchmark` / `category` / `scenario` fields on both direct canonical
  CSV surfaces
- benchmark-local documentation aligned to the new fields
- the iterative/eigensolver canonical normalization batch still cleanly
  deferred to Day 9

## Day 9 - Output Batch II

### Goal

Complete the canonical maintained performance surface by normalizing the
iterative and eigensolver reuse benchmarks and tightening the repo-level
classification story around the smaller Sprint 65 surface.

### Actions

1. Normalize the two remaining canonical benchmark binaries:
   - `bench_iterative_reuse`
   - `bench_eigs_reuse`
2. Keep the batch on output/schema and documentation consolidation only.
3. Align benchmark-local, README, and maintainer-facing ownership around the
   canonical / regression-sensitive / exploratory split.
4. Run the required benchmark-governance validation gate and rerun the two
   touched benchmark proof surfaces.

### Findings

#### 1. The main remaining contradiction after Day 8 was output shape, not benchmark selection

Before Day 9:

- the direct canonical surfaces already emitted stable CSV rows
- the iterative and eigensolver canonical surfaces still emitted
  human-readable prose summaries
- the benchmark selection itself was already settled by Day 6-8

Interpretation:

- the missing Day 9 work was canonical-surface consolidation, not another
  taxonomy rerank

#### 2. The full canonical maintained surface now emits stable machine-readable rows

The remaining two canonical surfaces now also begin with the stable governance
prefix:

- `benchmark`
- `category`
- `matrix`
- `scenario`

Landed interpretations:

- `bench_iterative_reuse`
  - `benchmark = bench_iterative_reuse`
  - `category = proof`
  - `scenario = iter_handle_reuse`
  - adds stable solver/timing/residual/convergence/status columns for:
    - `CG`
    - `GMRES`
    - `MINRES`
- `bench_eigs_reuse`
  - `benchmark = bench_eigs_reuse`
  - `category = proof`
  - `scenario = eigs_handle_reuse`
  - adds stable backend/timing/convergence/agreement columns for:
    - grow-m Lanczos
    - thick-restart Lanczos
    - explicit LOBPCG

Interpretation:

- the compact canonical maintained surface is now explicit in output, not just
  in docs
- the iterative/eigensolver surfaces stayed proof-oriented rather than turning
  into broad solver bake-off CSVs

#### 3. The classification story is now coherent across benchmark-local, top-level, and maintainer surfaces

The landed docs now split the benchmark world into:

- canonical maintained performance surface
- regression-sensitive runtime lane
- exploratory or broader comparison lane

Ownership is now explicit:

- benchmark binaries own emitted fields
- `benchmarks/README.md` owns benchmark-local schema explanation
- `README.md` owns only the compact top-level canonical-surface summary
- `docs/maintainer_guide.md` owns the authoritative category policy

Interpretation:

- Sprint 65 no longer relies on one giant benchmark catalog as the project’s
  “performance story”
- the maintained claim-bearing surface is now smaller and easier to keep honest

#### 4. Validation and retained output checks stayed clean

Because benchmark `*.c` files changed, the Day 9 validation gate was:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed. The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 571.75 sec`

The retained benchmark-proof spot checks were:

- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative retained normalized rows are now:

- `bench_iterative_reuse,proof,cg-tridiag-300,iter_handle_reuse,cg,300,400,112.1140,91.2320,1.23,17,17,5.192e-11,5.192e-11,1,1,OK,OK`
- `bench_iterative_reuse,proof,gmres-unsym-220,iter_handle_reuse,gmres,220,300,64.2980,60.8370,1.06,12,12,7.364e-11,7.364e-11,1,1,OK,OK`
- `bench_iterative_reuse,proof,minres-kkt-42,iter_handle_reuse,minres,42,250,20.7880,30.5190,0.68,39,39,3.870e-11,3.870e-11,1,1,OK,OK`
- `bench_eigs_reuse,proof,growm-nos4-k5,eigs_handle_reuse,lanczos_growm,100,5,40,3.9310,5.6480,0.70,115,115,5,5,4.326e-14,4.326e-14,100,100,0.000e+00,0.000e+00,lanczos_growm,OK,OK`
- `bench_eigs_reuse,proof,thick-bcsstk14-k5,eigs_handle_reuse,lanczos_thick_restart,1806,5,8,175.3030,153.8770,1.14,105,105,5,5,7.864e-14,7.864e-14,40,40,0.000e+00,0.000e+00,lanczos_thick_restart,OK,OK`
- `bench_eigs_reuse,proof,lobpcg-diag40-k3,eigs_handle_reuse,lobpcg,40,3,40,2.6230,2.4450,1.07,45,45,3,3,6.696e-11,6.696e-11,30,30,0.000e+00,0.000e+00,lobpcg,OK,OK`

### Day 9 Close

Sprint 65 now has:

- one fully normalized four-binary canonical maintained performance surface
- one explicit canonical / runtime / exploratory split across outputs and docs
- one compact top-level performance-governance story instead of a broad mixed
  benchmark catalog
- one fixed direct CSC/Cholesky efficiency target carried into Day 10
