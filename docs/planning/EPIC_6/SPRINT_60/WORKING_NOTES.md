# Sprint 60 Working Notes

## Day 1

**Objective:** Turn the Sprint 60 project-plan scope plus the merged Epic 5
close state into a concrete Epic 6 starting point by confirming the preserved
reviewed baseline, naming the Sprint 60 audit/contract workstreams explicitly,
and fixing the authoritative productization, architecture, validation, and
platform hotspots before implementation work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 60 project-plan source and the new sprint plan:
   - `sed -n '23,52p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
3. Re-read the strongest inherited Epic 6 review/todo inputs:
   - `sed -n '1,260p' docs/planning/EPIC_6/reviews/review-codex-2026-06-08.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/reviews/todo-codex-2026-06-08.md`
4. Re-read the strongest inherited Epic 5 closeout source:
   - `sed -n '1,220p' docs/planning/EPIC_5/EPIC_5_RETROSPECTIVE.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Measure the main Sprint 60 productization/architecture/quality hotspots:
   - `wc -l README.md docs/tutorial.md docs/maintainer_guide.md Makefile .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h src/sparse_analysis.c src/sparse_graph.c src/sparse_reorder_nd.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c src/sparse_eigs.c src/sparse_iterative.c benchmarks/README.md benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/README.md examples/example_analysis.c examples/example_iterative.c examples/example_eigs.c tests/test_integration.c tests/test_iterative.c tests/test_eigs.c tests/test_chol_csc.c tests/test_ldlt_csc.c`

### Day 1 Findings

#### 1. Sprint 60 starts from a fully closed Epic 5 baseline, not from reopened lifecycle or API design work

The inherited starting state is already explicit and stable:

- Epic 5 closed with:
  - explicit repeated-run direct lifecycle support centered on
    `sparse_analysis_t` / `sparse_factors_t`
  - explicit repeated-run iterative-handle support limited to:
    - `CG`
    - `GMRES`
    - `MINRES`
  - explicit repeated-run eigensolver-handle support limited to:
    - grow-m Lanczos
    - thick-restart Lanczos
    - explicit `LOBPCG`
  - large-source, giant-test, docs, and quality/platform closeout work landed
- the Epic 6 review and todo already assume:
  - no missing-core-algorithm crisis
  - no hidden Epic 5 recovery queue
  - productization and architecture convergence as the new main pressure

Interpretation:

- Sprint 60 is not a feature sprint
- Sprint 60 is not a public lifecycle rescue sprint
- Sprint 60 is the baseline, inventory, target-definition, and contract sprint
  that later Epic 6 implementation work must inherit

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the whole Epic 6 startup phase

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`
- the reviewed Makefile path still includes:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

Interpretation:

- Sprint 60 should preserve the exact `strongest local reviewed baseline`
  phrasing
- later code-touching Epic 6 work should keep reviewed CMake parity and
  Makefile/CMake agreement as the main truthfulness anchors
- the Epic 6 baseline should inherit the current dead-code contract honestly,
  not relitigate it on Day 1

#### 3. The Epic 6 review queue is concentrated in productization, configuration, backend/performance architecture, platform maturity, and assurance depth

The Epic 6 review and todo already reduce the remaining post-Epic-5 problem to
concrete work bands:

- direct-solver usability convergence
- typed configuration replacing process-global tuning where justified
- bounded backend/performance architecture modernization
- stronger benchmark and platform quality governance
- remaining maintainability hotspot reduction
- deeper numerical assurance
- clearer product-facing docs/examples/reference layering

Interpretation:

- Sprint 60 should treat the Epic 6 review as a real inventory source, but
  still refresh the strongest claims against the live repo
- the main Day 1 success criterion is to name the productization and contract
  workstreams clearly before any implementation sprint starts

#### 4. Sprint 60 reduces cleanly to six bounded work classes

The Sprint 60 project-plan items reduce to six bounded work classes:

1. baseline recheck
2. productization gap inventory
3. state-of-the-art target definition
4. configuration/performance surface audit
5. validation/platform contract freeze
6. sprint closeout package

The strongest architectural narrowing is:

- keep the sprint centered on:
  - truthfulness
  - ranking
  - target definition
  - architecture boundaries
  - validation/platform rules
- do not broaden into:
  - implementation churn
  - backend rewrites
  - platform enablement work
  - packaging redesign
  - benchmark-governance code before the target and contract are written

Interpretation:

- Sprint 60 should produce the written rules for Epic 6 rather than trying to
  satisfy them in the same sprint
- the right output shape is a smaller, measured ambiguity surface and a tighter
  implementation runway for Sprint 61+

#### 5. The strongest live Sprint 60 hotspots are now fixed directly from the current repo

The highest-value current Sprint 60 surfaces are now explicit:

- quality/platform contract surfaces:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `315`
  - `Makefile` = `881`
  - `.github/workflows/ci.yml` = `221`
  - `.github/workflows/macos-ci.yml` = `111`
  - `.github/workflows/windows-ci.yml` = `54`
- public product/control surfaces:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- major implementation/architecture surfaces:
  - `src/sparse_analysis.c` = `780`
  - `src/sparse_graph.c` = `801`
  - `src/sparse_reorder_nd.c` = `642`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_iterative.c` = `1985`
- benchmark/example story surfaces:
  - `benchmarks/README.md` = `246`
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
  - `examples/README.md` = `134`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `287`
- strongest assurance/proof surfaces likely to matter later:
  - `tests/test_integration.c` = `1976`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt_csc.c` = `3680`

Interpretation:

- the strongest early Epic 6 pressure is split between:
  - public product workflow wording and ergonomics
  - advanced configuration/control placement
  - residual architecture hotspots
  - performance-governance surfaces
  - hard assurance surfaces
- the repo is already much cleaner than the Epic 5 starting point, but the
  remaining large surfaces still justify an audit-first sprint

#### 6. The inherited compatibility and ambition fence gives Sprint 60 a clean non-goal boundary

The inherited fence remains:

- no fake state-of-the-art claim just because Epic 5 closed well
- no reopening the direct repeated-run lifecycle contract
- no widening iterative/eigensolver public repeated-run support by implication
- no broad backend or platform implementation batch before the architecture and
  validation contracts are frozen
- no broad packaging or CI redesign disguised as “baseline setup”

Interpretation:

- Sprint 60 should define which Epic 6 ambitions are real and which stay out of
  scope
- explicit non-goals are part of the deliverable, not a side note

#### 7. Epic 6 should be treated as a productization and architecture-convergence effort, not as another backlog bucket

The Epic 6 review already gives the key framing:

- the library is a strong single-node sparse linear algebra system
- the remaining distance to “state of the art” is mainly in:
  - usability polish
  - configuration coherence
  - backend/performance architecture
  - release/platform convergence
  - deeper assurance

Interpretation:

- Sprint 60 should translate that broad conclusion into:
  - a ranked live inventory
  - a precise target
  - a precise architecture contract
  - a precise validation/platform contract
- later implementation sprints should then inherit that contract instead of
  improvising toward “state of the art” piecemeal

## Day 1 Close

Sprint 60 now has an explicit starting point:

- preserved reviewed baseline
- inherited closed Epic 5 compatibility fence
- named productization and architecture workstreams
- explicit live hotspot map across docs, headers, implementations, benchmarks,
  examples, and tests
- explicit non-goal fence against premature implementation or inflated product
  claims

That is enough to move to the Day 2 validation and truthfulness-anchor recheck
without reopening Epic 5 decisions or pretending Epic 6 implementation work
can start safely without a written baseline package.

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline, the exact parity and
truthfulness anchors, and the authoritative rerun set that Epic 6
implementation work must preserve before the productization and architecture
audits deepen.

### Commands Run

1. Re-read the Day 2 sprint-plan contract:
   - `sed -n '76,150p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
2. Reconfirm the reviewed CMake count anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the reviewed local wrapper surface:
   - `make -n quality-review-full`
4. Re-read the strongest user-facing quality/truthfulness wording:
   - `sed -n '780,860p' README.md`
5. Re-read the strongest maintainer-policy quality/truthfulness wording:
   - `sed -n '100,220p' docs/maintainer_guide.md`
6. Re-read the exact maintained wrapper semantics in the live `Makefile`:
   - `sed -n '498,660p' Makefile`
7. Confirm the targeted Sprint 60 rerun set is present in `build/`:
   - `ls build | rg '^(test_integration|test_iterative|test_eigs|test_eigs_lobpcg|test_chol_csc|test_ldlt_csc|example_analysis|example_iterative|example_ic_minres|example_eigs|example_svd_lowrank|bench_refactor|bench_refactor_csc|bench_iterative_reuse|bench_eigs_reuse)$'`

### Day 2 Findings

#### 1. The strongest local reviewed baseline remains unchanged and should stay the default Epic 6 trust anchor

The maintained local quality baseline is still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity count anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The wrapper surface is still explicit:

- `quality-review-full`:
  - strongest local reviewed baseline
  - rerun guidance points to:
    - `make quality-review`
    - `make quality-review-cmake`
- `quality-review`:
  - reviewed local quality path
  - includes:
    - `format-check`
    - `lint`
    - `test`
    - `deadcode-check`
- `quality-review-cmake`:
  - reviewed CMake parity path with full suite execution
  - includes:
    - configure
    - clean rebuild
    - `ctest -N`
    - full `ctest`

Interpretation:

- Sprint 60 should keep treating `make quality-review-full` as the strongest
  local reviewed baseline
- reviewed CMake parity remains the main numerical truthfulness anchor
- no new “Epic 6 baseline” command is needed or justified

#### 2. The authoritative code-day gate remains smaller than the full reviewed baseline, but the stronger default for substantial work is still explicit

The mandatory gate for later `*.c` / `*.h` days remains:

- `make format`
- `make lint`
- `make test`

The stronger default for substantial implementation work remains:

- `make quality-review-full`

Interpretation:

- code-touching days can still use the lighter required gate when the change is
  bounded and local
- architecture-sensitive, performance-sensitive, or broad implementation days
  should still bias toward the stronger reviewed baseline
- Sprint 60 should write this split down explicitly so later Epic 6 sprints do
  not drift into ad hoc validation choices

#### 3. The README, maintainer guide, and Makefile still agree on the main quality/platform story

The current maintained quality story remains coherent across the main surfaces:

- `README.md` keeps the compact operator map:
  - dead-code is separate from `lint` and `test`
  - `quality-review-full` is the strongest local reviewed baseline
  - Linux is the enforced full reviewed path
  - macOS dead-code remains staged
  - Windows keeps the reviewed CMake subset enforced while Makefile reviewed
    wrappers and dead-code stay staged
- `docs/maintainer_guide.md` owns the policy interpretation:
  - reviewed baseline meaning
  - dead-code meaning
  - serialized dead-code topology
  - current staged residual dispositions
- `Makefile` owns the exact target semantics:
  - serial `deadcode*` topology
  - reviewed wrapper expansion
  - Makefile/CMake test-count parity check

Interpretation:

- Sprint 60 Day 2 does not expose a new wording contradiction that needs an
  immediate cleanup batch
- the baseline contract is already trustworthy enough to support the deeper
  productization and architecture audit work

#### 4. The targeted Sprint 60 rerun set is present and already maps well onto the later Epic 6 work bands

The confirmed targeted rerun set is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- iterative/eigensolver proof surfaces:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- representative benchmark drivers:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

Interpretation:

- the rerun set already spans the main Epic 6 future themes:
  - direct usability/lifecycle
  - CSC/direct repeated-run behavior
  - iterative/eigensolver product surfaces
  - examples as caller-story proof
  - benchmarks as performance-story proof
- Sprint 60 does not need to invent a new validation inventory; it needs to
  freeze this one cleanly

#### 5. The docs-only versus code-day versus substantial-work validation boundary is now explicit enough to carry through the epic

The current boundary is:

- docs-only days:
  - no automatic `make format` / `make lint` / `make test` requirement
  - use targeted sanity checks against the touched surfaces instead
- bounded `*.c` / `*.h` days:
  - required gate:
    - `make format`
    - `make lint`
    - `make test`
- substantial implementation, architecture, performance, or cross-surface
  code days:
  - stronger default:
    - `make quality-review-full`
  - recheck reviewed CMake parity and representative proof surfaces as needed

Interpretation:

- this split matches the live repo’s actual operating discipline
- Sprint 60 can now treat validation selection as a written contract instead of
  tacit tribal knowledge

## Day 2 Close

Sprint 60 now has an explicit validation and truthfulness baseline:

- strongest local reviewed baseline unchanged
- reviewed CMake parity count fixed at `53`
- code-day gate versus docs-only behavior versus stronger reviewed baseline
  split made explicit
- targeted rerun set fixed from the live `build/` tree
- no immediate contradiction across README, maintainer guide, and Makefile

That is enough to move to the Day 3 productization inventory with the baseline
and validation contract already frozen at the working-note level.

## Day 3

**Objective:** Reduce the broad Epic 6 review into a concrete, user-facing
productization inventory grounded in the current repo, with named usability,
configuration, onboarding, packaging, and benchmark-story gaps that later
Sprint 60 days can rank and turn into a state-of-the-art target.

### Commands Run

1. Re-read the Day 3 sprint-plan contract:
   - `sed -n '113,190p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
2. Re-read the strongest top-level user-facing product surfaces:
   - `sed -n '1,220p' README.md`
   - `sed -n '1,220p' docs/tutorial.md`
   - `sed -n '1,240p' examples/README.md`
   - `sed -n '1,260p' benchmarks/README.md`
3. Re-read the strongest public API/control surfaces:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,280p' include/sparse_iterative.h`
   - `sed -n '1,260p' include/sparse_eigs.h`
4. Audit the current advanced-control and env-var-driven seams:
   - `rg -n "SPARSE_(ND|FM|SUPERNODAL_POSTORDER)|getenv|environment" src include README.md docs/tutorial.md docs/maintainer_guide.md`
5. Audit the current packaging/install/build-shape surfaces:
   - `rg -n "add_library\\(|STATIC|SHARED|pkg-config|find_package|install\\(|EXPORT|VERSION|SOVERSION" CMakeLists.txt sparse.pc.in README.md INSTALL.md docs/maintainer_guide.md`
   - `sed -n '1,220p' INSTALL.md`
6. Re-read the strongest current shipped example and direct repeated-run proof
   surfaces:
   - `sed -n '1,220p' examples/example_analysis.c`
   - `sed -n '1,240p' benchmarks/bench_refactor.c`

### Day 3 Findings

#### 1. The strongest user-facing product gap is still the split direct-solver workflow model

The public story is materially better than it was pre-Epic-5:

- `README.md`, `docs/tutorial.md`, and `include/sparse_analysis.h` all now
  state the explicit repeated-run direct lifecycle clearly
- `examples/example_analysis.c` is a strong shipped reference for the
  analyze-once / factor-many path

But the overall direct-solver surface still reads as two partially different
product models:

- one-shot direct APIs remain the default entry points
- the repeated-run direct lifecycle is explicit and well-documented
- the boundary between the two is conceptually clear, but still carries caller
  discipline cost:
  - one-shot paths remain mutation-oriented
  - the safest usage pattern often still means copying or rebuilding matrices
  - advanced callers still need to understand the matrix-state contract in
    detail to avoid surprises

Impact assessment:

- **user-facing pain:** high
- **architectural leverage:** high
- **risk of misleading product claim:** medium-high
- **implementation cost:** medium

Interpretation:

- this remains the strongest productization gap because it affects the simplest
  “how should I solve my matrix?” story
- it is real Epic 6 product work, not a stretch-goal luxury item

#### 2. Advanced configuration remains too env-var-driven and under-exposed in the typed public surface

The current tree still contains a broad env-var-driven control surface:

- nested-dissection and graph/reordering knobs:
  - `SPARSE_ND_*`
  - `SPARSE_SUPERNODAL_POSTORDER`
- Fiduccia-Mattheyses and refinement knobs:
  - `SPARSE_FM_*`
- additional advanced toggles:
  - `SPARSE_SVD_LOWRANK_OUTER`
  - profiling/debug env vars for graph/reorder/eigensolver-adjacent paths

The problem is not only that these variables exist. It is that the strongest
advanced tuning story still lives:

- in internal `getenv(...)` logic
- in long README prose
- in implementation comments
- and only weakly in typed public call surfaces

Impact assessment:

- **user-facing pain:** high for advanced users, medium for general users
- **architectural leverage:** very high
- **risk of misleading product claim:** high
- **implementation cost:** medium-high

Interpretation:

- this is the strongest configuration-opacity problem in the repo
- it is unquestionably real Epic 6 product work, not a future research goal

#### 3. Onboarding and examples are coherent, but still denser and more policy-heavy than a product-like adoption path should be

The current onboarding surfaces are much more honest than earlier epics:

- `README.md` has a clear workflow-first front door
- `docs/tutorial.md` teaches the public workflow boundaries cleanly
- `examples/README.md` is intentionally scoped

But the live surfaces still ask a new user to absorb a lot at once:

- `README.md` is still very dense and mixes:
  - product overview
  - workflow selection
  - algorithm detail
  - benchmark notes
  - quality/platform contract
  - repository layout
- `docs/tutorial.md` is practical, but still reads more like a broad reference
  pass than a tightly tiered adoption path
- examples still mostly lean on one-shot APIs, which is consistent, but leaves
  the advanced repeated-run stories discoverable only after more reading

Impact assessment:

- **user-facing pain:** medium-high
- **architectural leverage:** medium
- **risk of misleading product claim:** medium
- **implementation cost:** medium

Interpretation:

- the onboarding story is not broken
- but it still reads more like a very capable engineering repo than a polished
  product surface
- this is real Epic 6 product work, though lower priority than direct
  usability and configuration modernization

#### 4. The benchmark story is rich, but still not yet a product-like performance-governance surface

`benchmarks/README.md` is already much cleaner than it used to be:

- workflow groups are explicit
- repeated-run direct, iterative-handle, and eigensolver-handle proof benches
  are all named clearly
- benchmark-local scope is better bounded

But the product gap remains:

- benchmark binaries still read more as individually maintained tools than one
  canonical performance program
- there is still no compact top-level answer to:
  - which results are regression-sensitive
  - which are exploratory characterization only
  - which numbers should anchor future product-performance claims
- `bench_refactor.c` is a strong workflow-proof harness, but it still behaves
  like an expert driver rather than a user-facing performance contract surface

Impact assessment:

- **user-facing pain:** medium
- **architectural leverage:** medium-high
- **risk of misleading product claim:** high
- **implementation cost:** medium

Interpretation:

- this is real Epic 6 product work because “state of the art” claims need a
  stronger performance-story contract
- but it should follow direct usability/configuration decisions rather than
  lead them

#### 5. Packaging and install quality are credible, but still read as developer-install quality more than full product-distribution quality

The current packaging surfaces are real and coherent:

- `INSTALL.md` is substantial and current
- `pkg-config` support exists
- `find_package(Sparse)` support exists
- `install(...)` and exported CMake targets are present

The current build/release shape is still bounded, though:

- `CMakeLists.txt` builds:
  - `add_library(sparse_lu_ortho STATIC ...)`
- the installed file list in `INSTALL.md` is static-library-first
- there is no stronger shared-library / ABI / SOVERSION product story
- Windows still uses the CMake path exclusively

Impact assessment:

- **user-facing pain:** medium
- **architectural leverage:** medium-high
- **risk of misleading product claim:** medium-high
- **implementation cost:** medium-high

Interpretation:

- this is real Epic 6 product work
- but it is not the first user-facing gap to attack
- it belongs behind the baseline, target, and architecture-contract writing
  rather than ahead of it

#### 6. The Day 3 gap list already separates real Epic 6 product work from stretch-goal inflation

The live repo supports a cleaner product/work split than the earlier review
alone could provide:

Real Epic 6 product work:

- direct-solver usability convergence
- typed configuration modernization
- benchmark/performance-governance clarification
- packaging/install maturity improvement
- clearer onboarding and workflow tiering

More likely later or bounded non-goal territory:

- broad vendor-backend parity
- distributed/HPC scope
- universal shared-library/ABI guarantees on every platform immediately
- massive new algorithm-family expansion

Interpretation:

- Day 3 already gives Day 4 and Day 5 a strong base for ranking and target
  definition
- the repo’s remaining productization gaps are now concrete enough to drive a
  real “state of the art for this project” decision instead of vague ambition

## Day 3 Close

Sprint 60 now has a concrete first productization inventory:

- strongest gap:
  - split direct-solver usability model
- strongest architecture/product-control gap:
  - env-var-driven advanced configuration
- strongest adoption-surface gap:
  - dense onboarding and example discovery
- strongest performance-story gap:
  - fragmented benchmark/performance-governance surface
- strongest release/productization gap:
  - static-first packaging and limited ABI/distribution story

That is enough to move to Day 4, where the non-usability gaps can be merged
into a broader ranked Epic 6 map instead of continuing to talk about
productization in generic terms.

## Day 4

**Objective:** Extend the Day 3 productization inventory into the strongest
remaining non-usability gaps, then collapse the full Epic 6 queue into one
ranked map with an explicit must-fix / important / maintainability / non-goal
split that can drive the Day 5 target definition.

### Commands Run

1. Re-read the Day 4 sprint-plan contract:
   - `sed -n '151,230p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
2. Re-read the strongest inherited Epic 6 review/todo findings:
   - `sed -n '1,240p' docs/planning/EPIC_6/reviews/review-codex-2026-06-08.md`
   - `sed -n '1,240p' docs/planning/EPIC_6/reviews/todo-codex-2026-06-08.md`
3. Re-read the Day 3 productization inventory:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day3-productization-gap-inventory-part1.md`
4. Re-rank the largest remaining implementation and proof hotspots:
   - `wc -l src/*.c tests/*.c | sort -nr | head -n 30`
5. Audit the current assurance/performance-governance evidence surface:
   - `rg -n "oracle|differential|property|fuzz|random|stress|regression-sensitive|performance" tests benchmarks README.md docs include src -g '!build'`
6. Re-read the Epic 5 residual queue for relevant carry-forward candidates:
   - `sed -n '1,240p' docs/planning/EPIC_5/EPIC_5_RETROSPECTIVE.md`

### Day 4 Findings

#### 1. The strongest remaining non-usability gap is performance/backend architecture, not missing benchmark binaries

The current repo already has real performance evidence and real high-value
fast paths:

- CSC direct kernels
- repeated-run direct workflows
- iterative/eigensolver reuse paths
- benchmark families for one-shot, repeated-run, and eigensolver workloads

But the strongest non-usability gap is still structural:

- no bounded backend abstraction for dense kernels or future acceleration
- no optional BLAS/LAPACK-style dense-kernel integration seam
- limited shared threading-policy story
- static-library-first build shape
- benchmark evidence is still largely local/manual rather than tied to a
  compact performance-governance layer

Impact assessment:

- **user-facing pain:** medium
- **architectural leverage:** very high
- **risk of misleading product claim:** high
- **implementation cost:** high

Interpretation:

- this is the strongest remaining quality/performance gap after direct
  usability and typed configuration
- it is real Epic 6 work, but it should be bounded and staged rather than
  treated as open-ended backend ambition

#### 2. Packaging/platform maturity is real product work, but still secondary to direct usability, typed configuration, and backend architecture

The current platform/packaging story is honest and significantly improved:

- Linux remains the enforced reviewed source-of-truth path
- macOS and Windows dispositions are explicit
- install/export support is real
- `INSTALL.md` is strong

But there is still a product-maturity gap:

- broader platform reviewed parity remains asymmetric
- dead-code topology remains serialized/staged on non-Linux paths
- Windows still closes on the reviewed CMake subset rather than a broader
  fully symmetric reviewed-wrapper surface
- packaging still reads static-library first with a bounded ABI/distribution
  story

Impact assessment:

- **user-facing pain:** medium
- **architectural leverage:** medium-high
- **risk of misleading product claim:** medium-high
- **implementation cost:** medium-high

Interpretation:

- this is important Epic 6 work
- it is not the first axis to define the Epic 6 target around
- the right Day 4 classification is “important quality/platform gap,” not
  “primary product anchor”

#### 3. Assurance depth is now the strongest quality-confidence gap after product-surface coherence

The current assurance surface is already serious:

- broad unit and integration coverage
- fuzz coverage exists
- stress and random-matrix checks exist in several solver families
- dense-oracle and reference-path comparisons exist in some deep test surfaces

But the assurance story still looks uneven when viewed as a product-quality
claim:

- property, oracle, and differential checks are present, but not yet clearly
  tiered across the hardest workflows
- fuzz remains excluded on Windows
- the biggest lifecycle/CSC/repeated-run paths still rely heavily on
  self-consistency and selected reference comparisons rather than a more
  uniform second-layer assurance strategy
- some of the strongest assurance logic still lives in the largest, hardest
  test files

Impact assessment:

- **user-facing pain:** low-medium directly
- **architectural leverage:** medium-high
- **risk of misleading product claim:** high
- **implementation cost:** medium-high

Interpretation:

- this is real Epic 6 work, not optional polish
- but it belongs after the baseline, target, and architecture contracts are
  written

#### 4. The remaining maintainability hotspots are still concentrated enough to justify further bounded work, but they no longer define the whole epic

The largest remaining hotspots are still explicit:

- tests:
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_etree.c` = `2962`
  - `tests/test_graph.c` = `2900`
  - `tests/test_iterative.c` = `2802`
- implementations:
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_lu_csr.c` = `1666`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_chol_csc.c` = `1532`

Epic 5 already removed the highest-value owned seams, so the remaining
maintainability queue is narrower:

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- later CSC residual cleanup if still justified
- deferred giant-test seams:
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - intentionally retained dense `tests/test_integration.c`
- graph/reorder residuals are now clearly live Epic 6 candidates

Impact assessment:

- **user-facing pain:** low directly
- **architectural leverage:** medium
- **risk of misleading product claim:** low-medium
- **implementation cost:** medium

Interpretation:

- this is bounded maintainability debt, not the main Epic 6 product story
- it still matters because the remaining hard files compete with future product
  work in exactly the graph/reorder and direct/CSC areas Epic 6 will touch

#### 5. The current benchmark/performance-governance story is an important quality/performance gap, but not a core architecture substitute

The benchmark evidence is better than the repo’s old state:

- reusable workflow-specific proof benches exist
- repeated-run direct/iterative/eigensolver paths all have dedicated drivers
- the README and benchmark docs already carry bounded “local evidence, not
  universal claim” caveats

But the current story still lacks:

- a canonical regression-sensitive benchmark tier
- machine-readable conventions tied to governance rather than only output
  convenience
- an explicit small set of stable product-performance baselines

Interpretation:

- this is an important quality/performance gap
- it should complement the backend-architecture work, not replace it

#### 6. The full Epic 6 queue now separates cleanly into four classes

Based on Days 3-4 plus the inherited review/todo and Epic 5 residual queue,
the current Epic 6 queue now separates into:

**Must-fix product gaps**

1. direct-solver usability convergence
2. typed configuration replacing the highest-value env-var-driven controls
3. bounded backend/performance architecture modernization

**Important quality/performance/platform gaps**

4. benchmark/performance-governance consolidation
5. packaging/platform/release-shape convergence
6. stronger second-layer assurance on the hardest workflows

**Bounded maintainability debt**

7. remaining large-source decomposition where the seam is real
8. remaining giant-test refactor where the seam is real
9. residual docs-density/reference layering cleanup

**Explicit non-goal or stretch territory**

10. distributed/HPC scope
11. immediate vendor-backend parity
12. broad universal shared-library/ABI guarantees without staged platform work
13. major new algorithm-family expansion as the defining Epic 6 story

Interpretation:

- the Day 4 result is now concrete enough to drive Day 5
- the queue is no longer “review findings” plus “some deferred work”; it is a
  ranked Epic 6 execution map

## Day 4 Close

Sprint 60 now has a unified ranked Epic 6 gap map:

- top tier:
  - direct usability
  - typed configuration
  - bounded backend/performance architecture
- second tier:
  - benchmark/performance governance
  - packaging/platform maturity
  - stronger assurance depth
- third tier:
  - bounded maintainability and docs-density cleanup
- explicit non-goals:
  - distributed/HPC scope
  - vendor-backend parity
  - broad algorithm-surface expansion as the epic’s center of gravity

That is enough to move to Day 5, where “state of the art for this project”
can now be defined against a ranked live-repo gap map rather than against a
generic aspiration list.

## Day 5

**Objective:** Convert the ranked Epic 6 gap map into an explicit target
definition, non-goal fence, and success scorecard so later implementation
sprints can be judged against one stable productization contract instead of a
vague “state-of-the-art” aspiration.

### Commands Run

1. Re-read the Day 5 sprint-plan contract:
   - `sed -n '189,270p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
2. Re-read the Sprint 60 project-plan section to keep the target aligned with
   the sprint-level epic charter:
   - `sed -n '1,260p' docs/planning/EPIC_6/PROJECT_PLAN.md`
3. Re-read the strongest inherited Epic 6 review/todo framing:
   - `sed -n '1,260p' docs/planning/EPIC_6/reviews/review-codex-2026-06-08.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/reviews/todo-codex-2026-06-08.md`
4. Re-read the Sprint 60 ranked inventory inputs:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day3-productization-gap-inventory-part1.md`
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_60/artifacts/day4-productization-gap-inventory-part2.md`
5. Re-read the Epic 5 close-state contract so the Epic 6 target does not
   silently reopen solved boundaries:
   - `sed -n '1,220p' docs/planning/EPIC_5/EPIC_5_RETROSPECTIVE.md`

### Day 5 Findings

#### 1. “State of the art” for this repo should mean best-in-class single-node product quality, not maximal algorithm or platform scope

The ranked Day 3-4 gap map supports a narrower and more defensible Epic 6
target than a generic “be a top sparse library” claim.

For this project, “state of the art” should mean:

- a coherent single-node sparse linear algebra product surface
- strong direct, iterative, eigensolver, and repeated-run workflow usability
- typed advanced configuration instead of ambient process-global tuning for the
  highest-value controls
- a credible bounded performance/backend architecture with a stronger
  performance-governance story
- a more distribution-ready packaging and platform contract
- stronger second-layer assurance on the hardest workflows

It should not mean:

- infinite algorithm breadth
- distributed/HPC scope
- vendor-tuned parity with specialized external solver stacks
- universal platform guarantees that the current repo cannot measure honestly

Interpretation:

- the Epic 6 target is product-quality convergence, not open-ended scope
  inflation

#### 2. The top Epic 6 product goals now collapse to five primary outcome bands

The current ranked inventory reduces cleanly to five primary outcome bands:

1. direct-solver usability convergence
2. typed advanced configuration and explicit precedence
3. bounded backend/performance architecture modernization
4. benchmark/platform/packaging maturity strong enough to support product
   claims
5. stronger assurance and residual maintainability cleanup where it unlocks the
   first four bands

Interpretation:

- this matches the 10-sprint Epic 6 project plan
- the goals are concrete enough to prioritize later implementation work

#### 3. The direct-solver, iterative-handle, and eigensolver-handle support boundaries remain part of the preserved product contract, not candidate target churn

Epic 5 already fixed the core workflow boundaries:

- repeated direct solves remain the explicit analysis/factors lifecycle
- iterative handles remain bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver handle remains bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

Interpretation:

- Epic 6 should improve usability, configuration, architecture, and proof
  around those boundaries
- it should not silently broaden them as part of the target definition

#### 4. The explicit Epic 6 non-goal fence is now narrow enough to prevent later sprint inflation

The ranked inventory and inherited Epic 5 fence support a real non-goal list:

- no distributed-memory or cluster/HPC scope
- no broad vendor-backend parity target
- no “support every iterative/eigensolver family with repeated-run handles”
  goal
- no major algorithm-family expansion as the center of gravity of the epic
- no fake broad platform closure beyond what reviewed measurement can support
- no broad packaging/ABI promise disconnected from the staged platform story
- no maintainability cleanup justified only by line-count aesthetics

Interpretation:

- these are not secondary caveats
- they are part of the Epic 6 target itself, because they keep the sprint queue
  bounded and technically honest

#### 5. A useful Epic 6 scorecard should be outcome-based rather than feature-count-based

The candidate success scorecard should ask whether Epic 6 leaves the repo with:

- a more coherent direct-solver story that reduces matrix-state surprise
- a typed advanced-configuration story for the highest-value controls
- a real bounded backend/performance seam on selected hot paths
- a clearer canonical performance-governance surface
- a more product-like packaging/platform story with explicit residual limits
- stronger second-layer assurance on the hardest lifecycle/CSC/repeated-run
  workflows
- smaller remaining architecture/test hotspots in the surfaces most affected by
  the epic
- a final caller/maintainer story that stays coherent across API, docs,
  examples, benchmarks, and validation surfaces

Interpretation:

- this scorecard can guide later sprint closeouts better than “did we add more
  features?”
- it is specific enough to evaluate the end of the epic without pretending to
  measure things the project does not promise

## Day 5 Close

Sprint 60 now has an explicit state-of-the-art target:

- best-in-class single-node product quality for this repo’s chosen solver and
  workflow scope
- no distributed/HPC or vendor-backend parity inflation
- no silent reopening of the Epic 5 workflow fence
- a concrete five-band goal structure
- an explicit outcome-based scorecard for later Epic 6 closeout evaluation

That is enough to move to Day 6, where the architectural seams behind those
goals can be audited without ambiguity about what the architecture is trying to
serve.

## Day 6

**Objective:** Map the strongest architecture-sensitive seams that later Epic 6
implementation work must preserve, so configuration, usability, backend, and
platform changes can land against one explicit ownership contract instead of
silently widening product-facing behavior.

### Commands Run

1. Re-read the Day 6 sprint-plan contract:
   - `sed -n '230,320p' docs/planning/EPIC_6/SPRINT_60/PLAN.md`
2. Re-read the Day 5 target-definition output:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day5-state-of-the-art-target-definition.md`
3. Scan the highest-value live product/validation/configuration seams:
   - `rg -n "sparse_analyze|sparse_factor_numeric|sparse_refactor_numeric|sparse_factor_solve|sparse_iter_handle|sparse_solve_.*with_handle|SPARSE_ND_|SPARSE_FM_|SPARSE_SUPERNODAL_POSTORDER|quality-review-full|quality-review-cmake|deadcode-check|bench_iterative_reuse|bench_eigs_reuse|bench_refactor|bench_refactor_csc" README.md docs include src tests benchmarks examples Makefile .github/workflows -g '!build'`
4. Re-read the strongest public direct lifecycle seam:
   - `sed -n '1,260p' include/sparse_analysis.h`
5. Re-read the strongest public iterative-handle seam:
   - `sed -n '200,420p' include/sparse_iterative.h`
6. Re-read the strongest public eigensolver-handle seam:
   - `sed -n '320,620p' include/sparse_eigs.h`
7. Re-read the current build/packaging entry surface:
   - `sed -n '1,240p' CMakeLists.txt`

### Day 6 Findings

#### 1. The direct repeated-run public lifecycle already has a strong ownership seam, and later Epic 6 work must preserve it

The direct repeated-run public model is now centered on one explicit seam:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

The ownership rule is already clear in `include/sparse_analysis.h`:

- symbolic/permutation setup lives in `sparse_analysis_t`
- numeric factor contents live in `sparse_factors_t`
- repeated-run reuse preserves symbolic/permutation setup only
- one-shot LU/Cholesky/LDL^T remain first-class peer entry points

Architecture consequence:

- Epic 6 may improve direct-solver usability and internal uniformity
- Epic 6 should not replace this model with a generic universal direct handle
- later usability work must converge the one-shot story toward this seam rather
  than trying to invent a second repeated-run ownership model

#### 2. The iterative and eigensolver repeated-run seams are intentionally explicit, opaque, and bounded

The iterative-handle seam is:

- `sparse_iter_handle_t`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_iter_handle_prepare_minres(...)`
- `sparse_solve_*_with_handle(...)`
- `sparse_iter_handle_free(...)`

The eigensolver-handle seam is:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_handle_free(...)`

Shared architectural rules across both:

- opaque public handle layout
- explicit prepare / run / free lifecycle
- reuse preserves allocation capacity, not prior numerical state
- support boundary remains intentionally bounded

Architecture consequence:

- later Epic 6 work may improve configuration, observability, and proof around
  these handles
- later Epic 6 work should not silently broaden the public repeated-run family
  set unless the target and contract explicitly reopen that decision

#### 3. The configuration/control architecture is the most fragile high-leverage seam in the repo

The current control placement is split across:

- public option structs:
  - direct analysis/factor options
  - iterative opts
  - eigensolver opts
- compile-time switches:
  - OpenMP
  - mutex support
  - sanitizers
- process-global env vars:
  - `SPARSE_ND_*`
  - `SPARSE_FM_*`
  - `SPARSE_SUPERNODAL_POSTORDER`
  - additional SVD/profile/debug toggles

This is fragile because later Epic 6 sprints could easily widen the product
surface accidentally by:

- moving internal controls straight into public docs without typed ownership
- adding more global env-var behavior instead of consolidating it
- conflating compile-time packaging switches with per-call algorithm control

Architecture consequence:

- later configuration work needs an explicit control-placement rule:
  - public typed option
  - internal typed option
  - compile-time build switch
  - narrow legacy env-var override
- this is the highest-risk seam for product-coherence drift

#### 4. Benchmark surfaces and benchmark-governance surfaces are related, but they are not the same architectural layer

The live benchmark surface already separates into workflow-proof binaries:

- direct lifecycle proof:
  - `bench_refactor`
  - `bench_refactor_csc`
- iterative-handle proof:
  - `bench_iterative_reuse`
- eigensolver-handle proof:
  - `bench_eigs_reuse`

But the architecture seam is:

- proof drivers should continue to prove supported workflows
- governance work should define:
  - which results are canonical
  - which outputs are machine-readable
  - which benches are regression-sensitive

Architecture consequence:

- later Epic 6 performance-governance work should build above the existing
  proof-driver layer, not replace it with a broad benchmark-framework rewrite
- workflow-proof binaries are product-surface evidence; governance is the
  policy layer on top

#### 5. The validation/platform contract is already a real architecture seam, not just documentation

The quality/validation path is concretely structured around:

- `quality-review-full`
- `quality-review`
- `quality-review-cmake`
- `deadcode-check`
- reviewed CMake parity
- platform-specific staged limits in CI and docs

The important architecture rule is:

- Linux reviewed quality remains the main enforced truth surface
- Windows and macOS surfaces are additive, staged, or subset-based according to
  explicit reviewed measurement
- `deadcode-check` is a completeness gate, not a zero-findings gate

Architecture consequence:

- later Epic 6 packaging/platform work must preserve this contract shape unless
  fresh measured evidence justifies changing it
- platform work is not free-form CI churn; it is contract-governed repo
  behavior

#### 6. The build/packaging seam is strong enough to extend, but still narrow enough that careless widening would create product confusion

`CMakeLists.txt` currently fixes several important architecture facts:

- one primary library target:
  - `sparse_lu_ortho`
- current build shape:
  - `STATIC`
- install/export surface exists
- optional build switches are present but bounded
- Windows benchmark/test gating is encoded explicitly in the build graph

Architecture consequence:

- later packaging/backend work must decide carefully what belongs in:
  - build-system target shape
  - public packaging contract
  - optional runtime/backend contract
- changing packaging or backend shape without a frozen architecture contract
  would blur the line between:
  - implementation convenience
  - public product promise

#### 7. The highest-risk Epic 6 seam interactions are now explicit

The strongest seam interactions later sprints must watch are:

1. direct usability changes vs preserved analysis/factors lifecycle ownership
2. typed configuration modernization vs legacy env-var compatibility
3. backend/performance abstraction vs self-contained default build contract
4. benchmark-governance policy vs existing workflow-proof benchmark drivers
5. packaging/platform ambition vs reviewed-measurement truthfulness
6. assurance expansion vs already-large proof surfaces and maintainability debt

Interpretation:

- these are the main places where one good local change could still widen the
  product surface accidentally
- Day 7-10 should convert these seam interactions into explicit contract rules

## Day 6 Close

Sprint 60 now has a concrete architecture seam map:

- direct repeated-run ownership seam fixed around `analysis` / `factors`
- iterative/eigensolver repeated-run seams fixed around opaque prepared handles
- configuration/control placement identified as the highest-risk drift seam
- benchmark proof surfaces separated from later governance policy
- validation/platform and packaging/build entry points identified as real
  product architecture, not just docs

That is enough to move to Day 7, where the env-var-driven and advanced control
surfaces can be audited against this seam map instead of in isolation.

## Day 7 — Configuration & Performance Surface Audit I

### Intent

Take the broad Epic 6 "advanced tuning is too env-var driven" claim and reduce
it to a concrete live-tree control map:

- environment variables
- compile-time switches
- public option structs
- process-global or thread-local behavior

The Day 7 goal was to distinguish real productization work from mere wording
cleanup.

### What I checked

- Day 7 scope/criteria in `PLAN.md`
- live `getenv(...)` call sites across `src/` and `include/`
- public typed option surfaces in:
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
- compile-time and build switches in:
  - `include/sparse_matrix.h`
  - `CMakeLists.txt`
  - `Makefile`
- representative runtime-control implementations in:
  - `src/sparse_analysis.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_svd.c`

### Main findings

#### 1. The typed public control surface is already strong in solver-family front doors

The repo is not missing a public option model in general.

Stable typed controls already exist for:

- direct repeated-run lifecycle:
  - `sparse_analysis_opts_t`
- one-shot direct families:
  - LU
  - Cholesky
  - LDL^T
  - QR
- iterative solvers:
  - CG / MINRES / BiCGSTAB via `sparse_iter_opts_t`
  - GMRES via `sparse_gmres_opts_t`
- eigensolvers:
  - `sparse_eigs_opts_t`
- SVD:
  - `sparse_svd_opts_t`

Interpretation:

- Epic 6 does not need a repo-wide configuration model from scratch
- the real productization gap is concentrated in a narrower set of advanced
  runtime controls

#### 2. The strongest env-var-driven gap is concentrated in ND/FM graph-ordering control

The densest process-global runtime tuning surface is the nested-dissection and
FM refinement stack, not the direct/iterative/eigensolver call layer.

Main algorithm-selection env vars:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_SVD_LOWRANK_OUTER`

Main tuning env vars:

- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Interpretation:

- one large high-impact subsystem still bypasses the typed public option model
- this is the clearest later Epic 6 candidate for typed public or typed
  analysis-local controls

#### 3. Compile-time switches are a smaller, cleaner layer — but not uniformly harmless

Clear build-shape switches:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`
- `SANITIZE`

These read like honest build controls, not accidental runtime policy.

But some compile-time macros are different because they shape runtime behavior
or AUTO routing:

- `SPARSE_NODES_PER_SLAB`
- `SPARSE_DROP_TOL`
- `SPARSE_CSC_THRESHOLD`
- `SPARSE_EIGS_THICK_RESTART_THRESHOLD`
- `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`

Interpretation:

- the build-switch layer is not the main productization problem
- the real concern is public compile-time leakage of performance policy and
  numerical heuristics

#### 4. Process-global control is reinforced by thread-local runtime state

The env-var issue is not only discoverability.

The implementation also uses `_Thread_local` runtime state for:

- ND profiling accumulators
- FM runtime configuration
- forced HEM override

and mutates FM runtime state through:

- `sparse_graph_fm_runtime_get(...)`
- `sparse_graph_fm_runtime_set(...)`

Interpretation:

- some behavior remains scoped to process or thread state rather than to a
  typed call or owned object
- this is a real architecture seam, not a docs-only problem

#### 5. The candidate public/internal split is now clearer

Strongest must-be-public-later candidates:

- ND algorithm-family selection
- FM pass-budget / strategy controls that survive as supported behavior
- optional supernodal postorder behavior if it remains caller-meaningful

Strongest must-be-internal-later candidates:

- `SPARSE_QG_PROFILE`
- `SPARSE_ND_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_*_DEBUG`
- forced-HEM retry plumbing
- FM runtime save/restore internals

Strongest stay-build-time candidates:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`
- `SANITIZE`

#### 6. The Epic 6 review claim was directionally right, but the live-tree version is narrower

The review's broad claim that advanced tuning is too env-var driven is
supported, but Day 7 narrows it:

- the main offender is ND/FM and a few advisory performance toggles
- solver-family front doors are already in better shape than the broad review
  wording implied
- compile-time build switches are not the main coherence problem

### Day 7 close

Sprint 60 now has a concrete first control-surface map:

- typed solver options are already a strong product base
- ND/FM tuning is the strongest remaining env-var-driven gap
- public compile-time thresholds leak runtime AUTO-policy more than ideal
- internal debug/profile surfaces can be separated cleanly from later public
  control work

That is enough to move to Day 8, where the public/internal/build/runtime split
can be tightened into an explicit architecture-contract direction instead of a
general complaint.

## Day 8 — Configuration & Performance Surface Audit II

### Intent

Extend the Day 7 control-surface audit into:

- backend-sensitive direct paths
- iterative/eigensolver workflow-sensitive surfaces
- benchmark driver and README performance-story layers
- packaging/platform surfaces that shape product performance claims

The Day 8 goal was to leave Sprint 60 with one unified configuration and
performance map rather than two separate audit fragments.

### What I checked

- Day 8 scope/criteria in `PLAN.md`
- benchmark taxonomy and workflow-proof claims in:
  - `benchmarks/README.md`
- performance/truthfulness/quality sections in:
  - `README.md`
- build/package/platform surfaces in:
  - `CMakeLists.txt`
  - `Makefile`
  - `INSTALL.md`
  - CI workflow comments
- backend-sensitive API/implementation and proof surfaces via targeted `rg` on:
  - `used_csc_path`
  - `backend_used`
  - `SPARSE_*THRESHOLD`
  - `SPARSE_OPENMP`
  - `SPARSE_MUTEX`
  - `AUTO`

### Main findings

#### 1. Direct backend sensitivity is already visible, but policy ownership is split

The direct side already exposes meaningful typed backend controls and telemetry:

- Cholesky:
  - `sparse_chol_backend_t`
  - `used_csc_path`
- LDL^T:
  - `sparse_ldlt_backend_t`
  - `used_csc_path`
- shared repeated-run direct lifecycle:
  - still routes into CSC-vs-linked-list policy internally

But the actual backend policy is spread across:

1. explicit per-call backend forcing
2. AUTO runtime dispatch
3. public compile-time thresholds like `SPARSE_CSC_THRESHOLD`

Interpretation:

- backend visibility is not the main missing piece
- the real gap is scattered policy ownership

#### 2. Iterative control is comparatively healthy; eigensolvers are more mixed

Iterative repeated-run support is already bounded and explicit:

- handles only for:
  - `CG`
  - `GMRES`
  - `MINRES`

So the iterative Epic 6 performance/control issue is mostly:

- build-shape sensitivity (`SPARSE_OPENMP`)
- proof-surface governance
- residual maintainability

Eigensolvers are more mixed:

- explicit backend selector already exists
- `backend_used` already exposes AUTO's choice
- `used_csc_path_ldlt` already exposes shift-invert LDL^T path selection
- AUTO still depends on compile-time thresholds:
  - `SPARSE_EIGS_THICK_RESTART_THRESHOLD`
  - `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`

Interpretation:

- iterative control is mostly product-bounded already
- eigensolvers still carry the same "public forcing plus compile-time AUTO
  heuristic" pattern seen on the direct side

#### 3. The benchmark drivers are already good workflow-proof binaries

The live benchmark layer is better than a generic "performance story is weak"
summary would imply.

It already separates into coherent groups:

- one-shot compatibility/comparison
- direct repeated-run lifecycle
- iterative public-handle reuse
- eigensolver public-handle reuse

Interpretation:

- later Epic 6 work should preserve these binaries as proof surfaces
- the benchmark gap is not missing drivers
- the gap is governance above the current drivers

#### 4. Performance governance is still fragmented across docs, wrappers, and per-binary conventions

The repo already has strong evidence surfaces, but weaker centralized policy for
how performance claims are made and maintained.

Current governance is distributed across:

- `README.md`
- `benchmarks/README.md`
- `Makefile`
- CI comments
- per-binary output conventions

The still-fragmented policy questions are:

- which benchmark surfaces are canonical
- which outputs should be treated as stable machine-readable formats
- which benches are regression-sensitive
- which runs are claim-bearing versus smoke-only

Interpretation:

- this is real Epic 6 work
- but it is policy-layer work above the existing proof binaries, not a reason
  to replace them

#### 5. Packaging/platform maturity still trails the workflow and quality story

The repo already has:

- install/export support
- pkg-config support
- `find_package(Sparse)` support
- reviewed CMake parity
- explicit Linux/macOS/Windows truthfulness wording

But the product shape still reads more like a strong developer-install surface
than a full release/distribution story:

- primary library target remains `STATIC`
- reviewed platform support remains intentionally asymmetric
- Windows and macOS residual limits are explicit, but still real

Interpretation:

- packaging/platform follow-through belongs in Epic 6
- but it is not the first lever compared with control-plane convergence and
  backend-policy cleanup

#### 6. The top-level performance story is now more coherent than the raw review implied, but still too policy-distributed

The current product story is not missing performance material.

What remains weaker is:

- one compact canonical policy for performance baselines
- one compact canonical policy for benchmark output conventions
- one compact canonical policy for claim-bearing versus smoke-only benchmark
  surfaces

Interpretation:

- the remaining story problem is governance coherence, not lack of evidence

### Unified Day 7-8 map

The repo now breaks cleanly into five configuration/performance bands:

1. healthy public typed-control band
   - direct lifecycle options
   - one-shot direct options
   - iterative options
   - eigensolver options
   - SVD options
   - explicit backend selectors
   - direct/eigensolver path telemetry
2. must-converge configuration band
   - ND/FM strategy and pass-budget controls
3. must-rationalize backend-policy band
   - direct/eigensolver AUTO threshold policy
4. governance/policy band
   - benchmark baselines
   - machine-readable conventions
   - regression-sensitive tiers
   - packaging/platform truthfulness rules
5. residual density/maintainability band
   - docs density
   - residual hotspot decomposition
   - residual giant-test seams

### Ranked future implementation queue

1. typed configuration convergence for ND/FM and adjacent advisory controls
2. backend/AUTO policy rationalization
3. benchmark-governance consolidation
4. packaging/platform/release-shape convergence
5. documentation/performance-story compression after policy stabilizes

### Day 8 close

Sprint 60 now has a unified configuration/performance surface map:

- typed solver controls are already a strong base
- ND/FM process-global tuning is the strongest remaining productization gap
- backend AUTO policy is the second strongest gap
- benchmark drivers are already good workflow-proof surfaces
- the main performance-story weakness is governance above those drivers
- packaging/platform maturity is real Epic 6 work, but not the first
  implementation lever

That is enough to move to Day 9, where the audits can be turned into an
explicit architecture contract instead of remaining as ranked observations.

## Day 9 — Architecture Contract Design

### Intent

Turn the Sprint 60 Day 5-8 target and audit work into an explicit Epic 6
architecture fence for later implementation sprints.

The Day 9 goal was to define:

- what later sprints may widen
- what later sprints must preserve
- how new controls should be placed
- what kinds of apparently-helpful changes are actually off-limits

### What I checked

- Day 9 scope/criteria in `PLAN.md`
- Day 5 target definition
- Day 6 architecture seam audit
- Day 7 configuration/control map
- Day 8 configuration/performance map

### Main contract decisions

#### 1. One-shot and repeated-run workflow boundaries are now explicitly frozen

The Day 9 contract keeps the public workflow fence intact:

- one-shot solver families remain first-class entry points
- repeated-run direct solves remain the explicit:
  - `analysis`
  - `factors`
  lifecycle
- iterative repeated-run handles remain bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver repeated-run handles remain bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`

Interpretation:

- Epic 6 can improve usability and coherence around these paths
- Epic 6 does not silently reopen the Epic 5 solver-boundary decision

#### 2. Control placement now has an explicit four-class rule

The contract fixes four allowed placement classes for future controls:

1. public typed option
2. internal typed policy
3. compile-time build switch
4. legacy compatibility override

Interpretation:

- later Epic 6 work should stop adding controls without an explicit ownership
  class
- this is the main anti-drift rule coming out of the Day 7 audit

#### 3. Public typed options are reserved for supported per-call or per-object behavior

The Day 9 contract narrows what counts as a public option:

- caller-visible supported behavior
- meaningful per-call or per-object scope
- stable enough for product documentation
- reasonable need for different values within one process

This strongly supports later typed ownership for the highest-value ND/FM
controls if they survive as supported behavior.

#### 4. Internal policy and build-switch lanes are intentionally distinct

The contract keeps separate:

- internal typed policy
  - AUTO heuristics
  - implementation routing
  - advisory strategy ownership
- compile-time build switches
  - `SPARSE_OPENMP`
  - `SPARSE_MUTEX`
  - `SANITIZE`

Interpretation:

- Epic 6 should not move build-shape controls into runtime API just for
  symmetry
- Epic 6 should not leave implementation policy leaking through public
  mechanisms by accident

#### 5. Benchmark proof versus governance is now a formal architecture rule

The contract explicitly preserves the workflow-proof binaries as the evidence
layer:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

and defines governance as a separate layer above them:

- canonical benchmark surfaces
- stable output conventions
- regression-sensitive tiers
- claim-bearing versus smoke-only policy

Interpretation:

- later Epic 6 performance work should consolidate policy above the existing
  binaries instead of replacing them casually

#### 6. Validation/platform/packaging truthfulness is now bound to the reviewed baseline

The Day 9 contract fixes a simple rule:

- later platform and packaging ambition must remain subordinate to the reviewed
  truth surface already established in Epic 5

That means:

- `quality-review-full`
- reviewed CMake parity
- current Linux/macOS/Windows disposition language

remain authoritative until fresh measured evidence justifies change.

#### 7. The non-goal fence is now directly attached to the architecture contract

The Day 9 draft makes the non-goal fence operational rather than aspirational:

- no distributed-memory / MPI scope
- no vendor-backend parity as the headline goal
- no broad new solver-family wave as the epic's center of gravity
- no fake platform closure beyond reviewed evidence
- no generic maintainability cleanup unless it materially supports product,
  control, backend, benchmark, packaging, or assurance outcomes

### Immediate implications for later Epic 6 work

The contract now gives a clean near-term order:

1. ND/FM control convergence is the first high-value configuration move
2. backend AUTO policy cleanup is the second high-value move
3. benchmark governance should build on existing proof binaries
4. packaging/platform work should follow rather than lead control coherence

### Day 9 close

Sprint 60 now has a draft Epic 6 architecture contract:

- workflow boundaries are explicit
- control-placement rules are explicit
- benchmark proof versus governance separation is explicit
- validation/platform/packaging truthfulness rules are explicit
- bounded widening and non-goal fences are explicit

That is enough to move to Day 10, where the validation and platform contract
can be frozen against this architecture fence instead of against raw audit
notes.

## Day 10 — Validation and Platform Contract Freeze

### Intent

Freeze the validation, gate-selection, and platform-truthfulness contract that
later Epic 6 implementation sprints must use.

The Day 10 goal was to define:

- the strongest reviewed local baseline
- which gates later work should default to
- when docs-only exceptions are valid
- how Linux/macOS/Windows truthfulness should be described

### What I checked

- Day 10 scope/criteria in `PLAN.md`
- Day 9 architecture contract draft
- live validation and truthfulness surfaces in:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

### Main contract decisions

#### 1. The strongest local reviewed baseline stays `make quality-review-full`

The Day 10 freeze keeps one top local authority:

- `make quality-review-full`

Interpretation:

- later Epic 6 work should treat this as the main local truth surface for
  substantial code and contract changes
- later work should not create a stronger-sounding but unreviewed local path
  and treat it as authoritative

#### 2. The reviewed sub-paths remain distinct and meaningful

The contract preserves the current reviewed path split:

- `quality-review-compile`
- `quality-review`
- `quality-review-cmake-compile`
- `quality-review-cmake`
- `quality-review-full`

Interpretation:

- these are not redundant aliases
- later Epic 6 work may extend them, but should not blur their meanings

#### 3. Gate selection is now explicit

Default code-touching local gate:

- `make format`
- `make lint`
- `make test`

Default stronger gate for substantial architecture/performance/platform work:

- `make quality-review-full`

This especially applies to:

- control-plane work
- backend/AUTO-policy work
- benchmark-governance work
- build/package/platform work
- validation/truthfulness contract work

#### 4. Docs-only exceptions are valid, but narrow

The Day 10 contract keeps docs-only exceptions real, but bounded:

- valid when work is docs-only and does not modify `*.c`, `*.h`, build logic,
  workflows, or executable scripts
- still requires live-tree grounding through targeted diff and truthfulness
  checks

Interpretation:

- docs-only does not mean aspirational or stale claims are acceptable
- build/workflow/script edits are not docs-only just because they are not in
  `src/`

#### 5. Dead-code remains reviewed, serialized, and non-zero-findings by contract

The Day 10 freeze preserves the live dead-code meaning:

- `deadcode-check` is a report-completeness gate
- it is not a zero-findings gate
- it is not automatic deletion authority
- serialized execution remains an intentional operational limit because the
  workflow still shares one build/artifact topology

Interpretation:

- later Epic 6 work must not imply dead-code closure beyond the current truth
  surface

#### 6. Coverage remains supplemental rather than an active reviewed-baseline residual

The current live policy remains:

- `make coverage`
- 80% line-coverage threshold
- Linux supplemental CI signal

Interpretation:

- coverage still matters
- but it is not the main active validation contract residual in Sprint 60

#### 7. Platform truthfulness is now frozen explicitly by lane

Linux remains the enforced reviewed source of truth:

- reviewed Makefile compile-quality path
- reviewed CMake parity path
- dead-code report/check path

Linux supplemental signals remain additive:

- direct runtime path
- `bench-fast`
- TSan
- coverage

macOS remains an enforced but narrower reviewed platform:

- Apple Clang reviewed Makefile compile-quality path
- Apple Clang reviewed CMake parity path
- `wall-check`
- Apple Clang `sanitize`

macOS remains staged or supplemental for:

- dead-code
- Homebrew GCC direct make/test path
- install/pkg-config verification

Windows remains an enforced reviewed CMake subset:

- configure
- build
- `ctest -N`
- full `ctest`

with explicit staged exclusions:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

and explicit staged surfaces:

- reviewed Makefile wrappers
- dead-code flow

#### 8. Platform claims now have one explicit truthfulness rule set

The Day 10 contract fixes a simple policy:

1. enforced, staged, and supplemental surfaces must stay labeled distinctly
2. staged surfaces must not be described as reviewed parity
3. platform-specific exclusions must remain explicit
4. docs and workflows must agree on the same reviewed baseline story

### Day 10 close

Sprint 60 now has a frozen validation and platform contract:

- strongest reviewed local baseline is explicit
- gate-selection policy is explicit
- docs-only exceptions are explicit
- dead-code and coverage meanings are explicit
- Linux/macOS/Windows truthfulness boundaries are explicit

That is enough to move to Day 11, where the Sprint 60 target, audits,
architecture contract, and validation contract can be re-read as one package
for cross-surface reconciliation.
