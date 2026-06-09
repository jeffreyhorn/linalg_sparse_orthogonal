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
