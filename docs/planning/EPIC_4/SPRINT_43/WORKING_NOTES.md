# Sprint 43 Working Notes

## Day 1

**Objective:** Turn the Sprint 43 project-plan scope plus the Sprint 40
hotspot baseline and the Sprint 41/42 internal-first execution rules into a
concrete graph-decomposition baseline by confirming the preserved reviewed
contracts, naming the Sprint 43 workstreams explicitly, and defining the
authoritative graph-hotspot and test-surface inputs before file extraction
begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 43 plan and the main prerequisite planning artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day4-hotspot-allocation-baseline.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
5. Measure the live graph hotspot and current graph-focused test surfaces:
   - `wc -l src/sparse_graph.c`
   - `rg --files tests | rg 'test_(graph|graph_fm_buckets|reorder_nd|reorder_amd_qg)\\.c$'`
   - `rg -n "SPARSE_PARTITION|partition|separator|coars|match|bisection|spectral|fm_" src/sparse_graph.c | sed -n '1,120p'`

### Day 1 Findings

#### 1. Sprint 43 starts from a preserved Sprint 40/41/42 baseline, not from reviewed-quality repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized
- Sprint 41 already left behind the shared internal arithmetic/allocation seam:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 42 already left behind:
  - internal factor-state scaffolding
  - shared matrix-state guard helpers
  - compatibility-preserving factor-path normalization

Interpretation:

- Sprint 43 is not a quality-baseline sprint
- Sprint 43 is a structural decomposition sprint layered on top of a preserved
  reviewed baseline and the internal-first Epic 4 execution contract

#### 2. `src/sparse_graph.c` is still the dominant source hotspot and the right Phase-1 target

The live hotspot baseline still matches Sprint 40's measured picture:

- `src/sparse_graph.c` = `3555` lines
- the file still visibly spans:
  - graph construction and ownership
  - hierarchy/coarsening
  - coarse bisection
  - FM refinement
  - separator lifting
  - runtime strategy parsing and mode switches

Interpretation:

- Sprint 43 is still correctly aimed at the highest-concentration structural
  hotspot in `src/`
- the graph decomposition queue is real and still bounded enough to begin with
  Phase 1 extraction rather than another audit-only sprint

#### 3. The Sprint 43 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's seven workstreams directly from the plan:

- graph-module boundary design
- graph ownership / construction extraction
- hierarchy / coarsening extraction
- coarse-bisection extraction
- build/include cleanup
- focused graph tests
- validation closeout

Interpretation:

- the front half of the sprint should stay audit/design first:
  - seam inventory
  - boundary design
  - build/include strategy
- implementation should then land through bounded subsystem slices rather than
  a broad "split the whole file" push

#### 4. Sprint 41 and Sprint 42 define the refactor style Sprint 43 should reuse

Sprint 43 inherits two execution rules directly:

- Sprint 41:
  - use shared internal helpers where generic safety logic is needed
  - keep specialized algorithm choreography local when it is not a generic seam
  - validate code changes with the Sprint 40 anchor
- Sprint 42:
  - land structural refactors through private/internal seams first
  - preserve public API compatibility and user-facing semantics
  - add focused regression coverage at the new seam boundaries

Interpretation:

- Sprint 43 should treat graph decomposition as an internal-first subsystem
  refactor, not as permission for algorithm or public-contract churn
- FM refinement and separator lifting are explicit later-phase candidates, not
  Day 1 excuses to broaden the scope

#### 5. The highest-risk Phase-1 graph seams are explicit before code changes begin

The main Day 1 high-risk graph seams are:

- graph construction / teardown ownership
- hierarchy-building and coarse-graph lifecycle
- heavy-edge matching / coarsening
- coarse-level bisection dispatch
- runtime strategy parsing and mode selection

The current graph-focused test surface is already concentrated in:

- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

Interpretation:

- Sprint 43 already has a bounded implementation cluster and a bounded focused
  test cluster
- the current test surface is broad enough to support extraction, but later
  days should still add seam-specific coverage where the new file boundaries
  justify it

#### 6. The Day 1 preserve-not-reopen boundary is clear

Sprint 43 is a graph/ND structural decomposition sprint. Day 1 confirms that
it should not reopen:

- public API redesign
- lifecycle-handle redesign beyond the inherited Sprint 42 seams
- cross-platform contract changes
- dead-code topology changes
- generic benchmark or script cleanup unrelated to graph extraction

Interpretation:

- the correct Sprint 43 shape is:
  - baseline and seam inventory
  - boundary/build design
  - bounded extraction batches
  - focused graph tests
  - validation
- broader graph-orchestration, FM, and separator cleanup remains later-phase
  work

## Day 2

**Objective:** Refresh the internal seam inventory inside `src/sparse_graph.c`
so Sprint 43's extraction order is grounded in the live monolith rather than
only in the project-plan labels, with explicit separation between stable
Phase-1 subsystem seams and later FM/separator-heavy regions.

### Commands Run

1. Re-read the Sprint 43 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
2. Re-read the Sprint 40 hotspot baseline:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day4-hotspot-allocation-baseline.md`
3. Sweep the live graph monolith for major seam markers and high-signal
   helper clusters:
   - `wc -l src/sparse_graph.c`
   - `rg -n "^(static |typedef struct|typedef enum|sparse_err_t |void |idx_t |double )|graph_construct|graph_coars|graph_uncoars|graph_refine_fm|separator|bisection|spectral|match|hierarch|strategy|dispatch|partition_once|sparse_graph_partition|sparse_graph_hierarchy_build|sparse_graph_subgraph" src/sparse_graph.c`
   - `rg -n "SPARSE_PARTITION|partition|separator|coars|match|bisection|spectral|fm_" src/sparse_graph.c | sed -n '1,120p'`
4. Re-read the current graph-internal header surfaces:
   - `sed -n '1,260p' src/sparse_graph_internal.h`
   - `sed -n '1,260p' src/sparse_graph_fm_buckets.h`
5. Reconfirm the current graph-focused test concentration:
   - `rg --files tests | rg 'test_(graph|graph_fm_buckets|reorder_nd|reorder_amd_qg)\\.c$'`

### Day 2 Findings

#### 1. The live monolith reduces cleanly to seven internal seam classes

The current `src/sparse_graph.c` implementation still maps cleanly to these
main regions:

- graph construction / ownership
  - `sparse_graph_from_sparse(...)`
  - `sparse_graph_free(...)`
  - `sparse_graph_subgraph(...)`
- hierarchy and coarse-graph lifecycle
  - `sparse_graph_hierarchy_t`
  - `sparse_graph_hierarchy_build(...)`
  - `sparse_graph_hierarchy_free(...)`
- matching / coarsening
  - `graph_coarsen_with_strategy(...)`
  - `graph_coarsen_heavy_edge_matching(...)`
  - `graph_coarsen_hcc(...)`
  - coarsening strategy parsing / scoring support
- coarse bisection
  - `bisect_brute_force(...)`
  - `bisect_gggp(...)`
  - `graph_build_laplacian(...)`
  - `graph_bisect_coarsest_spectral(...)`
  - `graph_bisect_coarsest(...)`
- FM refinement
  - bucket array implementation
  - annealing / thick-restart / ensemble thread-local strategy support
  - `graph_refine_fm(...)`
- separator lifting / final partition projection
  - `graph_edge_separator_to_vertex_separator(...)`
  - separator-lift strategy parsing and scoring helpers
- top-level orchestration and runtime strategy glue
  - `graph_uncoarsen(...)`
  - `partition_once(...)`
  - `sparse_graph_partition(...)`

Interpretation:

- Sprint 43's planned subsystem labels match the live code structure
- the file is large, but it is not structurally opaque anymore once these
  seven seam classes are named directly

#### 2. The strongest stable Phase-1 extraction seams are graph ownership, hierarchy/coarsening, and coarse bisection

The most stable extraction candidates now are:

- graph ownership / construction
  - already fronted by a small, coherent public-internal helper set
- low dependence on FM or separator-lifting internals
- hierarchy / coarsening
  - coherent around coarse-graph construction, `cmap` ownership, and
    strategy-selected matching
  - depends heavily on graph internals, but not on the FM hot loop
- coarse bisection
  - brute-force, GGGP, and spectral coarse split logic already form a bounded
    algorithm family
  - naturally separable from later uncoarsening and separator lifting

Interpretation:

- these three seams are the right Phase-1 module targets
- they can move without forcing Sprint 43 to solve the entire runtime-strategy
  or separator-lifting topology at the same time

#### 3. FM refinement is real subsystem material, but it is still too entangled for Phase 1

The FM region is not just one function. It currently includes:

- bucket-array implementation
- thread-local pop/annealing/thick-restart/ensemble controls
- gain-noise and perturbation strategy parsing
- `graph_refine_fm(...)` itself

The key complication is that this region is tightly coupled to:

- `graph_uncoarsen(...)`
- finest-level special-case control flow
- ensemble strategy loops
- separator-quality tuning expectations

Interpretation:

- FM refinement is a legitimate later subsystem, not dead weight inside the
  file
- it should remain explicitly deferred from Sprint 43 Phase 1 rather than
  being half-extracted under unstable boundaries

#### 4. Separator lifting is also a distinct subsystem, but it belongs to a later phase with FM

The separator region already has its own bounded vocabulary:

- separator-lift strategies
- per-vertex scoring modes
- weight-selection logic
- `graph_edge_separator_to_vertex_separator(...)`

But it still depends on:

- the partition state output of coarse bisection + FM
- top-level orchestration ordering
- strategy parsing and later-stage partition semantics

Interpretation:

- separator lifting is a valid extraction target
- it belongs in a later graph phase with the FM/toplevel cleanup rather than in
  Sprint 43's first bounded extraction push

#### 5. Cross-cutting runtime strategy state is a real risk cluster and should mostly stay in the orchestration layer for now

The monolith carries several cross-cutting runtime selectors:

- coarsening strategy parsing
- coarse-bisection strategy parsing
- FM annealing/thick-restart/ensemble toggles
- separator-lift strategy parsing
- degenerate-partition retry behavior

Interpretation:

- strategy parsing is not one flat extraction seam
- the safest Sprint 43 rule is:
  - move strategy parsing only when it belongs tightly to an extracted module
  - otherwise leave the wider runtime-glue surface in the remaining orchestration
    layer until Phase 2

#### 6. The current internal header surface is still underpowered for a multi-file graph subsystem

The current graph-internal headers cover:

- core graph representation and hierarchy types:
  - `src/sparse_graph_internal.h`
- FM bucket-array API:
  - `src/sparse_graph_fm_buckets.h`

What is missing for a clean multi-file Phase-1 split is an explicit shared
declaration surface for:

- coarsening-only helpers/types
- coarse-bisection-only helpers/types
- graph-ownership/constructor helpers once they stop living only in the
  monolith

Interpretation:

- Day 3/4 should explicitly design a stronger graph-internal header boundary
- Sprint 43 should not attempt extraction without first deciding which
  declarations stay shared and which remain translation-unit local

#### 7. The extraction order is now clearer than the project-plan text alone

The strongest initial Phase-1 landing order is:

1. graph ownership / construction
2. hierarchy / coarsening
3. coarse bisection
4. only then reconcile the remaining top-level orchestration glue

Explicit defer list for Sprint 43 Phase 1:

- FM refinement extraction
- separator lifting extraction
- broader finest-level strategy cleanup
- deeper top-level partition orchestration simplification

Interpretation:

- Day 3 should design around this order directly
- the sprint remains bounded if it treats the remaining monolith as:
  - FM refinement
  - separator lifting
  - orchestration glue
  rather than trying to erase `src/sparse_graph.c` completely in one pass

## Day 3

**Objective:** Turn the Day 2 seam map into a concrete Phase-1 graph module
layout, with explicit ownership boundaries, shared-header rules, and an
extract-now versus keep-in-monolith split that later code movement can follow
without reopening sprint scope.

### Commands Run

1. Re-read the Sprint 43 Day 3 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
2. Re-read the Day 2 seam-refresh inventory:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day2-monolith-seam-refresh-inventory.md`
3. Re-read the current graph internal header surface:
   - `sed -n '1,320p' src/sparse_graph_internal.h`
4. Sweep the live monolith for the current owners of the main Phase-1 seam
   entry points and parser/glue boundaries:
   - `rg -n "sparse_graph_from_sparse|sparse_graph_subgraph|graph_coarsen_with_strategy|graph_coarsen_heavy_edge_matching|graph_coarsen_hcc|sparse_graph_hierarchy_build|graph_build_laplacian|graph_bisect_coarsest_spectral|graph_bisect_coarsest|graph_uncoarsen|graph_edge_separator_to_vertex_separator|partition_once|sparse_graph_partition|parse_coarsening_strategy|parse_coarsest_bisect_strategy|parse_sep_lift_strategy|graph_refine_fm" src/sparse_graph.c`

### Day 3 Findings

#### 1. The Phase-1 file layout is now concrete rather than implied

The strongest Sprint 43 Phase-1 target layout is:

- `src/sparse_graph_core.c`
  - graph construction / ownership
  - graph subgraph extraction
- `src/sparse_graph_coarsen.c`
  - coarsening strategy parsing owned tightly by the coarsening seam
  - matching / coarsening helpers
  - hierarchy build/free logic
- `src/sparse_graph_bisect.c`
  - brute-force coarse bisection
  - GGGP coarse bisection
  - Laplacian builder
  - spectral coarse bisection
  - coarse-bisection strategy parsing
- remaining `src/sparse_graph.c`
  - FM refinement
  - separator lifting
  - top-level orchestration
  - cross-phase runtime glue still spanning multiple later seams

Interpretation:

- Sprint 43 does not need a large family of tiny files to succeed
- a three-module Phase-1 split is enough to remove the highest-value stable
  seams from the monolith while leaving later-phase regions intact

#### 2. `src/sparse_graph_internal.h` should remain the shared internal contract surface in Phase 1

Day 3's design choice is to keep one shared internal header for the extracted
graph subsystem in Phase 1 rather than introducing several new narrow internal
headers immediately.

That shared header should continue to own:

- `sparse_graph_t`
- `sparse_graph_hierarchy_t`
- graph construction / free / subgraph declarations
- coarsening declarations
- hierarchy declarations
- coarse-bisection declarations
- top-level partition declarations

What should remain translation-unit local for now:

- helper-only local structs such as score/comparator scratch records
- parser support enums that are not required cross-file
- FM-only thread-local controls
- separator-lifting-only scoring helpers

Interpretation:

- Phase 1 can succeed by strengthening one existing shared internal header
  surface rather than creating premature header fragmentation
- Day 4 should therefore focus on declaration placement discipline, not on a
  broad header-tree expansion

#### 3. The ownership boundaries for the extracted modules are now explicit

**`sparse_graph_core.c` owns:**

- raw graph object construction from sparse matrices
- graph object teardown
- induced subgraph creation
- the core representation invariants for `sparse_graph_t`

**`sparse_graph_coarsen.c` owns:**

- one-step coarsening implementations
- strategy-specific matching selection
- hierarchy growth and hierarchy teardown
- `cmap` ownership transitions across coarse levels

**`sparse_graph_bisect.c` owns:**

- all coarse-level partition-initialization logic
- coarse bisection strategy selection
- the Laplacian-builder dependency used by spectral bisection

**remaining `sparse_graph.c` owns in Phase 1:**

- FM refinement state and passes
- separator lifting and per-vertex separator scoring
- uncoarsening orchestration
- retry/fallback orchestration
- final public `sparse_graph_partition(...)` flow

Interpretation:

- each Phase-1 file has a real ownership story rather than being only a line
  split
- later graph phases can continue from the remaining monolith with a smaller,
  clearer orchestration-focused scope

#### 4. Parser ownership should follow the extracted seam only when the parser is tightly local

The parser decision for Phase 1 is intentionally mixed:

- move with extracted module:
  - `parse_coarsening_strategy(...)`
  - `parse_coarsest_bisect_strategy(...)`
- keep in remaining monolith:
  - FM strategy/annealing/thick-restart/ensemble parsers
  - separator-lift strategy parsing

Reason:

- coarsening and coarse-bisection parsers are tightly owned by the extracted
  algorithm families
- FM and separator parsers remain coupled to later-phase orchestration and
  should not be split early just to chase symmetry

Interpretation:

- Sprint 43 avoids fake consistency work
- parser movement follows real ownership rather than superficial naming

#### 5. The explicit keep-in-monolith set is now stable enough to protect scope

After Phase 1 extraction, the remaining `src/sparse_graph.c` should still
contain:

- FM bucket-array implementation and FM thread-local controls
- `graph_refine_fm(...)`
- `graph_uncoarsen(...)`
- separator-lifting helpers and
  `graph_edge_separator_to_vertex_separator(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- degenerate-partition retry logic and other cross-phase orchestration glue

Interpretation:

- the keep-in-monolith list is now explicit, not accidental
- Day 5/6/8 extraction work can proceed without reopening FM or separator scope

#### 6. The new file names are intentionally behavior-oriented, not sprint-history-oriented

Chosen naming direction:

- `sparse_graph_core.c`
- `sparse_graph_coarsen.c`
- `sparse_graph_bisect.c`

Rejected naming directions:

- sprint-day/history-derived names
- overly generic names like `graph_helpers.c`
- premature Phase-2 names that assume FM/separator extraction already happened

Interpretation:

- the file layout is meant to survive the sprint
- the names describe owned behavior rather than temporary extraction history

#### 7. Day 4 now has a narrower and better-defined job

Because Day 3 fixes the Phase-1 file layout and ownership map, Day 4 can stay
narrow:

- decide how `Makefile` / `CMakeLists.txt` absorb the new files
- decide which declarations become shared header surface versus local scope
- document include-order and dependency hygiene
- avoid reopening the basic extraction boundaries

Interpretation:

- Day 3 turns the later build/include work into a wiring problem, not another
  architecture-design pass

## Day 4

**Objective:** Turn the Day 3 file layout into a concrete build/include
strategy so the later extraction batches know exactly how to update source
lists, shared declarations, local-only helpers, and graph-focused test wiring
without reopening the architecture question.

### Commands Run

1. Re-read the Sprint 43 Day 4 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
2. Re-read the Day 3 module-boundary design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day3-graph-module-boundary-design.md`
3. Sweep the current build/test wiring for graph-adjacent sources and tests:
   - `rg -n "sparse_graph\\.c|sparse_graph_internal\\.h|sparse_graph_fm_buckets\\.h|sparse_reorder_nd|test_graph|test_graph_fm_buckets|test_reorder_nd|test_reorder_amd_qg" Makefile CMakeLists.txt tests src | sed -n '1,240p'`
4. Re-read the current maintained build surfaces:
   - `sed -n '1,220p' Makefile`
   - `sed -n '1,220p' CMakeLists.txt`

### Day 4 Findings

#### 1. The build-system change is a controlled source-list expansion, not a new build-model problem

Both maintained build surfaces already treat the library as an explicit source
list:

- `Makefile`
  - `LIB_SRCS = ...`
- `CMakeLists.txt`
  - `add_library(sparse_lu_ortho STATIC ...)`

Day 4 implication:

- Sprint 43 does not need generator logic, globbing changes, or a new graph
  sub-build model
- the extraction batches should simply:
  - add `src/sparse_graph_core.c`
  - add `src/sparse_graph_coarsen.c`
  - add `src/sparse_graph_bisect.c`
  - keep the remaining `src/sparse_graph.c`

Interpretation:

- build wiring for Phase 1 is a straightforward explicit-list update
- this stays aligned with the repo's existing truthfulness model for both
  `Makefile` and CMake

#### 2. `src/sparse_graph_internal.h` should remain the single shared graph contract surface for Phase 1

The current repo already routes graph and ND users through one main internal
header:

- `src/sparse_graph_internal.h`

It is consumed by:

- `src/sparse_reorder_nd.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- the graph monolith itself

Day 4 decision:

- keep `src/sparse_graph_internal.h` as the shared internal graph surface in
  Phase 1
- expand it only where extracted modules need cross-file declarations
- avoid creating several new narrow internal headers unless a later batch
  exposes a real need

Interpretation:

- the extracted files should share one authoritative graph contract surface
- Sprint 43 avoids premature header-tree fragmentation

#### 3. FM bucket support stays isolated in its own header and remains Phase-1 local to the remaining monolith

The repo already has a narrow FM support header:

- `src/sparse_graph_fm_buckets.h`

Current users:

- `src/sparse_graph.c`
- `tests/test_graph_fm_buckets.c`

Day 4 decision:

- keep `src/sparse_graph_fm_buckets.h` separate
- do not fold FM bucket declarations into `src/sparse_graph_internal.h`
- do not make extracted Phase-1 files depend on FM bucket internals

Interpretation:

- FM bucket support remains explicitly part of the later-phase FM seam
- this protects the Phase-1 extraction batches from accidental FM coupling

#### 4. Declaration placement should follow a simple shared-vs-local rule

Phase-1 shared declarations belong in `src/sparse_graph_internal.h` when they
are required by more than one of:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- remaining `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- graph-focused tests that already consume internal graph APIs

Phase-1 declarations should stay translation-unit local when they are:

- helper-only comparators or tiny support structs
- parser enums or support helpers used by just one implementation unit
- FM-only thread-local state
- separator-lift-only scoring helpers
- one-off local scoring/shuffle helpers with no stable external consumer

Interpretation:

- Day 5-9 code movement should add shared declarations only when the extracted
  seam genuinely needs them
- the default remains local scope, not shared scope

#### 5. Include-order risk is real around ND and graph tests, but the dependency graph is still simple

The current high-signal dependency edges are:

- `src/sparse_reorder_nd.c` -> `src/sparse_graph_internal.h`
- `tests/test_graph.c` -> `src/sparse_graph_internal.h`
- `tests/test_reorder_nd.c` -> `src/sparse_graph_internal.h`
- `tests/test_graph_fm_buckets.c` -> `src/sparse_graph_fm_buckets.h`

Day 4 implication:

- extracted Phase-1 files should include `src/sparse_graph_internal.h`
- ND and graph tests should continue to consume the same shared graph-internal
  header rather than being retargeted to per-file private declarations
- FM bucket tests should remain pointed at the narrow FM header only

Interpretation:

- include-order risk is mostly about keeping the shared graph contract surface
  coherent, not about deep cyclic dependency problems

#### 6. The graph-focused test wiring does not need structural change in Phase 1

The maintained graph-focused tests already live as ordinary test executables in
both build systems:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

Day 4 decision:

- no new test target families are needed
- no build-system specialization is needed for extracted graph modules
- later seam tests should simply extend the existing graph-focused binaries

Interpretation:

- the testing follow-through in later Sprint 43 days is a coverage problem, not
  a build-topology problem

#### 7. The extraction batches now have concrete wiring rules

When Phase-1 code movement starts, the rule set is:

1. update both library source lists in lockstep:
   - `Makefile`
   - `CMakeLists.txt`
2. keep `src/sparse_graph_internal.h` as the shared contract surface
3. keep `src/sparse_graph_fm_buckets.h` isolated to the FM seam
4. add declarations to the shared header only when they have multiple stable
   consumers
5. keep helper-only and phase-local support logic translation-unit local
6. do not change graph test target topology just because the implementation
   file count increases

Interpretation:

- Day 5 and later extraction work now has a concrete wiring checklist
- the remaining uncertainty is implementation detail, not build/include policy

## Day 5

**Objective:** Land the first real Phase-1 extraction batch by moving the
graph ownership / construction seam out of `src/sparse_graph.c` into its own
translation unit, wiring the new file into both maintained build systems, and
validating that the split preserves the current graph/ND behavior.

### Commands Run

1. Re-read the Sprint 43 Day 3/4 design inputs and inspect the current seam:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day3-graph-module-boundary-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day4-build-and-include-strategy-design.md`
   - `sed -n '1,360p' src/sparse_graph.c`
2. Land the extraction batch:
   - move `sparse_graph_from_sparse(...)`
   - move `sparse_graph_free(...)`
   - move `sparse_graph_subgraph(...)`
   - add `src/sparse_graph_core.c`
   - update `Makefile`
   - update `CMakeLists.txt`
3. Run the required full validation gate for `*.c` changes:
   - `make format`
   - `make lint`
   - `make test`
4. Capture the resulting diff and status:
   - `git diff --stat`
   - `git status --short`

### Day 5 Findings

#### 1. The first extraction seam was exactly as stable as the Day 2/3 inventory predicted

The moved batch was:

- `sparse_graph_from_sparse(...)`
- `sparse_graph_free(...)`
- `sparse_graph_subgraph(...)`

These functions form a coherent ownership / construction slice because they
own:

- graph-object initialization
- adjacency build-up from `SparseMatrix`
- partial-construction cleanup
- graph-object teardown
- the current subgraph helper stub

Interpretation:

- this was the right first batch because it is structurally important but
  algorithmically low-risk
- the extraction reduces the monolith without forcing premature movement of
  coarsening, FM, separator, or orchestration code

#### 2. `src/sparse_graph_core.c` is now the Phase-1 home for graph object lifecycle

Day 5 created:

- `src/sparse_graph_core.c`

The new file now owns the graph lifecycle seam, while the remaining
`src/sparse_graph.c` begins with the heavier algorithmic regions.

Interpretation:

- the decomposition is now real, not just planned
- later hierarchy/coarsening extraction can build on a cleaner residual
  monolith that no longer mixes object-lifecycle code with the core
  partitioning pipeline

#### 3. The remaining monolith is now less entangled with matrix-construction details

After the extraction:

- `src/sparse_graph.c` no longer needs `sparse_matrix_internal.h`
- graph construction / teardown no longer sits at the top of the main graph
  algorithm file

Interpretation:

- this is a good Phase-1 signal that seam ownership improved materially, not
  just cosmetically
- the residual monolith is now more purely about graph algorithms and
  strategy flow

#### 4. The Day 4 build/include strategy held exactly as designed

The build wiring change stayed bounded:

- `Makefile` gained `src/sparse_graph_core.c`
- `CMakeLists.txt` gained `src/sparse_graph_core.c`
- `src/sparse_graph.c` remained in both source lists
- no public headers changed
- `src/sparse_graph_internal.h` remained the shared internal contract surface

Interpretation:

- the Phase-1 extraction did not need build-system redesign
- the current explicit-source ownership model is sufficient for the next
  extraction batches

#### 5. Day 5 preserved the intended defer boundary

The batch explicitly did **not** move:

- hierarchy lifecycle
- heavy-edge matching / HCC coarsening
- coarse bisection
- FM refinement
- separator lifting
- top-level orchestration

Interpretation:

- Day 5 stayed within the Sprint 43 Phase-1 plan instead of drifting into a
  broad graph rewrite
- Day 6 can now focus cleanly on the hierarchy/coarsening seam without
  reopening the ownership batch

#### 6. Full validation passed after the split

Because `*.c` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- the extracted graph-core module is wired correctly into both maintained
  build surfaces
- the current graph/ND regression surface still holds after the split

## Day 6

**Objective:** Land the first hierarchy/coarsening extraction batch by moving
the multilevel coarsening core, hierarchy lifecycle, and the small internal
HEM/HCC strategy seam into a dedicated implementation unit while preserving
the remaining FM, separator-lifting, and orchestration boundaries.

### Commands Run

1. Re-read the Sprint 43 plan and the Day 3 boundary design, then inspect the
   live coarsening/hierarchy seam:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day3-graph-module-boundary-design.md`
   - `sed -n '400,1065p' src/sparse_graph.c`
   - `sed -n '1,420p' src/sparse_graph_internal.h`
2. Land the extraction batch:
   - add `src/sparse_graph_coarsen.c`
   - move coarsening strategy ownership
   - move heavy-edge/HCC coarsening core
   - move `sparse_graph_hierarchy_build(...)`
   - move `sparse_graph_hierarchy_free(...)`
   - update `src/sparse_graph_internal.h`
   - update `Makefile`
   - update `CMakeLists.txt`
   - reconnect the sep=0 retry path through explicit coarsening helpers
3. Run the required full validation gate:
   - `make format`
   - `make lint`
   - `make test`
4. Capture resulting status and diff shape:
   - `git diff --stat`
   - `git status --short`

### Day 6 Findings

#### 1. The hierarchy/coarsening seam was stable enough to extract as one real subsystem file

Day 6 moved the bounded hierarchy/coarsening family into:

- `src/sparse_graph_coarsen.c`

The moved ownership cluster was:

- coarsening strategy interpretation
- HEM-override control
- heavy-edge / HCC matching core
- coarse-graph construction and dedup
- hierarchy build / free lifecycle

Interpretation:

- this was the right Day 6 batch because the moved code already behaved like a
  real subsystem rather than a random set of helpers
- the batch reduced monolithic concentration materially without requiring FM or
  separator movement

#### 2. The small strategy seam is now explicit instead of being shared through raw file-local state

The remaining `sparse_graph_partition(...)` retry path still needs to know:

- what coarsening strategy is currently active
- how to force temporary HEM fallback for the current thread

Day 6 normalized that through a small internal seam:

- `sparse_graph_coarsening_strategy_current(...)`
- `sparse_graph_force_hem_override_begin(...)`
- `sparse_graph_force_hem_override_end(...)`

Interpretation:

- this is better subsystem ownership than leaving the retry path coupled to a
  raw thread-local defined in the same translation unit
- the seam stays small enough that it does not blur Phase-1 boundaries

#### 3. The remaining monolith is now more focused on the still-deferred graph phases

After the extraction, `src/sparse_graph.c` now retains primarily:

- coarse bisection
- FM refinement
- separator lifting
- top-level uncoarsening / partition orchestration

Interpretation:

- Day 6 successfully peeled away the hierarchy/coarsening layer
- the residual file is now closer to the Day 7/Day 8 and Day 9 target seams
  instead of carrying all graph phases together

#### 4. The build/include strategy still held cleanly under the second file split

The bounded wiring changes were:

- `Makefile` gained `src/sparse_graph_coarsen.c`
- `CMakeLists.txt` gained `src/sparse_graph_coarsen.c`
- `src/sparse_graph_internal.h` gained only the shared declarations the new
  seam actually needed

Interpretation:

- the Phase-1 extraction still does not need build-system redesign
- the current shared-header rule is sufficient for another file split

#### 5. One moved analyser suppression and one moved include dependency had to be re-landed explicitly

Two small extraction follow-ups surfaced during validation:

- `src/sparse_graph.c` still needed `<math.h>` because the FM/annealing path
  remained there
- the moved duplicate-bucket fill path in `src/sparse_graph_coarsen.c` needed
  the same targeted clang-analyzer suppression the pre-extraction code carried

Interpretation:

- these were seam-move maintenance details, not design regressions
- the extraction batch stayed behavior-preserving after restoring those
  load-bearing details

#### 6. Full validation passed after the split

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also covered the graph-focused regression
surface most relevant to the extraction:

- `test_graph`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

Interpretation:

- the new `src/sparse_graph_coarsen.c` seam is wired correctly
- the graph/ND behavior remained stable through the second extraction batch

## Day 7

**Objective:** Audit the post-Day-6 hierarchy/coarsening state so the second
coarsening-phase push stays bounded to real residual cleanup instead of
drifting into coarse-bisection, FM refinement, or separator-lifting work.

### Commands Run

1. Re-read the Sprint 43 Day 7/8 plan sections:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
2. Re-read the Day 6 extraction artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day6-hierarchy-coarsening-extraction-batch1.md`
3. Sweep the post-Day-6 graph surfaces for remaining coarsening, hierarchy,
   and coarse-level ownership seams:
   - `rg -n "graph_uncoarsen|graph_bisect_coarsest|graph_bisect_coarsest_spectral|bisect_brute_force|bisect_gggp|graph_build_laplacian|parse_coarsest_bisect_strategy|sparse_graph_hierarchy_build|graph_coarsen|coarsen|hierarchy|cmap|coarse" src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_internal.h`
   - `sed -n '340,980p' src/sparse_graph.c`
   - `sed -n '1,320p' src/sparse_graph_coarsen.c`
   - `sed -n '260,420p' src/sparse_graph_internal.h`

### Day 7 Findings

#### 1. The real hierarchy/coarsening implementation extraction is already substantially complete

After Day 6, `src/sparse_graph_coarsen.c` already owns the main
hierarchy/coarsening behavior:

- coarsening-strategy ownership and override helpers
- `graph_coarsen_with_strategy(...)`
- `graph_coarsen_heavy_edge_matching(...)`
- `graph_coarsen_hcc(...)`
- `sparse_graph_hierarchy_build(...)`
- `sparse_graph_hierarchy_free(...)`

Interpretation:

- there is no second large hidden coarsening core still stranded in
  `src/sparse_graph.c`
- Sprint 43's coarsening extraction is now closer to completion/consolidation
  than to another major code-move batch

#### 2. The remaining monolith now starts with coarse bisection, not residual coarsening

The first major post-Day-6 seam still in `src/sparse_graph.c` is:

- `bisect_brute_force(...)`
- `bisect_gggp(...)`
- `graph_build_laplacian(...)`
- `graph_bisect_coarsest_spectral(...)`
- `parse_coarsest_bisect_strategy(...)`
- `graph_bisect_coarsest(...)`

Interpretation:

- this is the correct Day 9 extraction seam
- it should not be pulled into Day 8 just because it still talks about coarse
  levels

#### 3. The other coarsening-adjacent region is really uncoarsening/orchestration work

The remaining lifecycle that still references hierarchy/coarse-level state is
primarily:

- `graph_uncoarsen(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

Interpretation:

- this logic is still coupled to FM refinement, separator lifting, and
  top-level retry/orchestration behavior
- it is intentionally out of scope for the second coarsening batch

#### 4. The residual Day 8 queue is mainly interface cleanup, not another big move

The bounded cleanup still worth doing before coarse-bisection extraction is:

- clearer coarsening-facing declaration grouping in
  `src/sparse_graph_internal.h`
- comment/ownership wording cleanup so the extracted seam is documented as an
  extracted subsystem rather than still described like one monolith
- preservation of the small strategy-helper ownership map that the top-level
  retry path still depends on

Interpretation:

- Day 8 should be a finish-and-consolidate batch
- Day 8 should not reopen FM, separator, or orchestration churn

#### 5. The keep-local / defer set remains explicit

These should stay out of the Day 8 batch:

- `graph_uncoarsen(...)`
- FM bucket/refinement machinery
- FM strategy parsing and thread-local controls
- separator lifting / final partition projection
- broader runtime strategy glue spanning multiple graph phases

Interpretation:

- Sprint 43 can stay bounded and still improve the graph subsystem structure
- the remaining graph phases do not need to move just because the
  coarsening/hierarchy split is now real

#### 6. Day 9 now has a cleaner handoff target

With Day 6 and Day 7 combined, the next extraction handoff is clearer:

- `src/sparse_graph_coarsen.c` owns hierarchy/coarsening
- `src/sparse_graph.c` retains coarse bisection, FM, separator lifting, and
  orchestration
- the next real implementation seam is coarse bisection, not another
  hierarchy/coarsening block

Interpretation:

- the Phase-1 extraction order is holding
- the sprint is positioned to move into bisection without re-auditing the
  coarsening module again

## Day 8

**Objective:** Complete the planned first-phase hierarchy/coarsening
extraction by landing the residual shared-interface and ownership cleanup the
Day 7 audit identified, without pulling coarse-bisection, FM refinement, or
separator-lifting work forward.

### Commands Run

1. Re-read the Sprint 43 Day 8 plan section and the Day 7 audit:
   - `sed -n '150,260p' docs/planning/EPIC_4/SPRINT_43/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day7-residual-coarsening-hierarchy-audit.md`
2. Re-audit the live declaration/ownership split:
   - `rg -n "coarsen|hierarch|coarsening_strategy|force_hem|parse_coarsening|sparse_graph_hierarchy|graph_coarsen|coarse_edge_t|splitmix64|fisher_yates" src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_internal.h`
   - `sed -n '130,360p' src/sparse_graph_internal.h`
   - `sed -n '1,140p' src/sparse_graph.c`
   - `sed -n '1,120p' src/sparse_graph_core.c`
3. After landing the Day 8 cleanup, run the full required gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 8 Findings

#### 1. The real residual Phase-1 risk was interface drift, not missing coarsening code movement

The Day 7 audit held up under the final Day 8 pass:

- `src/sparse_graph_coarsen.c` already owned the real hierarchy/coarsening
  implementation seam
- the remaining monolith did not hide another large coarsening block
- the main residual mismatch was that the shared internal header still grouped
  one coarse-bisection helper under the coarsening surface

Interpretation:

- the correct Day 8 batch was interface/ownership cleanup
- another broad code move would have been fake progress

#### 2. `graph_build_laplacian(...)` was the main declaration that still blurred the next seam

Before Day 8, `graph_build_laplacian(...)` still sat inside the coarsening
section of `src/sparse_graph_internal.h`, even though it exists only to
support spectral coarse bisection.

Day 8 moved it into the coarse-bisection / FM section.

Interpretation:

- the header now reflects the actual subsystem boundary more honestly
- Day 9 no longer inherits a coarsening-vs-bisection grouping ambiguity for
  the spectral path

#### 3. The shared internal header now documents the extracted coarsening seam explicitly

Day 8 also tightened the coarsening banner in
`src/sparse_graph_internal.h` so it now states directly that:

- the implementation lives in `src/sparse_graph_coarsen.c`
- new coarsening helpers should not drift back into `src/sparse_graph.c`
- the next grouped seam is coarse bisection, not more coarsening

Interpretation:

- the header is now a better ownership map for later graph work
- the coarsening module is less likely to regress into "split in code, merged
  in comments"

#### 4. The remaining monolith now documents its real Sprint 43 ownership more accurately

The top-level note in `src/sparse_graph.c` now names the file as the
remaining:

- coarse-bisection
- FM refinement
- uncoarsening
- separator lifting
- top-level orchestration

slice, while pointing construction/ownership to `src/sparse_graph_core.c` and
hierarchy/coarsening to `src/sparse_graph_coarsen.c`.

Interpretation:

- the file no longer reads like it still owns the full graph pipeline
- that makes the next extraction batches less error-prone

#### 5. The keep-bounded rule still held

Day 8 intentionally did **not** move:

- coarse-bisection implementation
- `graph_uncoarsen(...)`
- FM refinement
- separator lifting
- top-level retry/orchestration glue

Interpretation:

- Sprint 43 stayed inside the Day 7 boundary
- the extraction order remains:
  - graph ownership / construction
  - hierarchy / coarsening
  - coarse bisection
  - later FM / separator cleanup

#### 6. Full validation passed after the cleanup

Because `*.c` and `*.h` files changed, the full required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also covered the graph-focused regression
surface most relevant to the current subsystem split:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

All remained green.

Interpretation:

- the cleanup batch preserved behavior
- the graph subsystem is now in a cleaner state for Day 9 coarse-bisection
  extraction
