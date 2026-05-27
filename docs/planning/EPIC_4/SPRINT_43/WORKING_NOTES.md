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
