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
