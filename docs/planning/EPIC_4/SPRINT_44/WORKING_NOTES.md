# Sprint 44 Working Notes

## Day 1

**Objective:** Turn the Sprint 44 project-plan scope plus the Sprint 43 graph
closeout and the Sprint 40/41/42 execution rules into a concrete baseline by
confirming the preserved reviewed contracts, naming the Sprint 44 workstreams
explicitly, and defining the authoritative residual-graph and large-test
hotspot inputs before Phase-2 extraction begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short`
   - `git rev-parse --abbrev-ref HEAD`
2. Re-read the Sprint 44 plan and the main prerequisite planning artifacts:
   - `sed -n '156,185p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_43/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
5. Measure the live residual graph hotspot and current large-test / graph test
   concentration:
   - `wc -l src/sparse_graph.c src/sparse_graph_core.c src/sparse_graph_coarsen.c src/sparse_graph_bisect.c tests/test_chol_csc.c tests/test_svd.c tests/test_ldlt_csc.c tests/test_qr.c tests/test_graph.c tests/test_graph_fm_buckets.c tests/test_reorder_nd.c tests/test_reorder_amd_qg.c`
6. Refresh the residual graph seam markers:
   - `rg -n "graph_refine_fm|separator|SPARSE_ND_|parse_.*strategy|retry_with_forced_hem|partition" src/sparse_graph.c | sed -n '1,220p'`

### Day 1 Findings

#### 1. Sprint 44 starts from a preserved Sprint 40/41/42/43 baseline, not from baseline repair work

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
  - internal lifecycle/factor-state scaffolding
  - shared matrix-state guard helpers
  - compatibility-preserving factor-path normalization
- Sprint 43 already completed graph Phase 1:
  - `src/sparse_graph_core.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - narrowed residual `src/sparse_graph.c`

Interpretation:

- Sprint 44 is not a quality-baseline sprint
- Sprint 44 is a Phase-2 graph decomposition plus bounded large-test
  maintainability sprint on top of an already-validated Epic 4 baseline

#### 2. The residual graph hotspot is still large, but it is now clearly Phase-2 work

The live graph decomposition state is now:

- `src/sparse_graph.c` = `2153` lines
- extracted Phase-1 graph files:
  - `src/sparse_graph_core.c` = `264`
  - `src/sparse_graph_coarsen.c` = `597`
  - `src/sparse_graph_bisect.c` = `521`

The residual `src/sparse_graph.c` still visibly spans:

- FM refinement
- separator lifting
- top-level partition orchestration
- runtime strategy parsing / override glue
- fallback / retry policy

Interpretation:

- Sprint 43 reduced the graph monolith materially, but it did not eliminate
  the Phase-2 graph seams
- Sprint 44 is still correctly aimed at a real remaining hotspot rather than a
  synthetic follow-on task

#### 3. The largest inherited maintainability hotspots are now concentrated in the expected test binaries

The live large-test concentration is:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_svd.c` = `3746`
- `tests/test_ldlt_csc.c` = `3637`
- `tests/test_qr.c` = `3291`

The graph-focused regression surface remains concentrated in:

- `tests/test_graph.c` = `2753`
- `tests/test_graph_fm_buckets.c` = `404`
- `tests/test_reorder_nd.c` = `1594`
- `tests/test_reorder_amd_qg.c` = `273`

Interpretation:

- Sprint 44 does not need another exploratory maintainability sprint before it
  begins the first test-helper batch
- the strongest large-test maintainability targets are already explicit on Day 1

#### 4. The Sprint 44 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's seven workstreams directly from the plan:

- FM refinement extraction
- separator lifting extraction
- runtime strategy parsing cleanup
- final graph orchestration cleanup
- large-test helper audit
- first test-helper consolidation batch
- validation closeout

Interpretation:

- the front half of the sprint should stay graph-first:
  - residual seam inventory
  - FM boundary design
  - separator/runtime/test design
  - bounded extraction batches
- the back half should then pivot into focused graph seam protection and the
  first large-test helper consolidation batch

#### 5. Sprint 44 inherits a clear preserve-not-reopen boundary

Sprint 44 should not reopen:

- public API redesign
- lifecycle-handle redesign beyond the inherited Sprint 42 seams
- cross-platform CI contract changes
- dead-code topology changes
- broad benchmark/script cleanup unrelated to graph or the selected tests
- whole-file test splitting or test-framework redesign

Interpretation:

- the correct Sprint 44 shape is:
  - finish the residual graph decomposition
  - simplify orchestration and parser ownership
  - add focused graph seam tests
  - land a bounded helper/fixture consolidation batch in the biggest tests
- broader structural cleanup remains later Epic 4 work

#### 6. The Day 1 implementation order is fixed before code changes begin

The correct early sprint order is:

1. baseline and residual-graph inventory
2. FM boundary design
3. separator/runtime/test design
4. bounded FM and separator extraction batches
5. runtime/orchestration cleanup
6. focused graph seam tests
7. large-test audit and first helper consolidation batch

Interpretation:

- Sprint 44 should preserve Sprint 40's core rule: structural refactors should
  be guided by measured seams and explicit ownership boundaries before code
  movement lands

## Day 2

**Objective:** Refresh the internal seam inventory inside the residual
`src/sparse_graph.c` so Sprint 44's extraction order is grounded in the live
post-Sprint-43 file rather than only in the project-plan labels, with explicit
separation between extract-now FM/separator/runtime seams and the orchestration
glue that should remain until after those moves land.

### Commands Run

1. Re-read the Sprint 44 Day 2 plan section:
   - `sed -n '56,90p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
2. Re-read the Sprint 43 closeout notes for the residual graph ownership model:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day10-runtime-strategy-and-glue-reconciliation.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_43/artifacts/day14-closeout-and-handoff.md`
3. Refresh the live residual graph seam markers and size:
   - `wc -l src/sparse_graph.c`
   - `rg -n "graph_refine_fm|graph_uncoarsen|graph_edge_separator_to_vertex_separator|parse_sep_lift_strategy|SPARSE_ND_|partition_once|sparse_graph_partition|graph_partition_|compute_cut_weight|sep_lift|per_vertex|balanced_boundary|retry" src/sparse_graph.c`
4. Re-read the high-signal residual regions directly:
   - `sed -n '340,1155p' src/sparse_graph.c`
   - `sed -n '1155,2025p' src/sparse_graph.c`
   - `sed -n '2021,2165p' src/sparse_graph.c`
5. Re-read the current shared graph internal declarations:
   - `sed -n '1,260p' src/sparse_graph_internal.h`
   - `rg -n "graph_refine_fm|graph_uncoarsen|graph_edge_separator_to_vertex_separator|parse_sep_lift_strategy|parse_sep_lift_weight|is_per_vertex_strategy|per_vertex_score_cmp_desc|graph_partition_seed_coarsest|graph_partition_should_retry_with_forced_hem|partition_once|sparse_graph_partition|compute_cut_weight" src/sparse_graph_internal.h`

### Day 2 Findings

#### 1. The residual `src/sparse_graph.c` now reduces cleanly to five Phase-2 seam classes

The current file maps cleanly to these regions:

- FM refinement core
  - gain-bucket implementation
  - FM env/parser overlays
  - `compute_cut_weight(...)`
  - `graph_refine_fm(...)`
- uncoarsening and finest-level strategy orchestration
  - `graph_uncoarsen(...)`
  - finest-pass env parsing
  - annealing / thick-restart / ensemble strategy setup
- separator lifting and policy selection
  - separator strategy enums
  - `parse_sep_lift_strategy(...)`
  - `parse_sep_lift_weight(...)`
  - `is_per_vertex_strategy(...)`
  - `per_vertex_score_cmp_desc(...)`
  - `graph_edge_separator_to_vertex_separator(...)`
- top-level partition orchestration
  - `graph_hierarchy_coarsest(...)`
  - `graph_partition_seed_coarsest(...)`
  - `graph_partition_count_separator_vertices(...)`
  - `partition_once(...)`
  - `sparse_graph_partition(...)`
- retry / fallback glue
  - `graph_partition_should_retry_with_forced_hem(...)`
  - coarsening override begin/end composition

Interpretation:

- Sprint 43 successfully narrowed the graph monolith enough that Sprint 44 can
  now treat the residual file as a set of named Phase-2 seams rather than one
  generic cleanup region

#### 2. FM refinement is the strongest extract-now seam

The strongest Day 2 extract-now seam is the FM cluster:

- `compute_cut_weight(...)`
- gain-bucket implementation
- FM thread-local strategy controls
- `parse_fm_anneal_schedule(...)`
- `graph_refine_fm(...)`

Reasons:

- the cluster is already internally cohesive
- it has a direct focused test sibling:
  - `tests/test_graph_fm_buckets.c`
- the existing internal header already treats `graph_refine_fm(...)` as a
  stable cross-module seam
- Sprint 43 Day 10 already framed FM as residual module-owned logic rather than
  top-level orchestration glue

Interpretation:

- Sprint 44 should treat FM extraction as the first major implementation batch
- `compute_cut_weight(...)` can move with FM in Phase 2 because its strongest
  remaining consumers are FM and uncoarsening logic, not the already-extracted
  Phase-1 modules

#### 3. Separator lifting is the second strongest extract-now seam

The separator cluster is also now strongly self-contained:

- separator strategy enums
- per-vertex score type and comparator
- separator env parsers
- `graph_edge_separator_to_vertex_separator(...)`

Reasons:

- the whole region is already contiguous in `src/sparse_graph.c`
- the env/parser/config logic is tightly coupled to separator policy rather than
  to general partition orchestration
- the existing internal header already exposes only the behavior-level seam:
  - `graph_edge_separator_to_vertex_separator(...)`

Interpretation:

- Sprint 44 should treat separator lifting as the second direct extraction seam
- the `parse_sep_lift_*` helpers belong with separator lifting, not in a
  generic runtime-parser file on the first move

#### 4. The true residual monolith after FM and separator moves is smaller than the file size alone suggests

Once FM and separator lifting move out, the remaining `src/sparse_graph.c`
should primarily be:

- `graph_uncoarsen(...)`
- coarsest seed/orchestration helpers
- `partition_once(...)`
- `sparse_graph_partition(...)`
- sep=0 retry / override composition

Interpretation:

- Sprint 44's later graph cleanup is not "find another big subsystem"
- it is simplify the composition layer after FM and separator ownership move
- `graph_uncoarsen(...)` is the main bridge object between extracted FM,
  extracted separator lifting, and already-extracted Phase-1 graph modules

#### 5. Runtime/config parsing splits into three different ownership classes

The residual file does not have one generic parser seam. It has three classes:

- FM-owned parser/config logic:
  - `parse_fm_anneal_schedule(...)`
  - finest-level strategy env handling inside `graph_uncoarsen(...)`
  - thick-restart / ensemble / gain-noise parser glue
- separator-owned parser/config logic:
  - `parse_sep_lift_strategy(...)`
  - `parse_sep_lift_weight(...)`
- orchestration-owned retry/config logic:
  - `graph_partition_should_retry_with_forced_hem(...)`
  - explicit forced-HEM retry composition in `sparse_graph_partition(...)`

Interpretation:

- Sprint 44 should not create a generic "graph_runtime_config.c" file first
- parser ownership should follow the extracted subsystem seams
- only the residual top-level retry/config composition should remain with the
  orchestration layer after the FM and separator moves land

#### 6. Shared declarations are already narrower than the residual file body

The current internal header only exposes these residual behavior seams:

- `graph_refine_fm(...)`
- `graph_uncoarsen(...)`
- `graph_edge_separator_to_vertex_separator(...)`
- `sparse_graph_partition(...)`

The following remain translation-unit local today and should stay local unless
later extraction actually needs them cross-file:

- `compute_cut_weight(...)`
- `parse_sep_lift_strategy(...)`
- `parse_sep_lift_weight(...)`
- `is_per_vertex_strategy(...)`
- `per_vertex_score_cmp_desc(...)`
- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`

Interpretation:

- Sprint 44 can keep the shared-header expansion small
- Day 3 should promote only the minimum declarations needed for a real FM file
- Day 4 should do the same for separator lifting

#### 7. The correct Phase-2 extraction order is now explicit

The strongest Day 2 implementation order is:

1. FM refinement boundary design
2. separator/runtime/test design
3. FM extraction
4. separator extraction
5. runtime/orchestration cleanup after those moves land

Interpretation:

- this matches the Day 1 workstream order but is now grounded in the live
  residual file rather than only the project-plan labels
- Sprint 44 should not start with orchestration cleanup because the remaining
  orchestration shape depends on the FM and separator moves landing first
