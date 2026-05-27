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

## Day 3

**Objective:** Define the explicit module boundary for FM refinement extraction
so Sprint 44 can move the refinement core out of `src/sparse_graph.c` without
dragging `graph_uncoarsen(...)` or broader orchestration logic into the same
batch.

### Commands Run

1. Re-read the Sprint 44 Day 3 plan section:
   - `sed -n '90,122p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
2. Re-read the Day 2 residual seam inventory:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/artifacts/day2-residual-graph-seam-refresh-inventory.md`
3. Re-read the current public-ish FM helper header:
   - `sed -n '1,260p' src/sparse_graph_fm_buckets.h`
4. Re-read the live FM region in `src/sparse_graph.c`:
   - `sed -n '340,1165p' src/sparse_graph.c`
5. Sweep the current FM-related symbol ownership across implementation, header,
   and tests:
   - `rg -n "graph_refine_fm|compute_cut_weight|fm_bucket_|fm_use_|fm_anneal_|fm_thick_restart|fm_gain_noise|parse_fm_anneal_schedule|FINEST_FM_|SPARSE_FM_" src/sparse_graph.c src/sparse_graph_internal.h src/sparse_graph_fm_buckets.h tests/test_graph_fm_buckets.c tests/test_graph.c`

### Day 3 Findings

#### 1. The right Day 5 target is one dedicated FM implementation unit, not another mixed graph helper file

The best Sprint 44 FM target is:

- `src/sparse_graph_refine.c`

Reasoning:

- the extracted ownership is algorithm-focused rather than generic
- the live code already groups the FM region as one cohesive slice:
  - local score calculation
  - gain-bucket implementation
  - thread-local FM mode state
  - acceptance / perturbation overlays
  - `graph_refine_fm(...)`
- naming it `refine` matches the role better than a broader
  `runtime`/`helpers`/`phase2` label

Interpretation:

- Day 5 should land a real FM subsystem file, not a catch-all helper bucket

#### 2. The FM-owned extraction boundary is broader than just `graph_refine_fm(...)`

The FM-owned extraction set should include:

- FM thread-local controls:
  - `fm_pop_use_tail`
  - `fm_use_annealing`
  - `fm_anneal_schedule`
  - `fm_use_thick_restart`
  - `fm_thick_restart_perturb`
  - `fm_gain_noise_schedule`
  - `fm_anneal_pass_idx`
  - `fm_anneal_total_passes`
- FM parser/helpers:
  - `parse_fm_anneal_schedule(...)`
  - `parse_fm_thick_restart_perturb(...)`
  - `parse_fm_gain_noise_schedule(...)`
  - `thick_restart_perturb(...)`
- FM local score/update helper:
  - `compute_cut_weight(...)`
- FM bucket implementation:
  - `fm_bucket_array_init(...)`
  - `fm_bucket_array_free(...)`
  - `fm_bucket_insert(...)`
  - `fm_bucket_remove(...)`
  - `fm_bucket_pop_max(...)`
  - `fm_bucket_pop_max_tail(...)`
- FM algorithm entry point:
  - `graph_refine_fm(...)`

Interpretation:

- the FM module should own not just the hot loop, but also the local state and
  perturbation vocabulary that make the hot loop work
- leaving the thread-local FM control state behind in `src/sparse_graph.c`
  would produce a fake boundary

#### 3. `graph_uncoarsen(...)` should stay out of the FM extraction even though it currently parses most finest-level FM env vars

`graph_uncoarsen(...)` currently owns:

- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- the call-time setup/restoration of FM thread-local controls

Day 3 decision:

- keep `graph_uncoarsen(...)` in the residual orchestration file for Sprint 44
- do **not** move the finest-level orchestration loop with the FM module
- do move only the FM-local parser/helpers that describe refinement behavior
  itself:
  - annealing schedule
  - thick-restart perturbation mode
  - gain-noise schedule

Interpretation:

- Day 5 should extract the FM implementation seam
- Day 8 can then simplify `graph_uncoarsen(...)` against that cleaner FM seam
- this keeps the batch bounded and avoids treating orchestration as part of FM

#### 4. The existing shared-header surface is already close to the right final shape

Today the shared graph internal header exports:

- `graph_refine_fm(...)`

and the FM bucket header exports:

- `fm_bucket_array_t`
- bucket API functions

Day 3 decision:

- keep `graph_refine_fm(...)` as the only FM behavior seam in
  `src/sparse_graph_internal.h`
- keep the bucket API in `src/sparse_graph_fm_buckets.h`
- do **not** promote FM parser/helpers or thread-local state into shared
  headers
- do **not** expose `compute_cut_weight(...)`

Interpretation:

- Day 5 can land with little or no shared-header expansion
- most of the FM extraction is file movement plus include/ownership cleanup,
  not interface growth

#### 5. Bucket ownership is explicit and should remain separate from broader graph internals

The live code already treats the bucket API as a narrow reusable FM support
surface:

- dedicated header:
  - `src/sparse_graph_fm_buckets.h`
- dedicated focused tests:
  - `tests/test_graph_fm_buckets.c`

Day 3 decision:

- keep the bucket API and its tests as a dedicated FM-support seam
- the extracted FM implementation unit should include that header rather than
  reabsorbing bucket declarations into `src/sparse_graph_internal.h`

Interpretation:

- the FM extraction should preserve the existing narrow bucket abstraction
- Sprint 44 should not collapse bucket internals back into a broader graph
  header just because the implementation file moves

#### 6. `compute_cut_weight(...)` should move with FM in Day 5, even though `graph_uncoarsen(...)` also uses it

Current usage pattern:

- FM uses `compute_cut_weight(...)` internally
- `graph_uncoarsen(...)` uses it for thick-restart / ensemble bookkeeping

Day 3 decision:

- move `compute_cut_weight(...)` into `src/sparse_graph_refine.c`
- add a minimal shared declaration for it in `src/sparse_graph_internal.h`
  only if Day 5 needs `graph_uncoarsen(...)` to call across files
- otherwise prefer a translation-unit-local helper wrapper strategy if the move
  can stay narrower

Interpretation:

- the helper is more conceptually FM-owned than orchestration-owned
- but it is the main candidate for a tiny shared-header addition during the
  extraction batch

#### 7. The Day 5 FM batch should stay explicitly bounded

Do on Day 5:

- create `src/sparse_graph_refine.c`
- move the FM bucket implementation, FM parser/helpers, thread-local FM state,
  `compute_cut_weight(...)`, and `graph_refine_fm(...)`
- update build wiring and the minimal headers/includes needed

Do not do on Day 5:

- move `graph_uncoarsen(...)`
- move separator lifting
- redesign FM env-var contracts
- redesign `tests/test_graph_fm_buckets.c` or `tests/test_graph.c`
- broaden into runtime/orchestration cleanup

Interpretation:

- this preserves a real extracted FM subsystem while keeping the batch small
  enough to validate cleanly before Sprint 44 pivots into separator work

## Day 4

**Objective:** Bound the separator-lifting extraction seam, the residual
runtime/parser cleanup seam, and the first large-test maintainability target
set before Sprint 44 begins the main implementation batches.

### Commands Run

1. Re-read the Sprint 44 Day 4 plan section:
   - `sed -n '122,157p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
2. Re-read the Day 2 and Day 3 design artifacts:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/artifacts/day2-residual-graph-seam-refresh-inventory.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/artifacts/day3-fm-refinement-module-boundary-design.md`
3. Re-read the live separator and orchestration region:
   - `sed -n '1630,2165p' src/sparse_graph.c`
4. Reconfirm the largest-test size concentration and sweep for existing helper
   structures:
   - `wc -l tests/test_chol_csc.c tests/test_svd.c tests/test_ldlt_csc.c tests/test_qr.c`
   - `rg -n '^static ' tests/test_chol_csc.c tests/test_svd.c tests/test_ldlt_csc.c tests/test_qr.c`
   - `rg -n "static .*helper|fixture|make_.*matrix|build_.*matrix|assert_.*residual|check_.*residual|run_.*case|expect_.*|solve_.*case|compare_.*|setup_.*|teardown_.*" tests/test_chol_csc.c tests/test_svd.c tests/test_ldlt_csc.c tests/test_qr.c`
5. Re-read the highest-signal helper-heavy slices in the large tests:
   - `sed -n '2480,2915p' tests/test_qr.c`
   - `sed -n '3840,4665p' tests/test_chol_csc.c`
   - `sed -n '1160,1565p' tests/test_ldlt_csc.c`
   - `sed -n '3090,3415p' tests/test_svd.c`

### Day 4 Findings

#### 1. The separator extraction target is one dedicated policy-and-conversion module

The best Sprint 44 separator target is:

- `src/sparse_graph_separator.c`

The owned extraction set should include:

- separator strategy enums:
  - `sep_lift_strategy_t`
  - `sep_lift_weight_t`
- separator parser/helpers:
  - `parse_sep_lift_strategy(...)`
  - `parse_sep_lift_weight(...)`
  - `is_per_vertex_strategy(...)`
  - `per_vertex_score_cmp_desc(...)`
- separator algorithm entry point:
  - `graph_edge_separator_to_vertex_separator(...)`

Interpretation:

- Day 6 should land a real separator subsystem file, not just move the main
  conversion function by itself
- the parser/config logic should move with separator lifting because it is
  policy-specific rather than orchestration-generic

#### 2. The separator seam should stay narrower than the full post-FM orchestration layer

Even after FM moves out, these should stay outside the separator module:

- `graph_uncoarsen(...)`
- `graph_partition_seed_coarsest(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- `graph_partition_should_retry_with_forced_hem(...)`

Reason:

- those functions sequence multiple subsystems
- separator lifting is the final projection step, not the owner of the full
  partition pipeline

Interpretation:

- Day 6 should extract separator ownership without turning the separator module
  into a new orchestration file

#### 3. Runtime/config cleanup should follow subsystem ownership, not start from a generic parser file

Day 4 confirms three runtime/config ownership classes:

- FM-owned:
  - FM refinement behavior parsers and overlays
- separator-owned:
  - separator policy parsers and per-vertex scoring mode selection
- residual orchestration-owned:
  - finest-pass count / strategy selection
  - ensemble selector-list parsing
  - intermediate-pass parsing
  - sep=0 retry / forced-HEM composition

Day 4 decision:

- do not create a generic `src/sparse_graph_runtime.c` or parser bucket file
- let Day 5 and Day 6 move parser logic along with FM and separator ownership
- leave only the orchestration-scoped env parsing in the residual file for Day 8

Interpretation:

- Sprint 44's runtime cleanup is a post-extraction simplification step, not an
  independent first-wave module split

#### 4. The strongest large-test helper seams are real, but they differ by file

The live large-test surfaces suggest different first-batch targets:

- `tests/test_qr.c`
  - strongest helper seam:
    - `compare_dense_sparse_qr(...)`
  - repeated sparse-mode cases already fan into one helper
  - likely next opportunities:
    - repeated factor/solve/compare harnesses
    - repeated reconstruction/residual setup
- `tests/test_chol_csc.c`
  - strongest helper seam:
    - repeated supernodal cross-check / roundtrip / dispatch fixture harnesses
  - likely next opportunities:
    - Day 9 / Day 10 helper clusters
    - repeated SPD fixture builders and factor-match checks
- `tests/test_ldlt_csc.c`
  - strongest helper seam:
    - repeated indefinite fixture builders and two-pass factor harnesses
  - likely next opportunities:
    - KKT fixture builders
    - repeated solve-residual / factor-match harnesses
- `tests/test_svd.c`
  - strongest helper seam:
    - repeated dense fixture fill and full/economy/output comparison harnesses
  - likely next opportunities:
    - repeated 16×8 full-mode fixture setup
    - repeated low-rank corpus safety loops

Interpretation:

- Sprint 44's test maintainability work should target helper/fixture
  consolidation, not file splitting
- the strongest first batch is likely in QR and one of Chol/LDLT/SVD, not all
  four files at once

#### 5. The bounded Day 11 / Day 12 target shortlist is now explicit

Best Day 11 audit focus:

- `tests/test_qr.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`
- `tests/test_svd.c`

Best Day 12 likely implementation targets:

- `tests/test_qr.c`
  - because it already has a clear comparison-helper seam that can probably be
    extended without structural churn
- one of:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_svd.c`

Selection rule:

- choose the file with the clearest repeated helper/fixture pattern after the
  Day 11 audit, not the file that is merely largest

Interpretation:

- Sprint 44 should land a high-signal first maintainability batch instead of
  spreading across all four large tests

#### 6. Shared-header expansion for separator extraction should stay minimal, just like FM

Today the internal graph header exports:

- `graph_edge_separator_to_vertex_separator(...)`

Day 4 decision:

- keep that as the main shared separator behavior seam
- do not promote separator strategy enums or parser helpers into broader shared
  headers unless the extraction requires a narrow private header later
- prefer translation-unit-local policy helpers in the new separator file

Interpretation:

- Day 6 should be mostly file movement plus build/include cleanup, not interface
  growth

#### 7. The Day 5-Day 8 graph order is now completely fixed

The strongest implementation order is:

1. Day 5:
   - FM extraction
2. Day 6:
   - separator extraction
3. Day 7:
   - residual runtime/orchestration audit
4. Day 8:
   - runtime/orchestration cleanup after the moves land

Interpretation:

- Sprint 44 should not interleave separator extraction with runtime cleanup
- the residual orchestration layer can only be simplified honestly after the
  FM and separator moves are complete

## Day 5

**Objective:** Extract the bounded FM refinement subsystem out of the
residual graph monolith while preserving `graph_uncoarsen(...)`,
separator lifting, and top-level partition orchestration in
`src/sparse_graph.c`.

### Commands Run

1. Re-read the Day 5 plan and the Day 3 FM boundary design:
   - `sed -n '158,205p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/artifacts/day3-fm-refinement-module-boundary-design.md`
2. Re-read the live FM region and shared graph contracts:
   - `sed -n '1,1400p' src/sparse_graph.c`
   - `sed -n '1,520p' src/sparse_graph_internal.h`
   - `sed -n '1,260p' src/sparse_graph_fm_buckets.h`
3. Sweep live FM-owned symbols and orchestration consumers:
   - `rg -n "graph_refine_fm|compute_cut_weight|thick_restart_perturb|fm_|parse_.*fm|parse_.*anneal|parse_.*noise" src/sparse_graph.c src/sparse_graph_internal.h`
4. Validate the landed extraction batch:
   - `make format`
   - `make lint`
   - `make test`

### Day 5 Findings

#### 1. The FM Phase-2 batch landed as one real owned module

The extracted implementation unit is now:

- `src/sparse_graph_refine.c`

It owns the bounded Day 3 seam:

- FM thread-local runtime state
- FM parser/helpers
- thick-restart perturbation helper
- shared cut-weight evaluation
- FM bucket implementation
- `graph_refine_fm(...)`

Interpretation:

- Sprint 44 now has a real FM subsystem file rather than FM logic still living
  only inside the residual monolith

#### 2. The residual graph file is now narrower and more honest

`src/sparse_graph.c` no longer carries the FM implementation body.

It now starts at the intended residual seam:

- `graph_uncoarsen(...)`
- separator lifting
- top-level partition orchestration
- sep=`0` retry / fallback glue

Interpretation:

- the remaining graph monolith now reads as orchestration and
  separator-adjacent logic rather than as a mixed refinement +
  orchestration file

#### 3. The Day 5 interface growth stayed small and internal

The extraction did not promote FM behaviour into public headers.

The internal graph seam only grew enough to support orchestration:

- FM schedule / perturbation enums
- `sparse_graph_fm_runtime_t`
- parser helpers
- runtime get/set helpers
- `sparse_graph_compute_cut_weight(...)`
- `sparse_graph_thick_restart_perturb(...)`

Interpretation:

- Day 5 preserved the Day 3 rule: shared-header expansion is minimal and tied
  directly to the live orchestration seam

#### 4. The build wiring changed in the bounded expected way

Both maintained build systems now compile the extracted FM module:

- `Makefile`
- `CMakeLists.txt`

No broader graph build or test-matrix redesign was needed.

Interpretation:

- the FM extraction landed as a normal library-source split, not as a special
  build-path exception

#### 5. The runtime/orchestration boundary held

Day 5 intentionally did **not** move:

- `graph_uncoarsen(...)`
- ensemble/top-level FM pass composition
- separator lifting
- `partition_once(...)`
- `sparse_graph_partition(...)`

Interpretation:

- Sprint 44 stayed within the intended FM-only batch instead of pulling Day 6
  or Day 7 work forward

#### 6. Validation passed at the full required gate

The Day 5 batch touched `*.c` and `*.h`, so the full gate was required:

- `make format` — passed
- `make lint` — passed
- `make test` — passed

The authoritative `make test` sweep also re-covered the graph-focused
surfaces touched by the extraction:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

Interpretation:

- the FM extraction was structural only; the maintained graph and ND regression
  surface stayed green end to end

#### 7. Day 6 is now cleanly prepared

After Day 5, the next Phase-2 seam is better isolated:

- separator lifting and separator-policy parsing still remain together in the
  residual graph file
- FM-owned parsing/state is already gone from that file

Interpretation:

- Day 6 can now extract separator ownership from a cleaner residual
  orchestration layer instead of from the pre-Day-5 mixed FM/orchestration
  monolith

## Day 6

**Objective:** Land the bounded Sprint 44 Phase-2 separator extraction by
moving separator-policy parsing, per-vertex separator scoring, and final
edge-to-vertex separator conversion into a dedicated implementation unit
without reopening FM or broader orchestration work.

### Commands Run

1. Re-read the Day 6 plan/design inputs and separator seam:
   - `sed -n '170,220p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_44/artifacts/day4-separator-runtime-and-large-test-design.md`
   - `sed -n '680,1180p' src/sparse_graph.c`
   - `rg -n "sep_lift_strategy_t|sep_lift_weight_t|parse_sep_lift_strategy|parse_sep_lift_weight|is_per_vertex_strategy|per_vertex_score_cmp_desc|graph_edge_separator_to_vertex_separator|SPARSE_ND_SEP_LIFT" src/sparse_graph.c src/sparse_graph_internal.h tests/test_graph.c`
2. Inspect the live residual graph/header/build ownership before editing:
   - `sed -n '1,220p' src/sparse_graph.c`
   - `sed -n '1,220p' src/sparse_graph_internal.h`
   - `sed -n '56,78p' Makefile`
   - `sed -n '96,108p' CMakeLists.txt`
3. Land the Phase-2 separator extraction:
   - create `src/sparse_graph_separator.c`
   - remove the extracted separator block from `src/sparse_graph.c`
   - update ownership notes in `src/sparse_graph.c` and
     `src/sparse_graph_internal.h`
   - add `src/sparse_graph_separator.c` to `Makefile` and `CMakeLists.txt`
4. Reconfirm the post-edit ownership split:
   - `rg -n "graph_edge_separator_to_vertex_separator|SEP_LIFT_|parse_sep_lift_strategy|parse_sep_lift_weight|per_vertex_score_cmp_desc" src/sparse_graph.c src/sparse_graph_separator.c src/sparse_graph_internal.h`
   - `git status --short`
5. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 6 Findings

#### 1. Sprint 44 now has a real separator module

Day 6 created:

- `src/sparse_graph_separator.c`

The new module now owns:

- separator-lift strategy enums
- separator weight enums
- separator env-var parsers
- per-vertex separator scoring helpers
- `graph_edge_separator_to_vertex_separator(...)`

Interpretation:

- Day 6 delivered the intended Phase-2 separator seam as an actual
  implementation unit, not just a comment-level ownership change

#### 2. The residual graph file is now narrower and more honest

After the extraction, `src/sparse_graph.c` no longer owns:

- separator strategy enums
- separator weight enums
- separator env-var parsers
- per-vertex separator scoring helpers
- edge-to-vertex separator conversion

It now retains only the intended residual Phase-2 seam:

- `graph_uncoarsen(...)`
- top-level partition orchestration
- retry / fallback glue

Interpretation:

- the residual graph monolith is now closer to the final orchestration-only
  target that Day 7/Day 8 are supposed to audit and simplify

#### 3. Shared-header growth stayed minimal

The extraction did **not** promote separator-local enums or parser helpers into
`src/sparse_graph_internal.h`.

The shared internal contract still exports only the behavior seam that other
graph phases genuinely use:

- `graph_edge_separator_to_vertex_separator(...)`

Day 6 only updated the ownership comments around that seam.

Interpretation:

- the separator extraction preserved the Day 4 rule: move ownership by file,
  not by broadly expanding shared internal interfaces

#### 4. Build wiring changed in the expected bounded way

Both maintained build systems now compile the extracted separator module:

- `Makefile`
- `CMakeLists.txt`

No special-case graph build logic was required.

Interpretation:

- the separator extraction landed as a normal library-source split, consistent
  with the Sprint 43 and Sprint 44 Phase-1/Phase-2 graph module pattern

#### 5. Sprint 44 stayed within the planned mid-sprint boundary

Day 6 intentionally did **not** move:

- `graph_uncoarsen(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- sep=`0` retry / fallback composition

Interpretation:

- Sprint 44 preserved the Day 4 order:
  1. Day 5: FM extraction
  2. Day 6: separator extraction
  3. Day 7: runtime/orchestration audit
  4. Day 8: residual cleanup

#### 6. Validation passed at the full required gate

Because `*.c` and `*.h` files changed, the full gate was required:

- `make format` — passed
- `make lint` — passed
- `make test` — passed

The authoritative `make test` sweep also re-covered the touched graph/ND
surface directly:

- `test_graph`
- `test_graph_fm_buckets`
- `test_reorder_nd`
- `test_reorder_amd_qg`

Interpretation:

- the separator extraction was structural only; the maintained graph and ND
  regression surface stayed green end to end

#### 7. Day 7 is now cleanly prepared

With FM and separator seams both extracted, the remaining graph file now
mostly expresses:

- uncoarsening
- partition orchestration
- retry/fallback glue

Interpretation:

- Day 7 can now audit real residual runtime/orchestration coupling rather than
  rediscovering ownership noise from FM or separator-local code

## Day 7

**Objective:** Audit the post-Day-6 residual graph file so the remaining
runtime/config parsing, uncoarsening composition, and retry/fallback logic are
classified concretely before the Day 8 cleanup batch.

### Commands Run

1. Re-read the Day 7/Day 8 sprint-plan and Day 4 design targets:
   - `sed -n '200,255p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_44/artifacts/day4-separator-runtime-and-large-test-design.md`
2. Re-read the live residual graph ownership surface:
   - `sed -n '1,260p' src/sparse_graph.c`
   - `sed -n '220,420p' src/sparse_graph.c`
   - `sed -n '420,820p' src/sparse_graph.c`
   - `sed -n '660,760p' src/sparse_graph_internal.h`
3. Inventory the remaining runtime/config and orchestration seams:
   - `rg -n "getenv|strtol|strcmp\\(|SPARSE_" src/sparse_graph.c`
   - `rg -n "graph_uncoarsen|partition_once|sparse_graph_partition|retry|forced_hem|SPARSE_ND_|graph_hierarchy_coarsest|graph_partition_seed_coarsest|graph_partition_count_separator_vertices|graph_partition_should_retry_with_forced_hem|separator|coarsest" src/sparse_graph.c`

### Day 7 Findings

#### 1. The residual graph file now has one real runtime/config cluster, not a hidden fourth extraction seam

After the Day 5 and Day 6 moves, the live residual `src/sparse_graph.c`
contains two main classes of logic:

- uncoarsening / finest-level FM pass composition inside `graph_uncoarsen(...)`
- top-level partition orchestration / retry glue around:
  - `graph_hierarchy_coarsest(...)`
  - `graph_partition_seed_coarsest(...)`
  - `graph_partition_count_separator_vertices(...)`
  - `graph_partition_should_retry_with_forced_hem(...)`
  - `partition_once(...)`
  - `sparse_graph_partition(...)`

The remaining env-var parsing cluster is concentrated entirely inside
`graph_uncoarsen(...)`:

- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_THICK_RESTART_DEBUG`

Interpretation:

- Day 7 did **not** uncover another self-contained module like FM or separator
- the remaining cleanup target is a real orchestration/config simplification
  seam, which matches the Day 4 design

#### 2. The remaining parser logic is partly ready for direct consolidation, but not for another file extraction

The residual parsing now splits into two sub-classes.

Parser/config logic already ready for bounded Day 8 consolidation:

- finest-level strategy enum + parse block
- ensemble selector-list parsing block
- finest-pass count parsing block
- intermediate-pass count parsing block
- ensemble debug flag read
- thick-restart debug flag read

Parser/config logic that is still meaningfully coupled to live orchestration:

- per-level pass-count selection:
  - finest vs intermediate vs coarse levels
- runtime snapshot / restore choreography:
  - `sparse_graph_fm_runtime_get(...)`
  - `sparse_graph_fm_runtime_set(...)`
- thick-restart anchor allocation / restore
- ensemble buffer allocation / winner selection
- per-pass dispatch wiring into `graph_refine_fm(...)`

Interpretation:

- Day 8 should consolidate the parser/config blocks and comment structure
- Day 8 should **not** try to extract a new generic parser file or peel
  `graph_uncoarsen(...)` apart into fake standalone pieces

#### 3. Retry/fallback logic is now clearly a keep-local orchestration seam

The residual retry/fallback surface is tight and already isolated:

- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

These helpers compose:

- hierarchy build from `src/sparse_graph_coarsen.c`
- coarsest split from `src/sparse_graph_bisect.c`
- FM refinement from `src/sparse_graph_refine.c`
- separator lifting from `src/sparse_graph_separator.c`
- forced-HEM retry through the coarsening override seam

Interpretation:

- retry/fallback logic should stay in the residual orchestration file for now
- Day 8 should make this glue clearer, not try to move it into the coarsening
  module or invent a new wrapper layer

#### 4. The best Day 8 cleanup target is simplification around `graph_uncoarsen(...)`, not deeper partition-path surgery

The highest-volume residual implementation body is still
`graph_uncoarsen(...)`.

Its main cleanup opportunities are now explicit:

- consolidate the finest/intermediate env-var parse blocks
- make strategy dispatch comments match the post-Day-5/Day-6 ownership split
- tighten the ensemble / thick-restart / runtime-restore structure
- keep the FM algorithm implementation itself out of this file

The partition entry-point helpers are already comparatively small and honest.

Interpretation:

- Day 8 should focus on `graph_uncoarsen(...)` plus the small orchestration
  helper cluster
- it should not turn into a second separator/FM rewrite

#### 5. Internal-header cleanup is minor and comment-oriented

Day 7 did not find a large internal-header redesign need.

The main residual cleanup notes are:

- `src/sparse_graph_internal.h` already describes the live ownership split well
- the stronger cleanup need is comment accuracy in `src/sparse_graph.c`
- the section banner in `src/sparse_graph.c` still says:
  - `Uncoarsening + vertex-separator extraction`
  even though separator extraction now lives in
  `src/sparse_graph_separator.c`

Interpretation:

- Day 8 should include small comment/section-heading cleanup
- no broad shared-header contraction or expansion is needed first

#### 6. Day 8 target set is now concrete

Bounded Day 8 cleanup targets:

- simplify the remaining config parsing blocks inside `graph_uncoarsen(...)`
- make the per-level FM runtime/dispatched-orchestration structure easier to
  read without changing behavior
- clean up the residual file-level and section-level ownership comments
- preserve the existing helper layering for:
  - `partition_once(...)`
  - `sparse_graph_partition(...)`
  - forced-HEM retry

Explicit non-goals for Day 8:

- no new graph module beyond the Day 6 split
- no behavior change in finest/intermediate FM strategy selection
- no retry-policy semantic change
- no public API/header change

Interpretation:

- the remaining graph cleanup queue is now concrete rather than generic
- Sprint 44 can finish the Phase-2 graph pass with one bounded residual
  cleanup batch

## Day 8

**Objective:** Land one bounded residual cleanup pass in `src/sparse_graph.c`
so the remaining Phase-2 graph file reads like orchestration-owned runtime
composition rather than another hidden subsystem, while preserving the Day 7
keep-local boundary for retry/fallback glue and top-level partition control.

### Commands Run

1. Re-read the Sprint 44 Day 8 plan section and the Day 7 audit:
   - `sed -n '92,126p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_44/artifacts/day7-runtime-strategy-and-orchestration-audit.md`
2. Re-read the live residual orchestration path before editing:
   - `sed -n '140,520p' src/sparse_graph.c`
   - `rg -n "SPARSE_FM_FINEST_PASSES|SPARSE_FM_FINEST_STRATEGY|SPARSE_FM_ENSEMBLE_STRATEGIES|SPARSE_FM_ENSEMBLE_DEBUG|SPARSE_FM_INTERMEDIATE_PASSES|SPARSE_FM_THICK_RESTART_DEBUG|graph_uncoarsen|partition_once|sparse_graph_partition" src/sparse_graph.c`
3. Re-read the shared graph internal declarations for residual ownership
   context:
   - `sed -n '320,470p' src/sparse_graph_internal.h`
4. Edit the residual orchestration path in `src/sparse_graph.c`:
   - consolidate local env-var parsing / runtime-selection helpers
   - update residual section/ownership wording
5. Run the required validation gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 8 Findings

#### 1. The best remaining cleanup really was local orchestration parsing consolidation

Day 8 did **not** uncover another extractable module.

The real cleanup seam was the repeated env-var parsing and dispatch setup around
`graph_uncoarsen(...)`:

- finest-pass count parsing
- finest-strategy parsing
- ensemble-strategy list parsing
- ensemble debug flag reads
- intermediate-pass count parsing
- thick-restart debug flag reads

Those blocks are now routed through small local residual helpers instead of
being open-coded inline.

Interpretation:

- the Day 7 audit was accurate
- the right Phase-2 closeout move was simplification in place, not another file
  split

#### 2. `src/sparse_graph.c` now reads more clearly as orchestration-owned runtime composition

The Day 8 batch introduced a small local helper layer for the remaining
orchestration path:

- `graph_parse_env_int_range(...)`
- `graph_parse_finest_strategy(...)`
- `graph_parse_ensemble_strategy_list(...)`
- `graph_env_flag_enabled(...)`
- `graph_uncoarsen_level_passes(...)`
- `graph_uncoarsen_runtime_for_level(...)`

These helpers now own the repetitive residual decisions around:

- finest vs intermediate pass counts
- baseline/FIFO/annealing/thick-restart/ensemble dispatch selection
- debug-flag enablement
- per-level FM runtime setup

Interpretation:

- the file now expresses the residual queue as:
  - uncoarsening/runtime composition
  - top-level partition orchestration
  - retry/fallback glue
- it no longer reads like it still owns a hidden FM parser subsystem

#### 3. Ownership wording is now aligned with the live Sprint 44 split

Day 8 also corrected the strongest stale wording in the residual file:

- the old section banner:
  - `Uncoarsening + vertex-separator extraction`
- now reads as:
  - `Uncoarsening + residual orchestration runtime`

Interpretation:

- Day 5 and Day 6 already moved FM and separator ownership
- Day 8 now makes the residual file say that directly instead of preserving
  stale pre-extraction wording

#### 4. The keep-local orchestration boundary held

Day 8 intentionally did **not** move or redesign:

- `graph_uncoarsen(...)`
- `graph_hierarchy_coarsest(...)`
- `graph_partition_seed_coarsest(...)`
- `graph_partition_count_separator_vertices(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`

It also did **not** change:

- retry-policy semantics
- finest/intermediate FM strategy behavior
- public graph APIs
- shared graph header shape

Interpretation:

- the cleanup stayed inside the intended Sprint 44 Day 8 boundary
- Sprint 44 did not reopen Phase-1/Phase-2 design decisions that were already
  settled

#### 5. The required validation gate passed after the residual cleanup

Because `src/sparse_graph.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

Interpretation:

- the residual orchestration cleanup is validated as behavior-preserving
- Sprint 44 can now move on from graph-structure cleanup into the focused test
  maintainability work on the preserved Day 13-style quality baseline

## Day 9

**Objective:** Audit the post-Day-8 graph subsystem so the live ownership split
and existing regression surface are reviewed before the sprint shifts toward
large-test maintainability work, then define a bounded Day 10 graph-test batch
that protects the new Sprint 44 boundaries without overfitting to private
implementation details.

### Commands Run

1. Re-read the Sprint 44 Day 9 plan section and the earlier graph design notes:
   - `sed -n '260,330p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_44/artifacts/day4-separator-runtime-and-large-test-design.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_44/artifacts/day7-runtime-strategy-and-orchestration-audit.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_44/artifacts/day8-runtime-parsing-and-orchestration-cleanup.md`
2. Refresh the live Phase-2 graph ownership split:
   - `wc -l src/sparse_graph.c src/sparse_graph_core.c src/sparse_graph_coarsen.c src/sparse_graph_bisect.c src/sparse_graph_refine.c src/sparse_graph_separator.c`
   - `sed -n '1,260p' src/sparse_graph_internal.h`
   - `sed -n '1,260p' src/sparse_graph.c`
   - `sed -n '1,220p' src/sparse_graph_refine.c`
   - `sed -n '1,220p' src/sparse_graph_separator.c`
3. Refresh the current graph-focused regression surface:
   - `wc -l tests/test_graph.c tests/test_graph_fm_buckets.c tests/test_reorder_nd.c tests/test_reorder_amd_qg.c`
   - `rg -n "graph_refine_fm|graph_edge_separator_to_vertex_separator|graph_uncoarsen|SPARSE_FM_|SPARSE_ND_|gggp|brute|subgraph|separator|retry|forced_hem|balanced_boundary|per_vertex" tests/test_graph.c tests/test_graph_fm_buckets.c tests/test_reorder_nd.c tests/test_reorder_amd_qg.c`
4. Re-read the highest-signal direct graph seam tests:
   - `sed -n '1828,1935p' tests/test_graph.c`
   - `sed -n '1180,1550p' tests/test_graph.c`

### Day 9 Findings

#### 1. The live graph ownership split is now clean enough that Day 10 should protect boundaries, not implementation details

The current graph subsystem is now materially decomposed:

- `src/sparse_graph_core.c` = `264`
- `src/sparse_graph_coarsen.c` = `597`
- `src/sparse_graph_bisect.c` = `521`
- `src/sparse_graph_refine.c` = `619`
- `src/sparse_graph_separator.c` = `311`
- residual `src/sparse_graph.c` = `801`

The remaining ownership model is also explicit in the live file headers:

- core:
  - graph construction / free / induced subgraph
- coarsen:
  - hierarchy build
  - HEM/HCC coarsening
  - temporary forced-HEM override seam
- bisect:
  - coarsest split
  - brute/GGGP/spectral support
- refine:
  - FM runtime state
  - FM parser helpers
  - cut-weight evaluation
  - gain buckets
  - `graph_refine_fm(...)`
- separator:
  - separator policy parsers
  - separator scoring helpers
  - `graph_edge_separator_to_vertex_separator(...)`
- residual orchestration:
  - `graph_uncoarsen(...)`
  - top-level partition composition
  - retry/fallback glue

Interpretation:

- Sprint 44 has now crossed from "monolith being split" into "bounded module
  seams need protection"
- Day 10 should prefer behavior-level tests that pin cross-module contracts
  rather than direct unit tests of local residual helpers

#### 2. FM and bisection already have strong behavior-level coverage after the extraction

The current graph-focused test surface already protects several major seams:

- core/subgraph seam:
  - `test_graph_subgraph_argument_validation`
  - `test_graph_subgraph_path_slice`
- bisection dispatch/fallback seam:
  - `test_bisect_forced_gggp_small_graph`
  - `test_bisect_forced_brute_large_graph_falls_back_to_gggp`
  - spectral fallback coverage on star/small/disconnected fixtures
- FM direct behavior:
  - `test_fm_reduces_checkerboard_cut`
  - `test_fm_optimal_partition_no_regress`
  - `test_fm_null_args`
- FM/orchestration env dispatch:
  - `test_fm_intermediate_passes_smoke`
  - `test_finest_fm_strategy_fifo_smoke`
  - `test_finest_fm_gain_noise_formal_disrupts_baseline`
  - `test_finest_fm_ensemble_corpus_safety`
  - `test_finest_fm_ensemble_deterministic`
- end-to-end ND and fill-quality coverage:
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder_amd_qg.c`

Interpretation:

- Day 10 does not need to invent new FM-private tests just because the FM file
  moved to `src/sparse_graph_refine.c`
- the extracted FM seam is already mostly protected through direct behavior
  tests and end-to-end environment-driven coverage

#### 3. The strongest remaining direct gap is separator-policy coverage, not FM or coarsening

The extracted separator module now owns:

- separator strategy parsing
- separator weight parsing
- per-vertex selection/scoring
- `graph_edge_separator_to_vertex_separator(...)`

But the direct separator-helper coverage in `tests/test_graph.c` is still
narrow:

- `test_edge_to_vertex_separator_smaller_side`
- `test_edge_to_vertex_separator_null_args`

There is broader end-to-end separator-policy evidence through:

- `test_per_vertex_fixed_k_differs_from_dynamic_k`
- the ND fill-differentiation tests in `tests/test_reorder_nd.c`

Even so, the direct helper seam still lacks one small behavior-level policy
test after the module move.

Interpretation:

- the best Day 10 addition is a direct separator-policy contract test, not
  another FM or bisection test
- that gives the extracted `src/sparse_graph_separator.c` a stronger
  behavior-level anchor without exposing its private local helpers

#### 4. Residual orchestration already has the right kind of proof, so Day 10 should avoid private `graph_uncoarsen(...)` tests

The residual orchestration file now owns:

- `graph_uncoarsen(...)`
- `partition_once(...)`
- `sparse_graph_partition(...)`
- sep=`0` retry / forced-HEM composition

The current test surface already exercises those behaviors where they matter:

- structural partition validity on grids, meshes, cliques, and corpus fixtures
- HCC sep=`0` protection through the bcsstk14 contract
- spectral fallback and coarsest-dispatch behavior
- multiple non-default FM strategy/env paths

What is still missing is **not** a direct private-helper unit test. A new
`graph_uncoarsen(...)` choreography test would mostly pin internal buffer
arrangement and per-level dispatch details that Sprint 44 is not trying to
freeze.

Interpretation:

- Day 10 should keep residual coverage end-to-end and behavior-oriented
- it should not add private orchestration tests just because the file is now
  smaller

#### 5. The Day 10 graph-test batch is now concrete and intentionally small

Best Day 10 targets:

1. `tests/test_graph.c`
   - add one direct separator-policy test that exercises the extracted
     separator seam beyond the existing smaller-side default
   - strongest candidate:
     - `balanced_boundary` on a small crafted graph/partition where its choice
       is observably different from plain smaller-side lifting while still
       preserving the public partition invariant
2. `tests/test_graph.c`
   - add one compact end-to-end orchestration smoke that composes the extracted
     FM + separator modules through the residual `sparse_graph_partition(...)`
     path under a non-default but stable env configuration
   - goal:
     - protect the post-Day-8 module interaction boundary without overfitting
       to private helper structure

Explicit non-goals for Day 10:

- no new graph-specific test binary
- no direct tests of static/local parser helpers
- no attempt to pin exact intermediate FM runtime state
- no broad expansion of ND fill-quality corpus tests

Interpretation:

- the graph seam-test batch can now stay small, high-signal, and consistent
  with Sprint 44's behavior-first rule

## Day 10

**Objective:** Implement the bounded graph seam-test batch defined on Day 9 so
the extracted separator-policy seam and the post-Day-8 orchestration path gain
explicit regression coverage without introducing private helper tests or
another graph-specific test surface.

### Commands Run

1. Re-read the Sprint 44 Day 10 plan section and the Day 9 audit:
   - `sed -n '290,350p' docs/planning/EPIC_4/SPRINT_44/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_44/artifacts/day9-post-cleanup-graph-audit-and-test-design.md`
2. Re-read the current graph-test seam region:
   - `sed -n '1840,1935p' tests/test_graph.c`
   - `sed -n '2570,2765p' tests/test_graph.c`
   - `rg -n "edge_to_vertex_separator|balanced_boundary|fifo|test_graph" tests/test_graph.c`
3. Implement the bounded Day 10 test batch in `tests/test_graph.c`:
   - add one direct separator-policy contract test
   - add one compact end-to-end orchestration smoke
4. Run the required validation gate:
   - `make format`
   - `make lint`
   - `make test`

### Day 10 Findings

#### 1. The extracted separator seam now has a direct non-default policy test

Day 10 added a new small crafted-fixture test:

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`

The fixture is intentionally asymmetric:

- side 0 is smaller by weight
- side 0 has four boundary vertices
- side 1 has one boundary vertex

That makes the separator-policy choice observable:

- plain smaller-side lifting would lift the four boundary vertices on side 0
- `balanced_boundary` instead lifts the single boundary vertex on side 1 while
  still preserving a balanced post-lift split

The test asserts only behavior-level outcomes:

- `graph_edge_separator_to_vertex_separator(...)` succeeds
- the partition invariant still holds
- exactly one separator vertex is produced
- the chosen lifted vertex is the unique side-1 boundary vertex

Interpretation:

- the extracted `src/sparse_graph_separator.c` seam now has a stronger direct
  contract than the previous default-only smaller-side test
- the test protects policy behavior without pinning private separator scoring
  helpers or parser internals

#### 2. The residual orchestration path now has one compact post-split smoke under non-default FM + separator config

Day 10 also added:

- `test_partition_fifo_balanced_boundary_smoke`

This test runs the full `sparse_graph_partition(...)` path on a 10×10 grid
under a deliberately non-default but stable configuration:

- `SPARSE_ND_COARSENING=heavy_edge`
- `SPARSE_FM_FINEST_STRATEGY=fifo`
- `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary`

The test asserts only structural contracts:

- partition succeeds
- partition invariant holds
- separator count stays in a broad nondegenerate range
- both interior sides remain meaningfully populated

Interpretation:

- this gives Sprint 44 one direct post-split orchestration smoke that composes:
  - extracted coarsening
  - extracted FM refinement
  - extracted separator lifting
  - residual uncoarsening / partition glue
- it avoids freezing any private `graph_uncoarsen(...)` implementation detail

#### 3. The Day 9 keep-small boundary held

Day 10 intentionally stayed within the Day 9 design:

- only `tests/test_graph.c` changed
- no new graph-specific test binary was introduced
- no direct tests of static parser helpers were added
- no FM-private or bisection-private unit-test wave was reopened
- no production `src/` files changed

Interpretation:

- Sprint 44 added load-bearing regression protection without turning the graph
  test surface into another exploratory refactor

#### 4. The required validation gate passed, and the new graph tests passed inside the authoritative suite

Because `tests/test_graph.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative `make test` sweep also explicitly showed the new additions in
`test_graph`:

- `test_edge_to_vertex_separator_balanced_boundary_prefers_smaller_boundary`
- `test_partition_fifo_balanced_boundary_smoke`

Interpretation:

- the new graph seam tests are validated on the same maintained gate as the
  rest of Epic 4
- Sprint 44 can now shift from graph seam protection into the large-test
  maintainability batch with the post-split graph surface explicitly covered
