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
