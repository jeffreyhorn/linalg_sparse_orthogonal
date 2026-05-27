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
