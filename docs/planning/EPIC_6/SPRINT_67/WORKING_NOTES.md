# Sprint 67 Working Notes

## Day 1 - Scope Audit & Maintainability Baseline Setup

### Goal

Freeze the Sprint 67 starting point before implementation work begins by
reconfirming the inherited Sprint 66 contract, the preserved reviewed
baseline, the strongest live large-source maintainability hotspots, and the
most important docs/build/implementation/proof surfaces the sprint will touch
next.

### Actions

1. Re-read the Sprint 67 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 66 retrospective, and
   the Sprint 66 Day 14 closeout artifact.
2. Re-read the landed Sprint 67 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - residual hotspot audit
   - graph/reorder decomposition
   - CSC/iterative residual decomposition
   - comment/chronology cleanup
   - build/regression alignment
   - validation and closeout
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Materialized the reviewed CMake parity tree locally and rechecked:
   - `make quality-review-cmake-compile`
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 67 touch surfaces directly from the
   live tree across:
   - maintained truth/build surfaces
   - public coordination headers
   - graph/reorder implementation hotspots
   - CSC/iterative implementation hotspots
   - proof/adoption/regression support surfaces

### Findings

#### 1. Sprint 67 starts from the Sprint 66 productization close, not from renewed build or platform churn

Sprint 66 already closed the main packaging/productization contradiction and
made the platform truthfulness split explicit. That means Sprint 67 is not
reopening:

- static-first package-shape clarification
- install/export/productization tightening
- narrow ABI/version-story clarification
- workflow and CI contract reconciliation
- the deferred platform residual queue except where a touched maintainability
  seam proves it is truly necessary

Interpretation:

- Sprint 67 is the first post-productization Epic 6 sprint centered on source
  ownership and large-file maintainability again
- packaging/platform/build surfaces remain support surfaces only, not the
  implementation center

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 67 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The Day 1 supporting rechecks are now explicit:

- `make -n quality-review-full` still expands to the expected reviewed local
  quality path followed by the reviewed CMake parity path
- `make quality-review-cmake-compile` completed cleanly and re-established the
  local reviewed CMake tree before the parity count was recorded

Interpretation:

- Sprint 67 inherits the exact same reviewed baseline story as the Sprint 66
  close
- maintainability work does not get a weaker truth surface just because the
  main sprint topic is decomposition rather than new behavior

#### 3. The highest-value Sprint 67 problem is concentrated in mixed-ownership large-source seams, not in generic repo-wide cleanup

The live repo shows the strongest pressure in:

- graph/reorder orchestration mixed with family-local helpers
- CSC direct-family residuals still carrying too much local orchestration
- iterative/eigensolver residual seams that still compete with coordination in
  large permanent files
- stale sprint-local chronology inside touched permanent implementation
  surfaces

The project-plan scope therefore reduces cleanly to:

1. residual hotspot audit
2. graph/reorder decomposition
3. CSC/iterative residual decomposition
4. comment/chronology cleanup
5. build/regression alignment
6. validation and closeout

Interpretation:

- Sprint 67 should not pretend every large file is equally urgent
- the highest-value work is ownership extraction on the files where
  orchestration, family-local logic, and residual chronology are still
  colliding

#### 4. The strongest live Sprint 67 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained truth/build surfaces:
  - `README.md` = `1020`
  - `docs/maintainer_guide.md` = `548`
  - `CMakeLists.txt` = `413`
  - `Makefile` = `897`
- public coordination headers likely to matter if ownership boundaries move:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_reorder.h` = `186`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest graph/reorder implementation hotspots:
  - `src/sparse_graph.c` = `801`
  - `src/sparse_graph_coarsen.c` = `641`
  - `src/sparse_graph_bisect.c` = `528`
  - `src/sparse_graph_refine.c` = `629`
  - `src/sparse_graph_separator.c` = `297`
  - `src/sparse_reorder_nd.c` = `743`
  - `src/sparse_reorder_amd_qg.c` = `611`
- strongest CSC/iterative residual hotspots:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
- strongest proof/adoption/regression support surfaces:
  - `tests/test_integration.c` = `2367`
  - `tests/test_graph.c` = `2900`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_chol_csc.c` = `4716`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `examples/example_analysis.c` = `210`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`

Interpretation:

- the strongest remaining Epic 6 maintainability pressure is not generic across
  the whole repo
- it is concentrated in a smaller set of orchestration-heavy files and the
  proof surfaces that will have to move with them

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 67 Day 1 confirms the following non-goals:

- no fake maintainability wins that blur real ownership
- no broad feature work disguised as decomposition
- no reopening packaging/platform/build-surface churn unless a touched seam
  truly requires it
- no weakening of the reviewed truthfulness contract
- no broad style-only cleanup wave disconnected from actual ownership seams
- no chronology scrub that removes durable explanations just to erase sprint
  history references

### Day 1 Close

Sprint 67 now starts from one explicit maintainability implementation baseline:

- the Sprint 66 productization close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor is re-established locally at `53`
- the broad Epic 6 maintainability claim has already narrowed to hotspot
  audit, graph/reorder decomposition, CSC/iterative residual decomposition,
  chronology cleanup, and build/regression alignment
- the next step is to rank those live hotspot seams precisely before writing
  the bounded Day 2 validation and Day 3 audit follow-through

## Day 2 - Validation Baseline & Hotspot/Proof Rerun Recheck

### Goal

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 67
decomposition work must preserve before any implementation work lands.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the reviewed baseline wrapper surface:
   - `make -n quality-review-full`
3. Reconfirmed the authoritative validation split for:
   - bounded `*.c` / `*.h` days
   - substantial decomposition/build-alignment days
   - docs-only days
4. Rechecked build-tree availability of the most relevant Sprint 67 proof and
   regression surfaces:
   - graph/reorder proofs
   - CSC proofs
   - iterative/eigensolver proofs
   - representative examples
   - maintained benchmark/reporting surfaces
5. Reconfirmed the strongest likely Sprint 67 touched-surface classes from the
   live branch state after the Day 1 baseline.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 67 start

The strongest local reviewed baseline is still:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 67 starts from the same reviewed truthfulness baseline as the Sprint
  66 close
- decomposition work does not get a weaker local validation contract just
  because the main sprint topic is maintainability rather than new end-user
  behavior

#### 2. The Day 2 authority split is now explicit

The authoritative split for Sprint 67 is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial decomposition, ownership-boundary, or
  build/regression-alignment work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- Sprint 67 should treat ownership-boundary changes as contract-sensitive work,
  not as cheap refactors
- the stronger reviewed baseline remains the default for any change that could
  distort build, proof, or ownership truthfulness

#### 3. The high-signal Sprint 67 rerun set is now fixed around the actual decomposition-risk surface

The high-signal rerun set at Sprint 67 start is:

- cross-family and orchestration proof surfaces:
  - `./build/test_integration`
- graph/reorder family proofs:
  - `./build/test_graph`
  - `./build/test_reorder_nd`
- CSC direct-family proofs:
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- iterative and eigensolver residual proofs:
  - `./build/test_iterative`
  - `./build/test_eigs`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
- maintained benchmark/reporting surfaces likely to matter in alignment work:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

All of those surfaces were present in the current `build/` tree at Day 2.

Interpretation:

- the Sprint 67 rerun set is anchored to the actual decomposition-risk surface
  rather than to every executable in the repo
- maintained benchmark/reporting surfaces remain part of the live Sprint 67
  validation story because later build/regression alignment can still touch
  them indirectly

#### 4. The strongest likely Sprint 67 touch surfaces remain ownership-heavy implementation and proof surfaces, not packaging/productization files

The highest-signal likely Sprint 67 touch surfaces at Day 2 remain:

- implementation hotspots:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- proof/support surfaces:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
- likely coordination headers only if the design proves they must move:
  - `include/sparse_analysis.h`
  - `include/sparse_reorder.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

Interpretation:

- Sprint 67 still starts from a maintainability and ownership surface, not a
  productization or platform surface
- the heaviest likely touched source and proof seams are already explicit
  before the deeper hotspot audit begins

### Day 2 Close

Sprint 67 now has:

- one explicit reviewed validation contract for decomposition work
- one fixed rerun set centered on the actual graph/reorder, CSC, and
  iterative/eigs proof surface
- one clear Day 3 starting point for the residual hotspot audit

## Day 3 - Residual Hotspot Audit

### Goal

Reduce Sprint 67's broad maintainability claim to the live implementation seams
that still have the strongest mixed-ownership and chronology burden after the
earlier Epic 5 and Epic 6 decomposition work.

### Actions

1. Re-read the Day 2 validation baseline and the Sprint 67 plan fence.
2. Re-read representative top-level hotspot files directly:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_iterative.c`
3. Ran targeted `rg` scans across the likely Sprint 67 implementation and
   header surfaces for:
   - `Sprint` / `Day` chronology markers
   - helper ownership signals
   - internal API seams
   - configuration/runtime parser clusters
4. Re-ranked the current large-source set by:
   - mixed ownership pain
   - stale chronology density
   - extraction safety
   - proof burden
   - likely payoff
5. Fixed the likely Day 4 target boundary from the live repo state rather than
   from the project-plan summary.

### Findings

#### 1. The broad "large-source maintainability" problem is now reduced to a small ranked seam map

The live hotspot order is now:

1. graph/reorder orchestration residuals
2. CSC/analysis residuals
3. iterative/eigensolver residuals
4. public coordination-header truth follow-through

Interpretation:

- Sprint 67 should not spread evenly across every remaining large file
- the strongest remaining pain is still where graph partitioning, ND policy,
  retry/fallback glue, and sprint-history commentary remain mixed together

#### 2. Graph/reorder now owns the strongest remaining maintainability seam

The strongest first target is the graph/reorder lane:

- `src/sparse_graph.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Why this lane ranks first:

- the top-level orchestration files still carry the densest remaining sprint
  chronology blocks
- multiple files still mix durable algorithm explanation with landing-history
  notes, retry/fallback glue, parser/runtime state, and owned helper seams
- the proof surface is strong and already well isolated:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`

The strongest first exact seam is now:

- residual uncoarsening / orchestration in `src/sparse_graph.c`
- residual root-policy / profiling / fallback orchestration in
  `src/sparse_reorder_nd.c`

Interpretation:

- Sprint 43 and Sprint 44 already extracted meaningful graph subsystems, but
  the remaining orchestration shells still carry too much accumulated
  chronology and cross-seam policy
- this is the highest-payoff place to continue Phase 3 decomposition

#### 3. CSC/analysis is the strongest second lane, but it is no longer the best first landing

The strongest second target is:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

Why it ranks second instead of first:

- these files are still large, but their permanent file headers already read
  more like owned backend/analysis surfaces than the graph/reorder
  orchestration files do
- the chronology burden is real, especially in `src/sparse_analysis.c`, but it
  is more configuration-compatibility and policy-layer oriented than the graph
  lane's top-level ownership blur
- the proof burden is also broader because the touched behavior fans into:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`

Interpretation:

- CSC/analysis still matters in Sprint 67, but it is the better second landing
  after the graph/reorder seam is made cleaner

#### 4. Iterative/eigensolver residuals are real, but they are a later or narrower target than the headline sizes alone imply

The remaining iterative/eigensolver hotspots are:

- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

Why they rank below graph and CSC:

- both files are large, but they already read more like family-local
  orchestration plus shared helper surfaces than the graph/reorder files do
- the chronology burden is lighter and more localized than in the graph lane
- the proof burden is substantial:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - plus integration spillover if public-entry ownership changes

Interpretation:

- they remain valid Sprint 67 candidates only if the Day 4-5 design proves a
  bounded residual extraction is still justified after the first landing
- they should not displace the graph/reorder lane as the sprint's first target

#### 5. The strongest current contradiction is not raw file size; it is ownership blur plus stale chronology in permanent orchestration files

The current contradictions are:

- durable algorithm explanation mixed with sprint-history narration
- runtime/env-policy parsing mixed with top-level orchestration
- fallback and retry logic mixed with family-local ownership
- previously extracted subsystem boundaries still explained through old
  "Day X" archaeology in permanent files

This is strongest in:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`

Interpretation:

- Sprint 67 should optimize for clearer ownership and less permanent chronology,
  not just smaller line counts
- the most valuable extraction target is the file that still reads least like
  a durable owner and most like a sprint journal

### Day 3 Close

Sprint 67 Day 3 fixes the current ranked target order as:

1. graph/reorder decomposition first
2. CSC/analysis residual decomposition second
3. iterative/eigensolver residuals later or narrower only if still justified

That gives Day 4 one explicit job:

- turn the graph/reorder lane into the exact first landing fence instead of
  keeping it as a general hotspot bucket

## Day 4 - Hotspot Follow-Through & First-Landing Boundary

### Goal

Turn the Day 3 hotspot ranking into one exact first implementation fence so
Sprint 67 starts from a bounded graph/reorder landing instead of a generic
cleanup target set.

### Actions

1. Re-read the Day 3 ranked hotspot audit and the Sprint 67 plan fence.
2. Re-read the current internal ownership contract in:
   - `src/sparse_graph_internal.h`
3. Re-read the strongest likely first-landing files directly:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
4. Re-read the nearby already-extracted graph subsystem files to confirm what
   should stay out of the first landing:
   - `src/sparse_graph_coarsen.c`
   - `src/sparse_graph_refine.c`
5. Fixed the exact first-landing boundary from the live repo state:
   - required implementation surfaces
   - likely proof/support surfaces
   - explicit non-touch set

### Findings

#### 1. Sprint 67 now has one exact first landing boundary instead of a generic graph/reorder shortlist

The exact first landing is now fixed to:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

This is the right first batch because those two files still own the strongest
combination of:

- top-level orchestration
- retry/fallback glue
- runtime/env-policy handling
- residual chronology in permanent implementation surfaces

Interpretation:

- Sprint 67 should not start by touching every graph file that still looks big
- it should start by shrinking the remaining orchestration shells that still
  read least like durable owners

#### 2. The already-extracted graph subsystem files are support context, not the first implementation center

The following files now read as already-separated subsystem owners rather than
the best first extraction targets:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`

Why they stay out of the first landing:

- each already has a narrower ownership statement than `src/sparse_graph.c`
- the residual chronology burden is lower than in the orchestration shells
- widening into them immediately would blur whether Day 6-7 is still an
  ownership extraction or just a broad graph rewrite

Interpretation:

- these files remain relevant context for the Day 5 design
- they should move only if the landed first batch proves a truly necessary
  support edit

#### 3. The likely proof home is now bounded and explicit

The strongest proof surfaces for the first landing are now:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Likely support only if the design actually forces it:

- `src/sparse_graph_internal.h`
- `tests/test_integration.c`

Interpretation:

- the first landing can stay family-local if the design is disciplined
- cross-family proof should stay optional rather than assumed

#### 4. CSC/analysis and iterative/eigensolver remain explicitly outside the first landing fence

The following stay out of the first landing:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public coordination headers unless the landed graph/reorder design proves
  they truly need moving

Interpretation:

- Sprint 67 still has a real second lane after graph/reorder
- but the first implementation batch should not widen into CSC or iterative
  work just because those files are also large

#### 5. The strongest remaining Day 4 safety rule is "do not widen from orchestration cleanup into graph-family redesign"

The current non-widening fence is now:

- do not reopen graph construction ownership in `src/sparse_graph_core.c`
- do not reopen heavy-edge/HCC internals in `src/sparse_graph_coarsen.c`
- do not reopen FM-local runtime state in `src/sparse_graph_refine.c`
- do not reopen separator-policy surfaces in `src/sparse_graph_separator.c`
- do not widen into CSC/iterative/eigensolver work
- do not widen into packaging/platform/build churn

Interpretation:

- the first landing should optimize for clearer ownership in the remaining
  orchestration shells
- it should not behave like a fresh architecture phase for the whole graph
  subsystem

### Day 4 Close

Sprint 67 Day 4 now fixes the implementation order explicitly:

1. first landing:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
2. likely proof home:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
3. support only if needed:
   - `src/sparse_graph_internal.h`
   - `tests/test_integration.c`
4. later/deferred:
   - CSC/analysis residual decomposition
   - iterative/eigensolver residual decomposition

That gives Day 5 one exact job:

- define the ownership and extraction contract for the bounded
  `sparse_graph.c` / `sparse_reorder_nd.c` landing

## Day 5 - Graph/Reorder Decomposition Design

### Goal

Turn the Day 4 first-landing fence into one explicit ownership and extraction
contract so Day 6 can shrink the remaining graph/reorder orchestration shells
without widening into graph-family redesign.

### Actions

1. Re-read the Day 4 boundary artifact and the current Sprint 67 working
   baseline to keep the first-landing fence exact.
2. Re-read the live orchestration seams in:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
3. Re-read the already-extracted support context in:
   - `src/sparse_graph_internal.h`
   - `src/sparse_graph_coarsen.c`
   - `src/sparse_graph_refine.c`
4. Mapped the current helper and orchestration ownership split across the two
   first-landing files:
   - environment/runtime parsing
   - profiling/runtime accounting
   - recursive/top-level orchestration
   - uncoarsening/retry/fallback glue
   - public entry-point ownership
5. Reduced that seam map to one bounded Day 6-7 implementation contract and
   one explicit non-widening fence.

### Findings

#### 1. Sprint 67 now has one exact ownership contract for the first graph/reorder landing

The first landing should not try to "finish" graph decomposition across the
whole family. It should instead make the two remaining orchestration shells
read like durable owners:

- `src/sparse_graph.c` should converge toward:
  - graph partition top-level orchestration
  - coarsest-level seed selection/retry ownership
  - uncoarsening orchestration
- `src/sparse_reorder_nd.c` should converge toward:
  - ND policy normalization at the public boundary
  - ND recursion/top-level orchestration
  - ND profiling publication at the public boundary

Interpretation:

- Day 6 should optimize for cleaner ownership in the remaining orchestration
  shells
- it should not behave like a generic graph cleanup pass

#### 2. `src/sparse_graph.c` is now clearly the graph orchestration shell, not the home for every graph support helper

The live file still owns two different categories of logic:

- durable orchestration that belongs there:
  - `graph_uncoarsen(...)`
  - `graph_hierarchy_coarsest(...)`
  - `graph_partition_seed_coarsest(...)`
  - `graph_partition_should_retry_with_forced_hem(...)`
  - `partition_once(...)`
  - `sparse_graph_partition(...)`
- support/policy/runtime helpers that are weaker long-term owners:
  - `graph_parse_env_int_range(...)`
  - `graph_parse_finest_strategy(...)`
  - `graph_parse_ensemble_strategy_list(...)`
  - `graph_env_flag_enabled(...)`
  - `graph_uncoarsen_level_passes(...)`
  - `graph_uncoarsen_runtime_for_level(...)`

The Day 5 design implication is now explicit:

- the file should keep graph partition and uncoarsening orchestration
- support parsing/runtime-accounting helpers should move only if the Day 6
  landing needs that extraction to make the orchestration shell materially
  clearer

#### 3. `src/sparse_reorder_nd.c` is now clearly the ND orchestration shell, not the best home for every compatibility parser

The live file still owns three mixed layers:

- durable ND owners:
  - `nd_recurse(...)`
  - `sparse_reorder_nd_with_policy(...)`
  - `sparse_reorder_nd(...)`
- likely separable support helpers:
  - `nd_emit_natural(...)`
  - `nd_subgraph_to_sparse(...)`
- compatibility/policy parsing and profiling helpers:
  - `parse_nd_root_bisect_strategy_compat_override(...)`
  - `parse_nd_coarsening_compat_override(...)`
  - `parse_nd_coarsest_bisection_compat_override(...)`
  - `parse_nd_root_bisect_max_n_compat_override(...)`
  - `parse_nd_coarsen_floor_ratio_compat_override(...)`
  - `parse_nd_coarsening_cv_fallthrough_compat_override(...)`
  - `parse_nd_sep_lift_strategy_compat_override(...)`
  - `parse_nd_sep_lift_weight_compat_override(...)`
  - `sparse_reorder_nd_default_policy(...)`

The Day 5 design implication is now explicit:

- the file should keep ND public-boundary normalization and recursive
  orchestration
- compatibility parsers, leaf/base-case helpers, or local profiling helpers
  should move only where that materially reduces chronology and mixed ownership

#### 4. The Day 6-7 touched-file fence is now fixed and small

Required first-batch implementation surfaces:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

Likely proof home:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Support only if the landed extraction truly needs it:

- `src/sparse_graph_internal.h`
- `tests/test_integration.c`

Interpretation:

- the first implementation batch can still stay family-local by default
- support/header/integration widening is now explicitly conditional

#### 5. The explicit non-widening fence is now strong enough to keep the landing honest

The first graph/reorder landing should not widen into:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public coordination headers unless the landed extraction truly forces it
- packaging/platform/build churn

Interpretation:

- Sprint 67 still has real later lanes after Day 6-7
- but the first implementation success condition is a clearer ownership story
  in the two remaining orchestration shells, not broader source churn

### Day 5 Close

Sprint 67 Day 5 now fixes the first implementation contract exactly:

1. required first batch:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
2. likely proof home:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
3. support only if needed:
   - `src/sparse_graph_internal.h`
   - `tests/test_integration.c`
4. explicit non-touch set:
   - already-extracted graph subsystem files
   - `src/sparse_reorder_amd_qg.c`
   - CSC/analysis residuals
   - iterative/eigensolver residuals

That gives Day 6 one exact job:

- land one bounded graph/reorder ownership extraction batch without widening
  into graph-family redesign

## Day 6 - Graph/Reorder Ownership Extraction Batch 1

### Goal

Land the first bounded ownership extraction batch inside the Day 5 fence by
shrinking the two remaining graph/reorder orchestration shells without
widening into the already-extracted graph subsystem files.

### Actions

1. Re-read the Day 5 design contract and the live Day 6 target files:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
2. Identified the strongest mixed-ownership seams that could be extracted
   locally without creating a broader graph-family rewrite:
   - uncoarsen env/runtime setup in `src/sparse_graph.c`
   - leaf/partition/side-recursion support glue inside
     `src/sparse_reorder_nd.c`
3. Landed the bounded local helper extraction:
   - `graph_uncoarsen_options_t`
   - `graph_uncoarsen_options_from_env(...)`
   - `nd_emit_leaf_amd(...)`
   - `nd_partition_current_graph(...)`
   - `nd_recurse_side(...)`
4. Verified that the batch stayed inside the Day 5 fence:
   - no public-header widening
   - no already-extracted graph subsystem edits
   - no CSC/analysis or iterative/eigensolver widening
5. Ran the required validation and reviewed-quality paths.

### Findings

#### 1. `src/sparse_graph.c` now reads more like an orchestration shell and less like a mixed env-parser/runtime bucket

The Day 6 batch introduced:

- `graph_uncoarsen_options_t`
- `graph_uncoarsen_options_from_env(...)`

This consolidates the uncoarsening control-plane selection that was previously
spread across `graph_uncoarsen(...)`:

- finest/intermediate pass counts
- finest FM strategy
- annealing / thick-restart / gain-noise schedule choices
- ensemble strategy list and debug flag

Interpretation:

- `graph_uncoarsen(...)` now spends more of its visible surface on level-walk
  orchestration
- env/runtime selection is still local to the file, but it is no longer mixed
  directly into the orchestration body

#### 2. `src/sparse_reorder_nd.c` now separates recursive orchestration from three support responsibilities that previously lived inline

The Day 6 batch extracted:

- `nd_emit_leaf_amd(...)`
- `nd_partition_current_graph(...)`
- `nd_recurse_side(...)`

That removes three support responsibilities from the center of `nd_recurse(...)`:

- leaf AMD materialization/splice
- root spectral-versus-multilevel partition dispatch
- repeated side-subgraph build/map/recurse glue

Interpretation:

- `nd_recurse(...)` now reads more directly as ND recursion ownership
- the support helpers stay in the same file, preserving the bounded fence,
  while making the recursive driver itself materially smaller and clearer

#### 3. The first landed extraction stayed inside the exact Day 5 fence

Touched implementation surfaces:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`

The batch did not widen into:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

Interpretation:

- the landed batch is a real ownership extraction, not a disguised broad
  maintainability wave
- proof and support surfaces remain available for later follow-through if a
  subsequent batch truly needs them

#### 4. The reviewed baseline stayed intact after the ownership extraction

Because `*.c` changed, the required validation set was:

- `make format`
- `make lint`
- `make test`

And because this was substantial decomposition work on orchestration-heavy
files, the stronger reviewed path was also run:

- `make quality-review-full`

Interpretation:

- Sprint 67 maintainability work keeps the same reviewed truthfulness bar as
  the earlier Epic 6 code sprints
- the Day 6 extraction is validated as behavior-preserving, not just
  structurally cleaner

### Day 6 Close

Sprint 67 Day 6 now lands the first bounded graph/reorder ownership extraction:

1. `src/sparse_graph.c`
   - uncoarsen control-plane parsing/runtime setup is centralized into one
     local options seam
2. `src/sparse_reorder_nd.c`
   - leaf handling, partition dispatch, and side recursion glue are extracted
     out of the main recursive driver
3. the batch stayed inside the two-file first-landing fence
4. the full required validation and reviewed gate passed

That gives Day 7 one exact follow-through job:

- rerank the residual graph/reorder ownership seam after the first landed
  extraction and decide whether a second bounded graph batch is still justified

## Day 7 - Post-Landing Audit & Residual Rerank

### Goal

Rerank the live maintainability seams after the Day 6 graph/reorder landing and
decide whether Sprint 67 should spend another bounded batch inside the graph
lane or pivot to the stronger remaining shared-policy/CSC lane.

### Actions

1. Re-read the Day 6 landed artifact and the live Day 6 branch state.
2. Re-read the post-landing versions of:
   - `src/sparse_graph.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_analysis.c`
3. Rechecked the remaining mixed-ownership surfaces against the Sprint 67
   maintainability target:
   - graph partition retry/fallback orchestration
   - ND policy compatibility parsing
   - shared analysis/reorder ND policy normalization
4. Compared the residual graph/reorder seam against the next-ranked CSC/analysis
   seam to see which now carries the stronger contradiction.
5. Fixed the exact next target in writing.

### Findings

#### 1. The Day 6 landing closed the strongest pure graph/reorder ownership contradiction

After Day 6:

- `src/sparse_graph.c` reads more clearly as:
  - uncoarsening orchestration
  - partition sequencing
  - retry/fallback ownership
- `src/sparse_reorder_nd.c` reads more clearly as:
  - ND recursive driver
  - public reorder entry
  - local support helpers around the recursive path

Interpretation:

- the broad "remaining graph/reorder orchestration shell" problem is no longer
  the strongest maintainability seam on the branch
- a second graph-only batch is no longer automatically justified just because
  these files are still non-trivial

#### 2. The strongest remaining contradiction has shifted into shared ND policy normalization, not deeper graph extraction

The live strongest residual seam is now the duplicated ND compatibility-policy
surface split across:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

Why this now ranks above a second graph-only batch:

- both files still own parallel env-var compatibility parsers for:
  - root-bisect mode
  - coarsening mode
  - coarsest-bisection mode
  - root-bisect max-n
  - coarsen floor ratio
  - coarsening CV fallthrough
  - separator-lift strategy
  - separator-lift weight
- that duplication is a stronger remaining ownership contradiction than the
  smaller local retry/fallback seam still left in `src/sparse_graph.c`
- it already touches the CSC/analysis lane that Day 3 ranked second

Interpretation:

- the next maintainability win is no longer "more graph extraction"
- it is convergence of the shared ND policy normalization story behind the
  public analysis/reorder boundary

#### 3. `src/sparse_graph.c` still has residual glue, but it is now a lower-priority local seam rather than the sprint center

The main residual graph-local seam now is:

- `partition_once(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `sparse_graph_partition(...)`

That remains real because retry/fallback glue still lives beside the partition
orchestration shell, but it is now:

- more local
- lower-risk
- less contradictory than the duplicated ND policy compatibility path

Interpretation:

- this graph-local seam becomes support/deferred context rather than the next
  mandatory batch
- it should move later only if a shared-policy landing proves it necessary

#### 4. The exact next target is now the shared ND policy / CSC-analysis seam

The strongest next batch is now:

- shared ND policy normalization across `src/sparse_reorder_nd.c` and
  `src/sparse_analysis.c`

Likely touched surfaces:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

Likely proof home:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

Support only if the landing truly needs it:

- `src/sparse_reorder_nd_internal.h`
- `include/sparse_analysis.h`

Interpretation:

- Sprint 67’s second code lane now converges naturally into the CSC/analysis
  track instead of forcing a fake second graph batch
- this still stays consistent with the Day 3 ranking: graph first, CSC/analysis
  second

#### 5. The non-widening fence stays explicit even after the rerank

The next landing should still not widen into:

- already-extracted graph subsystem files
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public API redesign
- packaging/platform/build churn

Interpretation:

- the rerank changes target order, not the bounded-sprint safety contract
- the next step should still be a small ownership convergence batch, not a
  broad cross-family rewrite

### Day 7 Close

Sprint 67 Day 7 now fixes the post-Day-6 order explicitly:

1. graph/reorder first landing:
   - closed the strongest pure graph/reorder contradiction
2. strongest remaining seam:
   - shared ND policy normalization across `src/sparse_reorder_nd.c` and
     `src/sparse_analysis.c`
3. graph residuals now lower-priority/local:
   - retry/fallback glue in `src/sparse_graph.c`
4. likely next proof home:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`

That gives Day 8 one exact job:

- define the bounded shared ND policy / CSC-analysis convergence design instead
  of forcing a second graph-only batch

## Day 8 - Shared ND Policy Convergence Design

### Goal

Define the bounded convergence design for the strongest remaining Sprint 67
maintainability seam: duplicated ND compatibility-policy normalization across
`src/sparse_reorder_nd.c` and `src/sparse_analysis.c`.

### Actions

1. Re-read the Day 7 rerank artifact and the live policy surfaces in:
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
   - `include/sparse_analysis.h`
2. Compared the duplicated parser/default-policy responsibilities across the
   public analysis path and the direct ND reorder path.
3. Rechecked the live proof homes that already exercise the typed-policy versus
   compatibility-override contract:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`
4. Reduced the seam to one bounded design fence:
   - one shared internal ND policy normalization owner
   - preserved public analysis API surface
   - preserved direct `sparse_reorder_nd(...)` compatibility behavior
5. Fixed the Day 9-10 file fence and non-widening contract in writing.

### Findings

#### 1. Sprint 67 now has one exact second-lane design target

The strongest remaining maintainability seam is no longer generic CSC work.
It is the shared ND policy normalization story split across:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`

The exact design target is now:

- one internal owner for ND compatibility parsing and default policy
  normalization
- two consumers:
  - public repeated-run analysis path
  - direct `sparse_reorder_nd(...)` path

Interpretation:

- Day 9 should not redesign ND behavior
- it should reduce duplicated policy ownership while preserving the shipped
  analysis/reorder contract

#### 2. The natural shared owner is an internal ND-policy helper surface, not a public API move

The duplicated logic currently covers:

- root-bisect mode
- coarsening mode
- coarsest-bisection mode
- root-bisect max-n
- coarsen floor ratio
- coarsening CV fallthrough
- separator-lift strategy
- separator-lift weight

The Day 8 design implication is now explicit:

- keep `include/sparse_analysis.h` stable unless the code landing truly forces
  wording follow-through only
- move the compatibility/default-policy normalization behind an internal helper
  seam rather than widening the public API
- let:
  - `src/sparse_analysis.c` keep typed-option resolution ownership
  - `src/sparse_reorder_nd.c` keep direct ND entry ownership
  - the shared helper own compatibility-parser/default-value normalization

#### 3. The preserved compatibility contract is now explicit

The convergence batch must preserve:

- zero-init-safe `sparse_analysis_reorder_opts_t` behavior
- typed analysis values overriding compatibility env vars exactly as shipped
- direct `sparse_reorder_nd(...)` continuing to honor the compatibility path
  when no typed analysis layer is involved
- no public change to the meaning of:
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`

Interpretation:

- the Day 9-10 landing is an ownership convergence batch, not an option-model
  redesign
- proof must focus on behavioral equivalence, not new features

#### 4. The Day 9-10 file fence is now fixed and small

Required implementation surfaces:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`

Likely support only if the landed helper needs it:

- `src/sparse_reorder_nd_internal.h`

Likely proof home:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

Header/docs follow-through only if the landed code truly moves the wording:

- `include/sparse_analysis.h`

Interpretation:

- this stays inside the bounded second lane
- CSC backend files, iterative/eigensolver files, and public API redesign stay
  outside the batch

#### 5. The explicit non-widening fence is now strong enough for the second landing

The shared ND policy convergence batch should not widen into:

- `src/sparse_graph.c`
- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public API redesign
- packaging/platform/build churn

Interpretation:

- Sprint 67’s second lane remains a maintainability convergence batch
- it should not collapse into a broader analysis/CSC rewrite

### Day 8 Close

Sprint 67 Day 8 now fixes one exact second-lane design:

1. strongest target:
   - shared ND compatibility/default-policy normalization
2. required code surfaces:
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
3. likely proof home:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`
4. support only if needed:
   - `src/sparse_reorder_nd_internal.h`
5. header/docs follow-through only if wording actually moves:
   - `include/sparse_analysis.h`

That gives Day 9 one exact job:

- land the bounded shared ND policy convergence batch without widening into CSC
  backend implementation files or public API redesign

## Day 9 - Shared ND Policy Convergence Batch

### Goal

Land the bounded shared ND policy convergence batch by moving the compatibility
and default-policy baseline to one internal owner while preserving the shipped
typed-analysis override contract.

### Actions

1. Re-read the Day 8 design fence and the live ND policy duplication across:
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_reorder_nd_internal.h`
2. Reduced the landing to one shared internal baseline owner:
   - `sparse_reorder_nd_default_policy()`
3. Updated `src/sparse_analysis.c` to start from that shared baseline instead
   of carrying its own duplicated ND compatibility parsers and hard-coded
   default values.
4. Preserved the existing typed analysis option override behavior by leaving
   typed-field resolution in `src/sparse_analysis.c`.
5. Kept the batch inside the Day 8 fence:
   - no CSC backend widening
   - no public API redesign
   - no extra proof-surface widening

### Findings

#### 1. The strongest remaining duplicated ND policy seam is now closed at one internal owner

`src/sparse_reorder_nd.c` now owns the shared internal ND compatibility/default
baseline through:

- `sparse_reorder_nd_default_policy()`

`src/sparse_analysis.c` now consumes that baseline directly instead of
duplicating the following ND compatibility/default responsibilities locally:

- root-bisect mode
- coarsening mode
- coarsest-bisection mode
- root-bisect max-n
- coarsen floor ratio
- coarsening CV fallthrough
- separator-lift strategy
- separator-lift weight

Interpretation:

- the direct ND reorder path remains the natural owner of the compatibility
  baseline
- the repeated-run analysis path now layers typed analysis values on top of the
  same baseline instead of owning a second copy

#### 2. The shipped typed-analysis override contract stayed intact

The landed batch preserves the Day 8 compatibility fence:

- zero-init-safe `sparse_analysis_reorder_opts_t` behavior still starts from the
  same effective ND compatibility/default baseline
- typed analysis values still override that baseline exactly as shipped
- direct `sparse_reorder_nd(...)` still keeps its own compatibility-path
  behavior because the shared baseline owner remains in the reorder lane

One intentionally separate compatibility parser stayed local:

- `supernodal_postorder`

That stayed local because the Day 8 convergence target was ND policy
duplication, not every analysis compatibility field in the file.

#### 3. The batch stayed inside the bounded second-lane fence

Touched code surfaces:

- `src/sparse_analysis.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`

Untouched by design:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `include/sparse_analysis.h`
- CSC backend files
- iterative/eigensolver files
- graph subsystem files outside the existing reorder owner

Interpretation:

- this was an ownership-convergence batch, not a broader analysis or backend
  redesign
- the existing proof homes remain the right validation surfaces for behavioral
  equivalence

### Day 9 Close

Sprint 67 Day 9 now lands one bounded second-lane maintainability batch:

1. shared ND compatibility/default baseline owner:
   - `sparse_reorder_nd_default_policy()`
2. landed consumer convergence:
   - `src/sparse_analysis.c` now starts from that shared baseline
3. preserved contract:
   - typed analysis overrides still win over compatibility env vars
4. stayed inside the Day 8 fence:
   - no CSC widening
   - no public API redesign
   - no proof-surface widening unless validation proves it necessary

## Day 10 - Post-Landing Audit And Rerank

### Goal

Audit the post-Day-9 branch state and fix the next real Sprint 67 target in
writing instead of widening automatically from the shared ND policy lane.

### Actions

1. Re-read the Day 8 design fence and the landed Day 9 artifact.
2. Re-audited the live post-Day-9 ownership surfaces in:
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
3. Re-checked the strongest existing proof homes:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
4. Re-ranked the remaining Sprint 67 queue by ownership blur rather than
   by raw file size alone.

### Findings

#### 1. The Day 9 batch closed the strongest remaining ND-policy contradiction

The shared ND compatibility/default-policy seam is no longer the strongest
maintainability problem on the branch:

- `src/sparse_reorder_nd.c` now owns the ND compatibility/default baseline
- `src/sparse_analysis.c` now consumes that baseline instead of carrying its
  own second copy
- the residual `supernodal_postorder` compatibility parser in
  `src/sparse_analysis.c` is intentionally smaller and separate than the Day 8
  ND-policy target

Interpretation:

- another immediate ND-policy batch would now be fake symmetry rather than the
  highest-value next landing

#### 2. The strongest remaining seam has shifted to the large-n analysis → CSC handoff

The strongest post-Day-9 ownership blur now sits in the large-`n` explicit
analysis lifecycle handoff across:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`

The concrete reason is not just file length.  It is that the public repeated-
run direct lifecycle now fans into multiple partially parallel internal
surfaces:

- `factor_cholesky_with_analysis_csc(...)`
- `factor_ldlt_with_analysis_csc(...)`
- `chol_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- the CSC writeback/publication paths

Interpretation:

- the next real maintainability win is no longer graph-only
- it is analysis-to-CSC orchestration coherence on the large-`n` direct-family
  lane

#### 3. Cholesky now owns the strongest next bounded landing inside that lane

Within the CSC/analysis residual lane, Cholesky is the better next target than
LDL^T:

- `factor_cholesky_with_analysis_csc(...)` in `src/sparse_analysis.c` is
  simpler and more directly comparable to the public CSC helpers
- `src/sparse_chol_csc.c` already contains both the analysis-aware conversion
  and the CSC writeback/publication path in one family-local seam
- LDL^T still carries additional Bunch-Kaufman-specific ownership:
  - `D`
  - `D_offdiag`
  - `pivot_size`
  - composed permutation state
  - resolved-analysis preparation

Interpretation:

- Day 11 should target the Cholesky analysis/CSC handoff first
- LDL^T remains real follow-through, but not the best next bounded landing

#### 4. The exact Day 11 target is now fixed

Strongest next batch:

- large-`n` Cholesky analysis/CSC handoff coherence

Required code surfaces:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`

Likely proof home:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Support only if the landing truly needs it:

- `src/sparse_chol_csc_internal.h`

Likely deferred in the same batch:

- `src/sparse_ldlt_csc.c`
- `tests/test_ldlt_csc.c`
- `include/sparse_analysis.h`

#### 5. The non-widening fence is still explicit

The next landing should not widen into:

- `src/sparse_graph.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- packaging/platform/build churn
- public API redesign

Interpretation:

- Sprint 67 now transitions from the ND lane to the CSC/analysis lane
- it still stays bounded to maintainability follow-through, not feature work

### Day 10 Close

Sprint 67 Day 10 now fixes the post-Day-9 rerank explicitly:

1. the shared ND policy lane is no longer the strongest remaining seam
2. the strongest next seam is the large-`n` analysis-to-CSC direct-family handoff
3. Cholesky owns the best next bounded landing inside that lane
4. Day 11 should target:
   - `src/sparse_analysis.c`
   - `src/sparse_chol_csc.c`
   - likely proof in `tests/test_integration.c` and `tests/test_chol_csc.c`

## Day 11 - Large-n Cholesky analysis/CSC handoff batch

Date: 2026-06-13
Commit: `pending`

### Goal

Land the next bounded Sprint 67 maintainability batch by converging the
large-`n` explicit-analysis Cholesky CSC handoff onto one family-local factor
owner instead of keeping a second CSC orchestration shell in
`src/sparse_analysis.c`.

### Actions

1. audited the live large-`n` Cholesky analysis/CSC path across:
   - `src/sparse_analysis.c`
   - `src/sparse_chol_csc.c`
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`
2. changed `chol_csc_factor(...)` so the analysis-backed large-`n` lane now
   resolves through `chol_csc_eliminate_supernodal(...)` with the shared
   `SPARSE_CSC_SUPERNODE_MIN_SIZE` cutoff, matching the shipped public
   repeated-run Cholesky lifecycle
3. replaced the dedicated CSC elimination shell inside
   `factor_cholesky_with_analysis_csc(...)` with a call into the family-local
   `chol_csc_factor(...)` helper
4. tightened the family-local helper comment in
   `src/sparse_chol_csc_internal.h` so the large-`n` analysis-backed routing
   contract is stated directly
5. added two bounded proofs:
   - `tests/test_chol_csc.c` now proves `chol_csc_factor(A, &analysis, ...)`
     matches the explicit
     `chol_csc_from_sparse_with_analysis(...)` +
     `chol_csc_eliminate_supernodal(...)` route on a large SPD case
   - `tests/test_integration.c` now asserts the large-`n` one-shot Cholesky
     side of the existing public-path comparison actually resolved to the CSC
     lane via `used_csc_path == 1`

### Validation

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Results:

- all passed
- reviewed CMake parity anchor remained `53`
- Makefile/CMake parity remained `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 468.96 sec`

### Outcome

The Day 11 landing stayed inside the Day 10 fence:

- the large-`n` analysis-backed Cholesky CSC factor route now has one shared
  family-local owner
- the public repeated-run lifecycle no longer carries a second copy of the CSC
  elimination dispatch for that lane
- the proof burden stayed bounded to the Cholesky CSC family-local surface and
  the existing public large-`n` integration comparison

### Notes

- the first test cut used `build_tridiag_spd(...)` inside `tests/test_chol_csc.c`,
  but that helper does not exist on that family-local surface; the landed proof
  now builds its SPD tridiagonal matrix inline so the test remains self-contained
  and keeps the batch inside the intended touched-file fence

## Day 12 - Build and regression alignment

Date: 2026-06-13
Commit: `pending`

### Goal

Close the remaining Sprint 67 build/regression-surface contradiction after the
Day 6-11 maintainability landings: the code boundaries moved, but the
maintained docs still did not say clearly which proof surfaces now own the
shared ND-policy lane and the large-`n` Cholesky analysis/CSC handoff lane.

### Actions

1. re-read the Day 6-11 landing set against the Sprint 67 Day 12 plan
2. confirmed no real source-list or target-list contradiction remained in:
   - `Makefile`
   - `CMakeLists.txt`
3. identified the real residual alignment gap instead:
   - stale/underspecified proof-surface ownership in maintained docs
   - stale CSC direct-family suite inventory counts in `README.md`
4. updated maintained docs accordingly:
   - `README.md`
   - `docs/maintainer_guide.md`
   - `benchmarks/README.md`

### Validation

This was a docs-only alignment batch, so I did not rerun:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I ran the targeted Day 12 sanity set instead:

- diff review of touched docs
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch status recheck

### Outcome

The maintained contract is now explicit:

- `tests/test_reorder_nd.c` owns the shared ND compatibility/default-policy
  convergence proof lane
- `tests/test_chol_csc.c` owns the family-local large-`n` analysis-backed
  Cholesky CSC handoff proof lane
- `tests/test_integration.c` owns the public one-shot vs explicit repeated-run
  Cholesky parity and failure-preservation lane
- benchmark surfaces stay benchmark-side proof for repeated-run workflow and
  performance, not substitutes for those regression owners
- `README.md` no longer understates the live CSC Cholesky / CSC LDL^T suite
  sizes
