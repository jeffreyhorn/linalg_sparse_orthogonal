# Sprint 73 Working Notes

## Day 1 - Scope Audit and Baseline Setup

### Goal

Turn the Sprint 73 project-plan scope plus the Sprint 70 and Sprint 72
handoff into one bounded configuration-modernization sprint, with the
strongest live control surfaces and non-goal fence fixed before deeper audit
begins.

### Actions

1. Re-read the Sprint 73 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 architecture contract, the Sprint 72 retrospective, and the
   Sprint 72 closeout artifact.
2. Reconfirm the preserved Sprint 73 constraints:
   - no generic env-var purge detached from real ownership cost
   - no public capability widening disguised as configuration cleanup
   - no platform/install/reviewed-contract widening
   - no broad backend or product-model rewrite hidden inside control-surface
     modernization
3. Reconfirm the strongest local reviewed baseline shape from:
   - `make -n quality-review-full`
   - `make quality-review-cmake-compile`
4. Capture the live Day 1 hotspot map across the strongest likely Sprint 73
   configuration surfaces.
5. Record the intended Sprint 73 workstreams, touch surfaces, and proof-risk
   surfaces before Day 2 validation work begins.

### Findings

#### 1. Sprint 73 now starts from a precise residual-control queue

Sprint 73 does not need another broad Epic 7 planning reset and it does not
need another public-surface cleanup pass.

The strongest next queue is explicitly:

- residual env-var inventory and rerank
- typed versus internal-policy ownership design
- FM/graph policy integration
- debug/profile surface rationalization
- docs/maintainer/header follow-through only where landed behavior truly moves
  the contract
- focused regression coverage and validation

#### 2. The Sprint 70 non-goal fence remains the right constraint set

The live repo state still supports the same fence:

- no repo-wide env-var purge detached from real ownership cost
- no fake capability or product widening through undocumented internal knobs
- no platform/install/reviewed-contract widening
- no broad backend or product-model rewrite hidden inside configuration work

That means Sprint 73 should stay bounded to the strongest residual control
surfaces instead of trying to clean every process-global switch at once.

#### 3. The strongest live configuration pressure is concentrated in graph/FM,
ND, and residual advanced-control seams

The live `getenv(...)` and residual compatibility map is concentrated in:

- graph/FM strategy and pass-count surfaces in `src/sparse_graph.c`
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
- FM refinement and developer/debug surfaces in `src/sparse_graph_refine.c`
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`
- ND/coarsening and profile surfaces in:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph_bisect.c`
  with controls such as:
    - `SPARSE_ND_ROOT_BISECT`
    - `SPARSE_ND_COARSENING`
    - `SPARSE_ND_COARSEST_BISECTION`
    - `SPARSE_ND_ROOT_BISECT_MAX_N`
    - `SPARSE_ND_COARSEN_FLOOR_RATIO`
    - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
    - `SPARSE_ND_SEP_LIFT_STRATEGY`
    - `SPARSE_ND_SEP_LIFT_WEIGHT`
    - `SPARSE_ND_PROFILE`
- residual reorder/profile compatibility in `src/sparse_reorder_amd_qg.c`
  - `SPARSE_QG_PROFILE`
- residual analysis/SVD compatibility in:
  - `src/sparse_analysis.c`
    - `SPARSE_SUPERNODAL_POSTORDER`
    - `SPARSE_ND_SUPERNODAL_POSTORDER`
  - `src/sparse_svd.c`
    - `SPARSE_SVD_LOWRANK_OUTER`

This is the right Day 1 narrowing: Sprint 73 should start from graph/FM and ND
policy convergence, with debug/profile and residual compatibility spill as the
second modernization lane.

#### 4. The strongest likely Sprint 73 touch surfaces are now explicit

Raw Day 1 `wc -l` counts from the live tree:

##### Maintained public/policy surfaces

- `docs/maintainer_guide.md` = `585`
- `include/sparse_analysis.h` = `499`

##### Configuration-modernization implementation seams

- `src/sparse_graph.c` = `821`
- `src/sparse_graph_internal.h` = `850`
- `src/sparse_reorder_nd.c` = `739`
- `src/sparse_graph_refine.c` = `629`
- `src/sparse_graph_coarsen.c` = `641`
- `src/sparse_graph_separator.c` = `297`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_svd.c` = `1319`

##### Strongest proof and reporting surfaces

- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `2262`
- `tests/test_integration.c` = `2448`
- `tests/test_fuzz.c` = `651`
- `examples/example_analysis.c` = `210`
- `benchmarks/bench_reorder.c` = `321`

#### 5. The strongest reviewed baseline remains intact

The local reviewed baseline remains unchanged:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity was re-materialized live:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps Sprint 73 aligned with the Sprint 70 truthfulness fence before any
configuration precedence work lands.

### Validation

This was a docs-only Day 1 baseline/setup pass, so I did not run
`make format`, `make lint`, or `make test`.

I did recheck the reviewed baseline shape and parity anchors with:

- `make -n quality-review-full`
- `make quality-review-cmake-compile`

I also captured the live Day 1 raw `wc -l` hotspot measurements and the
residual `getenv(...)`/compatibility map across the strongest likely Sprint 73
configuration surfaces.

### Day 1 Exit State

Sprint 73 Day 1 closes with:

1. one configuration-modernization starting queue
2. one preserved Sprint 70 non-goal fence
3. one live reviewed baseline anchor
4. one ranked live residual-control hotspot map

## Day 2 - Validation Baseline and Truth-Surface Recheck

### Goal

Reconfirm the Sprint 73 implementation-day validation contract and fix the
highest-signal rerun set before any configuration-modernization batch lands.

### Actions

1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 73 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or precedence-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design/review days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 73 is most likely to stress:
   - graph/reorder proof owners
   - integration and compatibility proof owners
   - representative examples
   - maintained reorder/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 73.
5. Record the authoritative validation split in the working notes.

### Findings

#### 1. The strongest reviewed baseline remains unchanged

Sprint 73 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps the sprint aligned with the Sprint 70 truthfulness fence before any
typed/default/env precedence or policy-ownership work lands.

#### 2. The Sprint 73 authority split is now explicit before code work

The Day 2 recheck fixes the same three-part validation split Sprint 70 and
Sprint 72 used:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial architecture or precedence-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

That is the right split for Sprint 73 because the likely work crosses graph,
reorder, compatibility, and control-precedence boundaries rather than one tiny
file-local helper seam.

#### 3. The live proof-surface split is now fixed for Sprint 73

The Day 2 recheck shows this live local split:

- the reviewed CMake tree currently owns the key proof-owner tests,
  representative examples, and reorder benchmark binaries most relevant to
  Sprint 73:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_framework_optin`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- the root `build/` tree is not currently carrying the usual maintained
  benchmark binaries such as:
  - `build/bench_reorder`
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

That truth matters: Sprint 73 should anchor its Day 2 rerun set to the live
reviewed CMake tree and the maintained proof scripts, rather than assuming the
root benchmark binaries are materialized right now.

#### 4. The high-signal Sprint 73 rerun set is now explicit

The strongest likely rerun set for Sprint 73 is:

- graph/FM proof owners:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
- reorder/precedence proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
- compatibility/support proof owners:
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_framework_optin`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained reorder/reporting surfaces currently materialized in the reviewed
  tree:
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

This is the right Day 2 fix: the rerun contract is now tied to the live
configuration-risk surface and the current local binary split, not to a stale
assumption about which benchmark binaries happen to exist in `build/`.

### Validation

This was a docs-only Day 2 pass, so I did not run `make format`, `make lint`,
or `make test`.

I did recheck the reviewed baseline and proof-surface split with:

- `ctest -N --test-dir build/quality-review-cmake`
- direct existence checks on the reviewed CMake proof/test/example/benchmark
  binaries
- direct existence checks on the root `build/` benchmark binaries
- direct existence checks on the install/package regression scripts

### Day 2 Exit State

Sprint 73 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one truthful live proof-surface map
4. one exact rerun set for the strongest likely Sprint 73 configuration lanes

## Day 3 - Residual Env-Var Inventory Audit

### Goal

Re-rank the remaining configuration surfaces by live ownership cost instead of
by historical familiarity, so Sprint 73 can work from one concrete
contradiction map rather than one generic env-var-cleanup story.

### Actions

1. Re-read the strongest residual configuration seams directly in:
   - `src/sparse_graph.c`
   - `src/sparse_graph_refine.c`
   - `src/sparse_reorder_nd.c`
   - `src/sparse_reorder_amd_qg.c`
   - `src/sparse_analysis.c`
   - `src/sparse_svd.c`
2. Re-read the strongest proof owners that currently pin those controls:
   - `tests/test_graph.c`
   - `tests/test_reorder_nd.c`
3. Classify the remaining burdens into:
   - public process-global surprise
   - duplicated typed/default/env precedence
   - developer-only switches leaking into the public story
   - compatibility controls whose behavior is now better owned internally
4. Rank the strongest contradiction centers by:
   - caller confusion cost
   - implementation ownership blur
   - likely bounded Sprint 73 payoff
5. Write the Day 3 audit artifact.

### Findings

#### 1. The broad Sprint 73 configuration problem is now reduced to one ranked
live contradiction map

The strongest remaining control-surface problem is not “too many env vars” in
the abstract.

It is one ranked ownership map:

- strongest first target:
  - graph/FM strategy and pass-count policy
- strongest second target:
  - ND compatibility/default-policy overrides
- strongest third target:
  - developer-only debug/profile surfaces
- strongest later target:
  - residual SVD-routing and advanced compatibility controls

That is the useful Day 3 narrowing: Sprint 73 should not start by trying to
touch every residual `getenv(...)` call. It should start where the same
control story is still split across the most public, highest-cost graph and
reorder lanes.

#### 2. Graph/FM strategy and pass-count policy is the strongest first target

The strongest first contradiction center is the graph/FM lane split between:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Why this is first:

- `src/sparse_graph.c` still owns a dense public-facing control surface:
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
- `src/sparse_graph_refine.c` still owns the FM-local schedule, perturbation,
  and debug parsing:
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`
- the current model still makes advanced FM behavior depend on process-global
  parsing at the orchestration shell plus more process-global parsing inside
  the refinement subsystem
- `tests/test_graph.c` is already the strongest permanent proof owner for this
  lane, which makes the proof cost acceptable for a bounded first landing

This is the best first Sprint 73 target because it combines the largest raw
residual control surface with the clearest bounded payoff: shrink the
process-global public story and move more of the FM behavior into a clearer
typed or internal-policy contract.

#### 3. ND compatibility/default-policy overrides are the strongest second
target

The second contradiction center is the ND lane split between:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

Why this is second instead of first:

- Sprint 61 and Sprint 67 already improved the typed-precedence story here
- `sparse_reorder_nd_default_policy()` now centralizes more of the default
  policy surface than it used to
- but the lane still carries a dense compatibility parser bundle in
  `src/sparse_reorder_nd.c`:
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
- `src/sparse_analysis.c` still has residual compatibility parsing for:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_SUPERNODAL_POSTORDER`
- `tests/test_reorder_nd.c` already carries the strongest precedence and
  override proof cost for this lane

This is still a real Sprint 73 target, but it reads more like the strongest
second landing than the strongest first one because the graph/FM lane still
has the denser process-global spill and the weaker ownership center.

#### 4. Developer-only debug/profile surfaces are real, but better treated as
the second batch than the first

The strongest developer-only spill surfaces are:

- `SPARSE_ND_PROFILE` in `src/sparse_reorder_nd.c`
- `SPARSE_QG_PROFILE` in `src/sparse_reorder_amd_qg.c`
- FM debug flags in `src/sparse_graph.c` and `src/sparse_graph_refine.c`
- `SPARSE_HCC_DEBUG` in `src/sparse_graph_coarsen.c`

These are real contradictions because they leak operational or developer-only
control into permanent code paths and documentation pressure.

But they are weaker first targets than graph/FM policy or ND compatibility
because:

- the public correctness/behavior contract depends on them less
- they look more like rationalization and narrowing work than like the most
  valuable first ownership convergence
- they are good candidates for a second Sprint 73 batch once the main
  graph/FM or ND policy center is cleaned up

#### 5. Residual SVD-routing and advanced compatibility controls are valid but
lower-priority

The remaining advanced compatibility surface in:

- `src/sparse_svd.c`
  - `SPARSE_SVD_LOWRANK_OUTER`

is real, but it is a weaker Day 3 target because:

- it is narrower
- it carries lower public confusion cost than the graph/FM and ND lanes
- its proof and ownership surface is more isolated than the graph/reorder
  policy story

That makes it a later Sprint 73 or post-Sprint-73 queue item, not the best
first modernization center.

### Validation

This was a docs-only Day 3 audit pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the audit in direct rereads of the live parser and policy seams in:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_analysis.c`
- `src/sparse_svd.c`

and in the strongest proof owners:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

### Day 3 Exit State

Sprint 73 Day 3 closes with:

1. one ranked residual-control contradiction map
2. one strongest first target fixed to graph/FM policy convergence
3. one strongest second target fixed to ND compatibility/default-policy
   overrides
4. one bounded later queue for debug/profile and SVD-routing cleanup

## Day 4 - First Modernization Boundary

### Goal

Refine the Day 3 residual-control ranking and freeze the first bounded Sprint
73 modernization fence before implementation design begins.

### Actions

1. Re-rank the Day 3 contradiction centers against:
   - public process-global surprise
   - implementation leverage
   - compatibility risk
   - likely bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred configuration surfaces
3. Fix the strongest first Sprint 73 fence around the best first
   modernization lane.
4. Record the strongest non-goals for the first landing.
5. Write the Day 4 boundary artifact.

### Findings

#### 1. The strongest first Sprint 73 fence is graph/FM policy convergence, not
the ND compatibility lane and not the debug/profile lane

The Day 4 rerank confirms the best first bounded lane is:

- graph/FM strategy and pass-count policy convergence

That lane has the strongest combination of:

- public process-global surprise
- implementation ownership blur
- bounded cleanup payoff
- acceptable first-pass compatibility risk

The ND compatibility/default-policy lane remains real, but it is not the best
first landing because:

- Sprint 61 and Sprint 67 already improved typed-precedence there
- its current contradiction is denser in compatibility follow-through than in
  the broad public process-global story
- it reads more like the strongest second batch than the strongest first batch

The debug/profile lane also remains real, but it is weaker first work because
it carries less public behavior cost and reads more like rationalization than
like the highest-value first ownership convergence.

#### 2. The first-batch landing surfaces are now explicit

Required first landing:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Likely support only if the first landing forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

Deferred or explicitly later:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- broader public/doc surfaces
- capability/type work
- packaging/platform/workflow surfaces

#### 3. The first-batch contract is narrower than a generic graph-family cleanup

The Day 4 boundary treats the graph/FM lane as:

- a residual control-surface convergence problem
- not a broad graph partitioner redesign

So the first batch can touch only what clarifies:

- FM strategy ownership
- FM pass-count ownership
- FM schedule and perturbation ownership
- the line between typed/internal policy and compatibility-only process-global
  overrides

It should not widen into:

- general graph algorithm redesign
- ND/coarsening default-policy redesign
- broad debug/profile cleanup everywhere at once
- proof-owner churn beyond the strongest immediate FM lane

#### 4. The strongest non-goal fence is now explicit

Sprint 73 Day 4 fixes the first-lane non-goals:

- no repo-wide removal of every env var
- no premature public option-surface widening across all advanced controls
- no broad graph/reorder redesign without a ranked center
- no debug/profile rationalization wave as the first move
- no capability, backend, or platform work hidden inside configuration cleanup

### Validation

This was a docs-only Day 4 boundary pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded it in the Day 3 ranked contradiction map plus direct rereads of:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

### Day 4 Exit State

Sprint 73 Day 4 closes with:

1. one explicit first modernization boundary around graph/FM policy
   convergence
2. one fixed support-only map for proof and maintained-surface follow-through
3. one explicit deferred map for ND, debug/profile, and SVD-routing cleanup
4. one clear starting point for Day 5 implementation design

## Day 5 - Typed/Internal Policy Design

### Goal

Define the bounded implementation contract for the first Sprint 73
configuration-modernization landing before code edits begin.

### Actions

1. Re-read the Sprint 70 configuration target and non-goal fences against the
   Day 4 first-batch surfaces.
2. Decide the ownership split for the strongest remaining graph/FM controls:
   - public typed options
   - internal typed policy
   - compatibility-only env overrides
   - debug-only or narrowed developer-only behavior
3. Fix the precedence rules the first batch must preserve.
4. Freeze the exact first-batch non-touch set.
5. Write the Day 5 design artifact.

### Findings

#### 1. The first Sprint 73 batch should converge FM controls into one internal
policy owner, not widen the public option surface yet

The strongest useful Day 5 design decision is now explicit:

- the first Sprint 73 batch should not add a new public typed FM option
  surface
- it should converge the graph/FM lane behind one clearer internal policy
  owner

Why this is the right first move:

- the strongest current pain is split process-global parsing across
  `src/sparse_graph.c` and `src/sparse_graph_refine.c`
- the public maintained contract does not yet need a broad new FM option model
  to reduce that ownership blur
- adding new public typed controls now would widen the compatibility and
  documentation burden before the internal ownership center is cleaner

So the Day 6 batch should be an internal typed-policy convergence, not a
public API expansion.

#### 2. The first-batch ownership split is now fixed

Public typed options in the first batch:

- none required

Internal typed policy owner in the first batch:

- one graph/FM policy object should own:
  - finest FM strategy
  - ensemble member list
  - finest-level pass count
  - intermediate-level pass count
  - annealing schedule choice
  - thick-restart perturbation choice
  - gain-noise schedule choice
  - debug flags only as internal/runtime fields if still needed

Compatibility-only env overrides in the first batch:

- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Narrowed developer-only or debug-only behavior in the first batch:

- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

The key ownership rule is:

- compatibility env vars may still exist for back-compat
- but they should be parsed once at the orchestration boundary and lowered
  into one internal FM policy/runtime contract
- the refinement subsystem should stop behaving like a second independent
  public configuration parser

#### 3. The preserved precedence rules are now explicit

The first batch must preserve:

- if no FM compatibility env var is set, the existing default behavior stays
  bit-compatible
- recognized compatibility env vars still select the same effective FM
  strategies, pass counts, and schedule/perturbation choices as shipped today
- unrecognized or malformed compatibility env inputs still fall back to the
  current safe defaults rather than widening failure semantics
- developer-only debug flags do not become part of a broader public typed
  contract
- the batch must reduce split ownership, not silently broaden supported
  caller-visible controls

This is the most important Day 5 fence: preserve current caller-visible
compatibility while shrinking the number of places that independently interpret
the same FM control story.

#### 4. The first-batch touch and non-touch sets are now fixed

Required first implementation center:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Support only if the implementation truly forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`

Explicit non-touch set:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`
- broader README/tutorial/example/benchmark surfaces
- capability/type surfaces
- packaging/platform/workflow files

That keeps the first code batch bounded to FM ownership convergence rather
than letting it widen into the ND second batch or the debug/profile later
batch.

### Validation

This was a docs-only Day 5 design pass, so I did not run `make format`,
`make lint`, `make test`, or `make quality-review-full`.

I grounded the design in rereads of:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`
- the Sprint 70 configuration fence
- the Sprint 73 Day 4 modernization boundary

### Day 5 Exit State

Sprint 73 Day 5 closes with:

1. one explicit internal-policy-first design for the FM lane
2. one preserved compatibility-precedence checklist
3. one exact first-batch touch set
4. one explicit non-touch set before Day 6 implementation begins

## Day 6: FM/Graph Policy Integration Batch 1

### Objectives

Land the first bounded Sprint 73 implementation batch by converging the
graph/FM compatibility env surface behind one internal policy owner, while
preserving current defaults and back-compat behavior.

### Work Completed

#### 1. The graph/FM lane now resolves one internal policy object at the orchestration boundary

Landed implementation in:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`

The main Day 6 ownership change is:

- `src/sparse_graph.c` now parses the FM compatibility env surface once into
  `sparse_graph_fm_policy_t`
- that policy owns:
  - finest FM strategy
  - ensemble strategy list
  - finest/intermediate pass counts
  - annealing schedule choice
  - thick-restart perturbation choice
  - gain-noise schedule choice
  - retained debug/runtime flags
- `src/sparse_graph_refine.c` no longer behaves like a second independent
  parser for the annealing / thick-restart / gain-noise control lane

This preserves the Day 5 contract:

- no new public typed FM option surface
- compatibility envs remain supported
- orchestration owns parsing
- refinement consumes lowered runtime state

#### 2. The runtime seam is now narrower and more truthful

The runtime handoff now works like this:

- `graph_fm_policy_from_compat_env(...)` resolves the full FM compatibility
  story once
- `graph_uncoarsen_runtime_for_level(...)` lowers that policy into the
  finest-level runtime state only when the uncoarsening level actually needs
  it
- `sparse_graph_fm_runtime_set(...)` transfers the resolved runtime into the
  refinement subsystem
- `graph_refine_fm(...)` now reads the already-resolved runtime flags instead
  of calling `getenv(...)` for annealing and gain-noise debug behavior

The most important boundary improvement is:

- the refinement subsystem is no longer deciding schedule / perturbation /
  debug behavior from process-global state on its own

#### 3. The batch stayed inside the Day 5 fence

Touched implementation surfaces:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_internal.h`

Untouched deferred surfaces:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`
- `tests/test_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_svd.c`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

No support-surface widening was required:

- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_integration.c`

Existing proof already covered the preserved compatibility behavior well
enough for this ownership-convergence batch, so the landing stayed focused on
the configuration seam itself.

#### 4. One Day 6 implementation correction was required during validation

The first cut stored the finest-FM strategy in the new policy object as
`int`.

That compiled in the normal build, but the strict `make lint` path rejects the
implicit signedness conversion under `-Werror -Wsign-conversion`.

The landed correction was:

- move `finest_fm_strategy_t` into `src/sparse_graph_internal.h`
- make `sparse_graph_fm_policy_t.finest_strategy` use that enum type directly

That keeps the new internal ownership seam genuinely typed instead of fixing
the warning with an explicit cast.

### Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 337.63 sec`

### Day 6 Exit State

Sprint 73 Day 6 closes with:

1. one internal FM policy object as the graph/FM compatibility ownership center
2. one refinement runtime seam that consumes lowered state instead of
   re-parsing process-global controls
3. preserved default and compatibility env behavior without widening the public
   option surface
4. a fully validated first Sprint 73 implementation landing from
   `make quality-review-full`

## Day 7: Post-Landing Audit & Rerank

### Objectives

Reassess the residual configuration queue after the Day 6 FM/graph policy
landing and choose the strongest exact Day 8 target from the live post-landing
state.

### Audit Results

#### 1. The Day 6 landing closed the strongest graph/FM ownership contradiction

The Day 6 batch removed the main first-lane contradiction:

- `src/sparse_graph.c` is now the one compatibility-parser owner for the FM
  strategy/pass/schedule lane
- `src/sparse_graph_refine.c` now reads lowered runtime state instead of
  independently re-parsing that control surface

So a second same-family FM batch is no longer the highest-value next move.

The remaining FM-local follow-through is now weaker:

- residual env parsing in `src/sparse_graph.c` is concentrated instead of split
- retained FM debug flags are now runtime details rather than cross-file parser
  duplication
- the strongest contradiction is no longer "two places interpreting the same FM
  story"

#### 2. The strongest remaining contradiction has shifted to developer-only/profile spill

The strongest live residual-control seam is now the developer-only/profile
lane across:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Why this lane now ranks first:

- `src/sparse_graph_coarsen.c` still mixes real strategy/default routing with
  developer-only `SPARSE_HCC_DEBUG` and residual compatibility override reads
- `src/sparse_reorder_nd.c` still owns `SPARSE_ND_PROFILE` activation as a
  direct process-global check rather than a narrower internal runtime/policy
  seam
- `src/sparse_reorder_amd_qg.c` still does the same for
  `SPARSE_QG_PROFILE`
- this means the post-Day-6 queue is no longer mostly about FM policy
  ownership, but about instrumentation and developer-only controls still
  living as ad hoc process-global reads in multiple graph/reorder families

This is now a better second Sprint 73 lane than:

- another FM/graph batch in `src/sparse_graph.c` / `src/sparse_graph_refine.c`
- the later `SPARSE_SVD_LOWRANK_OUTER` routing seam in `src/sparse_svd.c`

#### 3. The lower-priority lanes are now explicit

FM/graph follow-through is now support/deferred context:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

Residual compatibility or later configuration lanes remain real, but lower
priority:

- `src/sparse_svd.c`
- `src/sparse_analysis.c`
- `src/sparse_graph_separator.c`

Why they rank lower now:

- `src/sparse_svd.c` is still only one narrow advisory control
- the separator-lift compatibility surface is real, but it is not currently as
  cross-family or process-global as the debug/profile lane
- `src/sparse_analysis.c` remains an important authority surface, but the
  strongest immediate contradiction is not there after Day 6

### Day 8 Target Fence

Required next design center:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`

Support only if the Day 8 design truly forces it:

- `src/sparse_graph_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Explicit deferred set for the next batch:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- `src/sparse_svd.c`
- `src/sparse_graph_separator.c`
- README/tutorial/example/benchmark surfaces
- capability/type/platform/workflow files

### Validation

This was a docs-only Day 7 audit pass, so I did not rerun:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I grounded the rerank in rereads of:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_svd.c`
- the Day 6 artifact and validation result

### Day 7 Exit State

Sprint 73 Day 7 closes with:

1. the Day 6 FM lane explicitly demoted from "next batch center" to
   support/deferred context
2. the developer-only/profile lane promoted to the strongest remaining
   contradiction
3. one exact Day 8 target fence around coarsening and ND/profile seams
4. a post-Day-6 queue that is explicit instead of assumed

## Day 8: Debug/Profile Rationalization Design

### Objectives

Define the bounded second Sprint 73 implementation batch around the strongest
remaining developer-only/profile spill, without widening into a broad
compatibility or public-API redesign.

### Design Results

#### 1. The second batch center is now narrower than the Day 7 fence

After the Day 7 rerank and the live reread of the remaining seams, the best
second implementation center is now:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`

Likely support only if the batch truly needs a shared internal runtime seam:

- `src/sparse_graph_internal.h`
- `src/sparse_reorder_amd_qg.c`

The most important Day 8 narrowing is:

- `src/sparse_reorder_amd_qg.c` is still part of the same general profile
  story, but it now reads more like support-only follow-through than the core
  second batch center
- the strongest immediate contradiction is the graph/ND lane where real
  routing/default policy and developer-only profile/debug activation are still
  most visibly mixed

#### 2. The exact second-batch ownership goal is now fixed

The Day 9 batch should:

- keep compatibility controls that still represent real maintained policy:
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- narrow the developer-only/profile spill so those controls stop reading like
  peer public configuration surfaces
- move profile/debug activation into a clearer internal runtime or
  entry-boundary ownership model

The strongest exact second-batch targets are:

- `SPARSE_HCC_DEBUG` in `src/sparse_graph_coarsen.c`
- `SPARSE_ND_PROFILE` in `src/sparse_reorder_nd.c`

Likely support-only target:

- `SPARSE_QG_PROFILE` in `src/sparse_reorder_amd_qg.c`

The key Day 8 design rule is:

- Day 9 should reduce process-global instrumentation sprawl
- it should not reopen the already-landed FM policy batch
- it should not broaden typed public options or claim broader stability for
  developer-only controls than the repo actually maintains

#### 3. The preserved compatibility checklist is now explicit

The second batch must preserve:

- current algorithm-routing defaults when no relevant compatibility env is set
- current recognized behavior for:
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- current opt-in behavior for developer-only instrumentation when it is
  enabled
- the existing narrow meaning of profile/debug surfaces as developer/bench or
  diagnostics aids, not production-facing policy promises

The second batch should avoid:

- turning debug/profile flags into new public typed analysis options
- mixing graph/FM policy follow-through back into the design
- widening into SVD-routing, separator-lift, or public docs/header cleanup

#### 4. Public/support follow-through remains bounded

No public header should move by default in the second batch:

- `include/sparse_analysis.h`
- `include/sparse_reorder.h`

No public-facing docs should move by default:

- `README.md`
- `docs/maintainer_guide.md`

Support only if the Day 9 implementation truly forces it:

- `src/sparse_graph_internal.h`
- `src/sparse_reorder_amd_qg.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_integration.c`
- `docs/maintainer_guide.md`

Explicit non-touch set:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_analysis.c`
- `include/sparse_analysis.h`
- `src/sparse_svd.c`
- `src/sparse_graph_separator.c`
- public README/tutorial/example/benchmark surfaces
- capability/type/platform/workflow files

### Validation

This was a docs-only Day 8 design pass, so I did not rerun:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I grounded the design in rereads of:

- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_graph_internal.h`
- the Day 7 rerank

### Day 8 Exit State

Sprint 73 Day 8 closes with:

1. one exact second-batch center around `SPARSE_HCC_DEBUG` and
   `SPARSE_ND_PROFILE`
2. one likely support-only follow-through map for `SPARSE_QG_PROFILE`
3. one preserved compatibility checklist for real maintained ND/coarsening
   controls
4. one explicit non-touch set before Day 9 implementation begins

## Sprint 73 Day 9: Debug/Profile Rationalization Batch

Date: 2026-06-16
Branch: `sprint-73`
Planned Time: 12 hours

### Context Snapshot

Day 8 fixed the second Sprint 73 implementation fence around the strongest
remaining developer-only/profile spill:

- `SPARSE_HCC_DEBUG` in `src/sparse_graph_coarsen.c`
- `SPARSE_ND_PROFILE` in `src/sparse_reorder_nd.c`

with explicit support-only follow-through for:

- `SPARSE_QG_PROFILE` in `src/sparse_reorder_amd_qg.c`

The key Day 9 rule was:

- narrow debug/profile activation into clearer internal entry-boundary seams
- preserve the real maintained coarsening compatibility controls
- avoid widening into a new public typed debug/profile surface

### Implementation

#### 1. `SPARSE_HCC_DEBUG` now has one explicit internal precedence seam

I landed the graph/coarsen side in:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_internal.h`

The main change is:

- HCC debug activation now resolves through
  `sparse_graph_hcc_debug_current()`
- current-thread begin/end override helpers now exist:
  - `sparse_graph_hcc_debug_override_begin(...)`
  - `sparse_graph_hcc_debug_override_end()`

That tightened the ownership boundary in the right place:

- the legacy `SPARSE_HCC_DEBUG` env var remains the compatibility/default
  source when no override is active
- the implementation no longer re-reads `getenv("SPARSE_HCC_DEBUG")` at each
  debug print site
- focused tests now have one explicit precedence seam instead of ambient
  process env dependence

The Day 8 support fence stayed intact:

- no `src/sparse_graph.c` or `src/sparse_graph_refine.c` follow-through was
  needed
- no public header or maintainer-doc wording moved

#### 2. `SPARSE_ND_PROFILE` now has one explicit internal precedence seam

I landed the ND side in:

- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`

The main change is:

- ND profile activation now resolves through
  `sparse_reorder_nd_profile_current()`
- current-thread begin/end override helpers now exist:
  - `sparse_reorder_nd_profile_override_begin(...)`
  - `sparse_reorder_nd_profile_override_end()`

That improved the ownership split without widening the feature:

- the legacy `SPARSE_ND_PROFILE` env var still controls the default behavior
  when no override is active
- the top-level ND entry path now consumes one explicit internal owner rather
  than open-coding another direct env read
- the change stayed developer-only and did not create a new typed public
  analysis option

#### 3. Focused proof landed in the right owners

I added the bounded precedence regressions in:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Those tests prove the exact Day 9 ownership contract:

- env-set default remains visible when no override is active
- explicit internal override wins while active
- clearing the override restores env-driven behavior
- explicit internal enable/disable also works with the env unset

The new proof owners are:

- `test_hcc_debug_override_precedence`
- `test_nd_profile_override_precedence`

#### 4. The support-only `SPARSE_QG_PROFILE` lane stayed deferred

I rechecked the Day 8 support-only candidate and did not widen into it:

- `src/sparse_reorder_amd_qg.c`

That keeps the batch inside the planned fence:

- graph/coarsen + ND profile ownership moved
- QG profile stayed support-only
- no SVD-routing, separator-lift, public-doc, or public-header spill landed

### Touched Surfaces

Code:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_internal.h`
- `src/sparse_reorder_nd.c`
- `src/sparse_reorder_nd_internal.h`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Raw `wc -l` counts after the landing:

- `src/sparse_graph_coarsen.c` = `659`
- `src/sparse_graph_internal.h` = `894`
- `src/sparse_reorder_nd.c` = `757`
- `src/sparse_reorder_nd_internal.h` = `116`
- `tests/test_graph.c` = `2925`
- `tests/test_reorder_nd.c` = `2287`

### Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 296.46 sec`

Focused proof highlights from the new boundary:

- `tests/test_graph.c`:
  - `test_hcc_debug_override_precedence`
- `tests/test_reorder_nd.c`:
  - `test_nd_profile_override_precedence`

### Day 9 Exit State

Sprint 73 Day 9 closes with:

1. one explicit internal precedence seam for `SPARSE_HCC_DEBUG`
2. one explicit internal precedence seam for `SPARSE_ND_PROFILE`
3. two focused regressions in the right proof owners
4. the `SPARSE_QG_PROFILE` support-only lane still deferred instead of
   widened into this batch

## Sprint 73 Day 10: Follow-Through Design

Date: 2026-06-16
Branch: `sprint-73`

### Goal

Decide the smallest maintained-surface follow-through actually required by the
Day 6 and Day 9 landed configuration contract.

### What I Rechecked

- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day9-debug-profile-rationalization-batch.md`
- `docs/maintainer_guide.md`
- `include/sparse_analysis.h`
- `src/sparse_reorder_amd_qg.c`

### Day 10 Design Result

The public header surface is already coherent after the landed code, so Sprint
73 does not need a broad docs/header cleanup batch.

The only maintained surface that now clearly needs follow-through is:

- `docs/maintainer_guide.md`

That is the one place still reading as if all residual FM-family env vars are
simply deferred, when the live code now has a narrower ownership split:

- recognized `SPARSE_FM_*` compatibility env vars are parsed once in
  `src/sparse_graph.c`
- they lower into one internal typed FM policy/runtime contract
- the refinement subsystem no longer behaves like a second independent parser
- developer-only FM debug flags remain intentionally internal

### Exact Day 11 Touch Set

Required:

- `docs/maintainer_guide.md`

Support only if wording truly forces it:

- `include/sparse_analysis.h`

Explicit non-touch set:

- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `src/sparse_analysis.c`
- `src/sparse_svd.c`
- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

### Preserved Truthfulness Checklist

Day 11 must preserve:

- no new public typed FM option family
- no public typed debug/profile option family
- `SPARSE_ND_PROFILE`, `SPARSE_HCC_DEBUG`, and `SPARSE_QG_PROFILE` remain
  internal or developer-only surfaces
- `include/sparse_analysis.h` stays truthful if left unchanged:
  - lower-level FM tuning and debug/profile env vars remain internal or
    compatibility-only for now
- `SPARSE_QG_PROFILE` remains support-only deferred follow-through, not a
  hidden Sprint 73 widening

### Sanity Outcome

The Day 10 recheck found one real policy-drift target, not a broader public or
header contradiction:

1. `docs/maintainer_guide.md` needs the Day 6 and Day 9 ownership split stated
   directly
2. `include/sparse_analysis.h` already remains accurate and should stay
   untouched unless the Day 11 wording forces a narrow consistency edit
3. `src/sparse_reorder_amd_qg.c` remains explicitly deferred support context,
   not a Day 11 follow-through center
