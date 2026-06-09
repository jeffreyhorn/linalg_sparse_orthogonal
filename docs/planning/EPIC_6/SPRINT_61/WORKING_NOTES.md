# Sprint 61 Working Notes

## Day 1

**Objective:** Turn the Sprint 61 project-plan scope plus the Sprint 60 frozen
architecture/validation contract into a concrete Phase 1 implementation
starting point by confirming the preserved reviewed baseline, naming the
configuration-modernization workstreams explicitly, and fixing the strongest
live env-var/control hotspots before design or code migration work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 61 project-plan source and the new sprint plan:
   - `sed -n '56,84p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
3. Re-read the strongest inherited Sprint 60 closeout source:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day14-closeout-and-handoff.md`
4. Re-read the strongest current control-plane public/implementation seam:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' src/sparse_analysis.c`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Inventory the current env-var and control-placement seams:
   - `rg -n "SPARSE_(ND|FM|SUPERNODAL_POSTORDER)|getenv\\(|environment" src include README.md docs/tutorial.md docs/maintainer_guide.md`
8. Measure the main Sprint 61 public/control/docs/proof hotspots:
   - `wc -l include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h src/sparse_analysis.c src/sparse_graph.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c docs/maintainer_guide.md README.md docs/tutorial.md tests/test_integration.c tests/test_graph.c`

### Day 1 Findings

#### 1. Sprint 61 starts from the Sprint 60 frozen contract, not from renewed target-definition or workflow-boundary debate

Sprint 60 already closed the Epic 6 opening contract work:

- the productization gap inventory is written
- the state-of-the-art target is written
- the control-placement rule is written
- the validation/platform truthfulness rule is written
- the workflow fence is still explicit and unchanged

Interpretation:

- Sprint 61 is not another audit-first sprint
- Sprint 61 is not a public repeated-run support redesign sprint
- Sprint 61 is the first bounded Epic 6 implementation sprint
- the main Day 1 job is to turn the frozen contract into a precise Phase 1
  configuration-modernization map

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the entire Phase 1 configuration batch

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the current reviewed wrapper still presents the expected truth surface:

- `quality-review-full: strongest local reviewed baseline`
- the reviewed path still centers on:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

Interpretation:

- Sprint 61 should inherit the exact Sprint 60 truthfulness wording
- later `*.c` / `*.h` landing days should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial control-plane days should still treat
  `make quality-review-full` as the stronger default

#### 3. The broad “too env-var driven” Epic 6 claim is already concentrated in the graph/reorder and analysis-time control seams

The live `rg` pass shows the strongest remaining process-global control seams
cluster in:

- graph/reorder and FM tuning:
  - `SPARSE_ND_*`
  - `SPARSE_FM_*`
- analysis-time advisory control:
  - `SPARSE_SUPERNODAL_POSTORDER`
- adjacent lower-priority or later controls:
  - `SPARSE_QG_PROFILE`
  - `SPARSE_SVD_LOWRANK_OUTER`

And the strongest concrete implementation ownership bands are now explicit:

- public/current control entry surface:
  - `include/sparse_analysis.h`
- main control-plane and translation seam:
  - `src/sparse_analysis.c`
- graph and ND/FM implementation cluster:
  - `src/sparse_graph.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`

Interpretation:

- Sprint 61 Phase 1 should not pretend every env-var in the repo is equally
  important
- the highest-value first batch is reorder/ND/FM plus selected
  analysis/postorder control placement
- `SVD` and some profile/debug seams remain real, but they are not the right
  opening Sprint 61 center of gravity

#### 4. Sprint 61 reduces cleanly to seven bounded implementation workstreams

The project-plan items collapse to:

1. env-var inventory
2. typed option design
3. reorder/ND integration
4. analysis/postorder integration
5. compatibility behavior
6. regression/docs updates
7. validation and closeout

Interpretation:

- the Sprint 61 implementation order is already smaller and clearer than the
  Epic 6 review sounded on first read
- the right Day 1 deliverable is not “solve configuration modernization”
- the right Day 1 deliverable is a bounded implementation map with a fixed
  non-goal fence

#### 5. The strongest likely Sprint 61 touch surfaces are now explicit from the live tree

The highest-value current Sprint 61 surfaces are:

- public/control surfaces:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest implementation/control seams:
  - `src/sparse_analysis.c` = `780`
  - `src/sparse_graph.c` = `801`
  - `src/sparse_reorder_nd.c` = `642`
  - `src/sparse_reorder_amd_qg.c` = `611`
- truthfulness/docs surfaces likely to need follow-through:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `315`
- strongest proof surfaces likely to matter in Phase 1:
  - `tests/test_integration.c` = `1976`
  - `tests/test_graph.c` = `2900`

Interpretation:

- the strongest early code pressure is concentrated enough to support a bounded
  first migration batch
- the strongest proof pressure is split between:
  - lifecycle/integration validation
  - graph/reorder-sensitive regression coverage
- the docs surface should follow the landed control story rather than lead it

#### 6. Sprint 61 needs an explicit Day 1 non-goal fence before any typed-option design begins

The preserved non-goal fence for Phase 1 is:

- no reopening the Epic 5 repeated-run workflow fence
- no broad backend/AUTO rewrite in the same batch
- no packaging/platform widening disguised as configuration work
- no fake removal of all legacy env-var behavior in Phase 1
- no broad migration of lower-priority debug/profile-only seams unless the
  Phase 1 landing proves they materially block the new control plane

Interpretation:

- Sprint 61 should land the highest-value typed control surfaces first
- compatibility behavior should be explicit and bounded, not magically removed
- Phase 1 success is coherent control placement, not total env-var eradication

### Day 1 Close

Sprint 61 now starts from one explicit Phase 1 configuration baseline:

- the Sprint 60 contract remains frozen and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 configuration problem has narrowed to a ranked set of
  graph/reorder and analysis-time control seams
- the public/implementation/docs/proof hotspots for the first migration batch
  are explicit
- the next step is to turn the live env-var inventory into an exact ranked
  Phase 1 candidate list before typed-option design begins

## Day 2

**Objective:** Freeze the validation and truthfulness baseline that Sprint 61
configuration-surface code changes must preserve by reconfirming the reviewed
baseline, the mandatory `*.c` / `*.h` gate, the stronger control-plane review
path, and the exact rerun set for graph/reorder plus lifecycle-sensitive work.

### Commands Run

1. Confirm branch cleanliness before the Day 2 pass:
   - `git status --short --branch`
2. Re-read the current Sprint 61 notes plus the Day 2 plan slice:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
   - `sed -n '85,140p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
3. Re-read the strongest inherited Day 2 shape from Sprint 60:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day2-validation-baseline-and-touched-surface-recheck.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Re-read the current quality/truthfulness wording:
   - `sed -n '1,220p' README.md`
   - `sed -n '1,240p' docs/maintainer_guide.md`
   - `rg -n "quality-review-full|quality-review-cmake|deadcode|Windows|macOS|Linux|coverage" Makefile README.md docs/maintainer_guide.md .github/workflows`
7. Confirm the Sprint 61 targeted rerun-set presence in the live build tree:
   - `for f in ./build/test_integration ./build/test_graph ./build/test_graph_fm_buckets ./build/test_reorder_nd ./build/test_reorder_amd_qg ./build/test_chol_csc ./build/test_ldlt_csc ./build/test_iterative ./build/test_eigs ./build/test_eigs_lobpcg ./build/example_analysis ./build/example_iterative ./build/example_ic_minres ./build/example_eigs ./build/example_svd_lowrank ./build/bench_refactor ./build/bench_refactor_csc ./build/bench_iterative_reuse ./build/bench_eigs_reuse; do [ -e "$f" ] && echo "$f"; done`
8. Re-read the current Windows reviewed-lane definition for staged exclusions:
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`

### Day 2 Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 61 inherits the same authoritative local validation command as Sprint
60:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

Interpretation:

- Sprint 61 should not invent a narrower local trust anchor for substantial
  control-plane work
- later implementation days can still use the bounded code-day gate when the
  touched surface is limited, but the stronger local reviewed proof point is
  unchanged

#### 2. The reviewed CMake parity count remains the main exact truthfulness anchor

The current reviewed CMake inventory remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

This still matters because it is the simplest exact proof that:

- the reviewed CMake path still sees the maintained full local suite
- Makefile/CMake parity has not drifted silently

Interpretation:

- Sprint 61 Day 2 should freeze `53` as the local parity-count anchor
- later Sprint 61 code days should treat any parity-count movement as a
  contract-level event, not incidental noise

#### 3. The code-day gate versus stronger reviewed path split remains stable

The maintained split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial control-plane or architecture-sensitive work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

Interpretation:

- Sprint 61 should keep the same operating discipline as Sprint 60
- the typed-configuration landing days should default upward to
  `make quality-review-full` when they span public headers plus graph/reorder
  implementation seams

#### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while reviewed Makefile
  wrappers and dead-code stay staged
- Windows staged exclusions remain explicit in the workflow:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- coverage remains a supplemental signal rather than an active reviewed-baseline
  residual
- dead-code remains operationally serialized and separate from `lint` and
  `test`

Interpretation:

- Sprint 61 can proceed from a stable truthfulness contract rather than needing
  a cleanup batch just to align validation wording
- the Windows reviewed subset remains relevant context, but it does not change
  the authoritative local Day 2 baseline

#### 5. The targeted Sprint 61 rerun set is now fixed around configuration-sensitive proof surfaces rather than just inherited broad coverage

The confirmed rerun set is:

- direct lifecycle and integration proofs:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- graph/reorder-sensitive proofs:
  - `./build/test_graph`
  - `./build/test_graph_fm_buckets`
  - `./build/test_reorder_nd`
  - `./build/test_reorder_amd_qg`
- adjacent repeated-run solver proofs that should not drift while public control
  surfaces move:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- representative benchmark surfaces:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

Interpretation:

- Sprint 61 now has a tighter rerun list aligned to its actual control-plane
  risk
- the graph/reorder-sensitive proofs are first-class for this sprint, not
  optional follow-ons

### Day 2 Close

Sprint 61 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged at `53`
- authoritative rerun set fixed from the current build tree around
  lifecycle-sensitive and graph/reorder-sensitive proof surfaces
- docs-only versus bounded code-day versus stronger reviewed-path split fixed
  explicitly
- no contradiction across the main quality/truthfulness surfaces

## Day 3

**Objective:** Reduce the broad Epic 6 “too env-var driven” claim to a
concrete ranked Phase 1 configuration list by inventorying the live
`SPARSE_ND_*`, `SPARSE_FM_*`, and adjacent analysis/reorder controls,
classifying them by future ownership, and fixing the strongest Sprint 61 cut
line before typed-option design begins.

### Commands Run

1. Confirm branch cleanliness before the Day 3 pass:
   - `git status --short --branch`
2. Re-read the Sprint 61 Day 3 plan slice and current sprint notes:
   - `sed -n '120,190p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
3. Re-read the strongest inherited Epic 6 configuration audit source:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_60/artifacts/day7-configuration-and-performance-surface-audit-part1.md`
4. Inventory the live control sites:
   - `rg -n "SPARSE_(ND|FM|SUPERNODAL_POSTORDER|QG_PROFILE|SVD_LOWRANK_OUTER)|getenv\\(" src/sparse_analysis.c src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_refine.c src/sparse_graph_separator.c src/sparse_graph_bisect.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c src/sparse_svd.c`
5. Re-read the strongest public and implementation seam:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' src/sparse_analysis.c`
   - `sed -n '1,240p' src/sparse_graph.c`
   - `sed -n '1,220p' src/sparse_reorder_nd.c`
   - `sed -n '1,220p' src/sparse_graph_refine.c`
   - `sed -n '1,220p' src/sparse_graph_coarsen.c`
6. Collapse the raw env-var surface to a unique control list:
   - `rg -o "SPARSE_[A-Z0-9_]+" src/sparse_analysis.c src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_refine.c src/sparse_graph_separator.c src/sparse_graph_bisect.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c src/sparse_svd.c | sort | uniq -c`
7. Reconfirm the runtime-state coupling around FM / ND profiling:
   - `rg -n "_Thread_local|sparse_graph_fm_runtime_get|sparse_graph_fm_runtime_set|forced_hem|profile" src/sparse_graph*.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c`
8. Reconfirm the strongest proof and user-facing reference surfaces:
   - `rg -n "SPARSE_(ND|FM|SUPERNODAL_POSTORDER|QG_PROFILE|SVD_LOWRANK_OUTER|HCC_DEBUG)" tests/test_graph.c tests/test_reorder_nd.c tests/test_reorder_amd_qg.c tests/test_integration.c README.md docs/tutorial.md docs/maintainer_guide.md`
9. Re-measure the main Day 3 control/proof/docs hotspots:
   - `wc -l include/sparse_analysis.h src/sparse_analysis.c src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_refine.c src/sparse_graph_separator.c src/sparse_graph_bisect.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c tests/test_graph.c tests/test_reorder_nd.c tests/test_reorder_amd_qg.c README.md docs/tutorial.md docs/maintainer_guide.md`

### Day 3 Findings

#### 1. The live process-global control surface is concentrated enough to rank cleanly

The unique current control list now reduces to four concrete classes rather than
one generic env-var complaint:

- caller-meaningful ND / analysis controls:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - legacy `SPARSE_ND_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
- lower-level ND/FM tuning controls:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
- debug/profile-only controls:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - `SPARSE_HCC_DEBUG`
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`
- adjacent non-Phase-1 controls:
  - `SPARSE_SVD_LOWRANK_OUTER`

Interpretation:

- the strongest Sprint 61 surface is smaller and more coherent than the Epic 6
  review summary implied
- the main problem is not “all env vars”
- the main problem is a mixed control plane inside the direct-analysis /
  graph-ordering subsystem

#### 2. The strongest Phase 1 public candidates are the controls that already read like analysis/reorder choices, not low-level FM experimentation

The public typed-option candidates with the best Phase 1 case are:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

Why these rank first:

- they change meaningful ordering/analysis behavior, not just internal search
  heuristics
- they are already close to the `sparse_analyze(...)` and `SPARSE_REORDER_ND`
  front door
- they are visible enough that tests and README-level wording already treat
  them like real user-facing choices
- they fit naturally under a future public analysis/reorder options surface

Interpretation:

- Sprint 61 Phase 1 should start with the analysis/reorder knobs that are
  already behaving like product controls
- this is a public control-surface modernization sprint, not a debug-flag
  cleanup sprint

#### 3. The strongest internal typed-policy candidates are the FM budget/schedule controls and adjacent ND tuning heuristics

The strongest internal typed-policy candidates are:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

Why these do not rank first as public knobs:

- they are more tightly coupled to the multilevel partitioner’s internal search
  and retry behavior
- several are reinforced by `_Thread_local` FM runtime state and orchestration
  save/restore helpers:
  - `sparse_graph_fm_runtime_get(...)`
  - `sparse_graph_fm_runtime_set(...)`
- they carry meaning mainly inside the implementation, not at the library’s
  main analysis front door

Interpretation:

- these controls should move away from raw process-global parsing too
- but their most natural first ownership is internal typed policy rather than a
  broad public API promise

#### 4. The debug/profile controls should stay out of Phase 1 public design

The clear stay-internal diagnostic/debug set is:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- `SPARSE_FM_ENSEMBLE_DEBUG`
- `SPARSE_FM_ANNEALING_DEBUG`
- `SPARSE_FM_THICK_RESTART_DEBUG`
- `SPARSE_FM_GAIN_NOISE_DEBUG`

These already map to:

- `_Thread_local` profiling accumulators in `src/sparse_reorder_nd.c`
- profile-only stderr output in `src/sparse_reorder_amd_qg.c`
- debug-only instrumentation in the graph subsystem

Interpretation:

- Sprint 61 should not promote these into public options
- at most, later Epic 6 work can rationalize them as internal typed diagnostic
  policy or leave them as bounded maintainer-only overrides

#### 5. The current proof surface confirms which controls are already treated like real compatibility commitments

The strongest proof burden sits in:

- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `1594`
- `tests/test_reorder_amd_qg.c` = `273`

And those tests actively mutate the same high-value knobs Sprint 61 is
considering:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_SUPERNODAL_POSTORDER`

Interpretation:

- these are not hypothetical knobs
- the repo already treats them as contract-bearing enough to pin in regression
  tests
- moving them behind typed control placement will need proof-surface updates,
  not just implementation edits

#### 6. The strongest Phase 1 cut line is now explicit

The highest-value Sprint 61 Phase 1 cut is:

- public typed-option candidates:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
- internal typed-policy candidates:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
  - selected `SPARSE_FM_*` budget/strategy/schedule controls
- explicit defer:
  - profile/debug-only controls
  - `SPARSE_SVD_LOWRANK_OUTER`
  - broader compile-time threshold policy

Interpretation:

- Day 4 can now design a typed options surface against a real ranked control
  set
- Sprint 61 does not need to solve every runtime tuning seam at once
- Phase 1 is analysis/reorder modernization first, deeper FM rationalization
  second

### Day 3 Close

Sprint 61 now has a concrete ranked env-var inventory:

- the strongest public candidates are explicit
- the strongest internal typed-policy candidates are explicit
- the debug/profile-only set is explicit
- the proof burden and migration complexity are explicit
- the next step is to define the exact typed-option and precedence contract for
  that ranked Phase 1 cut instead of designing against a generic env-var
  backlog

## Day 4

**Objective:** Turn the Day 3 ranked control inventory into an explicit Sprint
61 Phase 1 options contract by defining the public typed-option model, the
internal resolved-policy model, the exact precedence rules, the bounded legacy
env-var translation story, and the first integration fence before any code
changes begin.

### Commands Run

1. Confirm branch cleanliness before the Day 4 pass:
   - `git status --short --branch`
2. Re-read the Sprint 61 Day 4 plan slice and the current sprint notes:
   - `sed -n '150,240p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
   - `sed -n '1,700p' docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
3. Re-read the ranked Day 3 inventory:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_61/artifacts/day3-env-var-surface-inventory.md`
4. Re-read the strongest current public/control seams:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_reorder.h`
   - `sed -n '260,520p' src/sparse_analysis.c`
5. Re-read the current internal seam definitions:
   - `sed -n '1,260p' src/sparse_analysis_internal.h`
   - `sed -n '1,320p' src/sparse_graph_internal.h`
6. Reconfirm the current FM/runtime coupling and why it should stay internal
   first:
   - `sed -n '1,220p' src/sparse_graph_refine.c`
   - `sed -n '1,220p' src/sparse_graph_coarsen.c`

### Day 4 Findings

#### 1. Phase 1 should widen `sparse_analysis_opts_t`, not invent a parallel public configuration object

The strongest Phase 1 public controls all belong to the direct-analysis /
reorder front door:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

They therefore fit most naturally as a bounded extension of
`sparse_analysis_opts_t`, not as:

- a new standalone public config object
- graph-internal public headers
- a generic repo-wide option bag

Recommended public shape:

- keep `sparse_analysis_opts_t` as the public entry point
- add a nested advanced/reorder sub-struct rather than scattering many new
  top-level fields
- keep all new enum fields zero-init safe with `*_DEFAULT = 0`

Interpretation:

- callers already using the repeated-run direct lifecycle should not need a new
  top-level API concept just to set reorder/analysis controls
- Phase 1 should feel like a coherent analysis-surface widening, not a new
  subsystem

#### 2. The public typed-option model should cover caller-meaningful ND/postorder choices only

The Day 4 recommended public surface is:

- one reusable tri-state switch enum for bounded on/off/default controls:
  - `SPARSE_OPTION_DEFAULT = 0`
  - `SPARSE_OPTION_OFF = 1`
  - `SPARSE_OPTION_ON = 2`
- bounded analysis/reorder enums for:
  - ND coarsening strategy
  - ND coarsest bisection strategy
  - ND separator-lift strategy
  - ND separator-lift weight
- one nested public sub-struct under `sparse_analysis_opts_t` holding:
  - `supernodal_postorder`
  - `nd_coarsening`
  - `nd_coarsest_bisection`
  - `nd_root_bisect`
  - `nd_root_bisect_max_n`
  - `nd_sep_lift_strategy`
  - `nd_sep_lift_weight`

Recommended representation rule:

- enum fields:
  - `DEFAULT = 0` means unspecified / use the normal resolution path
- integer numeric threshold fields:
  - `0` means unspecified / use the normal resolution path

Interpretation:

- zero-initialized `sparse_analysis_opts_t` stays valid
- the existing caller ergonomics survive
- the strongest public knobs become explicit without widening into FM internals

#### 3. FM and lower-level ND tuning should move to an internal resolved-policy layer first, not straight into the public API

The strongest internal typed-policy set remains:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- `SPARSE_FM_INTERMEDIATE_PASSES`
- `SPARSE_FM_FINEST_PASSES`
- `SPARSE_FM_FINEST_STRATEGY`
- `SPARSE_FM_ENSEMBLE_STRATEGIES`
- `SPARSE_FM_ANNEALING_SCHEDULE`
- `SPARSE_FM_THICK_RESTART_PERTURB`
- `SPARSE_FM_GAIN_NOISE_SCHEDULE`

The Day 4 design consequence is:

- define one internal resolved-policy struct for the graph/reorder subsystem
- populate it once near the analysis/reorder front door
- pass typed values down instead of letting deep implementation files re-parse
  raw env vars indefinitely

Why this stays internal first:

- several FM controls are coupled to `_Thread_local` runtime state
- the graph partitioner already has save/restore runtime helpers
- the semantics are more implementation-tuning than stable public contract

Interpretation:

- Phase 1 public design should stop at caller-meaningful analysis/reorder
  choices
- Phase 1 internal design should still modernize the FM/runtime path enough to
  shrink raw `getenv(...)` ownership over time

#### 4. The exact Phase 1 precedence contract is now explicit

The Day 4 precedence rule is:

1. explicit typed option value
2. legacy compatibility override, but only when the typed field is left
   unspecified/default
3. internal typed policy default

And the “default typed option values” rule is intentionally constrained:

- a zero-initialized public options struct does not eagerly stamp concrete
  defaults into every field
- instead, `DEFAULT` / `0` means “unspecified; resolve through compatibility
  override then internal default”
- if a helper is added later to materialize recommended defaults, it must map
  to the same resolved values as the unspecified path with no compatibility
  override present

Interpretation:

- explicit typed values always win
- env vars keep backward-compatible meaning only where the caller leaves the new
  field unspecified
- internal defaults remain the final source of truth after compatibility
  translation

#### 5. The bounded legacy compatibility story is now explicit

The Day 4 compatibility rule is:

- keep canonical env-var names working in Phase 1 when the new typed field is
  unspecified
- preserve the existing legacy alias only where it already exists today:
  - `SPARSE_ND_SUPERNODAL_POSTORDER` as a compatibility alias for
    `SPARSE_SUPERNODAL_POSTORDER`
- do not create new legacy aliases just to soften the migration
- keep debug/profile env vars as maintainer-only internal overrides
- keep adjacent non-Phase-1 seams like `SPARSE_SVD_LOWRANK_OUTER` unchanged

Recommended translation ownership:

- compatibility parsing should be centralized behind one or a few translation
  helpers instead of remaining spread across many deep implementation files

Interpretation:

- Sprint 61 does not promise immediate env-var removal
- it promises explicit precedence and bounded compatibility

#### 6. The first code landing fence is now exact enough for Day 5

The Day 4 landing fence is:

- Phase 1 public widening:
  - `include/sparse_analysis.h`
- Phase 1 public contract follow-through:
  - likely small wording alignment in `README.md`, `docs/tutorial.md`, and
    `docs/maintainer_guide.md` only after the code lands
- Phase 1 front-door translation / resolved-policy seam:
  - `src/sparse_analysis.c`
- Phase 1 graph/reorder consumer seams:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_reorder_nd.c`
- proof-surface follow-through later:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_integration.c`

Explicit non-goals remain:

- no public FM tuning explosion
- no generic repo-wide configuration object
- no migration of debug/profile-only controls into the public API
- no packaging/platform widening
- no backend/AUTO-policy rewrite inside the same sprint slice

### Day 4 Close

Sprint 61 now has an explicit typed-options and precedence contract:

- the public Phase 1 surface is bounded to caller-meaningful
  analysis/reorder/postorder controls
- the lower-level FM and ND tuning controls are assigned to an internal
  resolved-policy lane first
- the precedence order is explicit
- the compatibility story is explicit
- the Day 5 landing design can now work from a real API/implementation contract
  instead of a generic modernization intention

## Day 5

**Objective:** Convert the Day 4 configuration contract into an exact Sprint 61
implementation map by fixing the minimum viable public API additions, the
internal policy bridge, the precise touched-file set, the Day 6 versus Day 7
batch split, and the explicit non-goals before public-header or implementation
edits begin.

### Commands Run

1. Confirm branch cleanliness before the Day 5 pass:
   - `git status --short --branch`
2. Re-read the Sprint 61 Day 5 plan slice and current notes:
   - `sed -n '185,280p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
   - `sed -n '1,920p' docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
3. Re-read the Day 4 contract artifact:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_61/artifacts/day4-typed-options-design-and-precedence-contract.md`
4. Reconfirm the exact current analysis/reorder seams:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_reorder.h`
   - `sed -n '260,520p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_reorder_nd_internal.h`
   - `sed -n '220,760p' src/sparse_graph_internal.h`
5. Reconfirm the exact env-var consumers and test pressure for the first batch:
   - `rg -n "parse_supernodal_postorder|sparse_analyze\\(|SPARSE_SUPERNODAL_POSTORDER|SPARSE_ND_SUPERNODAL_POSTORDER" src/sparse_analysis.c include/sparse_analysis.h tests/test_reorder_nd.c README.md`
   - `rg -n "SPARSE_ND_COARSENING|SPARSE_ND_COARSEST_BISECTION|SPARSE_ND_ROOT_BISECT|SPARSE_ND_SEP_LIFT_STRATEGY|SPARSE_ND_SEP_LIFT_WEIGHT|SPARSE_FM_" src/sparse_graph.c src/sparse_graph_coarsen.c src/sparse_graph_refine.c src/sparse_graph_separator.c src/sparse_graph_bisect.c src/sparse_reorder_nd.c tests/test_graph.c tests/test_reorder_nd.c`
   - `rg -n "sparse_reorder_nd\\(|sparse_graph_partition\\(|graph_bisect_coarsest_spectral|graph_edge_separator_to_vertex_separator|sparse_graph_hierarchy_build" src/sparse_reorder_nd.c src/sparse_graph.c src/sparse_graph_bisect.c src/sparse_graph_separator.c src/sparse_graph_coarsen.c tests/test_reorder_nd.c tests/test_graph.c`

### Day 5 Findings

#### 1. The minimum viable public API addition is a bounded widening of `sparse_analysis_opts_t` only

The smallest coherent public change is:

- widen `include/sparse_analysis.h`
- keep `include/sparse_reorder.h` unchanged
- do not add a new public graph or ND options header
- do not widen one-shot direct family option structs in the same batch

Recommended public additions:

- one tri-state enum for bounded on/off/default controls
- one nested `analysis_reorder` or similarly named sub-struct inside
  `sparse_analysis_opts_t`
- public enums only for the caller-meaningful controls selected on Day 4

Why this is the minimum viable surface:

- `sparse_analyze(...)` is the explicit repeated-run direct front door
- the targeted controls affect symbolic analysis and reorder behavior there
- callers should not need to learn a second public API surface just to express
  these choices

Interpretation:

- the first code batch should touch one public header, not many
- Sprint 61 should preserve the public API’s conceptual center of gravity

#### 2. The key implementation bridge is an internal policy-aware ND path, not a rewrite of the public reorder API

The main landing constraint is explicit now:

- `sparse_analyze(...)` can currently only reach ND through:
  - `sparse_reorder_nd(A, perm)`
- that public function has no typed options input today
- but the Day 4 contract requires `sparse_analyze(...)` to honor typed controls
  without breaking backward compatibility

The smallest viable bridge is:

- keep `sparse_reorder_nd(...)` as the public compatibility wrapper
- add one internal policy-aware ND entry point used by `sparse_analyze(...)`
- let the public `sparse_reorder_nd(...)` continue to resolve through the
  env-var compatibility path when called directly

Recommended ownership:

- declare the new internal entry point in `src/sparse_reorder_nd_internal.h`
- define the resolved policy structs/enums in `src/sparse_graph_internal.h`
  because the graph/reorder consumers need them
- resolve public options plus compatibility overrides in `src/sparse_analysis.c`
  and pass typed policy into the internal ND entry point

Interpretation:

- Sprint 61 does not need a public reorder API redesign
- it needs one internal bridge that lets `sparse_analyze(...)` bypass raw
  process-global parsing on the selected path

#### 3. The first code batch should avoid `src/sparse_graph.c` and `src/sparse_graph_refine.c`

The exact env-var consumer map shows:

- selected public-candidate controls are consumed in:
  - `src/sparse_analysis.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_reorder_nd.c`
- the heaviest FM tuning and runtime-save/restore coupling sits in:
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`

That means the first implementation batch should intentionally avoid:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

unless the code landing proves they are required for compilation or policy
threading.

Interpretation:

- the selected public-candidate path can stay narrower than the entire graph
  subsystem
- keeping `src/sparse_graph.c` and `src/sparse_graph_refine.c` out of the first
  batch materially lowers risk

#### 4. The exact Day 6 versus Day 7 split is now fixed

Recommended Day 6 batch:

- public header widening in:
  - `include/sparse_analysis.h`
- public-option resolution and compatibility translation in:
  - `src/sparse_analysis.c`
- internal policy scaffolding in:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
- first selected controls:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`

Why this is the strongest first batch:

- it exercises the full public-to-internal precedence path
- it stays close to the current `sparse_analyze(...)` and ND recursion seam
- it avoids starting with the broader graph partitioner policy set

Recommended Day 7 batch:

- finish the remaining selected public-candidate controls in:
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_reorder_nd.c`
- remaining selected controls:
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
- proof-surface follow-through:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - bounded `tests/test_integration.c` only if the public analysis lifecycle
    contract wording or behavior needs direct pinning

Interpretation:

- Day 6 proves the bridge and precedence model
- Day 7 completes the rest of the selected public analysis/reorder surface

#### 5. The exact touched-file plan is now explicit

Expected Day 6-7 touched files:

- public:
  - `include/sparse_analysis.h`
- internal headers:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
- implementation:
  - `src/sparse_analysis.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_separator.c`
- likely docs follow-through after code lands:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
- likely proof follow-through:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - optional bounded `tests/test_integration.c`

Expected non-touch set for the first landing unless implementation pressure
forces it:

- `include/sparse_reorder.h`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_svd.c`

Interpretation:

- the first landing now has a genuinely bounded file set
- the selected non-touch set is part of the safety fence, not an afterthought

#### 6. The code-batch non-goals are now operational

Do not widen the Day 6-7 implementation batch into:

- public FM tuning controls
- debug/profile option migration
- compile-time threshold policy cleanup
- new one-shot direct-family option surfaces
- generic repo-wide configuration helpers
- packaging/platform work
- broader docs simplification beyond touched-surface truthfulness follow-through

### Day 5 Close

Sprint 61 now has an exact landing design for the first code batch:

- the minimum viable public API addition is explicit
- the internal policy bridge is explicit
- the Day 6 versus Day 7 split is explicit
- the touched-file and non-touch file sets are explicit
- the code-batch non-goal fence is explicit before any header or implementation
  edits begin

## Day 6

**Objective:** Land the first bounded Phase 1 configuration-modernization code
batch by widening `sparse_analysis_opts_t` for the first selected
analysis/reorder controls, translating those typed fields through a resolved
internal ND policy seam, preserving legacy env-var compatibility when the typed
fields remain unspecified, and proving typed-over-env precedence on the live ND
and supernodal-postorder paths.

### Commands Run

1. Re-read the Day 5 landing design and the live Day 6 touch set:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/artifacts/day5-header-and-internal-surface-landing-design.md`
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_reorder_nd.c`
2. Edit the public typed-option surface, internal policy bridge, ND consumer,
   and proof surface:
   - `include/sparse_analysis.h`
   - `src/sparse_graph_internal.h`
   - `src/sparse_reorder_nd_internal.h`
   - `src/sparse_analysis.c`
   - `src/sparse_reorder_nd.c`
   - `tests/test_reorder_nd.c`
3. Review the landing diff:
   - `git diff -- include/sparse_analysis.h src/sparse_graph_internal.h src/sparse_reorder_nd_internal.h src/sparse_analysis.c src/sparse_reorder_nd.c tests/test_reorder_nd.c`
4. Run the required direct gate:
   - `make format`
   - `make lint`
   - `make test`
5. Resolve a stale-object false negative caused by the header layout change:
   - `make clean`
   - `make format`
   - `make lint`
   - `make test`
6. Run the stronger reviewed gate:
   - `make quality-review-full`
7. Capture the final touched-surface stats:
   - `git diff --stat`
   - `wc -l include/sparse_analysis.h src/sparse_graph_internal.h src/sparse_reorder_nd_internal.h src/sparse_analysis.c src/sparse_reorder_nd.c tests/test_reorder_nd.c`

### Day 6 Findings

#### 1. The first public typed-option widening now exists on `sparse_analysis_opts_t`, not in a new top-level configuration object

The landed public addition is bounded to one nested analysis/reorder sub-struct
inside `sparse_analysis_opts_t`:

- `sparse_analysis_reorder_opts_t`
  - `supernodal_postorder`
  - `nd_root_bisect`
  - `nd_root_bisect_max_n`

And the first public zero-init-safe enums are now explicit:

- `sparse_analysis_supernodal_postorder_t`
  - `DEFAULT`
  - `OFF`
  - `ON`
- `sparse_analysis_nd_root_bisect_t`
  - `DEFAULT`
  - `MULTILEVEL`
  - `SPECTRAL`

Interpretation:

- Phase 1 now has a real caller-facing typed configuration seam
- the widening stays on the existing direct-analysis configuration front door
- the Day 6 batch did not widen `include/sparse_reorder.h`
- the public surface stayed inside the Day 5 fence

#### 2. The public-to-internal precedence bridge now exists and matches the Day 4 contract

`src/sparse_analysis.c` now resolves the first selected controls through one
explicit internal policy translation step before dispatching ND ordering:

- explicit typed option when the field is not left at `DEFAULT` / `0`
- otherwise legacy compatibility override from the canonical env var
- otherwise internal typed policy default

This is now applied for:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`

The bridge is represented through the new internal policy types in
`src/sparse_graph_internal.h` and the new internal ND entry declaration in
`src/sparse_reorder_nd_internal.h`.

Interpretation:

- the precedence rule is no longer just documented; it now exists in code
- public typed options and legacy env compatibility no longer compete
  implicitly
- later Sprint 61 controls can extend the same bridge instead of inventing a
  parallel pattern

#### 3. `sparse_analyze(...)` and public `sparse_reorder_nd(...)` now share one ND policy model without changing the public reorder API

The Day 6 bridge kept the public reorder API intact:

- `sparse_reorder_nd(...)` remains the public compatibility wrapper

And added the internal policy-aware path needed by `sparse_analyze(...)`:

- `sparse_reorder_nd_with_policy(...)`

`src/sparse_reorder_nd.c` now threads a resolved
`sparse_graph_nd_policy_t` through the root spectral-bisect decision and the
ND recursion path, while the public wrapper still honors the legacy env-var
behavior when callers do not go through `sparse_analysis_opts_t`.

Interpretation:

- Sprint 61 now has the exact internal bridge the Day 5 design called for
- the Day 6 batch modernized control placement without widening the public
  reorder API
- compatibility behavior for legacy callers was preserved

#### 4. The typed-over-env precedence proof is now real on the live ND and supernodal-postorder paths

`tests/test_reorder_nd.c` now carries bounded Day 6 precedence tests for the
first selected controls:

- typed `nd_root_bisect = MULTILEVEL` overriding env
  `SPARSE_ND_ROOT_BISECT=spectral`
- typed `nd_root_bisect_max_n = 50000` overriding env
  `SPARSE_ND_ROOT_BISECT_MAX_N=1`
- typed `supernodal_postorder = OFF` overriding env
  `SPARSE_SUPERNODAL_POSTORDER=on`

The tests are intentionally proof-oriented rather than generic parser checks:

- they exercise `sparse_analyze(...)`
- they compare actual resulting permutations
- they use `SKIP_TEST` only when the matrix/data path cannot distinguish the
  compared strategies meaningfully

Interpretation:

- the Sprint 61 control-plane contract is now pinned on behavior, not just on
  field parsing
- the first Day 6 proof surface stayed bounded to `tests/test_reorder_nd.c`
- `tests/test_graph.c` and `tests/test_integration.c` remain available for the
  deeper Day 7 controls if needed

#### 5. The first `make test` failure was a stale-build artifact, not a Day 6 logic regression

The initial post-edit `make test` run produced false `SPARSE_ERR_BADARG`
failures in integration paths that pass `sparse_analysis_opts_t` into already
compiled direct-solver consumers.

The root cause was:

- `include/sparse_analysis.h` changed the `sparse_analysis_opts_t` layout
- the normal Makefile path did not fully rebuild every dependent object from
  that header change
- some objects were still compiled against the old struct layout

After `make clean`, the full gate passed cleanly.

Interpretation:

- the Day 6 code landing itself is sound
- the failure was a build dependency/staleness issue, not a contract mismatch
- the validated Day 6 baseline should be treated as the clean-rebuild result

### Day 6 Validation

After a clean rebuild:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 280.04 sec`

Representative retained proof points:

- `test_reorder_nd` passed with the new typed-over-env precedence tests
- the full reviewed path rebuilt and passed the live graph/reorder surfaces:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

Touched-surface diff summary:

- `6` files changed
- `428` insertions
- `36` deletions

### Day 6 Close

Sprint 61 now has the first landed Phase 1 configuration-modernization batch:

- the first public typed analysis/reorder options are live
- the public-to-internal precedence bridge is live
- the internal ND policy-aware entry is live
- legacy env-var compatibility remains intact when typed fields stay
  unspecified
- the first typed-over-env precedence proof is live
- the Day 7 queue is now narrowed to the deeper selected ND/FM-adjacent
  controls instead of the initial bridge slice

## Day 7

**Objective:** Complete the remaining bounded Sprint 61 Phase 1
analysis/reorder typed-configuration surface by landing the deeper selected ND
controls on `sparse_analysis_opts_t`, threading them through the resolved
internal ND policy seam, preserving legacy env-var compatibility for
unspecified fields, and proving typed-over-env precedence on the live
coarsening, coarsest-bisection, separator-lift-strategy, and
separator-lift-weight paths.

### Commands Run

1. Re-read the Day 5-Day 6 landing design and the remaining Day 7 touch set:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_61/artifacts/day5-header-and-internal-surface-landing-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_61/artifacts/day6-typed-analysis-reorder-option-batch1.md`
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_graph_internal.h`
   - `sed -n '1,260p' src/sparse_reorder_nd.c`
2. Edit the remaining public typed-option surface, internal ND policy seam,
   graph/ND consumers, and proof surface:
   - `include/sparse_analysis.h`
   - `src/sparse_analysis.c`
   - `src/sparse_graph_internal.h`
   - `src/sparse_graph_coarsen.c`
   - `src/sparse_graph_bisect.c`
   - `src/sparse_graph_separator.c`
   - `src/sparse_reorder_nd.c`
   - `tests/test_reorder_nd.c`
3. Start from a clean tree before rerunning the code gate because the
   public-analysis header widened again:
   - `make clean`
4. Run the required direct gate:
   - `make format`
   - `make lint`
   - `make test`
5. Run the stronger reviewed gate:
   - `make quality-review-full`
6. Capture the final touched-surface stats:
   - `git diff --stat`
   - `wc -l include/sparse_analysis.h src/sparse_analysis.c src/sparse_graph_internal.h src/sparse_graph_coarsen.c src/sparse_graph_bisect.c src/sparse_graph_separator.c src/sparse_reorder_nd.c tests/test_reorder_nd.c`

### Day 7 Findings

#### 1. The remaining selected public typed analysis/reorder controls now exist on `sparse_analysis_opts_t`

`include/sparse_analysis.h` now completes the selected Sprint 61 Phase 1
public analysis/reorder widening on `sparse_analysis_reorder_opts_t` by adding:

- `nd_coarsening`
- `nd_coarsest_bisection`
- `nd_sep_lift_strategy`
- `nd_sep_lift_weight`

The new public zero-init-safe enums are now explicit:

- `sparse_analysis_nd_coarsening_t`
  - `DEFAULT`
  - `HEAVY_EDGE`
  - `HCC`
- `sparse_analysis_nd_coarsest_bisection_t`
  - `DEFAULT`
  - `DEFAULT_ROUTING`
  - `SPECTRAL`
  - `GGGP`
  - `BRUTE`
- `sparse_analysis_nd_sep_lift_strategy_t`
  - `DEFAULT`
  - `SMALLER_WEIGHT`
  - `BALANCED_BOUNDARY`
  - `PER_VERTEX`
  - `PER_VERTEX_BALANCE`
  - `PER_VERTEX_DEGREE`
  - `PER_VERTEX_FIXED_K`
- `sparse_analysis_nd_sep_lift_weight_t`
  - `DEFAULT`
  - `HYBRID`
  - `BALANCE`
  - `DEGREE`

Interpretation:

- the public Phase 1 control-plane widening is now complete for the selected
  analysis/reorder controls
- the widening stayed on `sparse_analysis_opts_t`, not in a new top-level
  configuration object
- `include/sparse_reorder.h` still did not widen

#### 2. The resolved internal ND policy model now covers the full selected Phase 1 set

`src/sparse_graph_internal.h` now extends `sparse_graph_nd_policy_t` with the
remaining selected controls:

- `nd_coarsening`
- `nd_coarsest_bisection`
- `nd_sep_lift_strategy`
- `nd_sep_lift_weight`

It also now owns the internal typed policy enums used by the graph/ND
consumers:

- `sparse_graph_nd_coarsest_bisection_mode_t`
- `sparse_graph_nd_sep_lift_strategy_mode_t`
- `sparse_graph_nd_sep_lift_weight_mode_t`

And the internal override hooks needed to thread the resolved policy through
the existing graph subsystem:

- `sparse_graph_coarsening_override_begin(...)`
- `sparse_graph_coarsening_override_end(...)`
- `sparse_graph_coarsest_bisection_override_begin(...)`
- `sparse_graph_coarsest_bisection_override_end(...)`
- `sparse_graph_sep_lift_override_begin(...)`
- `sparse_graph_sep_lift_override_end(...)`

Interpretation:

- the Phase 1 bridge is no longer partial; the full selected ND policy set now
  exists as one internal resolved-policy model
- the public typed surface and the graph/ND implementation no longer need
  parallel ad hoc translations for these controls
- later Sprint 61+ work can extend this seam instead of inventing a new one

#### 3. The remaining graph/ND consumers now honor typed policy first while preserving bounded compatibility behavior

The Day 7 implementation widened the actual graph/ND consumer path:

- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- `src/sparse_graph_separator.c`
- `src/sparse_reorder_nd.c`

The effective precedence chain now holds across the selected Day 7 controls:

1. explicit typed option
2. legacy compatibility override, only when the typed field remains
   unspecified/default
3. internal typed policy default

Important bounded behavior was preserved:

- the existing forced heavy-edge fallback for degenerate separator recovery
  still wins when that internal recovery path is active
- public compatibility callers using `sparse_reorder_nd(...)` still keep the
  legacy env-var behavior
- typed policy callers using `sparse_analyze(...)` now override the legacy env
  path without a public reorder-API redesign

Interpretation:

- the public typed options now control the real graph/ND behavior instead of
  stopping at the analysis entry surface
- compatibility behavior for legacy env-var callers remains intact when typed
  fields stay unspecified
- the Sprint 61 contract from Day 4-Day 5 is now fully live in code

#### 4. The typed-over-env precedence proof is now real on the remaining selected ND controls

`tests/test_reorder_nd.c` now carries four new bounded Day 7 precedence tests:

- `test_analysis_typed_nd_coarsening_overrides_env`
- `test_analysis_typed_nd_coarsest_bisection_overrides_env`
- `test_analysis_typed_nd_sep_lift_strategy_overrides_env`
- `test_analysis_typed_nd_sep_lift_weight_overrides_env`

These remain behavior-level proofs rather than parser-only checks:

- they drive `sparse_analyze(...)`
- they compare resulting symbolic behavior through actual fill-sensitive
  analysis output
- they use `SKIP_TEST` only when the chosen matrix/data path cannot
  meaningfully distinguish the compared strategies

Representative live proof points recorded in the reviewed path:

- `bcsstk14`: HEM `nnz(L) = 129576`, HCC+Kuu-safe `nnz(L) = 130422`
- `bcsstk04` fixed-K weight path:
  - `hybrid = 3679`
  - `balance = 4469`
  - `degree = 4613`

Interpretation:

- the typed-over-env precedence contract now exists on the deeper ND policy
  controls, not only on the Day 6 bridge slice
- the proof surface remained bounded to `tests/test_reorder_nd.c`
- no broad proof-surface churn in `tests/test_graph.c` or
  `tests/test_integration.c` was needed for this batch

#### 5. Sprint 61 Phase 1 now has one coherent selected analysis/reorder modernization slice instead of a partial bridge plus a deferred backlog

Day 6 established the first public typed bridge for:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`

Day 7 now completes the selected remaining public-candidate controls:

- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

Interpretation:

- Sprint 61 now hands off one coherent Phase 1 typed analysis/reorder control
  surface
- the remaining Epic 6 configuration queue is now narrower and more honestly
  deferred
- the non-goal fence still holds:
  - no public FM tuning controls
  - no debug/profile option migration
  - no repo-wide configuration helper layer
  - no packaging/platform widening

### Day 7 Validation

Required gate after a clean rebuild:

- `make clean`
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 320.27 sec`

Representative retained proof points:

- `test_reorder_nd` passed with the new typed-over-env Day 7 precedence tests
- the reviewed path rebuilt and passed the live graph/reorder proof surface:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

One non-blocking Day 7 note is explicit:

- the reviewed CMake rebuild again emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
  cleanly and passed all parity gates

Touched-surface diff summary:

- `8` files changed
- `666` insertions
- `97` deletions

### Day 7 Close

Sprint 61 now has the full selected Phase 1 analysis/reorder typed-option
landing:

- the remaining selected public typed controls are live
- the full selected ND policy bridge is live
- the graph/ND consumers now honor typed policy first without reopening the
  public reorder API
- legacy env-var compatibility remains intact for unspecified typed fields
- the deeper selected typed-over-env precedence proof is live
- the remaining Epic 6 configuration queue is now narrower than the original
  “too env-var driven” backlog

## Day 8

**Objective:** Re-audit the remaining analysis-time and postorder-adjacent
env-var controls after the Day 6-7 landing, separate what still justifies
Sprint 61 movement from what should remain compatibility-only or explicitly
defer, and fix the exact bounded Day 9 landing target before any further code
touches.

### Commands Run

1. Re-read the landed Day 6-7 contract and the Sprint 61 Day 8 plan scope:
   - `sed -n '260,420p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
   - `tail -n 180 docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_61/artifacts/day6-typed-analysis-reorder-option-batch1.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_61/artifacts/day7-typed-analysis-reorder-option-batch2.md`
2. Re-inventory the still-live analysis/postorder-adjacent env-var seams:
   - `rg -n "getenv\\(|SPARSE_ND_|SPARSE_SUPERNODAL_POSTORDER|SPARSE_ND_SUPERNODAL_POSTORDER|SPARSE_FM_|PROFILE|DEBUG" src/sparse_analysis.c src/sparse_graph*.c src/sparse_reorder_nd.c src/sparse_reorder_amd_qg.c include/sparse_analysis.h README.md docs/maintainer_guide.md`
3. Inspect the remaining strongest implementation seams directly:
   - `sed -n '1,260p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_graph_internal.h`
   - `sed -n '60,220p' src/sparse_graph_coarsen.c`
   - `sed -n '460,560p' src/sparse_graph_coarsen.c`
   - `sed -n '430,520p' src/sparse_graph_bisect.c`
   - `sed -n '1,120p' src/sparse_graph_separator.c`
   - `sed -n '500,760p' src/sparse_reorder_nd.c`
   - `sed -n '1,120p' src/sparse_reorder_amd_qg.c`
4. Reconfirm the current proof/docs distribution for the residual controls:
   - `rg -n "SPARSE_ND_COARSEN_FLOOR_RATIO|SPARSE_ND_COARSENING_CV_FALLTHROUGH|SPARSE_ND_SUPERNODAL_POSTORDER|SPARSE_ND_PROFILE|SPARSE_QG_PROFILE|SPARSE_HCC_DEBUG" tests README.md docs include src`
5. Capture the current Sprint 61 branch shape and artifact set:
   - `git diff --stat master...HEAD`
   - `ls docs/planning/EPIC_6/SPRINT_61/artifacts`

### Day 8 Findings

#### 1. The strongest remaining analysis-time controls are now concentrated in one coarsening-policy seam, not spread across the whole ND path

After the Day 6-7 landing, the public typed analysis/reorder surface already
owns:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

What remains strongest on the analysis-time path is now narrow:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Both still sit in the coarsening/hierarchy seam:

- hierarchy stopping threshold in `sparse_graph_hierarchy_build(...)`
- HCC-to-HEM fallthrough threshold inside `graph_coarsen_with_strategy(...)`

Interpretation:

- the post-Day-7 queue is no longer a generic "more ND env vars" backlog
- the next justified Sprint 61 move is a bounded coarsening-policy slice
- the bisect, separator-lift, and supernodal-postorder paths are no longer the
  strongest residual Sprint 61 targets

#### 2. The legacy supernodal-postorder alias is still real, but it is now a compatibility-only seam rather than another integration target

`src/sparse_analysis.c` still accepts:

- canonical:
  - `SPARSE_SUPERNODAL_POSTORDER`
- legacy compatibility alias:
  - `SPARSE_ND_SUPERNODAL_POSTORDER`

The alias still matters for back-compat with older Sprint 28 captures and
advisory recipes, but it no longer justifies new public typed work:

- the canonical control is already on `sparse_analysis_opts_t`
- the alias is already explicitly subordinate to the canonical name
- removing or widening the alias would create churn without productization
  value

Interpretation:

- postorder itself is no longer an active Sprint 61 integration problem
- the alias should stay compatibility-only for now
- Day 9 should not widen into another postorder-specific batch

#### 3. The remaining debug/profile seams are real, but they are not Sprint 61 Phase 1 control-surface candidates

Still-live instrumentation/debug env vars include:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`

These remain implementation-support surfaces rather than caller-facing product
controls:

- they emit profiling/debug traces
- they do not define stable algorithm-choice intent
- they are still coupled to implementation-local instrumentation and debug
  output paths

Interpretation:

- they should not move into the public typed option surface in Sprint 61
- if they need modernization later, it should happen under a tooling/support
  or internal-policy lane, not in this Phase 1 caller-facing sprint
- Day 9 should explicitly defer them

#### 4. The FM family remains adjacent but still out of scope for this Sprint 61 slice

The live `SPARSE_FM_*` family still exists in:

- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`

But after the Day 6-7 landing it is still not the strongest remaining
analysis/postorder target:

- it is broader
- more tightly coupled to the refinement pipeline
- more debug/strategy-heavy
- riskier to widen without reopening the Sprint 61 non-goal fence

Interpretation:

- the FM family remains explicitly deferred after Day 8
- Sprint 61 should not drift from coarsening-policy cleanup into refinement
  strategy modernization
- the Day 9 target should stay narrower than the original Day 3 inventory

#### 5. The strongest Day 9 landing target is now one mixed public/internal coarsening-policy batch

The strongest bounded next slice is:

- public typed candidate:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- internal typed-policy candidate:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Why this is the best next batch:

- both are still genuinely analysis-time controls
- both live in the same coarsening seam
- both fit the existing Day 6-7 resolved-policy bridge pattern
- neither requires reopening the full FM or instrumentation surface

Recommended Day 9 target:

- widen `sparse_analysis_opts_t.reorder_opts` only if the field earns a real
  caller-facing story:
  - `nd_coarsen_floor_ratio`
- keep the HCC CV fallthrough threshold in the internal resolved-policy lane
  first:
  - no debug/profile widening
  - no FM strategy widening
- preserve env-var compatibility when the new field or policy remains
  unspecified

Likely Day 9-10 touched surfaces:

- `include/sparse_analysis.h`
- `src/sparse_analysis.c`
- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `src/sparse_reorder_nd.c`
- `tests/test_reorder_nd.c`
- optional bounded `tests/test_graph.c` only if the proof burden genuinely
  requires it

#### 6. The landed typed path exposed one new maintenance risk: default and compatibility drift now concentrates in a smaller number of duplicated parser seams

The Day 6-7 landing intentionally kept:

- public typed resolution in `src/sparse_analysis.c`
- env-compat behavior for direct graph entry surfaces inside the graph modules

This preserved compatibility and minimized public churn, but it also makes the
remaining risk more precise:

- if the remaining defaults or compat parsers drift, they will now drift in a
  small number of high-value coarsening-policy seams rather than across the
  whole ND pipeline

Interpretation:

- this is a manageable risk, not a blocker
- it argues for a narrow Day 9-10 batch that closes the strongest remaining
  coarsening-policy gap cleanly
- it argues against broadening into multiple unrelated residual env vars at
  once

### Day 8 Close

Sprint 61's post-Day-7 queue is now materially smaller and more concrete:

- the strongest remaining analysis-time controls are now narrowed to the
  coarsening-policy seam
- the supernodal-postorder legacy alias is compatibility-only
- debug/profile controls are explicitly deferred
- the FM family remains deferred
- the exact Day 9 landing target is now a mixed public/internal
  coarsening-policy batch instead of another broad ND sweep
