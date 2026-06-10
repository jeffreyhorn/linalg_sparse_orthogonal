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

## Day 9

**Objective:** Convert the Day 8 residual coarsening-policy queue into one
exact Day 10 implementation fence by defining the public/internal field split,
the precedence and compatibility behavior, the touched-file set, and the
explicit deferred-control list before more code lands.

### Commands Run

1. Re-read the Day 9-Day 10 plan slice and the Day 8 audit:
   - `sed -n '300,380p' docs/planning/EPIC_6/SPRINT_61/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_6/SPRINT_61/WORKING_NOTES.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_61/artifacts/day8-post-landing-analysis-postorder-audit.md`
2. Reinspect the live public option surface and the residual coarsening seam:
   - `sed -n '100,220p' include/sparse_analysis.h`
   - `sed -n '120,220p' src/sparse_graph_coarsen.c`
   - `sed -n '420,520p' src/sparse_graph_internal.h`
   - `sed -n '280,360p' tests/test_reorder_nd.c`
3. Reconfirm the branch cleanliness before a docs-only design landing:
   - `git status --short`
   - `git branch --show-current`

### Day 9 Findings

#### 1. The exact Day 10 control subset is now fixed: one public coarsening-threshold field plus one internal HCC fallback policy field

The Day 8 audit left two residual analysis-time controls:

- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

The Day 9 design now separates them explicitly:

- move publicly in Day 10:
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- keep internal-typed in Day 10:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Why this split is now fixed:

- `COARSEN_FLOOR_RATIO` has a real caller-facing story:
  it controls the coarse-level stopping threshold used by the multilevel ND
  hierarchy and already appears in public-facing algorithm notes
- `COARSENING_CV_FALLTHROUGH` is still implementation-shaped:
  it is an HCC-to-HEM safeguard threshold, not a stable user-level ordering
  intent knob

Interpretation:

- Day 10 should not treat the two residual knobs symmetrically
- the public/internal ownership split is now exact enough to implement without
  reopening the Day 8 audit

#### 2. The public Day 10 widening should stay on `sparse_analysis_reorder_opts_t` and use the existing zero-init-safe scalar pattern

The strongest public field shape is:

- `idx_t nd_coarsen_floor_ratio`

Placement:

- `include/sparse_analysis.h`
- inside `sparse_analysis_reorder_opts_t`

Semantics:

- `0` means unspecified/default, preserving the current precedence model
- positive values request an explicit typed override
- negative values remain invalid

Why this is the strongest public shape:

- it matches the existing scalar pattern already used by
  `nd_root_bisect_max_n`
- it avoids inventing a new nested coarsening-options struct for a one-field
  Phase 1 batch
- it keeps the public API change minimal and coherent with the Day 6-7 design

Recommended public field wording:

- "Optional ND coarsening floor ratio divisor. Use 0 to leave unspecified."

Interpretation:

- Day 10 should widen the existing reorder-options struct, not create a second
  Phase 1 configuration object
- the public scalar should follow the established Sprint 61 zero-init-safe
  convention exactly

#### 3. The internal typed-policy addition should carry the exact semantics of today's coarsening implementation rather than pretending to be a broader public control

The Day 10 internal addition should extend `sparse_graph_nd_policy_t` with
two fields:

- `idx_t nd_coarsen_floor_ratio`
- `double nd_coarsening_cv_fallthrough`

Internal defaults:

- `nd_coarsen_floor_ratio = 100`
- `nd_coarsening_cv_fallthrough = 0.30`

Internal validation/meaning:

- floor ratio:
  - valid typed/internal domain: `1..100000`
- HCC CV fallthrough:
  - valid typed/internal domain: `0.0..100.0`
  - `0.0` still means "disable the fallthrough threshold check", matching the
    current implementation

Why this is the strongest internal shape:

- it matches the live implementation semantics exactly
- it keeps the HCC fallthrough threshold out of the public API while still
  removing it from direct env-only selection on the `sparse_analyze(...)`
  path
- it avoids creating a generic "all coarsening knobs" internal struct when the
  only remaining justified fields are these two

Interpretation:

- Day 10 should modernize the residual coarsening seam without pretending
  Sprint 61 is solving the broader FM/control-plane problem

#### 4. The precedence and compatibility rules for Day 10 are now exact

Public floor-ratio field:

1. explicit typed `nd_coarsen_floor_ratio` when > 0
2. legacy compatibility override from `SPARSE_ND_COARSEN_FLOOR_RATIO` when the
   typed field is 0 / unspecified
3. internal typed default = `100`

Internal HCC CV fallthrough field:

1. internal resolved-policy value
2. compatibility override from `SPARSE_ND_COARSENING_CV_FALLTHROUGH` only when
   the internal field remains unset by the caller path
3. internal typed default = `0.30`

Important Day 10 compatibility rule:

- direct legacy callers going through public `sparse_reorder_nd(...)` still
  keep the env-var behavior unless and until a future sprint widens that API

Important Day 10 non-rule:

- do not introduce a new public typed field for the CV threshold

Interpretation:

- the public precedence story remains consistent with Day 6-7
- the internal precedence story is now explicit instead of implicit
- Day 10 can land without ambiguity around legacy env support

#### 5. The Day 10 touched-file fence is now precise and narrower than the Day 8 candidate set

Required Day 10 touch set:

- public:
  - `include/sparse_analysis.h`
- internal/public resolution:
  - `src/sparse_analysis.c`
  - `src/sparse_graph_internal.h`
- implementation:
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_reorder_nd.c`
- proof:
  - `tests/test_reorder_nd.c`

Optional only if proof burden forces it:

- `tests/test_graph.c`

Explicit non-touch set for Day 10:

- `src/sparse_graph_bisect.c`
- `src/sparse_graph_separator.c`
- `src/sparse_graph.c`
- `src/sparse_graph_refine.c`
- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

Interpretation:

- the next code batch is now narrower than the Day 8 candidate envelope
- there is no reason for Day 10 to reopen the bisect/separator slices or drift
  into docs before the code lands

#### 6. The regression obligation for Day 10 is now explicit and bounded

Required proof additions in Day 10:

- typed-over-env precedence proof for `nd_coarsen_floor_ratio`
- stable-default proof when the typed field remains unspecified
- bounded proof that the `sparse_analyze(...)` path now resolves the internal
  HCC fallthrough threshold through the same resolved-policy seam, not by
  directly competing parser paths

Preferred proof home:

- `tests/test_reorder_nd.c`

Optional `tests/test_graph.c` widening only if needed for one of:

- direct partition-shape differentiation that cannot be expressed cleanly
  through `sparse_analyze(...)`
- a tighter proof of the HCC fallthrough semantics

Interpretation:

- the proof burden is real but still bounded
- Day 10 should default to strengthening `tests/test_reorder_nd.c` first

### Day 9 Deferred-Control List

Stay compatibility-only for now:

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias

Explicitly defer:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`
- any widening of `sparse_reorder_nd(...)` itself
- any repo-wide configuration helper layer

### Day 9 Close

Sprint 61 now has an exact Day 10 implementation fence instead of a generic
“remaining analysis-time controls” queue:

- the exact public field to add is fixed
- the exact internal coarsening-policy fields are fixed
- the precedence and compatibility rules are fixed
- the required touched-file set is fixed
- the proof obligations are fixed
- the deferred-control list is now explicit rather than implicit

## Day 10

**Objective:** Land the remaining bounded analysis-time control batch by
moving `SPARSE_ND_COARSEN_FLOOR_RATIO` onto the typed
`sparse_analysis_reorder_opts_t` surface, resolving
`SPARSE_ND_COARSENING_CV_FALLTHROUGH` through the internal ND policy seam, and
proving the preserved precedence/compatibility rules.

1. Re-read the Day 9 design fence and keep the touch set exact:

- public:
  - `include/sparse_analysis.h`
- resolution/policy:
  - `src/sparse_analysis.c`
  - `src/sparse_graph_internal.h`
- implementation:
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_reorder_nd.c`
- proof:
  - `tests/test_reorder_nd.c`

2. Land the public/internal field split exactly as designed:

- added public typed field on `sparse_analysis_reorder_opts_t`:
  - `idx_t nd_coarsen_floor_ratio`
- completed the internal ND policy seam in `sparse_graph_nd_policy_t` with:
  - `idx_t nd_coarsen_floor_ratio`
  - `double nd_coarsening_cv_fallthrough`
- kept zero-init safety and the bounded public scalar contract:
  - `0` remains unspecified/default
  - positive floor-ratio values become explicit typed overrides
  - negative or out-of-range public values are rejected

3. Route both remaining controls through one resolved-policy bridge instead of
   direct competing parser paths:

- `src/sparse_analysis.c` now resolves:
  - typed `nd_coarsen_floor_ratio`
  - legacy `SPARSE_ND_COARSEN_FLOOR_RATIO` compatibility override
  - internal default `100`
- the analysis path now also resolves internal
  `SPARSE_ND_COARSENING_CV_FALLTHROUGH` through the same ND policy object,
  instead of leaving that seam as a separate late parser
- `src/sparse_reorder_nd.c` now carries both residual controls through the
  policy-aware ND entry point for the explicit analysis lifecycle

4. Complete the coarsening-side internal override plumbing without widening the
   public reorder API:

- `src/sparse_graph_coarsen.c` now owns thread-local begin/end overrides for:
  - coarsening floor-ratio divisor
  - HCC CV fallthrough threshold
- precedence at the coarsening site is now exact:
  - explicit internal override
  - legacy compatibility env var
  - internal default
- `sparse_reorder_nd(...)` remains a compatibility wrapper rather than a newly
  widened public configuration API

5. Prove the landed precedence behavior on the live ND analysis path:

- added `tests/test_reorder_nd.c` proof:
  - `test_analysis_typed_nd_coarsen_floor_ratio_overrides_env`
  - `test_analysis_nd_coarsening_cv_fallthrough_env_affects_policy_path`
- preserved the existing Day 6-7 typed-over-env proof set for:
  - root bisection
  - root bisection max-N
  - coarsening strategy
  - coarsest bisection
  - separator-lift strategy
  - separator-lift weight
  - supernodal postorder

### Day 10 Findings

#### 1. Sprint 61 Phase 1 is now materially complete for the justified public analysis/reorder controls

The highest-value public analysis-time controls identified on Days 3-5 are now
typed on `sparse_analysis_reorder_opts_t`:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`

Interpretation:

- the remaining Sprint 61 queue is no longer “move one more obvious public ND
  control”
- the broad env-var problem has now been reduced to deferred FM/debug/profile
  seams plus the compatibility-only legacy alias

#### 2. The residual HCC fallback threshold is no longer an ad hoc late parser on the analysis path

Before Day 10:

- `SPARSE_ND_COARSENING_CV_FALLTHROUGH` still lived only as a direct env read
  inside the coarsening implementation

After Day 10:

- the explicit analysis lifecycle resolves that threshold into the internal
  `sparse_graph_nd_policy_t`
- the ND driver passes it through begin/end override plumbing just like the
  other resolved policy fields
- the low-level env parser remains only as the bounded compatibility lane when
  no explicit internal override is active

Interpretation:

- the analysis path now has one coherent control-plane story instead of one
  lingering special-case env seam
- this closes the strongest remaining Day 8-Day 9 “post-landing analysis” gap

#### 3. The landed proof burden stayed bounded to `tests/test_reorder_nd.c`

Day 9 left open whether Day 10 would need to widen `tests/test_graph.c`.

It did not.

The new proof burden fit inside `tests/test_reorder_nd.c`:

- typed floor-ratio precedence is proven there
- HCC CV-threshold effect on the analysis path is proven there

Interpretation:

- the selected proof home was strong enough
- Day 10 did not have to reopen lower-level graph-test ownership

#### 4. The preserved non-goal fence stayed exact

Day 10 did not widen into:

- public FM tuning controls
- debug/profile option migration
- repo-wide configuration helper layers
- packaging/platform work
- backend/AUTO policy work
- public `sparse_reorder_nd(...)` signature changes

Interpretation:

- Sprint 61 still reads as one coherent Phase 1 configuration-modernization
  sprint rather than the start of a larger cross-cutting rewrite

### Day 10 Validation

Because `*.c` / `*.h` changed, I ran:

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
- `Total Test time (real) = 332.33 sec`

Focused retained proof points:

- `test_reorder_nd` passed with the two new Day 10 proofs:
  - `test_analysis_typed_nd_coarsen_floor_ratio_overrides_env`
  - `test_analysis_nd_coarsening_cv_fallthrough_env_affects_policy_path`
- the full graph/reorder-sensitive rerun surface stayed clean through the
  reviewed path:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

Non-blocking Day 10 note:

- the reviewed CMake rebuild again emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the reviewed path still completed cleanly
  and passed all parity gates

### Day 10 Deferred-Control List

Stay compatibility-only for now:

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias

Explicitly defer:

- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`
- any widening of `sparse_reorder_nd(...)`
- any repo-wide configuration helper layer

### Day 10 Close

Sprint 61 Day 10 lands the remaining justified analysis-time coarsening policy
work without widening beyond the planned Phase 1 fence:

- the last public ND scalar control moved onto the typed analysis options
- the residual HCC fallback threshold moved into the internal resolved-policy
  seam for the analysis path
- the ND driver now carries both residual controls through explicit override
  plumbing instead of competing parser paths
- the proof burden stayed bounded to `tests/test_reorder_nd.c`
- the full reviewed validation contract stayed clean

## Day 11 - Compatibility Layer & Regression Sweep

### Intent

Day 11 is the planned compatibility sweep after the Day 6-Day 10 typed
analysis/reorder landing:

- re-read the full Phase 1 precedence model
- prove stable default behavior explicitly
- tighten any stale env-only wording left behind in the internal seams
- close the batch from the full reviewed baseline

### Landed Code/Test Scope

Touched code/test surfaces:

- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`
- `tests/test_reorder_nd.c`

No public API/header widening was needed on Day 11. The shipped typed control
surface from Days 6-10 stayed unchanged; this batch tightened compatibility
proof and wording around it.

### What Changed

#### 1. The remaining stale env-only commentary on the landed ND/coarsening path is gone

Updated internal commentary now matches the actual Phase 1 contract:

- resolved ND policy comes first
- legacy env vars are compatibility overrides only when the typed field stays
  unspecified
- internal defaults remain the final fallback

This cleanup landed in:

- `src/sparse_graph_internal.h`
- `src/sparse_graph_coarsen.c`

Interpretation:

- the code comments no longer imply an older “env-var-only” control story on
  paths that now run through the typed analysis/reorder bridge
- the internal wording now matches the Day 4-Day 10 precedence contract:
  - explicit typed option
  - legacy compatibility override
  - internal typed policy default

#### 2. Stable default behavior is now explicitly proven for the two residual Day 10 controls

Day 10 proved:

- typed `nd_coarsen_floor_ratio` beats the legacy env override
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH` still affects the analysis path
  through the resolved-policy seam

Day 11 added the missing stable-default proofs in `tests/test_reorder_nd.c`:

- `test_analysis_default_nd_coarsen_floor_ratio_matches_internal_default`
- `test_analysis_nd_coarsening_cv_fallthrough_default_matches_compat_value`

Interpretation:

- the floor-ratio default path is now explicitly anchored to the shipped
  internal default of `100`
- the HCC CV-fallthrough compatibility lane is now explicitly anchored to the
  shipped internal default of `0.30`
- the Phase 1 story is no longer just “typed beats env”; it now also proves
  that the unspecified/default path is stable and intentional

#### 3. The proof burden stayed bounded to the selected reorder/ND proof home

Day 11 did not need to widen into `tests/test_graph.c` or `tests/test_integration.c`.

All new proof stayed in `tests/test_reorder_nd.c`, which now covers:

- typed precedence over env for the landed public controls
- compatibility fallback/default equivalence for the residual analysis-time
  controls

Interpretation:

- the selected proof home remained strong enough through the compatibility pass
- Sprint 61 still reads as one bounded Phase 1 configuration sprint rather
  than a broad graph-test rewrite

### Post-Landing Compatibility State

After Day 11, the landed Phase 1 ND/reorder compatibility model is:

1. explicit typed option value
2. legacy compatibility override when the typed field is left unspecified
3. internal default policy

This is now explicitly proven across the strongest landed controls:

- `SPARSE_SUPERNODAL_POSTORDER`
- `SPARSE_ND_ROOT_BISECT`
- `SPARSE_ND_ROOT_BISECT_MAX_N`
- `SPARSE_ND_COARSENING`
- `SPARSE_ND_COARSEST_BISECTION`
- `SPARSE_ND_SEP_LIFT_STRATEGY`
- `SPARSE_ND_SEP_LIFT_WEIGHT`
- `SPARSE_ND_COARSEN_FLOOR_RATIO`
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH`

Compatibility-only or deferred surfaces remain unchanged:

- legacy `SPARSE_ND_SUPERNODAL_POSTORDER` alias
- `SPARSE_ND_PROFILE`
- `SPARSE_QG_PROFILE`
- `SPARSE_HCC_DEBUG`
- all `SPARSE_FM_*`

### Day 11 Validation

Because `*.c` / `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 348.47 sec`

Focused retained proof points:

- `test_reorder_nd` passed with the new Day 11 default/compatibility proofs:
  - `test_analysis_default_nd_coarsen_floor_ratio_matches_internal_default`
  - `test_analysis_nd_coarsening_cv_fallthrough_default_matches_compat_value`
- the full graph/reorder-sensitive reviewed surface stayed clean:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

Non-blocking Day 11 note:

- the reviewed CMake rebuild again emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
  cleanly and passed all parity gates

### Day 11 Close

Sprint 61 Day 11 closes the Phase 1 compatibility sweep cleanly:

- stale env-only wording is removed from the landed internal seams
- stable default behavior is now explicitly proven for the remaining residual
  analysis-time controls
- the typed/default/env compatibility story is now bounded, intentional, and
  test-backed
- the proof burden stayed in the selected reorder/ND proof home
- the full reviewed validation contract remained clean
