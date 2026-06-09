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
