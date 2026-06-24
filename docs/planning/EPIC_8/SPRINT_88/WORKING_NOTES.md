# Sprint 88 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 88 baseline for Epic 8 by grounding the sprint in
the validated Sprint 87 close state, the live Sprint 88 project-plan section,
and the current front-door, example, install, benchmark-reference, and
public-narrative hotspots rather than another generic "improve docs" reset.

### Actions
- Re-read the Sprint 88 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 88 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the strongest Sprint 87 closeout context:
  - `docs/planning/EPIC_8/SPRINT_87/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_87/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly through the Day 1 parity
  rebuild:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 88
  touch surfaces across front-door docs, support references, example surfaces,
  maintained proof scripts, workflows, and highest-signal public headers.
- Opened Sprint 88 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 88 starts from the same strongest local reviewed baseline Sprint 87
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 88 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 88 is not a generic "improve docs" sprint. Its highest value is one
  bounded front-door usability and workflow-simplification package centered
  on:
  - user-journey audit
  - workflow-simplification design
  - README / tutorial batch
  - examples / workflow batch
  - support-surface consolidation
  - header / API narrative cleanup
  - validation and closeout
- The validated Sprint 87 close state already fixed the strongest handoff
  truth entering Sprint 88:
  - the repo now has a sharper static-first package/export contract
  - the maintained local consumer proof is stronger
  - the next first-tier Epic 8 contradiction is front-door usability and
    audience-boundary clarity rather than install/export truthfulness
- The strongest current maintained front-door contract is more truthful than
  earlier Epic 8 phases, but it still leaves adoption friction:
  - README, install, example references, and support surfaces still carry too
    much policy density for first adoption
  - example, benchmark, install, and maintainer references still need a
    clearer audience split
  - some highest-signal public headers still leak more internal policy than an
    adoption-focused public narrative needs
- The strongest likely Sprint 88 implementation, proof, and support surfaces
  are explicit from the live tree:
  - strongest front-door and support owners:
    - `README.md` = `1051`
    - `docs/maintainer_guide.md` = `727`
    - `INSTALL.md` = `266`
    - `benchmarks/README.md` = `399`
    - `CMakeLists.txt` = `416`
    - `Makefile` = `908`
  - strongest maintained proof and workflow surfaces:
    - `tests/test_cmake_install.sh` = `208`
    - `tests/test_install.sh` = `195`
    - `.github/workflows/ci.yml` = `223`
    - `.github/workflows/macos-ci.yml` = `104`
    - `.github/workflows/windows-ci.yml` = `63`
  - strongest example and high-signal public narrative surfaces:
    - `examples/cmake_example/main.c` = `50`
    - `examples/cmake_example/CMakeLists.txt` = `10`
    - `include/sparse_iterative.h` = `773`
    - `include/sparse_eigs.h` = `651`
    - `include/sparse_matrix.h` = `622`
    - `include/sparse_types.h` = `313`
- The strongest Day 1 clarification is now fixed:
  - Sprint 88 should not reopen package/platform semantics already stabilized
    in Sprint 87
  - Sprint 88 should first re-rank adoption-friction contradictions, then
    define one explicit front-door and support-layering contract
  - any widened usability wording must stay tied to maintained proof,
    realistic audience boundaries, and the preserved correctness contract
- The preserved Sprint 88 non-goal pressure is explicit before Day 2:
  - no package/platform contract rewrite
  - no correctness-ownership redistribution
  - no benchmark-policy rewrite detached from adoption guidance
  - no internal architectural rewrite disguised as usability cleanup
  - no support-surface churn detached from a real landed front-door seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live front-door / support / public-header hotspot map from
  direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 88 no longer starts from generic Epic 8 usability prose.
- The user-journey audit, workflow-simplification design, README/tutorial
  batch, examples/workflow batch, support-surface consolidation,
  header/API-narrative cleanup, and validation workstreams are fixed in
  writing.
- The strongest likely Sprint 88 touch surfaces and preserved non-goals are
  explicit before the validation / maintained-surface recheck begins.

## Day 2 - Validation and Maintained Support-Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
install/export, example, workflow, benchmark-reporting, and reviewed-surface
split before Sprint 88 changes any front-door, support, or public-narrative
surface.

### Actions
- Re-read the Day 2 validation-baseline expectations from
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the strongest recent validation/surface template from
  `docs/planning/EPIC_8/SPRINT_87/artifacts/day2-validation-baseline-and-maintained-consumer-surface-recheck.md`.
- Reconfirmed reviewed CMake parity directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the presence of the strongest reviewed representative binaries and
  examples that remain the main executable truth surfaces entering Sprint 88:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_reorder`
  - `./build/quality-review-cmake/test_reorder_amd_qg`
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical reporting command surface with:
  - `make -n bench-canonical-report`
- Rechecked the maintained package-proof, example, and support surfaces:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- Re-read the CI, macOS, and Windows workflow surfaces that constrain the
  current support and platform truth:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

### Findings
- Sprint 88 continues to inherit the strongest local reviewed baseline:
  - `make quality-review-full`
- The code-day and docs-day split is now fixed explicitly for this sprint:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial front-door, support-surface, or public-header narrative
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The reviewed CMake tree currently remains the strongest shared executable
  truth surface entering Sprint 88:
  - reviewed representative proof owners:
    - `./build/quality-review-cmake/test_reorder_nd`
    - `./build/quality-review-cmake/test_reorder`
    - `./build/quality-review-cmake/test_reorder_amd_qg`
    - `./build/quality-review-cmake/test_graph`
  - representative examples:
    - `./build/quality-review-cmake/example_analysis`
    - `./build/quality-review-cmake/example_basic_solve`
- Canonical benchmark reporting remains command- and script-owned rather than
  front-door-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/export proof remains script- and fixture-owned:
  - `bash tests/test_install.sh` proves the local Unix-side Make
    install/uninstall + `pkg-config` path
  - `bash tests/test_cmake_install.sh` proves the local Unix-side CMake
    install/export + `find_package(Sparse)` path
  - `examples/cmake_example/CMakeLists.txt` remains the representative
    downstream CMake consumer surface used by the CMake install/export proof
- Workflow-side support and platform truth remains intentionally narrower than
  a broad adoption or install/export parity claim:
  - Linux remains the strongest reviewed source of truth through the
    maintained reviewed paths
  - macOS carries a supplemental static-first Make install/`pkg-config`
    confidence lane only
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim a separate reviewed install-validation lane
- The strongest Day 2 clarification is now fixed:
  - reviewed CMake binaries remain the main executable truth anchor
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
  - downstream example proof remains local and bounded
  - workflow lanes remain support evidence rather than broad adoption or
    package parity claims

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked the presence of the strongest reviewed representative binaries and
  examples.
- Rechecked `make -n bench-canonical-report`.
- Rechecked `scripts/bench_canonical_report.sh`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `examples/cmake_example/CMakeLists.txt`, `README.md`, `INSTALL.md`,
  `docs/maintainer_guide.md`, `benchmarks/README.md`,
  `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and
  `.github/workflows/windows-ci.yml`.

### Day 2 Exit State
- Sprint 88 now has one explicit validation and maintained-support-surface
  contract before the user-journey audit begins.
- The live split across reviewed binaries, command-owned canonical reporting,
  script-owned install/export proof, example consumer proof, and narrower
  workflow-side platform evidence is fixed in writing.
- The highest-signal rerun set is explicit before the first adoption-friction
  rerank.

## Day 3 - User-Journey Audit

### Goal
Reduce Sprint 88's broad front-door usability problem to one ranked live
contradiction map so the sprint can choose one bounded adoption-guidance lane
instead of another generic docs or example bucket.

### Actions
- Re-read the Day 3 user-journey expectations from
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the strongest recent rerank template from
  `docs/planning/EPIC_8/SPRINT_87/artifacts/day3-release-package-gap-audit.md`.
- Re-read the current authoritative front-door and support wording in:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- Re-scanned the highest-signal adoption and public-narrative surfaces through
  targeted section searches in:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- Reconciled the usability rerank against the Sprint 87 close handoff and the
  preserved package/platform contract stabilized there.

### Findings
- Sprint 88's broad front-door usability problem is now reduced to one ranked
  live contradiction map:
  - strongest first target:
    - bounded front-door simplification centered on `README.md`, with direct
      follow-through only where the first user path currently leaks too much
      install, benchmark, or maintainer density
  - strongest second target:
    - bounded examples / workflow simplification centered on the example
      references in `README.md` plus the maintained downstream example surface
      in `examples/cmake_example/`
  - strongest third target:
    - bounded support-surface consolidation centered on `INSTALL.md`,
      `benchmarks/README.md`, and `docs/maintainer_guide.md` after the
      front-door contract is explicit
  - strongest fourth target:
    - bounded header / API narrative cleanup centered on the highest-signal
      public headers:
      - `include/sparse_iterative.h`
      - `include/sparse_eigs.h`
      - `include/sparse_matrix.h`
      - `include/sparse_types.h`
  - strongest support-only but real target:
    - workflow and proof-surface wording only where a landed usability batch
      truly changes how users should interpret those surfaces
- The strongest current contradiction is now explicit:
  - `README.md` already contains a real user entry path through:
    - `Choose a Workflow`
    - `Quick Start`
    - repeated-run workflow guidance
    - installation references
  - but the same file still carries advanced benchmark, dead-code, maintainer,
    and support references deep into the front-door reading path
  - the result is a truthful but over-dense front door: first-adoption
    decisions, advanced workflow interpretation, and maintainer-facing
    references still coexist too closely
- The strongest second contradiction is examples/workflow asymmetry:
  - the example surfaces are real and maintained
  - the downstream CMake example is minimal and bounded
  - but README still has to do too much work explaining how to move from
    one-shot examples to repeated-run, benchmark, and maintained proof lanes
- The strongest third contradiction is support-surface audience blur:
  - `INSTALL.md` already says README is the canonical front door
  - `benchmarks/README.md` already self-limits toward benchmark ownership
  - `docs/maintainer_guide.md` is already maintainer-facing
  - but the audience boundaries among these surfaces are still not quite sharp
    enough, so README still carries more support-routing burden than it should
- The strongest fourth contradiction is public-narrative spillover:
  - the highest-signal public headers remain large and valuable
  - but they still read with more internal workflow/policy context than an
    adoption-focused public narrative ideally needs
  - this remains real Sprint 88 work, but it is clearly later than the first
    front-door and example lanes
- The strongest Day 3 clarification is now fixed:
  - the best first Sprint 88 move is not generic "improve docs"
  - it is one bounded front-door simplification pass on the README-level user
    decision path
  - examples/workflow simplification follows next where the README contract
    exposes a real maintained adoption gap
  - support-surface consolidation comes after that where audience boundaries
    need sharpening
  - public header narrative cleanup remains real, but later than the first
    adoption-flow lanes

### Validation
- This was a docs-only audit day, so no full build/test rerun was required.
- Targeted sanity checks were completed:
  - re-read `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and
    `benchmarks/README.md`
  - re-scanned key user-facing and maintainer-facing section boundaries with
    targeted `rg`
  - rechecked the strongest example and public-header narrative surfaces

### Day 3 Exit State
- Sprint 88 now has one ranked live front-door usability contradiction map
  grounded in the current tree and the stabilized Sprint 87 package/platform
  contract.
- The first implementation center is fixed to bounded README/front-door
  simplification, not immediate support-surface or header cleanup.
- Later examples/workflow simplification, support-surface consolidation, and
  public narrative cleanup are explicitly ordered behind that first lane.

## Day 4 - First Usability Boundary Freeze

### Goal
Fix the first bounded Sprint 88 usability implementation fence so the next
design pass can define one real front-door contract instead of another broad
docs or support rewrite.

### Actions
- Re-read the Day 3 usability contradiction map against the Sprint 88
  project-plan scope.
- Reconciled the ranked front-door, examples, support-surface, and
  public-narrative lanes against the preserved Sprint 87 package/platform
  contract and proof-ownership split.
- Fixed the required first implementation center and the directly forced
  support-only surfaces that may move only if the first landing truly needs
  them.
- Fixed the preserved first-batch non-goal fence for README/front-door work.

### Findings
- Sprint 88 now has one explicit first implementation fence:
  - required first landing:
    - `README.md`
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `INSTALL.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
    - `examples/cmake_example/CMakeLists.txt`
    - `examples/cmake_example/main.c`
  - support-only proof and workflow surfaces that stay later unless the first
    landing truly forces movement:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
  - explicitly deferred from the first landing:
    - examples / workflow simplification as a first-batch center
    - support-surface consolidation as a first-batch center
    - public-header / API narrative cleanup as a first-batch center
    - package/platform contract reopening
    - benchmark-policy rewriting detached from adoption guidance
    - correctness-ownership redistribution
- The strongest Day 4 clarification is now explicit:
  - the best first Sprint 88 move is one bounded front-door simplification
    pass centered on `README.md`
  - the first landing should decide how the repo wants the first user path,
    support references, and adoption sequence to read before example or header
    widening moves
  - `INSTALL.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, and the
    example surfaces remain directly allowed support surfaces only if the
    README/front-door contract truly forces them to move
  - install/export proof, workflow surfaces, and public-header cleanup stay
    later unless the front-door landing truly changes their obligations
- The preserved first-batch non-goal fence is explicit now:
  - no package/platform contract reopening
  - no correctness-ownership redistribution
  - no benchmark-policy rewrite detached from adoption guidance
  - no internal architectural rewrite disguised as docs cleanup
  - no workflow/platform claim broadening beyond the already-maintained proof
    and support surfaces

### Validation
- This was a docs-only boundary-freeze day, so no full build/test rerun was
  required.
- Targeted sanity checks were completed:
  - re-read the Day 3 user-journey audit
  - re-read the Sprint 88 plan boundary expectations
  - rechecked the strongest first-tier and deferred support surfaces in the
    current tree

### Day 4 Exit State
- Sprint 88 now has one bounded first front-door landing center.
- Day 5 can design one explicit front-door and support-layering contract
  inside that fence.
- Later examples/workflow simplification, support-surface consolidation, and
  public-narrative cleanup are held back until later lanes.

## Day 5 - Workflow-Simplification Design

### Goal
Define the bounded adoption-guidance and support-layering contract Sprint 88
will actually support so Day 6 can land one real README/front-door change
instead of another broad docs cleanup.

### Actions
- Re-read the Day 5 contract from
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the fixed first-batch fence from
  `docs/planning/EPIC_8/SPRINT_88/artifacts/day4-first-usability-boundary.md`.
- Re-read the strongest recent Day 5 design template from
  `docs/planning/EPIC_8/SPRINT_87/artifacts/day5-product-matrix-design.md`.
- Re-audited the strongest front-door ownership seams in `README.md`,
  especially:
  - `Choose a Workflow`
  - `Quick Start`
  - repeated-run direct guidance
  - iterative/eigensolver workflow guidance
  - support and advanced-reference handoff points
- Fixed the intended ownership split between:
  - front-door guidance surfaces
  - example/workflow adoption surfaces
  - support-only advanced reference surfaces
  - maintainer-only detail surfaces
- Wrote the Day 5 design artifact and aligned the working-notes contract to
  the Day 4 fence.

### Findings
- Sprint 88 now has one explicit first implementation contract:
  - required implementation center:
    - `README.md`
  - directly forced support surfaces only if the first batch truly needs
    them:
    - `INSTALL.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
    - `examples/cmake_example/CMakeLists.txt`
    - `examples/cmake_example/main.c`
  - proof and workflow surfaces remain later owners unless the first batch
    truly changes their obligations:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
  - public-header narrative cleanup remains later than the first front-door
    landing:
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_matrix.h`
    - `include/sparse_types.h`
- The Day 5 ownership split is now fixed:
  - front-door adoption-guidance owner:
    - `README.md`
  - retained example/workflow adoption owners only if the front-door landing
    truly changes how downstream examples should be sequenced:
    - `examples/cmake_example/CMakeLists.txt`
    - `examples/cmake_example/main.c`
  - retained support-only advanced-reference owners only if the front-door
    landing truly changes where operational detail should live:
    - `INSTALL.md`
    - `benchmarks/README.md`
  - retained maintainer-only detail owner only if the front-door landing
    truly changes audience boundaries:
    - `docs/maintainer_guide.md`
  - retained proof and workflow evidence owners after the first landing:
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
- The strongest Day 5 design decision is now explicit:
  - Sprint 88 should treat `README.md` as an adoption-first surface, not as
    the place where every support, benchmark, workflow, or maintainer detail
    must be taught
  - the front door should preserve one clear user path:
    - choose a workflow
    - complete a minimal quick start
    - widen into repeated-run direct workflows
    - widen later into iterative/eigensolver, examples, install/support, and
      benchmark references only when the user actually needs them
  - advanced support and maintainer material should remain in their retained
    owners instead of being pulled forward into the first read path
- The useful Day 5 clarification is fixed now:
  - Day 6 should not try to solve examples, install/support, maintainer, and
    public-header narrative cleanup all at once
  - it should simplify the front door so the first adoption sequence reads
    clearly and hands off deliberately to later surfaces
  - it should preserve the maintained proof, workflow, and package-contract
    owners as later lanes rather than blending them into the first README
    pass
- The preserved first-batch fence remains explicit:
  - no package/platform contract reopening
  - no correctness-ownership redistribution
  - no benchmark-policy rewrite detached from adoption guidance
  - no workflow/platform claim broadening beyond the maintained proof
    surfaces
  - no public-header narrative widening folded into the first README batch
    unless the front-door contract truly forces it

### Validation
- This was a docs-only design day, so no full build/test rerun was required.
- Targeted sanity checks were completed:
  - re-read the Day 5 Sprint 88 plan contract
  - re-read the fixed Day 4 boundary artifact
  - re-read the strongest recent Day 5 design template
  - re-audited the highest-value README adoption sequence and handoff seams

### Day 5 Exit State
- Sprint 88 now has one bounded front-door usability design contract.
- Ownership between README, examples, support references, maintainer detail,
  and retained proof/workflow evidence is explicit before implementation.
- Day 6 can land one bounded README/tutorial batch without reopening the
  scope fence.

## Day 6 - README / Tutorial Batch

### Goal
Land one bounded README/front-door simplification batch that closes a real
adoption-friction contradiction without widening into examples, support docs,
workflow policy, or public-header cleanup.

### Actions
- Re-read the Day 6 implementation contract from
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the Day 5 design contract from
  `docs/planning/EPIC_8/SPRINT_88/artifacts/day5-workflow-simplification-design.md`.
- Reworked `README.md` so the first-user path is more explicit and adoption
  first:
  - added a compact `Start Here` front-door section near the top
  - tightened `Choose a Workflow` around the smallest real workflow choices
  - added a minimal first-adoption build path ahead of the deeper build
    reference
  - added explicit quick-start follow-through so users know when to widen
    into repeated-run direct, iterative, or install surfaces
- Kept the landing inside the required implementation center and rechecked
  that no support-surface or proof-owner follow-through was truly forced.
- Ran the substantial-docs validation gate required by the Sprint 88
  validation split:
  - `make quality-review-full`

### Findings
- Sprint 88's first implementation landing stayed inside the Day 5 fence:
  - required implementation center:
    - `README.md`
  - directly forced support follow-through actually needed:
    - none
  - not needed in the batch:
    - `INSTALL.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
    - `examples/cmake_example/CMakeLists.txt`
    - `examples/cmake_example/main.c`
    - `tests/test_install.sh`
    - `tests/test_cmake_install.sh`
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_matrix.h`
    - `include/sparse_types.h`
- The kept front-door usability win is explicit:
  - the README now opens with one compact `Start Here` path instead of making
    the first-time user infer that path from the larger feature surface
  - workflow choice is now organized around the smallest real solver/lifecycle
    surface rather than mixing first adoption with benchmark and proof-policy
    detail
  - local build guidance now starts with the shortest realistic first-adoption
    command path before widening into the full build reference
  - the quick start now hands off explicitly to repeated-run direct,
    iterative, and install surfaces instead of leaving the next step implicit
- The strongest Day 6 clarification is now explicit:
  - a real front-door contradiction closed without reopening package/platform
    semantics
  - examples, support-surface consolidation, workflow evidence, and
    public-header narrative cleanup remain later Sprint 88 lanes
  - the README now behaves more like an adoption-first surface and less like a
    mixed operator/reference dump

### Validation
- `make quality-review-full` passed.
- Reviewed parity remained exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - reviewed CMake `Total Test time (real)` = `412.54 sec`

### Day 6 Exit State
- Sprint 88 now has one landed bounded README/front-door batch.
- The first-user path from build to workflow choice to quick-start follow-on
  is sharper in the live repo.
- Later examples/workflow simplification, support-surface consolidation, and
  public-header narrative cleanup remain explicitly outside this first
  landing.

## Day 7 - Post-Landing Audit and Re-Rank

### Goal
Re-audit the touched front-door usability surfaces after the Day 6 landing and
fix the highest-value next implementation center so Sprint 88 does not drift
into generic docs follow-up.

### Actions
- Re-read the Day 7 rerank contract from
  `docs/planning/EPIC_8/SPRINT_88/PLAN.md`.
- Re-read the landed Day 6 batch from
  `docs/planning/EPIC_8/SPRINT_88/artifacts/day6-readme-tutorial-batch.md`.
- Re-audited the touched README front door and the strongest current handoff
  seams into examples, tutorial, install/support, benchmarks, and maintainer
  material.
- Rechecked the retained example/workflow adoption surfaces most likely to own
  the next contradiction:
  - `examples/README.md`
  - `docs/tutorial.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
- Reconfirmed which lower-value support and public-narrative surfaces remain
  real Sprint 88 work but still do not justify becoming the next center.
- Wrote the rerank artifact and aligned the working-notes priority map to the
  post-Day-6 repo state.

### Findings
- The Day 6 landing closed the strongest first usability contradiction:
  - `README.md` no longer stands out as the unclear first adoption center
  - a second immediate README-only batch is not the highest-value next move
- The strongest remaining contradiction is now examples / workflow
  simplification:
  - the README now tells users when to widen into examples and the fuller
    repeated-run flow
  - but the next-step example surfaces still live across multiple owners
    without one compact adoption package
  - `examples/README.md` and `docs/tutorial.md` now matter more because the
    README handoff is clearer than it was before Day 6
- The exact Day 8 design center is now fixed to the example/workflow adoption
  lane, centered first on:
  - `examples/README.md`
- The strongest directly adjacent support-only follow-through is now:
  - `docs/tutorial.md`
  - `examples/cmake_example/CMakeLists.txt`
  - `examples/cmake_example/main.c`
  - `README.md`
- Lower-value but still real later Sprint 88 lanes remain:
  - support-surface consolidation:
    - `INSTALL.md`
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - public-header / API narrative cleanup:
    - `include/sparse_iterative.h`
    - `include/sparse_eigs.h`
    - `include/sparse_matrix.h`
    - `include/sparse_types.h`
- The strongest Day 7 clarification is now explicit:
  - Sprint 88 should not widen next into install/support or header narrative
    cleanup just because the README now points at those surfaces
  - the best next move is one bounded example/workflow adoption package that
    makes the post-README handoff easier to follow
  - support-surface consolidation remains real, but it is later than the
    example/workflow lane unless that lane truly forces it

### Validation
- This was a docs-only rerank day, so no full build/test rerun was required.
- Targeted sanity checks were completed:
  - re-read the Day 6 artifact and the touched README sections
  - rechecked the strongest example/workflow ownership surfaces
  - rechecked the retained support-only and later public-narrative surfaces

### Day 7 Exit State
- Sprint 88 no longer needs another front-door-first rerank debate.
- The next implementation center is fixed to bounded example/workflow
  simplification.
- Day 8 can design one exact second implementation contract instead of
  reopening the front-door lane.
