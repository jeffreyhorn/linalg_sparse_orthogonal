# Sprint 59 Working Notes

## Day 1

**Objective:** Turn the Sprint 59 project-plan scope plus the Sprint 58
validated close state into a concrete final Epic 5 quality/platform and
closeout starting point by confirming the preserved reviewed baseline, naming
the Sprint 59 follow-through workstreams explicitly, and defining the
authoritative quality-contract, platform, and closeout hotspots before any
follow-through edits begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 59 project-plan source and the new sprint plan:
   - `sed -n '312,341p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
3. Re-read the strongest inherited Sprint 58 closeout sources:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md`
4. Re-read the Epic 5 review/todo guidance for the remaining quality/platform
   queue:
   - `sed -n '278,320p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '196,210p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
   - `rg -n "Quality-Platform|dead-code|coverage|Windows|macOS staging|reviewed-wrapper parity|deadcode|platform" docs/planning/EPIC_5 docs/maintainer_guide.md README.md Makefile .github/workflows -g '!build'`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Re-read the strongest live quality/platform contract surfaces:
   - `sed -n '840,872p' README.md`
   - `sed -n '498,650p' Makefile`
8. Measure the main Sprint 59 quality/platform and closeout hotspot surfaces:
   - `wc -l scripts/deadcode_workflow.sh scripts/deadcode_report.py .github/workflows/ci.yml README.md Makefile docs/maintainer_guide.md docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md docs/planning/EPIC_5/SPRINT_58/artifacts/day13-full-validation-sweep.md docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md`

### Day 1 Findings

#### 1. Sprint 59 starts from a validated final-sprint baseline, not from renewed feature, lifecycle, or API design work

The inherited starting state is already explicit and stable:

- Sprint 58 closed with:
  - public-surface simplification complete across README/tutorial/headers/
    examples/benchmarks
  - no public API redesign
  - no workflow-boundary drift
  - no implementation recovery queue hidden behind the docs work
- Sprint 58 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited product contract remains unchanged:
  - one-shot APIs remain first-class entry points
  - repeated direct-run lifecycle remains the validated Sprint 50-53 shape
  - repeated-run iterative/eigensolver handles remain the validated Sprint 54
    support boundary

Interpretation:

- Sprint 59 is not a feature sprint
- Sprint 59 is not a lifecycle-recovery sprint
- Sprint 59 is a bounded quality/platform/productization and closeout sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the final Epic 5 follow-through

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`
- `quality-review` still includes:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

Interpretation:

- Sprint 59 should keep using the exact `strongest local reviewed baseline`
  phrasing
- later code-touching days should keep the reviewed CMake count and
  Makefile/CMake parity contract as the main truthfulness anchors
- the final Epic 5 closeout should still treat `deadcode-check` as a report-
  completeness gate rather than a zero-findings gate

#### 3. The Epic 5 review queue is now concentrated in staged quality/platform follow-through rather than implementation ownership

The project plan and Epic 5 review/todo notes already fix the remaining
quality/platform queue:

- dead-code execution remains serialized
- macOS dead-code remains staged
- Windows local reviewed-wrapper parity remains staged
- Windows dead-code remains excluded
- coverage remains calibrated to the current enforced 80% reality

The inherited review guidance is explicit:

- the quality contract is already honest and strong
- the remaining gaps should be treated as a bounded productization pass
- the goal is to refresh each staged/excluded quality surface with a current
  disposition:
  - fixed
  - still intentionally staged
  - or explicitly deferred again with current rationale

Interpretation:

- Sprint 59 should treat the remaining quality/platform queue as real but
  bounded
- the strongest remaining maintainability pressure is now in platform-story
  truthfulness and residual disposition, not solver behavior

#### 4. Sprint 59 reduces cleanly to six bounded work classes

The Sprint 59 project-plan items reduce to six bounded work classes:

1. quality/platform residual audit
2. bounded quality follow-through batch
3. final cross-surface compatibility sweep
4. full validation sweep
5. Epic 5 summary and handoff
6. project-plan / residual-journal finalization

The strongest architectural narrowing is:

- keep the work centered on final truthfulness and bounded convergence
- prefer explicit residual disposition over broad platform ambition
- preserve the Sprint 50-58 public and validation fence exactly
- do not broaden into feature, API, benchmark-framework, or CI-redesign work

Interpretation:

- Sprint 59 is about closing the remaining stated gaps honestly
- the right output shape is a smaller, measured residual queue and a final
  integrated closeout package

#### 5. The authoritative Sprint 59 quality/platform hotspots are now fixed directly from the live repo

The strongest Sprint 59 touched surfaces are now explicit:

- quality-contract and platform-story surfaces:
  - `README.md` = `973`
  - `Makefile` = `878`
  - `docs/maintainer_guide.md` = `294`
  - `.github/workflows/ci.yml` = `221`
- dead-code and classification workflow surfaces:
  - `scripts/deadcode_workflow.sh` = `219`
  - `scripts/deadcode_report.py` = `550`
- project-level planning and closeout surfaces:
  - `docs/planning/EPIC_5/PROJECT_PLAN.md` = `340`
  - `docs/planning/EPIC_5/SPRINT_58/RETROSPECTIVE.md` = `226`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day13-full-validation-sweep.md` = `107`
  - `docs/planning/EPIC_5/SPRINT_58/artifacts/day14-closeout-and-handoff.md` = `125`

Interpretation:

- the strongest implementation-adjacent follow-through pressure is in
  `Makefile`, dead-code workflow scripts, and the platform contract wording
- the strongest closeout-writing pressure is now in Epic-level summary and
  residual disposition surfaces rather than sprint-local implementation notes

#### 6. The inherited public and quality compatibility fence gives Sprint 59 a clean non-goal boundary

The inherited fence remains:

- no public API redesign
- no reopening the direct lifecycle contract
- no reopening the repeated-run iterative/eigensolver support boundary
- no broad CI/platform redesign disguised as “final polish”
- no fake closure claims on staged quality surfaces without fresh evidence

Interpretation:

- Sprint 59 should improve residual disposition and final truthfulness under
  the already-validated product surface
- bounded convergence and honest closeout, not maximal platform expansion, are
  the success criteria

#### 7. The final Epic 5 closeout should now be treated as a measured integration package rather than a generic retrospective pass

The inherited Sprint 50-58 work already established:

- direct lifecycle, CSC, and factor-many behavior
- repeated-run iterative/eigensolver support boundaries
- large-source and giant-test maintainability improvements
- public-surface simplification

Interpretation:

- Sprint 59 should focus on:
  - final quality/platform disposition
  - final cross-surface caller-story agreement
  - final validation baseline
  - final project-level residual and handoff clarity
- the right closeout shape is a measured Epic 5 finish, not another broad
  storytelling sweep

## Day 1 Close

Sprint 59 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated public-contract and workflow fence from Sprint 58
- named quality/platform/productization hotspots
- clear final-sprint workstreams
- explicit non-goal fence against feature, API, or broad platform redesign

That is enough to move to the Day 2 validation and truthfulness-anchor recheck
without reopening Sprint 50-58 product-surface decisions.

## Day 2

**Objective:** Reconfirm the maintained reviewed validation baseline, the exact
truthfulness anchors, and the final-sprint rerun set Sprint 59 must preserve
before any quality/platform follow-through or Epic 5 closeout edits begin.

### Commands Run

1. Re-read the Sprint 59 Day 2 plan scope:
   - `sed -n '80,120p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the current Sprint 59 working-notes starting state:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/WORKING_NOTES.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
5. Re-read the live quality-contract wording across the main maintained
   authority surfaces:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check|dead-code|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
6. Reconfirm the targeted final-sprint rerun set from the live build tree:
   - `ls build/test_integration build/test_iterative build/test_eigs build/test_eigs_lobpcg build/test_chol_csc build/test_ldlt_csc build/example_analysis build/example_iterative build/example_ic_minres build/example_eigs build/example_svd_lowrank build/bench_refactor build/bench_refactor_csc build/bench_iterative_reuse build/bench_eigs_reuse`

### Day 2 Findings

#### 1. The maintained reviewed baseline remains unchanged and should stay the explicit final Epic 5 truthfulness anchor

The maintained Sprint 59 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The wrapper wording also remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 59 should keep using the exact `strongest local reviewed baseline`
  wording
- the reviewed CMake count and Makefile/CMake parity remain the main
  truthfulness anchors for the final sprint

#### 2. The authority split remains explicit and stable enough to carry unchanged through the final closeout

The maintained authority split remains:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 59 does not need to reopen the quality-contract hierarchy itself
- the final Epic 5 closeout can keep treating `deadcode-check` as a
  completeness and interpretation gate, not as a claim that dead-code findings
  must be zero

#### 3. The code-day validation boundary remains the same as the live repo contract

The mandatory gate for later `*.c` / `*.h` days remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial quality/platform follow-through
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- any Sprint 59 code-touching day still does
- substantial quality/platform changes should continue to use the stronger
  reviewed wrapper path too

#### 4. The maintained quality/platform wording is still aligned across the repo's authoritative surfaces

The quality-contract wording remains aligned across:

- `README.md`
  - strongest local reviewed baseline command map
  - dead-code completeness-gate meaning
  - reviewed platform matrix and staged-platform caveats
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - rerun guidance
  - test-count parity checks
- GitHub workflows
  - reviewed CMake execution path
  - dead-code report/check execution path
  - coverage-contract execution path

Interpretation:

- Sprint 59 does not need to reopen baseline wording just to restate the
  existing contract
- the real remaining work is bounded residual disposition on staged/excluded
  platform and dead-code surfaces, not rediscovering the quality hierarchy

#### 5. The authoritative Sprint 59 rerun set is now fixed from the live build tree

The main Sprint 59 follow-on binaries already present in `build/` are:

- `./build/test_integration`
- `./build/test_iterative`
- `./build/test_eigs`
- `./build/test_eigs_lobpcg`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/example_analysis`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Interpretation:

- Sprint 59 can keep its rerun focus on:
  - direct/public lifecycle proof
  - representative maintained examples
  - representative maintained benchmark drivers
  - the reviewed CMake parity anchor
- no broader default rerun set is required on Day 2

#### 6. Day 2 removes validation ambiguity before the final quality/platform audit begins

Day 2 now leaves Sprint 59 with:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit `*.c` / `*.h` code-day gate
- explicit stronger reviewed-baseline default
- authoritative final-sprint rerun set from the live build tree

Interpretation:

- the sprint can move to the Day 3 residual audit without uncertainty about
  what the maintained validation contract actually is
- any later change to the platform/quality story can now be judged against a
  fixed, explicit baseline

## Day 2 Close

Sprint 59 now has an explicit validation and truthfulness boundary:

- preserved reviewed baseline wording
- explicit authority split
- exact code-day validation gate
- stronger reviewed-wrapper default for substantial follow-through
- authoritative tests/examples/benchmarks rerun set from the live tree

That is enough to move to the Day 3 quality/platform residual audit without
validation ambiguity.

## Day 3

**Objective:** Reduce the remaining Sprint 59 quality/platform queue to named,
defensible residual classes by separating already-honest staged limits from
the few surfaces that still need fresh follow-through or measurement before the
final Epic 5 closeout.

### Commands Run

1. Re-read the Sprint 59 Day 3 plan scope:
   - `sed -n '120,170p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the main live quality/platform contract and policy surfaces:
   - `sed -n '1,520p' README.md`
   - `sed -n '1,360p' docs/maintainer_guide.md`
   - `sed -n '1,320p' .github/workflows/ci.yml`
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,240p' scripts/deadcode_report.py`
3. Re-read the Epic 5 review/todo guidance for the residual queue:
   - `sed -n '1,320p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
4. Reconfirm the cross-workflow platform coverage surfaces:
   - `ls .github/workflows`
   - `rg -n "windows|macos|quality-review|deadcode|coverage|ctest -N|quality-review-cmake" .github/workflows`
   - `sed -n '1,220p' .github/workflows/windows-ci.yml`
   - `sed -n '1,220p' .github/workflows/macos-ci.yml`
5. Re-read the most relevant maintained contract sections directly:
   - `sed -n '500,760p' Makefile`
   - `sed -n '620,840p' README.md`
6. Confirm branch cleanliness before writing the audit:
   - `git status --short --branch`

### Day 3 Findings

#### 1. The active residual queue is now smaller than the project-plan summary initially implied

The live repo already closes or stabilizes more of the quality/platform story
than the raw Sprint 59 item list suggests:

- Linux enforced reviewed paths are explicit and live in CI:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make deadcode-report`
  - `make deadcode-check`
- Windows enforced reviewed CMake subset is explicit and live in:
  - `.github/workflows/windows-ci.yml`
- macOS enforced Apple Clang reviewed path is explicit and live in:
  - `.github/workflows/macos-ci.yml`
- coverage is already a live supplemental CI signal:
  - `make coverage`

Interpretation:

- Sprint 59 is not recovering a vague or broken platform story
- it is deciding which remaining staged limits are still justified and which
  ones deserve one last bounded follow-through pass

#### 2. Coverage calibration is no longer a real open residual by itself

The inherited review queue named coverage calibration as a remaining quality
surface, but the live repo state now makes its disposition much clearer:

- the `Makefile` documents and enforces:
  - `COV_THRESHOLD = 80`
- the rationale for the 80% threshold remains embedded in the maintained
  comments near the coverage targets
- Linux CI still runs:
  - `make coverage`
- README and maintainer-facing wording already agree that coverage is:
  - supplemental
  - active
  - calibrated to the current 80% enforced reality

Interpretation:

- coverage calibration should drop out of the active Sprint 59 implementation
  queue
- it is now better treated as:
  - no longer justified by the current repo state
- Day 4+ should not broaden into a new coverage-policy rewrite unless a real
  contradiction surfaces elsewhere

#### 3. Serialized dead-code execution remains a truthful and still-justified operational residual

The live dead-code surfaces still explicitly require serialized execution:

- `Makefile`
  - `.NOTPARALLEL` covers the `deadcode*` targets
  - `deadcode-check` still says authoritative execution remains serialized
- `docs/maintainer_guide.md`
  - tells maintainers to run `deadcode*` serially because they share
    `build/deadcode-cmake` and `build/deadcode/`
- `.github/workflows/ci.yml`
  - keeps dead-code in one serial job because the shared build/artifact paths
    would otherwise become flaky or misleading

Interpretation:

- this residual is still justified by the current repo topology
- it is not primarily a truthfulness defect because the limitation is already
  explicit across the maintained surfaces
- it is better classified as:
  - already acceptable as deferred residual
- reworking it would imply broader path/topology redesign than Sprint 59 needs

#### 4. macOS dead-code staging is the strongest remaining measurement seam

The macOS story is now very specific:

- `macos-ci.yml` enforces:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make wall-check`
  - `make sanitize`
- README explicitly keeps macOS dead-code as:
  - staged
- no maintained macOS workflow currently runs:
  - `make deadcode-report`
  - `make deadcode-check`

Interpretation:

- this is still a real residual rather than historical wording
- but the repo does not yet provide fresh evidence that the dead-code workflow
  is either ready or clearly impossible on macOS
- it is best classified as:
  - needs measurement before any change

#### 5. Windows reviewed-wrapper parity and Windows dead-code exclusion remain real but honestly bounded residuals

The Windows surfaces are now clearer than the inherited review summary alone:

- `.github/workflows/windows-ci.yml` enforces the reviewed CMake subset:
  - configure
  - build
  - `ctest -N`
  - full `ctest`
- the workflow also publishes the current staged exclusions directly in job
  output:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- README keeps Windows as:
  - enforced reviewed CMake subset
  - staged Makefile reviewed wrappers
  - staged dead-code

Interpretation:

- Windows is not missing from CI
- the real residual is narrower:
  - local reviewed-wrapper parity remains staged
  - dead-code remains excluded
- both remain truthfully described today, so they are better classified as:
  - already acceptable as deferred residual
- if Sprint 59 lands a follow-through batch here, it should be wording and
  residual-disposition cleanup first, not a fake promise of full Windows
  Makefile/dead-code convergence

#### 6. The strongest bounded follow-through target is now residual-disposition truthfulness, not platform-expansion ambition

After separating the residual classes, the ranked queue becomes:

1. macOS dead-code staging
   - needs measurement before any change
   - highest remaining platform-story uncertainty
2. cross-surface residual wording reconciliation
   - needs bounded follow-through now
   - highest truthfulness value at low implementation cost
3. serialized dead-code execution
   - already acceptable as deferred residual
4. Windows reviewed-wrapper parity
   - already acceptable as deferred residual
5. Windows dead-code exclusion
   - already acceptable as deferred residual
6. coverage calibration
   - no longer justified as an active residual

Interpretation:

- Sprint 59 should not broaden into new platform ambition for its own sake
- the best first landing seam is a bounded quality-contract reconciliation
  batch driven by the measured residual classes above

#### 7. Day 4 now has a concrete first boundary instead of a generic "platform cleanup" bucket

The cleanest next step is now explicit:

- first Day 4 design target:
  - bounded residual-disposition reconciliation across the maintained
    quality/platform contract surfaces
- likely touched surfaces:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - workflow comments only if they materially disagree
- likely non-goal:
  - no attempt to force full macOS dead-code or full Windows reviewed-wrapper
    parity without fresh measurement and a clearly bounded path

Interpretation:

- Sprint 59 now has one concrete first implementation seam justified by
  truthfulness risk and implementation cost
- the sprint can design that batch without pretending the remaining staged
  platform limits are all equally urgent

## Day 3 Close

Sprint 59 now has a concrete residual map:

- coverage calibration drops out as an active residual
- serialized dead-code execution remains an explicit and acceptable deferred
  limit
- macOS dead-code staging is the strongest remaining measurement seam
- Windows reviewed-wrapper parity and dead-code remain honestly bounded staged
  residuals
- the strongest Day 4 landing target is bounded residual-disposition
  reconciliation, not broad platform expansion

That is enough to move to the Day 4 follow-through design from a ranked,
defensible queue instead of a generic final-sprint cleanup bucket.

## Day 4

**Objective:** Freeze the first bounded Sprint 59 follow-through boundary by
selecting the exact residual-disposition reconciliation seam, defining the
truthfulness invariants it must preserve, and recording the non-goal fence
before any maintained quality/platform surfaces are edited.

### Commands Run

1. Re-read the Sprint 59 Day 4 plan scope:
   - `sed -n '170,220p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the Day 3 residual audit:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/artifacts/day3-quality-platform-residual-audit.md`
3. Re-read the current Sprint 59 notes state:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/WORKING_NOTES.md`
4. Re-read the closest prior bounded design artifact shape for wording and
   fence discipline:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_58/artifacts/day4-readme-tutorial-reduction-design.md`

### Day 4 Findings

#### 1. The first follow-through seam should be residual-disposition reconciliation, not measurement or platform-expansion work

Day 3 already fixed the ranked queue:

- macOS dead-code staging
  - needs measurement before any change
- cross-surface residual wording reconciliation
  - needs bounded follow-through now
- serialized dead-code execution
  - already acceptable as deferred residual
- Windows reviewed-wrapper parity
  - already acceptable as deferred residual
- Windows dead-code exclusion
  - already acceptable as deferred residual
- coverage calibration
  - no longer justified as an active residual

Interpretation:

- Day 5 should not start with macOS dead-code execution itself
- the first landed batch should instead tighten how the current residual
  dispositions are described across the maintained contract surfaces
- that gives Sprint 59 one bounded patch with high truthfulness value and low
  blast radius

#### 2. The exact first landing boundary is now explicit

The first bounded quality/platform landing should cover:

- compact residual-disposition wording in:
  - `README.md`
  - `docs/maintainer_guide.md`
- executable/help wording in:
  - `Makefile`
    - only where the current residual interpretations or rerun guidance can be
      made clearer without changing the underlying targets
- workflow-comment wording only if it materially disagrees with the surfaces
  above:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

The first landing should intentionally defer:

- running or enabling macOS dead-code in CI
- expanding Windows to full Makefile reviewed-wrapper parity
- expanding Windows dead-code execution
- dead-code path/topology redesign
- coverage-policy rewrite

Interpretation:

- this is a wording-and-contract reconciliation batch first
- the touched logic surface should stay as close as possible to zero unless a
  tiny guard/comment/help change is required for truthfulness

#### 3. The preserved invariants are now fixed before Day 5

The first follow-through batch must preserve:

- reviewed baseline wording
  - `make quality-review-full` remains the strongest local reviewed baseline
- Makefile/CMake parity truthfulness
  - `ctest -N --test-dir build/quality-review-cmake` remains the authoritative
    parity-count anchor
- platform-story honesty
  - Linux enforced reviewed baseline remains the main source-of-truth path
  - macOS dead-code remains staged unless fresh evidence changes that
  - Windows reviewed CMake subset remains enforced while wrappers/dead-code
    stay staged
- stable local developer workflow
  - no change to the normal operator command map
  - no new target taxonomy
  - no renamed reviewed or dead-code targets

Interpretation:

- Day 5 should improve residual disposition without moving the underlying
  quality hierarchy
- any wording simplification that weakens the current truthfulness anchors is
  out of bounds

#### 4. The cleanup policy is now explicit and narrow

For the first quality/platform batch:

- minimize blast radius
- prefer measurement-backed wording or already-proven live CI facts
- remove ambiguity before adding new explanation
- avoid broad platform abstraction redesign
- keep repo-wide interpretation in:
  - `docs/maintainer_guide.md`
- keep executable command detail in:
  - `Makefile`
- keep platform enforcement/staging snapshots concise in:
  - `README.md`
  - workflow comments

Interpretation:

- the patch should read like final contract tightening, not like a new platform
  strategy document
- if a point requires deeper rationale than a concise maintained surface can
  support, it should remain deferred rather than being overexplained

#### 5. The Day 5 landing checklist is now concrete

Day 5 should:

1. tighten residual-disposition wording across the selected maintained
   surfaces
2. explicitly preserve:
   - serialized dead-code execution as a current deferred operational limit
   - macOS dead-code as staged pending measurement
   - Windows reviewed-wrapper/dead-code as staged rather than silently
     incomplete
3. remove coverage from any wording that still implies it is an unresolved
   residual
4. keep workflow comments aligned with the public and maintainer-facing
   surfaces where needed
5. avoid any behavior change that would force the batch to become a platform
   implementation sprint

Interpretation:

- the design boundary is now exact enough to implement without vague cleanup
  intent
- the batch can be judged on truthfulness improvement and discipline, not on
  how many platform knobs it touches

## Day 4 Close

Sprint 59 now has an explicit first follow-through boundary:

- first target:
  - residual-disposition reconciliation across maintained quality/platform
    surfaces
- likely touched files:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - workflow comments only if needed for agreement
- preserved invariants:
  - reviewed baseline wording
  - Makefile/CMake parity truthfulness
  - platform-story honesty
  - stable local developer workflow
- explicit non-goals:
  - no macOS dead-code enablement yet
  - no Windows wrapper/dead-code expansion
  - no coverage-policy rewrite
  - no topology redesign

That is enough to move to Day 5 implementation from one bounded residual seam
instead of a generic quality/platform cleanup intent.

## Day 5

**Objective:** Land the first bounded Sprint 59 quality/platform follow-through
batch by tightening residual-disposition wording across the maintained
contract surfaces without changing the reviewed baseline hierarchy or widening
the sprint into platform-expansion work.

### Commands Run

1. Re-read the selected Day 4 landing boundary:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/artifacts/day4-quality-follow-through-design.md`
2. Re-read the touched maintained contract sections:
   - `sed -n '780,880p' README.md`
   - `sed -n '1,180p' docs/maintainer_guide.md`
   - `sed -n '498,650p' Makefile`
   - `sed -n '1,120p' .github/workflows/ci.yml`
   - `sed -n '1,120p' .github/workflows/macos-ci.yml`
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`
3. Review the landed diff and touched-surface wording:
   - `git diff -- README.md docs/maintainer_guide.md Makefile`
4. Run targeted post-edit sanity checks:
   - `rg -n "deadcode|staged|coverage|quality-review-full|quality-review-cmake|Windows|macOS" README.md docs/maintainer_guide.md Makefile .github/workflows`
   - `wc -l README.md docs/maintainer_guide.md Makefile`

### Day 5 Findings

#### 1. The first landed batch stayed inside the Day 4 residual-disposition fence

The landed Day 5 patch is confined to:

- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`

And it stays focused on:

- making the dead-code residual disposition explicit
- making the macOS dead-code staged state explicit as pending measurement
- keeping Windows reviewed-wrapper/dead-code staging explicit
- removing any implication that coverage is still an unresolved reviewed-baseline
  residual

Interpretation:

- Day 5 did not widen into platform enablement or topology redesign
- the first follow-through seam stayed a contract-tightening batch, exactly as
  designed

#### 2. README now gives operators a clearer final-sprint residual map

The touched README sections now say more directly that:

- dead-code remains operationally serialized because of the current shared
  build/artifact topology
- Linux keeps dead-code in the enforced quality surface
- macOS keeps dead-code staged pending fresh measurement
- Windows keeps dead-code staged rather than pretending reviewed parity that is
  not yet enforced
- the readiness checklist now carries the remaining staged limits explicitly

Interpretation:

- the top-level operator story is more truthful and easier to scan
- the README still stays compact and does not collapse into a full maintainer
  policy document

#### 3. Maintainer policy now captures the final residual classes directly instead of leaving them implied

The maintainer guide now has an explicit `Current residual dispositions`
section covering:

- serialized dead-code execution
- macOS dead-code staging
- Windows reviewed-wrapper/dead-code staging
- coverage as a live supplemental signal rather than an unresolved baseline
  gap

Interpretation:

- the repo now has one clear maintainer-policy home for the remaining
  quality/platform limits
- later Epic 5 closeout writing will not need to infer these residual classes
  indirectly from several smaller surfaces

#### 4. The Makefile wording is tighter without changing the target surface

The touched Makefile wording now:

- calls serialized dead-code topology an intentional current operational limit
  in the reviewed-quality block comments
- makes the `deadcode-check` completion message clearer about the shared-path
  reason for serialized execution

Interpretation:

- the executable surface is more explicit without changing:
  - target names
  - target topology
  - reviewed baseline hierarchy
- this preserves stable local workflow while tightening truthfulness

#### 5. Workflow files stayed untouched because they already matched the reconciled story

The Day 4 plan allowed workflow-comment edits only if they materially disagreed
with the maintained contract surfaces. After the Day 5 review:

- `ci.yml` already matched Linux enforced + supplemental wording
- `macos-ci.yml` already matched staged dead-code wording
- `windows-ci.yml` already matched enforced reviewed CMake subset + staged
  wrappers/dead-code wording

Interpretation:

- no workflow edit was needed
- avoiding no-value comment churn kept the batch smaller and cleaner

## Day 5 Close

Sprint 59 now has its first landed follow-through batch:

- `README.md` gives a clearer operator-facing residual map
- `docs/maintainer_guide.md` owns the final residual dispositions directly
- `Makefile` help/comment wording is tighter without changing target topology
- workflow files did not need churn because they already matched the intended
  contract

That is enough to move to the next Sprint 59 measurement/reconciliation steps
from a more explicit maintained quality/platform story.

## Day 6

**Objective:** Re-audit the landed Day 5 quality/platform surfaces and decide
whether one more bounded follow-through batch is still justified or whether the
remaining items should now be recorded explicitly as consciously deferred
residuals.

### Commands Run

1. Re-read the Sprint 59 Day 6 plan scope:
   - `sed -n '220,280p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the landed Day 5 batch artifact:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/artifacts/day5-bounded-quality-follow-through-batch1.md`
3. Re-read the strongest remaining staged platform surfaces:
   - `sed -n '1,220p' .github/workflows/macos-ci.yml`
   - `sed -n '120,220p' .github/workflows/ci.yml`
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,260p' scripts/deadcode_report.py`
4. Re-read the Day 4 design fence and inherited Epic 5 residual guidance:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_59/artifacts/day4-quality-follow-through-design.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
   - `sed -n '278,320p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
5. Re-scan the maintained residual wording after the Day 5 patch:
   - `rg -n "macOS dead-code|dead-code remains staged|Windows reviewed-wrapper|coverage calibration|serialized dead-code|residual" README.md docs/maintainer_guide.md Makefile .github/workflows docs/planning/EPIC_5/SPRINT_59`

### Day 6 Findings

#### 1. Day 5 removed the last justified wording-only contradiction in the maintained quality/platform surfaces

After the Day 5 patch:

- `README.md` now gives operators an explicit residual map
- `docs/maintainer_guide.md` now owns the current residual classes directly
- `Makefile` now explains serialized dead-code topology more plainly
- workflow comments still agree with the maintained contract surfaces

Interpretation:

- there is no remaining high-signal wording contradiction that clearly
  justifies a second reconciliation patch
- any additional Day 6 patch would need to rest on fresh platform evidence, not
  on residual language drift

#### 2. macOS dead-code still lacks the fresh evidence needed to justify enablement work

The current macOS situation remains:

- `macos-ci.yml` enforces the Apple Clang reviewed path plus
  `wall-check`/`sanitize`
- it does not install or build the Linux dead-code toolchain path:
  - pinned `xunused`
  - LLVM/Clang cmake setup used by the Linux dead-code job
- `scripts/deadcode_workflow.sh` still carries Darwin-specific argument
  shaping for `xunused`, which confirms macOS is not a no-op path, but it does
  not by itself prove CI readiness

Interpretation:

- macOS dead-code remains a real residual
- but Sprint 59 still does not have the fresh measurement needed to justify
  either:
  - enabling it
  - or claiming it is permanently blocked
- the correct Day 6 disposition remains:
  - explicitly deferred pending measurement

#### 3. Windows wrapper/dead-code follow-through still fails the bounded-cost test

The live Windows surface remains:

- enforced reviewed CMake subset in `windows-ci.yml`
- explicit staged status for:
  - Makefile reviewed wrappers
  - dead-code
- explicit excluded-test printout already preserved in job output

Interpretation:

- there is no hidden Windows truthfulness defect left after Day 5
- expanding Windows further would move Sprint 59 toward platform implementation
  work rather than final bounded reconciliation
- the correct Day 6 disposition remains:
  - explicitly deferred with current rationale

#### 4. Serialized dead-code execution remains a consciously deferred operational limit, not a Day 6 fix target

The current dead-code topology still depends on shared paths:

- `build/deadcode-cmake`
- `build/deadcode/`

And the maintained surfaces already agree that:

- authoritative execution remains serialized
- the limitation is operational rather than a silent correctness problem

Interpretation:

- there is still no bounded Day 6 patch that would reduce this limit without
  reopening topology/workflow redesign
- the correct disposition remains:
  - explicitly deferred with current rationale

#### 5. Coverage calibration remains out of the active residual queue

Nothing in the Day 5 re-audit reopens the coverage question:

- `make coverage` is still live in CI
- `80%` is still the enforced threshold
- README, Makefile, and maintainer-facing wording still agree that coverage is
  supplemental rather than part of the reviewed baseline

Interpretation:

- coverage calibration stays closed as an active Sprint 59 residual
- Day 6 should not invent new work there

#### 6. Day 6 is best recorded as an explicit defer decision rather than a cosmetic second patch

The remaining quality/platform queue is now smaller and more concrete precisely
because the repo can state what is still deferred and why:

- macOS dead-code:
  - deferred pending fresh measurement
- Windows reviewed-wrapper parity:
  - deferred because the reviewed CMake subset is already the enforced Windows
    truth surface and broader parity would widen scope
- Windows dead-code:
  - deferred for the same bounded-scope reason
- serialized dead-code topology:
  - deferred because it still depends on shared-path workflow design rather
    than wording drift
- coverage calibration:
  - removed from the active residual queue

Interpretation:

- no second landed patch is justified on Day 6
- the correct output is a sharper deferred-residual note, not low-value churn

## Day 6 Close

Sprint 59 Day 6 closes with an explicit defer decision:

- no second follow-through patch was justified after the Day 5 reconciliation
- macOS dead-code remains the strongest measurement-dependent residual
- Windows wrapper/dead-code follow-through remains consciously deferred
- serialized dead-code topology remains a consciously deferred operational
  limit
- coverage stays out of the active residual queue

That leaves Sprint 59 with a smaller and more concrete quality/platform queue
before the final cross-surface compatibility audit begins.

## Day 7

**Objective:** Reduce the final Sprint 59 integration problem to explicit
caller-story drift classes by auditing the strongest public workflow,
example, benchmark, header, and proof surfaces before the last reconciliation
batch lands.

### Commands Run

1. Re-read the Sprint 59 Day 7 plan scope:
   - `sed -n '280,340p' docs/planning/EPIC_5/SPRINT_59/PLAN.md`
2. Re-read the highest-signal public workflow docs:
   - `sed -n '1,240p' README.md`
   - `sed -n '1,240p' docs/tutorial.md`
3. Re-read the example and benchmark caller-story surfaces:
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '1,260p' benchmarks/README.md`
4. Re-read the strongest direct lifecycle API/proof surfaces:
   - `sed -n '1,220p' include/sparse_analysis.h`
   - `sed -n '1,260p' tests/test_integration.c`
5. Re-scan the support-boundary wording across the main public surfaces:
   - `rg -n "repeated-run|analyze-once|factor-many|one-shot|CG|GMRES|MINRES|BiCGSTAB|LOBPCG|grow-m|thick-restart|block iterative|compatibility surfaces|staged" include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h README.md docs/tutorial.md examples/README.md benchmarks/README.md tests/test_integration.c`
6. Re-read the main iterative/eigensolver public headers where lifecycle
   language matters:
   - `sed -n '1,220p' include/sparse_iterative.h`
   - `sed -n '1,220p' include/sparse_eigs.h`
7. Re-scan the integration proof names for the public lifecycle story:
   - `rg -n "public_lifecycle|refactor|same_pattern|repeated_solve|factor_solve|analysis_free|factor_free" tests/test_integration.c`

### Day 7 Findings

#### 1. There is no blocker-level support-boundary contradiction left across the main public surfaces

The main stable workflow fence still agrees across:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`

The agreed caller story remains:

- one-shot APIs are still the default/front-door path
- repeated direct solves use the explicit analysis/factors lifecycle
- repeated-run iterative handles remain bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- repeated-run eigensolver handles remain bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

Interpretation:

- Day 7 did not surface a blocker-level API/docs mismatch
- the final integration problem is now about caller-story precision and
  terminology, not about conflicting support boundaries

#### 2. Example/docs alignment is mostly coherent, with one residual top-level emphasis seam

The example-side story is already disciplined:

- `examples/README.md` clearly says the shipped examples still lean on the
  one-shot public APIs
- `example_analysis` is still identified as the strongest shipped repeated-run
  direct example
- `example_eigs` is clearly one-shot by design even though the repeated-run
  eigensolver handle exists
- iterative handles are described as opt-in paths rather than default examples

Interpretation:

- there is no example/docs contradiction that requires a new example or a broad
  example rewrite
- the remaining seam is that the top-level docs can still present the repeated
  paths a little more explicitly as opt-in lifecycle workflows, matching the
  example-side phrasing

#### 3. Benchmark/docs alignment is also coherent, and now reads more stable than the top-level docs in a few places

The benchmark story is already explicit and workflow-first:

- `bench_refactor` and `bench_refactor_csc`
  - direct repeated-run lifecycle proof
- `bench_iterative_reuse`
  - public repeated-run iterative handle proof
- `bench_eigs_reuse`
  - public repeated-run eigensolver handle proof
- intentional exclusions are already called out directly in
  `benchmarks/README.md`

Interpretation:

- benchmark/docs mismatch is no longer a high-priority drift class
- if anything, `benchmarks/README.md` now uses the most precise stable
  lifecycle taxonomy of the audited caller-story surfaces

#### 4. The strongest remaining drift class is terminology/positioning in the top-level docs, not support-boundary truth

The strongest residual caller-story seam is now concentrated in:

- `README.md`
- `docs/tutorial.md`

The pattern is:

- top-level docs still sometimes say:
  - `repeated direct solves`
  - `repeated iterative solves`
  - `repeated symmetric eigensolves`
- while the tighter example/benchmark/header surfaces more consistently say:
  - explicit repeated-run direct lifecycle
  - explicit iterative handles
  - explicit eigensolver handle

Interpretation:

- the highest-value remaining reconciliation target is not a new support
  boundary or a new proof surface
- it is a bounded top-level terminology/positioning pass so the front-door docs
  match the more precise stable vocabulary already used lower in the tree

#### 5. Test/story mismatch is low-risk and should stay out of the final integration batch

The public lifecycle proof in `tests/test_integration.c` is now strong and
specific:

- repeated solve + zeroed free behavior
- same-pattern refactor parity with one-shot Cholesky
- LDL^T same-pattern indefinite repeated-run proof
- failure-preservation and mismatch rejection cases

Interpretation:

- those tests are the right proof surfaces
- but the top-level docs do not need to enumerate all of those regression
  details
- test/story mismatch is therefore a low-priority residual and not the best
  Day 8 target

#### 6. Day 8 now has one clean first reconciliation target

The strongest final integration target is now explicit:

- first Day 8 reconciliation seam:
  - top-level README/tutorial terminology and positioning alignment

Likely touched surfaces:

- `README.md`
- `docs/tutorial.md`

Likely preserved non-goals:

- no public-header batch
- no example README rewrite
- no benchmark README rewrite
- no broad long-form README history cleanup
- no test naming or proof-surface churn

Interpretation:

- Sprint 59 can land one caller-value-driven final integration batch without
  turning the closeout into another broad docs sprint

## Day 7 Close

Sprint 59 now has a reduced final integration map:

- no blocker-level support-boundary contradiction remains
- example/docs and benchmark/docs alignment are mostly coherent already
- the strongest remaining drift class is top-level terminology and workflow
  positioning in `README.md` and `docs/tutorial.md`
- test/story mismatch is low-risk and should stay out of the final batch

That is enough to move to Day 8 from a concrete caller-story seam instead of a
generic final integration backlog.

## Day 8

**Objective:** Land the first bounded final integration reconciliation patch by
tightening the top-level README/tutorial terminology and workflow positioning
so the front-door docs match the more precise stable vocabulary already used by
the headers, examples, and benchmark docs.

### Commands Run

1. Re-read the selected Day 7 landing boundary:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_59/artifacts/day7-cross-surface-compatibility-audit.md`
2. Re-read the touched top-level workflow sections:
   - `sed -n '70,120p' README.md`
   - `sed -n '1,60p' docs/tutorial.md`
   - `sed -n '160,190p' docs/tutorial.md`
   - `sed -n '230,265p' README.md`
   - `sed -n '300,330p' README.md`
3. Review the landed diff for the touched docs:
   - `git diff -- README.md docs/tutorial.md`
4. Run targeted post-edit alignment checks:
   - `rg -n "explicit repeated-run direct lifecycle|iterative handles|eigensolver handle|one-shot compatibility surfaces|BiCGSTAB|LOBPCG|grow-m|thick-restart" README.md docs/tutorial.md examples/README.md benchmarks/README.md include/sparse_analysis.h include/sparse_iterative.h include/sparse_eigs.h`
   - `wc -l README.md docs/tutorial.md`

### Day 8 Findings

#### 1. The landed batch stayed inside the Day 7 caller-story fence

The Day 8 patch is confined to:

- `README.md`
- `docs/tutorial.md`

And it stays focused on:

- replacing looser top-level phrases like:
  - `repeated direct solves`
  - `repeated iterative solves`
  - `repeated symmetric eigensolves`
- with the more precise stable vocabulary already used by the lower-level
  surfaces:
  - explicit repeated-run direct lifecycle
  - iterative handles
  - eigensolver handle

Interpretation:

- Day 8 did not widen into a header batch, example rewrite, benchmark rewrite,
  or long-form README cleanup
- the first final integration patch stayed caller-value driven and narrow

#### 2. README now matches the lower-level lifecycle vocabulary more closely

The touched README sections now:

- describe the direct repeated path as the explicit repeated-run direct
  lifecycle
- describe repeated iterative use as the explicit iterative-handle path
- describe repeated eigensolver use as the explicit eigensolver-handle path
- remove the stale `Sprint 49` framing from the repeated-run handle section

Interpretation:

- the top-level product map now matches the lifecycle terminology already used
  by:
  - `include/sparse_analysis.h`
  - `examples/README.md`
  - `benchmarks/README.md`
- the README front door is now closer to the repo's final stable product
  vocabulary

#### 3. Tutorial now reinforces the same explicit-lifecycle terminology

The touched tutorial sections now:

- describe the opt-in solver reuse paths as:
  - explicit iterative-handle lifecycle
  - explicit eigensolver-handle lifecycle
- describe the direct repeated path as the explicit repeated-run direct
  lifecycle

Interpretation:

- the tutorial now points users toward the same stable lifecycle categories the
  README and lower-level surfaces use
- the top-level docs are less likely to drift back toward generic
  "repeated solve" phrasing that obscures the real public model

#### 4. The strongest remaining cross-surface drift is now smaller and lower-risk

After the Day 8 patch:

- top-level docs, examples docs, benchmark docs, and public headers now agree
  more closely on the core lifecycle terminology
- the remaining density is mostly:
  - long-form detail
  - historical depth
  - proof detail the top-level docs intentionally do not mirror

Interpretation:

- the strongest final caller-story mismatch has been materially reduced
- the remaining queue is now more about optional density cleanup than about
  misleading integration wording

## Day 8 Close

Sprint 59 now has its first final integration reconciliation patch:

- `README.md` uses the more precise stable lifecycle vocabulary
- `docs/tutorial.md` reinforces the same explicit direct/iterative/eigensolver
  handle terminology
- the patch stayed bounded to the selected caller-story seam

That is enough to move to the Day 9 re-audit from a smaller and more stable
top-level integration story.
