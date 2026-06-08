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
